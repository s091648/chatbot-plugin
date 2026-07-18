from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from chatbot_plugin_sdk.contracts.responses import ChunkResult
from chatbot_plugin.llm.base import AllProvidersExhausted, LLMResult, ProviderHandler, ResilientLLMService, ToolCallRequest, ToolSpec
from chatbot_plugin_sdk.processors.retrieve import RetrieveProcessor

logger = logging.getLogger(__name__)

# Matches inline citations the system prompt asks the model to produce, e.g. "[1]" or the
# grouped form "[1, 2]" — used to work out which of the context articles were actually cited
# so unused ones aren't sent back to the frontend as source pills.
_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

SYSTEM_PROMPT = """\
You are a research assistant that answers questions based ONLY on the
provided context chunks. Each chunk is prefixed with [N] indicating its source article number.

Rules:
- Answer using only the information in the context below.
- If the context does not contain enough information to answer, say so.
- Use inline [N] citations (e.g. [1], [2]) immediately after each claim to indicate its source.
- Do not list sources separately at the end — citations must be inline only.
- Do not use external knowledge or make assumptions beyond the context.
- Respond in the same language as the user's question.
"""

PINNED_SYSTEM_PROMPT = """\
You are a research assistant answering a question about one or more articles the user
has explicitly pinned. Each context chunk is prefixed with [N] indicating its source article number.

Rules:
- Prefer answering using ONLY the pinned article context below.
- If the pinned context does not contain enough information to answer the question, call the
  search_articles tool with a focused query to look at the wider article database. Do not call it
  if the pinned context already answers the question.
- Use inline [N] citations (e.g. [1], [2]) immediately after each claim to indicate its source.
- Do not list sources separately at the end — citations must be inline only.
- Do not use external knowledge or make assumptions beyond the context and search results.
- Respond in the same language as the user's question.
"""

_NO_RELEVANT_INFO_REPLY = (
    "I couldn't find relevant information in the database for your question. "
    "Please try rephrasing or ask about a different topic."
)

SEARCH_TOOL = ToolSpec(
    name="search_articles",
    description=(
        "Search the full article database for content beyond the pinned articles. "
        "Use only when the pinned article context is insufficient to answer the question."
    ),
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)


@dataclass
class ArticleRef:
    id: str
    title: str | None
    url: str
    public_article_id: str | None = None


@dataclass
class ToolCallExecution:
    id: str
    name: str
    arguments: dict
    result_summary: str
    is_error: bool


@dataclass
class ChatResult:
    reply: str
    articles_used: list[ArticleRef]
    thinking: str | None = None
    chunks: list[ChunkResult] = field(default_factory=list)
    tool_calls_executed: list[ToolCallExecution] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        retriever: RetrieveProcessor,
        llm: ResilientLLMService,
        max_context_chunks: int = 10,
        max_tokens: int = 2048,
        min_score: float = 0.0,
        min_rerank_score: float = 0.0,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_context_chunks = max_context_chunks
        self._max_tokens = max_tokens
        self._min_score = min_score
        self._min_rerank_score = min_rerank_score

    async def chat(
        self,
        message: str,
        topic_id: str | None = None,
        pinned_article_ids: list[str] | None = None,
    ) -> ChatResult:
        if pinned_article_ids:
            return await self._chat_pinned(message, pinned_article_ids)

        search_result = await self._retriever.retrieve(
            message,
            top_k=self._max_context_chunks,
            min_score=self._min_score,
            min_rerank_score=self._min_rerank_score,
            filters={"topic_id": topic_id} if topic_id else None,
        )
        merged = search_result.chunks[: self._max_context_chunks]
        if not merged:
            return ChatResult(reply=_NO_RELEVANT_INFO_REPLY, articles_used=[], chunks=[])

        articles, article_index = self._collect_articles(merged)
        context = self._build_context(merged, article_index)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {message}"},
        ]

        result, _ = await self._llm.complete(messages, self._max_tokens)
        articles_used = self._filter_cited_articles(result.text, articles)
        return ChatResult(reply=result.text, articles_used=articles_used, thinking=result.thinking, chunks=merged)

    async def _chat_pinned(self, message: str, pinned_article_ids: list[str]) -> ChatResult:
        exclude: set[str] = set()
        while True:
            pinned_chunks = (await self._fetch_pinned_chunks(message, pinned_article_ids))[: self._max_context_chunks]
            articles, article_index = self._collect_articles(pinned_chunks)
            context = self._build_context(pinned_chunks, article_index) if pinned_chunks else "(no content available for the pinned article(s))"
            messages = [
                {"role": "system", "content": PINNED_SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nQuestion: {message}"},
            ]

            result, handler = await self._llm.complete(messages, self._max_tokens, tools=[SEARCH_TOOL], exclude=exclude)

            if not result.tool_calls:
                articles_used = self._filter_cited_articles(result.text, articles)
                return ChatResult(reply=result.text, articles_used=articles_used, thinking=result.thinking, chunks=pinned_chunks)

            tool_executions, tool_result_messages, all_chunks, articles, article_index = await self._execute_tool_calls(
                result.tool_calls, pinned_chunks, articles, article_index
            )
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in result.tool_calls],
            })
            messages.extend(tool_result_messages)

            try:
                final_result, _ = await self._llm.complete(messages, self._max_tokens, tools=None, pinned_handler=handler)
            except AllProvidersExhausted:
                exclude.add(handler.name)
                continue

            articles_used = self._filter_cited_articles(final_result.text, articles)
            return ChatResult(
                reply=final_result.text,
                articles_used=articles_used,
                thinking=final_result.thinking,
                chunks=all_chunks,
                tool_calls_executed=tool_executions,
            )

    async def _execute_tool_calls(
        self,
        calls: list[ToolCallRequest],
        pinned_chunks: list[ChunkResult],
        articles: list[ArticleRef],
        article_index: dict[str, int],
    ) -> tuple[list[ToolCallExecution], list[dict], list[ChunkResult], list[ArticleRef], dict[str, int]]:
        tool_executions: list[ToolCallExecution] = []
        tool_result_messages: list[dict] = []
        all_chunks = list(pinned_chunks)
        seen_chunk_ids = {c.chunk_id for c in pinned_chunks}
        seen_articles = {a.id: a for a in articles}

        for call in calls:
            if call.name != "search_articles":
                tool_executions.append(ToolCallExecution(call.id, call.name, call.arguments, "unknown tool", True))
                tool_result_messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": call.name,
                    "content": f"Unknown tool: {call.name}", "is_error": True,
                })
                continue

            query = call.arguments.get("query")
            if not query:
                tool_executions.append(ToolCallExecution(call.id, call.name, call.arguments, "missing 'query' argument", True))
                tool_result_messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": call.name,
                    "content": "missing 'query' argument", "is_error": True,
                })
                continue

            try:
                search_result = await self._retriever.retrieve(
                    query,
                    top_k=self._max_context_chunks,
                    min_score=self._min_score,
                    min_rerank_score=self._min_rerank_score,
                    filters=None,
                )
            except Exception:
                logger.exception("search_articles_tool_failed", extra={"query": query})
                tool_executions.append(ToolCallExecution(call.id, call.name, call.arguments, "search failed", True))
                tool_result_messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": call.name,
                    "content": "search failed", "is_error": True,
                })
                continue

            new_chunks = [c for c in search_result.chunks if c.chunk_id not in seen_chunk_ids]
            seen_chunk_ids.update(c.chunk_id for c in new_chunks)
            all_chunks.extend(new_chunks)
            new_articles, article_index = self._collect_articles(new_chunks, seen=seen_articles, index=article_index)
            seen_articles = {a.id: a for a in new_articles}
            result_content = self._build_context(new_chunks, article_index) if new_chunks else "No additional results found."

            tool_executions.append(ToolCallExecution(call.id, call.name, call.arguments, result_content, False))
            tool_result_messages.append({
                "role": "tool", "tool_call_id": call.id, "name": call.name,
                "content": result_content, "is_error": False,
            })

        return tool_executions, tool_result_messages, all_chunks, list(seen_articles.values()), article_index

    def _filter_cited_articles(self, reply: str, articles: list[ArticleRef]) -> list[ArticleRef]:
        """Keeps only the context articles whose [N] index is actually cited in the reply.

        Falls back to returning every context article when the reply has no citations at all
        (e.g. the model ignored the instruction) — better to over-show sources than show none.
        """
        cited: set[int] = set()
        for match in _CITATION_PATTERN.finditer(reply):
            for part in match.group(1).split(","):
                cited.add(int(part.strip()))
        if not cited:
            return articles
        return [article for i, article in enumerate(articles, start=1) if i in cited]

    async def _fetch_pinned_chunks(self, message: str, public_article_ids: list[str]) -> list[ChunkResult]:
        if not public_article_ids:
            return []

        try:
            result = await self._retriever.retrieve(
                message,
                top_k=self._max_context_chunks,
                min_score=self._min_score,
                min_rerank_score=self._min_rerank_score,
                filters={"public_article_id": public_article_ids},
            )
            return result.chunks
        except Exception:
            logger.exception("pinned_chunk_retrieve_failed", extra={"article_ids": public_article_ids})
            return []

    def _collect_articles(
        self,
        chunks: list[ChunkResult],
        seen: dict[str, ArticleRef] | None = None,
        index: dict[str, int] | None = None,
    ) -> tuple[list[ArticleRef], dict[str, int]]:
        """Returns (unique articles in first-appearance order, article_id → 1-based index).

        Pass in a previous call's (seen, index) to extend the numbering instead of restarting
        from 1 — used when tool-searched articles are added after the pinned ones.
        """
        seen = dict(seen) if seen else {}
        index = dict(index) if index else {}
        for chunk in chunks:
            if chunk.article_id not in seen:
                meta = chunk.article_metadata
                raw_pid = meta.get("public_article_id")
                seen[chunk.article_id] = ArticleRef(
                    id=chunk.article_id,
                    title=meta.get("title"),
                    url=meta.get("url") or "",
                    public_article_id=str(raw_pid) if raw_pid is not None else None,
                )
                index[chunk.article_id] = len(seen)
        return list(seen.values()), index

    def _build_context(self, chunks: list[ChunkResult], article_index: dict[str, int]) -> str:
        parts = []
        for chunk in chunks:
            n = article_index.get(chunk.article_id, 0)
            title = chunk.article_metadata.get("title") or "Unknown"
            parts.append(f"[{n}] {title}\n{chunk.content}")
        return "\n\n".join(parts)
