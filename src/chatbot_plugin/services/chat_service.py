from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from itertools import groupby
from typing import AsyncIterator

from chatbot_plugin_sdk.contracts.responses import ChunkResult
from chatbot_plugin.llm.base import (
    AllProvidersExhausted,
    LLMResult,
    ProviderHandler,
    ResilientLLMService,
    StreamError,
    TextDelta,
    ThinkingDelta,
    ToolCallRequest,
    ToolSpec,
)
from chatbot_plugin_sdk.processors.retrieve import RetrieveProcessor

logger = logging.getLogger(__name__)

# Matches inline citations the system prompt asks the model to produce, e.g. "[1]" or the
# grouped form "[1, 2]" — used to work out which of the context articles were actually cited
# so unused ones aren't sent back to the frontend as source pills.
_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

SYSTEM_PROMPT = """\
You are a research assistant that answers questions based ONLY on the
provided context. Context is grouped by source article; each group is prefixed with [N]
indicating its article number. A single [N] group can contain several paragraphs separated
by "---" — they are all still that one article, not separate numbered sources, so never cite
a number higher than the highest [N] group actually shown below.

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
has explicitly pinned. Context is grouped by source article; each group is prefixed with [N]
indicating its article number. A single [N] group can contain several paragraphs separated
by "---" — they are all still that one article, not separate numbered sources, so never cite
a number higher than the highest [N] group actually shown below.

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

# The tool-call follow-up turn (tools=None — see _chat_pinned_stream/_chat_pinned) sometimes
# finishes cleanly (finish_reason STOP) without producing any answer text at all: the model's own
# thinking can indicate it wants to search again, but with tools disabled on that turn it can't,
# and instead of falling back to answering with what it already has it just ends the turn with
# nothing. See gemini_provider.py's stream()/complete() "empty output" warning logs for the
# provider-side signal this pairs with. Unlike _NO_RELEVANT_INFO_REPLY, search *did* find
# something here — the model just failed to use it — so a distinct message avoids implying the
# database had nothing relevant.
_EMPTY_FOLLOWUP_REPLY = (
    "I found some information via search but wasn't able to put together a complete answer "
    "from it. Please try rephrasing your question or asking again."
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
    # 1-based index of this article's [N] group in the context sent to the LLM (same value as
    # article_index[id] in _collect_articles/_build_context). _filter_cited_articles narrows
    # articles down to only the ones actually cited, which can leave gaps (e.g. context had
    # articles 1-4 but only [1] and [3] got cited) — without carrying the original number along,
    # a consumer indexing into the resulting list by array position would silently map a citation
    # to the wrong article once the list is no longer a contiguous 1..k prefix.
    number: int = 0


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


# ── Streaming event types (ChatService.chat_stream) ────────────────────────────────────────
# ThinkingDelta / TextDelta are forwarded straight from the LLM provider layer (chatbot_plugin.
# llm.base) — same shape, no translation needed. The rest are domain-level events chat_stream
# adds on top: tool-call lifecycle and the final source-article list (only knowable once the
# full reply text has streamed in, since it's derived from which [N] citations appear in it).

@dataclass
class ToolCallStarted:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolCallFinished:
    id: str
    name: str
    result_summary: str
    is_error: bool


@dataclass
class SourcesReady:
    articles: list[ArticleRef]


@dataclass
class StreamFailed:
    """Terminal event: the provider failed after already streaming some output for this turn.
    See ResilientLLMService.stream_complete for why this can't fall back to another provider."""
    message: str


ChatStreamEvent = ThinkingDelta | TextDelta | ToolCallStarted | ToolCallFinished | SourcesReady | StreamFailed


class ChatService:
    def __init__(
        self,
        retriever: RetrieveProcessor,
        llm: ResilientLLMService,
        max_context_chunks: int = 10,
        max_tokens: int = 2048,
        min_score: float = 0.0,
        min_rerank_score: float = 0.0,
        max_tool_rounds: int = 3,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_context_chunks = max_context_chunks
        self._max_tokens = max_tokens
        self._min_score = min_score
        self._min_rerank_score = min_rerank_score
        # Only meaningful for the pinned-article flow (_chat_pinned/_chat_pinned_stream) — how
        # many turns the model may use the search_articles tool before it's forced to answer
        # with whatever it has (tools=None). Bounds an otherwise-unbounded agentic loop; not a
        # literal count of tool *calls* (each round can request more than one).
        self._max_tool_rounds = max_tool_rounds

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

    async def chat_stream(
        self,
        message: str,
        topic_id: str | None = None,
        pinned_article_ids: list[str] | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Streaming counterpart to chat(). Raises AllProvidersExhausted if the LLM call fails
        before producing any output at all (caller can still return a clean error response at
        that point); once anything has streamed, failures surface as a StreamFailed event
        instead (see ResilientLLMService.stream_complete)."""
        if pinned_article_ids:
            async for event in self._chat_pinned_stream(message, pinned_article_ids):
                yield event
            return

        search_result = await self._retriever.retrieve(
            message,
            top_k=self._max_context_chunks,
            min_score=self._min_score,
            min_rerank_score=self._min_rerank_score,
            filters={"topic_id": topic_id} if topic_id else None,
        )
        merged = search_result.chunks[: self._max_context_chunks]
        if not merged:
            yield TextDelta(text=_NO_RELEVANT_INFO_REPLY)
            yield SourcesReady(articles=[])
            return

        articles, article_index = self._collect_articles(merged)
        context = self._build_context(merged, article_index)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {message}"},
        ]

        reply_parts: list[str] = []
        async for _, event in self._llm.stream_complete(messages, self._max_tokens):
            if isinstance(event, StreamError):
                yield StreamFailed(message=event.message)
                return
            if isinstance(event, TextDelta):
                reply_parts.append(event.text)
            yield event

        yield SourcesReady(articles=self._filter_cited_articles("".join(reply_parts), articles))

    async def _chat_pinned_stream(self, message: str, pinned_article_ids: list[str]) -> AsyncIterator[ChatStreamEvent]:
        pinned_chunks = (await self._fetch_pinned_chunks(message, pinned_article_ids))[: self._max_context_chunks]
        articles, article_index = self._collect_articles(pinned_chunks)
        context = self._build_context(pinned_chunks, article_index) if pinned_chunks else "(no content available for the pinned article(s))"
        messages = [
            {"role": "system", "content": PINNED_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {message}"},
        ]

        all_chunks = pinned_chunks
        tool_executions: list[ToolCallExecution] = []
        handler: ProviderHandler | None = None
        round_num = 0
        # Only a round whose tool calls actually carried a thought_signature needs to stay pinned
        # to the exact same handler for the next call (Gemini's continuity requirement — echoing
        # a *different* model's function-call part back gets a 400 INVALID_ARGUMENT). A round
        # with no thought_signature at all (the handler wasn't a "thinking"-capable Gemini model,
        # or didn't emit one) has nothing that needs preserving, so the next call is free to
        # rotate across every other configured handler on failure, same as round 0 — pinning it
        # anyway would strand a turn on one quota-exhausted model while sibling models configured
        # specifically as fallbacks still had headroom, for no actual correctness benefit. See
        # ToolCallRequest.thought_signature.
        requires_pinning = False

        # A model can legitimately want to search again with a different query after seeing the
        # first result set — self._max_tool_rounds bounds how many times it's allowed to (rather
        # than the old hardcoded "exactly one round trip, forced to answer immediately after"),
        # while still guaranteeing termination: once round_num reaches the cap, tools is forced
        # to None so the *next* call cannot request another tool call.
        while True:
            tools_for_round = [SEARCH_TOOL] if round_num < self._max_tool_rounds else None
            call_tool_calls: list[ToolCallRequest] = []
            call_text: list[str] = []
            try:
                # Only round 0 is safe to retry across providers on total failure (nothing
                # streamed yet — that fallback lives inside stream_complete itself). Every round
                # after that can no longer silently retry regardless of requires_pinning: prior
                # rounds' output is already visible to the client, so a silent restart would
                # produce a duplicated turn — the difference requires_pinning makes is *which*
                # handler(s) stream_complete is allowed to try before giving up, not whether a
                # failure past this point can retry silently.
                stream = self._llm.stream_complete(
                    messages, self._max_tokens, tools=tools_for_round,
                    pinned_handler=handler if requires_pinning else None,
                )
                async for h, event in stream:
                    handler = h
                    if isinstance(event, StreamError):
                        yield StreamFailed(message=event.message)
                        return
                    if isinstance(event, ToolCallRequest):
                        call_tool_calls.append(event)
                        continue
                    if isinstance(event, TextDelta):
                        call_text.append(event.text)
                    yield event
            except AllProvidersExhausted:
                if round_num == 0:
                    raise
                yield StreamFailed(message="the model provider failed after the tool call")
                return

            if not call_tool_calls:
                break

            requires_pinning = any(c.thought_signature for c in call_tool_calls)

            # Emit "started" up front (id/name/arguments are already known from the tool_calls
            # themselves) so the tool card can show a running state for the retrieval below,
            # instead of only ever appearing already-finished.
            for call in call_tool_calls:
                yield ToolCallStarted(id=call.id, name=call.name, arguments=call.arguments)

            round_executions, tool_result_messages, all_chunks, articles, article_index = await self._execute_tool_calls(
                call_tool_calls, all_chunks, articles, article_index
            )
            tool_executions.extend(round_executions)
            for execution in round_executions:
                yield ToolCallFinished(
                    id=execution.id, name=execution.name,
                    result_summary=execution.result_summary, is_error=execution.is_error,
                )

            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments, "thought_signature": c.thought_signature}
                    for c in call_tool_calls
                ],
            })
            messages.extend(tool_result_messages)
            round_num += 1

        # A round that used the search tool can still finish cleanly with zero text (see
        # _EMPTY_FOLLOWUP_REPLY) — without this, the turn would settle with an empty assistant
        # message and no visible explanation. Not applied to a round-0 reply with no tool calls
        # at all: that's a different failure mode (no search context established yet) this isn't
        # trying to cover. This text was never part of the model's own stream, so it's yielded
        # here as its own TextDelta rather than folded silently into call_text.
        reply = "".join(call_text)
        if round_num > 0 and not reply.strip():
            reply = _EMPTY_FOLLOWUP_REPLY
            yield TextDelta(text=reply)

        yield SourcesReady(articles=self._filter_cited_articles(reply, articles))

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

            all_chunks = pinned_chunks
            tool_executions: list[ToolCallExecution] = []
            handler: ProviderHandler | None = None
            result: LLMResult | None = None
            round_num = 0
            provider_failed = False
            # See _chat_pinned_stream's equivalent flag for the full reasoning — only a round
            # whose tool calls actually carried a thought_signature needs the next call pinned to
            # the exact same handler; otherwise the next call is free to rotate across every
            # other configured handler (same as round 0), so a quota-exhausted model doesn't
            # strand the turn while sibling models configured as fallbacks still have headroom.
            requires_pinning = False

            # See _chat_pinned_stream's equivalent loop for the full reasoning — this mirrors it
            # for the non-streaming path: the model gets up to self._max_tool_rounds turns with
            # the search tool available (rather than the old hardcoded single round trip), with
            # the last one forced tools=None so a final answer is guaranteed.
            while True:
                tools_for_round = [SEARCH_TOOL] if round_num < self._max_tool_rounds else None
                try:
                    if requires_pinning:
                        result, handler = await self._llm.complete(messages, self._max_tokens, tools=tools_for_round, pinned_handler=handler)
                    else:
                        result, handler = await self._llm.complete(messages, self._max_tokens, tools=tools_for_round, exclude=exclude)
                except AllProvidersExhausted:
                    if round_num == 0:
                        raise
                    if not requires_pinning:
                        # Not pinned to a single handler — complete() above already walked every
                        # non-excluded candidate internally before raising, so there's nothing
                        # left to retry with.
                        raise
                    # Pinned: only that one handler was tried, and prior rounds' tool calls are
                    # only usable with it (Gemini continuity — see thought_signature), so a
                    # failure here can't retry with a different one in place; restart the whole
                    # attempt from scratch excluding it instead.
                    exclude.add(handler.name)
                    provider_failed = True
                    break

                if not result.tool_calls:
                    break

                round_executions, tool_result_messages, all_chunks, articles, article_index = await self._execute_tool_calls(
                    result.tool_calls, all_chunks, articles, article_index
                )
                tool_executions.extend(round_executions)
                requires_pinning = any(c.thought_signature for c in result.tool_calls)
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments, "thought_signature": c.thought_signature}
                        for c in result.tool_calls
                    ],
                })
                messages.extend(tool_result_messages)
                round_num += 1

            if provider_failed:
                continue

            # See _EMPTY_FOLLOWUP_REPLY — a round that used the search tool can still finish
            # cleanly with zero text. Not applied to a round-0 reply with no tool calls at all:
            # that's a different failure mode (no search context established yet) this isn't
            # trying to cover.
            reply = result.text if (round_num == 0 or result.text.strip()) else _EMPTY_FOLLOWUP_REPLY
            articles_used = self._filter_cited_articles(reply, articles)
            return ChatResult(
                reply=reply,
                articles_used=articles_used,
                thinking=result.thinking,
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

        The returned list can be a non-contiguous subset (e.g. only [1] and [3] cited out of
        four context articles) — each ArticleRef.number still holds its original context index,
        so a consumer can resolve a citation correctly without relying on array position.

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
                number = len(seen) + 1
                seen[chunk.article_id] = ArticleRef(
                    id=chunk.article_id,
                    title=meta.get("title"),
                    url=meta.get("url") or "",
                    public_article_id=str(raw_pid) if raw_pid is not None else None,
                    number=number,
                )
                index[chunk.article_id] = number
        return list(seen.values()), index

    def _build_context(self, chunks: list[ChunkResult], article_index: dict[str, int]) -> str:
        """Groups chunks under one [N] header per article instead of repeating an identical
        header on every chunk. A single pinned article can alone contribute up to
        max_context_chunks chunks (see _fetch_pinned_chunks) — repeating "[1] Title" back to
        back reads visually like a list of N distinct numbered things, and models (observed with
        gemini-3.1-flash-lite after the primary model hit its daily quota, but not exclusive to
        it) still cite invented numbers like [3] or [9] based on chunk/paragraph position rather
        than the literal repeated [1] label, producing citations SourcesReady/_filter_cited_articles
        then can't map back to any real article. The trailing range line below restates the exact
        valid bound for *this* request (rather than SYSTEM_PROMPT/PINNED_SYSTEM_PROMPT's generic
        wording) as one more layer against that — it reduces but doesn't eliminate the hallucination,
        so the frontend (cited-content.tsx's parseInline) is the layer that actually guarantees an
        invented number never reaches the user as a broken-looking citation."""
        ordered = sorted(chunks, key=lambda c: article_index.get(c.article_id, 0))
        parts = []
        for article_id, group in groupby(ordered, key=lambda c: c.article_id):
            group_chunks = list(group)
            n = article_index.get(article_id, 0)
            title = group_chunks[0].article_metadata.get("title") or "Unknown"
            body = "\n---\n".join(c.content for c in group_chunks)
            parts.append(f"[{n}] {title}\n{body}")
        if article_index:
            max_n = max(article_index.values())
            bound = "1" if max_n == 1 else f"1 to {max_n}"
            parts.append(
                f"(This context contains exactly {max_n} article(s), numbered {bound}. "
                "A single article's content may be split into several paragraphs above — that "
                "does not make it multiple sources. Every citation you write must be one of "
                f"{bound}; never invent any other number.)"
            )
        return "\n\n".join(parts)
