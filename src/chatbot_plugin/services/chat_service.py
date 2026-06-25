from __future__ import annotations

import logging
from dataclasses import dataclass, field

from chatbot_plugin_sdk.contracts.responses import ChunkResult
from chatbot_plugin.llm.base import ResilientLLMService
from chatbot_plugin_sdk.processors.retrieve import RetrieveProcessor

logger = logging.getLogger(__name__)

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

_NO_RELEVANT_INFO_REPLY = (
    "I couldn't find relevant information in the database for your question. "
    "Please try rephrasing or ask about a different topic."
)

@dataclass
class ArticleRef:
    id: str
    title: str | None
    url: str
    public_article_id: str | None = None


@dataclass
class ChatResult:
    reply: str
    articles_used: list[ArticleRef]
    thinking: str | None = None
    chunks: list[ChunkResult] = field(default_factory=list)


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
        search_result = await self._retriever.retrieve(
            message,
            top_k=self._max_context_chunks,
            min_score=self._min_score,
            min_rerank_score=self._min_rerank_score,
            filters={"topic_id": topic_id} if topic_id else None,
        )

        pinned_chunks = await self._fetch_pinned_chunks(message, pinned_article_ids or [])

        # Pinned chunks first, then semantic results — dedup by chunk_id
        seen: set[str] = set()
        merged: list[ChunkResult] = []
        for chunk in pinned_chunks + search_result.chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)
        merged = merged[:self._max_context_chunks]

        if not merged:
            return ChatResult(reply=_NO_RELEVANT_INFO_REPLY, articles_used=[], chunks=[])

        articles, article_index = self._collect_articles(merged)
        context = self._build_context(merged, article_index)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {message}"},
        ]

        thinking, reply = await self._llm.complete(messages, self._max_tokens)
        return ChatResult(reply=reply, articles_used=articles, thinking=thinking, chunks=merged)

    async def _fetch_pinned_chunks(self, message: str, public_article_ids: list[str]) -> list[ChunkResult]:
        if not public_article_ids:
            return []

        # Allocate slots fairly across articles; minimum 3 chunks per article
        per_article_k = max(self._max_context_chunks // len(public_article_ids), 3)

        results: list[ChunkResult] = []
        for pid in public_article_ids:
            try:
                result = await self._retriever.retrieve(
                    message,
                    top_k=per_article_k,
                    min_score=self._min_score,
                    min_rerank_score=self._min_rerank_score,
                    filters={"public_article_id": pid},
                )
                results.extend(result.chunks)
            except Exception:
                logger.exception("pinned_chunk_retrieve_failed", extra={"article_id": pid})

        return results

    def _collect_articles(
        self, chunks: list[ChunkResult]
    ) -> tuple[list[ArticleRef], dict[str, int]]:
        """Returns (unique articles in first-appearance order, article_id → 1-based index)."""
        seen: dict[str, ArticleRef] = {}
        index: dict[str, int] = {}
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
