from __future__ import annotations

import logging
from dataclasses import dataclass, field

from chatbot_plugin_sdk.contracts.responses import ChunkResult
from chatbot_plugin.llm.base import ResilientLLMService
from chatbot_plugin_sdk.processors.retrieve import RetrieveProcessor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research assistant that answers questions based ONLY on the
provided context chunks. Each chunk is annotated with its source article.

Rules:
- Answer using only the information in the context below.
- If the context does not contain enough information to answer, say so.
- Cite the source article title when referencing specific information.
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
    chunks: list[ChunkResult]


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

    async def chat(self, message: str, topic_id: str | None = None) -> ChatResult:
        search_result = await self._retriever.retrieve(
            message,
            top_k=self._max_context_chunks,
            min_score=self._min_score,
            min_rerank_score=self._min_rerank_score,
            topic_id=topic_id,
        )

        if not search_result.chunks:
            return ChatResult(reply=_NO_RELEVANT_INFO_REPLY, articles_used=[], chunks=[])

        context = self._build_context(search_result.chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {message}"},
        ]

        reply = await self._llm.complete(messages, self._max_tokens)

        if reply is None:
            reply = f"{context}\n\nQuestion: {message}"

        articles = self._collect_articles(search_result.chunks)

        return ChatResult(reply=reply, articles_used=articles, chunks=search_result.chunks)

    def _build_context(self, chunks: list[ChunkResult]) -> str:
        parts = []
        for chunk in chunks:
            title = chunk.article_title or "Unknown"
            parts.append(f"[source: {title}]\n{chunk.content}")
        return "\n\n".join(parts)

    def _collect_articles(self, chunks: list[ChunkResult]) -> list[ArticleRef]:
        seen: dict[str, ArticleRef] = {}
        for chunk in chunks:
            if chunk.article_id not in seen:
                seen[chunk.article_id] = ArticleRef(
                    id=chunk.article_id,
                    title=chunk.article_title,
                    url=chunk.article_url or "",
                    public_article_id=chunk.public_article_id,
                )
        return list(seen.values())
