"""Chatbot service — core chat logic.

Orchestrates context retrieval and LLM calls.
Spec reference: specs/chat-api.md
"""

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_plugin.config import settings
from chatbot_plugin.contracts import (
    ChatMessageResponse,
    SearchResponse,
    IndexResponse,
    StatusResponse,
    ArticleRef,
    ChunkResult,
)
from chatbot_plugin.llm.resilient_llm_service import ResilientLLMService
from chatbot_plugin.rag.chain import rag_generate
from chatbot_plugin.rag.retriever import Retriever


class ChatbotService:
    """Handles message processing: context retrieval + LLM response generation."""

    def __init__(self, db: AsyncSession, llm_service: ResilientLLMService) -> None:
        self.db = db
        self.llm_service = llm_service
        self.retriever = Retriever(db)
        self._indexing_articles: set[str] = set()

    async def chat(self, message: str, user_id: str | None = None) -> ChatMessageResponse:
        """Process a user message and return a chatbot reply.

        Args:
            message: The user's input message.
            user_id: Optional user identifier for per-user context.

        Returns:
            ChatMessageResponse with reply and articles_used.

        Raises:
            HTTPException: 503 if LLM provider is unavailable.
        """
        from fastapi import HTTPException

        # 1. Retrieve relevant articles via full-text search
        articles = await self.retriever.search(message, limit=settings.max_context_articles)

        # 2. Generate reply via RAG chain
        try:
            reply = await rag_generate(message, articles, self.llm_service)
        except RuntimeError:
            raise HTTPException(status_code=503, detail="LLM provider unavailable")
        except (ConnectionError, TimeoutError) as e:
            raise HTTPException(status_code=503, detail="LLM provider unavailable") from e

        # 3. Build response with article references
        articles_used = [
            ArticleRef(id=str(a["id"]), title=a["title"] or "Untitled")
            for a in articles
        ]
        return ChatMessageResponse(reply=reply, articles_used=articles_used)

    async def search(
        self, query: str, top_k: int = 10, topic_id: str | None = None
    ) -> SearchResponse:
        """Pure full-text search without LLM generation.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            topic_id: Optional topic filter.

        Returns:
            SearchResponse with ranked chunks (Phase 1: whole articles as chunks).
        """
        articles = await self.retriever.search(query, limit=top_k, topic_id=topic_id)
        chunks = [
            ChunkResult(
                content=a["content"] or "",
                article_id=str(a["id"]),
                article_title=a["title"] or "Untitled",
                score=a["rank"],
            )
            for a in articles
        ]
        return SearchResponse(chunks=chunks)

    async def trigger_index(self, article_id: str | None = None) -> IndexResponse:
        """Trigger embedding indexing (Phase 2 implementation).

        Args:
            article_id: Single article to index, or None for all unindexed.

        Returns:
            IndexResponse with job_id and status.

        Raises:
            HTTPException: 404 if article_id not found, 409 if already indexing.
        """
        from fastapi import HTTPException

        if article_id is not None:
            if article_id in self._indexing_articles:
                raise HTTPException(status_code=409, detail="Indexing already in progress")
            result = await self.db.execute(
                text("SELECT id FROM articles WHERE id = :id"),
                {"id": article_id},
            )
            if result.scalar() is None:
                raise HTTPException(status_code=404, detail="Article not found")
            self._indexing_articles.add(article_id)

        # Phase 1: stub response. Phase 2 will implement background indexing.
        job_id = str(uuid.uuid4())

        # Clean up tracking set (Phase 2 will remove after background task completes)
        if article_id is not None:
            self._indexing_articles.discard(article_id)

        return IndexResponse(job_id=job_id)

    async def get_status(self) -> StatusResponse:
        """Get indexing status and vector store stats.

        Returns:
            StatusResponse with total_chunks, last_indexed_at, pending_articles.
        """
        # Phase 1: no indexer, no chunks. Return zeros.
        return StatusResponse(
            total_chunks=0,
            last_indexed_at=None,
            pending_articles=0,
        )
