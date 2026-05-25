"""Chatbot service — core chat logic.

Orchestrates context retrieval and LLM calls.
Spec reference: specs/chat-api.md
"""

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
from chatbot_plugin.rag.chain import rag_generate


class ChatbotService:
    """Handles message processing: context retrieval + LLM response generation."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

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
        articles = await self._search_articles(message, limit=settings.max_context_articles)

        # 2. Generate reply via RAG chain
        try:
            reply = await rag_generate(message, articles)
        except Exception as e:
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
        articles = await self._search_articles(query, limit=top_k, topic_id=topic_id)
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
            result = await self.db.execute(
                text("SELECT id FROM articles WHERE id = :id"),
                {"id": article_id},
            )
            if result.scalar() is None:
                raise HTTPException(status_code=404, detail="Article not found")

        # Phase 1: stub response. Phase 2 will implement background indexing.
        return IndexResponse(job_id="stub-job-id")

    async def get_status(self) -> StatusResponse:
        """Get indexing status and vector store stats.

        Returns:
            StatusResponse with total_chunks, last_indexed_at, pending_articles.
        """
        # Phase 1: return article count as proxy. Phase 2 will query article_chunks.
        result = await self.db.execute(text("SELECT count(*) FROM articles"))
        count = result.scalar() or 0
        return StatusResponse(
            total_chunks=0,
            last_indexed_at=None,
            pending_articles=count,
        )

    async def _search_articles(
        self, query: str, limit: int = 10, topic_id: str | None = None
    ) -> list[dict]:
        """Full-text search on articles using PostgreSQL tsvector.

        Args:
            query: Search query string.
            limit: Max results.
            topic_id: Optional topic filter.

        Returns:
            List of article dicts with id, title, content, rank.
        """
        params: dict = {"query": query, "limit": limit}
        topic_filter = ""
        if topic_id is not None:
            topic_filter = "AND topic_id = :topic_id"
            params["topic_id"] = topic_id

        sql = text(f"""
            SELECT id, title, content,
                   ts_rank(to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,'')),
                           plainto_tsquery('english', :query)) AS rank
            FROM articles
            WHERE to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
                  @@ plainto_tsquery('english', :query)
              {topic_filter}
            ORDER BY rank DESC
            LIMIT :limit
        """)

        result = await self.db.execute(sql, params)
        rows = result.mappings().all()
        return [dict(row) for row in rows]
