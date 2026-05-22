"""Chatbot service — core chat logic.

Orchestrates context retrieval (from scrape-and-analyze articles) and LLM calls.
"""

from chatbot_plugin.config import settings
from chatbot_plugin.contracts import (
    ChatMessageResponse,
    SearchResponse,
    IndexResponse,
    StatusResponse,
)


class ChatbotService:
    """Handles message processing: context retrieval + LLM response generation."""

    async def chat(self, message: str, user_id: str | None = None) -> ChatMessageResponse:
        """Process a user message and return a chatbot reply.

        Args:
            message: The user's input message.
            user_id: Optional user identifier for per-user context.

        Returns:
            ChatMessageResponse with reply and articles_used.
        """
        # TODO: retrieve relevant articles from scrape-and-analyze DB
        # TODO: build prompt with article context
        # TODO: call LLM provider
        return ChatMessageResponse(
            reply=f"[chatbot-plugin] Not yet implemented. Message: {message}",
            articles_used=[],
        )

    async def search(
        self, query: str, top_k: int = 10, topic_id: str | None = None
    ) -> SearchResponse:
        """Pure hybrid search without LLM generation.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            topic_id: Optional topic filter.

        Returns:
            SearchResponse with ranked chunks.
        """
        # TODO: implement hybrid search (dense + sparse + RRF)
        return SearchResponse(chunks=[])

    async def trigger_index(self, article_id: str | None = None) -> IndexResponse:
        """Trigger embedding indexing.

        Args:
            article_id: Single article to index, or None for all unindexed.

        Returns:
            IndexResponse with job_id and status.
        """
        # TODO: implement background indexing
        return IndexResponse(job_id="stub-job-id")

    async def get_status(self) -> StatusResponse:
        """Get indexing status and vector store stats.

        Returns:
            StatusResponse with total_chunks, last_indexed_at, pending_articles.
        """
        # TODO: query actual stats from DB
        return StatusResponse(
            total_chunks=0,
            last_indexed_at=None,
            pending_articles=0,
        )
