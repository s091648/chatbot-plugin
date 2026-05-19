"""Chatbot service — core chat logic.

Orchestrates context retrieval (from scrape-and-analyze articles) and LLM calls.
"""

from chatbot_plugin.config import settings


class ChatbotService:
    """Handles message processing: context retrieval + LLM response generation."""

    async def chat(self, message: str, user_id: str | None = None) -> dict:
        """Process a user message and return a chatbot reply.

        Args:
            message: The user's input message.
            user_id: Optional user identifier for per-user context.

        Returns:
            Dict with "reply" and optional "articles_used" metadata.
        """
        # TODO: retrieve relevant articles from scrape-and-analyze DB
        # TODO: build prompt with article context
        # TODO: call LLM provider
        return {
            "reply": f"[chatbot-plugin] Not yet implemented. Message: {message}",
            "articles_used": 0,
        }
