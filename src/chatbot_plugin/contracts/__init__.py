"""Pydantic contracts — structured versions of specs/chat-api.md.

These models are the source of truth for request/response validation.
They must stay in sync with specs/chat-api.md. If the spec changes,
update the corresponding model here first.
"""

from chatbot_plugin.contracts.requests import (
    ChatMessageRequest,
    SearchRequest,
    IndexRequest,
)
from chatbot_plugin.contracts.responses import (
    ChatMessageResponse,
    ArticleRef,
    SearchResponse,
    ChunkResult,
    IndexResponse,
    StatusResponse,
)

__all__ = [
    "ChatMessageRequest",
    "SearchRequest",
    "IndexRequest",
    "ChatMessageResponse",
    "ArticleRef",
    "SearchResponse",
    "ChunkResult",
    "IndexResponse",
    "StatusResponse",
]
