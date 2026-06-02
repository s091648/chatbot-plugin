"""Pydantic contracts — structured versions of specs/toolbox-api.md.

These models are the source of truth for request/response validation.
They must stay in sync with specs/toolbox-api.md. If the spec changes,
update the corresponding model here first.
"""

from chatbot_plugin.contracts.requests import ArticleInfo, ChunkData, StoreChunksRequest
from chatbot_plugin.contracts.responses import StoreChunksResponse

__all__ = [
    "ArticleInfo",
    "ChunkData",
    "StoreChunksRequest",
    "StoreChunksResponse",
]
