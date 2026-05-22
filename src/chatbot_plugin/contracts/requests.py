"""Request contracts — mirrors specs/chat-api.md request bodies."""

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    """POST /chat/message request.

    Spec: specs/chat-api.md — POST /chat/message
    """

    message: str = Field(..., min_length=1, max_length=2000, description="User input")
    user_id: str | None = Field(default=None, description="Optional user identifier")


class SearchRequest(BaseModel):
    """POST /chat/search request.

    Spec: specs/chat-api.md — POST /chat/search
    """

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    topic_id: str | None = Field(default=None, description="Filter by topic UUID")


class IndexRequest(BaseModel):
    """POST /chat/index request.

    Spec: specs/chat-api.md — POST /chat/index
    """

    article_id: str | None = Field(
        default=None, description="Single article UUID, null = index all unindexed"
    )
