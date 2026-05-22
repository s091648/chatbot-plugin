"""Response contracts — mirrors specs/chat-api.md response bodies."""

from pydantic import BaseModel, Field


class ArticleRef(BaseModel):
    """Article reference used in chat reply citations."""

    id: str = Field(..., description="Article UUID")
    title: str = Field(..., description="Article title")


class ChatMessageResponse(BaseModel):
    """POST /chat/message response.

    Spec: specs/chat-api.md — POST /chat/message — 200
    """

    reply: str = Field(..., description="LLM-generated response")
    articles_used: list[ArticleRef] = Field(
        default_factory=list, description="Articles cited in the reply"
    )


class ChunkResult(BaseModel):
    """Single chunk from search results."""

    content: str = Field(..., description="Chunk text")
    article_id: str = Field(..., description="Parent article UUID")
    article_title: str = Field(..., description="Parent article title")
    score: float = Field(..., ge=0.0, le=1.0, description="RRF fusion score")


class SearchResponse(BaseModel):
    """POST /chat/search response.

    Spec: specs/chat-api.md — POST /chat/search — 200
    """

    chunks: list[ChunkResult] = Field(
        default_factory=list, description="Ranked search results"
    )


class IndexResponse(BaseModel):
    """POST /chat/index response.

    Spec: specs/chat-api.md — POST /chat/index — 202
    """

    job_id: str = Field(..., description="Background job UUID")
    status: str = Field(default="started", description="Always 'started'")


class StatusResponse(BaseModel):
    """GET /chat/status response.

    Spec: specs/chat-api.md — GET /chat/status — 200
    """

    total_chunks: int = Field(..., description="Total indexed chunks")
    last_indexed_at: str | None = Field(
        default=None, description="Last indexing time (ISO 8601), null if never"
    )
    pending_articles: int = Field(..., description="Articles not yet indexed")
