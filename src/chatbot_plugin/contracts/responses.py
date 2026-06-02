"""Response contracts — mirrors specs/toolbox-api.md response bodies."""

from pydantic import BaseModel, Field


class StoreChunksResponse(BaseModel):
    """POST /tools/chunks response.

    Spec: specs/toolbox-api.md — POST /tools/chunks — 201
    """

    stored: int = Field(..., ge=0, description="Number of chunks stored")
    article_id: str = Field(..., description="The article UUID (echoed back)")
