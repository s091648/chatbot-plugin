"""FastAPI router for the toolbox.

Mounted under /tools in the standalone server (main.py).
External services POST chunk + embedding data here.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_plugin.contracts import (
    StoreChunksRequest,
    StoreChunksResponse,
)
from chatbot_plugin.db import get_db
from chatbot_plugin.service import ToolboxService

toolbox_router = APIRouter()


@toolbox_router.post("/chunks", status_code=201, response_model=StoreChunksResponse)
async def store_chunks(
    req: StoreChunksRequest,
    db: AsyncSession = Depends(get_db),
) -> StoreChunksResponse:
    """Store or update an article and its pre-chunked pre-embedded data."""
    service = ToolboxService(db)
    return await service.store_chunks(req)
