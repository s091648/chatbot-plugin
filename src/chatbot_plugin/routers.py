"""FastAPI router for the chatbot plugin.

Mount this router into the scrape-and-analyze backend to enable chat endpoints.

Usage::

    from chatbot_plugin.routers import chat_router
    app.include_router(chat_router, prefix="/chat", tags=["chat"])
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_plugin.contracts import (
    ChatMessageRequest,
    ChatMessageResponse,
    SearchRequest,
    SearchResponse,
    IndexRequest,
    IndexResponse,
    StatusResponse,
)
from chatbot_plugin.db import get_db
from chatbot_plugin.llm.resilient_llm_service import ResilientLLMService
from chatbot_plugin.service import ChatbotService

chat_router = APIRouter()

# Module-level LLM service, initialized at app startup
_llm_service: ResilientLLMService | None = None


def init_llm_service() -> None:
    """Initialize the LLM service. Call once at app startup."""
    global _llm_service
    from chatbot_plugin.llm.bootstrap import build_llm_service
    _llm_service = build_llm_service()


def _service(db: AsyncSession = Depends(get_db)) -> ChatbotService:
    if _llm_service is None:
        init_llm_service()
    return ChatbotService(db, _llm_service)


def set_llm_service(service: ResilientLLMService) -> None:
    """Set the LLM service explicitly. Used for testing."""
    global _llm_service
    _llm_service = service


@chat_router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    req: ChatMessageRequest,
    service: ChatbotService = Depends(_service),
) -> ChatMessageResponse:
    """Send a message to the chatbot and receive a response."""
    return await service.chat(req.message, req.user_id)


@chat_router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    service: ChatbotService = Depends(_service),
) -> SearchResponse:
    """Pure full-text search without LLM generation."""
    return await service.search(req.query, req.top_k, req.topic_id)


@chat_router.post("/index", status_code=202, response_model=IndexResponse)
async def trigger_index(
    req: IndexRequest,
    service: ChatbotService = Depends(_service),
) -> IndexResponse:
    """Trigger embedding indexing for articles."""
    return await service.trigger_index(req.article_id)


@chat_router.get("/status", response_model=StatusResponse)
async def get_status(
    service: ChatbotService = Depends(_service),
) -> StatusResponse:
    """Check indexing status and vector store stats."""
    return await service.get_status()
