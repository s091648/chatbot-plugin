"""FastAPI router for the chatbot plugin.

Mount this router into the scrape-and-analyze backend to enable chat endpoints.

Usage::

    from chatbot_plugin.routers import chat_router
    app.include_router(chat_router, prefix="/chat", tags=["chat"])
"""

from fastapi import APIRouter

from chatbot_plugin.contracts import (
    ChatMessageRequest,
    ChatMessageResponse,
    SearchRequest,
    SearchResponse,
    IndexRequest,
    IndexResponse,
    StatusResponse,
)
from chatbot_plugin.service import ChatbotService

chat_router = APIRouter()
service = ChatbotService()


@chat_router.post("/message", response_model=ChatMessageResponse)
async def send_message(req: ChatMessageRequest) -> ChatMessageResponse:
    """Send a message to the chatbot and receive a response."""
    return await service.chat(req.message, req.user_id)


@chat_router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Pure hybrid search without LLM generation."""
    return await service.search(req.query, req.top_k, req.topic_id)


@chat_router.post("/index", status_code=202, response_model=IndexResponse)
async def trigger_index(req: IndexRequest) -> IndexResponse:
    """Trigger embedding indexing for articles."""
    return await service.trigger_index(req.article_id)


@chat_router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Check indexing status and vector store stats."""
    return await service.get_status()
