"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_plugin.routers import chat_router, set_llm_service
from chatbot_plugin.service import ChatbotService
from chatbot_plugin.llm.resilient_llm_service import ResilientLLMService
from fastapi import FastAPI


@pytest.fixture
def mock_db() -> AsyncMock:
    """Mock async DB session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    """Mock ResilientLLMService."""
    service = AsyncMock(spec=ResilientLLMService)
    service.generate.return_value = "Mocked LLM reply"
    return service


@pytest.fixture
def service(mock_db: AsyncMock, mock_llm_service: AsyncMock) -> ChatbotService:
    """ChatbotService with mocked DB and LLM."""
    return ChatbotService(mock_db, mock_llm_service)


@pytest.fixture
def app(mock_llm_service: AsyncMock) -> FastAPI:
    """Create a test FastAPI app with chat routes and mock LLM service."""
    set_llm_service(mock_llm_service)
    app = FastAPI()
    app.include_router(chat_router, prefix="/chat", tags=["chat"])
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async test client for the chat API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
