"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_plugin.routers import chat_router
from chatbot_plugin.service import ChatbotService
from fastapi import FastAPI


@pytest.fixture
def mock_db() -> AsyncMock:
    """Mock async DB session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def service(mock_db: AsyncMock) -> ChatbotService:
    """ChatbotService with mocked DB."""
    return ChatbotService(mock_db)


@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI app with chat routes mounted."""
    app = FastAPI()
    app.include_router(chat_router, prefix="/chat", tags=["chat"])
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async test client for the chat API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
