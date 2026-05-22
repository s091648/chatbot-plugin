"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient

from chatbot_plugin.routers import chat_router
from fastapi import FastAPI


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
