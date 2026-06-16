"""Shared test fixtures."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from chatbot_plugin.main import app as toolbox_app
from chatbot_plugin.chat_service import ChatService


@pytest.fixture
def app() -> FastAPI:
    """Return the toolbox FastAPI app with a mock ChatService attached."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="mock reply")
    mock_retriever = MagicMock()
    from chatbot_plugin_sdk.contracts.responses import SearchResponse
    mock_retriever.retrieve = AsyncMock(return_value=SearchResponse(chunks=[]))
    chat_service = ChatService(retriever=mock_retriever, llm=mock_llm)
    toolbox_app.state.chat_service = chat_service
    return toolbox_app


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async test client for the toolbox API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
