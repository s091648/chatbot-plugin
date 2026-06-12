"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from chatbot_plugin.main import app as toolbox_app
from chatbot_plugin_sdk import RagQueryProcessor


@pytest.fixture
def app() -> FastAPI:
    """Return the toolbox FastAPI app with a mock processor attached."""
    processor = RagQueryProcessor()
    processor.configure(dbname="test_db", user="test", password="test")
    toolbox_app.state.processor = processor
    return toolbox_app


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async test client for the toolbox API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
