"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from chatbot_plugin.main import app as toolbox_app


@pytest.fixture
def app() -> FastAPI:
    """Return the toolbox FastAPI app."""
    return toolbox_app


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async test client for the toolbox API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
