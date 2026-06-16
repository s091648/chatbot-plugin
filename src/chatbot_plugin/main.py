"""Toolbox standalone server — OpenAI-compatible backend.

Run with::

    uvicorn chatbot_plugin.main:app --reload
"""

from contextlib import asynccontextmanager
from os import getenv
from urllib.parse import urlparse

from fastapi import FastAPI

from chatbot_plugin.routers import api_router
from chatbot_plugin.chat_service import ChatService
from chatbot_plugin_sdk import RetrieveProcessor


def _get_db_url() -> str:
    """Read database URL from environment, fallback to local default."""
    return getenv(
        "CHATBOT_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = RetrieveProcessor()
    # NOTE: retriever.configure() requires a real backend; skipped here for startup.
    # ChatService and retriever should be configured via dependency injection in production.
    app.state.chat_service = ChatService(retriever=retriever, llm=None)  # type: ignore[arg-type]

    yield


app = FastAPI(
    title="Toolbox",
    description="Vector storage toolbox — OpenAI-compatible RAG chat API.",
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(api_router)
