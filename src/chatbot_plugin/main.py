"""Toolbox standalone server — OpenAI-compatible backend.

Run with::

    uvicorn chatbot_plugin.main:app --reload
"""

from contextlib import asynccontextmanager
from os import getenv
from urllib.parse import urlparse

from fastapi import FastAPI

from chatbot_plugin.routers import api_router
from chatbot_plugin_sdk import RagQueryProcessor


def _get_db_url() -> str:
    """Read database URL from environment, fallback to local default."""
    return getenv(
        "CHATBOT_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = _get_db_url()
    embedding_api = getenv("CHATBOT_EMBEDDING_MODEL_API", "")

    parsed = urlparse(db_url)
    dbname = parsed.path.lstrip("/")
    user = parsed.username or "postgres"
    password = parsed.password or ""
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432

    processor = RagQueryProcessor()
    processor.configure(
        dbname=dbname,
        user=user,
        password=password,
        embedding_model_api=embedding_api or None,
        host=host,
        port=port,
    )
    app.state.processor = processor

    yield


app = FastAPI(
    title="Toolbox",
    description="Vector storage toolbox — OpenAI-compatible RAG chat API.",
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(api_router)
