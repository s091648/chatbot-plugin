"""Toolbox standalone server.

Run with::

    uvicorn chatbot_plugin.main:app --reload

Or via the entry point::

    chatbox-toolbox
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from chatbot_plugin.db import init_db
from chatbot_plugin.routers import toolbox_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Toolbox",
    description="Vector storage toolbox — receives pre-chunked, pre-embedded data and serves retrieval APIs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(toolbox_router, prefix="/tools", tags=["tools"])
