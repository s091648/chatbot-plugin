"""Toolbox standalone server — OpenAI-compatible backend.

Run with::

    uvicorn chatbot_plugin.main:app --reload
"""

from contextlib import asynccontextmanager
from os import getenv

from fastapi import FastAPI

from chatbot_plugin.routers import api_router
from chatbot_plugin.chat_service import ChatService
from chatbot_plugin.llm.gemini_provider import GeminiProvider
from chatbot_plugin.llm.base import ResilientLLMService, ProviderHandler
from chatbot_plugin_sdk import (
    AsyncPgBackend,
    DatabaseConfig,
    FastEmbedReranker,
    FastEmbedSparseProvider,
    GeminiDenseProvider,
    RetrieveProcessor,
    SlidingWindowStrategy,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_config = DatabaseConfig(
        dbname=getenv("VECTOR_DB_NAME", ""),
        user=getenv("VECTOR_DB_USER", ""),
        password=getenv("VECTOR_DB_PASSWORD", ""),
        host=getenv("VECTOR_DB_HOST", "localhost"),
        port=int(getenv("VECTOR_DB_PORT", "5432")),
        schema=getenv("VECTOR_DB_SCHEMA", "vectors"),
        articles_table=getenv("VECTOR_DB_ARTICLES_TABLE", "articles"),
        chunks_table=getenv("VECTOR_DB_CHUNKS_TABLE", "article_chunks"),
    )
    backend = AsyncPgBackend(db_config)

    gemini_api_key = getenv("RAG_GEMINI_API_KEY", "")
    dense = GeminiDenseProvider(
        api_key=gemini_api_key,
        model=getenv("RAG_DENSE_MODEL", "gemini-embedding-001"),
        dimension=int(getenv("RAG_DENSE_DIMENSION", "768")),
    )
    sparse = FastEmbedSparseProvider(
        model=getenv("RAG_SPARSE_MODEL", "prithvida/Splade_PP_en_v1"),
        dimension=int(getenv("RAG_SPARSE_DIMENSION", "30522")),
    )
    reranker = FastEmbedReranker(
        model_name=getenv("RAG_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    )

    retriever = RetrieveProcessor()
    retriever.configure(backend=backend, dense=dense, sparse=sparse, reranker=reranker)

    gemini_llm = GeminiProvider(
        api_key=gemini_api_key,
        model=getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )
    llm_service = ResilientLLMService(handlers=[
        ProviderHandler(
            provider=gemini_llm,
            strategy=SlidingWindowStrategy(rpm=60),
            priority=1,
            name="gemini",
        )
    ])

    app.state.chat_service = ChatService(retriever=retriever, llm=llm_service)

    yield

    await backend.close()


app = FastAPI(
    title="Chatbot Plugin",
    description="Vector storage toolbox — OpenAI-compatible RAG chat API.",
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(api_router)
