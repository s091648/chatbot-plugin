"""Chatbot Plugin — OpenAI-compatible RAG chat backend.

Run with::

    uvicorn chatbot_plugin.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from chatbot_plugin.config import (
    VECTOR_DB_NAME, VECTOR_DB_USER, VECTOR_DB_PASSWORD,
    VECTOR_DB_HOST, VECTOR_DB_PORT, VECTOR_DB_SCHEMA,
    VECTOR_DB_ARTICLES_TABLE, VECTOR_DB_CHUNKS_TABLE,
    RAG_GEMINI_API_KEY, GEMINI_MODEL, GEMINI_RPM,
    RAG_DENSE_MODEL, RAG_DENSE_DIMENSION,
    RAG_SPARSE_ENDPOINT_URL, RAG_SPARSE_DIMENSION,
    RAG_RERANKER_MODEL,
    CHATBOT_MAX_CONTEXT_CHUNKS, CHATBOT_MAX_TOKENS,
    CHATBOT_RETRIEVAL_MIN_SCORE, CHATBOT_RERANKER_MIN_SCORE,
    APP_ENV, GRAFANA_LOKI_URL, GRAFANA_LOKI_USER, GRAFANA_API_KEY,
)
from chatbot_plugin.observability import configure_logging
from chatbot_plugin.routers import api_router
from chatbot_plugin.services.chat_service import ChatService
from chatbot_plugin.llm.gemini_provider import GeminiProvider
from chatbot_plugin.llm.base import ResilientLLMService, ProviderHandler
from chatbot_plugin_sdk import (
    AsyncPgBackend,
    DatabaseConfig,
    EndpointProvider,
    FastEmbedReranker,
    GeminiDenseProvider,
    RetrieveProcessor,
    SlidingWindowStrategy,
)

configure_logging(
    service="chatbot-plugin",
    loki_url=GRAFANA_LOKI_URL,
    loki_user=GRAFANA_LOKI_USER,
    loki_api_key=GRAFANA_API_KEY,
    app_env=APP_ENV,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_config = DatabaseConfig(
        dbname=VECTOR_DB_NAME,
        user=VECTOR_DB_USER,
        password=VECTOR_DB_PASSWORD,
        host=VECTOR_DB_HOST,
        port=VECTOR_DB_PORT,
        schema=VECTOR_DB_SCHEMA,
        articles_table=VECTOR_DB_ARTICLES_TABLE,
        chunks_table=VECTOR_DB_CHUNKS_TABLE,
    )
    backend = AsyncPgBackend(db_config)

    dense = GeminiDenseProvider(
        api_key=RAG_GEMINI_API_KEY,
        model=RAG_DENSE_MODEL,
        dimension=RAG_DENSE_DIMENSION,
    )
    sparse = EndpointProvider(
        url=RAG_SPARSE_ENDPOINT_URL,
        response_key="sparse",
        dimension=RAG_SPARSE_DIMENSION,
    )
    reranker = FastEmbedReranker(model_name=RAG_RERANKER_MODEL)

    retriever = RetrieveProcessor()
    retriever.configure(backend=backend, dense=dense, sparse=sparse, reranker=reranker)

    gemini_llm = GeminiProvider(api_key=RAG_GEMINI_API_KEY, model=GEMINI_MODEL)
    llm_service = ResilientLLMService(handlers=[
        ProviderHandler(
            provider=gemini_llm,
            strategy=SlidingWindowStrategy(rpm=GEMINI_RPM),
            priority=1,
            name="gemini",
        )
    ])

    app.state.chat_service = ChatService(
        retriever=retriever,
        llm=llm_service,
        max_context_chunks=CHATBOT_MAX_CONTEXT_CHUNKS,
        max_tokens=CHATBOT_MAX_TOKENS,
        min_score=CHATBOT_RETRIEVAL_MIN_SCORE,
        min_rerank_score=CHATBOT_RERANKER_MIN_SCORE,
    )

    yield

    await backend.close()


app = FastAPI(
    title="Chatbot Plugin",
    description="Vector storage toolbox — OpenAI-compatible RAG chat API.",
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(api_router)
