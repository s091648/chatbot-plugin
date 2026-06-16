# Integration Specification

> How external services interact with the chatbot plugin.

## Architecture

The chatbot plugin runs as a standalone FastAPI service backed by PostgreSQL + pgvector.

- Standalone FastAPI server (not embedded in other services)
- Owns its own database — no schema coupling with other services
- Ingestion is done **in-process via the SDK** (`IngestProcessor`), not via HTTP endpoints
- The chat API is OpenAI-compatible: `POST /v1/chat/completions`

## Running the Service

```bash
# Start the server
uvicorn chatbot_plugin.main:app --reload

# Or set the host/port
CHATBOT_DATABASE_URL=postgresql+asyncpg://... uvicorn chatbot_plugin.main:app --host 0.0.0.0 --port 8000
```

## Ingestion: SDK (In-Process)

Ingestion is handled in-process by the SDK's `IngestProcessor`. There is no HTTP ingestion endpoint.

```python
from chatbot_plugin_sdk import IngestProcessor, AsyncPgBackend, EndpointProvider

backend = AsyncPgBackend(database_url="postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin")
provider = EndpointProvider(embedding_api="http://localhost:8080")
ingest = IngestProcessor(backend=backend, provider=provider)

await ingest.ingest(
    full_text="Retrieval augmented generation is a technique ...",
    metadata={"url": "https://example.com/article", "title": "RAG 101"},
)
```

The `IngestProcessor` handles: text chunking → embedding generation → upsert into PostgreSQL + pgvector.

## Chat API: HTTP

External clients call the chat endpoint directly:

```python
import httpx

async def chat():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "chatbot-plugin",
                "messages": [
                    {"role": "user", "content": "Explain RAG in simple terms"}
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        print(data["choices"][0]["message"]["content"])
```

The endpoint is OpenAI-compatible — any OpenAI SDK client can point to this service.

## SDK Dependency

This service depends on `chatbot-plugin-sdk`, which provides:

| Symbol | Description |
|--------|-------------|
| `RetrieveProcessor` | Retrieval with score gating and optional reranking. Called as `retrieve(query, top_k, min_score, min_rerank_score)` |
| `AsyncPgBackend` | Async PostgreSQL connection backend |
| `EndpointProvider` | Embedding + reranker HTTP client |
| `SlidingWindowStrategy` | Chunking strategy for ingestion |
| `RateLimitExhausted` | Exception raised when a provider rate limit is hit |

`RetrieveProcessor` and `AsyncPgBackend` are wired up at startup in `main.py` lifespan and injected into `ChatService`.

## Environment Variables

All variables use the `CHATBOT_` prefix:

```bash
# Database
CHATBOT_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin

# Embedding + retrieval
CHATBOT_EMBEDDING_MODEL_API=http://localhost:8080
CHATBOT_ENABLE_RERANKER=true
CHATBOT_RETRIEVAL_MIN_SCORE=0.0
CHATBOT_RERANKER_MIN_SCORE=0.7
CHATBOT_MAX_CONTEXT_CHUNKS=10

# LLM providers (fallback chain: Claude → Gemini → OpenRouter)
CHATBOT_CLAUDE_API_KEY=sk-ant-api-...
CHATBOT_CLAUDE_MODEL=claude-sonnet-4-6-20250514

CHATBOT_GEMINI_API_KEY=...
CHATBOT_GEMINI_MODEL=gemini-2.0-flash

CHATBOT_OPENROUTER_API_KEY=...
CHATBOT_OPENROUTER_MODEL=meta-llama/llama-3-70b

CHATBOT_MAX_TOKENS=2048
```

## Database

- **Independent PostgreSQL + pgvector instance**
- Connection via `CHATBOT_DATABASE_URL` (async driver: `asyncpg`)
- Schema managed by SQLAlchemy models + Alembic
- Extension required: `CREATE EXTENSION IF NOT EXISTS vector;`

### Tables

```sql
CREATE TABLE articles (
    id          UUID PRIMARY KEY,
    url         VARCHAR NOT NULL UNIQUE,
    title       VARCHAR,
    source      VARCHAR,
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE article_chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id    UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    dense_vector  vector(1024),
    sparse_vector sparsevec(250002),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, chunk_index)
);

-- HNSW index for dense similarity search
CREATE INDEX hnsw_chunks_dense ON article_chunks
    USING hnsw (dense_vector vector_cosine_ops);
```

## Dependencies

- `fastapi` — HTTP server
- `pydantic>=2.0` — settings and contracts
- `structlog` — structured logging
- `sqlalchemy[asyncio]>=2.0` — ORM
- `asyncpg` — async PostgreSQL driver
- `pgvector` — SQLAlchemy/pgvector integration
- `alembic` — database migrations
- `anthropic` — Claude API client
- `google-generativeai` — Gemini API client
- `httpx` — async HTTP client (OpenRouter + embedding service)
- `chatbot-plugin-sdk` — `RetrieveProcessor`, `AsyncPgBackend`, `EndpointProvider`, `SlidingWindowStrategy`
