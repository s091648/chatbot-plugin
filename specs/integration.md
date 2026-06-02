# Integration Specification

> How external services interact with the toolbox.

## Architecture: Standalone Service

The toolbox runs as its own independent service with its own PostgreSQL + pgvector database.

- Standalone FastAPI server (not embedded in scrape-and-analyze)
- Owns its own database — no schema coupling with other services
- External services (e.g. scrape-and-analyze) send chunk + embedding data via HTTP
- Chat/retrieval APIs will be added in a later phase

## Running the Toolbox

```bash
# Start the server
uvicorn chatbot_plugin.main:app --reload

# Or set the host/port
CHATBOT_DATABASE_URL=postgresql+asyncpg://... uvicorn chatbot_plugin.main:app --host 0.0.0.0 --port 8000
```

## How External Services Send Data

Scrape-and-analyze (or any service) POSTs to the toolbox:

```python
import httpx

async def send_chunks():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://toolbox:8000/tools/chunks",
            json={
                "article": {"id": "...", "url": "..."},
                "chunks": [
                    {"chunk_index": 0, "content": "...", "dense_vector": [...], "sparse_vector": {...}},
                ],
            },
        )
        assert resp.status_code == 201
```

This exposes:
- `POST /tools/chunks` — store article + chunks

## Environment Variables

All variables use the `CHATBOT_` prefix:

```bash
# Database (independent PG + pgvector)
CHATBOT_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin

# Embedding model config (must match what scrape-and-analyze uses)
CHATBOT_EMBEDDING_MODEL=BAAI/bge-m3
CHATBOT_EMBEDDING_DIMENSION=1024
```

## Database

- **Independent PostgreSQL + pgvector instance** (not shared with scrape-and-analyze)
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
    sparse_vector JSONB,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, chunk_index)
);

-- HNSW index for dense similarity search
CREATE INDEX hnsw_chunks_dense ON article_chunks
    USING hnsw (dense_vector vector_cosine_ops);
```

## Dependencies

- `fastapi` — router mounting
- `pydantic>=2.0` — settings and contracts
- `structlog` — structured logging
- `sqlalchemy[asyncio]>=2.0` — ORM
- `asyncpg` — async PostgreSQL driver
- `pgvector` — SQLAlchemy/pgvector integration
- `httpx` — async HTTP client (if needed for future outbound calls)
- `alembic` — database migrations
