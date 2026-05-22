# Integration Specification

> How to install and integrate chatbot-plugin into scrape-and-analyze.

## Architecture: Independent Database

chatbot-plugin has its own PostgreSQL + pgvector database. It does **not** share scrape-and-analyze's database. This means:

- No schema coupling with the main app
- Can add pgvector extension independently
- Data sync between scrape-and-analyze → chatbot-plugin handled separately (Phase 2)

## Installation

```bash
# In scrape-and-analyze root
uv add chatbot-plugin
```

## Backend Mounting

In `backend/main.py`:

```python
from chatbot_plugin.routers import chat_router

app.include_router(chat_router, prefix="/chat", tags=["chat"])
```

This exposes all endpoints under `/chat/*`:
- `POST /chat/message`
- `POST /chat/search`
- `POST /chat/index`
- `GET /chat/status`

## Environment Variables

All variables use the `CHATBOT_` prefix. Add to scrape-and-analyze's `.env`:

```bash
# Database (independent PG + pgvector)
CHATBOT_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin

# LLM
CHATBOT_LLM_PROVIDER=claude
CHATBOT_LLM_MODEL=claude-sonnet-4-6-20250514

# Behavior
CHATBOT_MAX_CONTEXT_ARTICLES=10
CHATBOT_MAX_CONTEXT_TOKENS=8000

# Embedding (Phase 2+)
CHATBOT_EMBEDDING_MODEL=BAAI/bge-m3
CHATBOT_EMBEDDING_DIMENSION=1024
CHATBOT_RRF_K=60
CHATBOT_CHUNK_SIZE=512
CHATBOT_CHUNK_OVERLAP=50
```

## Database

- **Independent PostgreSQL + pgvector instance** (not shared with scrape-and-analyze)
- Connection via `CHATBOT_DATABASE_URL` (async driver: `asyncpg`)
- Plugin adds `article_chunks` table (see `specs/rag-pipeline.md`)
- Requires `pgvector` extension: `CREATE EXTENSION IF NOT EXISTS vector;`
- Migration via Alembic (Phase 2)
- Data sync from scrape-and-analyze handled by indexer (Phase 2)

## Dependencies

chatbot-plugin declares these runtime dependencies:
- `scrape-analyzer` — shared models and DB access
- `fastapi` — router mounting
- `pydantic>=2.0` — settings and contracts
- `structlog` — structured logging
- `sqlalchemy` — DB queries
- `anthropic` — Claude API (default LLM provider)

Phase 2 adds:
- `langchain`, `langchain-anthropic`, `langchain-google-genai`
- `sentence-transformers` — BGE-M3 embedding

## Frontend

Add a `/chat` page in the existing Next.js app. API shape is defined in `specs/chat-api.md`.

Key integration points:
1. **Chat page** — `POST /chat/message`, stream display of `reply`, citation cards from `articles_used`
2. **Search-only mode** — `POST /chat/search`, display `chunks` with scores
3. **Index management** — `POST /chat/index` + `GET /chat/status` for admin UI
