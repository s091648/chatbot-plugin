# CLAUDE.md

## Project Overview

Vector storage toolbox — standalone service that receives pre-chunked, pre-embedded article data from external services and stores it in PostgreSQL + pgvector. Will also serve chat/retrieval APIs for frontend UI. Two-person team: one frontend, one backend.

## SDD (Spec-Driven Development) Workflow

**Spec is source of truth. Code follows spec, not the other way around.**

### Rules

1. **Spec-first**: Before writing or changing any code, read and update `specs/` first
2. **Spec files**:
   - `specs/toolbox-api.md` — API contract (shared with external services)
   - `specs/rag-pipeline.md` — Storage internals (backend only)
   - `specs/integration.md` — How external services integrate with the toolbox
3. **Contracts = spec in code**: `src/chatbot_plugin/contracts/` contains Pydantic models that mirror the spec. If spec changes, update contracts first
4. **API changes must start from spec**: Change `specs/toolbox-api.md` → update `contracts/` → update router/service → tests pass
5. **Notify the other person**: If you change an API shape, the other developer needs to know. Spec is the communication channel
6. **Contract tests must pass**: `src/tests/contracts/` verifies Pydantic models match spec. These are non-negotiable

### Development Order

```
1. Write/update specs/*.md
2. Write/update contracts/*.py (Pydantic models from spec)
3. Write/update tests/ (based on spec cases)
4. Implement routers/ + service/ (make tests green)
```

### Scrape-and-Analyze Boundary

- External services only need `specs/toolbox-api.md` — do not look at backend implementation
- Backend can change internals freely as long as API shape (spec) is unchanged
- Disputes are resolved by reading the spec, not the code

## Project Structure

```
specs/                           # SDD spec files (source of truth)
  toolbox-api.md                 # API contract (POST /chunks)
  rag-pipeline.md                # Storage internals (backend only)
  integration.md                 # How external services integrate
src/chatbot_plugin/
  contracts/                     # Pydantic models = spec in code
    requests.py                  # ArticleInfo, ChunkData, StoreChunksRequest
    responses.py                 # StoreChunksResponse
  models/                        # SQLAlchemy ORM models
    article.py                   # Article + DeclarativeBase
    chunk.py                     # ArticleChunk (pgvector Vector column)
  config.py                      # CHATBOT_* env vars
  db.py                          # Async engine + session factory
  main.py                        # Standalone FastAPI app + lifespan
  routers.py                     # FastAPI endpoints
  service.py                     # Business logic (ToolboxService)
src/tests/
  contracts/                     # Contract conformance tests
  routers/                       # API endpoint tests
  test_service.py                # Unit tests
alembic/                         # Database migrations
```

## Commands

- **Run server:** `uvicorn chatbot_plugin.main:app --reload`
- **Test:** `uv run pytest src/tests/ -v`
- **Coverage:** `uv run pytest src/tests/ --cov=chatbot_plugin --cov-report=html`
