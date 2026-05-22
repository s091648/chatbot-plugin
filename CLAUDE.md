# CLAUDE.md

## Project Overview

Pluggable RAG-enabled chatbot for scrape-and-analyze. Two-person team: one frontend, one backend.

## SDD (Spec-Driven Development) Workflow

**Spec is source of truth. Code follows spec, not the other way around.**

### Rules

1. **Spec-first**: Before writing or changing any code, read and update `specs/` first
2. **Spec files**:
   - `specs/chat-api.md` — API contract (frontend + backend shared)
   - `specs/rag-pipeline.md` — RAG internals (backend only)
   - `specs/integration.md` — How to mount into scrape-and-analyze
3. **Contracts = spec in code**: `src/chatbot_plugin/contracts/` contains Pydantic models that mirror the spec. If spec changes, update contracts first
4. **API changes must start from spec**: Change `specs/chat-api.md` → update `contracts/` → update router/service → tests pass
5. **Notify the other person**: If you change an API shape, the other developer needs to know. Spec is the communication channel
6. **Contract tests must pass**: `src/tests/contracts/` verifies Pydantic models match spec. These are non-negotiable

### Development Order

```
1. Write/update specs/*.md
2. Write/update contracts/*.py (Pydantic models from spec)
3. Write/update tests/ (based on spec cases)
4. Implement routers/ + service/ (make tests green)
```

### Frontend-Backend Boundary

- Frontend only needs `specs/chat-api.md` — do not look at backend implementation
- Backend can change internals freely as long as API shape (spec) is unchanged
- Disputes are resolved by reading the spec, not the code

## Project Structure

```
specs/                           # SDD spec files (source of truth)
src/chatbot_plugin/
  contracts/                     # Pydantic models = spec in code
    requests.py                  # ChatMessageRequest, SearchRequest, IndexRequest
    responses.py                 # ChatMessageResponse, SearchResponse, etc.
  config.py                      # CHATBOT_* env vars
  routers.py                     # FastAPI endpoints
  service.py                     # Business logic
src/tests/
  contracts/                     # Contract conformance tests
  routers/                       # API endpoint tests
  test_service.py                # Unit tests
```

## Commands

- **Test:** `uv run pytest src/tests/ -v`
- **Coverage:** `uv run pytest src/tests/ --cov=chatbot_plugin --cov-report=html`
