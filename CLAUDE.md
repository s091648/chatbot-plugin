# CLAUDE.md

## Project Overview

OpenAI-compatible RAG chat backend. Receives user prompts, retrieves relevant chunks from PostgreSQL + pgvector via SDK's RetrieveProcessor, assembles context, and generates replies via a resilient LLM fallback chain (Claude → Gemini → OpenRouter).

## SDD (Spec-Driven Development) Workflow

**Spec is source of truth. Code follows spec, not the other way around.**

### Rules

1. **Spec-first**: Before writing or changing any code, read and update `specs/` first
2. **Spec files**:
   - `specs/toolbox-api.md` — API contract
   - `specs/rag-pipeline.md` — Retrieval and generation internals
   - `specs/integration.md` — How external services integrate
3. **Contracts = spec in code**: `src/chatbot_plugin/contracts/` contains Pydantic models that mirror the spec
4. **API changes must start from spec**: Change spec → update contracts → update router/service → tests pass

### Development Order

```
1. Write/update specs/*.md
2. Write/update contracts/*.py (Pydantic models from spec)
3. Write/update tests/ (based on spec cases)
4. Implement routers/ + service/ (make tests green)
```

## Project Structure

```
specs/                           # SDD spec files (source of truth)
  toolbox-api.md                 # API contract (POST /v1/chat/completions)
  rag-pipeline.md                # Retrieval and generation internals
  integration.md                 # How external services integrate
src/chatbot_plugin/
  contracts/                     # Pydantic models = spec in code
    chat_completion.py           # OpenAI-compatible request/response models
  llm/                           # LLM provider layer
    base.py                      # LLMProvider protocol, ResilientLLMService
    claude_provider.py           # Anthropic Claude
    gemini_provider.py           # Google Gemini
    openrouter_provider.py       # OpenRouter
  chat_service.py                # Core: retrieve → gate → assemble → generate
  main.py                        # FastAPI app + lifespan
  routers.py                     # POST /v1/chat/completions endpoint
src/tests/
  llm/                           # LLM provider tests
  routers/                       # API endpoint tests
  test_chat_service.py           # ChatService unit tests
alembic/                         # Database migrations
```

## Commands

- **Run server:** `uvicorn chatbot_plugin.main:app --reload`
- **Test:** `uv run pytest src/tests/ -v`
- **Coverage:** `uv run pytest src/tests/ --cov=chatbot_plugin --cov-report=html`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin` | Database URL |
| `CHATBOT_EMBEDDING_MODEL_API` | `""` | Embedding service URL |
| `CHATBOT_ENABLE_RERANKER` | `""` | Set to "true" to enable cross-encoder reranking |
| `CHATBOT_RETRIEVAL_MIN_SCORE` | `0.0` | Pre-rerank score threshold |
| `CHATBOT_RERANKER_MIN_SCORE` | `0.7` | Post-rerank score threshold |
| `CHATBOT_MAX_CONTEXT_CHUNKS` | `10` | Max chunks in context |
| `CHATBOT_MAX_TOKENS` | `2048` | Max LLM output tokens |
| `CHATBOT_CLAUDE_API_KEY` | `""` | Anthropic API key |
| `CHATBOT_CLAUDE_MODEL` | `claude-sonnet-4-6-20250514` | Claude model |
| `CHATBOT_GEMINI_API_KEY` | `""` | Google Gemini API key |
| `CHATBOT_GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model |
| `CHATBOT_OPENROUTER_API_KEY` | `""` | OpenRouter API key |
| `CHATBOT_OPENROUTER_MODEL` | `meta-llama/llama-3-70b` | OpenRouter model |
