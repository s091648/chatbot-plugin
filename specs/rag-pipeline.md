# RAG Pipeline Specification

> Backend-internal design. External services do not need this document to integrate.

## Architecture

The chatbot plugin is a **RAG chat backend** — it ingests article data via the SDK, stores chunks in PostgreSQL + pgvector, and serves OpenAI-compatible chat completions by retrieving relevant chunks and generating replies via an LLM fallback chain.

---

## Phase 1: Ingestion

Ingestion is handled in-process via the SDK's `IngestProcessor`. There is no HTTP ingestion endpoint.

```
Raw Article Text
    │
    ▼
SlidingWindowStrategy (SDK)
    │ → chunked text segments
    │
    ▼
EndpointProvider (SDK) — embedding HTTP call
    │ → dense_vector (1024-dim float)
    │ → sparse_vector (250002-dim sparsevec)
    │
    ▼
AsyncPgBackend (SDK)
    │ → upsert into article_chunks (PostgreSQL + pgvector)
```

---

## Phase 2: Retrieval

Retrieval is handled by `RetrieveProcessor` from the SDK, called as:

```python
chunks = await retrieve_processor.retrieve(
    query=user_message,
    top_k=10,
    min_score=0.0,          # pre-rerank score gate
    min_rerank_score=0.7,   # post-rerank score gate (if reranker enabled)
)
```

### Hybrid Search Algorithm

1. Embed query text via `EndpointProvider` → `(dense_vec, sparse_vec)`
2. **Dense candidates**: cosine similarity via pgvector HNSW on `dense_vector`
3. **Sparse candidates**: max inner product via pgvector sparsevec on `sparse_vector`
4. **RRF fusion** (k=60):
   - `score = 1/(60 + rank_dense) + 1/(60 + rank_sparse)`
   - Chunks that appear in only one list get 0 contribution from the missing rank
5. Filter by `min_score` (pre-rerank gate)
6. Optional cross-encoder reranking (if `CHATBOT_ENABLE_RERANKER=true`) → filter by `min_rerank_score`
7. Return top `top_k` chunks ordered by final score

### Score Gating

| Gate | Variable | Default | Applied |
|------|----------|---------|---------|
| Pre-rerank | `CHATBOT_RETRIEVAL_MIN_SCORE` | `0.0` | After RRF fusion, before reranker |
| Post-rerank | `CHATBOT_RERANKER_MIN_SCORE` | `0.7` | After cross-encoder reranking |

---

## Phase 3: Generation

Generation is handled by `ChatService` in `chat_service.py`.

### Pipeline

```
User Message (last "user" role in messages array)
    │
    ▼
RetrieveProcessor.retrieve(query, top_k, min_score, min_rerank_score)
    │ → list of scored chunks
    │
    ▼
Score Gate (ChatService)
    │ If no chunks pass → return raw context fallback
    │
    ▼
Context Assembly (ChatService)
    │ Format chunks as:
    │   [source: Article Title]
    │   Chunk content...
    │
    ▼
ResilientLLMService (llm/base.py)
    │ Fallback chain:
    │   1. Claude (anthropic) — if CHATBOT_CLAUDE_API_KEY set
    │   2. Gemini (google-generativeai) — if CHATBOT_GEMINI_API_KEY set
    │   3. OpenRouter (httpx) — if CHATBOT_OPENROUTER_API_KEY set
    │   4. Raw context fallback — if all providers fail or no keys configured
    │
    ▼
OpenAI-compatible response: POST /v1/chat/completions
```

### ChatService

`ChatService` in `src/chatbot_plugin/chat_service.py`:

- Accepts `messages: list[Message]` (OpenAI format)
- Extracts the last `user` message as the RAG query
- Calls `RetrieveProcessor.retrieve(...)` with configured thresholds
- Assembles context string with `[source: title]` annotations
- Calls `ResilientLLMService.generate(system_prompt, context, user_message)`
- Returns `ChatResult(reply: str, articles: list[ArticleRef])`

`SYSTEM_PROMPT` (defined in `chat_service.py`) instructs the LLM to answer based only on the provided context.

### ResilientLLMService

`ResilientLLMService` in `src/chatbot_plugin/llm/base.py`:

- Holds an ordered list of `ProviderHandler` instances
- Tries each provider in sequence; catches all exceptions and moves to the next
- Provider order: Claude → Gemini → OpenRouter
- If all providers fail or no keys are configured, returns the assembled context directly as the reply

### LLM Providers

| Provider | Module | Key Variable | Model Variable |
|----------|--------|-------------|----------------|
| Anthropic Claude | `llm/claude_provider.py` | `CHATBOT_CLAUDE_API_KEY` | `CHATBOT_CLAUDE_MODEL` |
| Google Gemini | `llm/gemini_provider.py` | `CHATBOT_GEMINI_API_KEY` | `CHATBOT_GEMINI_MODEL` |
| OpenRouter | `llm/openrouter_provider.py` | `CHATBOT_OPENROUTER_API_KEY` | `CHATBOT_OPENROUTER_MODEL` |

Each provider implements the `LLMProvider` protocol defined in `llm/base.py`.

---

## Database Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

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

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin` | Database URL |
| `CHATBOT_EMBEDDING_MODEL_API` | `""` | Embedding service URL |
| `CHATBOT_ENABLE_RERANKER` | `""` | Set to `"true"` to enable cross-encoder reranking |
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
