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
    │ Group chunks by article, one [N] header per article (not per chunk):
    │   [N] Article Title
    │   Chunk content...
    │   ---
    │   Another chunk from the same article...
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
- Assembles context string via `_build_context`, one `[N] Title` header per unique **article**
  (`_collect_articles` numbers articles, not chunks, in first-appearance order) — chunks from
  the same article are grouped under that one header, separated by `---`, rather than each chunk
  repeating its own `[N] Title` line
- Calls `ResilientLLMService.generate(system_prompt, context, user_message)`
- Returns `ChatResult(reply: str, articles: list[ArticleRef])`

`SYSTEM_PROMPT`/`PINNED_SYSTEM_PROMPT` (defined in `chat_service.py`) instruct the LLM to answer
based only on the provided context and to never cite an `[N]` higher than the highest group
actually present. This grouping (rather than one header per chunk) matters most for the
pinned-article flow: a single pinned article can alone contribute up to `max_context_chunks`
chunks (see `_fetch_pinned_chunks`), and repeating an identical `[1] Title` header that many
times in a row visually reads like a numbered list of that many distinct sources — weaker
fallback models (observed with `gemini-3.1-flash-lite`, a rate-limit fallback for the primary
model) have cited invented numbers like `[3]` or `[9]` based on chunk position instead of the
literal repeated `[1]` label. Those invented numbers don't correspond to any entry in
`articles`, so `_filter_cited_articles` silently drops them — the reply then contains `[3]`/`[9]`
markers the frontend renders as inert literal text (out of range for the returned `sources`
list) alongside one correctly-linked `[1]` citation.

### Pinned-Article Tool Calling

> This section covers only the pinned-article + tool-calling path (`_chat_pinned`/`_chat_pinned_stream`,
> `PINNED_SYSTEM_PROMPT`) and its `max_tool_rounds` cap. The rest of this document (streaming
> event types, thinking-delta passthrough) predates that flow and isn't fully reconciled with it yet.

When a request pins one or more articles, `ChatService` first answers from just the pinned
article context (`PINNED_SYSTEM_PROMPT`). If that context is insufficient, the model may call
`search_articles` to query the full corpus.

- The model is **not** limited to a single tool-call round trip. It may call `search_articles`
  again with a different query (e.g. after an unproductive first search) for up to
  `CHATBOT_MAX_TOOL_ROUNDS` turns (default `3`) before the next call is forced `tools=None`,
  guaranteeing the turn terminates with a text answer rather than looping indefinitely.
- A round's follow-up call is pinned to the same provider `ProviderHandler` **only if** that
  round's tool call(s) actually carried a `ToolCallRequest.thought_signature` — Gemini requires
  the exact same model for `thought_signature` continuity across a tool-call round trip and
  rejects a follow-up that echoes it back to a different one (400 INVALID_ARGUMENT). A round
  whose tool calls carried no signature (the handler wasn't a "thinking"-capable Gemini model, or
  didn't emit one) has nothing to preserve, so its follow-up is free to rotate across every other
  configured handler on failure, same as the very first call. Unconditional pinning previously
  meant a single quota-exhausted model could fail a whole turn outright even when sibling models
  configured specifically as fallbacks still had headroom.
- A round that *did* call the tool can still finish cleanly (no error, normal `finish_reason`)
  with zero answer text — the model's own reasoning can indicate an intent to search again that
  it can't act on once the round cap forces `tools=None`. When that happens the turn is not left
  empty: it's substituted with a fixed fallback string (`_EMPTY_FOLLOWUP_REPLY` in
  `chat_service.py`) instead of returning nothing. This substitution only applies to a round that
  used the tool at least once — a round-0 reply with no tool call and no text is a different,
  unrelated failure mode.

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
| `CHATBOT_MAX_TOOL_ROUNDS` | `3` | Pinned-article flow only: max turns the model may use `search_articles` before being forced to answer — see "Pinned-Article Tool Calling" above |
| `CHATBOT_CLAUDE_API_KEY` | `""` | Anthropic API key |
| `CHATBOT_CLAUDE_MODEL` | `claude-sonnet-4-6-20250514` | Claude model |
| `CHATBOT_GEMINI_API_KEY` | `""` | Google Gemini API key |
| `CHATBOT_GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model |
| `CHATBOT_OPENROUTER_API_KEY` | `""` | OpenRouter API key |
| `CHATBOT_OPENROUTER_MODEL` | `meta-llama/llama-3-70b` | OpenRouter model |
