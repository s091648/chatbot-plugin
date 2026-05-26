# RAG Pipeline Specification

> Backend-internal design. Frontend developers do not need this document.

## Architecture: Independent Database

chatbot-plugin uses its **own PostgreSQL + pgvector database**, completely independent from scrape-and-analyze's database. Data is synced from scrape-and-analyze via a separate process (Phase 2 indexer).

**Why independent:**
- No coupling with scrape-and-analyze's DB schema migrations
- Can add pgvector extension without affecting the main DB
- Plugin can be deployed/restarted independently

## Phase 1: Full-Text Search Fallback

Before embedding + hybrid search is ready, Phase 1 uses PostgreSQL `tsvector` full-text search:

```
User Query
    │
    ▼
PostgreSQL plainto_tsquery('english', query)
    │
    ▼
ts_rank() scoring
    │
    ▼
Top-K Articles (whole articles as context)
    │
    ▼
LLM Generate
```

**Limitations compared to Phase 2:**
- No semantic search (only keyword matching)
- No chunking (whole articles as context, less precise)
- No RRF fusion (single ranking method)

## Overview (Phase 2 Target)

The RAG pipeline retrieves relevant article chunks from PostgreSQL, assembles them as context, and feeds them to an LLM for generation.

```
User Query
    │
    ▼
Embed Query (BGE-M3)
    │ → query_dense_vec (1024-dim float)
    │ → query_sparse_vec (lexical weights)
    │
    ▼
Parallel Search
    │ → Dense:  cosine similarity via pgvector HNSW
    │ → Sparse: BM25 tsvector / neural sparse
    │
    ▼
RRF Fusion (k=60)
    │ score_d(i) = 1/(k + rank_d(i))
    │ score_s(i) = 1/(k + rank_s(i))
    │ final(i)   = score_d(i) + score_s(i)
    │
    ▼
Top-K Chunks (10-20)
    │ Deduplicate, assemble into context
    │
    ▼
LLM Generate
    │ Prompt template + context + query
    │ Each chunk annotated [source: article_title]
    │
    ▼
Chat Response
```

## Components

### Retriever (`rag/retriever.py`)

- **Input:** query string, top_k, optional topic_id filter
- **Output:** list of ranked chunks with scores
- **Strategy:** Hybrid (dense + sparse) with RRF fusion
- **Dense search:** `article_chunks.dense_vector <=> query_vec` via pgvector
- **Sparse search:** `articles.search_tsv` via GIN index (Phase 1), neural sparse via BGE-M3 (Phase 4)
- **Fusion:** Reciprocal Rank Fusion, k=60

### Prompt Builder (`rag/prompt.py`)

- **Input:** ranked chunks + user query
- **Output:** assembled prompt string
- **Template structure:**
  1. System prompt: "You are an assistant that answers based on the provided articles..."
  2. Context section: each chunk as `[source: article_title]\nchunk_content`
  3. User query
- **Token budget:** Truncate context if exceeds `max_context_tokens` (default 8000)
- **Citation:** LLM is instructed to reference sources inline

### RAG Chain (`rag/chain.py`)

- **Input:** user message + retrieved articles + `ResilientLLMService`
- **Steps:** retrieve → build messages (system + human) → call LLM → return reply
- **LLM call:** via `ResilientLLMService.generate()` with fallback chain
- **Prompt structure:** Separate `system_prompt` (instructions) and `human_prompt` (context + query)
- **Output:** reply string

## Database Schema

Plugin adds to the shared PostgreSQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE article_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id      UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    dense_vector    vector(1024),
    sparse_vector   JSONB,
    embedding_model VARCHAR(100) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, chunk_index, embedding_model)
);

-- BM25 fallback
ALTER TABLE articles ADD COLUMN search_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
    ) STORED;
CREATE INDEX idx_articles_tsv ON articles USING gin(search_tsv);

-- HNSW for dense search
CREATE INDEX idx_chunks_dense ON article_chunks
    USING hnsw (dense_vector vector_cosine_ops);
```

**Why chunking:** Whole-article embeddings dilute semantic meaning for long papers. Chunks (256-512 tokens) give more precise retrieval.

## Indexer (`indexer.py`)

- **Trigger:** `POST /chat/index` or polling (`articles.scraped_at > last_indexed_at`)
- **Pipeline:**
  1. Fetch unindexed articles
  2. Split into chunks (256-512 tokens, 50-token overlap)
  3. Generate BGE-M3 embeddings (dense + sparse)
  4. Upsert into `article_chunks`
- **Concurrency:** Runs as background task (rq or asyncio.Task)
- **Idempotency:** `UNIQUE(article_id, chunk_index, embedding_model)` prevents duplicates

## Embedding Model Strategy

| Stage | Approach | Latency |
|-------|----------|---------|
| Development | `sentence-transformers` CPU (BGE-M3) in-process | query ~300-500ms |
| Production (CPU) | TEI Docker container on CPU | Better concurrency |
| Production (GPU) | TEI on GPU | query <10ms |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `CHATBOT_EMBEDDING_DIMENSION` | `1024` | Dense vector dimension |
| `CHATBOT_RRF_K` | `60` | RRF constant k |
| `CHATBOT_CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHATBOT_CHUNK_OVERLAP` | `50` | Overlap tokens between chunks |

## LLM Provider Architecture

Provider configuration is loaded from `providers.toml` (project root). The actual file is gitignored; commit only `providers.example.toml` as a template.

### Provider TOML Format

```toml
[[providers]]
name = "gemini"              # Provider type: gemini | claude | openrouter
priority = 1                  # Lower = tried first
model = "gemini-2.5-flash"   # Model string
api_key_env = "GEMINI_API_KEY"  # Name of env var holding the API key

[providers.strategy]
type = "sliding_window"       # Rate limiting strategy
rpm = 5                       # Requests per minute
tpm = 250000                  # Tokens per minute
rpd = 500                     # Requests per day
```

### Architecture

```
providers.toml → load_providers() → build_llm_service()
                                           |
                                   ResilientLLMService
                                      (fallback chain)
                                    /       |       \
                            ClaudeProvider  GeminiProvider  OpenRouterProvider
                                    \       |       /
                                   BaseProvider (async tenacity retry)
                                           |
                                     QuotaStrategy (rate limiting)
```

### Key Design

- **Native SDKs** (no LangChain): `anthropic.AsyncAnthropic`, `google.genai.Client`, `httpx.AsyncClient`
- **Async throughout**: `asyncio.Lock`, `asyncio.sleep`, `AsyncRetrying`
- **Fallback chain**: `ResilientLLMService` tries providers in priority order, falls back on `RateLimitExhausted` or other errors
- **Env var names** match scrape-and-analyze: `GEMINI_API_KEY`, `CLAUDE_API_KEY`, `OPENROUTER_API_KEY`
- **`BaseProvider.generate(system_prompt, human_prompt)`**: Single method, returns `str | None`
