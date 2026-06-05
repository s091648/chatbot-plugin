# RAG Pipeline Specification

> Backend-internal design. Scrape-and-analyze does not need this document to integrate.

## Architecture

The toolbox is a **vector storage layer** — it receives already-chunked, already-embedded data from scrape-and-analyze and stores it in PostgreSQL + pgvector. All embedding generation and chunking logic lives in scrape-and-analyze (or a dedicated embedding service).

### Phase 2 (Current): Storage Toolbox

```
Scrape-and-analyze Pipeline
│
├── Chunk raw text into pieces
├── Generate BGE-M3 embeddings (dense + sparse)
│
└── POST /tools/chunks → {article, chunks[]}
         │
         ▼
    chatbot-plugin (this repo)
         │
         └── PostgreSQL + pgvector → article_chunks table
```

### Phase 3 (Future): Retrieval + Chat

When a search or chat API is added:

```
User Query
    │
    ▼
Embed Query (caller side or dedicated service)
    │ → query_dense_vec (1024-dim float)
    │ → query_sparse_vec (lexical weights)
    │
    ▼
Dense Similarity Search
    │ → pgvector HNSW: article_chunks.dense_vector <=> query_vec
    │
    ▼
Top-K Chunks (10-20)
    │ Deduplicate, assemble into context
    │
    ▼
Chat Response (caller handles LLM generation)
```

The toolbox **does not** generate embeddings or call LLMs. It stores data and serves it back on query. The search layer is retrieval-only.

---

## Database Schema

Plugin manages its own PostgreSQL:

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
    sparse_vector JSONB,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, chunk_index)
);

-- HNSW for dense similarity search
CREATE INDEX hnsw_chunks_dense ON article_chunks
    USING hnsw (dense_vector vector_cosine_ops);
```

**No `search_tsv` or full-text columns:** Chunking and search are embedding-based only in this architecture.

---

## Storage Rules

1. **Upsert semantics**: POST /tools/chunks replaces all existing chunks for an article if the article already exists.
2. **Vector dimension validation**: Reject chunks where `len(dense_vector)` != `CHATBOT_EMBEDDING_DIMENSION` (default 1024).
3. **No embedding generation here**: Scrape-and-analyze (or a separate embedding service) generates vectors before calling this API.

## Phase 3: Query Embedding + Hybrid Retrieval + Chat

### Architecture

```
User Query
    │
    ▼
Embed Query (BGE-M3 in-process CPU)
    │ → query_dense_vec (1024-dim float)
    │ → query_sparse_vec (250002-dim sparsevec)
    │
    ▼
Parallel Hybrid Search
    │
    ├─ Dense:  cosine similarity via pgvector HNSW
    │          ORDER BY dense_vector <=> query_vec LIMIT candidates
    │
    └─ Sparse: max inner product via pgvector sparsevec
               ORDER BY sparse_vector <#> query_sparse LIMIT candidates
    │
    ▼
RRF Fusion (k=60)
    score_d(i) = 1 / (k + rank_d(i))
    score_s(i) = 1 / (k + rank_s(i))
    final(i)   = score_d(i) + score_s(i)
    │
    ▼
Top-K Chunks (default 10)
    │ Deduplicate, assemble into context
    │
    ▼
Chat Response (Claude API via HTTP)
    System prompt + context chunks + user message
    Each context segment annotated [source: article_title]
```

### Embedding Model

- **Model**: `BAAI/bge-m3` (loaded via `FlagEmbedding` library)
- **Loading**: Singleton, loaded once at startup in FastAPI lifespan
- **CPU inference**: ~300-500ms per query, ~2GB RAM footprint
- **Output**: dense (1024-dim) + sparse (250002-dim token weights)
- Sparse dimension: `250002` (BGE-M3 tokenizer vocab size)

### Hybrid Search Algorithm

1. Embed query text via `embed_query(text)` → `(dense_vec, sparse_vec)`
2. **Dense candidates**: `SELECT chunk_id, article_id, ... FROM article_chunks ORDER BY dense_vector <=> :dense LIMIT :candidates`
3. **Sparse candidates**: `SELECT chunk_id, article_id, ... FROM article_chunks ORDER BY sparse_vector <#> :sparse LIMIT :candidates`
4. **RRF fusion**: Combine the two candidate lists
   - Rank each chunk in each list (1-indexed)
   - `score = 1/(rrf_k + rank_dense) + 1/(rrf_k + rank_sparse)`
   - If a chunk only appears in one list, its missing rank contributes 0
5. Sort by final score descending
6. Return top `top_k` chunks

### Chat Service

1. Call search service with `message` as query → top chunks
2. Deduplicate chunks by `article_id`
3. Build context string:
   ```
   [source: Article Title]
   Chunk content...

   [source: Another Title]
   Another chunk...
   ```
4. Send to Claude API:
   - System: "You are a helpful research assistant..."
   - User: context + "\n\nQuestion: {message}"
5. Return: `{reply, articles_used, chunks}`

### Sparse Vector Storage

- PostgreSQL type: `sparsevec(250002)` via pgvector
- SQLAlchemy type: `pgvector.sqlalchemy.SPARSEVEC(250002)`
- Index: no HNSW for sparsevec (pgvector 0.7+ uses btree or ivfflat)
  - For now create a standard btree index on `(sparse_vector)`
- Input translation: API receives `dict[str, float]` → `SparseVector(dict, dim=250002)` → DB
- Output translation: DB → `SparseVector.from_text()` → consumed by search

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_EMBEDDING_DIMENSION` | `1024` | Dense vector dimension (matches embedding model) |
| `CHATBOT_SPARSE_DIMENSION` | `250002` | Sparse vector dimension (BGE-M3 vocab size) |
| `CHATBOT_RRF_K` | `60` | RRF constant k |
| `CHATBOT_SEARCH_CANDIDATES` | `50` | Number of candidates from each sub-search before RRF |
| `CHATBOT_MAX_CONTEXT_CHUNKS` | `10` | Max chunks to include in chat context |
| `CHATBOT_LLM_API_KEY` | `""` | Anthropic API key (optional) |
| `CHATBOT_LLM_MODEL` | `"claude-sonnet-4-6-20250514"` | Claude model name |

## New Dependencies

- `FlagEmbedding` — BGE-M3 embedding (dense + sparse)
- `torch` — PyTorch backend (CPU mode)
- `anthropic` — Claude API client
