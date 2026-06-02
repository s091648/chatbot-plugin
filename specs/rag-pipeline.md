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

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_EMBEDDING_DIMENSION` | `1024` | Dense vector dimension (matches embedding model) |
