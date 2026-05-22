# RAG Pipeline Specification

> Backend-internal design. Frontend developers do not need this document.

## Overview

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

- **Input:** user message + optional user_id
- **Steps:** retrieve → build prompt → call LLM → parse response
- **LLM call:** via LangChain, provider determined by `CHATBOT_LLM_PROVIDER`
- **Output:** reply string + list of articles used

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
