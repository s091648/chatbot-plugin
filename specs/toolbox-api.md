# Toolbox API Specification

> **Source of truth for external services integrating with the toolbox.**
> Any service (e.g. scrape-and-analyze) only needs this document to integrate.

## Base URL

The toolbox runs as a standalone service. Default: `http://localhost:8000`

```bash
uvicorn chatbot_plugin.main:app --reload
```

All endpoints are under `/tools`.

---

## SDK Usage (Python)

Services may also import the toolbox as a Python SDK instead of calling HTTP endpoints.

```python
from chatbot_plugin.sdk import IngestToolboxSDK, QueryToolboxSDK

# Ingest SDK — handles text ingestion (normalise → chunk → embed → save)
from chatbot_plugin.sdk import IngestToolboxSDK

ingest_sdk = IngestToolboxSDK()
ingest_sdk.configure(
    dbname="chatbot_plugin",
    user="postgres",
    password="postgres",
    embedding_model_api="http://localhost:8080",  # embedding microservice
)

await ingest_sdk.ingest(
    full_text="Retrieval augmented generation is a technique ...",
    metadata={"url": "https://example.com/article", "title": "RAG 101"},
)

# Query SDK — handles read-only RAG queries
query_sdk = QueryToolboxSDK()
query_sdk.configure(
    dbname="chatbot_plugin",
    user="postgres",
    password="postgres",
    embedding_model_api="http://localhost:8080",
)

resp = await query_sdk.query("What is RAG?")
print(resp.reply)
```

### Class Hierarchy

| Class | DB | Embedding | Write | Query | LLM |
|-------|----|-----------|-------|-------|-----|
| `BaseRagProcessor` (not exported) | ✓ (internal) | — | — | `search()`, `chat()` | ✓ (fallback) |
| `RagArticleProcessor` | ✓ | ✓ (HTTP) | `ingest()` | inherits base | inherits base |
| `RagQueryProcessor` | ✓ | ✓ (HTTP) | — | `query()` → `chat()` | inherits base |

---

## `POST /tools/chunks`

Store or update an article and its pre-chunked, pre-embedded data.

### Request

```json
{
  "article": {
    "id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
    "url": "https://example.com/article",
    "title": "Article Title",
    "source": "example.com",
    "metadata": {"author": "Alice"}
  },
  "chunks": [
    {
      "chunk_index": 0,
      "content": "First chunk of text...",
      "dense_vector": [0.1, 0.2, ..., 0.1024],
      "sparse_vector": {"0": 0.5, "1": 0.3}
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `article` | `object` | yes | Article metadata |
| `article.id` | `string (uuid)` | yes | Article UUID (used as upsert key) |
| `article.url` | `string (url)` | yes | Source URL |
| `article.title` | `string` | no | Article title |
| `article.source` | `string` | no | Source domain / feed name |
| `article.metadata` | `object` | no | Arbitrary JSON metadata |
| `chunks` | `array (min 1)` | yes | Pre-chunked, pre-embedded data |
| `chunks[].chunk_index` | `integer` | yes | Position within article (0-based) |
| `chunks[].content` | `string` | yes | Chunk text content |
| `chunks[].dense_vector` | `array[float]` | yes | Dense embedding (must match CHATBOT_EMBEDDING_DIMENSION) |
| `chunks[].sparse_vector` | `object` | no | Lexical weights as {token_index: weight} |

### Response `201`

```json
{
  "stored": 1,
  "article_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `stored` | `integer` | Number of chunks stored |
| `article_id` | `string (uuid)` | The article UUID (echoed back) |

### Upsert Semantics

- If `article.id` already exists, the existing article metadata is **updated** and all existing chunks for that article are **deleted and re-inserted**.
- If `article.id` does not exist, a new article and its chunks are inserted.
- This allows scrape-and-analyze to re-index articles without manual deletion.

### Error Responses

| Code | Condition |
|------|-----------|
| `422` | `chunks` array is empty or missing (Pydantic validation) |
| `400` | `dense_vector` dimension does not match `CHATBOT_EMBEDDING_DIMENSION` (default 1024) |
| `422` | Invalid JSON shape or field types |

### Edge Cases

- `sparse_vector` omitted on individual chunks → stored as `NULL`/`{}`
- `title` omitted → stored as `NULL`
- Empty `metadata` → stored as `NULL`/`{}`
- Duplicate `chunk_index` values in the same request → last one wins (databases with `ON CONFLICT` will upsert)

---

## `POST /tools/search`

Hybrid dense + sparse search. The toolbox embeds the query text locally with BGE-M3, then queries pgvector for the most similar chunks.

### Request

```json
{
  "query": "What is retrieval augmented generation?",
  "top_k": 10
}
```

| Field   | Type     | Required | Description                                      |
|---------|----------|----------|--------------------------------------------------|
| `query` | `string` | yes      | Raw query text (will be embedded by toolbox)     |
| `top_k` | `integer`| no       | Number of top chunks to return (default: 10, max: 100) |

### Response `200`

```json
{
  "chunks": [
    {
      "chunk_id": "uuid-string",
      "article_id": "uuid-string",
      "article_title": "Article Title",
      "article_url": "https://example.com",
      "chunk_index": 0,
      "content": "RAG is a technique...",
      "score": 0.876
    }
  ]
}
```

| Field              | Type     | Description                                          |
|--------------------|----------|------------------------------------------------------|
| `chunks`           | `array`  | Ordered list of matching chunks (best first)         |
| `chunks[].chunk_id`| `string (uuid)` | Chunk UUID                                    |
| `chunks[].article_id`| `string (uuid)` | Parent article UUID                          |
| `chunks[].article_title`| `string` | Article title or `null`                      |
| `chunks[].article_url` | `string` | Source URL                                   |
| `chunks[].chunk_index` | `integer`| Position within article                            |
| `chunks[].content` | `string` | Chunk text content                                   |
| `chunks[].score`   | `float`  | RRF fusion score (higher is better)                  |

### Error Responses

| Code | Condition |
|------|-----------|
| `400` | Query text is empty or whitespace-only |
| `500` | Embedding model is not loaded or LLM call failed |

---

## `POST /tools/chat`

Chat with RAG context. Performs hybrid search, assembles top chunks into context, calls LLM, and returns the reply with citations.

### Request

```json
{
  "message": "Explain RAG in simple terms"
}
```

| Field     | Type     | Required | Description                                      |
|-----------|----------|----------|--------------------------------------------------|
| `message` | `string` | yes      | User message (will be embedded by toolbox)       |

### Response `200`

```json
{
  "reply": "RAG is a technique where...",
  "articles_used": [
    {
      "id": "uuid-string",
      "title": "Article Title",
      "url": "https://example.com"
    }
  ],
  "chunks": [
    {
      "chunk_id": "uuid-string",
      "article_id": "uuid-string",
      "chunk_index": 0,
      "content": "RAG is a technique...",
      "score": 0.876
    }
  ]
}
```

| Field           | Type     | Description                                          |
|-----------------|----------|------------------------------------------------------|
| `reply`         | `string` | LLM-generated answer                                 |
| `articles_used` | `array`  | Unique articles that contributed to the context      |
| `chunks`        | `array`  | All chunks used in the context (ordered by score)    |
| `chunks[].chunk_id`| `string (uuid)` | Chunk UUID                                    |
| `chunks[].article_id`| `string (uuid)` | Parent article UUID                          |
| `chunks[].chunk_index`| `integer`| Position within article                            |
| `chunks[].content` | `string` | Chunk text content                                   |
| `chunks[].score` | `float`  | RRF fusion score                                     |

### Error Responses

| Code | Condition |
|------|-----------|
| `400` | Message is empty or whitespace-only |
| `500` | Embedding model is not loaded or LLM call failed |
| `503` | LLM API key not configured |

---

## Sparse Vector Format

- Stored as `sparsevec` in PostgreSQL / pgvector
- Dimension: `250002` (BGE-M3 tokenizer vocab size)
- API layer still accepts `dict[str, float]` (token_index → weight)
- The toolbox translates the dict to sparsevec at storage time
