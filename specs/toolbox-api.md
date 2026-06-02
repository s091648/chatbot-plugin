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
