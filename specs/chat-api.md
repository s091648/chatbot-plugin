# Chat API Specification

> **Source of truth for frontend-backend contract.**
> Frontend developers only need this document to integrate. Any API change must update this spec first.

## Base Path

All endpoints are mounted under `/chat` in the parent scrape-and-analyze backend.

```
app.include_router(chat_router, prefix="/chat", tags=["chat"])
```

---

## `POST /chat/message`

Send a message and receive a chatbot reply with RAG context.

### Request

```json
{
  "message": "What articles discuss RAG implementation?",
  "user_id": "optional-user-id"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | `string` | yes | — | User input, 1–2000 chars |
| `user_id` | `string` | no | `null` | For per-user context / history |

### Response `200`

```json
{
  "reply": "Based on 3 articles, RAG implementation involves...",
  "articles_used": [
    {"id": "a1b2c3d4-...", "title": "RAG with LangChain"},
    {"id": "e5f6g7h8-...", "title": "Vector Search Best Practices"}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `reply` | `string` | LLM-generated response |
| `articles_used` | `array` | Articles cited in the reply |
| `articles_used[].id` | `string (uuid)` | Article UUID |
| `articles_used[].title` | `string` | Article title |

### Response `422` — Validation Error

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "message"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

### Edge Cases

- `message` empty string → 422
- `message` exceeds 2000 chars → 422
- No relevant articles found → `articles_used` is `[]`, `reply` still generated from LLM general knowledge with disclaimer
- LLM provider timeout → 503 `{"detail": "LLM provider unavailable"}`

---

## `POST /chat/search`

Pure hybrid search without LLM generation. Returns ranked chunks.

### Request

```json
{
  "query": "retrieval augmented generation",
  "top_k": 10,
  "topic_id": "optional-uuid-filter"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | `string` | yes | — | Search query, 1–500 chars |
| `top_k` | `integer` | no | `10` | Number of results, 1–50 |
| `topic_id` | `string (uuid)` | no | `null` | Filter by topic |

### Response `200`

```json
{
  "chunks": [
    {
      "content": "RAG combines retrieval with generation...",
      "article_id": "a1b2c3d4-...",
      "article_title": "RAG with LangChain",
      "score": 0.87
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `chunks` | `array` | Ranked search results |
| `chunks[].content` | `string` | Chunk text |
| `chunks[].article_id` | `string (uuid)` | Parent article UUID |
| `chunks[].article_title` | `string` | Parent article title |
| `chunks[].score` | `float` | RRF fusion score (0–1) |

### Edge Cases

- `top_k` < 1 or > 50 → 422
- No results → `chunks` is `[]`
- `topic_id` not found → `chunks` is `[]` (not 404, topic filter is optional)

---

## `POST /chat/index`

Trigger embedding indexing for articles.

### Request

```json
{
  "article_id": "optional-uuid-for-single-article"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `article_id` | `string (uuid)` | no | `null` | Index single article; `null` = index all unindexed |

### Response `202`

```json
{
  "job_id": "j1k2l3m4-...",
  "status": "started"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `string (uuid)` | Background job identifier |
| `status` | `string` | Always `"started"` |

### Edge Cases

- `article_id` not found → 404 `{"detail": "Article not found"}`
- Already indexing same article → 409 `{"detail": "Indexing already in progress"}`
- No new articles to index → 202 still returned with `job_id`, job completes immediately

---

## `GET /chat/status`

Check indexing status and vector store stats.

### Response `200`

```json
{
  "total_chunks": 1523,
  "last_indexed_at": "2026-05-20T10:00:00Z",
  "pending_articles": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_chunks` | `integer` | Total indexed chunks |
| `last_indexed_at` | `string (ISO 8601)` | Last successful indexing time, `null` if never indexed |
| `pending_articles` | `integer` | Articles not yet indexed |

---

## Error Response Format (shared)

All errors follow FastAPI's standard format:

```json
{
  "detail": "Human-readable error message"
}
```

For validation errors, `detail` is an array of objects (see 422 example above).

## Status Code Summary

| Code | Meaning |
|------|---------|
| 200 | Success |
| 202 | Accepted (async job started) |
| 404 | Resource not found |
| 409 | Conflict (duplicate operation) |
| 422 | Validation error |
| 503 | LLM/embedding provider unavailable |
