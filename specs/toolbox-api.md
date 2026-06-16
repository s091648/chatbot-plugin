# Toolbox API Specification

> **Source of truth for the chat completions API.**

## Overview

OpenAI-compatible chat completions with RAG context. The service receives a user message, retrieves relevant chunks from PostgreSQL + pgvector, assembles them into a context window, and generates a reply via a resilient LLM fallback chain (Claude → Gemini → OpenRouter).

## Base URL

```
http://localhost:8000
```

```bash
uvicorn chatbot_plugin.main:app --reload
```

---

## `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint with RAG augmentation.

### Request

```json
{
  "model": "chatbot-plugin",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain retrieval augmented generation"}
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `string` | yes | Model identifier (passed through to response; does not control LLM selection) |
| `messages` | `array` | yes | Conversation history. Must contain at least one message with `role: "user"` |
| `messages[].role` | `string` | yes | `"system"`, `"user"`, or `"assistant"` |
| `messages[].content` | `string` | yes | Message content |

The last `user` message in the array is used as the RAG query.

### Response `200`

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "chatbot-plugin",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "RAG is a technique where..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique completion ID |
| `object` | `string` | Always `"chat.completion"` |
| `model` | `string` | Echoed from request |
| `choices` | `array` | Always one element |
| `choices[].index` | `integer` | Always `0` |
| `choices[].message.role` | `string` | Always `"assistant"` |
| `choices[].message.content` | `string` | LLM-generated reply (or raw context if no LLM key is configured) |
| `choices[].finish_reason` | `string` | Always `"stop"` |
| `usage` | `object` | Token counts (not tracked; all fields are `0`) |

### Error Responses

| Code | Condition |
|------|-----------|
| `400` | No `user` message found in the `messages` array |
| `422` | Invalid request format or missing required fields (Pydantic validation) |
| `500` | All LLM providers failed and no fallback available |

### Notes

- If no LLM API keys are configured, the endpoint returns the assembled RAG context directly as the reply so callers can still inspect the retrieved sources.
- LLM provider selection is automatic: Claude → Gemini → OpenRouter. The first provider with a configured API key that succeeds is used.
- The `model` field in the request does not control which LLM is called. Provider selection is driven by environment variables.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin` | Database URL |
| `CHATBOT_EMBEDDING_MODEL_API` | `""` | Embedding service URL |
| `CHATBOT_ENABLE_RERANKER` | `""` | Set to `"true"` to enable cross-encoder reranking |
| `CHATBOT_RETRIEVAL_MIN_SCORE` | `0.0` | Pre-rerank score threshold |
| `CHATBOT_RERANKER_MIN_SCORE` | `0.7` | Post-rerank score threshold |
| `CHATBOT_MAX_CONTEXT_CHUNKS` | `10` | Max chunks included in the LLM context |
| `CHATBOT_MAX_TOKENS` | `2048` | Max LLM output tokens |
| `CHATBOT_CLAUDE_API_KEY` | `""` | Anthropic API key |
| `CHATBOT_CLAUDE_MODEL` | `claude-sonnet-4-6-20250514` | Claude model |
| `CHATBOT_GEMINI_API_KEY` | `""` | Google Gemini API key |
| `CHATBOT_GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model |
| `CHATBOT_OPENROUTER_API_KEY` | `""` | OpenRouter API key |
| `CHATBOT_OPENROUTER_MODEL` | `meta-llama/llama-3-70b` | OpenRouter model |
