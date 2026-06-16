# Generation Layer Design

> RAG chat generation layer for chatbot-plugin — prompt assembly, LLM call with fallback chain, and retrieval score gating.

## Overview

chatbot-plugin exposes a single endpoint: `POST /v1/chat/completions`. When a user prompt arrives, the service:

1. Embeds the prompt and retrieves relevant chunks via `RetrieveProcessor` (from SDK)
2. Gates results through score thresholds (dense cosine + optional reranker sigmoid)
3. Assembles a context prompt from surviving chunks
4. Calls an LLM via a resilient fallback chain (Claude → Gemini → OpenRouter)
5. Returns an OpenAI-compatible `ChatCompletionResponse`

If no relevant chunks survive threshold gating, the service returns a "no relevant information" reply without calling any LLM. If no LLM API keys are configured, the assembled context is returned as-is (raw context fallback).

## Obsolete Removals

The following are removed or rewritten:

| Item | Reason |
|------|--------|
| `RagQueryProcessor` imports in `main.py`, `routers.py`, `conftest.py`, `test_toolbox.py` | Class does not exist in SDK; replaced by `RetrieveProcessor` + `ChatService` |
| `specs/toolbox-api.md` `/tools/*` endpoints | Ingestion is done via SDK, not HTTP; chat endpoint is `/v1/chat/completions` |
| `specs/integration.md` ingestion HTTP examples | Same — ingestion uses SDK in-process |
| `specs/rag-pipeline.md` `/tools/chat` endpoint | Replaced by `/v1/chat/completions` |
| `scripts/seed.py` | References non-existent `RagArticleProcessor` |
| `scripts/query.py` | Direct ORM query script superseded by SDK `RetrieveProcessor` |
| `CLAUDE.md` stale project structure | Lists non-existent `models/`, `config.py`, `db.py`, `service.py` |

## File Structure

```
src/chatbot_plugin/
  __init__.py
  main.py                      # Updated: lifespan uses RetrieveProcessor + ChatService
  routers.py                   # Updated: uses ChatService instead of RagQueryProcessor
  chat_service.py              # NEW: orchestrate retrieve → gate → assemble → generate
  contracts/
    __init__.py
    chat_completion.py         # Unchanged
  llm/                         # NEW: LLM provider layer
    __init__.py
    base.py                    # LLMProvider protocol, ProviderHandler, ResilientLLMService
    claude_provider.py         # Anthropic Claude
    gemini_provider.py         # Google Gemini
    openrouter_provider.py     # OpenRouter
```

Rate limiting reuses `chatbot_plugin_sdk.SlidingWindowStrategy` — no separate rate_limit module in chatbot-plugin.

## LLM Provider Interface

```python
from typing import Protocol

class LLMProvider(Protocol):
    model: str

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        """Send messages array, return assistant text reply.

        messages format: [{"role": "system"|"user", "content": "..."}]
        """
        ...
```

Each concrete provider (Claude, Gemini, OpenRouter) translates `messages` into their API format and returns raw text. No JSON parsing, no domain-specific validation — this is a general-purpose chat completion, not an article analysis pipeline.

## ResilientLLMService

Mirrors the scrape-and-analyze `ResilientLLMService` pattern, adapted for async:

```python
@dataclass
class ProviderHandler:
    provider: LLMProvider
    strategy: SlidingWindowStrategy
    priority: int
    name: str

class ResilientLLMService:
    def __init__(self, handlers: list[ProviderHandler]) -> None:
        self._handlers = sorted(handlers, key=lambda h: h.priority)

    async def complete(self, messages: list[dict], max_tokens: int) -> str | None:
        """Try each handler in priority order. On RateLimitExhausted, move to end. On other errors, fall through."""
        ...
```

When `handlers` is empty, `complete()` returns `None` — ChatService treats this as raw context fallback.

## Retrieval Score Gating

Two thresholds, applied at different stages:

### Stage 1: Pre-rerank gate

After `RetrieveProcessor.retrieve()` returns, filter out chunks whose `score < CHATBOT_RETRIEVAL_MIN_SCORE`. This is a coarse filter that removes clearly irrelevant results before reranking.

- Default: `0.0` (disabled — no pre-rerank filtering)
- In **dense-only** mode, score = cosine similarity (0–1), so a threshold like 0.3 is meaningful
- In **hybrid** mode, score = RRF score (~0.01–0.05), so a threshold of 0.3 would filter everything out. Keep at 0.0 or set very low (e.g. 0.001) for hybrid
- The reranker (Stage 2) is the precise filter; Stage 1 is optional and mainly useful for dense-only setups

### Stage 2: Reranker sigmoid gate

When a reranker is configured, after reranking, filter out chunks whose reranker score < `CHATBOT_RERANKER_MIN_SCORE`. This is the precise relevance filter.

- Default: `0.7`
- Reranker scores are sigmoid-normalized (0–1), where 0.5 is the relevance boundary and 0.7+ means "confidently relevant"
- Only applies when a reranker is configured; without a reranker, Stage 1 is the only gate

### In SDK: threshold arguments for RetrieveProcessor

`RetrieveProcessor.retrieve()` gains two optional arguments:

```python
async def retrieve(
    self,
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
    min_rerank_score: float = 0.0,
) -> SearchResponse:
```

- `min_score`: filter out results below this threshold before reranking
- `min_rerank_score`: filter out results below this threshold after reranking
- Both default to 0.0 (no filtering), preserving backward compatibility

This keeps the gating logic inside the SDK's retrieval pipeline rather than requiring ChatService to post-process results.

### No relevant results

If no chunks survive gating, ChatService returns immediately without calling the LLM:

```python
return ChatResult(
    reply="I couldn't find relevant information in the database for your question. Please try rephrasing or ask about a different topic.",
    articles_used=[],
    chunks=[],
)
```

## ChatService

Orchestrates the full RAG chat flow:

```python
class ChatService:
    def __init__(
        self,
        retriever: RetrieveProcessor,
        llm: ResilientLLMService,
        max_context_chunks: int = 10,
        max_tokens: int = 2048,
        min_score: float = 0.0,
        min_rerank_score: float = 0.0,
    ) -> None: ...

    async def chat(self, message: str) -> ChatResult:
        # 1. Retrieve + gate
        search_result = await self._retriever.retrieve(
            message,
            top_k=self._max_context_chunks,
            min_score=self._min_score,
            min_rerank_score=self._min_rerank_score,
        )

        # 2. No relevant results → early return
        if not search_result.chunks:
            return ChatResult(no_relevant_info_reply, [], [])

        # 3. Assemble prompt
        context = self._build_context(search_result.chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {message}"},
        ]

        # 4. Generate
        reply = await self._llm.complete(messages, self._max_tokens)

        # 5. Fallback: no LLM available → return context as-is
        if reply is None:
            reply = f"{context}\n\nQuestion: {message}"

        # 6. Collect unique articles
        articles = self._collect_articles(search_result.chunks)

        return ChatResult(reply=reply, articles_used=articles, chunks=search_result.chunks)
```

### System Prompt

```
You are a research assistant that answers questions based ONLY on the
provided context chunks. Each chunk is annotated with its source article.

Rules:
- Answer using only the information in the context below.
- If the context does not contain enough information to answer, say so.
- Cite the source article title when referencing specific information.
- Do not use external knowledge or make assumptions beyond the context.
- Respond in the same language as the user's question.
```

### Context Assembly Format

```
[source: Article Title]
Chunk content text...

[source: Another Title]
Another chunk text...
```

## ChatResult

```python
@dataclass
class ChatResult:
    reply: str
    articles_used: list[ArticleRef]
    chunks: list[ChunkResult]

@dataclass
class ArticleRef:
    id: str
    title: str | None
    url: str
```

## Environment Variables

```bash
# Database + retrieval
CHATBOT_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin
CHATBOT_EMBEDDING_MODEL_API=http://localhost:8080

# Score gating
CHATBOT_RETRIEVAL_MIN_SCORE=0.0
CHATBOT_RERANKER_MIN_SCORE=0.7

# Generation
CHATBOT_MAX_CONTEXT_CHUNKS=10
CHATBOT_MAX_TOKENS=2048

# LLM providers (at least one needed for LLM generation; none = raw context fallback)
CHATBOT_LLM_PROVIDER=claude              # Default provider name

CHATBOT_CLAUDE_API_KEY=
CHATBOT_CLAUDE_MODEL=claude-sonnet-4-6-20250514

CHATBOT_GEMINI_API_KEY=
CHATBOT_GEMINI_MODEL=gemini-2.0-flash

CHATBOT_OPENROUTER_API_KEY=
CHATBOT_OPENROUTER_MODEL=
```

## main.py Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Retrieval
    db_config = DatabaseConfig(dbname=..., user=..., password=..., host=..., port=...)
    backend = AsyncPgBackend(db_config)
    dense = EndpointProvider(url=embedding_api, dimension=1024)
    retriever = RetrieveProcessor()
    retriever.configure(backend=backend, dense=dense, reranker=FastEmbedReranker())

    # LLM
    handlers = build_llm_handlers()  # from env vars
    llm = ResilientLLMService(handlers)

    # Chat
    app.state.chat_service = ChatService(
        retriever=retriever,
        llm=llm,
        max_context_chunks=int(getenv("CHATBOT_MAX_CONTEXT_CHUNKS", "10")),
        max_tokens=int(getenv("CHATBOT_MAX_TOKENS", "2048")),
        min_score=float(getenv("CHATBOT_RETRIEVAL_MIN_SCORE", "0.3")),
        min_rerank_score=float(getenv("CHATBOT_RERANKER_MIN_SCORE", "0.7")),
    )

    yield
```

## Dependencies (pyproject.toml additions)

```
anthropic          # Claude provider
google-genai>=1.0  # Gemini provider (already optional in SDK)
httpx              # OpenRouter provider (already a dev dep; move to core)
tenacity           # Retry logic for providers
```

## Spec Updates

- `specs/toolbox-api.md` — Remove all `/tools/*` endpoints. Document only `POST /v1/chat/completions`.
- `specs/integration.md` — Remove ingestion HTTP examples. Document SDK-based ingestion and the chat endpoint.
- `specs/rag-pipeline.md` — Remove `/tools/chat` section. Update Phase 3 design to reflect generation layer architecture.
- `CLAUDE.md` — Update project structure, commands, and architecture to match actual codebase.
