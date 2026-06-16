# Generation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the RAG generation layer (prompt assembly + LLM call with fallback chain + score gating) and remove all obsolete code.

**Architecture:** ChatService orchestrates: RetrieveProcessor.retrieve() with min_score/min_rerank_score gating → context assembly → ResilientLLMService.complete() with Claude/Gemini/OpenRouter fallback chain → OpenAI-compatible response. LLM providers implement a simple async protocol (messages in, text out). Rate limiting reuses SDK's SlidingWindowStrategy.

**Tech Stack:** Python 3.11+, FastAPI, anthropic SDK, google-genai SDK, httpx, tenacity, chatbot-plugin-sdk (SlidingWindowStrategy, RetrieveProcessor, AsyncPgBackend, EndpointProvider, FastEmbedReranker)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/chatbot_plugin/llm/__init__.py` | Re-exports |
| Create | `src/chatbot_plugin/llm/base.py` | LLMProvider protocol, ProviderHandler, ResilientLLMService |
| Create | `src/chatbot_plugin/llm/claude_provider.py` | Anthropic Claude LLM provider |
| Create | `src/chatbot_plugin/llm/gemini_provider.py` | Google Gemini LLM provider |
| Create | `src/chatbot_plugin/llm/openrouter_provider.py` | OpenRouter LLM provider |
| Create | `src/chatbot_plugin/chat_service.py` | ChatService, ChatResult, ArticleRef, SYSTEM_PROMPT, context assembly |
| Create | `src/tests/llm/test_base.py` | Tests for ResilientLLMService |
| Create | `src/tests/llm/test_providers.py` | Tests for LLM providers (mocked) |
| Create | `src/tests/test_chat_service.py` | Tests for ChatService |
| Modify | `src/chatbot_plugin/main.py` | Replace RagQueryProcessor with RetrieveProcessor + ChatService |
| Modify | `src/chatbot_plugin/routers.py` | Replace RagQueryProcessor with ChatService |
| Modify | `src/tests/conftest.py` | Replace RagQueryProcessor with ChatService mock |
| Modify | `src/tests/routers/test_toolbox.py` | Replace RagQueryProcessor patch with ChatService patch |
| Modify | `pyproject.toml` | Add anthropic, google-genai, tenacity; move httpx to core |
| Delete | `scripts/seed.py` | References non-existent RagArticleProcessor |
| Delete | `scripts/query.py` | Superseded by SDK RetrieveProcessor |
| Modify | `specs/toolbox-api.md` | Remove /tools/* endpoints, document /v1/chat/completions |
| Modify | `specs/integration.md` | Remove ingestion HTTP examples |
| Modify | `specs/rag-pipeline.md` | Remove /tools/chat, update Phase 3 |
| Modify | `CLAUDE.md` | Update project structure and commands |

**SDK changes** (separate repo `chatbot-plugin-sdk`):

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/chatbot_plugin_sdk/processors/retrieve.py` | Add min_score, min_rerank_score args to retrieve() |
| Modify | `src/tests/test_retrieve.py` | Add tests for min_score and min_rerank_score filtering |

---

## Task 1: Add min_score and min_rerank_score to SDK RetrieveProcessor

**Files:**
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin-sdk/src/chatbot_plugin_sdk/processors/retrieve.py:113-205`
- Test: `/home/pegaai/Desktop/sdd/chatbot-plugin-sdk/src/tests/test_retrieve.py`

- [ ] **Step 1: Write failing tests for min_score filtering**

Add to `test_retrieve.py`:

```python
class TestRetrieveScoreGating:
    @pytest.mark.asyncio
    async def test_min_score_filters_dense_results(self):
        """Chunks with score < min_score are removed before reranking."""
        retriever, backend = _configured_retriever()
        # Two results: score 0.9 and score 0.2
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "relevant", "T", "u", 0.1),  # score = 0.9
            _make_row("c2", "a2", 0, "irrelevant", "T", "u", 0.8),  # score = 0.2
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", min_score=0.5)

        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_min_score_zero_passes_everything(self):
        """min_score=0.0 (default) does not filter anything."""
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "text", "T", "u", 0.99),  # score = 0.01
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", min_score=0.0)

        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_min_rerank_score_filters_after_rerank(self):
        """Chunks with reranker score < min_rerank_score are removed."""
        backend = _mock_backend()
        row1 = _make_row("c1", "a1", 0, "relevant", "T", "u", 0.1)
        row2 = _make_row("c2", "a2", 0, "irrelevant", "T", "u", 0.3)
        backend.search_dense.return_value = [row1, row2]

        dense = EndpointProvider(url="http://x", dimension=768)
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[
            (row1, 0.9),   # above threshold
            (row2, 0.4),   # below threshold
        ])

        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, reranker=reranker)
        retriever._ready = True

        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", top_k=5, min_rerank_score=0.7)

        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == "c1"
        assert result.chunks[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_min_rerank_score_zero_passes_everything(self):
        """min_rerank_score=0.0 (default) does not filter after rerank."""
        backend = _mock_backend()
        row1 = _make_row("c1", "a1", 0, "text", "T", "u", 0.1)
        backend.search_dense.return_value = [row1]

        dense = EndpointProvider(url="http://x", dimension=768)
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[(row1, 0.15)])

        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, reranker=reranker)
        retriever._ready = True

        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", min_rerank_score=0.0)

        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_all_chunks_filtered_returns_empty(self):
        """When all chunks are below threshold, return empty chunks list."""
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = [
            _make_row("c1", "a1", 0, "text", "T", "u", 0.95),  # score = 0.05
        ]
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", min_score=0.5)

        assert result.chunks == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && uv run pytest src/tests/test_retrieve.py::TestRetrieveScoreGating -v`
Expected: FAIL — `retrieve()` doesn't accept `min_score` or `min_rerank_score` kwargs

- [ ] **Step 3: Implement min_score and min_rerank_score in retrieve()**

In `retrieve.py`, update the `retrieve()` signature and add filtering logic:

```python
async def retrieve(
    self,
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
    min_rerank_score: float = 0.0,
) -> SearchResponse:
```

Add `min_score` filtering **after** RRF merge / score assignment but **before** reranking. This means filtering the `ranked_rows` list:

For the dense-only path (after `ranked_rows = await self._backend.search_dense(...)`), add before the score-construction block:

```python
if min_score > 0:
    ranked_rows = [
        r for r in ranked_rows
        if (1.0 - r.distance) >= min_score
    ]
```

For the hybrid path (after `ranked_rows = [r for r, _ in merged]`), add:

```python
if min_score > 0:
    ranked_rows = [
        r for r in ranked_rows
        if rrf_scores.get(r.chunk_id, 0) >= min_score
    ]
```

For the sparse-only path (after `ranked_rows = await self._backend.search_sparse(...)`), add:

```python
if min_score > 0:
    ranked_rows = [
        r for r in ranked_rows
        if (-r.distance) >= min_score
    ]
```

Add `min_rerank_score` filtering **after** reranking, inside the reranker block:

```python
if self._reranker is not None:
    reranked = await self._reranker.rerank(query, ranked_rows)
    if min_rerank_score > 0:
        reranked = [(r, s) for r, s in reranked if s >= min_rerank_score]
    result = SearchResponse(chunks=[
        ChunkResult(
            chunk_id=r.chunk_id,
            article_id=r.article_id,
            article_title=r.title,
            article_url=r.url,
            chunk_index=r.chunk_index,
            content=r.content,
            score=round(s, 6),
        )
        for r, s in reranked[:top_k]
    ])
    logger.info("retrieve_complete", extra={"chunk_count": len(result.chunks), "reranked": True})
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && uv run pytest src/tests/test_retrieve.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full SDK test suite to check no regressions**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && uv run pytest src/tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit SDK changes**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk
git add src/chatbot_plugin_sdk/processors/retrieve.py src/tests/test_retrieve.py
git commit -m "feat: add min_score and min_rerank_score filtering to RetrieveProcessor.retrieve()"
```

---

## Task 2: Create LLM provider protocol and ResilientLLMService

**Files:**
- Create: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/chatbot_plugin/llm/__init__.py`
- Create: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/chatbot_plugin/llm/base.py`
- Test: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/tests/llm/test_base.py`

- [ ] **Step 1: Write failing tests for ResilientLLMService**

Create `src/tests/llm/__init__.py` (empty) and `src/tests/llm/test_base.py`:

```python
"""Tests for LLM provider protocol and ResilientLLMService."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot_plugin_sdk import SlidingWindowStrategy, RateLimitExhausted

from chatbot_plugin.llm.base import LLMProvider, ProviderHandler, ResilientLLMService


def _mock_provider(name: str = "mock", model: str = "mock-model") -> AsyncMock:
    provider = AsyncMock(spec=LLMProvider)
    provider.model = model
    provider.complete = AsyncMock(return_value="Generated text")
    return provider


def _handler(
    name: str = "mock",
    priority: int = 1,
    rpm: int = 0,
) -> ProviderHandler:
    provider = _mock_provider(name=name)
    strategy = SlidingWindowStrategy(rpm=rpm)
    return ProviderHandler(
        provider=provider,
        strategy=strategy,
        priority=priority,
        name=name,
    )


class TestResilientLLMService:
    @pytest.mark.asyncio
    async def test_calls_highest_priority_handler(self):
        h1 = _handler(name="first", priority=1)
        h2 = _handler(name="second", priority=2)
        service = ResilientLLMService([h2, h1])  # pass out of order
        result = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result == "Generated text"
        h1.provider.complete.assert_called_once()
        h2.provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_on_provider_failure(self):
        h1 = _handler(name="failing", priority=1)
        h1.provider.complete = AsyncMock(side_effect=RuntimeError("API down"))
        h2 = _handler(name="backup", priority=2)
        service = ResilientLLMService([h1, h2])
        result = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result == "Generated text"
        h2.provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_moves_handler_to_end_on_rate_limit(self):
        h1 = _handler(name="rate_limited", priority=1, rpm=1)
        h2 = _handler(name="backup", priority=2)
        # First call hits rate limit
        h1.provider.complete = AsyncMock(side_effect=RateLimitExhausted("daily cap"))
        service = ResilientLLMService([h1, h2])
        result = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result == "Generated text"
        # h1 should be moved to end
        assert service._handlers[0].name == "backup"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_handlers(self):
        service = ResilientLLMService([])
        result = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_all_handlers_fail(self):
        h1 = _handler(name="h1", priority=1)
        h2 = _handler(name="h2", priority=2)
        h1.provider.complete = AsyncMock(side_effect=RuntimeError("down"))
        h2.provider.complete = AsyncMock(side_effect=RuntimeError("down"))
        service = ResilientLLMService([h1, h2])
        result = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result is None


class TestProviderHandler:
    @pytest.mark.asyncio
    async def test_complete_delegates_to_provider(self):
        handler = _handler()
        result = await handler.complete(
            [{"role": "user", "content": "hello"}], 500
        )
        assert result == "Generated text"
        handler.provider.complete.assert_called_once_with(
            [{"role": "user", "content": "hello"}], 500
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/llm/test_base.py -v`
Expected: FAIL — module `chatbot_plugin.llm.base` not found

- [ ] **Step 3: Create llm/__init__.py**

Create `src/chatbot_plugin/llm/__init__.py`:

```python
from chatbot_plugin.llm.base import LLMProvider, ProviderHandler, ResilientLLMService

__all__ = ["LLMProvider", "ProviderHandler", "ResilientLLMService"]
```

- [ ] **Step 4: Implement base.py**

Create `src/chatbot_plugin/llm/base.py`:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chatbot_plugin_sdk import SlidingWindowStrategy, RateLimitExhausted

logger = logging.getLogger(__name__)


@runtime_checkable
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


@dataclass
class ProviderHandler:
    provider: LLMProvider
    strategy: SlidingWindowStrategy
    priority: int
    name: str

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        await self.strategy.acquire(estimated_tokens=estimated_tokens)
        result = await self.provider.complete(messages, max_tokens)
        self.strategy.record_usage(estimated_tokens)
        return result


class ResilientLLMService:
    """Walk an ordered list of ProviderHandlers. Fall back on rate-limit or failure."""

    def __init__(self, handlers: list[ProviderHandler]) -> None:
        self._handlers = sorted(handlers, key=lambda h: h.priority)

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str | None:
        if not self._handlers:
            return None

        handlers_snapshot = list(self._handlers)

        for handler in handlers_snapshot:
            try:
                result = await handler.complete(messages, max_tokens)
                if result is not None:
                    return result
                logger.warning("provider_returned_none", provider=handler.name)
            except RateLimitExhausted:
                logger.warning("provider_daily_limit_reached", provider=handler.name)
                self._handlers.remove(handler)
                self._handlers.append(handler)
                logger.warning("provider_moved_to_end", provider=handler.name)
            except Exception as e:
                logger.error("provider_failed", provider=handler.name, error=str(e))

        logger.error("all_providers_exhausted")
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/llm/test_base.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin
git add src/chatbot_plugin/llm/__init__.py src/chatbot_plugin/llm/base.py src/tests/llm/__init__.py src/tests/llm/test_base.py
git commit -m "feat: add LLMProvider protocol and ResilientLLMService"
```

---

## Task 3: Create Claude, Gemini, OpenRouter providers

**Files:**
- Create: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/chatbot_plugin/llm/claude_provider.py`
- Create: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/chatbot_plugin/llm/gemini_provider.py`
- Create: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/chatbot_plugin/llm/openrouter_provider.py`
- Test: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/tests/llm/test_providers.py`

- [ ] **Step 1: Write failing tests for providers**

Create `src/tests/llm/test_providers.py`:

```python
"""Tests for LLM provider implementations (mocked API calls)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin.llm.base import LLMProvider
from chatbot_plugin.llm.claude_provider import ClaudeProvider
from chatbot_plugin.llm.gemini_provider import GeminiProvider
from chatbot_plugin.llm.openrouter_provider import OpenRouterProvider


MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is RAG?"},
]


class TestClaudeProvider:
    def test_satisfies_protocol(self):
        provider = ClaudeProvider.__new__(ClaudeProvider)
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_complete_sends_messages(self):
        provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-6-20250514")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="RAG is retrieval-augmented generation.")]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=20)
        with patch.object(provider._client.messages, "create", return_value=mock_response) as mock_create:
            result = await provider.complete(MESSAGES, 1024)
        assert result == "RAG is retrieval-augmented generation."
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6-20250514"
        assert call_kwargs["max_tokens"] == 1024
        assert len(call_kwargs["messages"]) == 2

    @pytest.mark.asyncio
    async def test_complete_converts_roles(self):
        """Claude API uses same role names as OpenAI, so messages pass through."""
        provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-6-20250514")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.usage = MagicMock(input_tokens=0, output_tokens=0)
        with patch.object(provider._client.messages, "create", return_value=mock_response):
            await provider.complete(MESSAGES, 100)


class TestGeminiProvider:
    def test_satisfies_protocol(self):
        provider = GeminiProvider.__new__(GeminiProvider)
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_complete_sends_messages(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-2.0-flash")
        mock_response = MagicMock()
        mock_response.text = "RAG is a retrieval technique."
        mock_response.candidates = [MagicMock(finish_reason=MagicMock(name="STOP"))]
        with patch.object(provider._client.models, "generate_content", return_value=mock_response) as mock_gen:
            result = await provider.complete(MESSAGES, 1024)
        assert result == "RAG is a retrieval technique."
        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_returns_empty_on_blocked(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-2.0-flash")
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(finish_reason=MagicMock(name="SAFETY"))]
        mock_response.text = ""
        with patch.object(provider._client.models, "generate_content", return_value=mock_response):
            result = await provider.complete(MESSAGES, 1024)
        assert result == ""

    @pytest.mark.asyncio
    async def test_complete_raises_on_resource_exhausted(self):
        from chatbot_plugin_sdk import RateLimitExhausted
        provider = GeminiProvider(api_key="test-key", model="gemini-2.0-flash")
        error = Exception("429 RESOURCE_EXHAUSTED PerDay limit exceeded")
        with patch.object(provider._client.models, "generate_content", side_effect=error):
            with pytest.raises(RateLimitExhausted):
                await provider.complete(MESSAGES, 1024)


class TestOpenRouterProvider:
    def test_satisfies_protocol(self):
        provider = OpenRouterProvider.__new__(OpenRouterProvider)
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_complete_sends_messages(self):
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "RAG combines retrieval with generation."}}],
        }
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await provider.complete(MESSAGES, 1024)
        assert result == "RAG combines retrieval with generation."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/llm/test_providers.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement ClaudeProvider**

Create `src/chatbot_plugin/llm/claude_provider.py`:

```python
from __future__ import annotations

import asyncio
import logging

import anthropic

from chatbot_plugin.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class ClaudeProvider:
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6-20250514") -> None:
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        logger.info(
            "claude_api_called",
            extra={"model": self.model, "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
        )
        return response.content[0].text
```

- [ ] **Step 4: Implement GeminiProvider**

Create `src/chatbot_plugin/llm/gemini_provider.py`:

```python
from __future__ import annotations

import asyncio
import logging

from google import genai

from chatbot_plugin_sdk import RateLimitExhausted

logger = logging.getLogger(__name__)


class GeminiProvider:
    """Google Gemini LLM provider."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        # Gemini uses a single "contents" string — combine system + user messages
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System Instructions]\n{content}")
            else:
                parts.append(content)
        contents = "\n\n".join(parts)

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(max_output_tokens=max_tokens),
                ),
            )
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str and "PerDay" in error_str:
                raise RateLimitExhausted(f"Daily quota exceeded for {self.model}") from e
            raise

        if not response.candidates:
            return ""

        candidate = response.candidates[0]
        fr = candidate.finish_reason
        fr_name = fr.name if hasattr(fr, "name") else str(fr)
        if fr_name not in ("STOP", "1"):
            logger.warning("gemini_blocked", model=self.model, finish_reason=fr_name)
            return ""

        logger.info("gemini_api_called", model=self.model)
        return (response.text or "").strip()
```

- [ ] **Step 5: Implement OpenRouterProvider**

Create `src/chatbot_plugin/llm/openrouter_provider.py`:

```python
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider:
    """OpenRouter LLM provider via OpenAI-compatible chat completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._api_key = api_key

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                _API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

        logger.info("openrouter_api_called", model=self.model)
        return data["choices"][0]["message"]["content"]
```

- [ ] **Step 6: Run provider tests**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/llm/test_providers.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin
git add src/chatbot_plugin/llm/claude_provider.py src/chatbot_plugin/llm/gemini_provider.py src/chatbot_plugin/llm/openrouter_provider.py src/tests/llm/test_providers.py
git commit -m "feat: add Claude, Gemini, OpenRouter LLM providers"
```

---

## Task 4: Create ChatService

**Files:**
- Create: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/chatbot_plugin/chat_service.py`
- Test: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/tests/test_chat_service.py`

- [ ] **Step 1: Write failing tests for ChatService**

Create `src/tests/test_chat_service.py`:

```python
"""Tests for ChatService — prompt assembly, retrieval gating, LLM generation."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin.chat_service import ChatService, ArticleRef, SYSTEM_PROMPT


def _chunk(chunk_id="c1", article_id="a1", title="Test Article", url="https://test.com", content="RAG is retrieval-augmented generation.", score=0.9):
    return ChunkResult(
        chunk_id=chunk_id,
        article_id=article_id,
        article_title=title,
        article_url=url,
        chunk_index=0,
        content=content,
        score=score,
    )


def _mock_retriever(chunks=None):
    retriever = MagicMock()
    if chunks is None:
        chunks = [_chunk()]
    retriever.retrieve = AsyncMock(return_value=SearchResponse(chunks=chunks))
    return retriever


def _mock_llm(reply="Generated reply"):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=reply)
    return llm


class TestChatService:
    @pytest.mark.asyncio
    async def test_chat_returns_reply(self):
        service = ChatService(retriever=_mock_retriever(), llm=_mock_llm())
        result = await service.chat("What is RAG?")
        assert result.reply == "Generated reply"

    @pytest.mark.asyncio
    async def test_chat_passes_message_to_retriever(self):
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm())
        await service.chat("What is RAG?")
        retriever.retrieve.assert_called_once_with(
            "What is RAG?", top_k=10, min_score=0.0, min_rerank_score=0.0,
        )

    @pytest.mark.asyncio
    async def test_chat_respects_custom_thresholds(self):
        retriever = _mock_retriever()
        service = ChatService(
            retriever=retriever, llm=_mock_llm(),
            min_score=0.3, min_rerank_score=0.7,
        )
        await service.chat("What is RAG?")
        retriever.retrieve.assert_called_once_with(
            "What is RAG?", top_k=10, min_score=0.3, min_rerank_score=0.7,
        )

    @pytest.mark.asyncio
    async def test_chat_no_relevant_chunks_returns_fallback(self):
        retriever = _mock_retriever(chunks=[])
        service = ChatService(retriever=retriever, llm=_mock_llm())
        result = await service.chat("obscure question")
        assert "couldn't find" in result.reply.lower()
        assert result.articles_used == []
        assert result.chunks == []
        # LLM should NOT be called
        service._llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_assembles_context(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="RAG 101", content="RAG is retrieval-augmented generation."),
            _chunk(chunk_id="c2", article_id="a2", title="LLM Guide", content="LLMs are large language models."),
        ]
        llm = _mock_llm()
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=llm)
        await service.chat("What is RAG?")
        call_args = llm.complete.call_args
        messages = call_args.args[0]
        # System message should contain the SYSTEM_PROMPT
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        # User message should contain context with source annotations
        user_msg = messages[1]["content"]
        assert "[source: RAG 101]" in user_msg
        assert "[source: LLM Guide]" in user_msg
        assert "Question: What is RAG?" in user_msg

    @pytest.mark.asyncio
    async def test_chat_collects_unique_articles(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article 1", url="https://a.com"),
            _chunk(chunk_id="c2", article_id="a1", title="Article 1", url="https://a.com"),
            _chunk(chunk_id="c3", article_id="a2", title="Article 2", url="https://b.com"),
        ]
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=_mock_llm())
        result = await service.chat("q")
        assert len(result.articles_used) == 2
        assert result.articles_used[0].id == "a1"
        assert result.articles_used[1].id == "a2"

    @pytest.mark.asyncio
    async def test_chat_raw_context_fallback_when_no_llm(self):
        chunks = [_chunk(content="RAG is...")]
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=None)  # no LLM available
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=llm)
        result = await service.chat("What is RAG?")
        # Should return context as-is
        assert "[source: Test Article]" in result.reply
        assert "Question: What is RAG?" in result.reply

    @pytest.mark.asyncio
    async def test_chat_passes_max_tokens(self):
        llm = _mock_llm()
        service = ChatService(
            retriever=_mock_retriever(), llm=llm, max_tokens=512,
        )
        await service.chat("hi")
        call_args = llm.complete.call_args
        assert call_args.args[1] == 512

    @pytest.mark.asyncio
    async def test_chat_passes_max_context_chunks(self):
        retriever = _mock_retriever()
        service = ChatService(
            retriever=retriever, llm=_mock_llm(), max_context_chunks=5,
        )
        await service.chat("hi")
        retriever.retrieve.assert_called_once_with(
            "hi", top_k=5, min_score=0.0, min_rerank_score=0.0,
        )


class TestContextAssembly:
    @pytest.mark.asyncio
    async def test_build_context_format(self):
        chunks = [
            _chunk(title="My Title", content="First paragraph."),
            _chunk(title="Other Title", content="Second paragraph."),
        ]
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=_mock_llm())
        context = service._build_context(chunks)
        assert "[source: My Title]" in context
        assert "First paragraph." in context
        assert "[source: Other Title]" in context
        assert "Second paragraph." in context

    @pytest.mark.asyncio
    async def test_build_context_no_title(self):
        chunks = [_chunk(title=None, content="Some text.")]
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=_mock_llm())
        context = service._build_context(chunks)
        assert "[source: Unknown]" in context
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/test_chat_service.py -v`
Expected: FAIL — `chatbot_plugin.chat_service` not found

- [ ] **Step 3: Implement ChatService**

Create `src/chatbot_plugin/chat_service.py`:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from chatbot_plugin_sdk.contracts.responses import ChunkResult
from chatbot_plugin.llm.base import ResilientLLMService
from chatbot_plugin_sdk.processors.retrieve import RetrieveProcessor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research assistant that answers questions based ONLY on the
provided context chunks. Each chunk is annotated with its source article.

Rules:
- Answer using only the information in the context below.
- If the context does not contain enough information to answer, say so.
- Cite the source article title when referencing specific information.
- Do not use external knowledge or make assumptions beyond the context.
- Respond in the same language as the user's question.
"""

_NO_RELEVANT_INFO_REPLY = (
    "I couldn't find relevant information in the database for your question. "
    "Please try rephrasing or ask about a different topic."
)


@dataclass
class ArticleRef:
    id: str
    title: str | None
    url: str


@dataclass
class ChatResult:
    reply: str
    articles_used: list[ArticleRef]
    chunks: list[ChunkResult]


class ChatService:
    def __init__(
        self,
        retriever: RetrieveProcessor,
        llm: ResilientLLMService,
        max_context_chunks: int = 10,
        max_tokens: int = 2048,
        min_score: float = 0.0,
        min_rerank_score: float = 0.0,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_context_chunks = max_context_chunks
        self._max_tokens = max_tokens
        self._min_score = min_score
        self._min_rerank_score = min_rerank_score

    async def chat(self, message: str) -> ChatResult:
        # 1. Retrieve + gate
        search_result = await self._retriever.retrieve(
            message,
            top_k=self._max_context_chunks,
            min_score=self._min_score,
            min_rerank_score=self._min_rerank_score,
        )

        # 2. No relevant results → early return without LLM call
        if not search_result.chunks:
            return ChatResult(reply=_NO_RELEVANT_INFO_REPLY, articles_used=[], chunks=[])

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

    def _build_context(self, chunks: list[ChunkResult]) -> str:
        parts = []
        for chunk in chunks:
            title = chunk.article_title or "Unknown"
            parts.append(f"[source: {title}]\n{chunk.content}")
        return "\n\n".join(parts)

    def _collect_articles(self, chunks: list[ChunkResult]) -> list[ArticleRef]:
        seen: dict[str, ArticleRef] = {}
        for chunk in chunks:
            if chunk.article_id not in seen:
                seen[chunk.article_id] = ArticleRef(
                    id=chunk.article_id,
                    title=chunk.article_title,
                    url=chunk.article_url or "",
                )
        return list(seen.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/test_chat_service.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin
git add src/chatbot_plugin/chat_service.py src/tests/test_chat_service.py
git commit -m "feat: add ChatService with prompt assembly and retrieval gating"
```

---

## Task 5: Wire ChatService into main.py and routers.py, remove RagQueryProcessor

**Files:**
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/chatbot_plugin/main.py`
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/chatbot_plugin/routers.py`

- [ ] **Step 1: Rewrite main.py**

Replace entire `src/chatbot_plugin/main.py` with:

```python
"""Toolbox standalone server — OpenAI-compatible RAG chat backend.

Run with::

    uvicorn chatbot_plugin.main:app --reload
"""

from contextlib import asynccontextmanager
from os import getenv
from urllib.parse import urlparse

from fastapi import FastAPI

from chatbot_plugin.chat_service import ChatService
from chatbot_plugin.llm.base import ProviderHandler, ResilientLLMService
from chatbot_plugin.llm.claude_provider import ClaudeProvider
from chatbot_plugin.llm.gemini_provider import GeminiProvider
from chatbot_plugin.llm.openrouter_provider import OpenRouterProvider
from chatbot_plugin.routers import api_router
from chatbot_plugin_sdk import (
    AsyncPgBackend,
    DatabaseConfig,
    EndpointProvider,
    RetrieveProcessor,
    SlidingWindowStrategy,
)


def _get_db_url() -> str:
    return getenv(
        "CHATBOT_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin",
    )


def _build_llm_handlers() -> list[ProviderHandler]:
    handlers: list[ProviderHandler] = []
    priority = 1

    claude_key = getenv("CHATBOT_CLAUDE_API_KEY", "")
    if claude_key:
        handlers.append(ProviderHandler(
            provider=ClaudeProvider(api_key=claude_key, model=getenv("CHATBOT_CLAUDE_MODEL", "claude-sonnet-4-6-20250514")),
            strategy=SlidingWindowStrategy(rpm=5, tpm=100_000, rpd=1000),
            priority=priority,
            name="claude",
        ))
        priority += 1

    gemini_key = getenv("CHATBOT_GEMINI_API_KEY", "")
    if gemini_key:
        handlers.append(ProviderHandler(
            provider=GeminiProvider(api_key=gemini_key, model=getenv("CHATBOT_GEMINI_MODEL", "gemini-2.0-flash")),
            strategy=SlidingWindowStrategy(rpm=10, tpm=200_000, rpd=1500),
            priority=priority,
            name="gemini",
        ))
        priority += 1

    openrouter_key = getenv("CHATBOT_OPENROUTER_API_KEY", "")
    if openrouter_key:
        handlers.append(ProviderHandler(
            provider=OpenRouterProvider(api_key=openrouter_key, model=getenv("CHATBOT_OPENROUTER_MODEL", "meta-llama/llama-3-70b")),
            strategy=SlidingWindowStrategy(rpm=10, tpm=200_000, rpd=1500),
            priority=priority,
            name="openrouter",
        ))

    return handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Retrieval ──────────────────────────────────────────────────────────
    db_url = _get_db_url()
    embedding_api = getenv("CHATBOT_EMBEDDING_MODEL_API", "")
    parsed = urlparse(db_url)

    db_config = DatabaseConfig(
        dbname=parsed.path.lstrip("/"),
        user=parsed.username or "postgres",
        password=parsed.password or "",
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
    )
    backend = AsyncPgBackend(db_config)

    dense = EndpointProvider(url=embedding_api, dimension=1024) if embedding_api else None
    reranker = FastEmbedReranker() if getenv("CHATBOT_ENABLE_RERANKER", "").lower() in ("1", "true", "yes") else None

    retriever = RetrieveProcessor()
    retriever.configure(backend=backend, dense=dense, reranker=reranker)

    # ── LLM ───────────────────────────────────────────────────────────────
    handlers = _build_llm_handlers()
    llm = ResilientLLMService(handlers)

    # ── Chat ──────────────────────────────────────────────────────────────
    app.state.chat_service = ChatService(
        retriever=retriever,
        llm=llm,
        max_context_chunks=int(getenv("CHATBOT_MAX_CONTEXT_CHUNKS", "10")),
        max_tokens=int(getenv("CHATBOT_MAX_TOKENS", "2048")),
        min_score=float(getenv("CHATBOT_RETRIEVAL_MIN_SCORE", "0.0")),
        min_rerank_score=float(getenv("CHATBOT_RERANKER_MIN_SCORE", "0.7")),
    )

    yield


app = FastAPI(
    title="Toolbox",
    description="Vector storage toolbox — OpenAI-compatible RAG chat API.",
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(api_router)
```

- [ ] **Step 2: Rewrite routers.py**

Replace entire `src/chatbot_plugin/routers.py` with:

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from chatbot_plugin.chat_service import ChatService
from chatbot_plugin.contracts.chat_completion import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

api_router = APIRouter(prefix="/v1")


def _get_chat_service(request: Request) -> ChatService:
    service: ChatService | None = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="ChatService not initialised")
    return service


@api_router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse:
    service = _get_chat_service(request)

    last_message = req.get_last_user_message()
    if not last_message.strip():
        raise HTTPException(
            status_code=400,
            detail="messages must contain at least one user message with non-empty content",
        )

    result = await service.chat(last_message)

    return ChatCompletionResponse(
        model="rag-default",
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(content=result.reply),
            )
        ],
    )
```

- [ ] **Step 3: Run existing router tests to verify they still work (they need conftest update)**

The router tests will fail because conftest still uses RagQueryProcessor. We fix that in Task 6. For now, verify the import structure is correct:

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && python -c "from chatbot_plugin.main import app; print('OK')"`
Expected: `OK` (if env vars are not set, it will still import — lifespan runs at startup, not import time)

Actually, the import of `FastEmbedReranker` will fail if fastembed is not installed. Let's make that optional.

- [ ] **Step 4: Make FastEmbedReranker import optional in main.py**

Update the lifespan in `main.py` — the reranker import should be lazy:

```python
    reranker = None
    if getenv("CHATBOT_ENABLE_RERANKER", "").lower() in ("1", "true", "yes"):
        try:
            from chatbot_plugin_sdk import FastEmbedReranker
            reranker = FastEmbedReranker()
        except ImportError:
            pass
```

And remove `FastEmbedReranker` from the top-level imports in `main.py`.

- [ ] **Step 5: Verify import works**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && python -c "from chatbot_plugin.main import app; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin
git add src/chatbot_plugin/main.py src/chatbot_plugin/routers.py
git commit -m "feat: wire ChatService into main.py and routers.py, remove RagQueryProcessor"
```

---

## Task 6: Update tests (conftest.py and test_toolbox.py)

**Files:**
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/tests/conftest.py`
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/src/tests/routers/test_toolbox.py`

- [ ] **Step 1: Rewrite conftest.py**

Replace `src/tests/conftest.py` with:

```python
"""Shared test fixtures."""
import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI
from unittest.mock import AsyncMock, MagicMock

from chatbot_plugin.chat_service import ChatService, ChatResult
from chatbot_plugin.main import app as toolbox_app


@pytest.fixture
def app() -> FastAPI:
    """Return the toolbox FastAPI app with a mock ChatService attached."""
    mock_service = MagicMock(spec=ChatService)
    mock_service.chat = AsyncMock(return_value=ChatResult(
        reply="RAG is a retrieval technique...",
        articles_used=[],
        chunks=[],
    ))
    toolbox_app.state.chat_service = mock_service
    return toolbox_app


@pytest.fixture
def mock_service(app: FastAPI) -> MagicMock:
    """Return the mock ChatService attached to the app."""
    return app.state.chat_service


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async test client for the toolbox API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

- [ ] **Step 2: Rewrite test_toolbox.py**

Replace `src/tests/routers/test_toolbox.py` with:

```python
"""Router tests — validate /v1/chat/completions OpenAI-compatible endpoint."""
import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient

from chatbot_plugin.chat_service import ChatResult


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_accepts_valid_messages(self, client: AsyncClient, mock_service):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is RAG?"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "RAG is a retrieval technique..."
        assert "usage" in data
        mock_service.chat.assert_called_once_with("What is RAG?")

    @pytest.mark.asyncio
    async def test_rejects_empty_messages(self, client: AsyncClient):
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_no_messages_field(self, client: AsyncClient):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_default_model(self, client: AsyncClient, mock_service):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Hi"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "rag-default"
        mock_service.chat.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_only_system_message_returns_400(self, client: AsyncClient):
        """No user message → 400 because there's nothing to query."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                ],
            },
        )
        assert resp.status_code == 400
```

- [ ] **Step 3: Run all tests**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin
git add src/tests/conftest.py src/tests/routers/test_toolbox.py
git commit -m "test: update conftest and router tests to use ChatService"
```

---

## Task 7: Update pyproject.toml dependencies

**Files:**
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/pyproject.toml`

- [ ] **Step 1: Add new dependencies**

Update `pyproject.toml` — add `anthropic`, `google-genai`, `tenacity` to core dependencies, move `httpx` from dev to core:

```toml
[project]
name = "chatbot-plugin"
version = "0.3.0"
description = "OpenAI-compatible RAG chat backend"
requires-python = ">=3.11"
dependencies = [
    "fastapi",
    "uvicorn[standard]",
    "pydantic>=2.0",
    "chatbot-plugin-sdk",
    "anthropic",
    "google-genai>=1.0",
    "httpx",
    "tenacity",
]

[tool.uv.sources]
chatbot-plugin-sdk = { path = "../chatbot-plugin-sdk", editable = true }

[dependency-groups]
dev = [
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
]
```

Note: `httpx` removed from dev (moved to core). `anthropic`, `google-genai>=1.0`, `tenacity` added to core.

- [ ] **Step 2: Run uv lock and sync**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv lock && uv sync`
Expected: Dependencies resolved and installed

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin
git add pyproject.toml uv.lock
git commit -m "feat: add anthropic, google-genai, tenacity, httpx to dependencies"
```

---

## Task 8: Remove obsolete scripts

**Files:**
- Delete: `/home/pegaai/Desktop/sdd/chatbot-plugin/scripts/seed.py`
- Delete: `/home/pegaai/Desktop/sdd/chatbot-plugin/scripts/query.py`

- [ ] **Step 1: Delete obsolete scripts**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin
rm scripts/seed.py scripts/query.py
```

- [ ] **Step 2: Check if scripts/ directory is now empty, remove if so**

Run: `ls scripts/`
If only `fix_sparsevec.py` remains, keep the directory. If empty, remove it.

- [ ] **Step 3: Commit**

```bash
git add -u scripts/
git commit -m "chore: remove obsolete seed.py and query.py scripts"
```

---

## Task 9: Update specs and CLAUDE.md

**Files:**
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/specs/toolbox-api.md`
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/specs/integration.md`
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/specs/rag-pipeline.md`
- Modify: `/home/pegaai/Desktop/sdd/chatbot-plugin/CLAUDE.md`

- [ ] **Step 1: Rewrite specs/toolbox-api.md**

Replace with content documenting only the `/v1/chat/completions` endpoint (OpenAI-compatible format), removing all `/tools/*` endpoints. The spec should describe the request/response format which is already defined in `contracts/chat_completion.py`, plus the environment variables.

- [ ] **Step 2: Rewrite specs/integration.md**

Remove all ingestion HTTP examples. Document that:
- Ingestion is done via SDK (`IngestProcessor`) in-process
- Chat endpoint is `POST /v1/chat/completions`
- Environment variables for configuration

- [ ] **Step 3: Update specs/rag-pipeline.md**

Remove the `/tools/chat` endpoint section. Update Phase 3 architecture to reflect:
- `RetrieveProcessor` with `min_score`/`min_rerank_score` gating
- `ChatService` for prompt assembly + LLM generation
- `ResilientLLMService` with Claude/Gemini/OpenRouter fallback
- Raw context fallback when no LLM keys configured

- [ ] **Step 4: Rewrite CLAUDE.md**

Update to reflect actual codebase:

```markdown
# CLAUDE.md

## Project Overview

OpenAI-compatible RAG chat backend. Receives user prompts, retrieves relevant chunks from PostgreSQL + pgvector via SDK's RetrieveProcessor, assembles context, and generates replies via a resilient LLM fallback chain (Claude → Gemini → OpenRouter).

## SDD (Spec-Driven Development) Workflow

**Spec is source of truth. Code follows spec, not the other way around.**

### Rules

1. **Spec-first**: Before writing or changing any code, read and update `specs/` first
2. **Spec files**:
   - `specs/toolbox-api.md` — API contract
   - `specs/rag-pipeline.md` — Retrieval and generation internals
   - `specs/integration.md` — How external services integrate
3. **Contracts = spec in code**: `src/chatbot_plugin/contracts/` contains Pydantic models that mirror the spec
4. **API changes must start from spec**: Change spec → update contracts → update router/service → tests pass

### Development Order

```
1. Write/update specs/*.md
2. Write/update contracts/*.py (Pydantic models from spec)
3. Write/update tests/ (based on spec cases)
4. Implement routers/ + service/ (make tests green)
```

## Project Structure

```
specs/                           # SDD spec files (source of truth)
  toolbox-api.md                 # API contract (POST /v1/chat/completions)
  rag-pipeline.md                # Retrieval and generation internals
  integration.md                 # How external services integrate
src/chatbot_plugin/
  contracts/                     # Pydantic models = spec in code
    chat_completion.py           # OpenAI-compatible request/response models
  llm/                           # LLM provider layer
    base.py                      # LLMProvider protocol, ResilientLLMService
    claude_provider.py           # Anthropic Claude
    gemini_provider.py           # Google Gemini
    openrouter_provider.py       # OpenRouter
  chat_service.py                # Core: retrieve → gate → assemble → generate
  main.py                        # FastAPI app + lifespan
  routers.py                     # POST /v1/chat/completions endpoint
src/tests/
  llm/                           # LLM provider tests
  routers/                       # API endpoint tests
  test_chat_service.py           # ChatService unit tests
alembic/                         # Database migrations
```

## Commands

- **Run server:** `uvicorn chatbot_plugin.main:app --reload`
- **Test:** `uv run pytest src/tests/ -v`
- **Coverage:** `uv run pytest src/tests/ --cov=chatbot_plugin --cov-report=html`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin` | Database URL |
| `CHATBOT_EMBEDDING_MODEL_API` | `""` | Embedding service URL |
| `CHATBOT_ENABLE_RERANKER` | `""` | Set to "true" to enable cross-encoder reranking |
| `CHATBOT_RETRIEVAL_MIN_SCORE` | `0.0` | Pre-rerank score threshold |
| `CHATBOT_RERANKER_MIN_SCORE` | `0.7` | Post-rerank score threshold |
| `CHATBOT_MAX_CONTEXT_CHUNKS` | `10` | Max chunks in context |
| `CHATBOT_MAX_TOKENS` | `2048` | Max LLM output tokens |
| `CHATBOT_CLAUDE_API_KEY` | `""` | Anthropic API key |
| `CHATBOT_CLAUDE_MODEL` | `claude-sonnet-4-6-20250514` | Claude model |
| `CHATBOT_GEMINI_API_KEY` | `""` | Google Gemini API key |
| `CHATBOT_GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model |
| `CHATBOT_OPENROUTER_API_KEY` | `""` | OpenRouter API key |
| `CHATBOT_OPENROUTER_MODEL` | `meta-llama/llama-3-70b` | OpenRouter model |
```

- [ ] **Step 5: Commit**

```bash
cd /home/pegaai/Desktop/sdd/chatbot-plugin
git add specs/ CLAUDE.md
git commit -m "docs: update specs and CLAUDE.md to reflect generation layer architecture"
```

---

## Task 10: Run full test suite and verify

- [ ] **Step 1: Run all plugin tests**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && uv run pytest src/tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Run all SDK tests**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && uv run pytest src/tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 3: Verify server starts without errors**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin && timeout 5 uv run python -c "from chatbot_plugin.main import app; print('Server import OK')" || true`
Expected: `Server import OK`

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address test failures from integration"
```
