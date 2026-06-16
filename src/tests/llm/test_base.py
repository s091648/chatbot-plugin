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
        service = ResilientLLMService([h2, h1])
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
        h1.provider.complete = AsyncMock(side_effect=RateLimitExhausted("daily cap"))
        service = ResilientLLMService([h1, h2])
        result = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result == "Generated text"
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
