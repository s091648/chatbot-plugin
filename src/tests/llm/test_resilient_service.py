"""Tests for ResilientLLMService fallback chain."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from chatbot_plugin.llm.rate_limit import NoOpStrategy, RateLimitExhausted
from chatbot_plugin.llm.resilient_llm_service import ProviderHandler, ResilientLLMService


def _handler(name: str, priority: int, generate_return: str | None = "ok",
             generate_side_effect=None) -> ProviderHandler:
    """Create a ProviderHandler with a mock provider."""
    provider = AsyncMock()
    if generate_side_effect:
        provider.generate.side_effect = generate_side_effect
    else:
        provider.generate.return_value = generate_return
    return ProviderHandler(
        provider=provider,
        strategy=NoOpStrategy(),
        priority=priority,
        name=name,
    )


class TestResilientLLMService:
    @pytest.mark.asyncio
    async def test_first_provider_succeeds(self):
        h1 = _handler("gemini", 1, generate_return="gemini reply")
        h2 = _handler("claude", 2, generate_return="claude reply")
        service = ResilientLLMService([h1, h2])

        result = await service.generate("sys", "human")
        assert result == "gemini reply"
        h2.provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_on_none(self):
        h1 = _handler("gemini", 1, generate_return=None)
        h2 = _handler("claude", 2, generate_return="claude reply")
        service = ResilientLLMService([h1, h2])

        result = await service.generate("sys", "human")
        assert result == "claude reply"

    @pytest.mark.asyncio
    async def test_fallback_on_rate_limit_exhausted(self):
        h1 = _handler("gemini", 1, generate_side_effect=RateLimitExhausted("daily"))
        h2 = _handler("claude", 2, generate_return="claude reply")
        service = ResilientLLMService([h1, h2])

        result = await service.generate("sys", "human")
        assert result == "claude reply"

    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_none(self):
        h1 = _handler("gemini", 1, generate_return=None)
        h2 = _handler("claude", 2, generate_return=None)
        service = ResilientLLMService([h1, h2])

        result = await service.generate("sys", "human")
        assert result is None

    @pytest.mark.asyncio
    async def test_handlers_sorted_by_priority(self):
        h2 = _handler("claude", 2, generate_return="claude reply")
        h1 = _handler("gemini", 1, generate_return="gemini reply")
        service = ResilientLLMService([h2, h1])  # passed out of order

        result = await service.generate("sys", "human")
        assert result == "gemini reply"  # priority 1 tried first

    @pytest.mark.asyncio
    async def test_fallback_on_generic_exception(self):
        """Non-RateLimitExhausted exceptions also trigger fallback."""
        h1 = _handler("gemini", 1, generate_side_effect=RuntimeError("boom"))
        h2 = _handler("claude", 2, generate_return="claude reply")
        service = ResilientLLMService([h1, h2])

        result = await service.generate("sys", "human")
        assert result == "claude reply"

    @pytest.mark.asyncio
    async def test_mixed_failures_then_success(self):
        """First provider RateLimitExhausted, second returns None, third succeeds."""
        h1 = _handler("gemini", 1, generate_side_effect=RateLimitExhausted("daily"))
        h2 = _handler("claude", 2, generate_return=None)
        h3 = _handler("openrouter", 3, generate_return="or reply")
        service = ResilientLLMService([h1, h2, h3])

        result = await service.generate("sys", "human")
        assert result == "or reply"


class TestProviderHandler:
    @pytest.mark.asyncio
    async def test_generate_acquires_and_records_on_success(self):
        """ProviderHandler calls strategy.acquire and strategy.record_usage on success."""
        from chatbot_plugin.llm.resilient_llm_service import ProviderHandler
        strategy = AsyncMock()
        provider = AsyncMock()
        provider.generate.return_value = "success"
        handler = ProviderHandler(provider=provider, strategy=strategy, priority=1, name="test")

        result = await handler.generate("sys", "human")
        assert result == "success"
        strategy.acquire.assert_called_once()
        strategy.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_skips_record_usage_on_none(self):
        """ProviderHandler does not call record_usage when provider returns None."""
        from chatbot_plugin.llm.resilient_llm_service import ProviderHandler
        strategy = AsyncMock()
        provider = AsyncMock()
        provider.generate.return_value = None
        handler = ProviderHandler(provider=provider, strategy=strategy, priority=1, name="test")

        result = await handler.generate("sys", "human")
        assert result is None
        strategy.acquire.assert_called_once()
        strategy.record_usage.assert_not_called()
