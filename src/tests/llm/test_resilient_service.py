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
