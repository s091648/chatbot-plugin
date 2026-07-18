"""Tests for LLM provider protocol and ResilientLLMService."""
from unittest.mock import AsyncMock

import pytest

from chatbot_plugin_sdk import SlidingWindowStrategy, RateLimitExhausted

from chatbot_plugin.llm.base import (
    AllProvidersExhausted,
    LLMProvider,
    LLMResult,
    ProviderHandler,
    ResilientLLMService,
    ToolCallRequest,
    ToolSpec,
)


def _mock_provider(name: str = "mock", model: str = "mock-model", result: LLMResult | None = None) -> AsyncMock:
    provider = AsyncMock(spec=LLMProvider)
    provider.model = model
    provider.complete = AsyncMock(return_value=result or LLMResult(thinking=None, text="Generated text"))
    return provider


def _handler(name: str = "mock", priority: int = 1, rpm: int = 0, result: LLMResult | None = None) -> ProviderHandler:
    provider = _mock_provider(name=name, result=result)
    strategy = SlidingWindowStrategy(rpm=rpm)
    return ProviderHandler(provider=provider, strategy=strategy, priority=priority, name=name)


class TestResilientLLMService:
    @pytest.mark.asyncio
    async def test_calls_highest_priority_handler(self):
        h1 = _handler(name="first", priority=1)
        h2 = _handler(name="second", priority=2)
        service = ResilientLLMService([h2, h1])
        result, handler = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result.text == "Generated text"
        assert handler.name == "first"
        h1.provider.complete.assert_called_once()
        h2.provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_on_provider_failure(self):
        h1 = _handler(name="failing", priority=1)
        h1.provider.complete = AsyncMock(side_effect=RuntimeError("API down"))
        h2 = _handler(name="backup", priority=2)
        service = ResilientLLMService([h1, h2])
        result, handler = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result.text == "Generated text"
        assert handler.name == "backup"
        h2.provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_moves_handler_to_end_on_rate_limit(self):
        h1 = _handler(name="rate_limited", priority=1, rpm=1)
        h2 = _handler(name="backup", priority=2)
        h1.provider.complete = AsyncMock(side_effect=RateLimitExhausted("daily cap"))
        service = ResilientLLMService([h1, h2])
        await service.complete([{"role": "user", "content": "hi"}], 100)
        assert service._handlers[0].name == "backup"

    @pytest.mark.asyncio
    async def test_raises_when_no_handlers(self):
        service = ResilientLLMService([])
        with pytest.raises(AllProvidersExhausted):
            await service.complete([{"role": "user", "content": "hi"}], 100)

    @pytest.mark.asyncio
    async def test_raises_when_all_handlers_fail(self):
        h1 = _handler(name="h1", priority=1)
        h2 = _handler(name="h2", priority=2)
        h1.provider.complete = AsyncMock(side_effect=RuntimeError("down"))
        h2.provider.complete = AsyncMock(side_effect=RuntimeError("down"))
        service = ResilientLLMService([h1, h2])
        with pytest.raises(AllProvidersExhausted):
            await service.complete([{"role": "user", "content": "hi"}], 100)

    @pytest.mark.asyncio
    async def test_exclude_skips_named_handlers(self):
        h1 = _handler(name="first", priority=1)
        h2 = _handler(name="second", priority=2)
        service = ResilientLLMService([h1, h2])
        result, handler = await service.complete(
            [{"role": "user", "content": "hi"}], 100, exclude={"first"}
        )
        assert handler.name == "second"
        h1.provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_exclude_all_raises(self):
        h1 = _handler(name="first", priority=1)
        service = ResilientLLMService([h1])
        with pytest.raises(AllProvidersExhausted):
            await service.complete([{"role": "user", "content": "hi"}], 100, exclude={"first"})

    @pytest.mark.asyncio
    async def test_pinned_handler_bypasses_fallback_chain(self):
        h1 = _handler(name="pinned", priority=1)
        h2 = _handler(name="other", priority=2)
        service = ResilientLLMService([h2])  # "pinned" is intentionally not even in the chain
        result, handler = await service.complete([{"role": "user", "content": "hi"}], 100, pinned_handler=h1)
        assert handler is h1
        h1.provider.complete.assert_called_once()
        h2.provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_pinned_handler_failure_raises_without_fallback(self):
        h1 = _handler(name="pinned", priority=1)
        h1.provider.complete = AsyncMock(side_effect=RuntimeError("down"))
        service = ResilientLLMService([])
        with pytest.raises(AllProvidersExhausted):
            await service.complete([{"role": "user", "content": "hi"}], 100, pinned_handler=h1)

    @pytest.mark.asyncio
    async def test_passes_tools_through_to_provider(self):
        h1 = _handler(name="first", priority=1)
        service = ResilientLLMService([h1])
        tools = [ToolSpec(name="search_articles", description="search", input_schema={})]
        await service.complete([{"role": "user", "content": "hi"}], 100, tools=tools)
        h1.provider.complete.assert_called_once_with([{"role": "user", "content": "hi"}], 100, tools)

    @pytest.mark.asyncio
    async def test_result_can_carry_tool_calls(self):
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"})
        h1 = _handler(name="first", priority=1, result=LLMResult(thinking=None, text="", tool_calls=[tool_call]))
        service = ResilientLLMService([h1])
        result, _ = await service.complete([{"role": "user", "content": "hi"}], 100)
        assert result.tool_calls == [tool_call]


class TestProviderHandler:
    @pytest.mark.asyncio
    async def test_complete_delegates_to_provider(self):
        handler = _handler()
        result = await handler.complete([{"role": "user", "content": "hello"}], 500)
        assert result.text == "Generated text"
        handler.provider.complete.assert_called_once_with(
            [{"role": "user", "content": "hello"}], 500, None
        )
