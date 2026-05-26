"""Tests for BaseProvider retry logic and _is_retryable."""

import pytest

from chatbot_plugin.llm.base_provider import BaseProvider, _is_retryable
from chatbot_plugin.llm.rate_limit.quota_strategy import RateLimitExhausted


# ── _is_retryable() ──


def test_retryable_generic_exception():
    assert _is_retryable(ConnectionError("timeout")) is True


def test_retryable_runtime_error():
    assert _is_retryable(RuntimeError("oops")) is True


def test_not_retryable_rate_limit_exhausted():
    assert _is_retryable(RateLimitExhausted()) is False


def test_not_retryable_value_error():
    assert _is_retryable(ValueError("bad value")) is False


def test_not_retryable_key_error():
    assert _is_retryable(KeyError("missing")) is False


# ── BaseProvider.generate() ──


class _StubProvider(BaseProvider):
    """Test double that records calls and simulates _call_api behavior."""

    def __init__(self, side_effects: list):
        super().__init__(model="stub-model")
        self._side_effects = list(side_effects)
        self._call_count = 0

    async def _call_api(self, system_prompt: str, human_prompt: str) -> str:
        self._call_count += 1
        if self._side_effects:
            result = self._side_effects.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return "default"


@pytest.mark.asyncio
async def test_generate_success():
    provider = _StubProvider(["hello"])
    result = await provider.generate("sys", "human")
    assert result == "hello"


@pytest.mark.asyncio
async def test_generate_retries_on_transient_error():
    provider = _StubProvider([ConnectionError("timeout"), "recovered"])
    result = await provider.generate("sys", "human")
    assert result == "recovered"
    assert provider._call_count == 2


@pytest.mark.asyncio
async def test_generate_raises_rate_limit_exhausted():
    provider = _StubProvider([RateLimitExhausted()])
    with pytest.raises(RateLimitExhausted):
        await provider.generate("sys", "human")


@pytest.mark.asyncio
async def test_generate_returns_none_on_non_retryable_error():
    provider = _StubProvider([ValueError("bad data")])
    result = await provider.generate("sys", "human")
    assert result is None


@pytest.mark.asyncio
async def test_generate_returns_none_after_exhausted_retries():
    provider = _StubProvider([
        ConnectionError("fail1"),
        ConnectionError("fail2"),
        ConnectionError("fail3"),
    ])
    result = await provider.generate("sys", "human")
    assert result is None
    assert provider._call_count == 3
