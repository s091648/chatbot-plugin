"""Tests for LLM bootstrap factory."""

import os
import tempfile

import pytest

from chatbot_plugin.llm.bootstrap import build_llm_service, _create_provider, _create_strategy
from chatbot_plugin.llm.rate_limit import SlidingWindowStrategy, NoOpStrategy


# ── _create_provider() ──


def test_create_provider_claude():
    provider = _create_provider("claude", "key", "model")
    assert provider is not None
    assert provider._model == "model"


def test_create_provider_gemini():
    provider = _create_provider("gemini", "key", "model")
    assert provider is not None
    assert provider._model == "model"


def test_create_provider_openrouter():
    provider = _create_provider("openrouter", "key", "model")
    assert provider is not None
    assert provider._model == "model"


def test_create_provider_unknown_returns_none():
    assert _create_provider("unknown", "key", "model") is None


# ── _create_strategy() ──


def test_create_strategy_sliding_window():
    cfg = {"type": "sliding_window", "rpm": 5, "tpm": 1000, "rpd": 200}
    strategy = _create_strategy(cfg)
    assert isinstance(strategy, SlidingWindowStrategy)


def test_create_strategy_sliding_window_defaults():
    cfg = {"type": "sliding_window"}
    strategy = _create_strategy(cfg)
    assert isinstance(strategy, SlidingWindowStrategy)


def test_create_strategy_no_op_for_empty():
    strategy = _create_strategy({})
    assert isinstance(strategy, NoOpStrategy)


def test_create_strategy_no_op_for_unknown():
    strategy = _create_strategy({"type": "unknown"})
    assert isinstance(strategy, NoOpStrategy)


# ── build_llm_service() ──


def _write_toml(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def test_build_llm_service_raises_on_no_providers():
    path = _write_toml("")
    try:
        with pytest.raises(ValueError, match="No valid LLM providers"):
            build_llm_service(path)
    finally:
        os.unlink(path)


def test_build_llm_service_skips_missing_api_key(monkeypatch):
    path = _write_toml('[[providers]]\nname = "claude"\nmodel = "m"\napi_key_env = "NO_SUCH_KEY"\npriority = 1\n')
    # Ensure the env var is NOT set
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    try:
        with pytest.raises(ValueError, match="No valid LLM providers"):
            build_llm_service(path)
    finally:
        os.unlink(path)


def test_build_llm_service_skips_unknown_provider(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    path = _write_toml(
        '[[providers]]\nname = "unknown"\nmodel = "m"\napi_key_env = "TEST_KEY"\npriority = 1\n'
    )
    try:
        with pytest.raises(ValueError, match="No valid LLM providers"):
            build_llm_service(path)
    finally:
        os.unlink(path)


def test_build_llm_service_success(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    path = _write_toml(
        '[[providers]]\nname = "claude"\nmodel = "claude-sonnet-4-6-20250514"\napi_key_env = "TEST_KEY"\npriority = 1\n'
        '\n[[providers]]\nname = "gemini"\nmodel = "gemini-2.5-flash"\napi_key_env = "TEST_KEY"\npriority = 2\n'
    )
    try:
        service = build_llm_service(path)
        assert service is not None
    finally:
        os.unlink(path)
