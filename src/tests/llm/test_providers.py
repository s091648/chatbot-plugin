"""Tests for LLM provider config loading."""

import os
import tempfile

import pytest

from chatbot_plugin.llm.config import load_providers


def _write_toml(content: str) -> str:
    """Write content to a temp TOML file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "wb") as f:
        f.write(content.encode())
    return path


class TestLoadProviders:
    def test_loads_and_sorts_by_priority(self):
        toml = b"""
[[providers]]
name = "claude"
priority = 2
model = "claude-sonnet"
api_key_env = "CLAUDE_API_KEY"

[[providers]]
name = "gemini"
priority = 1
model = "gemini-flash"
api_key_env = "GEMINI_API_KEY"
"""
        fd, path = tempfile.mkstemp(suffix=".toml")
        with os.fdopen(fd, "wb") as f:
            f.write(toml)

        try:
            providers = load_providers(path)
            assert len(providers) == 2
            assert providers[0]["name"] == "gemini"
            assert providers[1]["name"] == "claude"
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_providers("/nonexistent/providers.toml")

    def test_empty_providers_returns_empty(self):
        toml = b"\n"
        fd, path = tempfile.mkstemp(suffix=".toml")
        with os.fdopen(fd, "wb") as f:
            f.write(toml)

        try:
            providers = load_providers(path)
            assert providers == []
        finally:
            os.unlink(path)

    def test_strategy_parsed_correctly(self):
        toml = b"""
[[providers]]
name = "gemini"
priority = 1
model = "gemini-flash"
api_key_env = "GEMINI_API_KEY"

[providers.strategy]
type = "sliding_window"
rpm = 5
tpm = 250000
rpd = 500
"""
        fd, path = tempfile.mkstemp(suffix=".toml")
        with os.fdopen(fd, "wb") as f:
            f.write(toml)

        try:
            providers = load_providers(path)
            assert providers[0]["strategy"]["rpm"] == 5
            assert providers[0]["strategy"]["tpm"] == 250000
        finally:
            os.unlink(path)
