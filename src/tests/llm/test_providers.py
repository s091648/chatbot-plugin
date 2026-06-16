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
