"""Tests for individual LLM provider implementations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chatbot_plugin.llm.rate_limit.quota_strategy import RateLimitExhausted


# ── ClaudeProvider ──


class TestClaudeProvider:
    def _make_provider(self):
        with patch("chatbot_plugin.llm.claude_provider.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            from chatbot_plugin.llm.claude_provider import ClaudeProvider
            provider = ClaudeProvider(api_key="sk-test", model="claude-sonnet-4-6-20250514")
            return provider, mock_client

    @pytest.mark.asyncio
    async def test_call_api_success(self):
        provider, mock_client = self._make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        result = await provider._call_api("sys", "human")
        assert result == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_call_api_uses_correct_params(self):
        provider, mock_client = self._make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.usage.input_tokens = 0
        mock_response.usage.output_tokens = 0
        mock_client.messages.create.return_value = mock_response

        await provider._call_api("system-instr", "user-msg")
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6-20250514"
        assert call_kwargs.kwargs["system"] == "system-instr"
        assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "user-msg"}]


# ── GeminiProvider ──


class TestGeminiProvider:
    @pytest.mark.asyncio
    @patch("chatbot_plugin.llm.gemini_provider.genai")
    async def test_call_api_success(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai.GenerateContentConfig = MagicMock()
        from chatbot_plugin.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.text = "Hello from Gemini"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=5
        )
        mock_client.models.generate_content.return_value = mock_response

        result = await provider._call_api("sys", "human")
        assert result == "Hello from Gemini"

    @pytest.mark.asyncio
    @patch("chatbot_plugin.llm.gemini_provider.genai")
    async def test_call_api_no_candidates_returns_empty(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai.GenerateContentConfig = MagicMock()
        from chatbot_plugin.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")

        mock_response = MagicMock()
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response

        result = await provider._call_api("sys", "human")
        assert result == ""

    @pytest.mark.asyncio
    @patch("chatbot_plugin.llm.gemini_provider.genai")
    async def test_call_api_blocked_finish_reason_returns_empty(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai.GenerateContentConfig = MagicMock()
        from chatbot_plugin.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")

        mock_response = MagicMock()
        mock_candidate = MagicMock(finish_reason="SAFETY")
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        result = await provider._call_api("sys", "human")
        assert result == ""

    @pytest.mark.asyncio
    @patch("chatbot_plugin.llm.gemini_provider.genai")
    async def test_call_api_daily_quota_raises_rate_limit_exhausted(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai.GenerateContentConfig = MagicMock()
        from chatbot_plugin.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")

        mock_client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED: PerDay limit exceeded"
        )

        with pytest.raises(RateLimitExhausted):
            await provider._call_api("sys", "human")

    @pytest.mark.asyncio
    @patch("chatbot_plugin.llm.gemini_provider.genai")
    async def test_call_api_other_exception_reraises(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai.GenerateContentConfig = MagicMock()
        from chatbot_plugin.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")

        mock_client.models.generate_content.side_effect = RuntimeError("network error")

        with pytest.raises(RuntimeError, match="network error"):
            await provider._call_api("sys", "human")

    @pytest.mark.asyncio
    @patch("chatbot_plugin.llm.gemini_provider.genai")
    async def test_call_api_no_usage_metadata(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai.GenerateContentConfig = MagicMock()
        from chatbot_plugin.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.text = "ok"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        result = await provider._call_api("sys", "human")
        assert result == "ok"


# ── OpenRouterProvider ──


class TestOpenRouterProvider:
    def _make_provider(self):
        with patch("chatbot_plugin.llm.openrouter_provider.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.AsyncClient.return_value = mock_client
            from chatbot_plugin.llm.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(api_key="sk-test", model="test-model")
            return provider, mock_client

    @pytest.mark.asyncio
    async def test_call_api_success(self):
        provider, mock_client = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenRouter"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_client.post.return_value = mock_response

        result = await provider._call_api("sys", "human")
        assert result == "Hello from OpenRouter"

    @pytest.mark.asyncio
    async def test_call_api_uses_correct_params(self):
        provider, mock_client = self._make_provider()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        mock_client.post.return_value = mock_response

        await provider._call_api("system-instr", "user-msg")
        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["model"] == "test-model"
        assert body["messages"][0] == {"role": "system", "content": "system-instr"}
        assert body["messages"][1] == {"role": "user", "content": "user-msg"}

    @pytest.mark.asyncio
    async def test_call_api_missing_usage_defaults_to_zero(self):
        provider, mock_client = self._make_provider()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        mock_client.post.return_value = mock_response

        result = await provider._call_api("sys", "human")
        assert result == "ok"
