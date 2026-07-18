"""Tests for LLM provider implementations (mocked API calls)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin.llm.base import LLMProvider, ToolCallRequest, ToolSpec
from chatbot_plugin.llm.claude_provider import ClaudeProvider
from chatbot_plugin.llm.gemini_provider import GeminiProvider
from chatbot_plugin.llm.openrouter_provider import OpenRouterProvider


MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is RAG?"},
]


class TestClaudeProvider:
    def test_satisfies_protocol(self):
        provider = ClaudeProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_complete_sends_system_separately(self):
        provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-6-20250514")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="RAG is retrieval-augmented generation.")]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=20)
        with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response) as mock_create:
            result = await provider.complete(MESSAGES, 1024)
        assert result.text == "RAG is retrieval-augmented generation."
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6-20250514"
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["system"] == "You are helpful."
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_complete_sends_tools_when_provided(self):
        provider = ClaudeProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]
        mock_response.usage = MagicMock(input_tokens=0, output_tokens=0)
        tools = [ToolSpec(name="search_articles", description="search", input_schema={"type": "object"})]
        with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response) as mock_create:
            await provider.complete(MESSAGES, 100, tools=tools)
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["tools"] == [{"name": "search_articles", "description": "search", "input_schema": {"type": "object"}}]

    @pytest.mark.asyncio
    async def test_complete_parses_tool_use_block(self):
        provider = ClaudeProvider(api_key="test-key")
        tool_block = MagicMock(type="tool_use", id="toolu_1", input={"query": "foo"})
        tool_block.name = "search_articles"
        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.usage = MagicMock(input_tokens=0, output_tokens=0)
        with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.complete(MESSAGES, 100, tools=[ToolSpec("search_articles", "d", {})])
        assert result.tool_calls == [ToolCallRequest(id="toolu_1", name="search_articles", arguments={"query": "foo"})]
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_complete_translates_tool_round_trip_messages(self):
        provider = ClaudeProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="final answer")]
        mock_response.usage = MagicMock(input_tokens=0, output_tokens=0)
        round_trip_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "toolu_1", "name": "search_articles", "arguments": {"query": "foo"}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "name": "search_articles", "content": "[1] Title\nchunk text", "is_error": False},
        ]
        with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response) as mock_create:
            await provider.complete(round_trip_messages, 100)
        sent = mock_create.call_args.kwargs["messages"]
        assert sent[1] == {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "search_articles", "input": {"query": "foo"}}]}
        assert sent[2] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "[1] Title\nchunk text", "is_error": False}],
        }


class TestGeminiProvider:
    def test_satisfies_protocol(self):
        provider = GeminiProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_complete_sends_messages(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-2.0-flash")
        mock_response = MagicMock()
        mock_response.text = "RAG is a retrieval technique."
        finish_reason = MagicMock()
        finish_reason.name = "STOP"
        mock_response.candidates = [MagicMock(finish_reason=finish_reason)]
        with patch.object(provider._client.models, "generate_content", return_value=mock_response) as mock_gen:
            result = await provider.complete(MESSAGES, 1024)
        assert result.text == "RAG is a retrieval technique."
        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_returns_empty_on_blocked(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-2.0-flash")
        mock_response = MagicMock()
        finish_reason = MagicMock()
        finish_reason.name = "SAFETY"
        mock_response.candidates = [MagicMock(finish_reason=finish_reason)]
        mock_response.text = ""
        with patch.object(provider._client.models, "generate_content", return_value=mock_response):
            result = await provider.complete(MESSAGES, 1024)
        assert result.text == ""

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
        provider = OpenRouterProvider(api_key="test-key", model="test-model")
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
        assert result.text == "RAG combines retrieval with generation."
