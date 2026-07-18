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

    @pytest.mark.asyncio
    async def test_complete_sends_tools_when_provided(self):
        provider = GeminiProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "ok"
        finish_reason = MagicMock()
        finish_reason.name = "STOP"
        mock_response.candidates = [MagicMock(finish_reason=finish_reason)]
        tools = [ToolSpec(name="search_articles", description="search", input_schema={"type": "object"})]
        with patch.object(provider._client.models, "generate_content", return_value=mock_response) as mock_gen:
            await provider.complete(MESSAGES, 100, tools=tools)
        config = mock_gen.call_args.kwargs["config"]
        assert config.tools is not None

    @pytest.mark.asyncio
    async def test_complete_parses_function_call_part(self):
        provider = GeminiProvider(api_key="test-key")
        fc_part = MagicMock(text=None, thought=False)
        fc_mock = MagicMock(args={"query": "foo"})
        fc_mock.name = "search_articles"  # MagicMock(name=...) is reserved for repr, not an attribute
        fc_part.function_call = fc_mock
        finish_reason = MagicMock()
        finish_reason.name = "STOP"
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(finish_reason=finish_reason, content=MagicMock(parts=[fc_part]))]
        with patch.object(provider._client.models, "generate_content", return_value=mock_response):
            result = await provider.complete(MESSAGES, 100, tools=[ToolSpec("search_articles", "d", {})])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_articles"
        assert result.tool_calls[0].arguments == {"query": "foo"}

    @pytest.mark.asyncio
    async def test_complete_translates_tool_round_trip_messages(self):
        provider = GeminiProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "final answer"
        finish_reason = MagicMock()
        finish_reason.name = "STOP"
        mock_response.candidates = [MagicMock(finish_reason=finish_reason)]
        round_trip_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "name": "search_articles", "arguments": {"query": "foo"}}]},
            {"role": "tool", "tool_call_id": "call_0", "name": "search_articles", "content": "[1] Title\nchunk text", "is_error": False},
        ]
        with patch.object(provider._client.models, "generate_content", return_value=mock_response) as mock_gen:
            await provider.complete(round_trip_messages, 100)
        contents = mock_gen.call_args.kwargs["contents"]
        assert contents[1].role == "model"
        assert contents[1].parts[0].function_call.name == "search_articles"
        assert contents[2].role == "user"
        assert contents[2].parts[0].function_response.name == "search_articles"


class TestOpenRouterProvider:
    def test_satisfies_protocol(self):
        provider = OpenRouterProvider(api_key="test-key", model="test-model")
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_complete_sends_messages(self):
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "RAG combines retrieval with generation."}}]}
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await provider.complete(MESSAGES, 1024)
        assert result.text == "RAG combines retrieval with generation."

    @pytest.mark.asyncio
    async def test_complete_sends_tools_when_provided(self):
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status = MagicMock()
        tools = [ToolSpec(name="search_articles", description="search", input_schema={"type": "object"})]
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            await provider.complete(MESSAGES, 1024, tools=tools)
        sent_payload = mock_client.post.call_args.kwargs["json"]
        assert sent_payload["tools"] == [{"type": "function", "function": {"name": "search_articles", "description": "search", "parameters": {"type": "object"}}}]

    @pytest.mark.asyncio
    async def test_complete_parses_tool_calls_response(self):
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "search_articles", "arguments": '{"query": "foo"}'}}
            ]}}]
        }
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await provider.complete(MESSAGES, 1024)
        assert result.tool_calls == [ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "foo"})]
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_complete_translates_tool_round_trip_messages(self):
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "final answer"}}]}
        mock_response.raise_for_status = MagicMock()
        round_trip_messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "search_articles", "arguments": {"query": "foo"}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "search_articles", "content": "[1] Title\nchunk text", "is_error": False},
        ]
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            await provider.complete(round_trip_messages, 100)
        sent_messages = mock_client.post.call_args.kwargs["json"]["messages"]
        assert sent_messages[1]["tool_calls"][0]["function"]["arguments"] == '{"query": "foo"}'
        assert sent_messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "[1] Title\nchunk text"}
