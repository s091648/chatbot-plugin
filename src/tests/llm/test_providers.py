"""Tests for LLM provider implementations (mocked API calls)."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot_plugin.llm.base import LLMProvider, TextDelta, ThinkingDelta, ToolCallRequest, ToolSpec
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
        assert "thinking" in call_kwargs

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
    async def test_complete_disables_thinking_when_tools_provided(self):
        """Extended thinking + tools is rejected by the real API unless the original signed
        thinking block is threaded through the tool-call round-trip, which this provider's
        round-trip reconstruction does not do. Thinking must be omitted whenever tools are sent."""
        provider = ClaudeProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]
        mock_response.usage = MagicMock(input_tokens=0, output_tokens=0)
        tools = [ToolSpec(name="search_articles", description="search", input_schema={"type": "object"})]
        with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response) as mock_create:
            await provider.complete(MESSAGES, 100, tools=tools)
        call_kwargs = mock_create.call_args.kwargs
        assert "thinking" not in call_kwargs

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

    @pytest.mark.asyncio
    async def test_stream_yields_thinking_and_text_deltas(self):
        provider = ClaudeProvider(api_key="test-key")
        events = [
            MagicMock(type="content_block_delta", index=0, delta=MagicMock(type="thinking_delta", thinking="pondering")),
            MagicMock(type="content_block_delta", index=0, delta=MagicMock(type="text_delta", text="Hello")),
            MagicMock(type="content_block_delta", index=0, delta=MagicMock(type="text_delta", text=" world")),
        ]
        with patch.object(provider._client.messages, "stream", return_value=_FakeClaudeStream(events)):
            emitted = [e async for e in provider.stream(MESSAGES, 100)]
        assert emitted == [ThinkingDelta(text="pondering"), TextDelta(text="Hello"), TextDelta(text=" world")]

    @pytest.mark.asyncio
    async def test_stream_assembles_tool_use_from_input_json_deltas(self):
        provider = ClaudeProvider(api_key="test-key")
        tool_use_block = MagicMock(type="tool_use", id="toolu_1")
        tool_use_block.name = "search_articles"  # MagicMock(name=...) is reserved for repr, not an attribute
        events = [
            MagicMock(type="content_block_start", index=0, content_block=tool_use_block),
            MagicMock(type="content_block_delta", index=0, delta=MagicMock(type="input_json_delta", partial_json='{"query"')),
            MagicMock(type="content_block_delta", index=0, delta=MagicMock(type="input_json_delta", partial_json=': "foo"}')),
            MagicMock(type="content_block_stop", index=0),
        ]
        with patch.object(provider._client.messages, "stream", return_value=_FakeClaudeStream(events)):
            emitted = [e async for e in provider.stream(MESSAGES, 100, tools=[ToolSpec("search_articles", "d", {})])]
        assert emitted == [ToolCallRequest(id="toolu_1", name="search_articles", arguments={"query": "foo"})]


class _FakeClaudeStream:
    """Stands in for anthropic's `async with client.messages.stream(...) as stream:` context
    manager — `stream` itself is what's async-iterated, not something returned by __aenter__."""
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for event in self._events:
            yield event


class TestGeminiProvider:
    def test_satisfies_protocol(self):
        provider = GeminiProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_complete_requests_thought_summaries_with_automatic_budget(self):
        """include_thoughts alone isn't enough — some models (gemini-2.5-flash-lite) default
        thinking to budget 0 (disabled), so there'd be nothing to summarize without also
        forcing a non-zero (automatic) budget."""
        provider = GeminiProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "ok"
        finish_reason = MagicMock()
        finish_reason.name = "STOP"
        mock_response.candidates = [MagicMock(finish_reason=finish_reason)]
        with patch.object(provider._client.models, "generate_content", return_value=mock_response) as mock_gen:
            await provider.complete(MESSAGES, 100)
        thinking_config = mock_gen.call_args.kwargs["config"].thinking_config
        assert thinking_config.include_thoughts is True
        assert thinking_config.thinking_budget == -1

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
        fc_part = MagicMock(text=None, thought=False, thought_signature=b"sig-bytes")
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
        assert result.tool_calls[0].thought_signature == b"sig-bytes"

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

    @pytest.mark.asyncio
    async def test_complete_round_trip_reattaches_thought_signature(self):
        """Gemini 3 rejects a follow-up turn whose function-call part is missing the
        thought_signature the model originally attached to it — this must survive the
        round trip through the generic message dict, not just the raw SDK response."""
        provider = GeminiProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "final answer"
        finish_reason = MagicMock()
        finish_reason.name = "STOP"
        mock_response.candidates = [MagicMock(finish_reason=finish_reason)]
        round_trip_messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_0", "name": "search_articles", "arguments": {"query": "foo"},
                    "thought_signature": b"sig-bytes",
                }],
            },
            {"role": "tool", "tool_call_id": "call_0", "name": "search_articles", "content": "result", "is_error": False},
        ]
        with patch.object(provider._client.models, "generate_content", return_value=mock_response) as mock_gen:
            await provider.complete(round_trip_messages, 100)
        contents = mock_gen.call_args.kwargs["contents"]
        assert contents[1].parts[0].thought_signature == b"sig-bytes"

    @pytest.mark.asyncio
    async def test_stream_yields_thinking_and_text_deltas(self):
        provider = GeminiProvider(api_key="test-key")
        thinking_part = MagicMock(text="pondering", thought=True, function_call=None)
        text_part = MagicMock(text="Hello", thought=False, function_call=None)
        stop = MagicMock()
        stop.name = "STOP"

        async def _chunks():
            yield MagicMock(candidates=[MagicMock(finish_reason=None, content=MagicMock(parts=[thinking_part]))])
            yield MagicMock(candidates=[MagicMock(finish_reason=stop, content=MagicMock(parts=[text_part]))])

        with patch.object(provider._client.aio.models, "generate_content_stream", new_callable=AsyncMock, return_value=_chunks()):
            events = [e async for e in provider.stream(MESSAGES, 100)]
        assert events == [ThinkingDelta(text="pondering"), TextDelta(text="Hello")]

    @pytest.mark.asyncio
    async def test_stream_yields_tool_call_with_thought_signature(self):
        provider = GeminiProvider(api_key="test-key")
        fc_mock = MagicMock(args={"query": "foo"})
        fc_mock.name = "search_articles"
        fc_part = MagicMock(text=None, thought=False, thought_signature=b"sig-bytes")
        fc_part.function_call = fc_mock
        stop = MagicMock()
        stop.name = "STOP"

        async def _chunks():
            yield MagicMock(candidates=[MagicMock(finish_reason=stop, content=MagicMock(parts=[fc_part]))])

        with patch.object(provider._client.aio.models, "generate_content_stream", new_callable=AsyncMock, return_value=_chunks()):
            events = [e async for e in provider.stream(MESSAGES, 100, tools=[ToolSpec("search_articles", "d", {})])]
        assert events == [ToolCallRequest(
            id="call_0", name="search_articles", arguments={"query": "foo"}, thought_signature=b"sig-bytes",
        )]

    @pytest.mark.asyncio
    async def test_stream_raises_rate_limit_on_daily_quota_error(self):
        from chatbot_plugin_sdk import RateLimitExhausted
        provider = GeminiProvider(api_key="test-key")
        error = Exception("429 RESOURCE_EXHAUSTED PerDay limit exceeded")
        with patch.object(provider._client.aio.models, "generate_content_stream", new_callable=AsyncMock, side_effect=error):
            with pytest.raises(RateLimitExhausted):
                async for _ in provider.stream(MESSAGES, 100):
                    pass

    # ── finish_reason / empty-output diagnostics (regression) ──────────────────────────────
    # Root cause of the original silent-empty-response bug: the terminal chunk carrying
    # finish_reason typically has content=None and was being skipped before finish_reason was
    # ever read, so neither a blocked completion nor a clean-but-empty one was ever logged.

    @pytest.mark.asyncio
    async def test_stream_logs_warning_when_blocked(self, caplog):
        provider = GeminiProvider(api_key="test-key")
        blocked = MagicMock()
        blocked.name = "SAFETY"

        async def _chunks():
            # The terminal chunk: content=None, finish_reason set — exactly the chunk the old
            # code's `or chunk.candidates[0].content is None: continue` skipped before ever
            # looking at finish_reason.
            yield MagicMock(candidates=[MagicMock(finish_reason=blocked, content=None)])

        with caplog.at_level("WARNING"):
            with patch.object(provider._client.aio.models, "generate_content_stream", new_callable=AsyncMock, return_value=_chunks()):
                events = [e async for e in provider.stream(MESSAGES, 100)]
        assert events == []
        assert any(r.message == "gemini_stream_blocked" for r in caplog.records)

    @pytest.mark.asyncio
    async def test_stream_logs_warning_when_finished_cleanly_with_no_output(self):
        """The exact bug a real user hit: finish_reason STOP (not blocked), but only a thinking
        delta was ever produced — no text, no tool call."""
        provider = GeminiProvider(api_key="test-key")
        thinking_part = MagicMock(text="thinking about it...", thought=True, function_call=None)
        stop = MagicMock()
        stop.name = "STOP"

        async def _chunks():
            yield MagicMock(candidates=[MagicMock(finish_reason=None, content=MagicMock(parts=[thinking_part]))])
            yield MagicMock(candidates=[MagicMock(finish_reason=stop, content=None)])

        with patch.object(provider._client.aio.models, "generate_content_stream", new_callable=AsyncMock, return_value=_chunks()):
            with patch("chatbot_plugin.llm.gemini_provider.logger") as mock_logger:
                events = [e async for e in provider.stream(MESSAGES, 100)]
        assert events == [ThinkingDelta(text="thinking about it...")]
        mock_logger.warning.assert_any_call(
            "gemini_stream_empty_response",
            extra={"model": provider.model, "finish_reason": "STOP"},
        )

    @pytest.mark.asyncio
    async def test_stream_does_not_warn_when_text_was_produced(self):
        provider = GeminiProvider(api_key="test-key")
        text_part = MagicMock(text="Hello", thought=False, function_call=None)
        stop = MagicMock()
        stop.name = "STOP"

        async def _chunks():
            yield MagicMock(candidates=[MagicMock(finish_reason=stop, content=MagicMock(parts=[text_part]))])

        with patch.object(provider._client.aio.models, "generate_content_stream", new_callable=AsyncMock, return_value=_chunks()):
            with patch("chatbot_plugin.llm.gemini_provider.logger") as mock_logger:
                events = [e async for e in provider.stream(MESSAGES, 100)]
        assert events == [TextDelta(text="Hello")]
        warned = [c for c in mock_logger.warning.call_args_list if c.args and c.args[0].startswith("gemini_stream")]
        assert warned == []

    @pytest.mark.asyncio
    async def test_complete_logs_warning_when_no_reply_text_and_no_tool_calls(self):
        provider = GeminiProvider(api_key="test-key")
        thinking_part = MagicMock(text="thinking about it...", thought=True, function_call=None)
        mock_response = MagicMock()
        mock_response.text = ""
        finish_reason = MagicMock()
        finish_reason.name = "STOP"
        mock_response.candidates = [MagicMock(finish_reason=finish_reason, content=MagicMock(parts=[thinking_part]))]
        with patch.object(provider._client.models, "generate_content", return_value=mock_response):
            with patch("chatbot_plugin.llm.gemini_provider.logger") as mock_logger:
                result = await provider.complete(MESSAGES, 100)
        assert result.text == ""
        mock_logger.warning.assert_any_call(
            "gemini_no_actionable_output",
            extra={"model": provider.model, "finish_reason": "STOP", "has_thinking": True},
        )


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
    async def test_complete_requests_reasoning(self):
        """No-op for non-reasoning models (OpenRouter drops unsupported params), but needed
        for reasoning-capable models routed through OpenRouter to surface thinking at all."""
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            await provider.complete(MESSAGES, 1024)
        assert mock_client.post.call_args.kwargs["json"]["reasoning"] == {"enabled": True}

    @pytest.mark.asyncio
    async def test_complete_surfaces_reasoning_as_thinking(self):
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "answer", "reasoning": "pondering..."}}]}
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await provider.complete(MESSAGES, 1024)
        assert result.thinking == "pondering..."

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
    async def test_complete_handles_malformed_tool_call_json_gracefully(self):
        """Malformed JSON in a tool call's arguments must not crash complete() — it should
        fall back to an empty dict so ChatService's existing missing-argument handling kicks in."""
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "search_articles", "arguments": '{"query": "foo"'}}
            ]}}]
        }
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await provider.complete(MESSAGES, 1024)
        assert result.tool_calls[0].arguments == {}

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

    def _fake_sse_response(self, lines: list[str]) -> MagicMock:
        """Stands in for the httpx.Response yielded by `async with client.stream(...) as response`."""
        response = MagicMock()
        response.raise_for_status = MagicMock()

        async def _aiter_lines():
            for line in lines:
                yield line
        response.aiter_lines = _aiter_lines
        return response

    def _stream_context_manager(self, response: MagicMock) -> AsyncMock:
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=response)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    @pytest.mark.asyncio
    async def test_stream_yields_text_deltas(self):
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': 'Hello'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': ' world'}}]})}",
            "data: [DONE]",
        ]
        response = self._fake_sse_response(lines)
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.stream = MagicMock(return_value=self._stream_context_manager(response))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            events = [e async for e in provider.stream(MESSAGES, 1024)]
        assert events == [TextDelta(text="Hello"), TextDelta(text=" world")]
        sent_payload = mock_client.stream.call_args.kwargs["json"]
        assert sent_payload["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_assembles_fragmented_tool_call(self):
        """OpenAI-style streaming delivers a tool call's arguments as index-keyed fragments
        across multiple chunks — this must be buffered and only surfaced once complete."""
        provider = OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b")
        chunks = [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "search_articles", "arguments": ""}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"query"'}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ': "foo"}'}}
            ]}}]},
        ]
        lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
        response = self._fake_sse_response(lines)
        with patch("chatbot_plugin.llm.openrouter_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.stream = MagicMock(return_value=self._stream_context_manager(response))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            events = [e async for e in provider.stream(MESSAGES, 1024, tools=[ToolSpec("search_articles", "d", {})])]
        assert events == [ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "foo"})]
