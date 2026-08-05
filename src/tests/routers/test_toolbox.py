"""Router tests — validate /v1/chat/completions OpenAI-compatible endpoint."""

import json
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from chatbot_plugin.llm.base import AllProvidersExhausted, TextDelta, ThinkingDelta
from chatbot_plugin.services.chat_service import (
    ArticleRef,
    ChatResult,
    ChatService,
    SourcesReady,
    StreamFailed,
    ToolCallFinished,
    ToolCallStarted,
)


def _ok_result(reply="RAG is a retrieval technique..."):
    return ChatResult(reply=reply, articles_used=[], chunks=[])


def _fake_chat_stream(events):
    """Builds a ChatService.chat_stream replacement that yields a fixed event sequence,
    ignoring its arguments — mirrors how ChatService.chat's return value is mocked above,
    but chat_stream is an async generator method rather than a plain coroutine, so it can't
    be swapped in with AsyncMock(return_value=...)."""
    def _method(self, message, topic_id=None, pinned_article_ids=None):
        async def _gen():
            for event in events:
                yield event
        return _gen()
    return _method


def _failing_chat_stream(self, message, topic_id=None, pinned_article_ids=None):
    async def _gen():
        if False:
            yield  # pragma: no cover - makes this an async generator function
        raise AllProvidersExhausted()
    return _gen()


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_accepts_valid_messages(self, client: AsyncClient):
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _ok_result()
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "What is RAG?"},
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "RAG is a retrieval technique..."
            assert "usage" in data
            mock.assert_called_once_with("What is RAG?", topic_id=None, pinned_article_ids=None)

    @pytest.mark.asyncio
    async def test_rejects_empty_messages(self, client: AsyncClient):
        resp = await client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_no_messages_field(self, client: AsyncClient):
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_default_model(self, client: AsyncClient):
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _ok_result("Hello")
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
            assert resp.status_code == 200
            assert resp.json()["model"] == "rag-default"
            mock.assert_called_once_with("Hi", topic_id=None, pinned_article_ids=None)

    @pytest.mark.asyncio
    async def test_only_system_message_returns_400(self, client: AsyncClient):
        """No user message → 400 because there's nothing to query."""
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "You are helpful."}]},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_topic_id_forwarded_to_chat(self, client: AsyncClient):
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _ok_result()
            await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is RAG?"}],
                    "topic_id": "topic-uuid-abc",
                },
            )
            mock.assert_called_once_with(
                "What is RAG?", topic_id="topic-uuid-abc", pinned_article_ids=None
            )

    @pytest.mark.asyncio
    async def test_pinned_article_ids_forwarded_to_chat(self, client: AsyncClient):
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _ok_result()
            await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Summarise this article"}],
                    "pinned_article_ids": ["uuid-1", "uuid-2"],
                },
            )
            mock.assert_called_once_with(
                "Summarise this article",
                topic_id=None,
                pinned_article_ids=["uuid-1", "uuid-2"],
            )

    @pytest.mark.asyncio
    async def test_null_pinned_article_ids_forwarded_as_none(self, client: AsyncClient):
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = _ok_result()
            await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "pinned_article_ids": None,
                },
            )
            mock.assert_called_once_with("hi", topic_id=None, pinned_article_ids=None)


class TestStreamingResponse:
    @pytest.mark.asyncio
    async def test_stream_returns_sse_content_chunk(self, client: AsyncClient):
        events = [TextDelta(text="Hello world"), SourcesReady(articles=[])]
        with patch.object(ChatService, "chat_stream", _fake_chat_stream(events)):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = resp.text
            assert "Hello world" in body
            assert "data: [DONE]" in body

    @pytest.mark.asyncio
    async def test_stream_includes_thinking_event_when_present(self, client: AsyncClient):
        events = [
            ThinkingDelta(text="Chain of thought..."),
            TextDelta(text="Answer"),
            SourcesReady(articles=[]),
        ]
        with patch.object(ChatService, "chat_stream", _fake_chat_stream(events)):
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )
            thinking_line = next(
                (l for l in resp.text.splitlines() if l.startswith("data: ") and "thinking" in l),
                None,
            )
            assert thinking_line is not None
            assert json.loads(thinking_line[6:])["thinking"] == "Chain of thought..."

    @pytest.mark.asyncio
    async def test_stream_includes_sources_event_when_articles_used(self, client: AsyncClient):
        events = [
            TextDelta(text="Answer"),
            SourcesReady(articles=[
                ArticleRef(id="vec-id", title="My Article", url="https://example.com", public_article_id="pub-uuid", number=1),
            ]),
        ]
        with patch.object(ChatService, "chat_stream", _fake_chat_stream(events)):
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )
            sources_line = next(
                (l for l in resp.text.splitlines() if l.startswith("data: ") and "sources" in l),
                None,
            )
            assert sources_line is not None
            sources = json.loads(sources_line[6:])["sources"]
            assert sources[0]["id"] == "vec-id"
            assert sources[0]["public_article_id"] == "pub-uuid"
            # Frontend resolves a "[N]" citation marker by this number, not array position — see
            # ArticleRef.number and cited-content.tsx's resolveSourceIndex.
            assert sources[0]["number"] == 1

    @pytest.mark.asyncio
    async def test_stream_omits_sources_event_when_no_articles(self, client: AsyncClient):
        events = [TextDelta(text="Hi"), SourcesReady(articles=[])]
        with patch.object(ChatService, "chat_stream", _fake_chat_stream(events)):
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )
            has_sources = any(
                l.startswith("data: ") and "sources" in l for l in resp.text.splitlines()
            )
            assert not has_sources

    @pytest.mark.asyncio
    async def test_stream_omits_thinking_event_when_none(self, client: AsyncClient):
        events = [TextDelta(text="Hi"), SourcesReady(articles=[])]
        with patch.object(ChatService, "chat_stream", _fake_chat_stream(events)):
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )
            has_thinking = any(
                l.startswith("data: ") and "thinking" in l for l in resp.text.splitlines()
            )
            assert not has_thinking

    @pytest.mark.asyncio
    async def test_stream_emits_tool_call_frames_when_tools_executed(self, client: AsyncClient):
        events = [
            ToolCallStarted(id="call_1", name="search_articles", arguments={"query": "foo"}),
            ToolCallFinished(id="call_1", name="search_articles", result_summary="[1] T\ntext", is_error=False),
            TextDelta(text="Final answer [1]"),
            SourcesReady(articles=[ArticleRef(id="a1", title="T", url="http://x", public_article_id="pub1")]),
        ]
        with patch.object(ChatService, "chat_stream", _fake_chat_stream(events)):
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "question"}], "stream": True, "pinned_article_ids": ["pub1"]},
            )
            body = resp.text
            assert '"tool_calls":[{"id":"call_1"' in body.replace(" ", "")
            assert '"tool_result":{"tool_call_id":"call_1"' in body.replace(" ", "")

    @pytest.mark.asyncio
    async def test_stream_returns_503_when_no_provider_produces_any_output(self, client: AsyncClient):
        """Total failure before anything has streamed must still surface as a clean 503, same
        as the non-streaming path — the router pulls the first event outside the SSE generator
        specifically so this is still possible (see chat.py's peek-before-StreamingResponse)."""
        with patch.object(ChatService, "chat_stream", _failing_chat_stream):
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_stream_reports_failure_that_happens_mid_stream(self, client: AsyncClient):
        """Once output has already streamed, a provider failure can't retry silently (see
        ResilientLLMService.stream_complete) — it must show up inline instead of just cutting
        the connection, and the stream must still end cleanly with [DONE]."""
        events = [TextDelta(text="partial answer"), StreamFailed(message="boom")]
        with patch.object(ChatService, "chat_stream", _fake_chat_stream(events)):
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )
            assert resp.status_code == 200
            body = resp.text
            assert "partial answer" in body
            assert "interrupted" in body
            assert "data: [DONE]" in body
