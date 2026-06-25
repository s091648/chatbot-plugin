"""Router tests — validate /v1/chat/completions OpenAI-compatible endpoint."""

import json
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from chatbot_plugin.services.chat_service import ChatService, ChatResult, ArticleRef


def _ok_result(reply="RAG is a retrieval technique..."):
    return ChatResult(reply=reply, articles_used=[], chunks=[])


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
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResult(reply="Hello world", articles_used=[], chunks=[])
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
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResult(
                reply="Answer", articles_used=[], chunks=[], thinking="Chain of thought..."
            )
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
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResult(
                reply="Answer",
                articles_used=[
                    ArticleRef(
                        id="vec-id", title="My Article",
                        url="https://example.com", public_article_id="pub-uuid",
                    )
                ],
                chunks=[],
            )
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

    @pytest.mark.asyncio
    async def test_stream_omits_sources_event_when_no_articles(self, client: AsyncClient):
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResult(reply="Hi", articles_used=[], chunks=[])
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
        with patch.object(ChatService, "chat", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResult(reply="Hi", articles_used=[], chunks=[], thinking=None)
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )
            has_thinking = any(
                l.startswith("data: ") and "thinking" in l for l in resp.text.splitlines()
            )
            assert not has_thinking
