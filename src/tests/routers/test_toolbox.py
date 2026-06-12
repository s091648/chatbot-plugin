"""Router tests — validate /v1/chat/completions OpenAI-compatible endpoint."""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from chatbot_plugin_sdk import RagQueryProcessor


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_accepts_valid_messages(self, client: AsyncClient):
        with patch.object(
            RagQueryProcessor,
            "chat",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = type(
                "ChatResponse",
                (),
                {
                    "reply": "RAG is a retrieval technique...",
                    "articles_used": [],
                    "chunks": [],
                },
            )()
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
            mock.assert_called_once_with("What is RAG?")

    @pytest.mark.asyncio
    async def test_rejects_empty_messages(self, client: AsyncClient):
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_no_messages_field(self, client: AsyncClient):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_default_model(self, client: AsyncClient):
        with patch.object(
            RagQueryProcessor,
            "chat",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = type(
                "ChatResponse",
                (),
                {
                    "reply": "Hello",
                    "articles_used": [],
                    "chunks": [],
                },
            )()
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "Hi"},
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["model"] == "rag-default"
            mock.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_only_system_message_returns_400(self, client: AsyncClient):
        """No user message → 400 because there's nothing to query."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                ],
            },
        )
        assert resp.status_code == 400
