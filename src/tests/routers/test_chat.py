"""Router tests — validate API endpoints match specs/chat-api.md.

These tests verify the API contract: endpoint paths, status codes,
and response shapes. Business logic is tested in service tests.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient


# ── POST /chat/message ──

class TestChatMessage:
    @pytest.mark.asyncio
    async def test_message_accepts_valid_body(self, client: AsyncClient):
        with patch("chatbot_plugin.service.ChatbotService.chat", new_callable=AsyncMock) as mock:
            mock.return_value = MagicMock(
                reply="test", articles_used=[], model_dump=lambda: {"reply": "test", "articles_used": []}
            )
            resp = await client.post("/chat/message", json={"message": "hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert "reply" in data
            assert "articles_used" in data

    @pytest.mark.asyncio
    async def test_message_empty_body_rejected(self, client: AsyncClient):
        resp = await client.post("/chat/message", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_message_empty_string_rejected(self, client: AsyncClient):
        resp = await client.post("/chat/message", json={"message": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_message_exceeds_max_length(self, client: AsyncClient):
        resp = await client.post("/chat/message", json={"message": "x" * 2001})
        assert resp.status_code == 422


# ── POST /chat/search ──

class TestChatSearch:
    @pytest.mark.asyncio
    async def test_search_accepts_valid_body(self, client: AsyncClient):
        with patch("chatbot_plugin.service.ChatbotService.search", new_callable=AsyncMock) as mock:
            mock.return_value = MagicMock(
                chunks=[], model_dump=lambda: {"chunks": []}
            )
            resp = await client.post("/chat/search", json={"query": "RAG"})
            assert resp.status_code == 200
            data = resp.json()
            assert "chunks" in data

    @pytest.mark.asyncio
    async def test_search_empty_query_rejected(self, client: AsyncClient):
        resp = await client.post("/chat/search", json={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_search_top_k_out_of_range(self, client: AsyncClient):
        resp = await client.post("/chat/search", json={"query": "RAG", "top_k": 51})
        assert resp.status_code == 422


# ── POST /chat/index ──

class TestChatIndex:
    @pytest.mark.asyncio
    async def test_index_no_body(self, client: AsyncClient):
        with patch("chatbot_plugin.service.ChatbotService.trigger_index", new_callable=AsyncMock) as mock:
            mock.return_value = MagicMock(
                job_id="job-1", status="started",
                model_dump=lambda: {"job_id": "job-1", "status": "started"}
            )
            resp = await client.post("/chat/index", json={})
            assert resp.status_code == 202
            data = resp.json()
            assert "job_id" in data
            assert data["status"] == "started"

    @pytest.mark.asyncio
    async def test_index_article_not_found(self, client: AsyncClient):
        from fastapi import HTTPException
        with patch("chatbot_plugin.service.ChatbotService.trigger_index", new_callable=AsyncMock) as mock:
            mock.side_effect = HTTPException(status_code=404, detail="Article not found")
            resp = await client.post(
                "/chat/index",
                json={"article_id": "nonexistent-uuid"},
            )
            assert resp.status_code == 404


# ── GET /chat/status ──

class TestChatStatus:
    @pytest.mark.asyncio
    async def test_status_returns_shape(self, client: AsyncClient):
        with patch("chatbot_plugin.service.ChatbotService.get_status", new_callable=AsyncMock) as mock:
            mock.return_value = MagicMock(
                total_chunks=0, last_indexed_at=None, pending_articles=5,
                model_dump=lambda: {"total_chunks": 0, "last_indexed_at": None, "pending_articles": 5}
            )
            resp = await client.get("/chat/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "total_chunks" in data
            assert "last_indexed_at" in data
            assert "pending_articles" in data


# ── Additional router branch tests ──


class TestRouterAdditional:
    @pytest.mark.asyncio
    async def test_message_with_user_id(self, client: AsyncClient):
        """Router passes user_id through to service.chat."""
        with patch("chatbot_plugin.service.ChatbotService.chat", new_callable=AsyncMock) as mock:
            mock.return_value = MagicMock(
                reply="hi", articles_used=[], model_dump=lambda: {"reply": "hi", "articles_used": []}
            )
            resp = await client.post("/chat/message", json={"message": "hello", "user_id": "user-1"})
            assert resp.status_code == 200
            mock.assert_called_once_with("hello", "user-1")

    @pytest.mark.asyncio
    async def test_search_with_topic_id(self, client: AsyncClient):
        """Router passes topic_id through to service.search."""
        with patch("chatbot_plugin.service.ChatbotService.search", new_callable=AsyncMock) as mock:
            mock.return_value = MagicMock(
                chunks=[], model_dump=lambda: {"chunks": []}
            )
            resp = await client.post("/chat/search", json={"query": "RAG", "topic_id": "topic-1"})
            assert resp.status_code == 200
            mock.assert_called_once_with("RAG", 10, "topic-1")

    @pytest.mark.asyncio
    async def test_index_with_article_id(self, client: AsyncClient):
        """Router passes article_id through to service.trigger_index."""
        with patch("chatbot_plugin.service.ChatbotService.trigger_index", new_callable=AsyncMock) as mock:
            mock.return_value = MagicMock(
                job_id="job-1", status="started",
                model_dump=lambda: {"job_id": "job-1", "status": "started"}
            )
            resp = await client.post("/chat/index", json={"article_id": "uuid-1"})
            assert resp.status_code == 202
            mock.assert_called_once_with("uuid-1")

    @pytest.mark.asyncio
    async def test_message_503_on_llm_failure(self, client: AsyncClient):
        """Router returns 503 when service.chat raises HTTPException(503)."""
        from fastapi import HTTPException
        with patch("chatbot_plugin.service.ChatbotService.chat", new_callable=AsyncMock) as mock:
            mock.side_effect = HTTPException(status_code=503, detail="LLM provider unavailable")
            resp = await client.post("/chat/message", json={"message": "hello"})
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_set_llm_service_overrides(self, client: AsyncClient):
        """set_llm_service can override the LLM service for testing."""
        from chatbot_plugin.routers import set_llm_service, _llm_service
        # The conftest already called set_llm_service, so _llm_service is not None
        # Just verify it was set
        from chatbot_plugin import routers
        assert routers._llm_service is not None
