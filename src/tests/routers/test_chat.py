"""Router tests — validate API endpoints match specs/chat-api.md.

These tests verify the API contract: endpoint paths, status codes,
and response shapes. Business logic is tested in service tests.
"""

import pytest
from httpx import AsyncClient


# ── POST /chat/message ──

class TestChatMessage:
    @pytest.mark.asyncio
    async def test_message_accepts_valid_body(self, client: AsyncClient):
        resp = await client.post("/chat/message", json={"message": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "articles_used" in data

    @pytest.mark.asyncio
    async def test_message_with_user_id(self, client: AsyncClient):
        resp = await client.post(
            "/chat/message", json={"message": "hello", "user_id": "u1"}
        )
        assert resp.status_code == 200

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
        resp = await client.post("/chat/search", json={"query": "RAG"})
        assert resp.status_code == 200
        data = resp.json()
        assert "chunks" in data

    @pytest.mark.asyncio
    async def test_search_with_top_k(self, client: AsyncClient):
        resp = await client.post("/chat/search", json={"query": "RAG", "top_k": 5})
        assert resp.status_code == 200

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
        resp = await client.post("/chat/index", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "started"

    @pytest.mark.asyncio
    async def test_index_with_article_id(self, client: AsyncClient):
        resp = await client.post(
            "/chat/index",
            json={"article_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab"},
        )
        assert resp.status_code == 202


# ── GET /chat/status ──

class TestChatStatus:
    @pytest.mark.asyncio
    async def test_status_returns_shape(self, client: AsyncClient):
        resp = await client.get("/chat/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_chunks" in data
        assert "last_indexed_at" in data
        assert "pending_articles" in data
