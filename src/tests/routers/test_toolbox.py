"""Router tests — validate API endpoints match specs/toolbox-api.md.

These tests verify the API contract: endpoint paths, status codes,
and response shapes. Business logic is tested in service tests.
"""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from chatbot_plugin.contracts import (
    ChatResponse,
    SearchResponse,
    StoreChunksResponse,
)
from chatbot_plugin.service import ToolboxService


# ── POST /tools/chunks ──


class TestStoreChunks:
    @pytest.mark.asyncio
    async def test_accepts_valid_payload(self, client: AsyncClient):
        with patch.object(
            ToolboxService,
            "store_chunks",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = StoreChunksResponse(stored=1, article_id="uuid-1")
            resp = await client.post(
                "/tools/chunks",
                json={
                    "article": {
                        "id": "uuid-1",
                        "url": "https://example.com",
                    },
                    "chunks": [
                        {
                            "chunk_index": 0,
                            "content": "Text chunk",
                            "dense_vector": [0.1] * 1024,
                        }
                    ],
                },
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["stored"] == 1
            assert data["article_id"] == "uuid-1"

    @pytest.mark.asyncio
    async def test_rejects_empty_chunks(self, client: AsyncClient):
        resp = await client.post(
            "/tools/chunks",
            json={
                "article": {
                    "id": "uuid-1",
                    "url": "https://example.com",
                },
                "chunks": [],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_missing_article_id(self, client: AsyncClient):
        resp = await client.post(
            "/tools/chunks",
            json={
                "article": {
                    "url": "https://example.com",
                },
                "chunks": [
                    {
                        "chunk_index": 0,
                        "content": "Text",
                        "dense_vector": [0.1] * 1024,
                    }
                ],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_missing_chunks(self, client: AsyncClient):
        resp = await client.post(
            "/tools/chunks",
            json={
                "article": {
                    "id": "uuid-1",
                    "url": "https://example.com",
                },
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_wrong_dense_vector_dim(self, client: AsyncClient):
        resp = await client.post(
            "/tools/chunks",
            json={
                "article": {
                    "id": "uuid-1",
                    "url": "https://example.com",
                },
                "chunks": [
                    {
                        "chunk_index": 0,
                        "content": "Text",
                        "dense_vector": [0.1] * 512,  # wrong dimension
                    }
                ],
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_accepts_sparse_vector(self, client: AsyncClient):
        with patch.object(
            ToolboxService,
            "store_chunks",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = StoreChunksResponse(stored=1, article_id="uuid-1")
            resp = await client.post(
                "/tools/chunks",
                json={
                    "article": {
                        "id": "uuid-1",
                        "url": "https://example.com",
                    },
                    "chunks": [
                        {
                            "chunk_index": 0,
                            "content": "Text",
                            "dense_vector": [0.1] * 1024,
                            "sparse_vector": {"0": 0.5},
                        }
                    ],
                },
            )
            assert resp.status_code == 201


# ── POST /tools/search ──


class TestSearch:
    @pytest.mark.asyncio
    async def test_accepts_valid_payload(self, client: AsyncClient):
        with patch.object(
            ToolboxService,
            "search",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = SearchResponse(
                chunks=[
                    {
                        "chunk_id": "c1",
                        "article_id": "a1",
                        "article_title": "Title",
                        "article_url": "https://example.com",
                        "chunk_index": 0,
                        "content": "Hello",
                        "score": 0.9,
                    }
                ]
            )
            resp = await client.post(
                "/tools/search",
                json={"query": "What is RAG?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["chunks"]) == 1
            assert data["chunks"][0]["chunk_id"] == "c1"

    @pytest.mark.asyncio
    async def test_accepts_custom_top_k(self, client: AsyncClient):
        with patch.object(
            ToolboxService,
            "search",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = SearchResponse()
            resp = await client.post(
                "/tools/search",
                json={"query": "test", "top_k": 5},
            )
            assert resp.status_code == 200
            mock.assert_called_once()
            call_args = mock.call_args[0][0]
            assert call_args.top_k == 5

    @pytest.mark.asyncio
    async def test_rejects_empty_query(self, client: AsyncClient):
        resp = await client.post(
            "/tools/search",
            json={"query": ""},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_top_k_below_min(self, client: AsyncClient):
        resp = await client.post(
            "/tools/search",
            json={"query": "test", "top_k": 0},
        )
        assert resp.status_code == 422


# ── POST /tools/chat ──


class TestChat:
    @pytest.mark.asyncio
    async def test_accepts_valid_payload(self, client: AsyncClient):
        with patch.object(
            ToolboxService,
            "chat",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = ChatResponse(
                reply="RAG is a technique...",
                articles_used=[
                    {"id": "a1", "title": "Title", "url": "https://example.com"}
                ],
                chunks=[
                    {
                        "chunk_id": "c1",
                        "article_id": "a1",
                        "chunk_index": 0,
                        "content": "RAG is...",
                        "score": 0.8,
                    }
                ],
            )
            resp = await client.post(
                "/tools/chat",
                json={"message": "Explain RAG"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["reply"] == "RAG is a technique..."
            assert len(data["articles_used"]) == 1
            assert len(data["chunks"]) == 1

    @pytest.mark.asyncio
    async def test_rejects_empty_message(self, client: AsyncClient):
        resp = await client.post(
            "/tools/chat",
            json={"message": ""},
        )
        assert resp.status_code == 422
