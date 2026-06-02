"""Router tests — validate API endpoints match specs/toolbox-api.md.

These tests verify the API contract: endpoint paths, status codes,
and response shapes. Business logic is tested in service tests.
"""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from chatbot_plugin.contracts import StoreChunksResponse
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
