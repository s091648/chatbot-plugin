"""Tests for ToolboxService."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from chatbot_plugin.config import settings
from chatbot_plugin.contracts import (
    ArticleInfo,
    ChunkData,
    StoreChunksRequest,
    StoreChunksResponse,
)
from chatbot_plugin.models import Article, ArticleChunk
from chatbot_plugin.service import ToolboxService


@pytest.fixture
def mock_db() -> AsyncMock:
    """Mock async DB session."""
    db = AsyncMock()
    db.add = MagicMock()  # not async in SQLAlchemy
    return db


@pytest.fixture
def service(mock_db: AsyncMock) -> ToolboxService:
    """ToolboxService with mocked DB."""
    return ToolboxService(mock_db)


# ── store_chunks ──


@pytest.mark.asyncio
async def test_store_chunks_creates_new_article(service: ToolboxService, mock_db: AsyncMock):
    """When article doesn't exist, a new one is created."""
    # simulate no existing article
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = result

    req = StoreChunksRequest(
        article=ArticleInfo(id=str(uuid.uuid4()), url="https://example.com", title="T"),
        chunks=[ChunkData(chunk_index=0, content="hello", dense_vector=[0.1] * 1024)],
    )

    resp = await service.store_chunks(req)

    assert isinstance(resp, StoreChunksResponse)
    assert resp.stored == 1
    assert resp.article_id == req.article.id
    mock_db.add.assert_called()
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_store_chunks_updates_existing_article(service: ToolboxService, mock_db: AsyncMock):
    """When article exists, metadata is updated and old chunks deleted."""
    existing = MagicMock(spec=Article)
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    mock_db.execute.return_value = result

    req = StoreChunksRequest(
        article=ArticleInfo(id=str(uuid.uuid4()), url="https://example.com", title="New Title"),
        chunks=[ChunkData(chunk_index=0, content="hello", dense_vector=[0.1] * 1024)],
    )

    resp = await service.store_chunks(req)

    assert isinstance(resp, StoreChunksResponse)
    assert resp.stored == 1
    assert existing.url == "https://example.com"
    assert existing.title == "New Title"
    # Should have executed delete statement plus the select
    assert mock_db.execute.call_count >= 1
    mock_db.commit.assert_called_once()


def test_store_chunks_rejects_empty_chunks():
    """Empty chunks list is rejected at the contract layer by Pydantic min_length=1."""
    with pytest.raises(ValidationError):
        StoreChunksRequest(
            article=ArticleInfo(id=str(uuid.uuid4()), url="https://example.com"),
            chunks=[],
        )


@pytest.mark.asyncio
async def test_store_chunks_rejects_wrong_vector_dimension(service: ToolboxService):
    req = StoreChunksRequest(
        article=ArticleInfo(id=str(uuid.uuid4()), url="https://example.com"),
        chunks=[ChunkData(chunk_index=0, content="hello", dense_vector=[0.1] * 512)],
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.store_chunks(req)
    assert exc_info.value.status_code == 400
    assert "mismatch" in exc_info.value.detail.lower()
    assert str(settings.embedding_dimension) in exc_info.value.detail


@pytest.mark.asyncio
async def test_store_chunks_accepts_sparse_vector(service: ToolboxService, mock_db: AsyncMock):
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = result

    req = StoreChunksRequest(
        article=ArticleInfo(id=str(uuid.uuid4()), url="https://example.com"),
        chunks=[
            ChunkData(
                chunk_index=0,
                content="hello",
                dense_vector=[0.1] * 1024,
                sparse_vector={"0": 0.5, "1": 0.3},
            )
        ],
    )

    resp = await service.store_chunks(req)
    assert resp.stored == 1


@pytest.mark.asyncio
async def test_store_chunks_multiple_chunks(service: ToolboxService, mock_db: AsyncMock):
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = result

    req = StoreChunksRequest(
        article=ArticleInfo(id=str(uuid.uuid4()), url="https://example.com"),
        chunks=[
            ChunkData(chunk_index=0, content="chunk 0", dense_vector=[0.1] * 1024),
            ChunkData(chunk_index=1, content="chunk 1", dense_vector=[0.2] * 1024),
        ],
    )

    resp = await service.store_chunks(req)
    assert resp.stored == 2
    assert mock_db.add.call_count == 3  # 1 article + 2 chunks
