"""Tests for ToolboxService."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from chatbot_plugin.config import settings
from chatbot_plugin.contracts import (
    ArticleInfo,
    ChatRequest,
    ChunkData,
    SearchRequest,
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


# ── search ──


@pytest.fixture
def make_search_row():
    """Factory to create mock rows from search query results."""

    def _make(chunk_id, article_id, chunk_index, content, title, url):
        Row = type(
            "MockRow",
            (),
            {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "chunk_index": chunk_index,
                "content": content,
                "title": title,
                "url": url,
            },
        )
        return Row()

    return _make


@pytest.mark.asyncio
@patch("chatbot_plugin.embedding.embed_query")
async def test_search_fuses_dense_and_sparse(mock_embed, service, mock_db, make_search_row):
    """Dense and sparse candidates are RRF-fused and top_k returned."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    # Dense returns chunk-a, chunk-b; Sparse returns chunk-b, chunk-c
    row_a = make_search_row("chunk-a", "art-1", 0, "text a", "Title A", "https://a.com")
    row_b = make_search_row("chunk-b", "art-1", 1, "text b", "Title A", "https://a.com")
    row_c = make_search_row("chunk-c", "art-2", 0, "text c", "Title C", "https://c.com")

    # Two sequential execute() calls: dense then sparse
    dense_result = MagicMock()
    dense_result.all.return_value = [row_a, row_b]
    sparse_result = MagicMock()
    sparse_result.all.return_value = [row_b, row_c]

    mock_db.execute.side_effect = [dense_result, sparse_result]

    req = SearchRequest(query="hello", top_k=2)
    resp = await service.search(req)

    assert len(resp.chunks) == 2
    # chunk-b appears in both lists → highest fused score
    assert resp.chunks[0].chunk_id == "chunk-b"
    mock_embed.assert_called_once_with("hello")


@pytest.mark.asyncio
@patch("chatbot_plugin.embedding.embed_query")
async def test_search_only_dense_candidates(mock_embed, service, mock_db, make_search_row):
    """When sparse has no results, dense-only scores still work."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    row_a = make_search_row("chunk-a", "art-1", 0, "text a", "Title A", "https://a.com")

    dense_result = MagicMock()
    dense_result.all.return_value = [row_a]
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_db.execute.side_effect = [dense_result, sparse_result]

    resp = await service.search(SearchRequest(query="hello"))
    assert len(resp.chunks) == 1
    assert resp.chunks[0].chunk_id == "chunk-a"


@pytest.mark.asyncio
@patch("chatbot_plugin.embedding.embed_query")
async def test_search_respects_top_k(mock_embed, service, mock_db, make_search_row):
    """top_k limits the number of results returned."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    rows = [
        make_search_row(f"chunk-{i}", "art-1", i, f"text {i}", "T", "https://x.com")
        for i in range(5)
    ]

    dense_result = MagicMock()
    dense_result.all.return_value = rows
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_db.execute.side_effect = [dense_result, sparse_result]

    resp = await service.search(SearchRequest(query="hello", top_k=3))
    assert len(resp.chunks) == 3


# ── chat ──


@pytest.mark.asyncio
@patch.object(ToolboxService, "_call_llm")
@patch("chatbot_plugin.embedding.embed_query")
async def test_chat_returns_reply(mock_embed, mock_llm, service, mock_db, make_search_row):
    """Chat searches for context, calls LLM, and returns reply + citations."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})
    mock_llm.return_value = "Yes, RAG is..."

    row_a = make_search_row("chunk-a", "art-1", 0, "RAG is retrieval...", "RAG Article", "https://rag.com")
    dense_result = MagicMock()
    dense_result.all.return_value = [row_a]
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_db.execute.side_effect = [dense_result, sparse_result]

    req = ChatRequest(message="What is RAG?")
    resp = await service.chat(req)

    assert resp.reply == "Yes, RAG is..."
    assert len(resp.articles_used) == 1
    assert resp.articles_used[0].id == "art-1"
    assert resp.articles_used[0].title == "RAG Article"
    assert len(resp.chunks) == 1
    mock_llm.assert_called_once()


@pytest.mark.asyncio
@patch("chatbot_plugin.embedding.embed_query")
async def test_chat_no_results(mock_embed, service, mock_db):
    """When search returns no chunks, chat responds with a fallback message."""
    mock_embed.return_value = ([0.1] * 1024, {1: 0.5})

    dense_result = MagicMock()
    dense_result.all.return_value = []
    sparse_result = MagicMock()
    sparse_result.all.return_value = []

    mock_db.execute.side_effect = [dense_result, sparse_result]

    req = ChatRequest(message="What is RAG?")
    resp = await service.chat(req)

    assert "couldn't find any relevant context" in resp.reply
    assert resp.articles_used == []
    assert resp.chunks == []