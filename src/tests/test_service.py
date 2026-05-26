"""Tests for ChatbotService."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException

from chatbot_plugin.service import ChatbotService
from chatbot_plugin.contracts import ChatMessageResponse, SearchResponse


def _mock_result(rows: list[dict] | None = None, scalar_val=None):
    """Build a mock DB result that supports .mappings().all() and .scalar()."""
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows or []
    result.scalar.return_value = scalar_val
    return result


# ── chat() ──

@pytest.mark.asyncio
async def test_chat_returns_reply_and_articles(service, mock_db, mock_llm_service):
    mock_db.execute.return_value = _mock_result(
        rows=[{"id": "uuid-1", "title": "RAG Article", "content": "RAG content...", "rank": 0.5}]
    )
    mock_llm_service.generate.return_value = "RAG is retrieval-augmented generation."

    result = await service.chat("What is RAG?")

    assert isinstance(result, ChatMessageResponse)
    assert "RAG" in result.reply
    assert len(result.articles_used) == 1
    assert result.articles_used[0].title == "RAG Article"


@pytest.mark.asyncio
async def test_chat_no_articles_still_generates(service, mock_db, mock_llm_service):
    mock_db.execute.return_value = _mock_result(rows=[])
    mock_llm_service.generate.return_value = "I don't have specific articles on that."

    result = await service.chat("obscure topic")

    assert result.articles_used == []


@pytest.mark.asyncio
async def test_chat_llm_failure_raises_503(service, mock_db, mock_llm_service):
    mock_db.execute.return_value = _mock_result(rows=[])
    mock_llm_service.generate.return_value = None  # All providers failed

    with pytest.raises(HTTPException) as exc_info:
        await service.chat("hello")
    assert exc_info.value.status_code == 503


# ── search() ──

@pytest.mark.asyncio
async def test_search_returns_chunks(service, mock_db, mock_llm_service):
    mock_db.execute.return_value = _mock_result(
        rows=[
            {"id": "uuid-1", "title": "Article A", "content": "Content A", "rank": 0.8},
            {"id": "uuid-2", "title": "Article B", "content": "Content B", "rank": 0.5},
        ]
    )

    result = await service.search("RAG", top_k=10)
    assert isinstance(result, SearchResponse)
    assert len(result.chunks) == 2
    assert result.chunks[0].score == 0.8


@pytest.mark.asyncio
async def test_search_no_results(service, mock_db, mock_llm_service):
    mock_db.execute.return_value = _mock_result(rows=[])

    result = await service.search("nonexistent")
    assert result.chunks == []


# ── trigger_index() ──

@pytest.mark.asyncio
async def test_trigger_index_returns_202(service, mock_db, mock_llm_service):
    result = await service.trigger_index()
    assert result.status == "started"
    assert result.job_id


@pytest.mark.asyncio
async def test_trigger_index_article_not_found_raises_404(service, mock_db, mock_llm_service):
    mock_db.execute.return_value = _mock_result(scalar_val=None)

    with pytest.raises(HTTPException) as exc_info:
        await service.trigger_index(article_id="nonexistent-uuid")
    assert exc_info.value.status_code == 404


# ── get_status() ──

@pytest.mark.asyncio
async def test_get_status_returns_shape(service, mock_db, mock_llm_service):
    mock_db.execute.return_value = _mock_result(scalar_val=42)

    result = await service.get_status()
    assert result.pending_articles == 42
    assert result.total_chunks == 0
    assert result.last_indexed_at is None
