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


# ── Missing branch tests ──


@pytest.mark.asyncio
async def test_chat_generic_exception_raises_503(service, mock_db, mock_llm_service):
    """Non-RuntimeError exceptions from rag_generate also produce 503."""
    mock_db.execute.return_value = _mock_result(rows=[])
    mock_llm_service.generate.side_effect = Exception("unexpected error")

    with pytest.raises(HTTPException) as exc_info:
        await service.chat("hello")
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_chat_untitled_fallback_for_none_title(service, mock_db, mock_llm_service):
    """Articles with None/empty title get 'Untitled' fallback."""
    mock_db.execute.return_value = _mock_result(
        rows=[{"id": "uuid-1", "title": None, "content": "Some content", "rank": 0.5}]
    )
    mock_llm_service.generate.return_value = "Reply"

    result = await service.chat("hello")
    assert result.articles_used[0].title == "Untitled"


@pytest.mark.asyncio
async def test_search_with_topic_id(service, mock_db, mock_llm_service):
    """search() with topic_id passes it through to _search_articles."""
    mock_db.execute.return_value = _mock_result(
        rows=[{"id": "uuid-1", "title": "Article A", "content": "Content A", "rank": 0.8}]
    )

    result = await service.search("RAG", topic_id="topic-uuid")
    assert len(result.chunks) == 1
    # Verify the SQL was executed (topic_id passed to params)
    mock_db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_search_untitled_fallback(service, mock_db, mock_llm_service):
    """Search results with None/empty title get 'Untitled' fallback."""
    mock_db.execute.return_value = _mock_result(
        rows=[{"id": "uuid-1", "title": "", "content": "Content", "rank": 0.5}]
    )

    result = await service.search("test")
    assert result.chunks[0].article_title == "Untitled"


@pytest.mark.asyncio
async def test_search_empty_content_fallback(service, mock_db, mock_llm_service):
    """Search results with None content get empty string fallback."""
    mock_db.execute.return_value = _mock_result(
        rows=[{"id": "uuid-1", "title": "Title", "content": None, "rank": 0.5}]
    )

    result = await service.search("test")
    assert result.chunks[0].content == ""


@pytest.mark.asyncio
async def test_trigger_index_with_article_found(service, mock_db, mock_llm_service):
    """trigger_index with article_id where the article exists."""
    mock_db.execute.return_value = _mock_result(scalar_val="uuid-1")

    result = await service.trigger_index(article_id="uuid-1")
    assert result.status == "started"
    assert result.job_id


@pytest.mark.asyncio
async def test_get_status_with_none_scalar(service, mock_db, mock_llm_service):
    """get_status when count(*) returns None — should default to 0."""
    mock_db.execute.return_value = _mock_result(scalar_val=None)

    result = await service.get_status()
    assert result.pending_articles == 0
