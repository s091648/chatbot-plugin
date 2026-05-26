"""Tests for Retriever — full-text search via search_tsv + GIN index."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from chatbot_plugin.rag.retriever import Retriever


def _mock_result(rows: list[dict] | None = None):
    """Build a mock DB result that supports .mappings().all()."""
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows or []
    return result


@pytest.mark.asyncio
async def test_search_uses_search_tsv_column():
    """Retriever must use search_tsv column, not on-the-fly to_tsvector."""
    mock_db = AsyncMock()
    mock_db.execute.return_value = _mock_result([])

    retriever = Retriever(mock_db)
    await retriever.search("test query")

    sql_text = str(mock_db.execute.call_args[0][0])
    # Must use search_tsv directly (enables GIN index)
    assert "search_tsv @@" in sql_text
    assert "ts_rank(search_tsv" in sql_text
    # Must NOT compute to_tsvector in WHERE clause
    where_clause = sql_text.split("WHERE")[1].split("ORDER BY")[0]
    assert "to_tsvector" not in where_clause


@pytest.mark.asyncio
async def test_search_returns_correct_shape():
    mock_db = AsyncMock()
    mock_db.execute.return_value = _mock_result(
        rows=[
            {"id": "uuid-1", "title": "Article A", "content": "Content A", "rank": 0.8},
            {"id": "uuid-2", "title": "Article B", "content": "Content B", "rank": 0.5},
        ]
    )

    retriever = Retriever(mock_db)
    results = await retriever.search("RAG")

    assert len(results) == 2
    assert results[0]["id"] == "uuid-1"
    assert results[0]["rank"] == 0.8


@pytest.mark.asyncio
async def test_search_with_topic_id():
    mock_db = AsyncMock()
    mock_db.execute.return_value = _mock_result([])

    retriever = Retriever(mock_db)
    await retriever.search("RAG", topic_id="topic-uuid")

    sql_text = str(mock_db.execute.call_args[0][0])
    assert "topic_id" in sql_text
    params = mock_db.execute.call_args[0][1]
    assert params["topic_id"] == "topic-uuid"


@pytest.mark.asyncio
async def test_search_no_topic_id():
    mock_db = AsyncMock()
    mock_db.execute.return_value = _mock_result([])

    retriever = Retriever(mock_db)
    await retriever.search("RAG")

    sql_text = str(mock_db.execute.call_args[0][0])
    assert "topic_id" not in sql_text


@pytest.mark.asyncio
async def test_search_no_results():
    mock_db = AsyncMock()
    mock_db.execute.return_value = _mock_result([])

    retriever = Retriever(mock_db)
    results = await retriever.search("nonexistent")
    assert results == []


@pytest.mark.asyncio
async def test_search_uses_plainto_tsquery():
    mock_db = AsyncMock()
    mock_db.execute.return_value = _mock_result([])

    retriever = Retriever(mock_db)
    await retriever.search("test query")

    sql_text = str(mock_db.execute.call_args[0][0])
    assert "plainto_tsquery" in sql_text
