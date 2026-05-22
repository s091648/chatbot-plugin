"""Contract tests — validate Pydantic models match specs/chat-api.md.

These tests ensure:
1. Request models accept valid input and reject invalid input per spec
2. Response models produce JSON that matches the spec shape
3. Field constraints (min_length, max_length, ge, le) align with spec
"""

import pytest
from pydantic import ValidationError

from chatbot_plugin.contracts.requests import (
    ChatMessageRequest,
    SearchRequest,
    IndexRequest,
)
from chatbot_plugin.contracts.responses import (
    ChatMessageResponse,
    ArticleRef,
    SearchResponse,
    ChunkResult,
    IndexResponse,
    StatusResponse,
)


# ── ChatMessageRequest ──

class TestChatMessageRequest:
    def test_valid_minimal(self):
        req = ChatMessageRequest(message="hello")
        assert req.message == "hello"
        assert req.user_id is None

    def test_valid_with_user_id(self):
        req = ChatMessageRequest(message="hello", user_id="user-1")
        assert req.user_id == "user-1"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessageRequest(message="")

    def test_message_exceeds_max_length(self):
        with pytest.raises(ValidationError):
            ChatMessageRequest(message="x" * 2001)

    def test_message_at_max_length(self):
        req = ChatMessageRequest(message="x" * 2000)
        assert len(req.message) == 2000


# ── SearchRequest ──

class TestSearchRequest:
    def test_valid_defaults(self):
        req = SearchRequest(query="test")
        assert req.top_k == 10
        assert req.topic_id is None

    def test_top_k_bounds(self):
        SearchRequest(query="test", top_k=1)
        SearchRequest(query="test", top_k=50)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=51)

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_query_exceeds_max_length(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 501)


# ── IndexRequest ──

class TestIndexRequest:
    def test_valid_empty(self):
        req = IndexRequest()
        assert req.article_id is None

    def test_valid_with_article_id(self):
        req = IndexRequest(article_id="a1b2c3d4-5678-90ab-cdef-1234567890ab")
        assert req.article_id is not None


# ── ChatMessageResponse ──

class TestChatMessageResponse:
    def test_valid_with_articles(self):
        resp = ChatMessageResponse(
            reply="Based on articles...",
            articles_used=[
                ArticleRef(id="uuid-1", title="Article One"),
                ArticleRef(id="uuid-2", title="Article Two"),
            ],
        )
        assert len(resp.articles_used) == 2

    def test_no_articles_empty_list(self):
        resp = ChatMessageResponse(reply="No relevant articles found.", articles_used=[])
        assert resp.articles_used == []

    def test_json_matches_spec(self):
        resp = ChatMessageResponse(
            reply="test reply",
            articles_used=[ArticleRef(id="uuid-1", title="T1")],
        )
        data = resp.model_dump()
        assert "reply" in data
        assert "articles_used" in data
        assert data["articles_used"][0]["id"] == "uuid-1"
        assert data["articles_used"][0]["title"] == "T1"


# ── SearchResponse ──

class TestSearchResponse:
    def test_valid_with_chunks(self):
        resp = SearchResponse(
            chunks=[
                ChunkResult(
                    content="chunk text",
                    article_id="uuid-1",
                    article_title="Article One",
                    score=0.87,
                )
            ]
        )
        assert len(resp.chunks) == 1
        assert resp.chunks[0].score == 0.87

    def test_no_results_empty_list(self):
        resp = SearchResponse(chunks=[])
        assert resp.chunks == []

    def test_score_bounds(self):
        ChunkResult(content="t", article_id="u", article_title="T", score=0.0)
        ChunkResult(content="t", article_id="u", article_title="T", score=1.0)
        with pytest.raises(ValidationError):
            ChunkResult(content="t", article_id="u", article_title="T", score=1.1)
        with pytest.raises(ValidationError):
            ChunkResult(content="t", article_id="u", article_title="T", score=-0.1)


# ── IndexResponse ──

class TestIndexResponse:
    def test_valid(self):
        resp = IndexResponse(job_id="job-uuid-1")
        assert resp.status == "started"

    def test_json_matches_spec(self):
        resp = IndexResponse(job_id="job-uuid-1")
        data = resp.model_dump()
        assert data == {"job_id": "job-uuid-1", "status": "started"}


# ── StatusResponse ──

class TestStatusResponse:
    def test_valid_never_indexed(self):
        resp = StatusResponse(total_chunks=0, last_indexed_at=None, pending_articles=5)
        assert resp.last_indexed_at is None

    def test_valid_with_timestamp(self):
        resp = StatusResponse(
            total_chunks=1523,
            last_indexed_at="2026-05-20T10:00:00Z",
            pending_articles=5,
        )
        assert resp.total_chunks == 1523
