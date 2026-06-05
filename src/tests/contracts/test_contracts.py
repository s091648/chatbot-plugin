"""Contract tests — validate Pydantic models match specs/toolbox-api.md.

These tests ensure:
1. Request models accept valid input and reject invalid input per spec
2. Response models produce JSON that matches the spec shape
3. Field constraints (min_length, ge, etc.) align with spec
"""

import pytest
from pydantic import ValidationError

from chatbot_plugin.contracts.requests import (
    ArticleInfo,
    ChunkData,
    StoreChunksRequest,
    SearchRequest,
    ChatRequest,
)
from chatbot_plugin.contracts.responses import (
    StoreChunksResponse,
    ChunkResult,
    SearchResponse,
    ArticleCitation,
    ChatResponse,
)


# ── ArticleInfo ──


class TestArticleInfo:
    def test_valid(self):
        info = ArticleInfo(
            id="a1b2c3d4-5678-90ab-cdef-1234567890ab",
            url="https://example.com",
            title="Title",
            source="example.com",
            metadata={"author": "Alice"},
        )
        assert info.id == "a1b2c3d4-5678-90ab-cdef-1234567890ab"
        assert info.url == "https://example.com"

    def test_optional_fields_default_none(self):
        info = ArticleInfo(
            id="a1b2c3d4-5678-90ab-cdef-1234567890ab",
            url="https://example.com",
        )
        assert info.title is None
        assert info.source is None
        assert info.metadata is None

    def test_id_required(self):
        with pytest.raises(ValidationError):
            ArticleInfo(url="https://example.com")

    def test_url_required(self):
        with pytest.raises(ValidationError):
            ArticleInfo(id="a1b2c3d4-5678-90ab-cdef-1234567890ab")


# ── ChunkData ──


class TestChunkData:
    def test_valid(self):
        chunk = ChunkData(
            chunk_index=0,
            content="Hello world",
            dense_vector=[0.1] * 1024,
            sparse_vector={"0": 0.5, "1": 0.3},
        )
        assert chunk.chunk_index == 0
        assert chunk.content == "Hello world"
        assert len(chunk.dense_vector) == 1024

    def test_sparse_vector_optional(self):
        chunk = ChunkData(
            chunk_index=0,
            content="Hello world",
            dense_vector=[0.1] * 1024,
        )
        assert chunk.sparse_vector is None

    def test_chunk_index_negative_rejected(self):
        with pytest.raises(ValidationError):
            ChunkData(
                chunk_index=-1,
                content="text",
                dense_vector=[0.1] * 1024,
            )

    def test_chunk_index_zero_accepted(self):
        chunk = ChunkData(
            chunk_index=0,
            content="text",
            dense_vector=[0.1] * 1024,
        )
        assert chunk.chunk_index == 0


# ── StoreChunksRequest ──


class TestStoreChunksRequest:
    def test_valid(self):
        req = StoreChunksRequest(
            article=ArticleInfo(id="uuid-1", url="https://example.com"),
            chunks=[
                ChunkData(chunk_index=0, content="c1", dense_vector=[0.1] * 1024)
            ],
        )
        assert req.article.id == "uuid-1"
        assert len(req.chunks) == 1

    def test_multiple_chunks(self):
        req = StoreChunksRequest(
            article=ArticleInfo(id="uuid-1", url="https://example.com"),
            chunks=[
                ChunkData(chunk_index=0, content="c1", dense_vector=[0.1] * 1024),
                ChunkData(chunk_index=1, content="c2", dense_vector=[0.2] * 1024),
            ],
        )
        assert len(req.chunks) == 2
        assert req.chunks[1].chunk_index == 1


# ── StoreChunksResponse ──


class TestStoreChunksResponse:
    def test_valid(self):
        resp = StoreChunksResponse(stored=5, article_id="uuid-1")
        assert resp.stored == 5
        assert resp.article_id == "uuid-1"

    def test_json_matches_spec(self):
        resp = StoreChunksResponse(stored=3, article_id="uuid-1")
        data = resp.model_dump()
        assert data == {"stored": 3, "article_id": "uuid-1"}


# ── SearchRequest ──


class TestSearchRequest:
    def test_valid_with_defaults(self):
        req = SearchRequest(query="What is RAG?")
        assert req.query == "What is RAG?"
        assert req.top_k == 10

    def test_valid_with_top_k(self):
        req = SearchRequest(query="test", top_k=50)
        assert req.top_k == 50

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_top_k_below_min_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)

    def test_top_k_above_max_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=101)


# ── ChatRequest ──


class TestChatRequest:
    def test_valid(self):
        req = ChatRequest(message="Explain RAG")
        assert req.message == "Explain RAG"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")


# ── ChunkResult ──


class TestChunkResult:
    def test_valid(self):
        chunk = ChunkResult(
            chunk_id="uuid-c1",
            article_id="uuid-a1",
            article_title="Title",
            article_url="https://example.com",
            chunk_index=0,
            content="Hello",
            score=0.876,
        )
        assert chunk.chunk_id == "uuid-c1"
        assert chunk.score == 0.876

    def test_optional_fields_default_none(self):
        chunk = ChunkResult(
            chunk_id="uuid-c1",
            article_id="uuid-a1",
            chunk_index=0,
            content="Hello",
            score=0.5,
        )
        assert chunk.article_title is None
        assert chunk.article_url is None

    def test_json_matches_spec(self):
        chunk = ChunkResult(
            chunk_id="uuid-c1",
            article_id="uuid-a1",
            article_title="Title",
            article_url="https://example.com",
            chunk_index=0,
            content="Hello",
            score=0.876,
        )
        data = chunk.model_dump()
        assert data == {
            "chunk_id": "uuid-c1",
            "article_id": "uuid-a1",
            "article_title": "Title",
            "article_url": "https://example.com",
            "chunk_index": 0,
            "content": "Hello",
            "score": 0.876,
        }


# ── SearchResponse ──


class TestSearchResponse:
    def test_valid(self):
        resp = SearchResponse(
            chunks=[
                ChunkResult(
                    chunk_id="c1",
                    article_id="a1",
                    chunk_index=0,
                    content="Hello",
                    score=0.9,
                )
            ]
        )
        assert len(resp.chunks) == 1
        assert resp.chunks[0].score == 0.9

    def test_empty_chunks(self):
        resp = SearchResponse()
        assert resp.chunks == []

    def test_json_matches_spec(self):
        resp = SearchResponse(chunks=[])
        data = resp.model_dump()
        assert data == {"chunks": []}


# ── ArticleCitation ──


class TestArticleCitation:
    def test_valid(self):
        cite = ArticleCitation(id="uuid-1", title="Title", url="https://example.com")
        assert cite.id == "uuid-1"
        assert cite.title == "Title"

    def test_optional_fields_default_none(self):
        cite = ArticleCitation(id="uuid-1")
        assert cite.title is None
        assert cite.url is None


# ── ChatResponse ──


class TestChatResponse:
    def test_valid(self):
        resp = ChatResponse(
            reply="RAG is...",
            articles_used=[ArticleCitation(id="a1", title="T")],
            chunks=[
                ChunkResult(
                    chunk_id="c1", article_id="a1", chunk_index=0, content="X", score=0.5
                )
            ],
        )
        assert resp.reply == "RAG is..."
        assert len(resp.articles_used) == 1
        assert len(resp.chunks) == 1

    def test_empty_defaults(self):
        resp = ChatResponse(reply="No context")
        assert resp.articles_used == []
        assert resp.chunks == []

    def test_json_matches_spec(self):
        resp = ChatResponse(reply="Answer")
        data = resp.model_dump()
        assert data == {
            "reply": "Answer",
            "articles_used": [],
            "chunks": [],
        }
