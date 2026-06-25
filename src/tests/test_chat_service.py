"""Tests for ChatService — prompt assembly, retrieval gating, LLM generation, pinned articles."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin.services.chat_service import ChatService, ArticleRef, SYSTEM_PROMPT


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(
    chunk_id="c1",
    article_id="a1",
    title="Test Article",
    url="https://test.com",
    public_article_id=None,
    content="RAG is retrieval-augmented generation.",
    score=0.9,
):
    return ChunkResult(
        chunk_id=chunk_id,
        article_id=article_id,
        article_metadata={
            "title": title,
            "url": url,
            **({"public_article_id": public_article_id} if public_article_id else {}),
        },
        chunk_index=0,
        content=content,
        score=score,
    )


def _mock_retriever(chunks=None, pinned_chunks=None):
    """Return a retriever whose retrieve() result depends on whether filters are passed.

    If *pinned_chunks* is given the mock returns those when a ``public_article_id``
    filter is present, otherwise it returns *chunks* (the normal semantic results).
    """
    retriever = MagicMock()
    normal_chunks = [_chunk()] if chunks is None else chunks
    pinned = pinned_chunks or []

    async def _retrieve(query, top_k=10, min_score=0.0, min_rerank_score=0.0, filters=None):
        if filters and "public_article_id" in filters:
            return SearchResponse(chunks=pinned)
        return SearchResponse(chunks=normal_chunks)

    retriever.retrieve = AsyncMock(side_effect=_retrieve)
    return retriever


def _mock_llm(reply="Generated reply", thinking=None):
    """LLM mock that returns (thinking, reply) matching ResilientLLMService.complete()."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=(thinking, reply))
    return llm


def _service(chunks=None, pinned_chunks=None, reply="Generated reply", **kwargs):
    return ChatService(
        retriever=_mock_retriever(chunks=chunks, pinned_chunks=pinned_chunks),
        llm=_mock_llm(reply=reply),
        **kwargs,
    )


# ── Basic chat behaviour ──────────────────────────────────────────────────────

class TestChatService:
    @pytest.mark.asyncio
    async def test_chat_returns_reply(self):
        result = await _service().chat("What is RAG?")
        assert result.reply == "Generated reply"

    @pytest.mark.asyncio
    async def test_chat_passes_message_to_retriever(self):
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm())
        await service.chat("What is RAG?")
        retriever.retrieve.assert_called_with(
            "What is RAG?", top_k=10, min_score=0.0, min_rerank_score=0.0, filters=None,
        )

    @pytest.mark.asyncio
    async def test_chat_passes_topic_id_as_filter(self):
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm())
        await service.chat("What is RAG?", topic_id="topic-abc")
        retriever.retrieve.assert_called_with(
            "What is RAG?", top_k=10, min_score=0.0, min_rerank_score=0.0,
            filters={"topic_id": "topic-abc"},
        )

    @pytest.mark.asyncio
    async def test_chat_respects_custom_thresholds(self):
        retriever = _mock_retriever()
        service = ChatService(
            retriever=retriever, llm=_mock_llm(),
            min_score=0.3, min_rerank_score=0.7,
        )
        await service.chat("What is RAG?")
        retriever.retrieve.assert_called_with(
            "What is RAG?", top_k=10, min_score=0.3, min_rerank_score=0.7, filters=None,
        )

    @pytest.mark.asyncio
    async def test_chat_passes_max_context_chunks(self):
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm(), max_context_chunks=5)
        await service.chat("hi")
        retriever.retrieve.assert_called_with(
            "hi", top_k=5, min_score=0.0, min_rerank_score=0.0, filters=None,
        )

    @pytest.mark.asyncio
    async def test_chat_passes_max_tokens(self):
        llm = _mock_llm()
        service = ChatService(retriever=_mock_retriever(), llm=llm, max_tokens=512)
        await service.chat("hi")
        call_args = llm.complete.call_args
        assert call_args.args[1] == 512

    @pytest.mark.asyncio
    async def test_chat_no_relevant_chunks_returns_fallback(self):
        service = ChatService(retriever=_mock_retriever(chunks=[]), llm=_mock_llm())
        result = await service.chat("obscure question")
        assert "couldn't find" in result.reply.lower()
        assert result.articles_used == []
        assert result.chunks == []
        service._llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_assembles_context_with_numbered_citations(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="RAG 101", content="RAG is retrieval-augmented generation."),
            _chunk(chunk_id="c2", article_id="a2", title="LLM Guide", content="LLMs are large language models."),
        ]
        llm = _mock_llm()
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=llm)
        await service.chat("What is RAG?")
        messages = llm.complete.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        user_msg = messages[1]["content"]
        assert "[1] RAG 101" in user_msg
        assert "[2] LLM Guide" in user_msg
        assert "Question: What is RAG?" in user_msg

    @pytest.mark.asyncio
    async def test_chat_collects_unique_articles(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article 1", url="https://a.com"),
            _chunk(chunk_id="c2", article_id="a1", title="Article 1", url="https://a.com"),
            _chunk(chunk_id="c3", article_id="a2", title="Article 2", url="https://b.com"),
        ]
        result = await _service(chunks=chunks).chat("q")
        assert len(result.articles_used) == 2
        assert result.articles_used[0].id == "a1"
        assert result.articles_used[1].id == "a2"

    @pytest.mark.asyncio
    async def test_chat_exposes_public_article_id_on_article_ref(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", public_article_id="pub-uuid-1"),
        ]
        result = await _service(chunks=chunks).chat("q")
        assert result.articles_used[0].public_article_id == "pub-uuid-1"

    @pytest.mark.asyncio
    async def test_chat_returns_chunks_used(self):
        chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2", article_id="a2")]
        result = await _service(chunks=chunks).chat("q")
        assert len(result.chunks) == 2


# ── Pinned article behaviour ──────────────────────────────────────────────────

class TestPinnedArticles:
    @pytest.mark.asyncio
    async def test_pinned_article_chunks_prepended_before_semantic(self):
        """Chunks from pinned article appear first in the merged list."""
        semantic_chunks = [_chunk(chunk_id="sem1", article_id="a-semantic", title="Semantic")]
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]

        llm = _mock_llm()
        service = ChatService(
            retriever=_mock_retriever(chunks=semantic_chunks, pinned_chunks=pinned_chunks),
            llm=llm,
        )
        result = await service.chat("question", pinned_article_ids=["pub-uuid"])
        # Pinned chunk should be first
        assert result.chunks[0].chunk_id == "pin1"
        assert result.chunks[1].chunk_id == "sem1"

    @pytest.mark.asyncio
    async def test_pinned_retrieval_uses_public_article_id_filter(self):
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm())
        await service.chat("question", pinned_article_ids=["pub-uuid-123"])

        calls = retriever.retrieve.call_args_list
        pinned_call = next(
            (c for c in calls if (c.kwargs.get("filters") or {}).get("public_article_id")),
            None,
        )
        assert pinned_call is not None
        assert pinned_call.kwargs["filters"]["public_article_id"] == "pub-uuid-123"

    @pytest.mark.asyncio
    async def test_pinned_uses_min_score_and_rerank_thresholds(self):
        retriever = _mock_retriever()
        service = ChatService(
            retriever=retriever, llm=_mock_llm(),
            min_score=0.3, min_rerank_score=0.7,
        )
        await service.chat("q", pinned_article_ids=["pid"])
        pinned_call = next(
            c for c in retriever.retrieve.call_args_list
            if (c.kwargs.get("filters") or {}).get("public_article_id")
        )
        assert pinned_call.kwargs["min_score"] == 0.3
        assert pinned_call.kwargs["min_rerank_score"] == 0.7

    @pytest.mark.asyncio
    async def test_multiple_pinned_articles_each_get_retrieve_call(self):
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm())
        await service.chat("q", pinned_article_ids=["id1", "id2", "id3"])

        pinned_calls = [
            c for c in retriever.retrieve.call_args_list
            if "public_article_id" in (c.kwargs.get("filters") or {})
        ]
        assert len(pinned_calls) == 3
        ids_queried = {c.kwargs["filters"]["public_article_id"] for c in pinned_calls}
        assert ids_queried == {"id1", "id2", "id3"}

    @pytest.mark.asyncio
    async def test_multiple_pinned_articles_slot_allocation(self):
        """Each article gets max_context_chunks // n_articles slots (minimum 3)."""
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm(), max_context_chunks=10)
        await service.chat("q", pinned_article_ids=["id1", "id2"])

        pinned_calls = [
            c for c in retriever.retrieve.call_args_list
            if "public_article_id" in (c.kwargs.get("filters") or {})
        ]
        for call in pinned_calls:
            assert call.kwargs["top_k"] == 5  # 10 // 2

    @pytest.mark.asyncio
    async def test_slot_allocation_minimum_three_per_article(self):
        """With many articles the minimum is 3 chunks per article."""
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm(), max_context_chunks=6)
        await service.chat("q", pinned_article_ids=["id1", "id2", "id3", "id4"])

        pinned_calls = [
            c for c in retriever.retrieve.call_args_list
            if "public_article_id" in (c.kwargs.get("filters") or {})
        ]
        for call in pinned_calls:
            assert call.kwargs["top_k"] == 3  # max(6//4=1, 3) → 3

    @pytest.mark.asyncio
    async def test_pinned_chunks_deduplicated_with_semantic(self):
        """A chunk returned by both pinned and semantic retrieval appears only once."""
        shared = _chunk(chunk_id="shared", article_id="a1")
        semantic = _chunk(chunk_id="unique-sem", article_id="a2")

        async def _retrieve(query, top_k=10, min_score=0.0, min_rerank_score=0.0, filters=None):
            if filters and "public_article_id" in filters:
                return SearchResponse(chunks=[shared])
            return SearchResponse(chunks=[shared, semantic])

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=_retrieve)
        service = ChatService(retriever=retriever, llm=_mock_llm())
        result = await service.chat("q", pinned_article_ids=["pub"])

        chunk_ids = [c.chunk_id for c in result.chunks]
        assert chunk_ids.count("shared") == 1

    @pytest.mark.asyncio
    async def test_pinned_retrieve_failure_is_graceful(self):
        """A retrieval error for a pinned article does not crash the whole request."""
        semantic_chunks = [_chunk(chunk_id="sem1")]

        async def _retrieve(query, top_k=10, min_score=0.0, min_rerank_score=0.0, filters=None):
            if filters and "public_article_id" in filters:
                raise RuntimeError("vector service unavailable")
            return SearchResponse(chunks=semantic_chunks)

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=_retrieve)
        service = ChatService(retriever=retriever, llm=_mock_llm())
        result = await service.chat("q", pinned_article_ids=["bad-id"])

        assert result.reply == "Generated reply"
        assert any(c.chunk_id == "sem1" for c in result.chunks)

    @pytest.mark.asyncio
    async def test_no_pinned_ids_does_not_call_retrieve_with_filter(self):
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm())
        await service.chat("q", pinned_article_ids=None)

        pinned_calls = [
            c for c in retriever.retrieve.call_args_list
            if "public_article_id" in (c.kwargs.get("filters") or {})
        ]
        assert pinned_calls == []

    @pytest.mark.asyncio
    async def test_merged_chunks_capped_at_max_context_chunks(self):
        """Total chunks passed to LLM never exceeds max_context_chunks."""
        many = [_chunk(chunk_id=f"c{i}", article_id=f"a{i}") for i in range(8)]
        pinned = [_chunk(chunk_id=f"p{i}", article_id=f"pa{i}") for i in range(6)]

        async def _retrieve(query, top_k=10, min_score=0.0, min_rerank_score=0.0, filters=None):
            if filters and "public_article_id" in filters:
                return SearchResponse(chunks=pinned[:top_k])
            return SearchResponse(chunks=many[:top_k])

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=_retrieve)
        service = ChatService(retriever=retriever, llm=_mock_llm(), max_context_chunks=5)
        result = await service.chat("q", pinned_article_ids=["pid"])
        assert len(result.chunks) <= 5

    @pytest.mark.asyncio
    async def test_only_pinned_chunks_sufficient_when_semantic_empty(self):
        """If semantic retrieval returns nothing but pinned has results, chat proceeds."""
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]

        async def _retrieve(query, top_k=10, min_score=0.0, min_rerank_score=0.0, filters=None):
            if filters and "public_article_id" in filters:
                return SearchResponse(chunks=pinned_chunks)
            return SearchResponse(chunks=[])

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=_retrieve)
        service = ChatService(retriever=retriever, llm=_mock_llm())
        result = await service.chat("q", pinned_article_ids=["pub"])
        assert result.reply == "Generated reply"
        assert len(result.chunks) == 1


# ── Context assembly ──────────────────────────────────────────────────────────

class TestContextAssembly:
    def _make_service(self, chunks):
        return ChatService(retriever=_mock_retriever(chunks=chunks), llm=_mock_llm())

    def test_build_context_numbered_format(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="My Title", content="First."),
            _chunk(chunk_id="c2", article_id="a2", title="Other", content="Second."),
        ]
        service = self._make_service(chunks)
        _, index = service._collect_articles(chunks)
        context = service._build_context(chunks, index)
        assert "[1] My Title" in context
        assert "First." in context
        assert "[2] Other" in context
        assert "Second." in context

    def test_build_context_unknown_title_fallback(self):
        chunks = [_chunk(title=None, content="Some text.")]
        service = self._make_service(chunks)
        _, index = service._collect_articles(chunks)
        context = service._build_context(chunks, index)
        assert "Unknown" in context
        assert "Some text." in context

    def test_collect_articles_deduplicates(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1"),
            _chunk(chunk_id="c2", article_id="a1"),
            _chunk(chunk_id="c3", article_id="a2"),
        ]
        service = self._make_service(chunks)
        articles, _ = service._collect_articles(chunks)
        assert len(articles) == 2

    def test_collect_articles_preserves_first_appearance_order(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a2"),
            _chunk(chunk_id="c2", article_id="a1"),
        ]
        service = self._make_service(chunks)
        articles, _ = service._collect_articles(chunks)
        assert articles[0].id == "a2"
        assert articles[1].id == "a1"

    def test_collect_articles_extracts_public_article_id(self):
        chunks = [_chunk(article_id="vec-id", public_article_id="pub-id-123")]
        service = self._make_service(chunks)
        articles, _ = service._collect_articles(chunks)
        assert articles[0].public_article_id == "pub-id-123"
