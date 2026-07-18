"""Tests for ChatService — prompt assembly, retrieval gating, LLM generation, pinned articles."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin.services.chat_service import ChatService, ArticleRef, SYSTEM_PROMPT
from chatbot_plugin.llm.base import AllProvidersExhausted, LLMResult, ProviderHandler, ToolCallRequest


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


def _fake_handler(name: str = "mock") -> ProviderHandler:
    return ProviderHandler(provider=AsyncMock(), strategy=AsyncMock(), priority=1, name=name)


def _mock_llm(reply="Generated reply", thinking=None):
    """LLM mock that returns (LLMResult, ProviderHandler) matching ResilientLLMService.complete()."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=(LLMResult(thinking=thinking, text=reply), _fake_handler()))
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

    @pytest.mark.asyncio
    async def test_chat_filters_articles_to_only_those_cited_in_reply(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article 1"),
            _chunk(chunk_id="c2", article_id="a2", title="Article 2"),
            _chunk(chunk_id="c3", article_id="a3", title="Article 3"),
        ]
        result = await _service(chunks=chunks, reply="See [1] for details.").chat("q")
        assert [a.id for a in result.articles_used] == ["a1"]

    @pytest.mark.asyncio
    async def test_chat_filters_articles_with_grouped_citation(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article 1"),
            _chunk(chunk_id="c2", article_id="a2", title="Article 2"),
            _chunk(chunk_id="c3", article_id="a3", title="Article 3"),
        ]
        result = await _service(chunks=chunks, reply="See [1, 3] for details.").chat("q")
        assert [a.id for a in result.articles_used] == ["a1", "a3"]

    @pytest.mark.asyncio
    async def test_chat_returns_all_articles_when_reply_has_no_citations(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article 1"),
            _chunk(chunk_id="c2", article_id="a2", title="Article 2"),
        ]
        result = await _service(chunks=chunks, reply="No citations here.").chat("q")
        assert [a.id for a in result.articles_used] == ["a1", "a2"]


# ── Pinned article behaviour ──────────────────────────────────────────────────

class TestPinnedArticles:
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
        assert pinned_call.kwargs["filters"]["public_article_id"] == ["pub-uuid-123"]

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
    async def test_multiple_pinned_articles_single_batched_call(self):
        """All pinned article ids are queried in a single batched retrieve call."""
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm())
        await service.chat("q", pinned_article_ids=["id1", "id2", "id3"])

        pinned_calls = [
            c for c in retriever.retrieve.call_args_list
            if "public_article_id" in (c.kwargs.get("filters") or {})
        ]
        assert len(pinned_calls) == 1
        assert pinned_calls[0].kwargs["filters"]["public_article_id"] == ["id1", "id2", "id3"]

    @pytest.mark.asyncio
    async def test_pinned_batch_call_uses_max_context_chunks_as_top_k(self):
        """The batched pinned call requests up to max_context_chunks regardless of article count."""
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm(), max_context_chunks=10)
        await service.chat("q", pinned_article_ids=["id1", "id2"])

        pinned_calls = [
            c for c in retriever.retrieve.call_args_list
            if "public_article_id" in (c.kwargs.get("filters") or {})
        ]
        assert len(pinned_calls) == 1
        assert pinned_calls[0].kwargs["top_k"] == 10

    @pytest.mark.asyncio
    async def test_pinned_retrieve_failure_is_graceful(self):
        """A retrieval error for a pinned article does not crash the whole request, and must
        not silently fall back to unrelated full-corpus search results."""
        semantic_chunks = [_chunk(chunk_id="semantic1", article_id="a-semantic", title="Unrelated")]

        async def _retrieve(query, top_k=10, min_score=0.0, min_rerank_score=0.0, filters=None):
            if filters and "public_article_id" in filters:
                raise RuntimeError("vector service unavailable")
            return SearchResponse(chunks=semantic_chunks)

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=_retrieve)
        service = ChatService(retriever=retriever, llm=_mock_llm())
        result = await service.chat("q", pinned_article_ids=["bad-id"])

        assert result.reply == "Generated reply"
        assert result.chunks == []

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


# ── Pinned tool-calling agent loop ─────────────────────────────────────────────

class TestPinnedToolCalling:
    @pytest.mark.asyncio
    async def test_no_tool_call_when_pinned_context_sufficient(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        service = _service(
            pinned_chunks=pinned_chunks,
            reply="Generated reply",  # mock LLM never requests a tool by default
        )
        result = await service.chat("what is this about", pinned_article_ids=["pub"])
        assert result.tool_calls_executed == []
        assert result.reply == "Generated reply"

    @pytest.mark.asyncio
    async def test_tool_called_when_pinned_context_insufficient(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        searched_chunks = [_chunk(chunk_id="s1", article_id="a-searched", title="Searched")]

        llm = AsyncMock()
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "broader topic"})
        llm.complete = AsyncMock(side_effect=[
            (LLMResult(thinking=None, text="", tool_calls=[tool_call]), _fake_handler()),
            (LLMResult(thinking=None, text="Final answer citing [2]"), _fake_handler()),
        ])

        retriever = _mock_retriever(chunks=searched_chunks, pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7)

        result = await service.chat("broad question", pinned_article_ids=["pub"])

        assert len(result.tool_calls_executed) == 1
        assert result.tool_calls_executed[0].name == "search_articles"
        assert result.tool_calls_executed[0].is_error is False
        assert result.reply == "Final answer citing [2]"

        # search call must NOT filter by public_article_id (full corpus)
        search_calls = [c for c in retriever.retrieve.call_args_list if c.kwargs.get("filters") is None]
        assert len(search_calls) == 1

    @pytest.mark.asyncio
    async def test_tool_execution_failure_still_produces_final_answer(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"})

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[
            (LLMResult(thinking=None, text="", tool_calls=[tool_call]), _fake_handler()),
            (LLMResult(thinking=None, text="Sorry, I could not find more info"), _fake_handler()),
        ])

        retriever = AsyncMock()
        async def retrieve_side_effect(*args, **kwargs):
            if kwargs.get("filters") is None:
                raise RuntimeError("search backend down")
            return SearchResponse(chunks=pinned_chunks)
        retriever.retrieve = AsyncMock(side_effect=retrieve_side_effect)

        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7)
        result = await service.chat("question", pinned_article_ids=["pub"])

        assert result.tool_calls_executed[0].is_error is True
        assert result.reply == "Sorry, I could not find more info"

    @pytest.mark.asyncio
    async def test_provider_failure_mid_exchange_restarts_with_next_provider(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"})
        failing_handler = _fake_handler(name="failing")
        backup_handler = _fake_handler(name="backup")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[
            (LLMResult(thinking=None, text="", tool_calls=[tool_call]), failing_handler),  # round 1, attempt 1
            AllProvidersExhausted(),  # round 2, attempt 1 (pinned to failing_handler) -> fails
            (LLMResult(thinking=None, text="", tool_calls=[tool_call]), backup_handler),  # round 1, attempt 2
            (LLMResult(thinking=None, text="Final answer"), backup_handler),  # round 2, attempt 2 -> succeeds
        ])

        retriever = _mock_retriever(chunks=[], pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7)
        result = await service.chat("question", pinned_article_ids=["pub"])

        assert result.reply == "Final answer"
        assert llm.complete.call_count == 4
        # second attempt's first call must exclude the failed handler
        second_attempt_call = llm.complete.call_args_list[2]
        assert second_attempt_call.kwargs["exclude"] == {"failing"}

    @pytest.mark.asyncio
    async def test_non_pinned_requests_unaffected(self):
        semantic_chunks = [_chunk(chunk_id="s1", article_id="a1")]
        service = _service(chunks=semantic_chunks, reply="Normal reply")
        result = await service.chat("question", pinned_article_ids=None)
        assert result.tool_calls_executed == []
        assert result.reply == "Normal reply"


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
