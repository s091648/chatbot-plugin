"""Tests for ChatService — prompt assembly, retrieval gating, LLM generation, pinned articles."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin.services.chat_service import (
    ArticleRef,
    ChatService,
    SourcesReady,
    StreamFailed,
    SYSTEM_PROMPT,
    ToolCallFinished,
    ToolCallStarted,
    _EMPTY_FOLLOWUP_REPLY,
)
from chatbot_plugin.llm.base import (
    AllProvidersExhausted,
    LLMResult,
    ProviderHandler,
    StreamError,
    TextDelta,
    ThinkingDelta,
    ToolCallRequest,
)


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


def _mock_stream_llm(events=None, reply="Generated reply"):
    """LLM mock whose stream_complete() yields a fixed (handler, event) sequence, matching
    ResilientLLMService.stream_complete(). Ignores which turn (tools vs. pinned_handler) it's
    called for — tests that need call-dependent behaviour (e.g. the tool-call round trip) wire
    llm.stream_complete directly instead of using this helper."""
    handler = _fake_handler()
    stream_events = events if events is not None else [TextDelta(text=reply)]

    def _stream_complete(*args, **kwargs):
        async def _gen():
            for event in stream_events:
                yield (handler, event)
        return _gen()

    llm = MagicMock()
    llm.stream_complete = _stream_complete
    return llm


def _service_stream(chunks=None, pinned_chunks=None, events=None, reply="Generated reply", **kwargs):
    return ChatService(
        retriever=_mock_retriever(chunks=chunks, pinned_chunks=pinned_chunks),
        llm=_mock_stream_llm(events=events, reply=reply),
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
        # The returned list is a compacted (non-contiguous) subset of the 3 context articles —
        # .number must still hold each article's *original* context index so a consumer can
        # resolve the literal "[1]"/"[3]" in the reply without relying on array position.
        assert [a.number for a in result.articles_used] == [1, 3]

    @pytest.mark.asyncio
    async def test_chat_cited_article_numbers_survive_non_contiguous_filtering(self):
        """Regression test: citing articles 2 and 4 out of 4 (skipping 1 and 3) used to leave
        articles_used == [article2, article4] with no way to tell that the reply's "[2]" refers
        to the first entry and "[4]" to the second — a naive array-position lookup would map
        "[2]" to the second entry (article4) instead. See ArticleRef.number."""
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article 1"),
            _chunk(chunk_id="c2", article_id="a2", title="Article 2"),
            _chunk(chunk_id="c3", article_id="a3", title="Article 3"),
            _chunk(chunk_id="c4", article_id="a4", title="Article 4"),
        ]
        result = await _service(chunks=chunks, reply="See [2] and [4] for details.").chat("q")
        assert [a.id for a in result.articles_used] == ["a2", "a4"]
        assert [a.number for a in result.articles_used] == [2, 4]

    @pytest.mark.asyncio
    async def test_chat_returns_all_articles_when_reply_has_no_citations(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article 1"),
            _chunk(chunk_id="c2", article_id="a2", title="Article 2"),
        ]
        result = await _service(chunks=chunks, reply="No citations here.").chat("q")
        assert [a.id for a in result.articles_used] == ["a1", "a2"]


# ── Streaming (non-pinned) ─────────────────────────────────────────────────────

class TestChatServiceStreaming:
    @pytest.mark.asyncio
    async def test_chat_stream_forwards_deltas_then_sources(self):
        chunks = [_chunk(chunk_id="c1", article_id="a1", title="Article 1")]
        service = _service_stream(
            chunks=chunks,
            events=[ThinkingDelta(text="pondering"), TextDelta(text="RAG "), TextDelta(text="is cool [1]")],
        )
        events = [e async for e in service.chat_stream("What is RAG?")]
        assert events[:-1] == [
            ThinkingDelta(text="pondering"),
            TextDelta(text="RAG "),
            TextDelta(text="is cool [1]"),
        ]
        assert isinstance(events[-1], SourcesReady)
        assert [a.id for a in events[-1].articles] == ["a1"]

    @pytest.mark.asyncio
    async def test_chat_stream_yields_fallback_reply_when_nothing_retrieved(self):
        service = _service_stream(chunks=[])
        events = [e async for e in service.chat_stream("obscure question")]
        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) == 1
        assert "couldn't find relevant information" in text_events[0].text
        assert isinstance(events[-1], SourcesReady)
        assert events[-1].articles == []

    @pytest.mark.asyncio
    async def test_chat_stream_surfaces_stream_error_as_stream_failed(self):
        service = _service_stream(
            chunks=[_chunk()],
            events=[TextDelta(text="partial"), StreamError("boom")],
        )
        events = [e async for e in service.chat_stream("q")]
        assert events == [TextDelta(text="partial"), StreamFailed(message="boom")]


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
        # thought_signature present — this exercises the "must stay pinned" path (see
        # TestFollowupHandlerPinning for the no-signature "free to rotate" counterpart).
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"}, thought_signature=b"sig-bytes")
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


# ── Follow-up handler pinning (regression) ──────────────────────────────────────
# Pinning the follow-up call to whichever handler made the tool call exists only because Gemini
# rejects a follow-up turn that echoes back a function-call part's thought_signature to a
# *different* model (400 INVALID_ARGUMENT). A tool call with no thought_signature at all has
# nothing that constraint applies to, so the follow-up should be free to rotate across every
# other configured handler — pinning it anyway would strand a turn on one quota-exhausted model
# while sibling models configured specifically as fallbacks still had headroom (the bug a real
# user hit: gemini-3-flash-preview's 20/day cap exhausted, gemini-3.1-flash-lite's 500/day
# sitting unused, and the pinned follow-up failed outright instead of trying it).

class TestFollowupHandlerPinning:
    @pytest.mark.asyncio
    async def test_followup_is_not_pinned_when_tool_call_had_no_thought_signature(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"})  # no thought_signature

        captured_kwargs = []

        async def _complete(messages, max_tokens, tools=None, **kwargs):
            captured_kwargs.append(kwargs)
            if tools:
                return (LLMResult(thinking=None, text="", tool_calls=[tool_call]), _fake_handler())
            return (LLMResult(thinking=None, text="Final answer"), _fake_handler())

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=_complete)
        retriever = _mock_retriever(chunks=[], pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=1)

        result = await service.chat("question", pinned_article_ids=["pub"])

        assert result.reply == "Final answer"
        assert len(captured_kwargs) == 2
        assert "pinned_handler" not in captured_kwargs[1]
        assert "exclude" in captured_kwargs[1]

    @pytest.mark.asyncio
    async def test_followup_is_pinned_when_tool_call_had_a_thought_signature(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"}, thought_signature=b"sig")
        first_round_handler = _fake_handler(name="round0-handler")

        captured_kwargs = []

        async def _complete(messages, max_tokens, tools=None, **kwargs):
            captured_kwargs.append(kwargs)
            if tools:
                return (LLMResult(thinking=None, text="", tool_calls=[tool_call]), first_round_handler)
            return (LLMResult(thinking=None, text="Final answer"), first_round_handler)

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=_complete)
        retriever = _mock_retriever(chunks=[], pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=1)

        result = await service.chat("question", pinned_article_ids=["pub"])

        assert result.reply == "Final answer"
        assert len(captured_kwargs) == 2
        assert captured_kwargs[1].get("pinned_handler") is first_round_handler
        assert "exclude" not in captured_kwargs[1]

    @pytest.mark.asyncio
    async def test_streaming_followup_is_not_pinned_when_tool_call_had_no_thought_signature(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"})  # no thought_signature

        captured_kwargs = []

        def stream_complete(messages, max_tokens, tools=None, **kwargs):
            captured_kwargs.append(kwargs)
            if tools:
                return _stream_gen([tool_call])
            return _stream_gen([TextDelta(text="Final answer")])

        llm = MagicMock()
        llm.stream_complete = stream_complete
        retriever = _mock_retriever(chunks=[], pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=1)

        events = [e async for e in service.chat_stream("question", pinned_article_ids=["pub"])]

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Final answer"
        assert len(captured_kwargs) == 2
        assert captured_kwargs[1].get("pinned_handler") is None

    @pytest.mark.asyncio
    async def test_streaming_followup_is_pinned_when_tool_call_had_a_thought_signature(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"}, thought_signature=b"sig")
        handler = _fake_handler()

        captured_kwargs = []

        def stream_complete(messages, max_tokens, tools=None, **kwargs):
            captured_kwargs.append(kwargs)
            if tools:
                return _stream_gen([tool_call], handler=handler)
            return _stream_gen([TextDelta(text="Final answer")], handler=handler)

        llm = MagicMock()
        llm.stream_complete = stream_complete
        retriever = _mock_retriever(chunks=[], pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=1)

        events = [e async for e in service.chat_stream("question", pinned_article_ids=["pub"])]

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Final answer"
        assert len(captured_kwargs) == 2
        assert captured_kwargs[1].get("pinned_handler") is handler


# ── Pinned tool calling — multiple rounds (non-streaming) ──────────────────────
# The model isn't limited to exactly one tool-call round trip: it may search again with a
# different query before answering, up to max_tool_rounds turns.

class TestPinnedToolCallingMultiRound:
    @pytest.mark.asyncio
    async def test_answers_after_a_second_search_with_a_different_query(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        searched_chunks = [_chunk(chunk_id="s1", article_id="a-searched", title="Searched")]
        first_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "AWS IoT Core"})
        second_call = ToolCallRequest(id="call_2", name="search_articles", arguments={"query": "AWS IoT"})

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[
            (LLMResult(thinking=None, text="", tool_calls=[first_call]), _fake_handler()),
            (LLMResult(thinking=None, text="", tool_calls=[second_call]), _fake_handler()),
            (LLMResult(thinking=None, text="Final answer citing [2]"), _fake_handler()),
        ])

        retriever = _mock_retriever(chunks=searched_chunks, pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=3)

        result = await service.chat("broad question", pinned_article_ids=["pub"])

        assert llm.complete.call_count == 3
        assert len(result.tool_calls_executed) == 2
        assert [c.name for c in result.tool_calls_executed] == ["search_articles", "search_articles"]
        assert result.reply == "Final answer citing [2]"

    @pytest.mark.asyncio
    async def test_forces_a_final_answer_once_max_tool_rounds_is_reached(self):
        """A model that keeps wanting to search forever must still be cut off — the round after
        the cap gets tools=None, which the fake LLM here always answers with `tool_calls=[]` for
        (matching what the real provider guarantees: no tools offered, none can be returned)."""
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        searched_chunks = [_chunk(chunk_id="s1", article_id="a-searched", title="Searched")]
        endless_call = ToolCallRequest(id="call_x", name="search_articles", arguments={"query": "x"})

        call_log: list[list] = []

        async def _complete(messages, max_tokens, tools=None, pinned_handler=None, exclude=None):
            call_log.append(tools)
            if tools:
                return (LLMResult(thinking=None, text="", tool_calls=[endless_call]), _fake_handler())
            return (LLMResult(thinking=None, text="Giving up, here's what I found"), _fake_handler())

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=_complete)
        retriever = _mock_retriever(chunks=searched_chunks, pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=2)

        result = await service.chat("question", pinned_article_ids=["pub"])

        # 2 tool-enabled rounds + 1 forced tools=None round = 3 calls total
        assert len(call_log) == 3
        assert call_log[0] and call_log[1] and not call_log[2]
        assert len(result.tool_calls_executed) == 2
        assert result.reply == "Giving up, here's what I found"


# ── Empty follow-up fallback ────────────────────────────────────────────────────
# The model can finish a post-tool-call turn cleanly (no error) with zero answer text — see
# gemini_provider.py's stream()/complete() "empty output" warning logs for the provider-side
# signal this pairs with.

class TestEmptyFollowupFallback:
    @pytest.mark.asyncio
    async def test_substitutes_fallback_text_when_followup_produces_no_text(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"})

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[
            (LLMResult(thinking=None, text="", tool_calls=[tool_call]), _fake_handler()),
            (LLMResult(thinking="wanted to search again but couldn't", text=""), _fake_handler()),
        ])

        retriever = _mock_retriever(chunks=[], pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=1)

        result = await service.chat("question", pinned_article_ids=["pub"])
        assert result.reply == _EMPTY_FOLLOWUP_REPLY

    @pytest.mark.asyncio
    async def test_does_not_substitute_fallback_when_no_tool_call_was_ever_made(self):
        """An empty reply with no tool call at all is a different failure mode this fallback
        isn't meant to cover — the raw (possibly empty) text passes through unchanged."""
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        service = _service(pinned_chunks=pinned_chunks, reply="")
        result = await service.chat("question", pinned_article_ids=["pub"])
        assert result.reply == ""


# ── Pinned tool calling (streaming) ─────────────────────────────────────────────

def _stream_gen(events, handler=None):
    h = handler or _fake_handler()
    async def _gen():
        for event in events:
            yield (h, event)
    return _gen()


class TestPinnedToolCallingStreaming:
    @pytest.mark.asyncio
    async def test_no_tool_call_streams_straight_through(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        llm = MagicMock()
        llm.stream_complete = lambda *a, **kw: _stream_gen(
            [ThinkingDelta(text="checking pinned content"), TextDelta(text="Generated reply")]
        )
        service = ChatService(retriever=_mock_retriever(pinned_chunks=pinned_chunks), llm=llm)

        events = [e async for e in service.chat_stream("what is this about", pinned_article_ids=["pub"])]

        assert events[:-1] == [ThinkingDelta(text="checking pinned content"), TextDelta(text="Generated reply")]
        assert isinstance(events[-1], SourcesReady)
        assert not any(isinstance(e, (ToolCallStarted, ToolCallFinished)) for e in events)

    @pytest.mark.asyncio
    async def test_tool_call_streams_started_finished_then_follow_up_text(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        searched_chunks = [_chunk(chunk_id="s1", article_id="a-searched", title="Searched")]
        # thought_signature present — this is what actually requires the follow-up call to stay
        # pinned to the same handler (see TestFollowupHandlerPinning for the no-signature case).
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "broader topic"}, thought_signature=b"sig-bytes")
        handler = _fake_handler()

        def stream_complete(messages, max_tokens, tools=None, pinned_handler=None, exclude=None):
            if tools:
                return _stream_gen([tool_call], handler=handler)
            assert pinned_handler is handler  # follow-up must reuse the same provider
            return _stream_gen([TextDelta(text="Final answer citing [2]")], handler=handler)

        llm = MagicMock()
        llm.stream_complete = stream_complete
        retriever = _mock_retriever(chunks=searched_chunks, pinned_chunks=pinned_chunks)
        # This mock's stream_complete branches only on "tools truthy or not" (no round
        # awareness) — max_tool_rounds=1 keeps it a single-round scenario, matching what it was
        # testing before the multi-round loop existed. See TestPinnedToolCallingMultiRound for
        # coverage of rounds beyond the first.
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=1)

        events = [e async for e in service.chat_stream("broad question", pinned_article_ids=["pub"])]

        started = [e for e in events if isinstance(e, ToolCallStarted)]
        finished = [e for e in events if isinstance(e, ToolCallFinished)]
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert started == [ToolCallStarted(id="call_1", name="search_articles", arguments={"query": "broader topic"})]
        assert len(finished) == 1 and finished[0].is_error is False
        assert text == "Final answer citing [2]"
        # tool card must appear before the follow-up text in the stream, not after
        assert events.index(started[0]) < [i for i, e in enumerate(events) if isinstance(e, TextDelta)][0]

    @pytest.mark.asyncio
    async def test_followup_failure_after_tool_call_yields_stream_failed_not_silent_retry(self):
        """Once the tool call has streamed, a failed follow-up can't restart with a different
        provider (that output is already visible) — it must surface as StreamFailed instead."""
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"})

        def stream_complete(messages, max_tokens, tools=None, pinned_handler=None, exclude=None):
            if tools:
                return _stream_gen([tool_call])

            async def _gen():
                if False:
                    yield
                raise AllProvidersExhausted()
            return _gen()

        llm = MagicMock()
        llm.stream_complete = stream_complete
        retriever = _mock_retriever(chunks=[], pinned_chunks=pinned_chunks)
        # Same reasoning as the previous test — this mock isn't round-aware.
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=1)

        events = [e async for e in service.chat_stream("question", pinned_article_ids=["pub"])]
        assert isinstance(events[-1], StreamFailed)
        assert not any(isinstance(e, SourcesReady) for e in events)

    @pytest.mark.asyncio
    async def test_total_failure_before_any_output_propagates(self):
        """The very first call is still safe to fail loudly — nothing has streamed yet, so the
        router can still turn this into a clean 503 (see routers/chat.py)."""
        def stream_complete(*args, **kwargs):
            async def _gen():
                if False:
                    yield
                raise AllProvidersExhausted()
            return _gen()

        llm = MagicMock()
        llm.stream_complete = stream_complete
        retriever = _mock_retriever(chunks=[], pinned_chunks=[_chunk()])
        service = ChatService(retriever=retriever, llm=llm)

        with pytest.raises(AllProvidersExhausted):
            async for _ in service.chat_stream("question", pinned_article_ids=["pub"]):
                pass


# ── Pinned tool calling — multiple rounds (streaming) ───────────────────────────

class TestPinnedToolCallingStreamingMultiRound:
    @pytest.mark.asyncio
    async def test_streams_two_tool_rounds_then_final_text(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        searched_chunks = [_chunk(chunk_id="s1", article_id="a-searched", title="Searched")]
        # thought_signature present on both — each round's follow-up must stay pinned to the
        # same handler (see TestFollowupHandlerPinning for the no-signature case).
        first_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "AWS IoT Core"}, thought_signature=b"sig-1")
        second_call = ToolCallRequest(id="call_2", name="search_articles", arguments={"query": "AWS IoT"}, thought_signature=b"sig-2")
        handler = _fake_handler()

        call_count = 0

        def stream_complete(messages, max_tokens, tools=None, pinned_handler=None, exclude=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _stream_gen([first_call], handler=handler)
            if call_count == 2:
                assert pinned_handler is handler
                return _stream_gen([second_call], handler=handler)
            assert pinned_handler is handler
            return _stream_gen([TextDelta(text="Final answer citing [2]")], handler=handler)

        llm = MagicMock()
        llm.stream_complete = stream_complete
        retriever = _mock_retriever(chunks=searched_chunks, pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=3)

        events = [e async for e in service.chat_stream("broad question", pinned_article_ids=["pub"])]

        started = [e for e in events if isinstance(e, ToolCallStarted)]
        finished = [e for e in events if isinstance(e, ToolCallFinished)]
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert [e.id for e in started] == ["call_1", "call_2"]
        assert len(finished) == 2
        assert text == "Final answer citing [2]"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_forces_final_answer_once_max_tool_rounds_is_reached(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        searched_chunks = [_chunk(chunk_id="s1", article_id="a-searched", title="Searched")]
        endless_call = ToolCallRequest(id="call_x", name="search_articles", arguments={"query": "x"})

        call_log: list[list] = []

        def stream_complete(messages, max_tokens, tools=None, pinned_handler=None, exclude=None):
            call_log.append(tools)
            if tools:
                return _stream_gen([endless_call])
            return _stream_gen([TextDelta(text="Giving up, here's what I found")])

        llm = MagicMock()
        llm.stream_complete = stream_complete
        retriever = _mock_retriever(chunks=searched_chunks, pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=2)

        events = [e async for e in service.chat_stream("question", pinned_article_ids=["pub"])]

        assert len(call_log) == 3
        assert call_log[0] and call_log[1] and not call_log[2]
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Giving up, here's what I found"

    @pytest.mark.asyncio
    async def test_yields_fallback_text_delta_when_followup_produces_no_text(self):
        pinned_chunks = [_chunk(chunk_id="pin1", article_id="a-pinned", title="Pinned")]
        tool_call = ToolCallRequest(id="call_1", name="search_articles", arguments={"query": "x"})
        handler = _fake_handler()

        def stream_complete(messages, max_tokens, tools=None, pinned_handler=None, exclude=None):
            if tools:
                return _stream_gen([tool_call], handler=handler)
            return _stream_gen([ThinkingDelta(text="wanted to search again but couldn't")], handler=handler)

        llm = MagicMock()
        llm.stream_complete = stream_complete
        retriever = _mock_retriever(chunks=[], pinned_chunks=pinned_chunks)
        service = ChatService(retriever=retriever, llm=llm, min_score=0.3, min_rerank_score=0.7, max_tool_rounds=1)

        events = [e async for e in service.chat_stream("question", pinned_article_ids=["pub"])]

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == _EMPTY_FOLLOWUP_REPLY
        assert isinstance(events[-1], SourcesReady)


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

    # Regression: a single pinned article can alone contribute up to max_context_chunks chunks
    # (see _fetch_pinned_chunks). Previously _build_context repeated "[1] Title" once per chunk,
    # which a weaker fallback model (gemini-3.1-flash-lite) mistook for a numbered list of
    # distinct sources — inventing citations like [3]/[9] from chunk position instead of the
    # literal repeated [1] label. Grouping under one header per article removes that misleading
    # repeated-numbered-list shape entirely.
    def test_build_context_groups_multiple_chunks_from_the_same_article_under_one_header(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Same Article", content="First chunk."),
            _chunk(chunk_id="c2", article_id="a1", title="Same Article", content="Second chunk."),
            _chunk(chunk_id="c3", article_id="a1", title="Same Article", content="Third chunk."),
        ]
        service = self._make_service(chunks)
        _, index = service._collect_articles(chunks)
        context = service._build_context(chunks, index)
        assert context.count("[1] Same Article") == 1
        assert "First chunk." in context
        assert "Second chunk." in context
        assert "Third chunk." in context

    def test_build_context_orders_by_article_index_even_when_chunks_interleave(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article One", content="A1 chunk."),
            _chunk(chunk_id="c2", article_id="a2", title="Article Two", content="A2 chunk."),
            _chunk(chunk_id="c3", article_id="a1", title="Article One", content="A1 chunk 2."),
        ]
        service = self._make_service(chunks)
        _, index = service._collect_articles(chunks)
        context = service._build_context(chunks, index)
        assert context.count("[1] Article One") == 1
        assert context.count("[2] Article Two") == 1
        assert context.index("[1] Article One") < context.index("[2] Article Two")

    def test_build_context_appends_valid_citation_range_for_single_article(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Same Article", content="First chunk."),
            _chunk(chunk_id="c2", article_id="a1", title="Same Article", content="Second chunk."),
        ]
        service = self._make_service(chunks)
        _, index = service._collect_articles(chunks)
        context = service._build_context(chunks, index)
        assert "exactly 1 article" in context
        assert "must be one of 1;" in context

    def test_build_context_appends_valid_citation_range_for_multiple_articles(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article One"),
            _chunk(chunk_id="c2", article_id="a2", title="Article Two"),
            _chunk(chunk_id="c3", article_id="a3", title="Article Three"),
        ]
        service = self._make_service(chunks)
        _, index = service._collect_articles(chunks)
        context = service._build_context(chunks, index)
        assert "exactly 3 article(s), numbered 1 to 3" in context

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
