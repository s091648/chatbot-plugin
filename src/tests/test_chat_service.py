"""Tests for ChatService — prompt assembly, retrieval gating, LLM generation."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot_plugin_sdk.contracts.responses import ChunkResult, SearchResponse
from chatbot_plugin.services.chat_service import ChatService, ArticleRef, SYSTEM_PROMPT


def _chunk(chunk_id="c1", article_id="a1", title="Test Article", url="https://test.com", content="RAG is retrieval-augmented generation.", score=0.9):
    return ChunkResult(
        chunk_id=chunk_id,
        article_id=article_id,
        article_title=title,
        article_url=url,
        chunk_index=0,
        content=content,
        score=score,
    )


def _mock_retriever(chunks=None):
    retriever = MagicMock()
    if chunks is None:
        chunks = [_chunk()]
    retriever.retrieve = AsyncMock(return_value=SearchResponse(chunks=chunks))
    return retriever


def _mock_llm(reply="Generated reply"):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=reply)
    return llm


class TestChatService:
    @pytest.mark.asyncio
    async def test_chat_returns_reply(self):
        service = ChatService(retriever=_mock_retriever(), llm=_mock_llm())
        result = await service.chat("What is RAG?")
        assert result.reply == "Generated reply"

    @pytest.mark.asyncio
    async def test_chat_passes_message_to_retriever(self):
        retriever = _mock_retriever()
        service = ChatService(retriever=retriever, llm=_mock_llm())
        await service.chat("What is RAG?")
        retriever.retrieve.assert_called_once_with(
            "What is RAG?", top_k=10, min_score=0.0, min_rerank_score=0.0,
        )

    @pytest.mark.asyncio
    async def test_chat_respects_custom_thresholds(self):
        retriever = _mock_retriever()
        service = ChatService(
            retriever=retriever, llm=_mock_llm(),
            min_score=0.3, min_rerank_score=0.7,
        )
        await service.chat("What is RAG?")
        retriever.retrieve.assert_called_once_with(
            "What is RAG?", top_k=10, min_score=0.3, min_rerank_score=0.7,
        )

    @pytest.mark.asyncio
    async def test_chat_no_relevant_chunks_returns_fallback(self):
        retriever = _mock_retriever(chunks=[])
        service = ChatService(retriever=retriever, llm=_mock_llm())
        result = await service.chat("obscure question")
        assert "couldn't find" in result.reply.lower()
        assert result.articles_used == []
        assert result.chunks == []
        # LLM should NOT be called
        service._llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_assembles_context(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="RAG 101", content="RAG is retrieval-augmented generation."),
            _chunk(chunk_id="c2", article_id="a2", title="LLM Guide", content="LLMs are large language models."),
        ]
        llm = _mock_llm()
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=llm)
        await service.chat("What is RAG?")
        call_args = llm.complete.call_args
        messages = call_args.args[0]
        # System message should contain the SYSTEM_PROMPT
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        # User message should contain context with source annotations
        user_msg = messages[1]["content"]
        assert "[source: RAG 101]" in user_msg
        assert "[source: LLM Guide]" in user_msg
        assert "Question: What is RAG?" in user_msg

    @pytest.mark.asyncio
    async def test_chat_collects_unique_articles(self):
        chunks = [
            _chunk(chunk_id="c1", article_id="a1", title="Article 1", url="https://a.com"),
            _chunk(chunk_id="c2", article_id="a1", title="Article 1", url="https://a.com"),
            _chunk(chunk_id="c3", article_id="a2", title="Article 2", url="https://b.com"),
        ]
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=_mock_llm())
        result = await service.chat("q")
        assert len(result.articles_used) == 2
        assert result.articles_used[0].id == "a1"
        assert result.articles_used[1].id == "a2"

    @pytest.mark.asyncio
    async def test_chat_raw_context_fallback_when_no_llm(self):
        chunks = [_chunk(content="RAG is...")]
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=None)  # no LLM available
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=llm)
        result = await service.chat("What is RAG?")
        # Should return context as-is
        assert "[source: Test Article]" in result.reply
        assert "Question: What is RAG?" in result.reply

    @pytest.mark.asyncio
    async def test_chat_passes_max_tokens(self):
        llm = _mock_llm()
        service = ChatService(
            retriever=_mock_retriever(), llm=llm, max_tokens=512,
        )
        await service.chat("hi")
        call_args = llm.complete.call_args
        assert call_args.args[1] == 512

    @pytest.mark.asyncio
    async def test_chat_passes_max_context_chunks(self):
        retriever = _mock_retriever()
        service = ChatService(
            retriever=retriever, llm=_mock_llm(), max_context_chunks=5,
        )
        await service.chat("hi")
        retriever.retrieve.assert_called_once_with(
            "hi", top_k=5, min_score=0.0, min_rerank_score=0.0,
        )


class TestContextAssembly:
    @pytest.mark.asyncio
    async def test_build_context_format(self):
        chunks = [
            _chunk(title="My Title", content="First paragraph."),
            _chunk(title="Other Title", content="Second paragraph."),
        ]
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=_mock_llm())
        context = service._build_context(chunks)
        assert "[source: My Title]" in context
        assert "First paragraph." in context
        assert "[source: Other Title]" in context
        assert "Second paragraph." in context

    @pytest.mark.asyncio
    async def test_build_context_no_title(self):
        chunks = [_chunk(title=None, content="Some text.")]
        service = ChatService(retriever=_mock_retriever(chunks=chunks), llm=_mock_llm())
        context = service._build_context(chunks)
        assert "[source: Unknown]" in context
