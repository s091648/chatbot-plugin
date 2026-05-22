"""Tests for ChatbotService."""

import pytest

from chatbot_plugin.service import ChatbotService


@pytest.fixture
def service() -> ChatbotService:
    return ChatbotService()


@pytest.mark.asyncio
async def test_chat_returns_reply(service: ChatbotService):
    result = await service.chat("hello")
    assert result.reply
    assert isinstance(result.articles_used, list)


@pytest.mark.asyncio
async def test_search_returns_chunks(service: ChatbotService):
    result = await service.search("RAG")
    assert isinstance(result.chunks, list)


@pytest.mark.asyncio
async def test_trigger_index_returns_job(service: ChatbotService):
    result = await service.trigger_index()
    assert result.job_id
    assert result.status == "started"


@pytest.mark.asyncio
async def test_get_status_returns_shape(service: ChatbotService):
    result = await service.get_status()
    assert isinstance(result.total_chunks, int)
    assert isinstance(result.pending_articles, int)
