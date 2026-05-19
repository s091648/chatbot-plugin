"""Tests for ChatbotService."""

import pytest

from chatbot_plugin.service import ChatbotService


@pytest.fixture
def service() -> ChatbotService:
    return ChatbotService()


@pytest.mark.asyncio
async def test_chat_returns_reply(service: ChatbotService):
    result = await service.chat("hello")
    assert "reply" in result
    assert "articles_used" in result
