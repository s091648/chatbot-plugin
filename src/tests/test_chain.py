"""Tests for RAG chain (rag_generate)."""

import pytest
from unittest.mock import AsyncMock

from chatbot_plugin.llm.rate_limit.quota_strategy import RateLimitExhausted
from chatbot_plugin.rag.chain import rag_generate


@pytest.mark.asyncio
async def test_rag_generate_success():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "RAG is retrieval-augmented generation."
    result = await rag_generate("What is RAG?", [], mock_llm)
    assert "RAG" in result


@pytest.mark.asyncio
async def test_rag_generate_none_raises_runtime_error():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = None
    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        await rag_generate("hello", [], mock_llm)


@pytest.mark.asyncio
async def test_rag_generate_passes_articles_to_prompt():
    articles = [{"title": "AI", "content": "Artificial intelligence."}]
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "AI response"
    result = await rag_generate("What is AI?", articles, mock_llm)
    assert result == "AI response"
    # Verify generate was called with system and human prompts
    call_args = mock_llm.generate.call_args
    assert len(call_args.args) == 2  # system_prompt, human_prompt
