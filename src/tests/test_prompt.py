"""Tests for RAG prompt building."""

import pytest

from chatbot_plugin.rag.prompt import build_context, build_messages, SYSTEM_PROMPT


# ── build_context() ──


def test_build_context_empty_articles():
    assert build_context([]) == ""


def test_build_context_single_article_fits():
    articles = [{"title": "AI", "content": "Artificial intelligence."}]
    result = build_context(articles, max_tokens=100)
    assert "[source: AI]" in result
    assert "Artificial intelligence." in result


def test_build_context_multiple_articles():
    articles = [
        {"title": "A", "content": "Content A."},
        {"title": "B", "content": "Content B."},
    ]
    result = build_context(articles, max_tokens=100)
    assert "[source: A]" in result
    assert "[source: B]" in result


def test_build_context_truncation_with_ellipsis():
    long_content = "x" * 500
    articles = [{"title": "Long", "content": long_content}]
    result = build_context(articles, max_tokens=50)
    assert "[source: Long]" in result
    assert result.endswith("...")


def test_build_context_budget_exhausted_skips_remaining():
    articles = [
        {"title": "First", "content": "x" * 200},
        {"title": "Second", "content": "y" * 200},
    ]
    result = build_context(articles, max_tokens=30)
    assert "[source: First]" in result
    assert "[source: Second]" not in result


def test_build_context_too_small_budget_skips_article():
    articles = [{"title": "Tiny", "content": "x" * 500}]
    result = build_context(articles, max_tokens=5)
    assert "[source: Tiny]" not in result or len(result) < 200


def test_build_context_untitled_default():
    articles = [{"content": "No title here."}]
    result = build_context(articles, max_tokens=100)
    assert "[source: Untitled]" in result


def test_build_context_empty_content():
    articles = [{"title": "Empty", "content": ""}]
    result = build_context(articles, max_tokens=100)
    assert "[source: Empty]" in result


def test_build_context_defaults_to_settings_max_tokens():
    articles = [{"title": "Test", "content": "Hello"}]
    result = build_context(articles)
    assert "[source: Test]" in result


# ── build_messages() ──


def test_build_messages_returns_tuple():
    system, human = build_messages("What is AI?", [])
    assert system == SYSTEM_PROMPT
    assert "What is AI?" in human


def test_build_messages_empty_articles_fallback():
    system, human = build_messages("Hello", [])
    assert "No relevant articles found." in human


def test_build_messages_with_articles():
    articles = [{"title": "AI", "content": "Artificial intelligence."}]
    system, human = build_messages("What is AI?", articles)
    assert system == SYSTEM_PROMPT
    assert "[source: AI]" in human
    assert "What is AI?" in human
