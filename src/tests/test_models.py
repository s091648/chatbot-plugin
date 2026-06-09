"""Tests for model __repr__."""

import uuid

from chatbot_plugin.models.article import Article
from chatbot_plugin.models.chunk import ArticleChunk


def test_article_repr():
    article = Article(id=uuid.uuid4(), title="Hello")
    r = repr(article)
    assert "Article" in r
    assert "Hello" in r


def test_chunk_repr():
    chunk = ArticleChunk(
        id=uuid.uuid4(), article_id=uuid.uuid4(), chunk_index=3
    )
    r = repr(chunk)
    assert "ArticleChunk" in r
    assert "index=3" in r
