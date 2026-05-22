"""Article model — mirrors scrape-and-analyze articles table schema.

This model maps to the articles table in chatbot-plugin's independent
PostgreSQL database. Data is synced from scrape-and-analyze separately.
"""

import uuid

from sqlalchemy import Column, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String, unique=True, nullable=False)
    url_hash = Column(String, nullable=False)
    source = Column(String, nullable=True)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    published_at = Column(String, nullable=True)
    scraped_at = Column(String, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    correlation_id = Column(UUID(as_uuid=True), nullable=True)
    topic_id = Column(UUID(as_uuid=True), nullable=True)

    __table_args__ = (
        Index("idx_articles_source", "source"),
        Index("idx_articles_scraped_at", "scraped_at"),
        Index("idx_articles_topic_id", "topic_id"),
    )

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title={self.title!r})>"
