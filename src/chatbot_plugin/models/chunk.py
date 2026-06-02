"""Chunk model — stores pre-chunked, pre-embedded article data.

Each chunk belongs to an Article and holds a dense vector (pgvector)
and optional sparse vector (JSONB lexical weights).
"""

from sqlalchemy import Column, ForeignKey, Integer, Text, DateTime, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from pgvector.sqlalchemy import Vector

from chatbot_plugin.config import settings
from chatbot_plugin.models.article import Base


class ArticleChunk(Base):
    __tablename__ = "article_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    article_id = Column(
        UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    dense_vector = Column(
        Vector(settings.embedding_dimension),
        nullable=True,
    )
    sparse_vector = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # String reference avoids any import-order issues
    article = relationship("Article", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("article_id", "chunk_index", name="uq_article_chunk_idx"),
    )

    def __repr__(self) -> str:
        return f"<ArticleChunk(id={self.id}, article_id={self.article_id}, index={self.chunk_index})>"
