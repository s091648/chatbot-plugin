"""Toolbox schema: simplified articles + article_chunks

Revision ID: 002
Revises: 001
Create Date: 2026-05-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old indexes + columns
    op.execute("DROP INDEX IF EXISTS idx_articles_tsv")
    op.drop_index("idx_articles_topic_id", table_name="articles")
    op.drop_index("idx_articles_scraped_at", table_name="articles")
    op.drop_index("idx_articles_source", table_name="articles")

    # Rebuild articles table with minimal schema
    op.drop_table("articles")

    op.create_table(
        "articles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_articles_source", "articles", ["source"])
    op.create_index("idx_articles_url", "articles", ["url"])
    op.create_unique_constraint("articles_url_key", "articles", ["url"])

    # pgvector: enable extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create article_chunks table
    op.create_table(
        "article_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("article_id", UUID(as_uuid=True), sa.ForeignKey("articles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("dense_vector", sa.Text(), nullable=True),
        sa.Column("sparse_vector", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Cast dense_vector to proper vector type
    op.execute("ALTER TABLE article_chunks ALTER COLUMN dense_vector TYPE vector(1024) USING dense_vector::vector")
    # Cast sparse_vector to proper sparsevec type
    op.execute("ALTER TABLE article_chunks ALTER COLUMN sparse_vector TYPE sparsevec(250002) USING sparse_vector::sparsevec")

    op.create_unique_constraint("uq_article_chunk_idx", "article_chunks", ["article_id", "chunk_index"])
    op.create_index("idx_chunks_article_id", "article_chunks", ["article_id"])

    # HNSW index for dense similarity search
    op.execute(
        "CREATE INDEX hnsw_chunks_dense ON article_chunks USING hnsw (dense_vector vector_cosine_ops)"
    )
    # Simple index for sparse vector (sparsevec HNSW not available in pgvector 0.8.0)
    op.execute(
        "CREATE INDEX idx_chunks_sparse ON article_chunks (sparse_vector)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_sparse")
    op.execute("DROP INDEX IF EXISTS hnsw_chunks_dense")

    op.drop_table("article_chunks")
    op.drop_table("articles")

    # Recreate old articles table (simplified, just for rollback compat)
    op.create_table(
        "articles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("url", sa.String(), nullable=False, unique=True),
        sa.Column("url_hash", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("published_at", sa.String(), nullable=True),
        sa.Column("scraped_at", sa.String(), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("correlation_id", UUID(as_uuid=True), nullable=True),
        sa.Column("topic_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "search_tsv",
            sa.Text(),  # simplified placeholder
            sa.Computed(
                "to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))",
                persisted=True,
            ),
            nullable=True,
        ),
    )
    op.create_index("idx_articles_source", "articles", ["source"])
    op.create_index("idx_articles_scraped_at", "articles", ["scraped_at"])
    op.create_index("idx_articles_topic_id", "articles", ["topic_id"])
