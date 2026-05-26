"""Initial articles table with search_tsv and GIN index

Revision ID: 001
Revises: None
Create Date: 2026-05-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

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
            TSVECTOR,
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
    op.execute(
        "CREATE INDEX idx_articles_tsv ON articles USING gin(search_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_tsv")
    op.drop_index("idx_articles_topic_id", table_name="articles")
    op.drop_index("idx_articles_scraped_at", table_name="articles")
    op.drop_index("idx_articles_source", table_name="articles")
    op.drop_table("articles")
