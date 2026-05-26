"""Retriever — retrieves relevant articles/chunks from PostgreSQL.

Spec reference: specs/rag-pipeline.md — Retriever

Phase 1: full-text search via search_tsv (generated tsvector column) + GIN index.
Phase 2: hybrid (dense + sparse) with RRF fusion.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class Retriever:
    """Retrieves ranked articles using PostgreSQL full-text search.

    Uses the ``search_tsv`` stored generated column with a GIN index
    for efficient full-text search. Phase 2 will add dense vector
    search and RRF fusion.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def search(
        self, query: str, limit: int = 10, topic_id: str | None = None
    ) -> list[dict]:
        """Full-text search on articles using the search_tsv GIN-indexed column.

        Args:
            query: Search query string.
            limit: Max results.
            topic_id: Optional topic filter.

        Returns:
            List of article dicts with id, title, content, rank.
        """
        params: dict = {"query": query, "limit": limit}
        topic_filter = ""
        if topic_id is not None:
            topic_filter = "AND topic_id = :topic_id"
            params["topic_id"] = topic_id

        sql = text(f"""
            SELECT id, title, content,
                   ts_rank(search_tsv, plainto_tsquery('english', :query)) AS rank
            FROM articles
            WHERE search_tsv @@ plainto_tsquery('english', :query)
              {topic_filter}
            ORDER BY rank DESC
            LIMIT :limit
        """)

        result = await self.db.execute(sql, params)
        rows = result.mappings().all()
        # Normalize ts_rank (unbounded) to 0-1 using rank/(rank+1)
        return [
            {**dict(row), "rank": row["rank"] / (row["rank"] + 1) if row["rank"] else 0.0}
            for row in rows
        ]
