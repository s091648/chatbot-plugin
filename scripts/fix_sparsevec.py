#!/usr/bin/env python3
"""Fix existing databases: sparse_vector column type jsonb -> sparsevec.

Migration 002 originally created sparse_vector as JSONB (a bug in the migration).
This script safely converts existing data to the correct sparsevec(250002) type.
"""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from pgvector import SparseVector
from chatbot_plugin.config import settings


async def fix_sparse_vector() -> None:
    engine = create_async_engine(settings.database_url)

    async with engine.begin() as conn:
        # Check current type
        result = await conn.execute(text(
            "SELECT data_type "
            "FROM information_schema.columns "
            "WHERE table_name = 'article_chunks' AND column_name = 'sparse_vector'"
        ))
        row = result.fetchone()
        if row is None:
            print("Column sparse_vector not found — nothing to fix.")
            return

        current_type = row[0]

        if current_type == "sparsevec":
            print(f"sparse_vector is already {current_type} — nothing to fix.")
            return

        print(f"Current type: {current_type}. Converting to sparsevec(250002)...")

        # Ensure extension is available
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Drop any existing index on sparse_vector first
        await conn.execute(text("DROP INDEX IF EXISTS idx_chunks_sparse"))

        # Add a temporary sparsevec column
        await conn.execute(text(
            "ALTER TABLE article_chunks "
            "ADD COLUMN IF NOT EXISTS sparse_vector_new sparsevec(250002)"
        ))

        # Transfer data: read jsonb dicts and write as SparseVector
        result = await conn.execute(text(
            "SELECT id, sparse_vector FROM article_chunks WHERE sparse_vector IS NOT NULL"
        ))
        rows = result.fetchall()
        print(f"Converting {len(rows)} rows...")

        for chunk_id, sparse_data in rows:
            if sparse_data is None:
                continue
            # PostgreSQL sparsevec uses 1-based indices in text format:
            # '{1:val,2:val}/dims'.
            items = ",".join(f"{int(k) + 1}:{float(v)}" for k, v in sorted(sparse_data.items()))
            sparse_text = f"{{{items}}}/{settings.sparse_dimension}"
            await conn.execute(
                text("UPDATE article_chunks SET sparse_vector_new = text(:sv)::sparsevec(250002) WHERE id = :id"),
                {"sv": sparse_text, "id": chunk_id},
            )

        # Drop old column and rename
        await conn.execute(text("ALTER TABLE article_chunks DROP COLUMN sparse_vector"))
        await conn.execute(text("ALTER TABLE article_chunks RENAME COLUMN sparse_vector_new TO sparse_vector"))

        # Note: pgvector 0.8.0 does not provide sparsevec HNSW indexing.
        # A simple index on the column is sufficient for now.
        await conn.execute(text(
            "CREATE INDEX idx_chunks_sparse ON article_chunks (sparse_vector)"
        ))

        print("Done. sparse_vector is now sparsevec(250002).")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(fix_sparse_vector())
