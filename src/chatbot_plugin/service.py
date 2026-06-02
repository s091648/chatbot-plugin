"""Toolbox service — handles chunk storage.

Orchestrates article upserts and chunk persistence.
Spec reference: specs/toolbox-api.md
"""

from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_plugin.config import settings
from chatbot_plugin.contracts import (
    StoreChunksRequest,
    StoreChunksResponse,
)
from chatbot_plugin.models import Article, ArticleChunk


class ToolboxService:
    """Handles article metadata storage and chunk persistence."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def store_chunks(self, request: StoreChunksRequest) -> StoreChunksResponse:
        """Upsert an article and store its chunks.

        If the article already exists, its metadata is updated and all
        existing chunks are replaced.

        Args:
            request: StoreChunksRequest with article info and chunk data.

        Returns:
            StoreChunksResponse with count of stored chunks.

        Raises:
            HTTPException: 400 if dense_vector dimension does not match settings.
        """
        expected_dim = settings.embedding_dimension
        for chunk in request.chunks:
            if len(chunk.dense_vector) != expected_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Dense vector dimension mismatch at chunk_index={chunk.chunk_index}: "
                        f"expected {expected_dim}, got {len(chunk.dense_vector)}"
                    ),
                )

        article_id = UUID(request.article.id)

        # Check for existing article
        result = await self.db.execute(select(Article).where(Article.id == article_id))
        existing = result.scalar_one_or_none()

        if existing is not None:
            # Update metadata
            existing.url = request.article.url
            existing.title = request.article.title
            existing.source = request.article.source
            existing.metadata_ = request.article.metadata

            # Delete old chunks
            await self.db.execute(
                delete(ArticleChunk).where(ArticleChunk.article_id == article_id)
            )
        else:
            # Create new article
            new_article = Article(
                id=article_id,
                url=request.article.url,
                title=request.article.title,
                source=request.article.source,
                metadata_=request.article.metadata,
            )
            self.db.add(new_article)
            await self.db.flush()

        # Insert new chunks
        for chunk_data in request.chunks:
            chunk = ArticleChunk(
                article_id=article_id,
                chunk_index=chunk_data.chunk_index,
                content=chunk_data.content,
                dense_vector=chunk_data.dense_vector,
                sparse_vector=chunk_data.sparse_vector,
            )
            self.db.add(chunk)

        await self.db.commit()

        return StoreChunksResponse(
            stored=len(request.chunks),
            article_id=request.article.id,
        )
