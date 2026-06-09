"""Toolbox service — handles chunk storage, search, and chat.

Orchestrates article upserts, chunk persistence, hybrid retrieval, and LLM calls.
Spec reference: specs/toolbox-api.md, specs/rag-pipeline.md
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import HTTPException
from pgvector import SparseVector
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_plugin.config import settings
from chatbot_plugin.contracts import (
    ArticleCitation,
    ChatRequest,
    ChatResponse,
    ChunkResult,
    SearchRequest,
    SearchResponse,
    StoreChunksRequest,
    StoreChunksResponse,
)
from chatbot_plugin.models import Article, ArticleChunk


class ToolboxService:
    """Handles article metadata storage, chunk persistence, search, and chat."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Storage ──

    async def store_chunks(self, request: StoreChunksRequest) -> StoreChunksResponse:
        """Upsert an article and store its chunks.

        If the article already exists, its metadata is updated and all
        existing chunks are replaced.
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
            sparse = None
            if chunk_data.sparse_vector:
                sparse = SparseVector(chunk_data.sparse_vector, settings.sparse_dimension)

            chunk = ArticleChunk(
                article_id=article_id,
                chunk_index=chunk_data.chunk_index,
                content=chunk_data.content,
                dense_vector=chunk_data.dense_vector,
                sparse_vector=sparse,
            )
            self.db.add(chunk)

        await self.db.commit()

        return StoreChunksResponse(
            stored=len(request.chunks),
            article_id=request.article.id,
        )

    # ── Search ──

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Hybrid dense + sparse search with RRF fusion."""
        from chatbot_plugin.embedding import embed_query

        dense_vec, sparse_weights = embed_query(request.query)
        sparse_vec = SparseVector(sparse_weights, settings.sparse_dimension)

        candidates = settings.search_candidates
        k = settings.rrf_k

        # Dense candidates
        dense_stmt = (
            select(
                ArticleChunk.id.label("chunk_id"),
                ArticleChunk.article_id,
                ArticleChunk.chunk_index,
                ArticleChunk.content,
                Article.title,
                Article.url,
            )
            .join(Article, ArticleChunk.article_id == Article.id)
            .where(ArticleChunk.dense_vector.isnot(None))
            .order_by(ArticleChunk.dense_vector.cosine_distance(dense_vec))
            .limit(candidates)
        )
        dense_result = await self.db.execute(dense_stmt)
        dense_rows = dense_result.all()

        # Sparse candidates
        sparse_stmt = (
            select(
                ArticleChunk.id.label("chunk_id"),
                ArticleChunk.article_id,
                ArticleChunk.chunk_index,
                ArticleChunk.content,
                Article.title,
                Article.url,
            )
            .join(Article, ArticleChunk.article_id == Article.id)
            .where(ArticleChunk.sparse_vector.isnot(None))
            .order_by(ArticleChunk.sparse_vector.max_inner_product(sparse_vec))
            .limit(candidates)
        )
        sparse_result = await self.db.execute(sparse_stmt)
        sparse_rows = sparse_result.all()

        # RRF fusion
        chunk_scores: dict[str, tuple[float, Any]] = {}

        for rank, row in enumerate(dense_rows, start=1):
            chunk_id = str(row.chunk_id)
            chunk_scores[chunk_id] = (1.0 / (k + rank), row)

        for rank, row in enumerate(sparse_rows, start=1):
            chunk_id = str(row.chunk_id)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] = (
                    chunk_scores[chunk_id][0] + 1.0 / (k + rank),
                    chunk_scores[chunk_id][1],
                )
            else:
                chunk_scores[chunk_id] = (1.0 / (k + rank), row)

        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )[:request.top_k]

        chunks = [
            ChunkResult(
                chunk_id=chunk_id,
                article_id=str(row.article_id),
                article_title=row.title,
                article_url=row.url,
                chunk_index=row.chunk_index,
                content=row.content,
                score=round(score, 6),
            )
            for chunk_id, (score, row) in sorted_chunks
        ]

        return SearchResponse(chunks=chunks)

    # ── Chat ──

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Chat with RAG context.

        1. Search for relevant chunks using the message as query.
        2. Assemble context from top chunks.
        3. Call LLM with system prompt + context + user message.
        """
        search_result = await self.search(
            SearchRequest(query=request.message, top_k=settings.max_context_chunks)
        )

        if not search_result.chunks:
            return ChatResponse(
                reply="I couldn't find any relevant context to answer your question.",
                articles_used=[],
                chunks=[],
            )

        # Assemble context
        context_parts = []
        for chunk in search_result.chunks:
            source = chunk.article_title or "Unknown source"
            context_parts.append(f"[source: {source}]\n{chunk.content}")
        context = "\n\n".join(context_parts)

        reply = await self._call_llm(context, request.message)

        # Deduplicate articles_used
        seen_ids: set[str] = set()
        articles_used: list[ArticleCitation] = []
        for chunk in search_result.chunks:
            if chunk.article_id not in seen_ids:
                seen_ids.add(chunk.article_id)
                articles_used.append(
                    ArticleCitation(
                        id=chunk.article_id,
                        title=chunk.article_title,
                        url=chunk.article_url,
                    )
                )

        return ChatResponse(
            reply=reply,
            articles_used=articles_used,
            chunks=search_result.chunks,
        )

    async def _call_llm(self, context: str, question: str) -> str:
        """Call LLM for chat. Tries Anthropic first, falls back to Gemini."""
        system = (
            "You are a helpful research assistant. Answer the user's question "
            "using only the provided context. Cite sources using the [source: Title] "
            "annotations already present in the context. If the context does not "
            "contain enough information, say so clearly."
        )
        user_prompt = f"{context}\n\nQuestion: {question}"

        # Try Anthropic first
        if settings.llm_api_key:
            try:
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=settings.llm_api_key)
                response = await client.messages.create(
                    model=settings.llm_model,
                    max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except Exception:
                pass  # Fallback to Gemini

        # Fallback to Gemini, or return raw context if no LLM keys configured
        if settings.gemini_api_key:
            try:
                return await self._call_gemini(system, user_prompt)
            except Exception:
                pass  # Gemini failed (unreachable, rate limited, etc.)

        return (
            "[No LLM configured — returning raw retrieved context]\n\n"
            f"{user_prompt}\n\n"
            "[Set CHATBOT_LLM_API_KEY (Anthropic) or CHATBOT_GEMINI_API_KEY "
            "to enable LLM-generated responses.]"
        )

    async def _call_gemini(self, system: str, prompt: str) -> str:
        """Call Google Gemini REST API."""
        import httpx

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_model}:generateContent"
        )
        params = {"key": settings.gemini_api_key}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {"maxOutputTokens": 2048},
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, params=params, json=payload)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {resp.status_code} {resp.text}",
                )
            data = resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected Gemini response: {data}",
                ) from e
