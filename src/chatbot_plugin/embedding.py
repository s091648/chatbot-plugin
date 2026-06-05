"""Embedding module — lazy singleton BGE-M3 model.

Loads `FlagEmbedding.BGEM3FlagModel` on first use and caches it.
Produces dense (1024-dim) vectors and sparse (250002-dim) token weights.
"""

from __future__ import annotations

import structlog

from chatbot_plugin.config import settings

logger = structlog.get_logger()

# Lazy singleton
_model: "BGEM3FlagModel" | None = None


def _get_model() -> "BGEM3FlagModel":
    """Load and cache BGE-M3 model (CPU only)."""
    global _model
    if _model is None:
        logger.info("loading_embedding_model", model=settings.embedding_model)
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=False,
            device="cpu",
        )
        logger.info("embedding_model_loaded")
    return _model


def embed_query(text: str) -> tuple[list[float], dict[int, float]]:
    """Embed a single query text.

    Returns:
        (dense_vector, sparse_weights) where sparse_weights is
        {token_index: weight} as produced by BGE-M3.
    """
    model = _get_model()
    results = model.encode(
        [text],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense: list[float] = results["dense_vecs"][0].tolist()
    sparse_raw: dict = results["lexical_weights"][0]
    # Sparse keys are string token indices from FlagEmbedding
    sparse: dict[int, float] = {int(k): float(v) for k, v in sparse_raw.items()}
    return dense, sparse


def embed_texts(texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Embed multiple texts in a single batch (for indexing).

    Returns:
        (dense_vectors, sparse_weights_list)
    """
    model = _get_model()
    results = model.encode(
        texts,
        batch_size=16,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_list: list[list[float]] = [v.tolist() for v in results["dense_vecs"]]
    sparse_list: list[dict[int, float]] = [
        {int(k): float(v) for k, v in s.items()} for s in results["lexical_weights"]
    ]
    return dense_list, sparse_list


def get_sparse_vector(sparse_weights: dict[int, float]) -> "SparseVector":
    """Convert token-index weights to pgvector SparseVector."""
    from pgvector import SparseVector

    return SparseVector(sparse_weights, settings.sparse_dimension)
