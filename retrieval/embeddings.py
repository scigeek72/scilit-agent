"""
retrieval/embeddings.py — Provider-agnostic embedding wrapper.

Delegates to get_embeddings() from llm_provider.py so the embedding
provider (OpenAI or local HuggingFace) is controlled entirely by config.

Usage:
    from retrieval.embeddings import EmbeddingModel
    model = EmbeddingModel()
    vecs = model.embed_texts(["text one", "text two"])
    qvec = model.embed_query("my query")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Batch size for embedding calls — keeps memory bounded for large doc sets
_BATCH_SIZE = 64


class EmbeddingModel:
    """Lazy-initialised wrapper around the configured embedding provider."""

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from llm_provider import get_embeddings
            self._model = get_embeddings()
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents. Returns one vector per text.
        Processes in batches to avoid OOM on large corpora.
        """
        if not texts:
            return []
        model = self._get_model()
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            results.extend(model.embed_documents(batch))
        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self._get_model().embed_query(query)

    def dimension(self) -> int | None:
        """Return embedding dimension, or None if not yet initialised."""
        if self._model is None:
            return None
        sample = self.embed_query("test")
        return len(sample)


# Module-level singleton — shared across all callers
_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Return the shared EmbeddingModel instance (lazy init)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
