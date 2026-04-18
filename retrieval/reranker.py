"""
retrieval/reranker.py — Cross-encoder reranker.

Uses sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2, a fast
and accurate passage reranker trained on MS MARCO.

The reranker sees (query, passage) pairs and assigns a relevance score,
providing finer-grained ranking than the bi-encoder retrieval stage.

Model is downloaded once on first use (~80 MB) and cached by
sentence-transformers in ~/.cache/torch/sentence_transformers/.

Usage:
    from retrieval.reranker import Reranker
    reranker = Reranker()
    top_chunks = reranker.rerank(query, chunks, top_k=5)
"""

from __future__ import annotations

import logging

from config import Config

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_LENGTH = 512    # token limit for the cross-encoder

# Module-level import so @patch("retrieval.reranker.CrossEncoder") works in tests
try:
    from sentence_transformers import CrossEncoder as CrossEncoder
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, Exception):
    CrossEncoder = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


class Reranker:
    """Lazy-loaded cross-encoder reranker."""

    def __init__(self, model_name: str = _RERANKER_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            if CrossEncoder is None:
                raise ImportError("sentence_transformers is not available")
            self._model = CrossEncoder(
                self._model_name,
                max_length=_MAX_LENGTH,
            )
            logger.debug("Reranker model '%s' loaded", self._model_name)
        return self._model

    def is_available(self) -> bool:
        """Return True if sentence_transformers CrossEncoder is importable."""
        return CrossEncoder is not None

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank chunks using the cross-encoder.

        Returns the top_k most relevant chunks (or all if top_k is None),
        each with an updated 'score' key (raw cross-encoder logit).
        """
        top_k = top_k or Config.RERANK_TOP_K
        if not chunks:
            return []

        model = self._get_model()
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        return [
            {**chunk, "score": float(score)}
            for score, chunk in ranked[:top_k]
        ]


# Module-level singleton
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
