"""
retrieval/bm25_index.py — BM25 keyword index with persistence.

Uses rank-bm25 (BM25Okapi).  The index and its document store are
serialised to disk with pickle so they survive process restarts.

Usage:
    from retrieval.bm25_index import BM25Index
    idx = BM25Index()
    idx.add_chunks(chunks)
    idx.save()
    results = idx.search("transformer attention", top_k=20)
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

from config import Config

if TYPE_CHECKING:
    from retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

_INDEX_FILE = "bm25.pkl"
_DOCS_FILE  = "docs.pkl"

# Simple tokeniser: lowercase, split on non-alphanumeric
_TOK_RE = re.compile(r"[^a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return [t for t in _TOK_RE.split(text.lower()) if t]


class BM25Index:
    """BM25 index over paper chunks, with load/save support."""

    def __init__(self, index_dir: str | None = None) -> None:
        self._dir = Path(index_dir or str(Config.bm25_index_dir()))
        self._bm25 = None          # rank_bm25.BM25Okapi — lazy
        self._docs: list[dict] = []  # parallel list of chunk dicts (text + meta)
        self._dirty = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list["Chunk"]) -> None:
        """
        Add chunks to the index.  Existing chunks for the same paper_id
        are replaced (delete-then-add) to keep the index consistent.
        """
        if not chunks:
            return

        paper_id = chunks[0]["paper_id"]
        # Remove old entries for this paper
        self._docs = [d for d in self._docs if d.get("paper_id") != paper_id]

        for chunk in chunks:
            self._docs.append({
                "chunk_id":        chunk["chunk_id"],
                "text":            chunk["text"],
                "paper_id":        chunk["paper_id"],
                "title":           chunk["title"],
                "authors":         chunk["authors"],
                "year":            chunk["year"],
                "source":          chunk["source"],
                "section_heading": chunk["section_heading"],
            })

        self._bm25 = None    # invalidate — rebuilt on next search
        self._dirty = True
        logger.debug("BM25: added %d chunks for %s (total %d)", len(chunks), paper_id, len(self._docs))

    def delete_paper(self, paper_id: str) -> None:
        before = len(self._docs)
        self._docs = [d for d in self._docs if d.get("paper_id") != paper_id]
        if len(self._docs) != before:
            self._bm25 = None
            self._dirty = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        year_filter: int | None = None,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        BM25 keyword search.  Returns up to top_k results with a 'score' key.
        """
        if not self._docs:
            return []

        bm25 = self._get_bm25()
        tokens = _tokenise(query)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)

        # Apply metadata filters
        candidates = []
        for i, doc in enumerate(self._docs):
            if year_filter is not None and doc.get("year", 0) < year_filter:
                continue
            if source_filter and doc.get("source", "") != source_filter:
                continue
            candidates.append((float(scores[i]), i))

        candidates.sort(reverse=True)
        top = candidates[:top_k]

        return [
            {**self._docs[i], "score": score}
            for score, i in top
            if score > 0
        ]

    def count(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist index and documents to disk."""
        if not self._dirty:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        bm25 = self._get_bm25()
        with open(self._dir / _INDEX_FILE, "wb") as f:
            pickle.dump(bm25, f)
        with open(self._dir / _DOCS_FILE, "wb") as f:
            pickle.dump(self._docs, f)
        self._dirty = False
        logger.debug("BM25 index saved to %s (%d docs)", self._dir, len(self._docs))

    def load(self) -> bool:
        """Load index from disk. Returns True if loaded successfully."""
        idx_path = self._dir / _INDEX_FILE
        docs_path = self._dir / _DOCS_FILE
        if not idx_path.exists() or not docs_path.exists():
            return False
        try:
            with open(idx_path, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(docs_path, "rb") as f:
                self._docs = pickle.load(f)
            self._dirty = False
            logger.debug("BM25 index loaded from %s (%d docs)", self._dir, len(self._docs))
            return True
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s", exc)
            self._bm25 = None
            self._docs = []
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            corpus = [_tokenise(d["text"]) for d in self._docs]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25


# Module-level singleton
_bm25_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
        _bm25_index.load()   # no-op if file doesn't exist yet
    return _bm25_index
