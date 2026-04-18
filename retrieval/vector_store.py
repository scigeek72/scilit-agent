"""
retrieval/vector_store.py — ChromaDB persistent vector store wrapper.

One ChromaDB collection per topic (named "scilit_{topic_slug}").
Stores pre-computed embeddings alongside chunk metadata so the embedding
model can be swapped without rebuilding the DB.

Usage:
    from retrieval.vector_store import VectorStore
    vs = VectorStore()
    vs.add_chunks(chunks, embeddings)
    results = vs.search(query_embedding, top_k=20)
"""

from __future__ import annotations

import logging
from typing import Any

from config import Config
from retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class VectorStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(self, persist_dir: str | None = None) -> None:
        self._persist_dir = persist_dir or str(Config.vector_db_dir())
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            Config.ensure_dirs()
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            name = f"scilit_{Config.topic_slug()}"
            self._collection = self._client.get_or_create_collection(
                name=name,
                metadata=_COLLECTION_METADATA,
            )
            logger.debug(
                "ChromaDB collection '%s' opened at %s (%d docs)",
                name, self._persist_dir, self._collection.count(),
            )
        return self._collection

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Upsert chunks into the collection.
        Existing chunks with the same chunk_id are overwritten (idempotent).
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be same length"
            )

        col = self._get_collection()
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [_chunk_to_meta(c) for c in chunks]

        # ChromaDB upsert in batches of 500 (avoids gRPC size limits)
        batch = 500
        for i in range(0, len(ids), batch):
            col.upsert(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                embeddings=embeddings[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

        logger.debug("Upserted %d chunks for paper %s", len(chunks), chunks[0]["paper_id"])

    def delete_paper(self, paper_id: str) -> None:
        """Remove all chunks belonging to a paper."""
        col = self._get_collection()
        col.delete(where={"paper_id": paper_id})
        logger.debug("Deleted chunks for paper %s", paper_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        year_filter: int | None = None,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Semantic search.  Returns up to top_k results, each as a dict:
        {chunk_id, text, paper_id, title, authors, year, source,
         section_heading, score}
        """
        col = self._get_collection()
        if col.count() == 0:
            return []

        where = _build_where(year_filter, source_filter)
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        hits: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                **meta,
                "text": doc,
                "score": float(1 - dist),   # cosine distance → similarity
            })
        return hits

    def count(self) -> int:
        return self._get_collection().count()

    def paper_ids(self) -> list[str]:
        """Return deduplicated list of paper_ids in the store."""
        col = self._get_collection()
        if col.count() == 0:
            return []
        result = col.get(include=["metadatas"])
        seen: set[str] = set()
        ids: list[str] = []
        for meta in result["metadatas"]:
            pid = meta.get("paper_id", "")
            if pid and pid not in seen:
                seen.add(pid)
                ids.append(pid)
        return ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_to_meta(chunk: Chunk) -> dict:
    """Convert a Chunk to ChromaDB-compatible metadata (str/int/float/bool only)."""
    return {
        "chunk_id":        chunk["chunk_id"],
        "paper_id":        chunk["paper_id"],
        "title":           chunk["title"][:500],
        "authors":         chunk["authors"][:500],
        "year":            int(chunk["year"]),
        "source":          chunk["source"],
        "chunk_index":     int(chunk["chunk_index"]),
        "section_heading": chunk["section_heading"][:200],
    }


def _build_where(
    year_filter: int | None,
    source_filter: str | None,
) -> dict | None:
    conditions: list[dict] = []
    if year_filter is not None:
        conditions.append({"year": {"$gte": year_filter}})
    if source_filter:
        conditions.append({"source": source_filter})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# Module-level singleton
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
