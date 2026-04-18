"""
retrieval/hybrid_search.py — Reciprocal Rank Fusion of BM25 + semantic search.

RRF formula: score(d) = Σ  1 / (k + rank_i(d))
  where k=60 (standard), rank_i is the 1-based position in each ranked list.

Weights from config (BM25_WEIGHT, SEMANTIC_WEIGHT) bias the fusion toward
one signal.  Weighted RRF multiplies each system's contribution by its weight
before summing:
  score(d) = BM25_WEIGHT * (1/(k+rank_bm25)) + SEMANTIC_WEIGHT * (1/(k+rank_sem))

Usage:
    from retrieval.hybrid_search import hybrid_search
    results = hybrid_search("attention mechanism", top_k=20)
"""

from __future__ import annotations

import logging

from config import Config

logger = logging.getLogger(__name__)

_RRF_K = 60   # standard constant; high values reduce the impact of top ranks


def hybrid_search(
    query: str,
    top_k: int | None = None,
    year_filter: int | None = None,
    source_filter: str | None = None,
    *,
    bm25_index=None,
    vector_store=None,
    embedding_model=None,
) -> list[dict]:
    """
    Run BM25 + semantic search and fuse with Reciprocal Rank Fusion.

    Dependencies (bm25_index, vector_store, embedding_model) are injected
    for testability; if None, the module-level singletons are used.

    Returns a list of dicts with all chunk metadata plus a 'score' key
    (fused RRF score, higher = more relevant).
    """
    top_k = top_k or Config.RETRIEVAL_K

    # Resolve singletons lazily
    if bm25_index is None:
        from retrieval.bm25_index import get_bm25_index
        bm25_index = get_bm25_index()
    if vector_store is None:
        from retrieval.vector_store import get_vector_store
        vector_store = get_vector_store()
    if embedding_model is None:
        from retrieval.embeddings import get_embedding_model
        embedding_model = get_embedding_model()

    # Run both searches in parallel (enough results to rank before trimming)
    fetch_k = max(top_k * 2, 50)

    bm25_hits = bm25_index.search(
        query, top_k=fetch_k,
        year_filter=year_filter, source_filter=source_filter,
    )
    query_embedding = embedding_model.embed_query(query)
    semantic_hits = vector_store.search(
        query_embedding, top_k=fetch_k,
        year_filter=year_filter, source_filter=source_filter,
    )

    fused = _rrf_fuse(bm25_hits, semantic_hits, top_k=top_k)

    logger.debug(
        "hybrid_search: bm25=%d, semantic=%d, fused=%d results",
        len(bm25_hits), len(semantic_hits), len(fused),
    )
    return fused


def _rrf_fuse(
    bm25_hits: list[dict],
    semantic_hits: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Fuse two ranked lists with weighted RRF.

    Uses chunk_id as the document identifier.  Documents that appear in
    only one list still get a score from that list alone.
    """
    bm25_weight = Config.BM25_WEIGHT
    sem_weight  = Config.SEMANTIC_WEIGHT

    # Build rank maps: chunk_id → 1-based rank
    bm25_rank = {h["chunk_id"]: i + 1 for i, h in enumerate(bm25_hits)}
    sem_rank  = {h["chunk_id"]: i + 1 for i, h in enumerate(semantic_hits)}

    # Collect all unique chunk_ids and the best metadata for each
    all_ids: dict[str, dict] = {}
    for h in bm25_hits:
        all_ids[h["chunk_id"]] = h
    for h in semantic_hits:
        all_ids.setdefault(h["chunk_id"], h)

    # Compute fused scores
    scored: list[tuple[float, dict]] = []
    for chunk_id, doc in all_ids.items():
        rrf_bm25 = bm25_weight / (_RRF_K + bm25_rank[chunk_id]) if chunk_id in bm25_rank else 0.0
        rrf_sem  = sem_weight  / (_RRF_K + sem_rank[chunk_id])  if chunk_id in sem_rank  else 0.0
        fused_score = rrf_bm25 + rrf_sem
        scored.append((fused_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {**doc, "score": score}
        for score, doc in scored[:top_k]
    ]
