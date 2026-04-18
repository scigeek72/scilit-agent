"""
tests/test_retrieval.py — Unit tests for Phase 4 retrieval layer.

All external services (ChromaDB, sentence-transformers, LLM) are mocked.
No real models, no real DB, no real LLM calls required.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_PARSED_PAPER = {
    "paper_id": "arxiv:2301.00001",
    "title": "Attention Is All You Need",
    "authors": ["Vaswani, A.", "Shazeer, N."],
    "abstract": "We propose the Transformer, a model architecture eschewing recurrence.",
    "year": 2017,
    "source": "arxiv",
    "sections": [
        {"heading": "Introduction", "text": "Recurrent neural networks have been firmly established.", "level": 1},
        {"heading": "Model Architecture", "text": "The Transformer uses stacked self-attention layers.", "level": 1},
    ],
    "references": [],
    "figures": [],
    "tables": [],
    "equations": [],
    "parser_used": "grobid",
    "math_fraction": 0.05,
    "is_abstract_only": False,
}

SAMPLE_CHUNK = {
    "chunk_id": "arxiv:2301.00001_0",
    "text": "We propose the Transformer, a model architecture eschewing recurrence.",
    "paper_id": "arxiv:2301.00001",
    "title": "Attention Is All You Need",
    "authors": "Vaswani, A. | Shazeer, N.",
    "year": 2017,
    "source": "arxiv",
    "chunk_index": 0,
    "section_heading": "Abstract",
}


# ===========================================================================
# Chunker
# ===========================================================================

class TestChunker:

    def test_chunk_produces_at_least_abstract(self):
        from retrieval.chunker import chunk_parsed_paper
        chunks = chunk_parsed_paper(SAMPLE_PARSED_PAPER)
        assert len(chunks) >= 1

    def test_chunk_ids_are_unique(self):
        from retrieval.chunker import chunk_parsed_paper
        chunks = chunk_parsed_paper(SAMPLE_PARSED_PAPER)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_populated(self):
        from retrieval.chunker import chunk_parsed_paper
        chunks = chunk_parsed_paper(SAMPLE_PARSED_PAPER)
        for c in chunks:
            assert c["paper_id"] == SAMPLE_PARSED_PAPER["paper_id"]
            assert c["year"] == SAMPLE_PARSED_PAPER["year"]
            assert c["source"] == SAMPLE_PARSED_PAPER["source"]
            assert isinstance(c["text"], str) and c["text"]

    def test_abstract_only_paper_produces_chunks(self):
        from retrieval.chunker import chunk_parsed_paper
        paper = {**SAMPLE_PARSED_PAPER, "is_abstract_only": True, "sections": []}
        chunks = chunk_parsed_paper(paper)
        assert len(chunks) >= 1
        assert chunks[0]["section_heading"] == "Abstract"

    def test_paper_with_no_content_produces_no_chunks(self):
        from retrieval.chunker import chunk_parsed_paper
        paper = {**SAMPLE_PARSED_PAPER, "abstract": "", "sections": []}
        chunks = chunk_parsed_paper(paper)
        assert chunks == []

    def test_authors_joined_with_pipe(self):
        from retrieval.chunker import chunk_parsed_paper
        chunks = chunk_parsed_paper(SAMPLE_PARSED_PAPER)
        assert "|" in chunks[0]["authors"] or len(SAMPLE_PARSED_PAPER["authors"]) == 1

    def test_sliding_window_respects_chunk_size(self):
        from retrieval.chunker import _sliding_window
        words = "word " * 600   # 600 words
        windows = _sliding_window(words.strip(), size=512, overlap=50)
        for w in windows:
            assert len(w.split()) <= 512

    def test_sliding_window_overlap(self):
        from retrieval.chunker import _sliding_window
        text = " ".join(f"w{i}" for i in range(100))
        windows = _sliding_window(text, size=60, overlap=10)
        assert len(windows) >= 2
        # First word of second window should appear near end of first window
        first_end_words = set(windows[0].split()[-15:])
        second_start_words = set(windows[1].split()[:15])
        assert first_end_words & second_start_words  # overlap exists

    def test_short_text_returns_single_window(self):
        from retrieval.chunker import _sliding_window
        text = "only a few words"
        windows = _sliding_window(text, size=512, overlap=50)
        assert len(windows) == 1
        assert windows[0] == text


# ===========================================================================
# BM25Index
# ===========================================================================

class TestBM25Index:

    def _make_chunks(self, n=3, paper_id="arxiv:test_001"):
        texts = [
            "transformer attention mechanism self-attention",
            "recurrent neural network sequence model",
            "convolutional neural network image classification",
        ]
        return [
            {
                "chunk_id": f"{paper_id}_{i}",
                "text": texts[i % len(texts)],
                "paper_id": paper_id,
                "title": "Test Paper",
                "authors": "Author A",
                "year": 2023,
                "source": "arxiv",
                "chunk_index": i,
                "section_heading": "Section",
            }
            for i in range(n)
        ]

    def test_add_and_search(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index(index_dir=tempfile.mkdtemp())
        idx.add_chunks(self._make_chunks())
        results = idx.search("transformer attention")
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_search_returns_relevant_result_first(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index(index_dir=tempfile.mkdtemp())
        idx.add_chunks(self._make_chunks())
        results = idx.search("transformer attention")
        assert "transformer" in results[0]["text"].lower()

    def test_search_with_source_filter(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index(index_dir=tempfile.mkdtemp())
        idx.add_chunks(self._make_chunks())
        results = idx.search("transformer", source_filter="pubmed")
        assert results == []   # all chunks are source=arxiv

    def test_search_with_year_filter(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index(index_dir=tempfile.mkdtemp())
        idx.add_chunks(self._make_chunks())
        results = idx.search("transformer", year_filter=2030)
        assert results == []   # all chunks are year=2023

    def test_delete_paper(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index(index_dir=tempfile.mkdtemp())
        idx.add_chunks(self._make_chunks(paper_id="arxiv:paper_a"))
        idx.add_chunks(self._make_chunks(paper_id="arxiv:paper_b"))
        idx.delete_paper("arxiv:paper_a")
        results = idx.search("transformer")
        assert all(r["paper_id"] != "arxiv:paper_a" for r in results)

    def test_save_and_load(self):
        from retrieval.bm25_index import BM25Index
        tmpdir = tempfile.mkdtemp()
        idx = BM25Index(index_dir=tmpdir)
        idx.add_chunks(self._make_chunks())
        idx.save()

        idx2 = BM25Index(index_dir=tmpdir)
        loaded = idx2.load()
        assert loaded is True
        results = idx2.search("transformer")
        assert len(results) > 0

    def test_empty_query_returns_empty(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index(index_dir=tempfile.mkdtemp())
        idx.add_chunks(self._make_chunks())
        results = idx.search("")
        assert results == []

    def test_search_on_empty_index_returns_empty(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index(index_dir=tempfile.mkdtemp())
        results = idx.search("transformer")
        assert results == []

    def test_add_chunks_same_paper_replaces_old(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index(index_dir=tempfile.mkdtemp())
        idx.add_chunks(self._make_chunks(n=3))
        assert idx.count() == 3
        idx.add_chunks(self._make_chunks(n=2))   # same paper_id, fewer chunks
        assert idx.count() == 2


# ===========================================================================
# VectorStore (ChromaDB mocked)
# ===========================================================================

class TestVectorStore:

    def _make_mock_collection(self, count=5):
        col = MagicMock()
        col.count.return_value = count
        col.query.return_value = {
            "documents": [["chunk text one", "chunk text two"]],
            "metadatas": [
                [
                    {"chunk_id": "p_0", "paper_id": "arxiv:001", "title": "T",
                     "authors": "A", "year": 2023, "source": "arxiv",
                     "chunk_index": 0, "section_heading": "Intro"},
                    {"chunk_id": "p_1", "paper_id": "arxiv:001", "title": "T",
                     "authors": "A", "year": 2023, "source": "arxiv",
                     "chunk_index": 1, "section_heading": "Methods"},
                ]
            ],
            "distances": [[0.1, 0.3]],
        }
        return col

    def test_search_returns_results(self):
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        vs._collection = self._make_mock_collection()
        results = vs.search([0.1] * 10, top_k=2)
        assert len(results) == 2
        assert results[0]["score"] == pytest.approx(0.9)

    def test_search_on_empty_store_returns_empty(self):
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        vs._collection = self._make_mock_collection(count=0)
        results = vs.search([0.1] * 10)
        assert results == []

    def test_add_chunks_calls_upsert(self):
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        vs._collection = mock_col

        chunks = [SAMPLE_CHUNK]
        embeddings = [[0.1] * 10]
        vs.add_chunks(chunks, embeddings)
        mock_col.upsert.assert_called_once()

    def test_add_chunks_length_mismatch_raises(self):
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        vs._collection = MagicMock()
        with pytest.raises(ValueError):
            vs.add_chunks([SAMPLE_CHUNK], [[0.1], [0.2]])

    def test_delete_calls_collection_delete(self):
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        mock_col = MagicMock()
        vs._collection = mock_col
        vs.delete_paper("arxiv:001")
        mock_col.delete.assert_called_once_with(where={"paper_id": "arxiv:001"})

    def test_search_score_is_1_minus_distance(self):
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        vs._collection = self._make_mock_collection()
        results = vs.search([0.1] * 10, top_k=2)
        assert results[0]["score"] == pytest.approx(1 - 0.1)
        assert results[1]["score"] == pytest.approx(1 - 0.3)

    def test_where_clause_year_filter(self):
        from retrieval.vector_store import _build_where
        where = _build_where(year_filter=2020, source_filter=None)
        assert where == {"year": {"$gte": 2020}}

    def test_where_clause_source_filter(self):
        from retrieval.vector_store import _build_where
        where = _build_where(year_filter=None, source_filter="pubmed")
        assert where == {"source": "pubmed"}

    def test_where_clause_combined(self):
        from retrieval.vector_store import _build_where
        where = _build_where(year_filter=2020, source_filter="arxiv")
        assert where["$and"][0] == {"year": {"$gte": 2020}}
        assert where["$and"][1] == {"source": "arxiv"}

    def test_where_clause_none(self):
        from retrieval.vector_store import _build_where
        assert _build_where(None, None) is None


# ===========================================================================
# Hybrid search (RRF fusion)
# ===========================================================================

class TestHybridSearch:

    def _make_hits(self, ids, base_score=0.9):
        return [
            {
                "chunk_id": cid,
                "text": f"text {cid}",
                "paper_id": "p1",
                "title": "T",
                "authors": "A",
                "year": 2023,
                "source": "arxiv",
                "section_heading": "S",
                "score": base_score - i * 0.1,
            }
            for i, cid in enumerate(ids)
        ]

    def test_rrf_fuse_top_results(self):
        from retrieval.hybrid_search import _rrf_fuse
        bm25 = self._make_hits(["a", "b", "c", "d"])
        sem  = self._make_hits(["b", "a", "c", "e"])
        fused = _rrf_fuse(bm25, sem, top_k=3)
        assert len(fused) == 3
        # "a" and "b" appear in both lists near the top — should lead
        top_ids = {r["chunk_id"] for r in fused[:2]}
        assert "a" in top_ids or "b" in top_ids

    def test_rrf_fuse_document_in_one_list_only(self):
        from retrieval.hybrid_search import _rrf_fuse
        bm25 = self._make_hits(["a", "b"])
        sem  = self._make_hits(["c", "d"])
        fused = _rrf_fuse(bm25, sem, top_k=4)
        ids = {r["chunk_id"] for r in fused}
        assert ids == {"a", "b", "c", "d"}

    def test_hybrid_search_calls_both_backends(self):
        from retrieval.hybrid_search import hybrid_search

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = self._make_hits(["a", "b"])
        mock_vs = MagicMock()
        mock_vs.search.return_value = self._make_hits(["b", "c"])
        mock_vs.count.return_value = 5
        mock_em = MagicMock()
        mock_em.embed_query.return_value = [0.1] * 10

        results = hybrid_search(
            "query",
            top_k=5,
            bm25_index=mock_bm25,
            vector_store=mock_vs,
            embedding_model=mock_em,
        )
        mock_bm25.search.assert_called_once()
        mock_em.embed_query.assert_called_once_with("query")
        mock_vs.search.assert_called_once()
        assert len(results) <= 5

    def test_hybrid_search_empty_stores_return_empty(self):
        from retrieval.hybrid_search import hybrid_search

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = []
        mock_vs = MagicMock()
        mock_vs.search.return_value = []
        mock_vs.count.return_value = 0
        mock_em = MagicMock()
        mock_em.embed_query.return_value = [0.1] * 10

        results = hybrid_search(
            "query",
            bm25_index=mock_bm25,
            vector_store=mock_vs,
            embedding_model=mock_em,
        )
        assert results == []


# ===========================================================================
# Reranker
# ===========================================================================

class TestReranker:

    def _make_chunks(self):
        texts = ["attention mechanism in transformers",
                 "recurrent networks are sequential",
                 "convolutional filters for images"]
        return [
            {**SAMPLE_CHUNK, "chunk_id": f"c{i}", "text": t, "score": 0.5}
            for i, t in enumerate(texts)
        ]

    @patch("retrieval.reranker.CrossEncoder")
    def test_rerank_reorders_by_score(self, mock_ce_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.9, 0.5]
        mock_ce_cls.return_value = mock_model

        from retrieval.reranker import Reranker
        r = Reranker()
        r._model = mock_model

        chunks = self._make_chunks()
        result = r.rerank("transformer attention", chunks, top_k=3)
        assert result[0]["score"] == pytest.approx(0.9)
        assert result[1]["score"] == pytest.approx(0.5)
        assert result[2]["score"] == pytest.approx(0.2)

    @patch("retrieval.reranker.CrossEncoder")
    def test_rerank_top_k_limits_output(self, mock_ce_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.9, 0.5]
        mock_ce_cls.return_value = mock_model

        from retrieval.reranker import Reranker
        r = Reranker()
        r._model = mock_model

        chunks = self._make_chunks()
        result = r.rerank("query", chunks, top_k=2)
        assert len(result) == 2

    def test_rerank_empty_chunks_returns_empty(self):
        from retrieval.reranker import Reranker
        r = Reranker()
        result = r.rerank("query", [], top_k=5)
        assert result == []

    def test_is_available_reflects_import(self):
        from retrieval.reranker import Reranker, _SENTENCE_TRANSFORMERS_AVAILABLE
        r = Reranker()
        assert r.is_available() is _SENTENCE_TRANSFORMERS_AVAILABLE


# ===========================================================================
# HyDE
# ===========================================================================

class TestHyDE:

    def test_returns_string(self):
        from retrieval.hyde import generate_hyde_document
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="A hypothetical excerpt about attention.")
        result = generate_hyde_document("What is self-attention?", llm=mock_llm)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_query_on_llm_failure(self):
        from retrieval.hyde import generate_hyde_document
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        result = generate_hyde_document("What is self-attention?", llm=mock_llm)
        assert result == "What is self-attention?"

    def test_returns_query_on_empty_response(self):
        from retrieval.hyde import generate_hyde_document
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        result = generate_hyde_document("test query", llm=mock_llm)
        assert result == "test query"

    def test_uses_query_in_prompt(self):
        from retrieval.hyde import _HYDE_PROMPT
        # Verify the prompt template contains the query placeholder
        assert "{query}" in _HYDE_PROMPT
        # Verify the formatted prompt contains the actual query
        formatted = _HYDE_PROMPT.format(query="CRISPR delivery mechanisms")
        assert "CRISPR delivery mechanisms" in formatted


# ===========================================================================
# Retrieval tools (integration-level, all mocked)
# ===========================================================================

class TestRetrievalTools:

    @patch("retrieval.embeddings.get_embedding_model")
    @patch("retrieval.vector_store.get_vector_store")
    @patch("retrieval.bm25_index.get_bm25_index")
    def test_index_paper_returns_chunk_count(self, mock_bm25, mock_vs, mock_em):
        from tools.retrieval_tools import index_paper

        mock_em.return_value.embed_texts.return_value = [[0.1] * 10] * 20
        mock_vs.return_value.add_chunks = MagicMock()
        mock_bm25.return_value.add_chunks = MagicMock()
        mock_bm25.return_value.save = MagicMock()

        n = index_paper(SAMPLE_PARSED_PAPER)
        assert n > 0

    @patch("retrieval.embeddings.get_embedding_model")
    @patch("retrieval.vector_store.get_vector_store")
    def test_search_vector_db_calls_store(self, mock_vs, mock_em):
        from tools.retrieval_tools import search_vector_db

        mock_em.return_value.embed_query.return_value = [0.1] * 10
        mock_vs.return_value.search.return_value = [SAMPLE_CHUNK]

        results = search_vector_db("attention mechanism", top_k=5)
        assert results == [SAMPLE_CHUNK]
        mock_vs.return_value.search.assert_called_once()

    @patch("retrieval.hyde.generate_hyde_document")
    @patch("retrieval.hybrid_search.hybrid_search")
    def test_hybrid_search_calls_hyde_when_enabled(self, mock_hs, mock_hyde):
        from tools.retrieval_tools import hybrid_search as tool_hybrid

        mock_hyde.return_value = "hypothetical excerpt"
        mock_hs.return_value = []

        tool_hybrid("attention", use_hyde=True)
        mock_hyde.assert_called_once_with("attention")

    @patch("retrieval.reranker.get_reranker")
    def test_rerank_chunks_delegates_to_reranker(self, mock_get_reranker):
        from tools.retrieval_tools import rerank_chunks

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [SAMPLE_CHUNK]
        mock_get_reranker.return_value = mock_reranker

        result = rerank_chunks("query", [SAMPLE_CHUNK], top_k=1)
        assert result == [SAMPLE_CHUNK]
        mock_reranker.rerank.assert_called_once_with("query", [SAMPLE_CHUNK], top_k=1)

    @patch("retrieval.vector_store.get_vector_store")
    @patch("retrieval.bm25_index.get_bm25_index")
    def test_delete_paper_calls_both_stores(self, mock_bm25, mock_vs):
        from tools.retrieval_tools import delete_paper_from_index

        mock_vs.return_value.delete_paper = MagicMock()
        mock_bm25.return_value.delete_paper = MagicMock()
        mock_bm25.return_value.save = MagicMock()

        delete_paper_from_index("arxiv:001")
        mock_vs.return_value.delete_paper.assert_called_once_with("arxiv:001")
        mock_bm25.return_value.delete_paper.assert_called_once_with("arxiv:001")
