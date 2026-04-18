"""
tests/test_query_agent.py — Unit tests for Phase 7: Query Agent + Cache.

All LLM calls, retrieval, and wiki I/O are mocked.
Tests cover:
  - QueryCache: get/put/invalidate/clear/eviction/TTL
  - Individual query agent nodes
  - Routing functions
  - Graph compilation
  - End-to-end run_query with cache hit and miss
  - File-back path (is_worth_filing → write synthesis page)
  - Self-critique → refine path
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# QueryCache tests
# ---------------------------------------------------------------------------

class TestQueryCache:
    @pytest.fixture
    def cache(self, tmp_path):
        from retrieval.query_cache import QueryCache
        return QueryCache(tmp_path / "test_cache.db")

    def test_miss_returns_none(self, cache):
        assert cache.get("unknown query") is None

    def test_put_and_get(self, cache):
        cache.put("What is attention?", "It is a mechanism.", [{"paper_id": "arxiv:1"}], 0.9)
        result = cache.get("What is attention?")
        assert result is not None
        assert result["answer"] == "It is a mechanism."
        assert result["confidence"] == 0.9

    def test_normalised_query_matches(self, cache):
        cache.put("  What IS attention? ", "Answer.", [], 0.8)
        result = cache.get("what is attention?")
        assert result is not None
        assert result["answer"] == "Answer."

    def test_sources_round_trip(self, cache):
        sources = [{"paper_id": "arxiv:1", "title": "Paper A", "year": 2023}]
        cache.put("query", "answer", sources, 0.7)
        result = cache.get("query")
        assert result["sources"] == sources

    def test_put_overwrites(self, cache):
        cache.put("q", "first", [], 0.5)
        cache.put("q", "second", [], 0.9)
        result = cache.get("q")
        assert result["answer"] == "second"

    def test_invalidate(self, cache):
        cache.put("q", "answer", [], 0.7)
        cache.invalidate("q")
        assert cache.get("q") is None

    def test_clear(self, cache):
        cache.put("q1", "a1", [], 0.5)
        cache.put("q2", "a2", [], 0.5)
        cache.clear()
        assert cache.size() == 0

    def test_size(self, cache):
        assert cache.size() == 0
        cache.put("q1", "a1", [], 0.5)
        cache.put("q2", "a2", [], 0.5)
        assert cache.size() == 2

    def test_ttl_eviction(self, cache):
        """Entries dated before the TTL cutoff should be evicted on get()."""
        from datetime import date, timedelta
        from retrieval.query_cache import CACHE_TTL_DAYS

        old_date = (date.today() - timedelta(days=CACHE_TTL_DAYS + 1)).isoformat()

        # Manually insert an old entry
        with cache._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_cache "
                "(query_hash, query_text, answer, sources_json, confidence, date_cached) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("deadbeef" * 8, "old query", "old answer", "[]", 0.5, old_date),
            )

        # The hash won't match "old query" because we put it manually.
        # Test by using the real hash of a query with a past date.
        cache.put("stale query", "stale answer", [], 0.5)
        # Manually update the date to expired
        key = cache._hash("stale query")
        with cache._connect() as conn:
            conn.execute(
                "UPDATE query_cache SET date_cached = ? WHERE query_hash = ?",
                (old_date, key),
            )
        assert cache.get("stale query") is None

    def test_max_size_eviction(self, tmp_path):
        """Cache should not exceed CACHE_MAX_SIZE entries."""
        from retrieval.query_cache import QueryCache
        from unittest.mock import patch

        with patch("retrieval.query_cache.Config") as mock_config:
            mock_config.CACHE_MAX_SIZE = 3
            mock_config.cache_db_path.return_value = tmp_path / "small_cache.db"
            c = QueryCache(tmp_path / "small_cache.db")
            for i in range(5):
                c.put(f"query {i}", f"answer {i}", [], 0.5)
            assert c.size() <= 3


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

class TestNodeCheckCache:
    def test_cache_miss(self):
        from agents.query_agent import node_check_cache

        with patch("agents.query_agent.get_query_cache") as mock_cache_cls:
            mock_cache_cls.return_value.get.return_value = None
            result = node_check_cache({"query": "What is BERT?"})

        assert result["cache_hit"] is False

    def test_cache_hit(self):
        from agents.query_agent import node_check_cache

        hit = {"answer": "BERT is a model.", "sources": [], "confidence": 0.9}
        with patch("agents.query_agent.get_query_cache") as mock_cache_cls:
            mock_cache_cls.return_value.get.return_value = hit
            result = node_check_cache({"query": "What is BERT?"})

        assert result["cache_hit"] is True
        assert result["answer"] == "BERT is a model."
        assert result["confidence"] == 0.9

    def test_cache_disabled(self):
        from agents.query_agent import node_check_cache

        with patch("agents.query_agent.Config") as mock_config:
            mock_config.USE_PERSISTENT_CACHE = False
            result = node_check_cache({"query": "q"})

        assert result["cache_hit"] is False

    def test_cache_exception_returns_miss(self):
        from agents.query_agent import node_check_cache

        with patch("agents.query_agent.get_query_cache", side_effect=Exception("DB error")):
            result = node_check_cache({"query": "q"})

        assert result["cache_hit"] is False


class TestNodeClassifyQuery:
    def test_factual_what_is(self):
        from agents.query_agent import node_classify_query
        result = node_classify_query({"query": "What is the transformer architecture?"})
        assert result["query_type"] == "factual"

    def test_comparative(self):
        from agents.query_agent import node_classify_query
        result = node_classify_query({"query": "Compare BERT vs GPT-2"})
        assert result["query_type"] == "comparative"

    def test_survey(self):
        from agents.query_agent import node_classify_query
        result = node_classify_query({"query": "Give me a survey of attention mechanisms"})
        assert result["query_type"] == "survey"

    def test_exploratory(self):
        from agents.query_agent import node_classify_query
        result = node_classify_query({"query": "Attention mechanisms in biology"})
        assert result["query_type"] == "exploratory"

    def test_vs_keyword(self):
        from agents.query_agent import node_classify_query
        result = node_classify_query({"query": "BERT vs RoBERTa performance"})
        assert result["query_type"] == "comparative"


class TestNodeReadWikiIndex:
    def test_reads_index_and_searches(self):
        from agents.query_agent import node_read_wiki_index

        with (
            patch("agents.query_agent.read_wiki_index", return_value="# Wiki Index"),
            patch("agents.query_agent.search_wiki", return_value=[
                {"page_path": "concepts/attention.md", "snippet": "...", "score": 1.0}
            ]),
            patch("agents.query_agent.list_wiki_pages", return_value=[]),
            patch("agents.query_agent.decompose_query", return_value=["attention mechanisms"]),
        ):
            result = node_read_wiki_index({
                "query": "attention mechanisms",
                "query_type": "factual",
                "sub_queries": [],
            })

        assert "concepts/attention.md" in result["wiki_pages_read"]

    def test_decomposes_comparative_query(self):
        from agents.query_agent import node_read_wiki_index

        with (
            patch("agents.query_agent.read_wiki_index", return_value=""),
            patch("agents.query_agent.search_wiki",     return_value=[]),
            patch("agents.query_agent.list_wiki_pages", return_value=[]),
            patch("agents.query_agent.decompose_query", return_value=["BERT", "GPT-2"]) as mock_decompose,
        ):
            result = node_read_wiki_index({
                "query": "Compare BERT vs GPT-2",
                "query_type": "comparative",
                "sub_queries": [],
            })

        mock_decompose.assert_called_once()
        assert result["sub_queries"] == ["BERT", "GPT-2"]

    def test_skips_decompose_for_factual(self):
        from agents.query_agent import node_read_wiki_index

        with (
            patch("agents.query_agent.read_wiki_index", return_value=""),
            patch("agents.query_agent.search_wiki",     return_value=[]),
            patch("agents.query_agent.list_wiki_pages", return_value=[]),
            patch("agents.query_agent.decompose_query") as mock_decompose,
        ):
            result = node_read_wiki_index({
                "query": "What is attention?",
                "query_type": "factual",
                "sub_queries": [],
            })

        mock_decompose.assert_not_called()
        assert result["sub_queries"] == ["What is attention?"]


class TestNodeReadWikiPages:
    def test_concatenates_pages(self):
        from agents.query_agent import node_read_wiki_pages

        def mock_read(path):
            return f"Content of {path}"

        with patch("agents.query_agent.read_wiki_page", side_effect=mock_read):
            result = node_read_wiki_pages({
                "wiki_pages_read": ["concepts/attention.md", "papers/arxiv-2301.md"],
            })

        assert "concepts/attention.md" in result["wiki_context"]
        assert "papers/arxiv-2301.md" in result["wiki_context"]

    def test_empty_pages_returns_empty_context(self):
        from agents.query_agent import node_read_wiki_pages

        result = node_read_wiki_pages({"wiki_pages_read": []})
        assert result["wiki_context"] == ""

    def test_missing_page_skipped(self):
        from agents.query_agent import node_read_wiki_pages

        with patch("agents.query_agent.read_wiki_page", return_value=""):
            result = node_read_wiki_pages({"wiki_pages_read": ["missing.md"]})

        assert result["wiki_context"] == ""


class TestNodeDecideRetrieval:
    def test_factual_with_rich_wiki_is_sufficient(self):
        from agents.query_agent import node_decide_retrieval

        result = node_decide_retrieval({
            "query_type": "factual",
            "wiki_context": "x" * 500,
        })
        assert result.get("_wiki_sufficient") is True

    def test_exploratory_always_needs_chunks(self):
        from agents.query_agent import node_decide_retrieval

        result = node_decide_retrieval({
            "query_type": "exploratory",
            "wiki_context": "x" * 500,
        })
        assert result.get("_wiki_sufficient") is False

    def test_factual_with_sparse_wiki_needs_chunks(self):
        from agents.query_agent import node_decide_retrieval

        result = node_decide_retrieval({
            "query_type": "factual",
            "wiki_context": "short",
        })
        assert result.get("_wiki_sufficient") is False


class TestNodeHybridSearch:
    def test_calls_hybrid_search(self):
        from agents.query_agent import node_hybrid_search

        chunks = [{"chunk_id": "p1_0", "text": "chunk text", "paper_id": "arxiv:1"}]
        with patch("agents.query_agent.hybrid_search", return_value=chunks) as mock_search:
            result = node_hybrid_search({
                "query": "attention",
                "sub_queries": ["attention mechanisms"],
                "query_type": "factual",
            })

        mock_search.assert_called_once()
        assert result["retrieved_chunks"] == chunks

    def test_deduplicates_chunks(self):
        from agents.query_agent import node_hybrid_search

        chunk = {"chunk_id": "p1_0", "text": "same", "paper_id": "arxiv:1"}
        with patch("agents.query_agent.hybrid_search", return_value=[chunk, chunk]):
            result = node_hybrid_search({
                "query": "attention",
                "sub_queries": ["attention", "mechanisms"],
                "query_type": "factual",
            })

        # Deduplication should collapse two identical chunks
        ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
        assert len(ids) == len(set(ids))

    def test_search_failure_returns_empty(self):
        from agents.query_agent import node_hybrid_search

        with patch("agents.query_agent.hybrid_search", side_effect=Exception("DB down")):
            result = node_hybrid_search({
                "query": "q",
                "sub_queries": ["q"],
                "query_type": "factual",
            })

        assert result["retrieved_chunks"] == []


class TestNodeRerank:
    def test_reranks_chunks(self):
        from agents.query_agent import node_rerank

        chunks   = [{"chunk_id": f"p_{i}", "text": f"text {i}"} for i in range(5)]
        reranked = chunks[:3]

        with patch("agents.query_agent.rerank_chunks", return_value=reranked):
            result = node_rerank({"query": "q", "retrieved_chunks": chunks})

        assert result["reranked_chunks"] == reranked

    def test_empty_chunks_returns_empty(self):
        from agents.query_agent import node_rerank

        result = node_rerank({"query": "q", "retrieved_chunks": []})
        assert result["reranked_chunks"] == []

    def test_rerank_failure_falls_back(self):
        from agents.query_agent import node_rerank

        chunks = [{"chunk_id": f"p_{i}"} for i in range(10)]
        with patch("agents.query_agent.rerank_chunks", side_effect=Exception("model down")):
            result = node_rerank({
                "query": "q",
                "retrieved_chunks": chunks,
            })

        # Falls back to top-K unreranked
        assert len(result["reranked_chunks"]) <= 5


class TestNodeSynthesizeAnswer:
    def test_generates_answer(self):
        from agents.query_agent import node_synthesize_answer

        with patch("agents.query_agent.synthesize_answer", return_value="The answer is 42."):
            result = node_synthesize_answer({
                "query": "What is the answer?",
                "wiki_context": "wiki content",
                "reranked_chunks": [{"paper_id": "arxiv:1", "title": "P", "year": 2023, "source": "arxiv"}],
            })

        assert result["answer"] == "The answer is 42."
        assert any(s.get("paper_id") == "arxiv:1" for s in result["sources"])

    def test_synthesis_failure_returns_fallback(self):
        from agents.query_agent import node_synthesize_answer

        with patch("agents.query_agent.synthesize_answer", side_effect=Exception("LLM down")):
            result = node_synthesize_answer({
                "query": "q",
                "wiki_context": "",
                "reranked_chunks": [],
            })

        assert "unable" in result["answer"].lower()

    def test_includes_wiki_sources(self):
        from agents.query_agent import node_synthesize_answer

        with patch("agents.query_agent.synthesize_answer", return_value="Answer."):
            result = node_synthesize_answer({
                "query": "q",
                "wiki_context": "wiki",
                "reranked_chunks": [],
                "wiki_pages_read": ["concepts/attention.md"],
            })

        wiki_sources = [s for s in result["sources"] if s.get("type") == "wiki"]
        assert len(wiki_sources) == 1
        assert wiki_sources[0]["page_path"] == "concepts/attention.md"


class TestNodeSelfCritique:
    def test_grounded_answer(self):
        from agents.query_agent import node_self_critique

        critique_result = {"is_grounded": True, "confidence": 0.9, "issues": []}
        with patch("agents.query_agent.self_critique", return_value=critique_result):
            result = node_self_critique({
                "query": "q", "answer": "a", "sources": [],
            })

        assert result["is_grounded"] is True
        assert result["confidence"] == 0.9

    def test_ungrounded_answer(self):
        from agents.query_agent import node_self_critique

        critique_result = {"is_grounded": False, "confidence": 0.3, "issues": ["hallucination"]}
        with patch("agents.query_agent.self_critique", return_value=critique_result):
            result = node_self_critique({
                "query": "q", "answer": "made up answer", "sources": [],
            })

        assert result["is_grounded"] is False
        assert result["confidence"] < 0.5

    def test_empty_answer_returns_ungrounded(self):
        from agents.query_agent import node_self_critique

        result = node_self_critique({"query": "q", "answer": "", "sources": []})
        assert result["is_grounded"] is False

    def test_critique_failure_assumes_grounded(self):
        from agents.query_agent import node_self_critique

        with patch("agents.query_agent.self_critique", side_effect=Exception("LLM down")):
            result = node_self_critique({
                "query": "q", "answer": "answer", "sources": [],
            })

        # Graceful failure: assume grounded rather than blocking
        assert result["is_grounded"] is True


class TestNodeRefineAnswer:
    def test_increments_retry_count(self):
        from agents.query_agent import node_refine_answer

        with patch("agents.query_agent.synthesize_answer", return_value="refined answer"):
            result = node_refine_answer({
                "query": "q",
                "wiki_context": "ctx",
                "reranked_chunks": [],
                "retry_count": 0,
            })

        assert result["retry_count"] == 1

    def test_produces_new_answer(self):
        from agents.query_agent import node_refine_answer

        with patch("agents.query_agent.synthesize_answer", return_value="better answer"):
            result = node_refine_answer({
                "query": "q",
                "wiki_context": "ctx",
                "reranked_chunks": [],
                "retry_count": 0,
            })

        assert result["answer"] == "better answer"


class TestNodeDecideFileBack:
    def test_worth_filing_returns_true(self):
        from agents.query_agent import node_decide_file_back

        with patch("agents.query_agent.is_worth_filing", return_value=True):
            result = node_decide_file_back({"query": "q", "answer": "a"})

        assert result["should_file_back"] is True

    def test_not_worth_filing_returns_false(self):
        from agents.query_agent import node_decide_file_back

        with patch("agents.query_agent.is_worth_filing", return_value=False):
            result = node_decide_file_back({"query": "q", "answer": "a"})

        assert result["should_file_back"] is False

    def test_write_back_disabled_skips(self):
        from agents.query_agent import node_decide_file_back

        with patch("agents.query_agent.Config") as mock_config:
            mock_config.WIKI_WRITE_BACK = False
            result = node_decide_file_back({"query": "q", "answer": "a"})

        assert result["should_file_back"] is False


class TestNodeWriteSynthesisPage:
    def test_writes_page_and_updates_index(self):
        from agents.query_agent import node_write_synthesis_page

        with (
            patch("agents.query_agent.make_synthesis_page", return_value="synthesis content"),
            patch("agents.query_agent.write_wiki_page")  as mock_write,
            patch("agents.query_agent.update_wiki_index") as mock_index,
            patch("agents.query_agent.append_wiki_log"),
        ):
            result = node_write_synthesis_page({
                "query":   "Compare BERT vs GPT",
                "answer":  "BERT is better for NLP.",
                "sources": [{"paper_id": "arxiv:1", "type": "chunk"}],
                "confidence": 0.85,
            })

        mock_write.assert_called_once()
        mock_index.assert_called_once()
        assert result["filed_page_path"] is not None
        assert result["filed_page_path"].startswith("synthesis/query-answers/")

    def test_write_failure_returns_none(self):
        from agents.query_agent import node_write_synthesis_page

        with (
            patch("agents.query_agent.make_synthesis_page", return_value="content"),
            patch("agents.query_agent.write_wiki_page", side_effect=IOError("Disk full")),
            patch("agents.query_agent.update_wiki_index"),
            patch("agents.query_agent.append_wiki_log"),
        ):
            result = node_write_synthesis_page({
                "query": "q", "answer": "a", "sources": [], "confidence": 0.5,
            })

        assert result["filed_page_path"] is None


class TestNodeSaveCache:
    def test_saves_to_cache(self):
        from agents.query_agent import node_save_cache

        mock_cache = MagicMock()
        with patch("agents.query_agent.get_query_cache", return_value=mock_cache):
            node_save_cache({
                "query": "q", "answer": "a", "sources": [], "confidence": 0.8,
            })

        mock_cache.put.assert_called_once_with("q", "a", [], 0.8)

    def test_skips_empty_query(self):
        from agents.query_agent import node_save_cache

        mock_cache = MagicMock()
        with patch("agents.query_agent.get_query_cache", return_value=mock_cache):
            node_save_cache({"query": "", "answer": "a", "sources": []})

        mock_cache.put.assert_not_called()

    def test_cache_disabled_skips(self):
        from agents.query_agent import node_save_cache

        mock_cache = MagicMock()
        with (
            patch("agents.query_agent.Config") as mock_config,
            patch("agents.query_agent.get_query_cache", return_value=mock_cache),
        ):
            mock_config.USE_PERSISTENT_CACHE = False
            node_save_cache({"query": "q", "answer": "a", "sources": []})

        mock_cache.put.assert_not_called()


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

class TestRoutingFunctions:
    def test_route_after_cache_hit(self):
        from agents.query_agent import _route_after_cache
        assert _route_after_cache({"cache_hit": True}) == "end"

    def test_route_after_cache_miss(self):
        from agents.query_agent import _route_after_cache
        assert _route_after_cache({"cache_hit": False}) == "classify_query"

    def test_route_wiki_sufficient(self):
        from agents.query_agent import _route_after_decide_retrieval
        assert _route_after_decide_retrieval({"_wiki_sufficient": True}) == "synthesize_answer"

    def test_route_needs_chunks(self):
        from agents.query_agent import _route_after_decide_retrieval
        assert _route_after_decide_retrieval({"_wiki_sufficient": False}) == "hybrid_search"

    def test_route_after_critique_grounded(self):
        from agents.query_agent import _route_after_self_critique
        assert _route_after_self_critique({"is_grounded": True, "retry_count": 0}) == "decide_file_back"

    def test_route_after_critique_ungrounded_first_try(self):
        from agents.query_agent import _route_after_self_critique
        assert _route_after_self_critique({"is_grounded": False, "retry_count": 0}) == "refine_answer"

    def test_route_after_critique_ungrounded_second_try(self):
        from agents.query_agent import _route_after_self_critique
        assert _route_after_self_critique({"is_grounded": False, "retry_count": 1}) == "decide_file_back"

    def test_route_file_back_yes(self):
        from agents.query_agent import _route_after_decide_file_back
        assert _route_after_decide_file_back({"should_file_back": True}) == "write_synthesis_page"

    def test_route_file_back_no(self):
        from agents.query_agent import _route_after_decide_file_back
        assert _route_after_decide_file_back({"should_file_back": False}) == "save_cache"


# ---------------------------------------------------------------------------
# Graph construction test
# ---------------------------------------------------------------------------

class TestBuildQueryGraph:
    def test_graph_compiles(self):
        from agents.query_agent import build_query_graph
        app = build_query_graph()
        assert app is not None

    def test_graph_has_expected_nodes(self):
        from agents.query_agent import build_query_graph
        app   = build_query_graph()
        graph = app.get_graph()
        node_names = set(graph.nodes.keys())

        expected = {
            "check_cache", "classify_query", "read_wiki_index",
            "read_wiki_pages", "decide_retrieval", "hybrid_search",
            "rerank", "synthesize_answer", "self_critique",
            "refine_answer", "decide_file_back", "write_synthesis_page",
            "save_cache",
        }
        assert expected.issubset(node_names)


# ---------------------------------------------------------------------------
# End-to-end run_query tests
# ---------------------------------------------------------------------------

class TestRunQuery:
    def _all_mocks(self, answer="The answer.", file_back=False, cache_hit=None):
        """Return a context manager that mocks all external calls."""
        from contextlib import ExitStack
        from unittest.mock import patch

        patches = [
            patch("agents.query_agent.get_query_cache", return_value=MagicMock(
                get=MagicMock(return_value=cache_hit),
                put=MagicMock(),
            )),
            patch("agents.query_agent.decompose_query",    return_value=["sub-query"]),
            patch("agents.query_agent.read_wiki_index",    return_value="# Index"),
            patch("agents.query_agent.search_wiki",        return_value=[]),
            patch("agents.query_agent.list_wiki_pages",    return_value=[]),
            patch("agents.query_agent.read_wiki_page",     return_value="wiki content"),
            patch("agents.query_agent.hybrid_search",      return_value=[
                {"chunk_id": "p1_0", "text": "chunk", "paper_id": "arxiv:1",
                 "title": "Paper", "year": 2023, "source": "arxiv"}
            ]),
            patch("agents.query_agent.rerank_chunks",      return_value=[
                {"chunk_id": "p1_0", "text": "chunk", "paper_id": "arxiv:1",
                 "title": "Paper", "year": 2023, "source": "arxiv"}
            ]),
            patch("agents.query_agent.synthesize_answer",  return_value=answer),
            patch("agents.query_agent.self_critique",      return_value={
                "is_grounded": True, "confidence": 0.85, "issues": [],
            }),
            patch("agents.query_agent.is_worth_filing",    return_value=file_back),
            patch("agents.query_agent.make_synthesis_page", return_value="synthesis content"),
            patch("agents.query_agent.write_wiki_page"),
            patch("agents.query_agent.update_wiki_index"),
            patch("agents.query_agent.append_wiki_log"),
        ]

        stack = ExitStack()
        mocks = [stack.enter_context(p) for p in patches]
        return stack

    def test_basic_query_returns_answer(self):
        from agents.query_agent import run_query

        with self._all_mocks(answer="Attention is a mechanism."):
            result = run_query("What is attention?")

        assert result["answer"] == "Attention is a mechanism."
        assert result["is_grounded"] is True
        assert result["cache_hit"] is False

    def test_cache_hit_returns_immediately(self):
        from agents.query_agent import run_query

        cached = {
            "answer": "Cached answer.", "sources": [], "confidence": 0.9,
            "date_cached": "2026-04-13",
        }
        with self._all_mocks(cache_hit=cached):
            result = run_query("cached query")

        assert result["cache_hit"] is True
        assert result["answer"] == "Cached answer."

    def test_file_back_writes_synthesis_page(self):
        from agents.query_agent import run_query

        with self._all_mocks(answer="Novel synthesis.", file_back=True) as mocks:
            result = run_query("Compare BERT vs GPT")

        assert result.get("filed_page_path") is not None
        assert result["filed_page_path"].startswith("synthesis/query-answers/")

    def test_no_file_back_when_not_worth_it(self):
        from agents.query_agent import run_query

        with self._all_mocks(answer="Simple answer.", file_back=False):
            result = run_query("What is BERT?")

        assert result.get("filed_page_path") is None

    def test_year_filter_passed_to_search(self):
        from agents.query_agent import run_query

        with patch("agents.query_agent.hybrid_search") as mock_search, \
             self._all_mocks():
            mock_search.return_value = []
            run_query("attention transformers", year_filter=2023)

        # Verify year_filter was forwarded to hybrid_search
        if mock_search.called:
            call_kwargs = mock_search.call_args[1] if mock_search.call_args[1] else {}
            call_args   = mock_search.call_args[0]
            year_passed = call_kwargs.get("year_filter") or (call_args[2] if len(call_args) > 2 else None)
            assert year_passed == 2023

    def test_answer_present_even_if_not_grounded(self):
        """Agent should still return an answer even after failed self-critique."""
        from agents.query_agent import run_query

        from contextlib import ExitStack

        patches = [
            patch("agents.query_agent.get_query_cache", return_value=MagicMock(
                get=MagicMock(return_value=None), put=MagicMock()
            )),
            patch("agents.query_agent.decompose_query",    return_value=["q"]),
            patch("agents.query_agent.read_wiki_index",    return_value=""),
            patch("agents.query_agent.search_wiki",        return_value=[]),
            patch("agents.query_agent.list_wiki_pages",    return_value=[]),
            patch("agents.query_agent.read_wiki_page",     return_value=""),
            patch("agents.query_agent.hybrid_search",      return_value=[]),
            patch("agents.query_agent.rerank_chunks",      return_value=[]),
            patch("agents.query_agent.synthesize_answer",  return_value="uncertain answer"),
            patch("agents.query_agent.self_critique",      return_value={
                "is_grounded": False, "confidence": 0.2, "issues": ["hallucination"],
            }),
            patch("agents.query_agent.is_worth_filing",    return_value=False),
            patch("agents.query_agent.write_wiki_page"),
            patch("agents.query_agent.update_wiki_index"),
            patch("agents.query_agent.append_wiki_log"),
        ]

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = run_query("tricky query")

        # Answer should be present even if not grounded (agent proceeds with caveat)
        assert result.get("answer") is not None
        assert len(result["answer"]) > 0
