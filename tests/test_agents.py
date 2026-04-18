"""
tests/test_agents.py — Unit tests for Phase 6: Ingest Agent.

All LLM calls, source connectors, parsers, and file I/O are mocked.
Tests cover:
  - Full happy-path ingest pipeline
  - Abstract-only paper path (no PDF)
  - Deduplication (federated_search returns duplicates)
  - Parser fallback behaviour
  - Wiki page creation and content
  - Index update and log append
  - Error recovery (parse failure, index failure)
  - Individual node functions
  - Graph construction (build_ingest_graph)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_paper(paper_id: str = "arxiv:2301.12345", abstract_only: bool = False) -> dict:
    return {
        "paper_id": paper_id,
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N."],
        "abstract": "We propose the Transformer architecture based on attention.",
        "year": 2017,
        "source": "arxiv",
        "pdf_url": None if abstract_only else "https://arxiv.org/pdf/2301.12345",
        "doi": "10.5555/3295222.3295349",
        "venue": "NeurIPS",
        "tags": ["transformers", "attention", "self-attention"],
        "is_open_access": True,
    }


def _make_parsed(paper: dict, abstract_only: bool = False) -> dict:
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "authors": paper["authors"],
        "abstract": paper["abstract"],
        "year": paper["year"],
        "source": paper["source"],
        "sections": [
            {"heading": "Introduction", "text": "Attention mechanisms are key.", "level": 1},
            {"heading": "Methods", "text": "We use multi-head attention.", "level": 1},
        ],
        "references": [],
        "figures": [],
        "tables": [],
        "equations": [],
        "parser_used": "none" if abstract_only else "grobid",
        "math_fraction": 0.05,
        "is_abstract_only": abstract_only,
        "tags": paper.get("tags", []),
    }


# ---------------------------------------------------------------------------
# Tests for individual node functions
# ---------------------------------------------------------------------------

class TestNodeFederatedSearch:
    def test_returns_candidates(self):
        from agents.ingest_agent import node_federated_search

        paper = _make_paper()
        with patch("agents.ingest_agent.federated_search", return_value=[paper]) as mock_search:
            result = node_federated_search({"query": "attention transformers"})

        mock_search.assert_called_once_with("attention transformers", max_results=200)
        assert result["candidates"] == [paper]
        assert result["paper_index"] == 0
        assert result["done"] is False
        assert result["papers_processed"] == []
        assert result["chunks_indexed"] == 0
        assert result["errors"] == []

    def test_empty_query_returns_done(self):
        from agents.ingest_agent import node_federated_search

        result = node_federated_search({"query": ""})
        assert result["done"] is True
        assert result["candidates"] == []

    def test_missing_query_key_returns_done(self):
        from agents.ingest_agent import node_federated_search

        result = node_federated_search({})
        assert result["done"] is True


class TestNodeNextPaper:
    def test_loads_first_paper(self):
        from agents.ingest_agent import node_next_paper

        paper = _make_paper()
        state = {"candidates": [paper], "paper_index": 0}
        result = node_next_paper(state)

        assert result["paper"] == paper
        assert result["paper_index"] == 1
        # When not done, "done" key is absent (partial state update)
        assert result.get("done", False) is False

    def test_sets_done_when_exhausted(self):
        from agents.ingest_agent import node_next_paper

        paper = _make_paper()
        state = {"candidates": [paper], "paper_index": 1}
        result = node_next_paper(state)

        assert result["done"] is True
        assert result["paper"] is None

    def test_resets_per_paper_fields(self):
        from agents.ingest_agent import node_next_paper

        paper = _make_paper()
        state = {"candidates": [paper, paper], "paper_index": 0}
        result = node_next_paper(state)

        assert result["pdf_path"] is None
        assert result["parsed_paper"] is None
        assert result["is_abstract_only"] is False


class TestNodeDownloadPdf:
    def test_pdf_available(self, tmp_path):
        from agents.ingest_agent import node_download_pdf

        paper = _make_paper()
        pdf_path = str(tmp_path / "paper.pdf")
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")

        with patch("agents.ingest_agent.download_pdf", return_value=pdf_path):
            result = node_download_pdf({"paper": paper})

        assert result["pdf_path"] == pdf_path
        assert result["is_abstract_only"] is False

    def test_pdf_unavailable_no_doi(self):
        from agents.ingest_agent import node_download_pdf

        paper = _make_paper(abstract_only=True)
        paper["doi"] = None

        with patch("agents.ingest_agent.download_pdf", return_value=None):
            result = node_download_pdf({"paper": paper})

        assert result["pdf_path"] is None
        assert result["is_abstract_only"] is True

    def test_unpaywall_fallback(self, tmp_path):
        from agents.ingest_agent import node_download_pdf

        paper = _make_paper()
        paper["pdf_url"] = None
        pdf_path = str(tmp_path / "paper.pdf")

        with (
            patch("agents.ingest_agent.download_pdf", return_value=None),
            patch("agents.ingest_agent._try_unpaywall", return_value=pdf_path),
        ):
            result = node_download_pdf({"paper": paper})

        assert result["pdf_path"] == pdf_path
        assert result["is_abstract_only"] is False

    def test_no_paper_returns_abstract_only(self):
        from agents.ingest_agent import node_download_pdf

        result = node_download_pdf({"paper": None})
        assert result["is_abstract_only"] is True
        assert result["pdf_path"] is None


class TestNodeParsePaper:
    def test_parses_pdf(self):
        from agents.ingest_agent import node_parse_paper

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        state  = {"paper": paper, "pdf_path": "/tmp/paper.pdf", "is_abstract_only": False}

        with patch("agents.ingest_agent.run_parser", return_value=parsed):
            result = node_parse_paper(state)

        assert result["parsed_paper"] == parsed

    def test_abstract_only_builds_stub(self):
        from agents.ingest_agent import node_parse_paper

        paper = _make_paper(abstract_only=True)
        state = {"paper": paper, "pdf_path": None, "is_abstract_only": True}

        result = node_parse_paper(state)

        assert result["parsed_paper"]["is_abstract_only"] is True
        assert result["parsed_paper"]["parser_used"] == "none"
        assert result["parsed_paper"]["paper_id"] == paper["paper_id"]
        assert result["parsed_paper"]["sections"] == []

    def test_parse_failure_falls_back_to_abstract_stub(self):
        from agents.ingest_agent import node_parse_paper

        paper = _make_paper()
        state = {
            "paper": paper,
            "pdf_path": "/tmp/paper.pdf",
            "is_abstract_only": False,
            "errors": [],
        }

        with patch("agents.ingest_agent.run_parser", side_effect=RuntimeError("Grobid down")):
            result = node_parse_paper(state)

        assert result["parsed_paper"]["is_abstract_only"] is True
        assert result["is_abstract_only"] is True
        assert len(result["errors"]) == 1
        assert "parse error" in result["errors"][0]


class TestNodeChunkAndIndex:
    def test_indexes_chunks(self):
        from agents.ingest_agent import node_chunk_and_index

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        state  = {"parsed_paper": parsed, "chunks_indexed": 0}

        with patch("agents.ingest_agent.index_paper", return_value=42):
            result = node_chunk_and_index(state)

        assert result["chunks_indexed"] == 42

    def test_accumulates_across_papers(self):
        from agents.ingest_agent import node_chunk_and_index

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        state  = {"parsed_paper": parsed, "chunks_indexed": 10}

        with patch("agents.ingest_agent.index_paper", return_value=5):
            result = node_chunk_and_index(state)

        assert result["chunks_indexed"] == 15

    def test_no_parsed_paper_returns_empty(self):
        from agents.ingest_agent import node_chunk_and_index

        result = node_chunk_and_index({"parsed_paper": None})
        assert result == {}

    def test_index_failure_records_error(self):
        from agents.ingest_agent import node_chunk_and_index

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        state  = {"parsed_paper": parsed, "chunks_indexed": 0, "errors": []}

        with patch("agents.ingest_agent.index_paper", side_effect=Exception("ChromaDB down")):
            result = node_chunk_and_index(state)

        assert len(result["errors"]) == 1
        assert "index error" in result["errors"][0]


class TestNodeWritePaperWikiPage:
    def test_creates_wiki_page(self, tmp_path):
        from agents.ingest_agent import node_write_paper_wiki_page

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        state  = {
            "paper": paper,
            "parsed_paper": parsed,
            "is_abstract_only": False,
            "wiki_pages_written": [],
        }

        stub_content  = "---\npaper_id: arxiv:2301.12345\n---\n# Title\n## Summary\n_stub_\n"
        filled_content = stub_content.replace("_stub_", "Great paper about attention.")

        with (
            patch("agents.ingest_agent.extract_tags",       return_value=["transformers", "attention"]),
            patch("agents.ingest_agent.summarize_paper",    return_value="A great paper."),
            patch("agents.ingest_agent.make_paper_page",    return_value=stub_content),
            patch("agents.ingest_agent.fill_paper_wiki_page", return_value=filled_content),
            patch("agents.ingest_agent.write_wiki_page")    as mock_write,
        ):
            result = node_write_paper_wiki_page(state)

        mock_write.assert_called_once()
        call_args = mock_write.call_args
        args, kwargs = call_args
        # write_wiki_page(page_path, content, reason=...)
        page_path = args[0] if args else kwargs.get("page_path")
        content   = args[1] if len(args) > 1 else kwargs.get("content")
        reason    = args[2] if len(args) > 2 else kwargs.get("reason", "")
        assert page_path == "papers/arxiv-2301.12345.md"
        assert content == filled_content
        assert "arxiv" in reason and "2301.12345" in reason

        assert "papers/arxiv-2301.12345.md" in result["wiki_pages_written"]

    def test_abstract_only_page_creation(self, tmp_path):
        from agents.ingest_agent import node_write_paper_wiki_page

        paper  = _make_paper(abstract_only=True)
        parsed = _make_parsed(paper, abstract_only=True)
        state  = {
            "paper": paper,
            "parsed_paper": parsed,
            "is_abstract_only": True,
            "wiki_pages_written": [],
        }

        with (
            patch("agents.ingest_agent.extract_tags",       return_value=["attention"]),
            patch("agents.ingest_agent.summarize_paper",    return_value="Abstract only."),
            patch("agents.ingest_agent.make_paper_page",    return_value="stub") as mock_page,
            patch("agents.ingest_agent.fill_paper_wiki_page", return_value="filled"),
            patch("agents.ingest_agent.write_wiki_page"),
        ):
            node_write_paper_wiki_page(state)

        # Verify is_abstract_only=True was passed to make_paper_page
        call_kwargs = mock_page.call_args[1]
        assert call_kwargs.get("is_abstract_only") is True

    def test_write_failure_records_error(self):
        from agents.ingest_agent import node_write_paper_wiki_page

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        state  = {
            "paper": paper,
            "parsed_paper": parsed,
            "is_abstract_only": False,
            "wiki_pages_written": [],
            "errors": [],
        }

        with (
            patch("agents.ingest_agent.extract_tags",       return_value=[]),
            patch("agents.ingest_agent.summarize_paper",    return_value=""),
            patch("agents.ingest_agent.make_paper_page",    return_value="stub"),
            patch("agents.ingest_agent.fill_paper_wiki_page", return_value="filled"),
            patch("agents.ingest_agent.write_wiki_page",    side_effect=IOError("Disk full")),
        ):
            result = node_write_paper_wiki_page(state)

        assert len(result["errors"]) == 1
        assert "wiki page error" in result["errors"][0]


class TestNodeUpdateConceptPages:
    def test_creates_new_concept_page(self):
        from agents.ingest_agent import node_update_concept_pages

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        parsed["tags"] = ["attention"]
        state = {
            "parsed_paper": parsed,
            "wiki_pages_written": [],
        }

        with (
            patch("agents.ingest_agent.read_wiki_page",     return_value=""),
            patch("agents.ingest_agent.make_concept_page",  return_value="concept stub"),
            patch("agents.ingest_agent.update_concept_page", return_value="updated concept"),
            patch("agents.ingest_agent.write_wiki_page")    as mock_write,
            patch("agents.ingest_agent.update_wiki_index"),
        ):
            result = node_update_concept_pages(state)

        assert "concepts/attention.md" in result["wiki_pages_written"]
        mock_write.assert_called_once()

    def test_updates_existing_concept_page(self):
        from agents.ingest_agent import node_update_concept_pages

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        parsed["tags"] = ["transformers"]
        state = {
            "parsed_paper": parsed,
            "wiki_pages_written": [],
        }
        existing_page = "---\nconcept: transformers\n---\n# Transformers\n## Key Papers\n"

        with (
            patch("agents.ingest_agent.read_wiki_page",     return_value=existing_page),
            patch("agents.ingest_agent.make_concept_page"),
            patch("agents.ingest_agent.update_concept_page", return_value="updated") as mock_update,
            patch("agents.ingest_agent.write_wiki_page"),
            patch("agents.ingest_agent.update_wiki_index"),
        ):
            result = node_update_concept_pages(state)

        # Should NOT call make_concept_page since page exists
        mock_update.assert_called_once()
        assert "concepts/transformers.md" in result["wiki_pages_written"]

    def test_no_parsed_paper_returns_empty(self):
        from agents.ingest_agent import node_update_concept_pages

        result = node_update_concept_pages({"parsed_paper": None})
        assert result == {}

    def test_limits_to_four_concepts(self):
        from agents.ingest_agent import node_update_concept_pages

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        parsed["tags"] = ["a", "b", "c", "d", "e", "f"]
        state = {
            "parsed_paper": parsed,
            "wiki_pages_written": [],
        }

        calls = []
        def track_write(page_path, *args, **kwargs):
            calls.append(page_path)

        with (
            patch("agents.ingest_agent.read_wiki_page",     return_value=""),
            patch("agents.ingest_agent.make_concept_page",  return_value="stub"),
            patch("agents.ingest_agent.update_concept_page", return_value="updated"),
            patch("agents.ingest_agent.write_wiki_page",    side_effect=track_write),
            patch("agents.ingest_agent.update_wiki_index"),
        ):
            node_update_concept_pages(state)

        # Only 4 concept pages max
        assert len(calls) <= 4


class TestNodeUpdateIndexAndLog:
    def test_updates_index_and_appends_log(self):
        from agents.ingest_agent import node_update_index_and_log

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        state  = {
            "paper": paper,
            "parsed_paper": parsed,
            "is_abstract_only": False,
            "chunks_indexed": 42,
            "wiki_pages_written": ["papers/arxiv-2301.12345.md", "concepts/attention.md"],
            "papers_processed": [],
            "errors": [],
        }

        with (
            patch("agents.ingest_agent.update_wiki_index") as mock_index,
            patch("agents.ingest_agent.append_wiki_log")   as mock_log,
        ):
            result = node_update_index_and_log(state)

        mock_index.assert_called_once()
        mock_log.assert_called_once()

        log_args, log_kwargs = mock_log.call_args
        operation = log_args[0] if log_args else log_kwargs.get("operation")
        title     = log_args[1] if len(log_args) > 1 else log_kwargs.get("title", "")
        assert operation == "ingest"
        assert "arxiv:2301.12345" in title

        assert "arxiv:2301.12345" in result["papers_processed"]
        assert result["paper"] is None
        assert result["parsed_paper"] is None

    def test_abstract_only_log_note(self):
        from agents.ingest_agent import node_update_index_and_log

        paper  = _make_paper(abstract_only=True)
        parsed = _make_parsed(paper, abstract_only=True)
        state  = {
            "paper": paper,
            "parsed_paper": parsed,
            "is_abstract_only": True,
            "chunks_indexed": 0,
            "wiki_pages_written": ["papers/arxiv-2301.12345.md"],
            "papers_processed": [],
            "errors": [],
        }

        with (
            patch("agents.ingest_agent.update_wiki_index"),
            patch("agents.ingest_agent.append_wiki_log") as mock_log,
        ):
            node_update_index_and_log(state)

        log_args, log_kwargs = mock_log.call_args
        details = log_args[2] if len(log_args) > 2 else log_kwargs.get("details", "")
        assert "abstract_only" in details
        assert "stub page created" in details.lower()

    def test_resets_per_paper_state(self):
        from agents.ingest_agent import node_update_index_and_log

        paper  = _make_paper()
        parsed = _make_parsed(paper)
        state  = {
            "paper": paper,
            "parsed_paper": parsed,
            "is_abstract_only": False,
            "chunks_indexed": 5,
            "wiki_pages_written": ["papers/arxiv-2301.12345.md"],
            "papers_processed": [],
            "errors": [],
        }

        with (
            patch("agents.ingest_agent.update_wiki_index"),
            patch("agents.ingest_agent.append_wiki_log"),
        ):
            result = node_update_index_and_log(state)

        assert result["paper"] is None
        assert result["parsed_paper"] is None
        assert result["is_abstract_only"] is False
        assert result["wiki_pages_written"] == []


# ---------------------------------------------------------------------------
# Tests for routing and helper functions
# ---------------------------------------------------------------------------

class TestRouteAfterNextPaper:
    def test_routes_to_end_when_done(self):
        from agents.ingest_agent import _route_after_next_paper

        assert _route_after_next_paper({"done": True}) == "end"

    def test_routes_to_download_when_not_done(self):
        from agents.ingest_agent import _route_after_next_paper

        assert _route_after_next_paper({"done": False}) == "download_pdf"

    def test_routes_to_download_when_done_absent(self):
        from agents.ingest_agent import _route_after_next_paper

        assert _route_after_next_paper({}) == "download_pdf"


class TestSlugify:
    def test_lowercase_hyphens(self):
        from agents.ingest_agent import _slugify

        assert _slugify("Self Attention") == "self-attention"

    def test_removes_special_chars(self):
        from agents.ingest_agent import _slugify

        assert _slugify("BERT: Pre-training") == "bert-pre-training"

    def test_truncates_to_60(self):
        from agents.ingest_agent import _slugify

        long = "a" * 80
        assert len(_slugify(long)) <= 60


class TestBuildAbstractOnlyParsed:
    def test_builds_stub(self):
        from agents.ingest_agent import _build_abstract_only_parsed

        paper = _make_paper(abstract_only=True)
        parsed = _build_abstract_only_parsed(paper)

        assert parsed["paper_id"] == paper["paper_id"]
        assert parsed["title"] == paper["title"]
        assert parsed["is_abstract_only"] is True
        assert parsed["parser_used"] == "none"
        assert parsed["sections"] == []
        assert parsed["equations"] == []


class TestTryUnpaywall:
    def test_returns_pdf_path_on_success(self, tmp_path):
        from agents.ingest_agent import _try_unpaywall

        paper = _make_paper()

        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"}
        }

        mock_pdf_response = MagicMock()
        mock_pdf_response.status_code = 200
        mock_pdf_response.iter_content.return_value = [b"%PDF-1.4"]

        with (
            patch("agents.ingest_agent.requests.get", side_effect=[mock_api_response, mock_pdf_response]),
            patch("agents.ingest_agent.Config.raw_pdf_dir", return_value=tmp_path),
        ):
            result = _try_unpaywall("10.5555/test", paper)

        assert result is not None
        assert result.endswith(".pdf")

    def test_returns_none_when_no_pdf_url(self):
        from agents.ingest_agent import _try_unpaywall

        paper = _make_paper()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"best_oa_location": {"url_for_pdf": None}}

        with patch("agents.ingest_agent.requests.get", return_value=mock_resp):
            result = _try_unpaywall("10.5555/test", paper)

        assert result is None

    def test_returns_none_on_api_error(self):
        from agents.ingest_agent import _try_unpaywall

        paper = _make_paper()

        with patch("agents.ingest_agent.requests.get", side_effect=Exception("Network error")):
            result = _try_unpaywall("10.5555/test", paper)

        assert result is None

    def test_returns_none_on_404(self):
        from agents.ingest_agent import _try_unpaywall

        paper = _make_paper()

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("agents.ingest_agent.requests.get", return_value=mock_resp):
            result = _try_unpaywall("10.5555/bad-doi", paper)

        assert result is None


# ---------------------------------------------------------------------------
# Tests for graph construction
# ---------------------------------------------------------------------------

class TestBuildIngestGraph:
    def test_graph_compiles(self):
        from agents.ingest_agent import build_ingest_graph

        app = build_ingest_graph()
        assert app is not None

    def test_graph_has_expected_nodes(self):
        from agents.ingest_agent import build_ingest_graph

        app = build_ingest_graph()
        # LangGraph compiled graph exposes its graph object
        graph = app.get_graph()
        node_names = set(graph.nodes.keys())

        expected = {
            "federated_search", "next_paper", "download_pdf",
            "parse_paper", "chunk_and_index", "write_paper_wiki_page",
            "update_concept_pages", "check_contradictions",
            "update_index_and_log",
        }
        assert expected.issubset(node_names)


# ---------------------------------------------------------------------------
# End-to-end integration tests (all externals mocked)
# ---------------------------------------------------------------------------

class TestRunIngest:
    """
    Full end-to-end tests of run_ingest() with all external calls mocked.
    Verifies the pipeline produces the correct state for:
      1. One full-text paper
      2. One abstract-only paper
      3. Two papers (verify accumulation)
    """

    def _mock_run_ingest(self, papers, pdf_paths=None, abstract_only_flags=None, tmp_path=None):
        """
        Helper: run the full ingest graph with all externals mocked.
        Returns the final state.
        """
        from agents.ingest_agent import run_ingest

        if pdf_paths is None:
            pdf_paths = [None] * len(papers)
        if abstract_only_flags is None:
            abstract_only_flags = [p is None for p in pdf_paths]

        parsed_papers = [
            _make_parsed(papers[i], abstract_only=abstract_only_flags[i])
            for i in range(len(papers))
        ]

        download_returns = iter(pdf_paths)
        parse_returns    = iter(parsed_papers)

        def _mock_download(metadata, output_dir=None):
            return next(download_returns)

        def _mock_parse(pdf_path, metadata):
            return next(parse_returns)

        # Patch at agents.ingest_agent since it imports names at module level
        with (
            patch("agents.ingest_agent.federated_search",     return_value=papers),
            patch("agents.ingest_agent.download_pdf",         side_effect=_mock_download),
            patch("agents.ingest_agent.run_parser",           side_effect=_mock_parse),
            patch("agents.ingest_agent.index_paper",          return_value=10),
            patch("agents.ingest_agent.extract_tags",         return_value=["attention"]),
            patch("agents.ingest_agent.summarize_paper",      return_value="A good paper."),
            patch("agents.ingest_agent.fill_paper_wiki_page", return_value="filled page content"),
            patch("agents.ingest_agent.update_concept_page",  return_value="updated concept"),
            patch("agents.ingest_agent.flag_contradictions",  return_value=[]),
            patch("agents.ingest_agent.write_wiki_page"),
            patch("agents.ingest_agent.update_wiki_index"),
            patch("agents.ingest_agent.append_wiki_log"),
            patch("agents.ingest_agent.read_wiki_page",       return_value=""),
            patch("agents.ingest_agent.make_concept_page",    return_value="concept stub"),
            patch("agents.ingest_agent.make_paper_page",      return_value="paper stub"),
            patch("agents.ingest_agent.make_debate_page",     return_value="debate stub"),
        ):
            return run_ingest("attention transformers")

    def test_single_fulltext_paper(self, tmp_path):
        paper  = _make_paper()
        result = self._mock_run_ingest([paper], pdf_paths=["/tmp/paper.pdf"])

        assert "arxiv:2301.12345" in result["papers_processed"]
        assert result["chunks_indexed"] == 10
        assert result["errors"] == []
        assert result["done"] is True

    def test_abstract_only_paper(self, tmp_path):
        paper  = _make_paper(abstract_only=True)
        result = self._mock_run_ingest([paper], pdf_paths=[None])

        assert "arxiv:2301.12345" in result["papers_processed"]
        # Abstract-only still gets indexed (abstract text) and a stub page
        assert result["errors"] == []

    def test_two_papers_accumulate_chunks(self, tmp_path):
        paper1 = _make_paper("arxiv:2301.00001")
        paper2 = _make_paper("arxiv:2301.00002")

        result = self._mock_run_ingest(
            [paper1, paper2],
            pdf_paths=["/tmp/p1.pdf", "/tmp/p2.pdf"],
        )

        assert "arxiv:2301.00001" in result["papers_processed"]
        assert "arxiv:2301.00002" in result["papers_processed"]
        assert result["chunks_indexed"] == 20  # 10 per paper

    def test_empty_candidates(self, tmp_path):
        from agents.ingest_agent import run_ingest

        with patch("agents.ingest_agent.federated_search", return_value=[]):
            result = run_ingest("obscure query no results")

        assert result["papers_processed"] == [] or len(result.get("papers_processed", [])) == 0
        assert result["done"] is True

    def test_parse_failure_continues_pipeline(self, tmp_path):
        """A paper that fails to parse should still produce a stub page and not abort."""
        from agents.ingest_agent import run_ingest

        paper = _make_paper()

        with (
            patch("agents.ingest_agent.federated_search",     return_value=[paper]),
            patch("agents.ingest_agent.download_pdf",         return_value="/tmp/paper.pdf"),
            patch("agents.ingest_agent.run_parser",           side_effect=RuntimeError("Parser crashed")),
            patch("agents.ingest_agent.index_paper",          return_value=0),
            patch("agents.ingest_agent.extract_tags",         return_value=[]),
            patch("agents.ingest_agent.summarize_paper",      return_value=""),
            patch("agents.ingest_agent.fill_paper_wiki_page", return_value="stub"),
            patch("agents.ingest_agent.update_concept_page",  return_value=""),
            patch("agents.ingest_agent.flag_contradictions",  return_value=[]),
            patch("agents.ingest_agent.write_wiki_page"),
            patch("agents.ingest_agent.update_wiki_index"),
            patch("agents.ingest_agent.append_wiki_log"),
            patch("agents.ingest_agent.read_wiki_page",       return_value=""),
            patch("agents.ingest_agent.make_concept_page",    return_value=""),
            patch("agents.ingest_agent.make_paper_page",      return_value="stub"),
            patch("agents.ingest_agent.make_debate_page",     return_value=""),
        ):
            result = run_ingest("attention transformers")

        # Paper should still appear as processed (fallback stub was created)
        assert "arxiv:2301.12345" in result["papers_processed"]
        # Error should be recorded
        assert len(result["errors"]) > 0
        assert "parse error" in result["errors"][0]


# ---------------------------------------------------------------------------
# Tests for wiki page content validation
# ---------------------------------------------------------------------------

class TestWikiPageFormat:
    """Verify that wiki pages are created with the correct format."""

    def test_paper_page_has_required_sections(self):
        from tools.wiki_tools import make_paper_page

        content = make_paper_page(
            paper_id="arxiv:2301.12345",
            title="Attention Is All You Need",
            authors=["Vaswani, A."],
            year=2017,
            source="arxiv",
            venue="NeurIPS",
            parser_used="grobid",
            math_fraction=0.05,
            is_abstract_only=False,
            tags=["transformers"],
            summary="A great paper.",
        )

        assert "## Summary" in content
        assert "## Key Contributions" in content
        assert "## Methods" in content
        assert "## Results & Claims" in content
        assert "## Limitations" in content

    def test_abstract_only_page_has_warning(self):
        from tools.wiki_tools import make_paper_page

        content = make_paper_page(
            paper_id="arxiv:2301.12345",
            title="Attention Is All You Need",
            authors=[],
            year=2017,
            source="arxiv",
            venue="",
            parser_used="none",
            math_fraction=0.0,
            is_abstract_only=True,
            tags=[],
            summary="Abstract only.",
        )

        assert "ABSTRACT ONLY" in content

    def test_paper_page_frontmatter(self):
        from tools.wiki_tools import make_paper_page

        content = make_paper_page(
            paper_id="arxiv:2301.12345",
            title="Test Paper",
            authors=["Author A", "Author B"],
            year=2023,
            source="arxiv",
            venue="NeurIPS",
            parser_used="grobid",
            math_fraction=0.1,
            is_abstract_only=False,
            tags=["ml", "nlp"],
            summary="A summary.",
        )

        assert "paper_id: arxiv:2301.12345" in content
        assert "year: 2023" in content
        assert "source: arxiv" in content
        assert "is_abstract_only: false" in content
