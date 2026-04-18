"""
tests/test_interfaces.py — Unit tests for Phase 9 interfaces.

Tests the CLI argument parser, command handlers, and Gradio backend
functions. No real agents or LLMs are invoked — all external calls
are mocked. Desktop app tests are limited to import and backend
functions (no GUI event loop is started).
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# CLI — argument parser
# ---------------------------------------------------------------------------

class TestCLIParser:
    def _parse(self, *args):
        from interfaces.cli import build_parser
        return build_parser().parse_args(list(args))

    def test_ingest_command(self):
        args = self._parse("ingest", "attention", "transformers")
        assert args.command == "ingest"
        assert args.query == ["attention", "transformers"]

    def test_query_command(self):
        # parse_args receives a list of strings; a quoted phrase is one element
        args = self._parse("query", "What", "is", "BERT?")
        assert args.command == "query"
        assert args.question == ["What", "is", "BERT?"]

    def test_query_with_year_filter(self):
        args = self._parse("query", "BERT", "--year", "2023")
        assert args.year == 2023

    def test_query_with_source_filter(self):
        args = self._parse("query", "BERT", "--source", "arxiv")
        assert args.source == "arxiv"

    def test_lint_command(self):
        args = self._parse("lint")
        assert args.command == "lint"

    def test_status_command(self):
        args = self._parse("status")
        assert args.command == "status"

    def test_rebuild_requires_confirm(self):
        args = self._parse("rebuild")
        assert args.confirm is False

    def test_rebuild_with_confirm(self):
        args = self._parse("rebuild", "--confirm")
        assert args.confirm is True

    def test_verbose_flag(self):
        args = self._parse("-v", "query", "What is BERT?")
        assert args.verbose is True

    def test_missing_command_exits(self):
        from interfaces.cli import build_parser
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


# ---------------------------------------------------------------------------
# CLI — command handlers
# ---------------------------------------------------------------------------

class TestCLICommands:
    def _make_args(self, **kwargs):
        """Build a minimal Namespace for a command handler."""
        import argparse
        defaults = {"verbose": False}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_cmd_ingest_success(self, capsys):
        from interfaces.cli import cmd_ingest

        mock_result = {
            "papers_processed": ["arxiv:1", "arxiv:2"],
            "chunks_indexed":   42,
            "errors":           [],
        }
        args = self._make_args(query=["attention", "transformers"])
        with patch("interfaces.cli.run_ingest", return_value=mock_result):
            rc = cmd_ingest(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "2" in out          # 2 papers
        assert "42" in out         # 42 chunks

    def test_cmd_ingest_shows_errors_in_verbose(self, capsys):
        from interfaces.cli import cmd_ingest

        mock_result = {
            "papers_processed": ["arxiv:1"],
            "chunks_indexed":   0,
            "errors":           ["parse error for arxiv:1: timeout"],
        }
        args = self._make_args(query=["q"], verbose=True)
        with patch("interfaces.cli.run_ingest", return_value=mock_result):
            cmd_ingest(args)

        out = capsys.readouterr().out
        assert "parse error" in out

    def test_cmd_query_prints_answer(self, capsys):
        from interfaces.cli import cmd_query

        mock_result = {
            "answer":     "Attention is a mechanism that…",
            "confidence": 0.9,
            "cache_hit":  False,
            "is_grounded": True,
            "filed_page_path": None,
            "sources":    [],
        }
        args = self._make_args(question=["What", "is", "attention?"],
                               year=None, source=None)
        with patch("interfaces.cli.run_query", return_value=mock_result):
            rc = cmd_query(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Attention is a mechanism" in out
        assert "90%" in out

    def test_cmd_query_shows_cache_hit(self, capsys):
        from interfaces.cli import cmd_query

        mock_result = {
            "answer": "Cached answer.", "confidence": 0.8,
            "cache_hit": True, "is_grounded": True,
            "filed_page_path": None, "sources": [],
        }
        args = self._make_args(question=["q"], year=None, source=None)
        with patch("interfaces.cli.run_query", return_value=mock_result):
            cmd_query(args)

        out = capsys.readouterr().out
        assert "cache hit" in out

    def test_cmd_query_shows_filed_path(self, capsys):
        from interfaces.cli import cmd_query

        mock_result = {
            "answer": "A.", "confidence": 0.9,
            "cache_hit": False, "is_grounded": True,
            "filed_page_path": "synthesis/query-answers/2026-04-17-q.md",
            "sources": [],
        }
        args = self._make_args(question=["q"], year=None, source=None)
        with patch("interfaces.cli.run_query", return_value=mock_result):
            cmd_query(args)

        out = capsys.readouterr().out
        assert "synthesis/query-answers" in out

    def test_cmd_lint_prints_summary(self, capsys):
        from interfaces.cli import cmd_lint

        mock_result = {
            "orphans":              ["concepts/a.md"],
            "contradictions":       [],
            "stale_claims":         ["old claim"],
            "missing_concept_pages": [],
            "gaps":                 ["Search for X"],
            "report_path":          "synthesis/lint-2026-04-17.md",
        }
        args = self._make_args()
        with patch("interfaces.cli.run_lint", return_value=mock_result):
            rc = cmd_lint(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "1" in out          # 1 orphan
        assert "Search for X" in out

    def test_cmd_status_runs(self, capsys):
        from interfaces.cli import cmd_status

        args = self._make_args()
        with patch("interfaces.cli.services.print_service_status"):
            rc = cmd_status(args)

        assert rc == 0

    def test_cmd_rebuild_without_confirm(self, capsys):
        from interfaces.cli import cmd_rebuild

        args = self._make_args(confirm=False)
        rc   = cmd_rebuild(args)
        assert rc == 1
        out  = capsys.readouterr().out
        assert "--confirm" in out

    def test_cmd_rebuild_with_confirm(self, tmp_path, capsys):
        from interfaces.cli import cmd_rebuild

        args = self._make_args(confirm=True)
        with (
            patch("interfaces.cli.Config.vector_db_dir",  return_value=tmp_path / "vdb"),
            patch("interfaces.cli.Config.bm25_index_dir", return_value=tmp_path / "bm25"),
            patch("interfaces.cli.Config.wiki_dir",       return_value=tmp_path / "wiki"),
            patch("interfaces.cli.Config.cache_db_path",  return_value=tmp_path / "cache" / "q.db"),
            patch("interfaces.cli.Config.ensure_dirs"),
            patch("interfaces.cli._init_wiki"),
        ):
            rc = cmd_rebuild(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Rebuild complete" in out


# ---------------------------------------------------------------------------
# CLI — _init_wiki helper
# ---------------------------------------------------------------------------

class TestInitWiki:
    def test_creates_index_and_log(self, tmp_path):
        from interfaces.cli import _init_wiki

        with patch("interfaces.cli.Config.wiki_dir", return_value=tmp_path):
            _init_wiki()

        assert (tmp_path / "index.md").exists()
        assert (tmp_path / "log.md").exists()

    def test_index_has_required_sections(self, tmp_path):
        from interfaces.cli import _init_wiki

        with patch("interfaces.cli.Config.wiki_dir", return_value=tmp_path):
            _init_wiki()

        content = (tmp_path / "index.md").read_text()
        for section in ("Papers", "Concepts", "Methods", "Debates", "Synthesis"):
            assert f"## {section}" in content

    def test_does_not_overwrite_existing(self, tmp_path):
        from interfaces.cli import _init_wiki

        existing = tmp_path / "index.md"
        existing.write_text("# Existing index\n")

        with patch("interfaces.cli.Config.wiki_dir", return_value=tmp_path):
            _init_wiki()

        assert existing.read_text() == "# Existing index\n"


# ---------------------------------------------------------------------------
# Gradio backend functions
# ---------------------------------------------------------------------------

class TestGradioBackends:
    def test_do_query_returns_answer(self):
        from interfaces.gradio_interface import _do_query

        mock_result = {
            "answer": "The answer.", "confidence": 0.85,
            "cache_hit": False, "is_grounded": True,
            "filed_page_path": None,
            "sources": [{"paper_id": "arxiv:1", "title": "Paper A", "year": 2023}],
        }
        with patch("interfaces.gradio_interface.run_query", return_value=mock_result):
            answer_md, sources_md = _do_query("What is attention?", "", "")

        assert "The answer." in answer_md
        assert "85%" in answer_md
        assert "Paper A" in sources_md

    def test_do_query_empty_question(self):
        from interfaces.gradio_interface import _do_query

        answer_md, sources_md = _do_query("", "", "")
        assert "Please enter" in answer_md

    def test_do_query_year_filter_parsed(self):
        from interfaces.gradio_interface import _do_query

        mock_result = {
            "answer": "A.", "confidence": 0.7,
            "cache_hit": False, "is_grounded": True,
            "filed_page_path": None, "sources": [],
        }
        with patch("interfaces.gradio_interface.run_query", return_value=mock_result) as mock_q:
            _do_query("question", "2023", "arxiv")

        args, kwargs = mock_q.call_args
        year_passed = kwargs.get("year_filter") if kwargs else (args[1] if len(args) > 1 else None)
        assert year_passed == 2023

    def test_do_query_cache_hit_badge(self):
        from interfaces.gradio_interface import _do_query

        mock_result = {
            "answer": "Cached.", "confidence": 0.9,
            "cache_hit": True, "is_grounded": True,
            "filed_page_path": None, "sources": [],
        }
        with patch("interfaces.gradio_interface.run_query", return_value=mock_result):
            answer_md, _ = _do_query("q", "", "")

        assert "cache hit" in answer_md

    def test_do_query_low_confidence_warning(self):
        from interfaces.gradio_interface import _do_query

        mock_result = {
            "answer": "Uncertain.", "confidence": 0.3,
            "cache_hit": False, "is_grounded": False,
            "filed_page_path": None, "sources": [],
        }
        with patch("interfaces.gradio_interface.run_query", return_value=mock_result):
            answer_md, _ = _do_query("q", "", "")

        assert "low confidence" in answer_md.lower()

    def test_do_ingest_returns_summary(self):
        from interfaces.gradio_interface import _do_ingest

        mock_result = {
            "papers_processed": ["a", "b", "c"],
            "chunks_indexed":   30,
            "errors":           [],
        }
        with patch("interfaces.gradio_interface.run_ingest", return_value=mock_result):
            result = _do_ingest("attention transformers")

        assert "3" in result
        assert "30" in result

    def test_do_ingest_empty_query(self):
        from interfaces.gradio_interface import _do_ingest

        result = _do_ingest("")
        assert "Please enter" in result

    def test_do_lint_returns_summary(self):
        from interfaces.gradio_interface import _do_lint

        mock_result = {
            "orphans": ["a.md"], "contradictions": [],
            "stale_claims": [], "missing_concept_pages": [],
            "gaps": ["Search for X"],
            "report_path": "synthesis/lint-today.md",
        }
        with patch("interfaces.gradio_interface.run_lint", return_value=mock_result):
            result = _do_lint()

        assert "1" in result          # 1 orphan
        assert "Search for X" in result
        assert "synthesis/lint-today.md" in result

    def test_do_status_contains_config(self):
        from interfaces.gradio_interface import _do_status
        import services

        with patch.object(services, "grobid_status", return_value={"running": False, "url": "http://localhost:8070"}):
            with patch.object(services, "marker_status", return_value={"installed": True}):
                result = _do_status()

        assert "Config" in result
        assert "LLM" in result


# ---------------------------------------------------------------------------
# Gradio — build_app (import-only; no browser launched)
# ---------------------------------------------------------------------------

class TestGradioBuildApp:
    def test_build_app_returns_blocks(self):
        try:
            import gradio as gr
        except ImportError:
            pytest.skip("gradio not installed")

        from interfaces.gradio_interface import build_app
        app = build_app()
        assert app is not None

    def test_build_app_raises_without_gradio(self):
        import sys
        orig = sys.modules.pop("gradio", None)
        try:
            # Force re-import without gradio
            import importlib
            import interfaces.gradio_interface as gi
            importlib.reload(gi)
            with pytest.raises(ImportError, match="gradio"):
                gi.build_app()
        finally:
            if orig is not None:
                sys.modules["gradio"] = orig


# ---------------------------------------------------------------------------
# Desktop app — backend functions (no GUI event loop)
# ---------------------------------------------------------------------------

class TestDesktopBackends:
    def test_run_query_returns_answer_and_sources(self):
        from interfaces.desktop_app import _run_query

        mock_result = {
            "answer": "The answer.", "confidence": 0.8,
            "cache_hit": False, "is_grounded": True,
            "filed_page_path": None,
            "sources": [{"paper_id": "arxiv:1", "title": "Paper A", "year": 2023}],
        }
        with patch("interfaces.desktop_app.run_query", return_value=mock_result):
            answer, sources = _run_query("What is attention?", None, None)

        assert "The answer." in answer
        assert "Paper A" in sources

    def test_run_ingest_returns_summary(self):
        from interfaces.desktop_app import _run_ingest

        mock_result = {
            "papers_processed": ["a", "b"],
            "chunks_indexed":   20,
            "errors":           [],
        }
        with patch("interfaces.desktop_app.run_ingest", return_value=mock_result):
            result = _run_ingest("attention")

        assert "2" in result
        assert "20" in result

    def test_run_lint_returns_summary(self):
        from interfaces.desktop_app import _run_lint

        mock_result = {
            "orphans": [], "contradictions": ["debates/open.md"],
            "stale_claims": [], "missing_concept_pages": [],
            "gaps": ["Search for Y"], "report_path": "synthesis/lint.md",
        }
        with patch("interfaces.desktop_app.run_lint", return_value=mock_result):
            result = _run_lint()

        assert "1" in result
        assert "Search for Y" in result

    def test_run_status_contains_config(self):
        from interfaces.desktop_app import _run_status
        import services

        with patch.object(services, "grobid_status", return_value={"running": False, "url": ""}):
            with patch.object(services, "marker_status", return_value={"installed": False}):
                result = _run_status()

        assert "Config" in result
        assert "Topic" in result

    def test_run_query_cache_hit_annotated(self):
        from interfaces.desktop_app import _run_query

        mock_result = {
            "answer": "Fast answer.", "confidence": 0.9,
            "cache_hit": True, "is_grounded": True,
            "filed_page_path": None, "sources": [],
        }
        with patch("interfaces.desktop_app.run_query", return_value=mock_result):
            answer, _ = _run_query("q", None, None)

        assert "cache hit" in answer
