"""
tests/test_wiki_tools.py — Unit tests for Phase 5 wiki tools.

All tests use a temporary directory as the wiki root — no real wiki/ is
touched.  No LLM calls required.  Pure file I/O.

Tests verify:
  - read_wiki_index / read_wiki_page
  - write_wiki_page (including protection of log.md and path escaping)
  - update_wiki_index (row insertion, dedup, stat updates)
  - append_wiki_log (append-only invariant)
  - list_wiki_pages (category filter)
  - search_wiki (BM25 relevance)
  - Page template helpers (make_paper_page, make_concept_page, etc.)
  - build_link_graph
  - extract_open_questions
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wiki_dir(tmp_path):
    """Create a minimal wiki directory structure and patch Config to use it."""
    d = tmp_path / "wiki"
    for sub in ["papers", "concepts", "methods", "debates", "authors",
                "synthesis", "synthesis/query-answers"]:
        (d / sub).mkdir(parents=True, exist_ok=True)

    # Create minimal index.md and log.md
    (d / "index.md").write_text(
        "# Wiki Index — Test Topic\n\n"
        "Last updated: 2026-01-01 | Papers: 0 | Total pages: 0\n"
        "Sources active: arxiv\n\n"
        "## Papers (0)\n| Page | Title | Year | Source | Tags |\n|---|---|---|---|---|\n\n"
        "## Concepts (0)\n| Page | Summary | Domains |\n|---|---|---|\n\n"
        "## Methods (0)\n| Page | Summary | Domains |\n|---|---|---|\n\n"
        "## Debates (0)\n| Page | Status | Domains |\n|---|---|---|\n\n"
        "## Synthesis (0)\n| Page | Query | Date |\n|---|---|---|\n",
        encoding="utf-8",
    )
    (d / "log.md").write_text(
        "# Wiki Log — Test Topic\n\n"
        "<!-- Append-only. Never edit or delete existing entries. -->\n\n"
        "## [2026-01-01] init | Wiki initialized\nTopic: Test Topic\n",
        encoding="utf-8",
    )

    with patch("tools.wiki_tools.Config") as mock_cfg:
        mock_cfg.wiki_dir.return_value = d
        mock_cfg.TOPIC_NAME = "Test Topic"
        mock_cfg.ACTIVE_SOURCES = ["arxiv"]
        mock_cfg.topic_slug.return_value = "test_topic"
        yield d


# ---------------------------------------------------------------------------
# read_wiki_index
# ---------------------------------------------------------------------------

class TestReadWikiIndex:

    def test_returns_content(self, wiki_dir):
        from tools.wiki_tools import read_wiki_index
        content = read_wiki_index()
        assert "Wiki Index" in content

    def test_returns_empty_when_missing(self, tmp_path):
        with patch("tools.wiki_tools.Config") as mock_cfg:
            mock_cfg.wiki_dir.return_value = tmp_path / "nonexistent"
            from tools.wiki_tools import read_wiki_index
            assert read_wiki_index() == ""


# ---------------------------------------------------------------------------
# read_wiki_page
# ---------------------------------------------------------------------------

class TestReadWikiPage:

    def test_reads_existing_page(self, wiki_dir):
        (wiki_dir / "concepts" / "attention.md").write_text("# Attention\nContent here.")
        from tools.wiki_tools import read_wiki_page
        content = read_wiki_page("concepts/attention.md")
        assert "Attention" in content

    def test_returns_empty_for_missing_page(self, wiki_dir):
        from tools.wiki_tools import read_wiki_page
        assert read_wiki_page("concepts/nonexistent.md") == ""


# ---------------------------------------------------------------------------
# write_wiki_page
# ---------------------------------------------------------------------------

class TestWriteWikiPage:

    def test_writes_new_page(self, wiki_dir):
        from tools.wiki_tools import write_wiki_page
        write_wiki_page("concepts/transformer.md", "# Transformer\nContent.", "test")
        assert (wiki_dir / "concepts" / "transformer.md").exists()

    def test_overwrites_existing_page(self, wiki_dir):
        from tools.wiki_tools import write_wiki_page
        write_wiki_page("concepts/transformer.md", "# v1", "first")
        write_wiki_page("concepts/transformer.md", "# v2", "update")
        content = (wiki_dir / "concepts" / "transformer.md").read_text()
        assert "v2" in content
        assert "v1" not in content

    def test_creates_subdirectory(self, wiki_dir):
        from tools.wiki_tools import write_wiki_page
        write_wiki_page("synthesis/query-answers/2026-01-01-test.md", "# Q", "test")
        assert (wiki_dir / "synthesis" / "query-answers" / "2026-01-01-test.md").exists()

    def test_refuses_to_write_log_md(self, wiki_dir):
        from tools.wiki_tools import write_wiki_page
        with pytest.raises(ValueError, match="protected"):
            write_wiki_page("log.md", "tampered", "bad")

    def test_refuses_path_traversal(self, wiki_dir):
        from tools.wiki_tools import write_wiki_page
        with pytest.raises(ValueError):
            write_wiki_page("../../../etc/passwd", "evil", "attack")

    def test_log_md_unchanged_after_write(self, wiki_dir):
        from tools.wiki_tools import write_wiki_page
        original_log = (wiki_dir / "log.md").read_text()
        try:
            write_wiki_page("log.md", "tampered", "bad")
        except ValueError:
            pass
        assert (wiki_dir / "log.md").read_text() == original_log


# ---------------------------------------------------------------------------
# append_wiki_log
# ---------------------------------------------------------------------------

class TestAppendWikiLog:

    def test_appends_entry(self, wiki_dir):
        from tools.wiki_tools import append_wiki_log
        append_wiki_log("ingest", "arxiv:2301.12345 — Attention Is All You Need",
                        "Parser: grobid | Chunks: 47 | Access: full")
        log = (wiki_dir / "log.md").read_text()
        assert "ingest" in log
        assert "arxiv:2301.12345" in log

    def test_entry_format_has_date_header(self, wiki_dir):
        from tools.wiki_tools import append_wiki_log
        import re
        append_wiki_log("query-filed", "Compare BERT vs GPT",
                        "Confidence: high | Wiki pages read: 3")
        log = (wiki_dir / "log.md").read_text()
        assert re.search(r"## \[\d{4}-\d{2}-\d{2}\] query-filed", log)

    def test_preserves_existing_entries(self, wiki_dir):
        from tools.wiki_tools import append_wiki_log
        append_wiki_log("ingest", "paper A", "detail A")
        append_wiki_log("ingest", "paper B", "detail B")
        log = (wiki_dir / "log.md").read_text()
        assert "paper A" in log
        assert "paper B" in log
        # Original init entry also preserved
        assert "Wiki initialized" in log

    def test_multiple_appends_are_ordered(self, wiki_dir):
        from tools.wiki_tools import append_wiki_log
        append_wiki_log("ingest", "first", "d1")
        append_wiki_log("ingest", "second", "d2")
        log = (wiki_dir / "log.md").read_text()
        assert log.index("first") < log.index("second")

    def test_creates_log_if_missing(self, tmp_path):
        d = tmp_path / "wiki2"
        d.mkdir()
        with patch("tools.wiki_tools.Config") as mock_cfg:
            mock_cfg.wiki_dir.return_value = d
            mock_cfg.TOPIC_NAME = "T"
            mock_cfg.ACTIVE_SOURCES = ["arxiv"]
            from tools.wiki_tools import append_wiki_log
            append_wiki_log("ingest", "test", "detail")
            assert (d / "log.md").exists()


# ---------------------------------------------------------------------------
# update_wiki_index
# ---------------------------------------------------------------------------

class TestUpdateWikiIndex:

    def test_adds_paper_to_papers_section(self, wiki_dir):
        from tools.wiki_tools import update_wiki_index
        update_wiki_index("papers/arxiv-2301-12345.md", "Attention Is All You Need", "papers")
        index = (wiki_dir / "index.md").read_text()
        assert "arxiv-2301-12345" in index

    def test_adds_concept_to_concepts_section(self, wiki_dir):
        from tools.wiki_tools import update_wiki_index
        update_wiki_index("concepts/attention.md", "Scaled dot-product attention", "concepts")
        index = (wiki_dir / "index.md").read_text()
        assert "attention" in index

    def test_updates_paper_count_in_stats(self, wiki_dir):
        from tools.wiki_tools import update_wiki_index
        update_wiki_index("papers/arxiv-2301-12345.md", "Paper A", "papers")
        index = (wiki_dir / "index.md").read_text()
        assert "Papers: 1" in index

    def test_updates_total_pages_in_stats(self, wiki_dir):
        from tools.wiki_tools import update_wiki_index
        update_wiki_index("papers/arxiv-2301-12345.md", "Paper A", "papers")
        update_wiki_index("concepts/attention.md", "Attention concept", "concepts")
        index = (wiki_dir / "index.md").read_text()
        assert "Total pages: 2" in index

    def test_idempotent_same_paper_twice(self, wiki_dir):
        from tools.wiki_tools import update_wiki_index
        update_wiki_index("papers/arxiv-2301-12345.md", "Paper A", "papers")
        update_wiki_index("papers/arxiv-2301-12345.md", "Paper A updated", "papers")
        index = (wiki_dir / "index.md").read_text()
        # Should appear only once
        assert index.count("arxiv-2301-12345") == 1
        assert "Papers: 1" in index

    def test_adds_debate_to_debates_section(self, wiki_dir):
        from tools.wiki_tools import update_wiki_index
        update_wiki_index("debates/attention-vs-rnn.md", "Attention vs RNN", "debates")
        index = (wiki_dir / "index.md").read_text()
        assert "attention-vs-rnn" in index

    def test_updates_last_updated_date(self, wiki_dir):
        from tools.wiki_tools import update_wiki_index
        from datetime import date
        update_wiki_index("papers/arxiv-test.md", "Test", "papers")
        index = (wiki_dir / "index.md").read_text()
        today = date.today().isoformat()
        assert today in index

    def test_multiple_papers_all_appear(self, wiki_dir):
        from tools.wiki_tools import update_wiki_index
        update_wiki_index("papers/arxiv-001.md", "Paper One", "papers")
        update_wiki_index("papers/arxiv-002.md", "Paper Two", "papers")
        update_wiki_index("papers/arxiv-003.md", "Paper Three", "papers")
        index = (wiki_dir / "index.md").read_text()
        assert "arxiv-001" in index
        assert "arxiv-002" in index
        assert "arxiv-003" in index
        assert "Papers: 3" in index


# ---------------------------------------------------------------------------
# list_wiki_pages
# ---------------------------------------------------------------------------

class TestListWikiPages:

    def _populate(self, wiki_dir):
        (wiki_dir / "papers" / "arxiv-001.md").write_text("# P1")
        (wiki_dir / "papers" / "arxiv-002.md").write_text("# P2")
        (wiki_dir / "concepts" / "attention.md").write_text("# C1")
        (wiki_dir / "methods" / "transformer.md").write_text("# M1")

    def test_list_all(self, wiki_dir):
        self._populate(wiki_dir)
        from tools.wiki_tools import list_wiki_pages
        pages = list_wiki_pages()
        assert len(pages) == 4
        assert not any(p.endswith("index.md") for p in pages)
        assert not any(p.endswith("log.md") for p in pages)

    def test_list_papers_only(self, wiki_dir):
        self._populate(wiki_dir)
        from tools.wiki_tools import list_wiki_pages
        pages = list_wiki_pages("papers")
        assert len(pages) == 2
        assert all("papers/" in p for p in pages)

    def test_list_concepts_only(self, wiki_dir):
        self._populate(wiki_dir)
        from tools.wiki_tools import list_wiki_pages
        pages = list_wiki_pages("concepts")
        assert len(pages) == 1

    def test_empty_category_returns_empty(self, wiki_dir):
        from tools.wiki_tools import list_wiki_pages
        assert list_wiki_pages("debates") == []

    def test_nonexistent_wiki_returns_empty(self, tmp_path):
        with patch("tools.wiki_tools.Config") as mock_cfg:
            mock_cfg.wiki_dir.return_value = tmp_path / "no_wiki"
            from tools.wiki_tools import list_wiki_pages
            assert list_wiki_pages() == []


# ---------------------------------------------------------------------------
# search_wiki
# ---------------------------------------------------------------------------

class TestSearchWiki:

    def _populate(self, wiki_dir):
        (wiki_dir / "concepts" / "attention.md").write_text(
            "# Attention Mechanism\nScaled dot-product attention allows transformers to focus."
        )
        (wiki_dir / "concepts" / "rnn.md").write_text(
            "# Recurrent Neural Networks\nRNNs process sequences step by step."
        )
        (wiki_dir / "methods" / "transformer.md").write_text(
            "# Transformer Architecture\nEncoder-decoder with multi-head attention."
        )

    def test_returns_relevant_results(self, wiki_dir):
        self._populate(wiki_dir)
        from tools.wiki_tools import search_wiki
        results = search_wiki("attention transformer", top_k=3)
        assert len(results) > 0
        top_page = results[0]["page_path"]
        assert "attention" in top_page or "transformer" in top_page

    def test_results_have_required_keys(self, wiki_dir):
        self._populate(wiki_dir)
        from tools.wiki_tools import search_wiki
        results = search_wiki("attention", top_k=3)
        for r in results:
            assert "page_path" in r
            assert "snippet" in r
            assert "score" in r

    def test_returns_empty_on_empty_wiki(self, wiki_dir):
        from tools.wiki_tools import search_wiki
        results = search_wiki("anything")
        assert results == []

    def test_top_k_limits_results(self, wiki_dir):
        self._populate(wiki_dir)
        from tools.wiki_tools import search_wiki
        results = search_wiki("attention", top_k=1)
        assert len(results) <= 1

    def test_scores_positive(self, wiki_dir):
        self._populate(wiki_dir)
        from tools.wiki_tools import search_wiki
        results = search_wiki("attention mechanism", top_k=3)
        assert all(r["score"] > 0 for r in results)


# ---------------------------------------------------------------------------
# Page template helpers
# ---------------------------------------------------------------------------

class TestPageTemplates:

    def test_make_paper_page_has_frontmatter(self):
        from tools.wiki_tools import make_paper_page
        page = make_paper_page(
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
            summary="Proposed the Transformer model.",
        )
        assert "paper_id: arxiv:2301.12345" in page
        assert "math_fraction: 0.050" in page
        assert "is_abstract_only: false" in page
        assert "## Summary" in page
        assert "## Open Questions Raised" in page

    def test_make_paper_page_abstract_only_has_warning(self):
        from tools.wiki_tools import make_paper_page
        page = make_paper_page(
            paper_id="pubmed:123",
            title="Paywalled Paper",
            authors=[],
            year=2024,
            source="pubmed",
            venue="Nature",
            parser_used="pymupdf",
            math_fraction=0.0,
            is_abstract_only=True,
            tags=[],
            summary="Abstract only.",
        )
        assert "⚠️ ABSTRACT ONLY" in page
        assert "is_abstract_only: true" in page

    def test_make_concept_page_has_key_sections(self):
        from tools.wiki_tools import make_concept_page
        page = make_concept_page("Self-Attention", tags=["transformers"])
        assert "concept: " in page
        assert "## Definition" in page
        assert "## Key Papers" in page
        assert "## Cross-domain Notes" in page
        assert "## Open Debates" in page

    def test_make_method_page_has_comparison_table(self):
        from tools.wiki_tools import make_method_page
        page = make_method_page("Transformer", domains=["cs"])
        assert "method: " in page
        assert "## Comparison Table" in page
        assert "## When to use" in page

    def test_make_debate_page_has_positions(self):
        from tools.wiki_tools import make_debate_page
        page = make_debate_page("Attention vs RNN")
        assert "## Position A" in page
        assert "## Position B" in page
        assert "## Resolution" in page
        assert "status: open" in page

    def test_make_synthesis_page_has_sources(self):
        from tools.wiki_tools import make_synthesis_page
        page = make_synthesis_page(
            query="What is self-attention?",
            answer="Self-attention allows each token to attend to all others.",
            sources_wiki=["concepts/attention.md"],
            sources_chromadb=["arxiv:2301.12345"],
        )
        assert "query: " in page
        assert "## Sources" in page
        assert "concepts/attention.md" in page
        assert "arxiv:2301.12345" in page


# ---------------------------------------------------------------------------
# build_link_graph
# ---------------------------------------------------------------------------

class TestBuildLinkGraph:

    def test_finds_inbound_links(self, wiki_dir):
        (wiki_dir / "papers" / "arxiv-001.md").write_text(
            "# Paper\nSee [[concepts/attention.md]] and [[methods/transformer.md]]."
        )
        (wiki_dir / "concepts" / "attention.md").write_text("# Attention\n")
        (wiki_dir / "methods" / "transformer.md").write_text("# Transformer\n")

        from tools.wiki_tools import build_link_graph
        graph = build_link_graph()
        assert "papers/arxiv-001.md" in graph.get("concepts/attention.md", [])
        assert "papers/arxiv-001.md" in graph.get("methods/transformer.md", [])

    def test_orphan_has_empty_inbound(self, wiki_dir):
        (wiki_dir / "concepts" / "orphan.md").write_text("# Orphan\nNo one links here.")
        from tools.wiki_tools import build_link_graph
        graph = build_link_graph()
        assert graph.get("concepts/orphan.md", []) == []


# ---------------------------------------------------------------------------
# extract_open_questions
# ---------------------------------------------------------------------------

class TestExtractOpenQuestions:

    def test_extracts_questions(self, wiki_dir):
        (wiki_dir / "papers" / "arxiv-001.md").write_text(
            "---\npaper_id: arxiv:001\n---\n\n"
            "## Open Questions Raised\n"
            "- Why does attention scale poorly with sequence length?\n"
            "- Can transformers replace all RNNs?\n"
        )
        from tools.wiki_tools import extract_open_questions
        questions = extract_open_questions()
        texts = [q["question"] for q in questions]
        assert any("attention" in t.lower() for t in texts)
        assert any("RNN" in t for t in texts)

    def test_returns_empty_when_no_papers(self, wiki_dir):
        from tools.wiki_tools import extract_open_questions
        assert extract_open_questions() == []

    def test_extracts_paper_id(self, wiki_dir):
        (wiki_dir / "papers" / "arxiv-001.md").write_text(
            "---\npaper_id: arxiv:001\n---\n\n"
            "## Open Questions Raised\n- A question?\n"
        )
        from tools.wiki_tools import extract_open_questions
        questions = extract_open_questions()
        assert questions[0]["paper_id"] == "arxiv:001"
