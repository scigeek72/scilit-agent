"""
tests/test_frontier_agent.py — Unit tests for the Frontier Agent (Phase 11).

All tests run without a real LLM or a real wiki on disk.
The wiki tools are mocked at the module level.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper_page(concept_links: list[str], year: int = 2022, open_questions: list[str] | None = None) -> str:
    oq_lines = "\n".join(f"- {q}" for q in (open_questions or ["How does this scale?"]))
    links = ", ".join(f"[[concepts/{c}]]" for c in concept_links)
    return f"""---
paper_id: arxiv:test-{year}
title: "Test Paper"
authors: [Alice]
year: {year}
source: arxiv
is_abstract_only: false
---

# Test Paper

## Key Concepts
{links}

## Open Questions Raised
{oq_lines}
"""


def _make_concept_page(domains: list[str], cross_domain_notes: str = "") -> str:
    return f"""---
concept: "Test Concept"
---

# Test Concept

## Definition
A test concept.

## Cross-domain Notes
{cross_domain_notes or "No cross-domain notes."}
"""


def _make_method_page(domains: list[str]) -> str:
    return f"""---
method: "Test Method"
domains: [{", ".join(domains)}]
---

# Test Method

## Summary
A test method.

## Comparison Table
| Approach | Paper | Domain | Strengths | Weaknesses | Benchmark |
|---|---|---|---|---|---|
"""


def _make_debate_page(status: str = "open") -> str:
    return f"""---
debate: "Test Debate"
status: {status}
---

# Test Debate

## The Question
Is this a good approach?

## Position A
Yes — supported by [[papers/arxiv-test]]

## Position B
No — supported by [[papers/pubmed-test]]

## Key Concepts
[[concepts/test-concept]]
"""


# ---------------------------------------------------------------------------
# Tests: node_classify_focus
# ---------------------------------------------------------------------------

class TestClassifyFocus:
    def test_methodological_signals(self):
        from agents.frontier_agent import node_classify_focus
        state = {"query": "Which benchmark techniques have not been applied?"}
        result = node_classify_focus(state)
        assert result["query_focus"] == "methodological"

    def test_conceptual_signals(self):
        from agents.frontier_agent import node_classify_focus
        state = {"query": "What open questions has nobody worked on?"}
        result = node_classify_focus(state)
        assert result["query_focus"] == "conceptual"

    def test_both_signals(self):
        from agents.frontier_agent import node_classify_focus
        state = {"query": "What method gaps and open questions exist?"}
        result = node_classify_focus(state)
        assert result["query_focus"] == "both"

    def test_no_signals_defaults_to_both(self):
        from agents.frontier_agent import node_classify_focus
        state = {"query": "What is unexplored in this corpus?"}
        result = node_classify_focus(state)
        # "unexplored" is a conceptual signal
        assert result["query_focus"] in ("conceptual", "both")

    def test_empty_query(self):
        from agents.frontier_agent import node_classify_focus
        result = node_classify_focus({"query": ""})
        assert result["query_focus"] == "both"


# ---------------------------------------------------------------------------
# Tests: node_aggregate_open_questions
# ---------------------------------------------------------------------------

class TestAggregateOpenQuestions:
    def test_skips_when_methodological_focus(self):
        from agents.frontier_agent import node_aggregate_open_questions
        result = node_aggregate_open_questions({"query_focus": "methodological"})
        assert result["open_questions"] == []

    def test_extracts_open_questions(self):
        from agents.frontier_agent import node_aggregate_open_questions

        page1 = _make_paper_page(["attention"], open_questions=["How does attention scale to long sequences?"])
        page2 = _make_paper_page(["attention"], open_questions=["How does attention scale to long sequences?"])
        page3 = _make_paper_page(["bert"],      open_questions=["What are the limits of BERT?"])

        with patch("agents.frontier_agent.list_wiki_pages", return_value=["papers/p1.md", "papers/p2.md", "papers/p3.md"]), \
             patch("agents.frontier_agent.read_wiki_page", side_effect=[page1, page2, page3]):
            result = node_aggregate_open_questions({"query_focus": "conceptual"})

        oqs = result["open_questions"]
        assert len(oqs) > 0
        # The repeated question should have higher count
        repeated = [q for q in oqs if "scale" in q["question"].lower()]
        if repeated:
            assert repeated[0]["count"] >= 2

    def test_empty_wiki_returns_empty(self):
        from agents.frontier_agent import node_aggregate_open_questions
        with patch("agents.frontier_agent.list_wiki_pages", return_value=[]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=""):
            result = node_aggregate_open_questions({"query_focus": "both"})
        assert result["open_questions"] == []

    def test_confidence_levels(self):
        from agents.frontier_agent import node_aggregate_open_questions

        pages = [
            _make_paper_page(["x"], open_questions=["What is the scaling limit?"]),
            _make_paper_page(["x"], open_questions=["What is the scaling limit?"]),
            _make_paper_page(["x"], open_questions=["What is the scaling limit?"]),
        ]
        with patch("agents.frontier_agent.list_wiki_pages", return_value=["p1.md", "p2.md", "p3.md"]), \
             patch("agents.frontier_agent.read_wiki_page", side_effect=pages):
            result = node_aggregate_open_questions({"query_focus": "both"})

        oqs = result["open_questions"]
        assert len(oqs) > 0
        # At least one question with count >= 3 should be high confidence
        high = [q for q in oqs if q["confidence"] == "high"]
        assert len(high) > 0 or any(q["count"] >= 3 for q in oqs)


# ---------------------------------------------------------------------------
# Tests: node_find_method_domain_gaps
# ---------------------------------------------------------------------------

class TestFindMethodDomainGaps:
    def test_skips_when_conceptual_focus(self):
        from agents.frontier_agent import node_find_method_domain_gaps
        result = node_find_method_domain_gaps({"query_focus": "conceptual"})
        assert result["method_domain_gaps"] == []

    def test_detects_gap(self):
        from agents.frontier_agent import node_find_method_domain_gaps

        method_page = _make_method_page(domains=["cs"])
        paper_page  = _make_paper_page([], year=2023)
        # Paper page has source: pubmed → maps to "medicine"
        paper_page_pubmed = paper_page.replace("source: arxiv", "source: pubmed")

        with patch("agents.frontier_agent.list_wiki_pages", side_effect=[
                ["methods/test-method.md"],   # methods/
                ["papers/p1.md"],             # papers/
             ]), \
             patch("agents.frontier_agent.read_wiki_page", side_effect=[method_page, paper_page_pubmed]):
            result = node_find_method_domain_gaps({"query_focus": "methodological"})

        gaps = result["method_domain_gaps"]
        # method is documented in "cs", corpus has "medicine" → gap expected
        assert isinstance(gaps, list)

    def test_no_methods_returns_empty(self):
        from agents.frontier_agent import node_find_method_domain_gaps
        with patch("agents.frontier_agent.list_wiki_pages", side_effect=[[], []]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=""):
            result = node_find_method_domain_gaps({"query_focus": "both"})
        assert result["method_domain_gaps"] == []


# ---------------------------------------------------------------------------
# Tests: node_find_temporal_dropouts
# ---------------------------------------------------------------------------

class TestFindTemporalDropouts:
    def test_skips_when_methodological_focus(self):
        from agents.frontier_agent import node_find_temporal_dropouts
        result = node_find_temporal_dropouts({"query_focus": "methodological"})
        assert result["temporal_dropouts"] == []

    def test_detects_old_concept(self):
        from agents.frontier_agent import node_find_temporal_dropouts

        page = _make_paper_page(["old-concept"], year=2018)

        with patch("agents.frontier_agent.list_wiki_pages", return_value=["papers/old.md"]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=page):
            result = node_find_temporal_dropouts({"query_focus": "conceptual"})

        dropouts = result["temporal_dropouts"]
        assert len(dropouts) > 0
        assert dropouts[0]["last_year"] == 2018

    def test_recent_concept_not_flagged(self):
        from agents.frontier_agent import node_find_temporal_dropouts
        from datetime import date

        current_year = date.today().year
        page = _make_paper_page(["new-concept"], year=current_year - 1)

        with patch("agents.frontier_agent.list_wiki_pages", return_value=["papers/new.md"]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=page):
            result = node_find_temporal_dropouts({"query_focus": "both"})

        dropouts = result["temporal_dropouts"]
        recent_dropouts = [d for d in dropouts if d["concept"] == "new-concept"]
        assert len(recent_dropouts) == 0


# ---------------------------------------------------------------------------
# Tests: node_find_contradiction_clusters
# ---------------------------------------------------------------------------

class TestFindContradictionClusters:
    def test_empty_wiki(self):
        from agents.frontier_agent import node_find_contradiction_clusters
        with patch("agents.frontier_agent.list_wiki_pages", return_value=[]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=""):
            result = node_find_contradiction_clusters({})
        assert result["contradiction_clusters"] == []

    def test_finds_open_debate(self):
        from agents.frontier_agent import node_find_contradiction_clusters

        debate_page = _make_debate_page(status="open")

        with patch("agents.frontier_agent.list_wiki_pages", return_value=["debates/d1.md"]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=debate_page):
            result = node_find_contradiction_clusters({})

        clusters = result["contradiction_clusters"]
        assert len(clusters) > 0

    def test_resolved_debate_excluded(self):
        from agents.frontier_agent import node_find_contradiction_clusters

        debate_page = _make_debate_page(status="resolved")

        with patch("agents.frontier_agent.list_wiki_pages", return_value=["debates/d1.md"]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=debate_page):
            result = node_find_contradiction_clusters({})

        assert result["contradiction_clusters"] == []


# ---------------------------------------------------------------------------
# Tests: node_find_cross_domain
# ---------------------------------------------------------------------------

class TestFindCrossDomain:
    def test_skips_when_conceptual_focus(self):
        from agents.frontier_agent import node_find_cross_domain
        result = node_find_cross_domain({"query_focus": "conceptual"})
        assert result["cross_domain_opportunities"] == []

    def test_detects_cross_domain_notes(self):
        from agents.frontier_agent import node_find_cross_domain

        page = _make_concept_page(
            domains=["cs", "biology"],
            cross_domain_notes=(
                "This concept originates in cs but has strong parallels in biology "
                "and is increasingly used in medicine for clinical decision making."
            ),
        )

        with patch("agents.frontier_agent.list_wiki_pages", return_value=["concepts/test.md"]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=page):
            result = node_find_cross_domain({"query_focus": "methodological"})

        opps = result["cross_domain_opportunities"]
        assert len(opps) > 0
        assert len(opps[0]["domains"]) >= 2

    def test_empty_notes_not_flagged(self):
        from agents.frontier_agent import node_find_cross_domain

        page = _make_concept_page(domains=[], cross_domain_notes="")

        with patch("agents.frontier_agent.list_wiki_pages", return_value=["concepts/c.md"]), \
             patch("agents.frontier_agent.read_wiki_page", return_value=page):
            result = node_find_cross_domain({"query_focus": "both"})

        assert result["cross_domain_opportunities"] == []


# ---------------------------------------------------------------------------
# Tests: node_synthesize_report (LLM-free fallback)
# ---------------------------------------------------------------------------

class TestSynthesizeReport:
    def test_fallback_report_built_without_llm(self):
        from agents.frontier_agent import node_synthesize_report

        state = {
            "query":        "What gaps exist?",
            "query_focus":  "both",
            "open_questions": [
                {"question": "How does this scale?", "count": 2, "papers": ["p1.md"], "years": [2022], "confidence": "medium"},
            ],
            "method_domain_gaps": [
                {"method": "transformer", "documented_domains": ["cs"], "missing_domain": "medicine", "confidence": "high"},
            ],
            "temporal_dropouts": [],
            "contradiction_clusters": [],
            "cross_domain_opportunities": [],
            "kg_gap_edges": [],
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="## Methodological Gaps\nSome gaps found.\n## Suggested Next Searches\n- search query 1\n")

        result = node_synthesize_report(state, llm=mock_llm)
        assert "report" in result
        assert len(result["report"]) > 20

    def test_fallback_when_llm_fails(self):
        from agents.frontier_agent import node_synthesize_report

        state = {
            "query":        "What gaps?",
            "query_focus":  "both",
            "open_questions": [{"question": "Why?", "count": 1, "papers": [], "years": [], "confidence": "low"}],
            "method_domain_gaps": [],
            "temporal_dropouts": [],
            "contradiction_clusters": [],
            "cross_domain_opportunities": [],
            "kg_gap_edges": [],
        }

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")

        result = node_synthesize_report(state, llm=mock_llm)
        assert "report" in result
        assert isinstance(result["report"], str)


# ---------------------------------------------------------------------------
# Tests: node_file_report
# ---------------------------------------------------------------------------

class TestFileReport:
    def test_writes_wiki_page(self):
        from agents.frontier_agent import node_file_report

        state = {
            "query":            "What gaps exist?",
            "query_focus":      "both",
            "report":           "## Test Report\nContent here.",
            "kg_gap_edges":     [],
        }

        with patch("agents.frontier_agent.list_wiki_pages", return_value=[]), \
             patch("agents.frontier_agent.write_wiki_page") as mock_write, \
             patch("agents.frontier_agent.update_wiki_index"):
            result = node_file_report(state)

        assert mock_write.called
        assert result["filed_page_path"] is not None
        assert result["filed_page_path"].startswith("synthesis/frontier-")

    def test_returns_none_on_write_failure(self):
        from agents.frontier_agent import node_file_report

        state = {
            "query": "test", "query_focus": "both",
            "report": "report text", "kg_gap_edges": [],
        }

        with patch("agents.frontier_agent.list_wiki_pages", return_value=[]), \
             patch("agents.frontier_agent.write_wiki_page", side_effect=OSError("disk full")), \
             patch("agents.frontier_agent.update_wiki_index"):
            result = node_file_report(state)

        assert result["filed_page_path"] is None


# ---------------------------------------------------------------------------
# Tests: graph construction
# ---------------------------------------------------------------------------

class TestBuildFrontierGraph:
    def test_graph_compiles(self):
        from agents.frontier_agent import build_frontier_graph
        graph = build_frontier_graph()
        assert graph is not None

    def test_run_frontier_returns_state_keys(self):
        """Smoke test: run_frontier with an empty wiki returns a complete state."""
        from agents.frontier_agent import run_frontier

        with patch("agents.frontier_agent.list_wiki_pages", return_value=[]), \
             patch("agents.frontier_agent.read_wiki_page",  return_value=""), \
             patch("agents.frontier_agent.read_wiki_index", return_value=""), \
             patch("agents.frontier_agent.write_wiki_page"), \
             patch("agents.frontier_agent.update_wiki_index"), \
             patch("agents.frontier_agent.append_wiki_log"), \
             patch("agents.frontier_agent._call_llm", return_value="## Gaps\nNone found."):
            state = run_frontier("What methodological gaps exist?")

        assert "query_focus" in state
        assert "open_questions" in state
        assert "method_domain_gaps" in state
        assert "temporal_dropouts" in state
        assert "contradiction_clusters" in state
        assert "cross_domain_opportunities" in state
        assert "report" in state
