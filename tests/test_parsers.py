"""
tests/test_parsers.py — Unit tests for Phase 3 parsers.

Covers:
- ParsedPaper / empty_parsed_paper (base.py)
- PyMuPDFParser — math_fraction, sections, refs, figures, tables
- GrobidParser — TEI XML parsing (no live Grobid service needed)
- MarkerParser — markdown parsing (no live marker binary needed)
- router.route_and_parse — routing logic, fallback, abstract-only shortcut

No external services or real PDFs required.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.base import ParsedPaper, empty_parsed_paper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METADATA = {
    "paper_id": "arxiv:2301.12345",
    "title": "Attention Is All You Need",
    "authors": ["Vaswani, A.", "Shazeer, N."],
    "abstract": "The dominant sequence transduction models...",
    "year": 2017,
    "source": "arxiv",
    "is_abstract_only": False,
}

SAMPLE_TEXT = textwrap.dedent("""\
    Attention Is All You Need

    Abstract
    The dominant sequence transduction models are based on recurrent neural networks.

    1. Introduction
    Recurrent neural networks, long short-term memory and gated recurrent neural networks
    have been firmly established as state of the art approaches.

    2. Background
    The goal of reducing sequential computation forms the foundation of the Extended
    Neural GPU, ByteNet and ConvS2S.

    Figure 1. The Transformer architecture.

    Table 1. Results on WMT 2014 English-to-German translation.

    References
    [1] Bahdanau, D., Cho, K., Bengio, Y. Neural Machine Translation. 2015.
    [2] Vaswani, A. et al. Attention Is All You Need. NeurIPS 2017.
""")

SAMPLE_MATH_TEXT = textwrap.dedent("""\
    We define the attention function as:
    $\\text{Attention}(Q,K,V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$
    where $Q$, $K$, $V$ are queries, keys, values.
    \\[ \\sum_{i=1}^{n} \\alpha_i = 1 \\]
    \\[ \\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2} \\]
    The gradient ∇L is computed via backpropagation.
    We use \\frac{d}{dx} and \\sum notation throughout.
""")


# ---------------------------------------------------------------------------
# ParsedPaper / empty_parsed_paper
# ---------------------------------------------------------------------------

class TestParsedPaperBase:

    def test_empty_parsed_paper_defaults(self):
        p = empty_parsed_paper()
        assert p["paper_id"] == ""
        assert p["sections"] == []
        assert p["references"] == []
        assert p["figures"] == []
        assert p["tables"] == []
        assert p["equations"] == []
        assert p["math_fraction"] == 0.0
        assert p["is_abstract_only"] is False

    def test_empty_parsed_paper_with_values(self):
        p = empty_parsed_paper(
            paper_id="arxiv:1234",
            title="Test",
            authors=["Alice"],
            year=2022,
            parser_used="grobid",
            math_fraction=0.15,
            is_abstract_only=True,
        )
        assert p["paper_id"] == "arxiv:1234"
        assert p["title"] == "Test"
        assert p["authors"] == ["Alice"]
        assert p["year"] == 2022
        assert p["parser_used"] == "grobid"
        assert p["math_fraction"] == 0.15
        assert p["is_abstract_only"] is True

    def test_parsed_paper_is_dict(self):
        p = empty_parsed_paper()
        assert isinstance(p, dict)

    def test_all_required_keys_present(self):
        required = {
            "paper_id", "title", "authors", "abstract", "year", "source",
            "sections", "references", "figures", "tables", "equations",
            "parser_used", "math_fraction", "is_abstract_only",
        }
        p = empty_parsed_paper()
        assert required.issubset(p.keys())


# ---------------------------------------------------------------------------
# Math fraction estimation
# ---------------------------------------------------------------------------

class TestMathFraction:

    def test_empty_text_returns_zero(self):
        from parsers.pymupdf_parser import _compute_math_fraction
        assert _compute_math_fraction("") == 0.0

    def test_plain_text_has_low_math_fraction(self):
        from parsers.pymupdf_parser import _compute_math_fraction
        plain = "The cat sat on the mat. We observed significant results in the study."
        assert _compute_math_fraction(plain) < 0.05

    def test_math_heavy_text_has_high_math_fraction(self):
        from parsers.pymupdf_parser import _compute_math_fraction
        score = _compute_math_fraction(SAMPLE_MATH_TEXT)
        assert score > 0.05

    def test_fraction_capped_at_one(self):
        from parsers.pymupdf_parser import _compute_math_fraction
        pure_math = "$x$ " * 1000
        assert _compute_math_fraction(pure_math) <= 1.0

    def test_greek_letters_count_as_math(self):
        from parsers.pymupdf_parser import _compute_math_fraction
        text = "The parameters α and β control the rate γ of convergence."
        score = _compute_math_fraction(text)
        assert score > 0.0


# ---------------------------------------------------------------------------
# PyMuPDFParser
# ---------------------------------------------------------------------------

class TestPyMuPDFParser:

    def test_parser_name(self):
        from parsers.pymupdf_parser import PyMuPDFParser
        assert PyMuPDFParser().parser_name == "pymupdf"

    def test_is_available(self):
        from parsers.pymupdf_parser import PyMuPDFParser
        # PyMuPDF is in the env — should be True
        assert PyMuPDFParser().is_available() is True

    def test_abstract_only_returns_partial(self):
        from parsers.pymupdf_parser import PyMuPDFParser
        meta = {**SAMPLE_METADATA, "is_abstract_only": True}
        result = PyMuPDFParser().parse("", meta)
        assert result["is_abstract_only"] is True
        assert result["sections"] == []

    @patch("parsers.pymupdf_parser.fitz")
    def test_parse_returns_valid_schema(self, mock_fitz):
        from parsers.pymupdf_parser import PyMuPDFParser

        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = SAMPLE_TEXT
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        result = PyMuPDFParser().parse("/fake/path.pdf", SAMPLE_METADATA)

        assert result["parser_used"] == "pymupdf"
        assert result["paper_id"] == "arxiv:2301.12345"
        assert isinstance(result["sections"], list)
        assert isinstance(result["references"], list)
        assert isinstance(result["figures"], list)
        assert isinstance(result["tables"], list)
        assert isinstance(result["math_fraction"], float)

    @patch("parsers.pymupdf_parser.fitz")
    def test_fitz_failure_returns_empty_result(self, mock_fitz):
        from parsers.pymupdf_parser import PyMuPDFParser
        mock_fitz.open.side_effect = Exception("cannot open")
        result = PyMuPDFParser().parse("/fake/bad.pdf", SAMPLE_METADATA)
        assert result["sections"] == []
        assert result["parser_used"] == "pymupdf"

    def test_extract_sections_finds_numbered_headings(self):
        from parsers.pymupdf_parser import _extract_sections
        sections = _extract_sections(SAMPLE_TEXT)
        headings = [s["heading"] for s in sections]
        assert any("Introduction" in h for h in headings)

    def test_extract_references_finds_entries(self):
        from parsers.pymupdf_parser import _extract_references
        refs = _extract_references(SAMPLE_TEXT)
        assert len(refs) >= 1
        assert all("year" in r for r in refs)

    def test_extract_figure_captions(self):
        from parsers.pymupdf_parser import _extract_figure_captions
        figs = _extract_figure_captions(SAMPLE_TEXT)
        assert any("Transformer" in f["caption"] for f in figs)

    def test_extract_table_captions(self):
        from parsers.pymupdf_parser import _extract_table_captions
        tables = _extract_table_captions(SAMPLE_TEXT)
        assert len(tables) >= 1

    def test_year_extracted_from_reference(self):
        from parsers.pymupdf_parser import _extract_references
        refs = _extract_references(SAMPLE_TEXT)
        years = [r["year"] for r in refs if r["year"] > 0]
        assert 2015 in years or 2017 in years


# ---------------------------------------------------------------------------
# GrobidParser (TEI XML parsing — no live service)
# ---------------------------------------------------------------------------

SAMPLE_TEI = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Attention Is All You Need</title>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>The dominant sequence transduction models are based on complex RNNs.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head n="1">Introduction</head>
        <p>Recurrent neural networks have been firmly established.</p>
      </div>
      <div>
        <head n="2">Background</head>
        <p>The goal of reducing sequential computation.</p>
      </div>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct xml:id="b0">
            <analytic>
              <title level="a">Neural Machine Translation</title>
              <author>
                <persName><forename>D.</forename><surname>Bahdanau</surname></persName>
              </author>
            </analytic>
            <monogr>
              <imprint><date when="2015"/></imprint>
            </monogr>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


class TestGrobidParser:

    def test_parser_name(self):
        from parsers.grobid_parser import GrobidParser
        assert GrobidParser().parser_name == "grobid"

    @patch("parsers.grobid_parser.requests.get")
    def test_is_available_true(self, mock_get):
        from parsers.grobid_parser import GrobidParser
        mock_get.return_value = MagicMock(status_code=200)
        assert GrobidParser().is_available() is True

    @patch("parsers.grobid_parser.requests.get", side_effect=Exception("down"))
    def test_is_available_false(self, mock_get):
        from parsers.grobid_parser import GrobidParser
        assert GrobidParser().is_available() is False

    def test_abstract_only_returns_partial(self):
        from parsers.grobid_parser import GrobidParser
        meta = {**SAMPLE_METADATA, "is_abstract_only": True}
        result = GrobidParser().parse("", meta)
        assert result["is_abstract_only"] is True
        assert result["sections"] == []

    @patch("parsers.grobid_parser.GrobidParser._call_grobid")
    def test_parse_extracts_title(self, mock_call):
        from parsers.grobid_parser import GrobidParser
        mock_call.return_value = SAMPLE_TEI
        result = GrobidParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert result["title"] == "Attention Is All You Need"

    @patch("parsers.grobid_parser.GrobidParser._call_grobid")
    def test_parse_extracts_abstract(self, mock_call):
        from parsers.grobid_parser import GrobidParser
        mock_call.return_value = SAMPLE_TEI
        result = GrobidParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert "RNN" in result["abstract"] or "recurrent" in result["abstract"].lower()

    @patch("parsers.grobid_parser.GrobidParser._call_grobid")
    def test_parse_extracts_sections(self, mock_call):
        from parsers.grobid_parser import GrobidParser
        mock_call.return_value = SAMPLE_TEI
        result = GrobidParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert len(result["sections"]) == 2
        assert result["sections"][0]["heading"] == "Introduction"
        assert result["sections"][1]["heading"] == "Background"

    @patch("parsers.grobid_parser.GrobidParser._call_grobid")
    def test_parse_extracts_references(self, mock_call):
        from parsers.grobid_parser import GrobidParser
        mock_call.return_value = SAMPLE_TEI
        result = GrobidParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert len(result["references"]) == 1
        assert result["references"][0]["title"] == "Neural Machine Translation"
        assert result["references"][0]["year"] == 2015

    @patch("parsers.grobid_parser.GrobidParser._call_grobid")
    def test_parse_grobid_returns_no_xml(self, mock_call):
        from parsers.grobid_parser import GrobidParser
        mock_call.return_value = None
        result = GrobidParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        # Should return partial result with metadata fields populated
        assert result["parser_used"] == "grobid"
        assert result["sections"] == []

    @patch("parsers.grobid_parser.GrobidParser._call_grobid")
    def test_equations_list_empty_for_grobid(self, mock_call):
        from parsers.grobid_parser import GrobidParser
        mock_call.return_value = SAMPLE_TEI
        result = GrobidParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert result["equations"] == []

    @patch("parsers.grobid_parser.GrobidParser._call_grobid")
    def test_section_levels_from_n_attribute(self, mock_call):
        from parsers.grobid_parser import GrobidParser
        mock_call.return_value = SAMPLE_TEI
        result = GrobidParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert result["sections"][0]["level"] == 1   # n="1"
        assert result["sections"][1]["level"] == 1   # n="2"


# ---------------------------------------------------------------------------
# MarkerParser (markdown parsing — no live binary)
# ---------------------------------------------------------------------------

SAMPLE_MMD = textwrap.dedent("""\
    # Attention Is All You Need

    ## Abstract
    The dominant sequence transduction models are based on complex recurrent networks.

    ## 1 Introduction
    Recurrent neural networks have been firmly established as state of the art.

    ## 2 Background
    The goal of reducing sequential computation.

    We study the attention function:
    \\[ \\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V \\]

    And the loss:
    $$ \\mathcal{L} = -\\sum_{t} \\log p(y_t | y_{<t}, x) $$

    **Figure 1.** The Transformer model architecture with encoder-decoder stacks.

    **Table 1.** BLEU scores on WMT 2014 English-to-German translation.

    ## References
    [1] Bahdanau, D., Cho, K., Bengio, Y. Neural Machine Translation. arXiv 2015.
    [2] Vaswani, A. Attention Is All You Need. NeurIPS 2017.
""")


class TestMarkerParser:

    def test_parser_name(self):
        from parsers.marker_parser import MarkerParser
        assert MarkerParser().parser_name == "marker"

    @patch("parsers.marker_parser.subprocess.run")
    def test_is_available_true(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = MagicMock(returncode=0)
        assert MarkerParser().is_available() is True

    @patch("parsers.marker_parser.subprocess.run", side_effect=FileNotFoundError)
    def test_is_available_false(self, mock_run):
        from parsers.marker_parser import MarkerParser
        assert MarkerParser().is_available() is False

    def test_abstract_only_returns_partial(self):
        from parsers.marker_parser import MarkerParser
        meta = {**SAMPLE_METADATA, "is_abstract_only": True}
        result = MarkerParser().parse("", meta)
        assert result["is_abstract_only"] is True

    @patch("parsers.marker_parser.MarkerParser._run_marker")
    def test_parse_extracts_sections(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = SAMPLE_MMD
        result = MarkerParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        headings = [s["heading"] for s in result["sections"]]
        assert any("Introduction" in h for h in headings)
        assert any("Background" in h for h in headings)

    @patch("parsers.marker_parser.MarkerParser._run_marker")
    def test_parse_extracts_equations(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = SAMPLE_MMD
        result = MarkerParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert len(result["equations"]) >= 2

    @patch("parsers.marker_parser.MarkerParser._run_marker")
    def test_parse_extracts_references(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = SAMPLE_MMD
        result = MarkerParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert len(result["references"]) == 2
        assert result["references"][0]["year"] == 2015
        assert result["references"][1]["year"] == 2017

    @patch("parsers.marker_parser.MarkerParser._run_marker")
    def test_parse_extracts_figures(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = SAMPLE_MMD
        result = MarkerParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert len(result["figures"]) >= 1
        assert "Transformer" in result["figures"][0]["caption"]

    @patch("parsers.marker_parser.MarkerParser._run_marker")
    def test_parse_extracts_tables(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = SAMPLE_MMD
        result = MarkerParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert len(result["tables"]) >= 1

    @patch("parsers.marker_parser.MarkerParser._run_marker")
    def test_math_fraction_high_for_math_heavy_md(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = SAMPLE_MMD
        result = MarkerParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert result["math_fraction"] > 0.0

    @patch("parsers.marker_parser.MarkerParser._run_marker")
    def test_marker_no_output_returns_empty(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = None
        result = MarkerParser().parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert result["sections"] == []
        assert result["equations"] == []

    @patch("parsers.marker_parser.MarkerParser._run_marker")
    def test_abstract_extracted_from_md(self, mock_run):
        from parsers.marker_parser import MarkerParser
        mock_run.return_value = SAMPLE_MMD
        meta = {**SAMPLE_METADATA, "abstract": ""}
        result = MarkerParser().parse("/fake/paper.pdf", meta)
        assert "recurrent" in result["abstract"].lower()


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestRouter:

    def test_abstract_only_skips_parse(self):
        from parsers.router import route_and_parse
        meta = {**SAMPLE_METADATA, "is_abstract_only": True}
        result = route_and_parse(None, meta)
        assert result["is_abstract_only"] is True
        assert result["sections"] == []

    def test_none_pdf_path_returns_abstract_only(self):
        from parsers.router import route_and_parse
        result = route_and_parse(None, SAMPLE_METADATA)
        assert result["is_abstract_only"] is True

    @patch("parsers.router.estimate_math_fraction", return_value=0.05)
    @patch("parsers.router._get_grobid")
    def test_low_math_routes_to_grobid(self, mock_get_grobid, mock_math):
        from parsers.router import route_and_parse

        mock_parser = MagicMock()
        mock_parser.is_available.return_value = True
        mock_parser.parse.return_value = empty_parsed_paper(
            parser_used="grobid", **{k: SAMPLE_METADATA[k] for k in
            ["paper_id", "title", "authors", "abstract", "year", "source"]}
        )
        mock_get_grobid.return_value = mock_parser

        result = route_and_parse("/fake/paper.pdf", SAMPLE_METADATA)
        mock_parser.parse.assert_called_once()
        assert result["parser_used"] == "grobid"

    @patch("parsers.router.estimate_math_fraction", return_value=0.45)
    @patch("parsers.router._get_marker")
    def test_high_math_routes_to_marker(self, mock_get_marker, mock_math):
        from parsers.router import route_and_parse

        mock_parser = MagicMock()
        mock_parser.is_available.return_value = True
        mock_parser.parse.return_value = empty_parsed_paper(
            parser_used="marker", **{k: SAMPLE_METADATA[k] for k in
            ["paper_id", "title", "authors", "abstract", "year", "source"]}
        )
        mock_get_marker.return_value = mock_parser

        result = route_and_parse("/fake/paper.pdf", SAMPLE_METADATA)
        mock_parser.parse.assert_called_once()
        assert result["parser_used"] == "marker"

    @patch("parsers.router.estimate_math_fraction", return_value=0.05)
    @patch("parsers.router._get_grobid")
    @patch("parsers.router._get_pymupdf")
    def test_grobid_unavailable_falls_back_to_pymupdf(
        self, mock_get_pymupdf, mock_get_grobid, mock_math
    ):
        from parsers.router import route_and_parse

        grobid_mock = MagicMock()
        grobid_mock.is_available.return_value = False
        mock_get_grobid.return_value = grobid_mock

        pymupdf_mock = MagicMock()
        pymupdf_mock.parse.return_value = empty_parsed_paper(
            parser_used="pymupdf", **{k: SAMPLE_METADATA[k] for k in
            ["paper_id", "title", "authors", "abstract", "year", "source"]}
        )
        mock_get_pymupdf.return_value = pymupdf_mock

        result = route_and_parse("/fake/paper.pdf", SAMPLE_METADATA)
        pymupdf_mock.parse.assert_called_once()
        assert result["parser_used"] == "pymupdf"

    @patch("parsers.router.estimate_math_fraction", return_value=0.45)
    @patch("parsers.router._get_marker")
    @patch("parsers.router._get_pymupdf")
    def test_marker_unavailable_falls_back_to_pymupdf(
        self, mock_get_pymupdf, mock_get_marker, mock_math
    ):
        from parsers.router import route_and_parse

        nougat_mock = MagicMock()
        nougat_mock.is_available.return_value = False
        mock_get_marker.return_value = nougat_mock

        pymupdf_mock = MagicMock()
        pymupdf_mock.parse.return_value = empty_parsed_paper(
            parser_used="pymupdf", **{k: SAMPLE_METADATA[k] for k in
            ["paper_id", "title", "authors", "abstract", "year", "source"]}
        )
        mock_get_pymupdf.return_value = pymupdf_mock

        result = route_and_parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert result["parser_used"] == "pymupdf"

    @patch("parsers.router.estimate_math_fraction", return_value=0.05)
    @patch("parsers.router._get_grobid")
    @patch("parsers.router._get_pymupdf")
    def test_grobid_exception_falls_back_to_pymupdf(
        self, mock_get_pymupdf, mock_get_grobid, mock_math
    ):
        from parsers.router import route_and_parse

        grobid_mock = MagicMock()
        grobid_mock.is_available.return_value = True
        grobid_mock.parse.side_effect = RuntimeError("unexpected crash")
        mock_get_grobid.return_value = grobid_mock

        pymupdf_mock = MagicMock()
        pymupdf_mock.parse.return_value = empty_parsed_paper(
            parser_used="pymupdf", **{k: SAMPLE_METADATA[k] for k in
            ["paper_id", "title", "authors", "abstract", "year", "source"]}
        )
        mock_get_pymupdf.return_value = pymupdf_mock

        result = route_and_parse("/fake/paper.pdf", SAMPLE_METADATA)
        assert result["parser_used"] == "pymupdf"

    def test_auto_route_false_uses_default_parser(self):
        from parsers.router import route_and_parse
        from config import Config

        original = Config.PARSER_AUTO_ROUTE
        Config.PARSER_AUTO_ROUTE = False
        Config.DEFAULT_PARSER = "pymupdf"

        with patch("parsers.router._get_pymupdf") as mock_get:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = empty_parsed_paper(
                parser_used="pymupdf", **{k: SAMPLE_METADATA[k] for k in
                ["paper_id", "title", "authors", "abstract", "year", "source"]}
            )
            mock_get.return_value = mock_parser
            result = route_and_parse("/fake/paper.pdf", SAMPLE_METADATA)

        Config.PARSER_AUTO_ROUTE = original
        assert result["parser_used"] == "pymupdf"
