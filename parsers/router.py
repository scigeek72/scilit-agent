"""
parsers/router.py — classify paper and route to the correct parser.

Decision logic:
  1. If PARSER_AUTO_ROUTE is False → use DEFAULT_PARSER
  2. Estimate math_fraction from first N pages (PyMuPDF, always available)
  3. math_fraction >= MATH_THRESHOLD → try Nougat
     math_fraction <  MATH_THRESHOLD → try Grobid
  4. If chosen parser is unavailable (service down) → PyMuPDF fallback
  5. If primary parser fails → PyMuPDF fallback
  6. Log which parser was actually used

All downstream code receives the same ParsedPaper schema regardless of
which parser ran.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import fitz as _fitz
except ImportError:
    _fitz = None  # type: ignore[assignment]

from config import Config
from parsers.base import ParsedPaper, empty_parsed_paper
from parsers.pymupdf_parser import PyMuPDFParser, _compute_math_fraction

logger = logging.getLogger(__name__)

# Lazy imports for heavy parsers — only instantiated when needed
_pymupdf: PyMuPDFParser | None = None
_grobid = None
_nougat = None


def _get_pymupdf() -> PyMuPDFParser:
    global _pymupdf
    if _pymupdf is None:
        _pymupdf = PyMuPDFParser()
    return _pymupdf


def _get_grobid():
    global _grobid
    if _grobid is None:
        from parsers.grobid_parser import GrobidParser
        _grobid = GrobidParser()
    return _grobid


def _get_marker():
    global _nougat
    if _nougat is None:
        from parsers.marker_parser import MarkerParser
        _nougat = MarkerParser()
    return _nougat


# ---------------------------------------------------------------------------
# Math fraction estimation (fast, no external service)
# ---------------------------------------------------------------------------

def estimate_math_fraction(pdf_path: str, max_pages: int = 5) -> float:
    """
    Quickly estimate the math density of a PDF by reading the first
    max_pages pages with PyMuPDF. Returns 0.0 if PyMuPDF is unavailable
    or the file cannot be opened.
    """
    try:
        if _fitz is None:
            return 0.0
        doc = _fitz.open(pdf_path)
        pages_to_scan = min(max_pages, len(doc))
        text = "\n".join(doc[i].get_text() for i in range(pages_to_scan))
        doc.close()
        return _compute_math_fraction(text)
    except Exception as exc:
        logger.debug("math_fraction estimation failed for %s: %s", pdf_path, exc)
        return 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _ensure_services_once() -> None:
    """Auto-start Grobid the first time route_and_parse is called."""
    global _services_checked
    if _services_checked:
        return
    _services_checked = True
    try:
        from services import ensure_grobid
        ensure_grobid(auto_start=True)
    except Exception as exc:
        logger.debug("Service auto-start skipped: %s", exc)


_services_checked: bool = False


def route_and_parse(pdf_path: str | None, metadata: dict) -> ParsedPaper:
    """
    Classify the paper, choose a parser, and return a ParsedPaper.

    pdf_path: absolute path to PDF, or None for abstract-only papers.
    metadata: PaperMetadata.to_dict()

    Never raises — always returns a valid ParsedPaper (possibly with
    only abstract-level fields populated if PDF is unavailable).
    """
    _ensure_services_once()

    # Abstract-only shortcut
    if not pdf_path or metadata.get("is_abstract_only", False):
        logger.info(
            "Abstract-only paper %s — skipping parse",
            metadata.get("paper_id", "?"),
        )
        result = empty_parsed_paper(
            paper_id=metadata.get("paper_id", ""),
            title=metadata.get("title", ""),
            authors=metadata.get("authors", []),
            abstract=metadata.get("abstract", ""),
            year=metadata.get("year", 0),
            source=metadata.get("source", ""),
            parser_used="pymupdf",
            is_abstract_only=True,
        )
        return result

    paper_id = metadata.get("paper_id", pdf_path)

    # Determine which parser to use
    if not Config.PARSER_AUTO_ROUTE:
        parser_name = Config.DEFAULT_PARSER
        math_fraction = 0.0
        logger.info(
            "Auto-routing disabled — using default parser '%s' for %s",
            parser_name, paper_id,
        )
    else:
        math_fraction = estimate_math_fraction(pdf_path)
        parser_name = (
            "marker" if math_fraction >= Config.MATH_THRESHOLD else "grobid"
        )
        logger.info(
            "math_fraction=%.3f → routed to '%s' for %s",
            math_fraction, parser_name, paper_id,
        )

    # Attempt primary parser, fall back to PyMuPDF on any failure
    result = _try_parse(parser_name, pdf_path, metadata)

    if result is None:
        logger.info(
            "Primary parser '%s' unavailable/failed — falling back to pymupdf for %s",
            parser_name, paper_id,
        )
        result = _get_pymupdf().parse(pdf_path, metadata)
        result["parser_used"] = "pymupdf"

    # Stamp math_fraction from our fast estimate if the parser didn't set one
    if result.get("math_fraction", 0.0) == 0.0 and math_fraction > 0.0:
        result["math_fraction"] = math_fraction

    logger.info(
        "Parsed %s with '%s': %d sections, %d refs, %d equations",
        paper_id,
        result.get("parser_used", "?"),
        len(result.get("sections", [])),
        len(result.get("references", [])),
        len(result.get("equations", [])),
    )
    return result


def _try_parse(
    parser_name: str, pdf_path: str, metadata: dict
) -> ParsedPaper | None:
    """
    Attempt to parse with the named parser.
    Returns None if the parser is unavailable or raises an unexpected error.
    """
    try:
        if parser_name == "grobid":
            parser = _get_grobid()
        elif parser_name == "marker":
            parser = _get_marker()
        elif parser_name == "pymupdf":
            return _get_pymupdf().parse(pdf_path, metadata)
        else:
            logger.warning("Unknown parser name '%s' — falling back", parser_name)
            return None

        if not parser.is_available():
            logger.warning(
                "Parser '%s' is not available (service down?)", parser_name
            )
            return None

        return parser.parse(pdf_path, metadata)

    except Exception as exc:
        logger.warning(
            "Parser '%s' raised unexpectedly for %s: %s",
            parser_name, metadata.get("paper_id", pdf_path), exc,
        )
        return None
