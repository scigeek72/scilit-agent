"""
tools/parse_tools.py — Agent-callable wrapper over the parser router.

Single function: run_parser()
Delegates entirely to parsers/router.py which handles:
  - math_fraction estimation → Grobid vs marker routing
  - fallback to PyMuPDF if primary parser unavailable
  - abstract-only shortcut (no PDF)
  - always returns a valid ParsedPaper dict

Usage:
    from tools.parse_tools import run_parser
    result = run_parser(pdf_path, metadata)   # ParsedPaper dict
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_parser(pdf_path: str | None, metadata: dict) -> dict:
    """
    Parse a PDF (or produce an abstract-only ParsedPaper) using the router.

    pdf_path: absolute local path to PDF, or None for abstract-only papers.
    metadata: PaperMetadata.to_dict()

    Returns a ParsedPaper dict.  Never raises — falls back gracefully.

    The router decides which parser to use:
      - math_fraction >= MATH_THRESHOLD → marker (LaTeX / equations)
      - math_fraction <  MATH_THRESHOLD → Grobid (sections / references)
      - Either unavailable              → PyMuPDF fallback
      - pdf_path is None                → abstract-only stub
    """
    from parsers.router import route_and_parse
    result = route_and_parse(pdf_path, metadata)
    logger.info(
        "run_parser(%s): parser=%s sections=%d refs=%d",
        metadata.get("paper_id", "?"),
        result.get("parser_used", "?"),
        len(result.get("sections", [])),
        len(result.get("references", [])),
    )
    return result
