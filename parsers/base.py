"""
parsers/base.py — Parser ABC and ParsedPaper TypedDict.

Both Grobid and Nougat return a ParsedPaper dict.
All downstream code is parser-agnostic — never depend on parser-specific fields.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


# ---------------------------------------------------------------------------
# ParsedPaper — the unified output schema for all parsers
# ---------------------------------------------------------------------------

class SectionDict(TypedDict):
    heading: str
    text: str
    level: int          # 1 = top-level, 2 = subsection, etc.


class ReferenceDict(TypedDict):
    ref_id: str
    title: str
    authors: list[str]
    year: int


class FigureDict(TypedDict):
    caption: str
    label: str


class TableDict(TypedDict):
    caption: str
    label: str


class ParsedPaper(TypedDict):
    paper_id:         str
    title:            str
    authors:          list[str]
    abstract:         str
    year:             int
    source:           str
    sections:         list[SectionDict]
    references:       list[ReferenceDict]
    figures:          list[FigureDict]
    tables:           list[TableDict]
    equations:        list[str]         # LaTeX strings; Nougat only, [] otherwise
    parser_used:      str               # "grobid" | "nougat" | "pymupdf"
    math_fraction:    float             # 0.0–1.0 fraction of math tokens
    is_abstract_only: bool              # True if PDF was unavailable


def empty_parsed_paper(
    paper_id: str = "",
    title: str = "",
    authors: list[str] | None = None,
    abstract: str = "",
    year: int = 0,
    source: str = "",
    parser_used: str = "pymupdf",
    math_fraction: float = 0.0,
    is_abstract_only: bool = False,
) -> ParsedPaper:
    """Return a ParsedPaper with all list fields initialised to empty lists."""
    return ParsedPaper(
        paper_id=paper_id,
        title=title,
        authors=authors or [],
        abstract=abstract,
        year=year,
        source=source,
        sections=[],
        references=[],
        figures=[],
        tables=[],
        equations=[],
        parser_used=parser_used,
        math_fraction=math_fraction,
        is_abstract_only=is_abstract_only,
    )


# ---------------------------------------------------------------------------
# Parser ABC
# ---------------------------------------------------------------------------

class Parser(ABC):
    """
    Abstract base class for all document parsers.

    Every parser receives a PDF path (or None for abstract-only papers) and
    the pre-populated PaperMetadata, and returns a ParsedPaper dict.
    The returned dict must conform to the ParsedPaper schema — no
    parser-specific keys should be added.
    """

    @abstractmethod
    def parse(self, pdf_path: str, metadata: dict) -> ParsedPaper:
        """
        Parse the PDF at pdf_path using the pre-populated metadata dict.

        pdf_path: absolute path to the PDF file on disk.
        metadata: PaperMetadata.to_dict() — provides paper_id, title,
                  authors, abstract, year, source as authoritative fields.

        Returns a fully-populated ParsedPaper dict.
        Never raises — catch all internal errors and fall back to the
        best available partial result.
        """

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """Unique identifier: 'grobid' | 'nougat' | 'pymupdf'."""

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if this parser is currently available.
        For service-based parsers (Grobid, Nougat), check the service.
        For PyMuPDF, always returns True (pure Python, no external service).
        """
