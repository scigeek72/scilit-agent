"""
parsers/pymupdf_parser.py — PyMuPDF plain-text fallback parser.

Always available (pure Python). Used when Grobid and Nougat are both
unreachable. Extracts sections by heading heuristics, references by
pattern matching, and computes a math_fraction estimate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

try:
    import fitz   # PyMuPDF — available in ds-project-2026 env
except ImportError:
    fitz = None  # type: ignore[assignment]

from parsers.base import (
    FigureDict,
    ParsedPaper,
    Parser,
    ReferenceDict,
    SectionDict,
    TableDict,
    empty_parsed_paper,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Math-token patterns used for math_fraction estimation
# ---------------------------------------------------------------------------
_MATH_PATTERNS = re.compile(
    r"""
    \$[^$]+\$               |   # inline LaTeX  $...$
    \$\$[^$]+\$\$           |   # display LaTeX $$...$$
    \\[a-zA-Z]+\{           |   # LaTeX commands  \frac{  \sum{
    \\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|sigma|tau|
         phi|psi|omega|nabla|partial|infty|leq|geq|neq|approx|cdot|times|
         int|sum|prod|lim|frac|sqrt|hat|bar|vec|mathbb|mathcal)\b  |
    [∀∃∈∉⊂⊃∪∩∧∨¬→↔⟹⟺]     |   # Unicode math symbols
    [αβγδεζηθικλμνξπρστυφχψω]   # Greek letters
    """,
    re.VERBOSE,
)

# Section-heading heuristics: lines that look like headings
_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z]"      # "1. Introduction" or "2.1 Methods"
    r"|[A-Z][A-Z\s]{3,40}$"           # ALL CAPS headings
    r"|(?:Abstract|Introduction|Related Work|Background|Methods?|"
    r"Methodology|Results?|Discussion|Conclusion|References?|"
    r"Acknowledgments?|Appendix)\s*$"
    r")",
    re.MULTILINE,
)

# Reference line patterns
_REF_LINE_RE = re.compile(
    r"^\s*\[?\d+\]?\s+(.+)$"
)
_YEAR_IN_REF_RE = re.compile(r"\b(19|20)\d{2}\b")


class PyMuPDFParser(Parser):

    @property
    def parser_name(self) -> str:
        return "pymupdf"

    def is_available(self) -> bool:
        return fitz is not None

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str, metadata: dict) -> ParsedPaper:
        result = empty_parsed_paper(
            paper_id=metadata.get("paper_id", ""),
            title=metadata.get("title", ""),
            authors=metadata.get("authors", []),
            abstract=metadata.get("abstract", ""),
            year=metadata.get("year", 0),
            source=metadata.get("source", ""),
            parser_used="pymupdf",
            is_abstract_only=metadata.get("is_abstract_only", False),
        )

        if result["is_abstract_only"] or not pdf_path:
            return result

        try:
            doc = fitz.open(pdf_path)
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as exc:
            logger.warning("PyMuPDF failed to open %s: %s", pdf_path, exc)
            return result

        result["math_fraction"] = _compute_math_fraction(full_text)
        result["sections"] = _extract_sections(full_text)
        result["references"] = _extract_references(full_text)
        result["figures"] = _extract_figure_captions(full_text)
        result["tables"] = _extract_table_captions(full_text)

        # Upgrade title/abstract from PDF text if metadata fields are empty
        if not result["title"]:
            lines = [l.strip() for l in full_text.splitlines() if l.strip()]
            result["title"] = lines[0] if lines else ""

        if not result["abstract"]:
            m = re.search(
                r"(?i)abstract[.\s:]*(.+?)(?=\n(?:introduction|keywords|1\s*\.?\s*intro)|\Z)",
                full_text,
                re.DOTALL,
            )
            if m:
                result["abstract"] = re.sub(r"\s+", " ", m.group(1)).strip()[:2000]

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_math_fraction(text: str) -> float:
    """
    Return the fraction of tokens that look like math.
    Capped at 1.0. Returns 0.0 for empty text.
    """
    if not text:
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    math_hits = len(_MATH_PATTERNS.findall(text))
    return min(math_hits / max(len(tokens), 1), 1.0)


def _extract_sections(text: str) -> list[SectionDict]:
    """Split text into sections by heading heuristics."""
    sections: list[SectionDict] = []
    lines = text.splitlines()
    current_heading = ""
    current_level = 1
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        if _HEADING_RE.match(stripped):
            # Save previous section
            body = "\n".join(current_lines).strip()
            if body or current_heading:
                sections.append(
                    SectionDict(
                        heading=current_heading,
                        text=body,
                        level=current_level,
                    )
                )
            current_heading = stripped
            current_level = _heading_level(stripped)
            current_lines = []
        else:
            current_lines.append(stripped)

    # Final section
    body = "\n".join(current_lines).strip()
    if body or current_heading:
        sections.append(
            SectionDict(heading=current_heading, text=body, level=current_level)
        )

    return sections


def _heading_level(heading: str) -> int:
    """Infer heading depth from numbering prefix."""
    m = re.match(r"^(\d+(?:\.\d+)*)", heading)
    if m:
        return len(m.group(1).split("."))
    return 1


def _extract_references(text: str) -> list[ReferenceDict]:
    """Extract reference entries after the References section."""
    refs: list[ReferenceDict] = []
    # Find the references section
    m = re.search(r"(?i)\breferences\b", text)
    if not m:
        return refs

    ref_text = text[m.end():]
    entries = re.split(r"\n\s*\[?\d+\]?\s+", ref_text)

    for i, entry in enumerate(entries[1:], start=1):  # skip text before first ref
        entry = entry.strip()
        if not entry or len(entry) < 10:
            continue
        year_m = _YEAR_IN_REF_RE.search(entry)
        year = int(year_m.group(0)) if year_m else 0
        # First line heuristic for title
        first_line = entry.splitlines()[0].strip() if entry else ""
        refs.append(
            ReferenceDict(
                ref_id=str(i),
                title=first_line[:200],
                authors=[],
                year=year,
            )
        )
        if len(refs) >= 100:   # cap to avoid huge lists
            break

    return refs


def _extract_figure_captions(text: str) -> list[FigureDict]:
    figures: list[FigureDict] = []
    for m in re.finditer(
        r"(?i)(?:fig(?:ure)?\.?\s*)(\d+[a-z]?)[.:]\s*(.+?)(?=\n\n|\Z)",
        text,
        re.DOTALL,
    ):
        caption = re.sub(r"\s+", " ", m.group(2)).strip()[:500]
        figures.append(FigureDict(label=f"Figure {m.group(1)}", caption=caption))
        if len(figures) >= 30:
            break
    return figures


def _extract_table_captions(text: str) -> list[TableDict]:
    tables: list[TableDict] = []
    for m in re.finditer(
        r"(?i)(?:table\.?\s*)(\d+[a-z]?)[.:]\s*(.+?)(?=\n\n|\Z)",
        text,
        re.DOTALL,
    ):
        caption = re.sub(r"\s+", " ", m.group(2)).strip()[:500]
        tables.append(TableDict(label=f"Table {m.group(1)}", caption=caption))
        if len(tables) >= 20:
            break
    return tables
