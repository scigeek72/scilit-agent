"""
parsers/marker_parser.py — marker-pdf → ParsedPaper.

marker (https://github.com/VikParuchuri/marker) converts PDFs to structured
markdown, handling headings, tables, figures, and equations.  It runs as a
local CLI subprocess — no daemon, no Docker required — and works on Intel Mac,
Apple Silicon (MPS), Linux (CPU/CUDA), and Windows.

Installation:
  pip install marker-pdf

Usage (run as subprocess):
  marker_single {pdf_path} {output_dir}

Output: {output_dir}/{stem}/{stem}.md

Sections are delimited by # / ## / ### headings.
Equations are delimited by $$ ... $$ or \\[ ... \\].
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from parsers.base import (
    FigureDict,
    ParsedPaper,
    Parser,
    ReferenceDict,
    SectionDict,
    TableDict,
    empty_parsed_paper,
)
from parsers.pymupdf_parser import _compute_math_fraction

logger = logging.getLogger(__name__)

# Equation delimiters (same as Nougat .mmd)
_DISPLAY_EQ_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_DISPLAY_EQ2_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

# Reference entry patterns
_REF_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+(.+)$", re.MULTILINE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


class MarkerParser(Parser):

    @property
    def parser_name(self) -> str:
        return "marker"

    def is_available(self) -> bool:
        """Check that the marker_single CLI is on PATH."""
        try:
            result = subprocess.run(
                ["marker_single", "--help"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

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
            parser_used="marker",
            is_abstract_only=metadata.get("is_abstract_only", False),
        )

        if result["is_abstract_only"] or not pdf_path:
            return result

        md_text = self._run_marker(pdf_path)
        if md_text is None:
            logger.warning("marker produced no output for %s", pdf_path)
            return result

        result["math_fraction"] = _compute_math_fraction(md_text)
        result["equations"] = _extract_equations(md_text)
        result["sections"] = _extract_sections(md_text)
        result["references"] = _extract_references(md_text)
        result["figures"] = _extract_figure_captions(md_text)
        result["tables"] = _extract_table_captions(md_text)

        # Upgrade title/abstract from markdown if missing in metadata
        if not result["title"]:
            result["title"] = _extract_title(md_text)
        if not result["abstract"]:
            result["abstract"] = _extract_abstract(md_text)

        return result

    # ------------------------------------------------------------------
    # marker subprocess
    # ------------------------------------------------------------------

    def _run_marker(self, pdf_path: str) -> str | None:
        """Run marker_single CLI and return the markdown text, or None on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    ["marker_single", pdf_path, tmpdir],
                    capture_output=True,
                    timeout=300,   # 5 min hard limit
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "marker process failed (exit %d) for %s: %s",
                    exc.returncode, pdf_path,
                    exc.stderr.decode("utf-8", errors="replace")[:200],
                )
                return None
            except subprocess.TimeoutExpired:
                logger.warning("marker timed out for %s", pdf_path)
                return None
            except FileNotFoundError:
                logger.warning("marker_single binary not found")
                return None

            # marker_single writes to {tmpdir}/{stem}/{stem}.md
            stem = Path(pdf_path).stem
            md_path = Path(tmpdir) / stem / f"{stem}.md"
            if not md_path.exists():
                # Fallback: search recursively for any .md output
                candidates = list(Path(tmpdir).rglob("*.md"))
                if candidates:
                    md_path = candidates[0]
                else:
                    logger.warning("marker produced no .md file for %s", pdf_path)
                    return None

            return md_path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Markdown parsing helpers
# ---------------------------------------------------------------------------

def _extract_title(md: str) -> str:
    """First H1 heading, or first non-empty line."""
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:300]
        if stripped and not stripped.startswith("#"):
            return stripped[:300]
    return ""


def _extract_abstract(md: str) -> str:
    m = re.search(
        r"(?i)#+\s*abstract\s*\n(.+?)(?=\n#+\s|\Z)",
        md,
        re.DOTALL,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:2000]
    return ""


def _extract_sections(md: str) -> list[SectionDict]:
    """Split markdown by heading lines (# / ## / ###)."""
    sections: list[SectionDict] = []
    parts = re.split(r"(\n#+\s+.+)", md)

    current_heading = ""
    current_level = 1
    current_body_parts: list[str] = []

    for part in parts:
        heading_m = re.match(r"\n(#+)\s+(.+)", part)
        if heading_m:
            body = "".join(current_body_parts).strip()
            if body or current_heading:
                sections.append(
                    SectionDict(
                        heading=current_heading,
                        text=body,
                        level=current_level,
                    )
                )
            current_heading = heading_m.group(2).strip()
            current_level = len(heading_m.group(1))
            current_body_parts = []
        else:
            current_body_parts.append(part)

    body = "".join(current_body_parts).strip()
    if body or current_heading:
        sections.append(
            SectionDict(heading=current_heading, text=body, level=current_level)
        )

    return sections


def _extract_equations(md: str) -> list[str]:
    """Extract display-math LaTeX strings."""
    eqs: list[str] = []
    for m in _DISPLAY_EQ_RE.finditer(md):
        eqs.append(m.group(1).strip())
    for m in _DISPLAY_EQ2_RE.finditer(md):
        eqs.append(m.group(1).strip())
    return eqs[:200]


def _extract_references(md: str) -> list[ReferenceDict]:
    refs: list[ReferenceDict] = []
    ref_section_m = re.search(r"(?i)#+\s*references?\s*\n", md)
    ref_text = md[ref_section_m.end():] if ref_section_m else md

    for m in _REF_ENTRY_RE.finditer(ref_text):
        ref_id = m.group(1)
        body = m.group(2).strip()
        year_m = _YEAR_RE.search(body)
        year = int(year_m.group(0)) if year_m else 0
        refs.append(
            ReferenceDict(
                ref_id=ref_id,
                title=body[:200],
                authors=[],
                year=year,
            )
        )
        if len(refs) >= 150:
            break
    return refs


def _extract_figure_captions(md: str) -> list[FigureDict]:
    figures: list[FigureDict] = []
    for m in re.finditer(
        r"(?i)\*\*(?:fig(?:ure)?\.?\s*)(\d+[a-z]?)[.:]\*\*\s*(.+?)(?=\n\n|\Z)",
        md,
        re.DOTALL,
    ):
        caption = re.sub(r"\s+", " ", m.group(2)).strip()[:500]
        figures.append(FigureDict(label=f"Figure {m.group(1)}", caption=caption))
        if len(figures) >= 30:
            break
    return figures


def _extract_table_captions(md: str) -> list[TableDict]:
    tables: list[TableDict] = []
    for m in re.finditer(
        r"(?i)\*\*(?:table\.?\s*)(\d+[a-z]?)[.:]\*\*\s*(.+?)(?=\n\n|\Z)",
        md,
        re.DOTALL,
    ):
        caption = re.sub(r"\s+", " ", m.group(2)).strip()[:500]
        tables.append(TableDict(label=f"Table {m.group(1)}", caption=caption))
        if len(tables) >= 20:
            break
    return tables
