"""
parsers/nougat_parser.py — Nougat .mmd → ParsedPaper.

Nougat (Meta AI) converts math-heavy PDFs to Mathpix Markdown (.mmd),
preserving LaTeX equations faithfully.

Installation:
  pip install nougat-ocr

Usage (run as subprocess):
  nougat {pdf_path} -o {output_dir} --no-skipping

Output: {output_dir}/{stem}.mmd

Sections are delimited by ## headings.
Equations are delimited by \[ ... \] or $$ ... $$.
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

# Equation delimiters
_DISPLAY_EQ_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_DISPLAY_EQ2_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_EQ_RE = re.compile(r"\$([^$\n]+?)\$")

# Reference entry in .mmd: often "[N] Author..." or "## References" section
_REF_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+(.+)$", re.MULTILINE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


class NougatParser(Parser):

    @property
    def parser_name(self) -> str:
        return "nougat"

    def is_available(self) -> bool:
        """Check that the nougat CLI is on PATH."""
        try:
            result = subprocess.run(
                ["nougat", "--help"],
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
            parser_used="nougat",
            is_abstract_only=metadata.get("is_abstract_only", False),
        )

        if result["is_abstract_only"] or not pdf_path:
            return result

        mmd_text = self._run_nougat(pdf_path)
        if mmd_text is None:
            logger.warning("Nougat produced no output for %s", pdf_path)
            return result

        result["math_fraction"] = _compute_math_fraction(mmd_text)
        result["equations"] = _extract_equations(mmd_text)
        result["sections"] = _extract_sections(mmd_text)
        result["references"] = _extract_references(mmd_text)
        result["figures"] = _extract_figure_captions(mmd_text)
        result["tables"] = _extract_table_captions(mmd_text)

        # Upgrade title/abstract if missing in metadata
        if not result["title"]:
            result["title"] = _extract_title(mmd_text)
        if not result["abstract"]:
            result["abstract"] = _extract_abstract(mmd_text)

        return result

    # ------------------------------------------------------------------
    # Nougat subprocess
    # ------------------------------------------------------------------

    def _run_nougat(self, pdf_path: str) -> str | None:
        """Run nougat CLI and return the .mmd text, or None on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    [
                        "nougat",
                        pdf_path,
                        "-o", tmpdir,
                        "--no-skipping",
                        "--markdown",
                    ],
                    capture_output=True,
                    timeout=300,   # 5 min hard limit
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "Nougat process failed (exit %d) for %s: %s",
                    exc.returncode, pdf_path,
                    exc.stderr.decode("utf-8", errors="replace")[:200],
                )
                return None
            except subprocess.TimeoutExpired:
                logger.warning("Nougat timed out for %s", pdf_path)
                return None
            except FileNotFoundError:
                logger.warning("nougat binary not found")
                return None

            stem = Path(pdf_path).stem
            mmd_path = Path(tmpdir) / f"{stem}.mmd"
            if not mmd_path.exists():
                # Nougat may use only the first few chars of the stem
                candidates = list(Path(tmpdir).glob("*.mmd"))
                if candidates:
                    mmd_path = candidates[0]
                else:
                    logger.warning("Nougat produced no .mmd file for %s", pdf_path)
                    return None

            return mmd_path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# .mmd parsing helpers
# ---------------------------------------------------------------------------

def _extract_title(mmd: str) -> str:
    """First non-empty line that is not a heading marker."""
    for line in mmd.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped and not stripped.startswith("\\"):
            return stripped[:300]
    return ""


def _extract_abstract(mmd: str) -> str:
    m = re.search(
        r"(?i)#+\s*abstract\s*\n(.+?)(?=\n#+\s|\Z)",
        mmd,
        re.DOTALL,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:2000]
    return ""


def _extract_sections(mmd: str) -> list[SectionDict]:
    """Split .mmd by ## / ### headings."""
    sections: list[SectionDict] = []
    # Split on heading lines
    parts = re.split(r"(\n#+\s+.+)", mmd)

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
            current_level = len(heading_m.group(1))   # # = 1, ## = 2, etc.
            current_body_parts = []
        else:
            current_body_parts.append(part)

    body = "".join(current_body_parts).strip()
    if body or current_heading:
        sections.append(
            SectionDict(heading=current_heading, text=body, level=current_level)
        )

    return sections


def _extract_equations(mmd: str) -> list[str]:
    """Extract all display-math LaTeX strings."""
    eqs: list[str] = []
    for m in _DISPLAY_EQ_RE.finditer(mmd):
        eqs.append(m.group(1).strip())
    for m in _DISPLAY_EQ2_RE.finditer(mmd):
        eqs.append(m.group(1).strip())
    return eqs[:200]   # cap


def _extract_references(mmd: str) -> list[ReferenceDict]:
    refs: list[ReferenceDict] = []
    # Find the references section
    ref_section_m = re.search(r"(?i)#+\s*references?\s*\n", mmd)
    ref_text = mmd[ref_section_m.end():] if ref_section_m else mmd

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


def _extract_figure_captions(mmd: str) -> list[FigureDict]:
    figures: list[FigureDict] = []
    for m in re.finditer(
        r"(?i)\*\*(?:fig(?:ure)?\.?\s*)(\d+[a-z]?)[.:]\*\*\s*(.+?)(?=\n\n|\Z)",
        mmd,
        re.DOTALL,
    ):
        caption = re.sub(r"\s+", " ", m.group(2)).strip()[:500]
        figures.append(FigureDict(label=f"Figure {m.group(1)}", caption=caption))
        if len(figures) >= 30:
            break
    return figures


def _extract_table_captions(mmd: str) -> list[TableDict]:
    tables: list[TableDict] = []
    for m in re.finditer(
        r"(?i)\*\*(?:table\.?\s*)(\d+[a-z]?)[.:]\*\*\s*(.+?)(?=\n\n|\Z)",
        mmd,
        re.DOTALL,
    ):
        caption = re.sub(r"\s+", " ", m.group(2)).strip()[:500]
        tables.append(TableDict(label=f"Table {m.group(1)}", caption=caption))
        if len(tables) >= 20:
            break
    return tables
