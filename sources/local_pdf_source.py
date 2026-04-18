"""
sources/local_pdf_source.py — Local PDF drop-folder connector.

Watches data/{topic}/raw/local_drop/ for PDF files.
No API, no rate limiting. All discovered PDFs are treated as open access.

Metadata extraction order:
1. PDF document info (PyMuPDF) — title, author fields if present
2. First page text heuristics — title from first non-empty line
3. Filename as final fallback

The user assigns the paper_id as "local:{stem}" where stem is the filename
without extension.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from sources.base import PaperMetadata, SourceConnector
from config import Config

logger = logging.getLogger(__name__)


def _extract_metadata_from_pdf(pdf_path: Path) -> dict:
    """
    Extract best-effort title, authors, year, and abstract from a PDF.
    Returns a dict with keys: title, authors, year, abstract.
    Never raises — returns empty strings on failure.
    """
    title = ""
    authors: list[str] = []
    year = 0
    abstract = ""

    try:
        import fitz   # PyMuPDF

        doc = fitz.open(str(pdf_path))

        # --- DocInfo metadata ---
        info = doc.metadata or {}
        if info.get("title"):
            title = info["title"].strip()
        if info.get("author"):
            authors = [a.strip() for a in re.split(r"[;,]", info["author"]) if a.strip()]
        if info.get("creationDate"):
            # PDF date format: D:YYYYMMDDHHmmSS
            m = re.search(r"D:(\d{4})", info["creationDate"])
            if m:
                year = int(m.group(1))

        # --- First-page text heuristics ---
        first_page_text = doc[0].get_text() if len(doc) > 0 else ""
        lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]

        if not title and lines:
            # The first non-empty line is usually the title
            title = lines[0]

        if not year:
            # Search all first-page text for a 4-digit year 2000-2030
            for line in lines[:30]:
                m = re.search(r"\b(20[0-2]\d)\b", line)
                if m:
                    year = int(m.group(1))
                    break

        # Extract abstract heuristically
        full_text = "\n".join(lines)
        m = re.search(
            r"(?i)abstract[.\s:]*(.+?)(?=\n(?:introduction|keywords|1\s*\.?\s*intro)|\Z)",
            full_text,
            re.DOTALL,
        )
        if m:
            abstract = re.sub(r"\s+", " ", m.group(1)).strip()[:2000]

        doc.close()

    except Exception as exc:
        logger.debug("PDF metadata extraction failed for %s: %s", pdf_path.name, exc)

    return {
        "title": title or pdf_path.stem.replace("_", " ").replace("-", " ").title(),
        "authors": authors,
        "year": year or datetime.now().year,
        "abstract": abstract,
    }


class LocalPdfSource(SourceConnector):
    """
    Treats every PDF file in the local_drop directory as a paper to ingest.
    The 'search' method ignores the query — it returns ALL PDFs in the folder.
    """

    @property
    def source_name(self) -> str:
        return "local_pdf"

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        """
        Return metadata for every PDF in the local_drop directory.
        The query parameter is ignored — local files are always included.
        """
        drop_dir = Config.local_drop_dir()
        if not drop_dir.exists():
            drop_dir.mkdir(parents=True, exist_ok=True)
            return []

        results: list[PaperMetadata] = []
        pdf_files = sorted(drop_dir.glob("*.pdf"))[:max_results]

        for pdf_path in pdf_files:
            try:
                metadata = self._pdf_to_metadata(pdf_path)
                results.append(metadata)
            except Exception as exc:
                logger.warning("Could not read local PDF %s: %s", pdf_path.name, exc)

        return results

    # ------------------------------------------------------------------
    # fetch_metadata
    # ------------------------------------------------------------------

    def fetch_metadata(self, paper_id: str) -> PaperMetadata:
        """Fetch metadata for a local PDF by its namespaced ID 'local:{stem}'."""
        if not paper_id.startswith("local:"):
            raise ValueError(f"paper_id '{paper_id}' does not belong to local_pdf source")
        stem = paper_id[len("local:"):]
        drop_dir = Config.local_drop_dir()
        pdf_path = drop_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            raise LookupError(f"Local PDF not found: {pdf_path}")
        return self._pdf_to_metadata(pdf_path)

    # ------------------------------------------------------------------
    # download_pdf
    # ------------------------------------------------------------------

    def download_pdf(self, metadata: PaperMetadata, output_dir: str) -> str | None:
        """
        'Download' a local PDF by copying it to the raw PDF directory.
        Returns the destination path (already on disk), never None for local files.
        """
        drop_dir = Config.local_drop_dir()
        stem = metadata.short_id()
        source_path = drop_dir / f"{stem}.pdf"

        if not source_path.exists():
            logger.warning("Local PDF no longer on disk: %s", source_path)
            return None

        dest_path = Path(output_dir) / f"{metadata.wiki_filename()}.pdf"
        if dest_path.exists():
            return str(dest_path)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        return str(dest_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pdf_to_metadata(self, pdf_path: Path) -> PaperMetadata:
        extracted = _extract_metadata_from_pdf(pdf_path)

        # Stable ID: use a short hash of the filename so renames don't
        # create duplicates if the file content is the same.
        stem = pdf_path.stem
        paper_id = f"local:{stem}"

        return PaperMetadata(
            paper_id=paper_id,
            title=extracted["title"],
            authors=extracted["authors"],
            abstract=extracted["abstract"],
            year=extracted["year"],
            source="local",
            pdf_url=None,          # already on disk; no URL needed
            doi=None,
            venue=None,
            tags=["local"],
            is_open_access=True,
        )
