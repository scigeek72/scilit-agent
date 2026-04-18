"""
sources/arxiv_source.py — arXiv source connector.

Uses the official `arxiv` Python library. All arXiv papers are open access.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import arxiv

from sources.base import PaperMetadata, SourceConnector

logger = logging.getLogger(__name__)


class ArxivSource(SourceConnector):

    # arXiv API recommends <= 3 requests/sec for anonymous access
    _REQUEST_DELAY: float = 0.5

    @property
    def source_name(self) -> str:
        return "arxiv"

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        """Search arXiv for papers matching plain-text query."""
        results: list[PaperMetadata] = []
        try:
            client = arxiv.Client(
                page_size=min(max_results, 100),
                delay_seconds=self._REQUEST_DELAY,
                num_retries=3,
            )
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for paper in client.results(search):
                metadata = self._to_metadata(paper)
                results.append(metadata)
        except Exception as exc:
            logger.warning("arXiv search failed: %s", exc)
        return results

    # ------------------------------------------------------------------
    # fetch_metadata
    # ------------------------------------------------------------------

    def fetch_metadata(self, paper_id: str) -> PaperMetadata:
        """
        Fetch metadata for a paper by its namespaced ID, e.g. 'arxiv:2301.12345'.
        """
        if not paper_id.startswith("arxiv:"):
            raise ValueError(f"paper_id '{paper_id}' does not belong to arxiv source")
        arxiv_id = paper_id[len("arxiv:"):]
        client = arxiv.Client(num_retries=3)
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        if not results:
            raise LookupError(f"arXiv paper not found: {arxiv_id}")
        return self._to_metadata(results[0])

    # ------------------------------------------------------------------
    # download_pdf
    # ------------------------------------------------------------------

    def download_pdf(self, metadata: PaperMetadata, output_dir: str) -> str | None:
        """Download the arXiv PDF. Returns local path, or None on failure."""
        if not metadata.pdf_url:
            logger.info("No PDF URL for %s", metadata.paper_id)
            return None

        arxiv_id = metadata.short_id()
        output_path = Path(output_dir) / f"{metadata.wiki_filename()}.pdf"

        if output_path.exists():
            logger.debug("PDF already cached: %s", output_path)
            return str(output_path)

        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            client = arxiv.Client(num_retries=3)
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(client.results(search))
            if not results:
                logger.warning("Could not retrieve arXiv entry for download: %s", arxiv_id)
                return None
            paper = results[0]
            paper.download_pdf(dirpath=output_dir, filename=output_path.name)
            time.sleep(self._REQUEST_DELAY)
            return str(output_path)
        except Exception as exc:
            logger.warning("PDF download failed for %s: %s", metadata.paper_id, exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_metadata(self, paper: arxiv.Result) -> PaperMetadata:
        """Convert an arxiv.Result object to PaperMetadata."""
        arxiv_id = paper.entry_id.split("/")[-1]   # strip version suffix later
        # Keep version-stripped ID: "2301.12345v2" → "2301.12345"
        arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

        doi = paper.doi if paper.doi else None

        # arXiv PDF URL: use pdf_url from the result if available
        pdf_url = str(paper.pdf_url) if paper.pdf_url else None

        # Tags: primary category + all categories
        tags = list({paper.primary_category} | set(paper.categories))

        return PaperMetadata(
            paper_id=f"arxiv:{arxiv_id_clean}",
            title=paper.title,
            authors=[str(a) for a in paper.authors],
            abstract=paper.summary,
            year=paper.published.year if paper.published else 0,
            source="arxiv",
            pdf_url=pdf_url,
            doi=doi,
            venue=paper.journal_ref or paper.primary_category,
            tags=tags,
            is_open_access=True,
        )
