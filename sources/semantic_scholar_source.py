"""
sources/semantic_scholar_source.py — Semantic Scholar source connector.

Semantic Scholar indexes papers from Nature, Springer, Elsevier, IEEE, ACM,
and many others. It is the recommended universal fallback because it
frequently has open-access PDF links for otherwise paywalled papers.

Uses the `semanticscholar` Python library (rate-limited; API key optional).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from sources.base import PaperMetadata, SourceConnector
from config import Config

logger = logging.getLogger(__name__)

_SS_API_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = (
    "paperId,title,authors,abstract,year,externalIds,"
    "openAccessPdf,isOpenAccess,publicationVenue,fieldsOfStudy,tldr"
)


class SemanticScholarSource(SourceConnector):

    _REQUEST_DELAY: float = 1.0   # 1 req/sec without API key; 10/sec with key

    def __init__(self) -> None:
        self._api_key = Config.semantic_scholar_api_key()
        self._headers: dict[str, str] = {}
        if self._api_key:
            self._headers["x-api-key"] = self._api_key

    @property
    def source_name(self) -> str:
        return "semantic_scholar"

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        """Search Semantic Scholar for papers matching plain-text query."""
        results: list[PaperMetadata] = []
        limit = min(max_results, 100)   # SS API max per call is 100
        offset = 0

        while len(results) < max_results:
            batch = min(limit, max_results - len(results))
            try:
                resp = requests.get(
                    f"{_SS_API_BASE}/paper/search",
                    headers=self._headers,
                    params={
                        "query": query,
                        "fields": _FIELDS,
                        "limit": batch,
                        "offset": offset,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("Semantic Scholar search failed: %s", exc)
                break

            papers = data.get("data", [])
            if not papers:
                break

            for p in papers:
                try:
                    results.append(self._to_metadata(p))
                except Exception as exc:
                    logger.debug("Could not parse SS paper: %s", exc)

            total = data.get("total", 0)
            offset += len(papers)
            if offset >= total or offset >= max_results:
                break

            time.sleep(self._REQUEST_DELAY)

        return results[:max_results]

    # ------------------------------------------------------------------
    # fetch_metadata
    # ------------------------------------------------------------------

    def fetch_metadata(self, paper_id: str) -> PaperMetadata:
        """Fetch metadata by namespaced ID, e.g. 'semantic_scholar:abc123'."""
        if not paper_id.startswith("semantic_scholar:"):
            raise ValueError(
                f"paper_id '{paper_id}' does not belong to semantic_scholar source"
            )
        ss_id = paper_id[len("semantic_scholar:"):]
        try:
            resp = requests.get(
                f"{_SS_API_BASE}/paper/{ss_id}",
                headers=self._headers,
                params={"fields": _FIELDS},
                timeout=30,
            )
            resp.raise_for_status()
            return self._to_metadata(resp.json())
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise LookupError(f"Semantic Scholar paper not found: {ss_id}") from exc
            raise

    # ------------------------------------------------------------------
    # download_pdf
    # ------------------------------------------------------------------

    def download_pdf(self, metadata: PaperMetadata, output_dir: str) -> str | None:
        """Download PDF from openAccessPdf URL if available."""
        if not metadata.pdf_url:
            logger.info("No open-access PDF URL for %s", metadata.paper_id)
            return None

        output_path = Path(output_dir) / f"{metadata.wiki_filename()}.pdf"
        if output_path.exists():
            logger.debug("PDF already cached: %s", output_path)
            return str(output_path)

        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            resp = requests.get(metadata.pdf_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(output_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
            return str(output_path)
        except Exception as exc:
            logger.warning("PDF download failed for %s: %s", metadata.paper_id, exc)
            # Clean up partial file
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_metadata(self, data: dict) -> PaperMetadata:
        ss_id = data.get("paperId", "")
        external = data.get("externalIds") or {}
        doi = external.get("DOI")
        arxiv_id = external.get("ArXiv")

        # Prefer ArXiv paper_id if available so deduplication works with arxiv source
        if arxiv_id:
            paper_id = f"arxiv:{arxiv_id}"
            source = "arxiv"
        else:
            paper_id = f"semantic_scholar:{ss_id}"
            source = "semantic_scholar"

        oa_pdf = data.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url")
        is_oa = bool(data.get("isOpenAccess", False)) or bool(pdf_url)

        venue_obj = data.get("publicationVenue") or {}
        venue = venue_obj.get("name") or data.get("venue")

        fields = data.get("fieldsOfStudy") or []
        tags = [f for f in fields if f]

        authors = []
        for a in data.get("authors") or []:
            name = a.get("name", "").strip()
            if name:
                authors.append(name)

        abstract = data.get("abstract") or ""
        # Fall back to tldr if abstract missing
        if not abstract:
            tldr = data.get("tldr") or {}
            abstract = tldr.get("text", "")

        return PaperMetadata(
            paper_id=paper_id,
            title=data.get("title", ""),
            authors=authors,
            abstract=abstract,
            year=data.get("year") or 0,
            source=source,
            pdf_url=pdf_url,
            doi=doi,
            venue=venue,
            tags=tags,
            is_open_access=is_oa,
        )
