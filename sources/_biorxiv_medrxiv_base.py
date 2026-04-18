"""
sources/_biorxiv_medrxiv_base.py — shared base for bioRxiv and medRxiv connectors.

Both servers expose an identical REST API at api.biorxiv.org / api.medrxiv.org.
This base class implements all the logic; BiorxivSource and MedrxivSource
each set _SERVER = "biorxiv" | "medrxiv" and override source_name.

Search strategy:
  The API does not support free-text search directly. We use the
  /details/{server}/{interval}/{cursor}/json endpoint to pull recent
  preprints, then filter locally by keyword match against title + abstract.
  A 2-year lookback window gives broad coverage without excessive API calls.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from sources.base import PaperMetadata, SourceConnector

logger = logging.getLogger(__name__)

_API_BASE = "https://api.biorxiv.org"   # same domain for both servers
_LOOKBACK_DAYS = 730   # 2-year window
_PAGE_SIZE = 100       # max records per API call
_REQUEST_DELAY = 0.5


class BiorxivMedrxivBase(SourceConnector):
    """
    Shared implementation for bioRxiv and medRxiv connectors.
    Subclasses set _SERVER = "biorxiv" or "medrxiv".
    """

    _SERVER: str = "biorxiv"   # overridden in subclasses

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        """
        Pull recent preprints and filter by keyword match.
        Returns up to max_results papers whose title or abstract contain
        at least one query keyword.
        """
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        end_date = date.today()
        start_date = end_date - timedelta(days=_LOOKBACK_DAYS)
        interval = f"{start_date.isoformat()}/{end_date.isoformat()}"

        results: list[PaperMetadata] = []
        cursor = 0

        while len(results) < max_results:
            url = (
                f"{_API_BASE}/details/{self._SERVER}/"
                f"{interval}/{cursor}/json"
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("%s API request failed: %s", self._SERVER, exc)
                break

            collection = data.get("collection", [])
            if not collection:
                break

            for item in collection:
                if len(results) >= max_results:
                    break
                if self._matches_keywords(item, keywords):
                    try:
                        results.append(self._to_metadata(item))
                    except Exception as exc:
                        logger.debug("Could not parse %s item: %s", self._SERVER, exc)

            total = int(data.get("messages", [{}])[0].get("total", 0))
            cursor += len(collection)
            if cursor >= total or cursor >= 5000:   # safety cap
                break

            time.sleep(_REQUEST_DELAY)

        return results[:max_results]

    # ------------------------------------------------------------------
    # fetch_metadata
    # ------------------------------------------------------------------

    def fetch_metadata(self, paper_id: str) -> PaperMetadata:
        """
        Fetch metadata by DOI-based namespaced ID.
        e.g. 'biorxiv:10.1101/2024.01.01.123456'
        """
        prefix = f"{self._SERVER}:"
        if not paper_id.startswith(prefix):
            raise ValueError(
                f"paper_id '{paper_id}' does not belong to {self._SERVER} source"
            )
        doi = paper_id[len(prefix):]
        url = f"{_API_BASE}/details/{self._SERVER}/{doi}/na/json"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise LookupError(
                    f"{self._SERVER} paper not found: {doi}"
                ) from exc
            raise

        collection = data.get("collection", [])
        if not collection:
            raise LookupError(f"{self._SERVER} paper not found: {doi}")
        # Return the most recent version
        return self._to_metadata(collection[-1])

    # ------------------------------------------------------------------
    # download_pdf
    # ------------------------------------------------------------------

    def download_pdf(self, metadata: PaperMetadata, output_dir: str) -> str | None:
        """Download the preprint PDF (always open access)."""
        if not metadata.pdf_url:
            return None

        output_path = Path(output_dir) / f"{metadata.wiki_filename()}.pdf"
        if output_path.exists():
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
            logger.warning(
                "PDF download failed for %s: %s", metadata.paper_id, exc
            )
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _matches_keywords(self, item: dict, keywords: list[str]) -> bool:
        """Return True if any keyword appears in the title or abstract."""
        if not keywords:
            return True
        haystack = (
            (item.get("title") or "") + " " + (item.get("abstract") or "")
        ).lower()
        return any(kw in haystack for kw in keywords)

    def _to_metadata(self, item: dict) -> PaperMetadata:
        doi = (item.get("doi") or "").strip()
        paper_id = f"{self._SERVER}:{doi}" if doi else f"{self._SERVER}:{item.get('biorxiv_doi', '')}"

        # Published date → year
        date_str = item.get("date") or ""
        year = int(date_str[:4]) if len(date_str) >= 4 and date_str[:4].isdigit() else 0

        # PDF URL: biorxiv/medrxiv PDFs are at https://www.biorxiv.org/content/{doi}v{version}.full.pdf
        version = str(item.get("version", "1"))
        server_host = f"www.{self._SERVER}.org"
        pdf_url = f"https://{server_host}/content/{doi}v{version}.full.pdf" if doi else None

        # Authors: comma-separated string in the API
        raw_authors = item.get("authors") or ""
        authors = [a.strip() for a in raw_authors.split(";") if a.strip()]
        if not authors:
            authors = [a.strip() for a in raw_authors.split(",") if a.strip()]

        # Category tag
        category = item.get("category") or ""
        tags = [category] if category else []

        return PaperMetadata(
            paper_id=paper_id,
            title=(item.get("title") or "").strip(),
            authors=authors,
            abstract=(item.get("abstract") or "").strip(),
            year=year,
            source=self._SERVER,
            pdf_url=pdf_url,
            doi=doi or None,
            venue=f"{self._SERVER.capitalize()} preprint",
            tags=tags,
            is_open_access=True,
        )
