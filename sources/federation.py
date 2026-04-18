"""
sources/federation.py — federated_search() and deduplication logic.

federated_search() fans out a plain-text query to all active sources
in parallel, deduplicates the results, and returns a unified ranked list.
The caller never knows which source each paper came from — that
information is in PaperMetadata.source.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from thefuzz import fuzz

from config import Config
from sources.base import PaperMetadata, SourceConnector

if TYPE_CHECKING:
    pass   # avoid circular import at runtime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

def _build_registry() -> dict[str, type[SourceConnector]]:
    """Return a mapping of source_name → connector class."""
    from sources.arxiv_source import ArxivSource
    from sources.semantic_scholar_source import SemanticScholarSource
    from sources.local_pdf_source import LocalPdfSource

    registry: dict[str, type[SourceConnector]] = {
        "arxiv": ArxivSource,
        "semantic_scholar": SemanticScholarSource,
        "local_pdf": LocalPdfSource,
    }

    # Phase 2 sources — import only if available
    try:
        from sources.pubmed_source import PubMedSource
        registry["pubmed"] = PubMedSource
    except ImportError:
        pass

    try:
        from sources.biorxiv_source import BiorxivSource
        registry["biorxiv"] = BiorxivSource
    except ImportError:
        pass

    try:
        from sources.medrxiv_source import MedrxivSource
        registry["medrxiv"] = MedrxivSource
    except ImportError:
        pass

    return registry


_REGISTRY: dict[str, type[SourceConnector]] | None = None


def _get_registry() -> dict[str, type[SourceConnector]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_connector(source_name: str) -> SourceConnector:
    """Instantiate and return a connector for the given source name."""
    registry = _get_registry()
    cls = registry.get(source_name)
    if cls is None:
        raise ValueError(
            f"Unknown source '{source_name}'. "
            f"Available: {sorted(registry.keys())}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

# Source priority for deduplication (index 0 = highest priority)
_SOURCE_PRIORITY: list[str] = [
    "arxiv",
    "biorxiv",
    "medrxiv",
    "pubmed",
    "semantic_scholar",
    "local_pdf",
    "local",
]

_TITLE_SIMILARITY_THRESHOLD: float = 92.0   # fuzz.token_sort_ratio score 0–100


def _source_priority(source: str) -> int:
    """Lower = higher priority."""
    try:
        return _SOURCE_PRIORITY.index(source)
    except ValueError:
        return len(_SOURCE_PRIORITY)


def _titles_are_duplicates(title_a: str, title_b: str) -> bool:
    """Return True if two titles are similar enough to be the same paper."""
    score = fuzz.token_sort_ratio(title_a.lower(), title_b.lower())
    return score >= _TITLE_SIMILARITY_THRESHOLD


def deduplicate(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """
    Remove duplicate papers from a mixed-source list.

    Deduplication priority order:
    1. Exact DOI match → keep highest-priority source copy
    2. Title similarity >= 92 → keep highest-priority source copy
    3. No match → keep both

    The input list order determines ranking within each priority bucket.
    """
    seen_dois: dict[str, PaperMetadata] = {}     # doi → kept paper
    kept: list[PaperMetadata] = []               # final output list (in order)

    for paper in papers:
        # --- Pass 1: DOI deduplication ---
        if paper.doi:
            existing = seen_dois.get(paper.doi)
            if existing is None:
                seen_dois[paper.doi] = paper
                kept.append(paper)
            else:
                # Replace if current paper comes from a higher-priority source
                if _source_priority(paper.source) < _source_priority(existing.source):
                    # Swap the kept entry for the higher-priority version
                    idx = kept.index(existing)
                    kept[idx] = paper
                    seen_dois[paper.doi] = paper
                    logger.debug(
                        "DOI dedup: replaced %s (%s) with %s (%s)",
                        existing.paper_id, existing.source,
                        paper.paper_id, paper.source,
                    )
            continue

        # --- Pass 2: title-similarity deduplication ---
        duplicate_found = False
        for existing in kept:
            if _titles_are_duplicates(paper.title, existing.title):
                duplicate_found = True
                if _source_priority(paper.source) < _source_priority(existing.source):
                    idx = kept.index(existing)
                    kept[idx] = paper
                    # Update doi index if the replacement has a DOI
                    if paper.doi:
                        seen_dois[paper.doi] = paper
                    logger.debug(
                        "Title dedup: replaced %s with %s",
                        existing.paper_id, paper.paper_id,
                    )
                break

        if not duplicate_found:
            kept.append(paper)
            if paper.doi:
                seen_dois[paper.doi] = paper

    return kept


# ---------------------------------------------------------------------------
# Open-access resolution
# ---------------------------------------------------------------------------

def resolve_open_access(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """
    For each paper without a pdf_url, try the Unpaywall API to find an OA version.
    Papers without a DOI are skipped (Unpaywall requires a DOI).
    Modifies papers in-place and returns the same list.
    """
    import requests

    unpaywall_email = "scilit-agent@example.com"   # Unpaywall requires any email

    for paper in papers:
        if paper.pdf_url or not paper.doi:
            continue
        try:
            resp = requests.get(
                f"https://api.unpaywall.org/v2/{paper.doi}",
                params={"email": unpaywall_email},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                oa_location = data.get("best_oa_location") or {}
                url_for_pdf = oa_location.get("url_for_pdf")
                if url_for_pdf:
                    paper.pdf_url = url_for_pdf
                    paper.is_open_access = True
                    logger.info(
                        "Unpaywall OA found for %s: %s", paper.paper_id, url_for_pdf
                    )
        except Exception as exc:
            logger.debug("Unpaywall lookup failed for %s: %s", paper.doi, exc)

    return papers


# ---------------------------------------------------------------------------
# Federated search
# ---------------------------------------------------------------------------

def federated_search(
    query: str,
    max_total: int = 200,
) -> list[PaperMetadata]:
    """
    Submit the plain-text query to all active sources in parallel.
    Deduplicate by DOI then by title similarity.
    Attempt Unpaywall OA resolution for paywalled results.
    Return a unified ranked list.

    The caller never needs to name a source — all routing is internal.
    """
    active = Config.ACTIVE_SOURCES
    max_per_source = Config.SOURCE_MAX_RESULTS
    registry = _get_registry()

    # Only search sources that are both active and have a connector
    sources_to_search = [s for s in active if s in registry]
    unknown = [s for s in active if s not in registry]
    if unknown:
        logger.warning("No connector for active sources: %s — skipping", unknown)

    all_results: list[PaperMetadata] = []

    def _search_one(source_name: str) -> list[PaperMetadata]:
        limit = max_per_source.get(source_name, 50)
        connector = get_connector(source_name)
        try:
            results = connector.search(query, max_results=limit)
            logger.info(
                "Source '%s' returned %d results for query: %r",
                source_name, len(results), query,
            )
            return results
        except Exception as exc:
            logger.warning("Source '%s' search raised unexpectedly: %s", source_name, exc)
            return []

    # Fan out in parallel (one thread per source)
    with ThreadPoolExecutor(max_workers=len(sources_to_search) or 1) as pool:
        futures = {
            pool.submit(_search_one, source_name): source_name
            for source_name in sources_to_search
        }
        for future in as_completed(futures):
            source_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                logger.error("Unexpected error from source '%s': %s", source_name, exc)

    logger.info(
        "Total pre-dedup results: %d from %d sources",
        len(all_results), len(sources_to_search),
    )

    # Deduplicate
    deduped = deduplicate(all_results)
    logger.info("Post-dedup count: %d", len(deduped))

    # Open-access resolution for paywalled papers
    deduped = resolve_open_access(deduped)

    # Cap at max_total
    return deduped[:max_total]
