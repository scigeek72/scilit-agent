"""
tools/source_tools.py — Agent-callable wrappers over the source federation layer.

These are thin wrappers that convert PaperMetadata dataclass objects to plain
dicts (JSON-serialisable) for use inside LangGraph state.

Functions:
  federated_search    — fan-out search across all active sources
  fetch_paper_metadata — fetch full metadata for a known paper_id
  download_pdf        — download PDF to disk, return path or None
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


def federated_search(query: str, max_results: int = 200) -> list[dict]:
    """
    Search all configured sources for papers matching the plain-text query.

    Returns a deduplicated, ranked list of PaperMetadata dicts.
    Use this as the entry point for any new paper discovery.
    The user never needs to specify which source to search.

    Gracefully handles individual source failures — a source that errors
    is skipped and logged; the remaining sources still contribute results.
    """
    from sources.federation import federated_search as _search
    results = _search(query, max_total=max_results)
    dicts = [p.to_dict() for p in results]
    logger.info("federated_search('%s'): %d papers", query[:60], len(dicts))
    return dicts


def fetch_paper_metadata(paper_id: str) -> dict | None:
    """
    Fetch full metadata for a known paper by its namespaced ID.
    e.g. 'arxiv:2301.12345', 'pubmed:38291847'

    Routes to the correct source connector automatically.
    Returns None if the paper cannot be found.
    """
    from sources.federation import get_connector

    source = paper_id.split(":")[0] if ":" in paper_id else "semantic_scholar"
    try:
        connector = get_connector(source)
        meta = connector.fetch_metadata(paper_id)
        return meta.to_dict() if meta else None
    except Exception as exc:
        logger.warning("fetch_paper_metadata(%s) failed: %s", paper_id, exc)
        return None


def download_pdf(metadata: dict, output_dir: str | None = None) -> str | None:
    """
    Download the PDF for a paper. Returns local file path, or None if unavailable
    (paywalled or no PDF link). Never raises — logs failure and returns None.

    If None is returned: paper will be indexed as abstract_only.
    output_dir defaults to Config.raw_pdf_dir().
    """
    from sources.federation import get_connector
    from sources.base import PaperMetadata

    output_dir = output_dir or str(Config.raw_pdf_dir())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    source = metadata.get("source", "")
    if not source:
        logger.warning("download_pdf: no source in metadata for %s", metadata.get("paper_id"))
        return None

    try:
        connector = get_connector(source)
        paper = PaperMetadata.from_dict(metadata)
        path = connector.download_pdf(paper, output_dir)
        if path:
            logger.info("PDF downloaded: %s → %s", metadata.get("paper_id"), path)
        else:
            logger.info("PDF unavailable for %s (abstract only)", metadata.get("paper_id"))
        return path
    except Exception as exc:
        logger.warning("download_pdf(%s) failed: %s", metadata.get("paper_id"), exc)
        return None
