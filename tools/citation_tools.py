"""
tools/citation_tools.py — Citation graph tools via Semantic Scholar API.

Uses the semanticscholar Python library which handles rate-limiting and
retries automatically.  Works for papers from any source (arxiv, pubmed,
DOI-based) — Semantic Scholar normalises identifiers internally.

Functions:
  get_references  — papers cited BY this paper
  get_cited_by    — papers that CITE this paper
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Map our source prefixes to Semantic Scholar identifier prefixes
_ID_MAP = {
    "arxiv":   "arXiv:",
    "pubmed":  "PMID:",
    "doi":     "DOI:",
}


def _to_ss_id(paper_id: str) -> str:
    """
    Convert a namespaced paper_id to a Semantic Scholar identifier string.

    arxiv:2301.12345 → arXiv:2301.12345
    pubmed:38291847  → PMID:38291847
    doi:10.1234/...  → DOI:10.1234/...
    semantic_scholar:abc123 → abc123  (already an S2 corpusId or paperId)
    """
    if ":" not in paper_id:
        return paper_id
    prefix, rest = paper_id.split(":", 1)
    ss_prefix = _ID_MAP.get(prefix.lower())
    if ss_prefix:
        return ss_prefix + rest
    # semantic_scholar or local — return the rest directly
    return rest


def get_references(paper_id: str) -> list[dict]:
    """
    Papers cited by this paper. Uses Semantic Scholar API.

    Works for any source (arxiv, pubmed, doi-based).
    Returns list of {title, authors, year, paper_id, doi, url}.

    Returns empty list on any API error — graceful degradation.
    """
    try:
        import semanticscholar as ss
        api = ss.SemanticScholar()
        ss_id = _to_ss_id(paper_id)
        paper = api.get_paper(ss_id, fields=["references"])
        if paper is None or not hasattr(paper, "references"):
            return []
        return [_ref_to_dict(r) for r in (paper.references or [])]
    except Exception as exc:
        logger.warning("get_references(%s) failed: %s", paper_id, exc)
        return []


def get_cited_by(paper_id: str, limit: int = 20) -> list[dict]:
    """
    Papers that cite this paper. Uses Semantic Scholar API.

    Use to find follow-up work and assess impact.
    Returns list of {title, authors, year, paper_id, doi, url}.

    Returns empty list on any API error — graceful degradation.
    """
    try:
        import semanticscholar as ss
        api = ss.SemanticScholar()
        ss_id = _to_ss_id(paper_id)
        paper = api.get_paper(ss_id, fields=["citations"])
        if paper is None or not hasattr(paper, "citations"):
            return []
        citations = (paper.citations or [])[:limit]
        return [_ref_to_dict(c) for c in citations]
    except Exception as exc:
        logger.warning("get_cited_by(%s) failed: %s", paper_id, exc)
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ref_to_dict(ref) -> dict:
    """Convert a Semantic Scholar reference/citation object to a plain dict."""
    try:
        authors = [a.get("name", "") for a in (ref.authors or [])] if hasattr(ref, "authors") else []
        year = ref.year if hasattr(ref, "year") and ref.year else 0
        doi = ref.externalIds.get("DOI", "") if hasattr(ref, "externalIds") and ref.externalIds else ""

        # Build a namespaced paper_id
        pid = ""
        if hasattr(ref, "externalIds") and ref.externalIds:
            if ref.externalIds.get("ArXiv"):
                pid = f"arxiv:{ref.externalIds['ArXiv']}"
            elif ref.externalIds.get("PubMed"):
                pid = f"pubmed:{ref.externalIds['PubMed']}"
            elif doi:
                pid = f"doi:{doi}"
        if not pid and hasattr(ref, "paperId") and ref.paperId:
            pid = f"semantic_scholar:{ref.paperId}"

        return {
            "paper_id": pid,
            "title":    ref.title if hasattr(ref, "title") and ref.title else "",
            "authors":  authors,
            "year":     int(year) if year else 0,
            "doi":      doi,
            "url":      ref.url if hasattr(ref, "url") and ref.url else "",
        }
    except Exception:
        return {"paper_id": "", "title": "", "authors": [], "year": 0, "doi": "", "url": ""}
