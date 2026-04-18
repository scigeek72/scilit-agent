"""
agents/ingest_agent.py — LangGraph Ingest Agent.

Orchestrates the full ingest pipeline:
  1. Federated search across all configured sources
  2. Deduplication by DOI / title
  3. Open-access resolution (Unpaywall fallback)
  4. Per-paper: download PDF → parse → chunk & index → write wiki

Graph edges handle:
  - Abstract-only papers (no PDF available)
  - Parser fallback (Grobid → marker → PyMuPDF)
  - Multiple papers iterated via paper_index counter

Usage:
    from agents.ingest_agent import build_ingest_graph, run_ingest
    result = run_ingest("attention mechanisms transformer models")
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

import requests

from langgraph.graph import END, StateGraph

from agents.state import IngestState
from config import Config
from tools.llm_tools import (
    extract_tags,
    fill_paper_wiki_page,
    flag_contradictions,
    summarize_paper,
    update_concept_page,
)
from tools.parse_tools import run_parser
from tools.retrieval_tools import index_paper
from tools.source_tools import download_pdf, federated_search
from tools.wiki_tools import (
    append_wiki_log,
    make_concept_page,
    make_debate_page,
    make_paper_page,
    read_wiki_page,
    update_wiki_index,
    write_wiki_page,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def node_federated_search(state: IngestState) -> dict:
    """Fan out to all active sources, return deduplicated candidate list."""
    query = state.get("query", "")
    if not query:
        logger.warning("ingest: no query provided")
        return {"candidates": [], "paper_index": 0, "done": True}

    candidates = federated_search(query, max_results=200)
    logger.info("federated_search: %d candidates for '%s'", len(candidates), query[:60])
    return {
        "candidates": candidates,
        "paper_index": 0,
        "papers_processed": [],
        "wiki_pages_written": [],
        "chunks_indexed": 0,
        "errors": [],
        "done": False,
    }


def node_next_paper(state: IngestState) -> dict:
    """
    Advance paper_index and load the next paper into state.paper.
    Sets done=True when all candidates have been processed.
    """
    candidates = state.get("candidates", [])
    index = state.get("paper_index", 0)

    if index >= len(candidates):
        return {"done": True, "paper": None}

    paper = candidates[index]
    return {
        "paper": paper,
        "paper_index": index + 1,
        "pdf_path": None,
        "parsed_paper": None,
        "is_abstract_only": False,
    }


def node_download_pdf(state: IngestState) -> dict:
    """
    Download PDF for the current paper. If unavailable, mark is_abstract_only.
    Tries Unpaywall as a fallback for paywalled papers.
    """
    paper = state.get("paper")
    if not paper:
        return {"is_abstract_only": True, "pdf_path": None}

    # First attempt: normal download
    pdf_path = download_pdf(paper)

    if pdf_path:
        return {"pdf_path": pdf_path, "is_abstract_only": False}

    # Fallback: try Unpaywall if we have a DOI
    doi = paper.get("doi")
    if doi:
        pdf_path = _try_unpaywall(doi, paper)

    if pdf_path:
        logger.info("Unpaywall PDF: %s → %s", paper.get("paper_id"), pdf_path)
        return {"pdf_path": pdf_path, "is_abstract_only": False}

    logger.info("No PDF available for %s — abstract only", paper.get("paper_id"))
    return {"pdf_path": None, "is_abstract_only": True}


def node_parse_paper(state: IngestState) -> dict:
    """
    Route to the correct parser (Grobid / marker / PyMuPDF) and parse the paper.
    For abstract-only papers, builds a stub ParsedPaper from metadata.
    """
    paper     = state.get("paper", {})
    pdf_path  = state.get("pdf_path")
    abstract_only = state.get("is_abstract_only", False)

    if abstract_only:
        # Build a minimal ParsedPaper stub from metadata
        parsed = _build_abstract_only_parsed(paper)
        return {"parsed_paper": parsed}

    try:
        parsed = run_parser(pdf_path, paper)
        return {"parsed_paper": parsed}
    except Exception as exc:
        logger.error("parse_paper failed for %s: %s", paper.get("paper_id"), exc)
        errors = list(state.get("errors", []))
        errors.append(f"parse error for {paper.get('paper_id', '?')}: {exc}")
        # Fall back to abstract-only stub
        parsed = _build_abstract_only_parsed(paper)
        return {"parsed_paper": parsed, "is_abstract_only": True, "errors": errors}


def node_chunk_and_index(state: IngestState) -> dict:
    """
    Chunk the parsed paper and add to ChromaDB + BM25 index.
    Skips if parsed_paper is missing (error recovery).
    """
    parsed = state.get("parsed_paper")
    if not parsed:
        return {}

    try:
        n_chunks = index_paper(parsed)
        total = state.get("chunks_indexed", 0) + n_chunks
        logger.info("Indexed %d chunks for %s", n_chunks, parsed.get("paper_id"))
        return {"chunks_indexed": total}
    except Exception as exc:
        logger.error("chunk_and_index failed for %s: %s", parsed.get("paper_id"), exc)
        errors = list(state.get("errors", []))
        errors.append(f"index error for {parsed.get('paper_id', '?')}: {exc}")
        return {"errors": errors}


def node_write_paper_wiki_page(state: IngestState) -> dict:
    """
    Create or update wiki/papers/{source}-{id}.md using LLM-filled sections.
    """
    parsed = state.get("parsed_paper")
    paper  = state.get("paper", {})
    if not parsed:
        return {}

    paper_id      = parsed.get("paper_id", paper.get("paper_id", ""))
    abstract_only = state.get("is_abstract_only", parsed.get("is_abstract_only", False))

    try:
        # Build wiki page filename: replace colons and slashes
        filename  = paper_id.replace(":", "-").replace("/", "-") + ".md"
        page_path = f"papers/{filename}"

        # Generate tags and summary via LLM
        tags    = extract_tags(parsed)
        summary = summarize_paper(parsed)

        # Build stub then fill sections
        stub = make_paper_page(
            paper_id=paper_id,
            title=parsed.get("title", paper.get("title", "")),
            authors=parsed.get("authors", paper.get("authors", [])),
            year=parsed.get("year", paper.get("year", 0)),
            source=parsed.get("source", paper.get("source", "")),
            venue=paper.get("venue", ""),
            parser_used=parsed.get("parser_used", "pymupdf"),
            math_fraction=parsed.get("math_fraction", 0.0),
            is_abstract_only=abstract_only,
            tags=tags,
            pdf_url=paper.get("pdf_url", "") or "",
            summary=summary,
        )
        page_content = fill_paper_wiki_page(stub, parsed)

        write_wiki_page(page_path, page_content, reason=f"ingest {paper_id}")

        written = list(state.get("wiki_pages_written", []))
        written.append(page_path)
        return {"wiki_pages_written": written}

    except Exception as exc:
        logger.error("write_paper_wiki_page failed for %s: %s", paper_id, exc)
        errors = list(state.get("errors", []))
        errors.append(f"wiki page error for {paper_id}: {exc}")
        return {"errors": errors}


def node_update_concept_pages(state: IngestState) -> dict:
    """
    Update or create wiki/concepts/*.md for each concept tag in the paper.
    Uses LLM to merge new paper into existing concept page.
    """
    parsed = state.get("parsed_paper")
    if not parsed:
        return {}

    tags     = parsed.get("tags", [])
    paper_id = parsed.get("paper_id", "")
    written  = list(state.get("wiki_pages_written", []))

    for tag in tags[:4]:  # limit to 4 concepts per paper to avoid excess LLM calls
        concept_slug = _slugify(tag)
        concept_path = f"concepts/{concept_slug}.md"

        try:
            existing = read_wiki_page(concept_path)
            if not existing:
                existing = make_concept_page(concept=tag, tags=[tag])

            updated = update_concept_page(existing, parsed, concept_name=tag)
            write_wiki_page(concept_path, updated, reason=f"concept update from {paper_id}")
            update_wiki_index(
                page_path=concept_path,
                summary=f"Concept page for {tag}",
                category="concepts",
            )
            written.append(concept_path)

        except Exception as exc:
            logger.warning("update_concept_pages failed for '%s': %s", tag, exc)

    return {"wiki_pages_written": written}


def node_check_contradictions(state: IngestState) -> dict:
    """
    Detect contradictions between new paper claims and existing wiki claims.
    Updates wiki/debates/*.md if contradictions are found.
    """
    parsed = state.get("parsed_paper")
    if not parsed:
        return {}

    paper_id = parsed.get("paper_id", "")

    # Gather existing claims from related concept pages
    existing_claims: list[str] = []
    for tag in (parsed.get("tags") or [])[:3]:
        concept_path = f"concepts/{_slugify(tag)}.md"
        page = read_wiki_page(concept_path)
        if page:
            # Extract bullet-point claims
            claims = re.findall(r"^[-*]\s+(.+)$", page, re.MULTILINE)
            existing_claims.extend(claims[:5])

    if not existing_claims:
        return {}

    try:
        contradictions = flag_contradictions(existing_claims, parsed)
        if not contradictions:
            return {}

        # Create or update debates page
        slug  = _slugify(paper_id.split(":")[-1][:40]) + "-debate"
        dpath = f"debates/{slug}.md"
        existing_debate = read_wiki_page(dpath)
        if not existing_debate:
            existing_debate = make_debate_page(
                topic=f"Contradictions involving {paper_id}",
                domains=[parsed.get("source", "unknown")],
            )

        # Append contradiction notes
        contradiction_block = "\n".join(f"- {c}" for c in contradictions)
        updated_debate = existing_debate.rstrip() + (
            f"\n\n### From [[papers/{paper_id.replace(':', '-').replace('/', '-')}]]\n"
            f"{contradiction_block}\n"
        )
        write_wiki_page(dpath, updated_debate, reason=f"contradiction from {paper_id}")
        update_wiki_index(dpath, f"Debate involving {paper_id}", "debates")

        written = list(state.get("wiki_pages_written", []))
        written.append(dpath)
        return {"wiki_pages_written": written}

    except Exception as exc:
        logger.warning("check_contradictions failed for %s: %s", paper_id, exc)
        return {}


def node_update_index_and_log(state: IngestState) -> dict:
    """
    Update wiki/index.md and append an entry to wiki/log.md.
    Records every successfully processed paper.
    """
    parsed  = state.get("parsed_paper")
    paper   = state.get("paper", {})
    written = state.get("wiki_pages_written", [])

    paper_id = (parsed or {}).get("paper_id", paper.get("paper_id", "?"))
    title    = (parsed or {}).get("title",    paper.get("title", "?"))

    # Update index for the paper page
    paper_page = next((p for p in written if p.startswith("papers/")), None)
    if paper_page:
        try:
            update_wiki_index(
                page_path=paper_page,
                summary=title[:80],
                category="papers",
            )
        except Exception as exc:
            logger.warning("update_wiki_index failed: %s", exc)

    # Append log entry
    abstract_only = state.get("is_abstract_only", False)
    chunks        = state.get("chunks_indexed", 0)
    parser_used   = (parsed or {}).get("parser_used", "none")
    access        = "abstract_only" if abstract_only else "full"
    n_wiki        = len(written)

    log_details = (
        f"Parser: {parser_used} | Chunks: {chunks} | "
        f"Wiki pages updated: {n_wiki} | Access: {access}"
    )
    if abstract_only:
        log_details += f"\nNote: PDF unavailable — stub page created"

    try:
        append_wiki_log(operation="ingest", title=f"{paper_id} — {title[:50]}", details=log_details)
    except Exception as exc:
        logger.warning("append_wiki_log failed: %s", exc)

    # Accumulate processed paper_ids
    processed = list(state.get("papers_processed", []))
    processed.append(paper_id)

    # Reset per-paper fields, keep accumulators
    return {
        "papers_processed": processed,
        "paper": None,
        "pdf_path": None,
        "parsed_paper": None,
        "is_abstract_only": False,
        "wiki_pages_written": [],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _route_after_next_paper(state: IngestState) -> str:
    """After node_next_paper: if done → end, else → download_pdf."""
    if state.get("done"):
        return "end"
    return "download_pdf"


def build_ingest_graph() -> Any:
    """
    Build and compile the Ingest Agent LangGraph StateGraph.

    Graph topology:
      federated_search
           │
      next_paper ──(done)──→ END
           │
      download_pdf
           │
      parse_paper
           │
      chunk_and_index
           │
      write_paper_wiki_page
           │
      update_concept_pages
           │
      check_contradictions
           │
      update_index_and_log
           │
      next_paper  (loop back)
    """
    graph = StateGraph(IngestState)

    # Register nodes
    graph.add_node("federated_search",       node_federated_search)
    graph.add_node("next_paper",             node_next_paper)
    graph.add_node("download_pdf",           node_download_pdf)
    graph.add_node("parse_paper",            node_parse_paper)
    graph.add_node("chunk_and_index",        node_chunk_and_index)
    graph.add_node("write_paper_wiki_page",  node_write_paper_wiki_page)
    graph.add_node("update_concept_pages",   node_update_concept_pages)
    graph.add_node("check_contradictions",   node_check_contradictions)
    graph.add_node("update_index_and_log",   node_update_index_and_log)

    # Entry point
    graph.set_entry_point("federated_search")

    # Linear edges
    graph.add_edge("federated_search", "next_paper")

    # Conditional: done → END, else → download_pdf
    graph.add_conditional_edges(
        "next_paper",
        _route_after_next_paper,
        {"end": END, "download_pdf": "download_pdf"},
    )

    # Main pipeline
    graph.add_edge("download_pdf",          "parse_paper")
    graph.add_edge("parse_paper",           "chunk_and_index")
    graph.add_edge("chunk_and_index",       "write_paper_wiki_page")
    graph.add_edge("write_paper_wiki_page", "update_concept_pages")
    graph.add_edge("update_concept_pages",  "check_contradictions")
    graph.add_edge("check_contradictions",  "update_index_and_log")

    # Loop back to pick up the next paper
    graph.add_edge("update_index_and_log",  "next_paper")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ingest(query: str) -> IngestState:
    """
    Run the full ingest pipeline for a plain-text query.

    Returns the final IngestState with:
      papers_processed: list of paper_ids that were ingested
      chunks_indexed:   total number of chunks added
      errors:           any non-fatal errors encountered
    """
    app = build_ingest_graph()
    final_state = app.invoke({"query": query})
    n = len(final_state.get("papers_processed", []))
    logger.info(
        "Ingest complete: %d papers, %d chunks, %d errors",
        n,
        final_state.get("chunks_indexed", 0),
        len(final_state.get("errors", [])),
    )
    return final_state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert a string to a lowercase hyphenated slug for filenames."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60]


def _build_abstract_only_parsed(paper: dict) -> dict:
    """
    Build a minimal ParsedPaper stub from PaperMetadata when no PDF is available.
    Marks is_abstract_only=True and parser_used='none'.
    """
    return {
        "paper_id":       paper.get("paper_id", ""),
        "title":          paper.get("title", ""),
        "authors":        paper.get("authors", []),
        "abstract":       paper.get("abstract", ""),
        "year":           paper.get("year", 0),
        "source":         paper.get("source", ""),
        "sections":       [],
        "references":     [],
        "figures":        [],
        "tables":         [],
        "equations":      [],
        "parser_used":    "none",
        "math_fraction":  0.0,
        "is_abstract_only": True,
        "tags":           paper.get("tags", []),
    }


def _try_unpaywall(doi: str, paper: dict) -> str | None:
    """
    Try Unpaywall API for an open-access PDF URL.
    Returns a local file path if download succeeds, else None.
    No API key required for low volume; falls back silently on any error.
    """
    try:
        email = "scilit-agent@example.com"
        url   = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        resp  = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        best_oa = data.get("best_oa_location") or {}
        pdf_url = best_oa.get("url_for_pdf")
        if not pdf_url:
            return None

        # Download to raw PDF dir
        output_dir = Config.raw_pdf_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        pid      = paper.get("paper_id", "unknown").replace(":", "-").replace("/", "-")
        filename = f"{pid}.pdf"
        pdf_path = output_dir / filename

        pdf_resp = requests.get(pdf_url, timeout=60, stream=True)
        if pdf_resp.status_code == 200:
            with open(pdf_path, "wb") as f:
                for chunk in pdf_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return str(pdf_path)
        return None

    except Exception as exc:
        logger.debug("Unpaywall lookup failed for DOI %s: %s", doi, exc)
        return None
