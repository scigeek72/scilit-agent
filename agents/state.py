"""
agents/state.py — TypedDict state schemas for all LangGraph agents.

Each agent has its own state type.  Nodes receive the full state dict and
return a partial dict with only the fields they modify — LangGraph merges
the updates automatically.

IngestState  — used by the Ingest Agent (Phase 6)
QueryState   — used by the Query Agent  (Phase 7)
LintState    — used by the Lint Agent   (Phase 8)
FrontierState— used by the Frontier Agent (Phase 11)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Ingest Agent state
# ---------------------------------------------------------------------------

class IngestState(TypedDict, total=False):
    # Input
    query: str                               # original search query

    # Federation output
    candidates: List[Dict]                   # list of PaperMetadata.to_dict()

    # Per-paper processing (one paper at a time through the pipeline)
    paper: Optional[Dict]                    # current PaperMetadata dict
    pdf_path: Optional[str]                  # local path to downloaded PDF
    parsed_paper: Optional[Dict]             # ParsedPaper dict from router
    is_abstract_only: bool                   # True if no PDF available

    # Accumulation across all papers
    papers_processed: List[str]              # paper_ids successfully processed
    wiki_pages_written: List[str]            # relative wiki paths written
    chunks_indexed: int                      # total chunks added to ChromaDB + BM25
    errors: List[str]                        # non-fatal error messages

    # Control flow
    paper_index: int                         # index into candidates list
    done: bool                               # True when all candidates processed


# ---------------------------------------------------------------------------
# Query Agent state
# ---------------------------------------------------------------------------

class QueryState(TypedDict, total=False):
    # Input
    query: str
    year_filter: Optional[int]
    source_filter: Optional[str]

    # Classification
    query_type: Optional[str]               # 'factual' | 'comparative' | 'exploratory' | 'survey'
    sub_queries: List[str]                  # from decompose_query()

    # Wiki retrieval
    wiki_pages_read: List[str]              # relative paths of wiki pages consulted
    wiki_context: str                       # concatenated wiki page content

    # ChromaDB retrieval
    retrieved_chunks: List[Dict]            # raw chunks from hybrid_search
    reranked_chunks: List[Dict]             # after cross-encoder rerank

    # Answer generation
    answer: Optional[str]
    sources: List[Dict]                     # chunks / wiki pages cited
    is_grounded: bool                       # self_critique result
    confidence: float                       # 0.0–1.0
    retry_count: int

    # Cache + file-back
    cache_hit: bool
    should_file_back: bool
    filed_page_path: Optional[str]


# ---------------------------------------------------------------------------
# Lint Agent state
# ---------------------------------------------------------------------------

class LintState(TypedDict, total=False):
    link_graph: Dict[str, List[str]]        # {page_path: [pages_linking_to_it]}
    orphans: List[str]                      # pages with zero inbound links
    contradictions: List[str]               # unresolved debate pages
    stale_claims: List[str]                 # claims older than newest papers
    missing_concept_pages: List[str]        # concepts referenced but no page exists
    gaps: List[str]                         # suggested next searches
    report_path: Optional[str]              # wiki/synthesis/lint-{date}.md


# ---------------------------------------------------------------------------
# Frontier Agent state (Phase 11)
# ---------------------------------------------------------------------------

class FrontierState(TypedDict, total=False):
    query: str
    query_focus: str                        # 'methodological' | 'conceptual' | 'both'
    open_questions: List[Dict]              # from extract_open_questions()
    method_domain_gaps: List[Dict]
    temporal_dropouts: List[Dict]
    contradiction_clusters: List[Dict]
    cross_domain_opportunities: List[Dict]
    kg_gap_edges: List[Dict]               # populated only if USE_KG=True
    wiki_context: Dict
    report: Optional[str]
    filed_page_path: Optional[str]
