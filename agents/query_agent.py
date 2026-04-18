"""
agents/query_agent.py — LangGraph Query Agent.

Handles the full query pipeline:
  1. Cache check (SQLite) — return immediately on hit
  2. Classify query type (factual / comparative / exploratory / survey)
  3. Read wiki index → find relevant pages
  4. Read relevant wiki pages
  5. Decide if wiki alone suffices or ChromaDB chunks are needed
  6. HyDE for exploratory queries / decompose for comparative/survey
  7. Hybrid search + rerank
  8. Synthesize answer
  9. Self-critique → refine once if not grounded
  10. File back to wiki if is_worth_filing()
  11. Save to cache

Usage:
    from agents.query_agent import build_query_graph, run_query
    result = run_query("What are the main challenges in CRISPR delivery?")
    print(result["answer"])
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from langgraph.graph import END, StateGraph

from agents.state import QueryState
from config import Config
from retrieval.query_cache import get_query_cache
from tools.llm_tools import (
    decompose_query,
    is_worth_filing,
    self_critique,
    synthesize_answer,
)
from tools.retrieval_tools import hybrid_search, rerank_chunks, search_vector_db
from tools.wiki_tools import (
    append_wiki_log,
    list_wiki_pages,
    make_synthesis_page,
    read_wiki_index,
    read_wiki_page,
    search_wiki,
    update_wiki_index,
    write_wiki_page,
)

logger = logging.getLogger(__name__)

# Minimum self_critique confidence to accept without retry
_CONFIDENCE_THRESHOLD = 0.6
# Number of wiki pages to read before falling back to ChromaDB
_MAX_WIKI_PAGES_TO_READ = 5
# Keywords that indicate a complex query needing decomposition
_COMPLEX_QUERY_KEYWORDS = re.compile(
    r"\b(compare|comparison|vs\.?|versus|difference|survey|review|explain|overview|"
    r"what are|how do|advantages|disadvantages|pros|cons|contrast)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def node_check_cache(state: QueryState) -> dict:
    """Check the SQLite cache. On hit, populate answer and set cache_hit=True."""
    if not Config.USE_PERSISTENT_CACHE:
        return {"cache_hit": False, "retry_count": 0}

    query = state.get("query", "")
    if not query:
        return {"cache_hit": False, "retry_count": 0}

    try:
        cache = get_query_cache()
        hit = cache.get(query)
        if hit:
            logger.info("Cache HIT for query: %s…", query[:60])
            return {
                "cache_hit":  True,
                "answer":     hit["answer"],
                "sources":    hit["sources"],
                "confidence": hit["confidence"],
                "is_grounded": True,
                "should_file_back": False,
            }
    except Exception as exc:
        logger.warning("Cache lookup failed: %s", exc)

    return {"cache_hit": False, "retry_count": 0}


def node_classify_query(state: QueryState) -> dict:
    """
    Classify the query into: factual | comparative | exploratory | survey.

    - factual:      specific question with a definite answer
    - comparative:  comparing two or more things
    - exploratory:  broad/vague question needing synthesis
    - survey:       request for a structured overview of a topic
    """
    query = state.get("query", "")

    if _COMPLEX_QUERY_KEYWORDS.search(query):
        # Distinguish survey from comparative
        if re.search(r"\b(survey|review|overview|summarize|list)\b", query, re.IGNORECASE):
            qtype = "survey"
        else:
            qtype = "comparative"
    elif re.search(r"\b(what is|define|who|when|where|which)\b", query, re.IGNORECASE):
        qtype = "factual"
    else:
        qtype = "exploratory"

    logger.debug("Query classified as: %s", qtype)
    return {"query_type": qtype}


def node_read_wiki_index(state: QueryState) -> dict:
    """
    Read wiki/index.md and find the most relevant page paths for this query.
    Returns sub_queries if query should be decomposed.
    """
    query      = state.get("query", "")
    query_type = state.get("query_type", "factual")

    # Decompose complex queries into sub-queries for better coverage
    sub_queries: list[str] = state.get("sub_queries", [])
    if not sub_queries and query_type in ("comparative", "survey"):
        try:
            sub_queries = decompose_query(query)
        except Exception as exc:
            logger.warning("decompose_query failed: %s", exc)
            sub_queries = [query]
    if not sub_queries:
        sub_queries = [query]

    index_content = read_wiki_index()

    # BM25 search over wiki pages
    search_query = " ".join(sub_queries[:2])  # use first two sub-queries for coverage
    wiki_hits = search_wiki(search_query, top_k=_MAX_WIKI_PAGES_TO_READ)
    relevant_paths = [h["page_path"] for h in wiki_hits]

    # Also include any concept pages matching query terms
    concept_pages = list_wiki_pages("concepts")
    query_words = set(re.findall(r"[a-z]{4,}", query.lower()))
    for cp in concept_pages:
        slug = cp.replace("concepts/", "").replace(".md", "").replace("-", " ")
        if any(w in slug for w in query_words):
            if cp not in relevant_paths:
                relevant_paths.append(cp)

    return {
        "sub_queries":     sub_queries,
        "wiki_pages_read": relevant_paths[:_MAX_WIKI_PAGES_TO_READ],
        "wiki_context":    index_content[:500],  # seed with index snippet
    }


def node_read_wiki_pages(state: QueryState) -> dict:
    """Read each relevant wiki page and concatenate into wiki_context."""
    pages_to_read = state.get("wiki_pages_read", [])
    if not pages_to_read:
        return {"wiki_context": ""}

    parts: list[str] = []
    for page_path in pages_to_read[:_MAX_WIKI_PAGES_TO_READ]:
        content = read_wiki_page(page_path)
        if content:
            parts.append(f"--- {page_path} ---\n{content[:1200]}")

    wiki_context = "\n\n".join(parts)
    return {"wiki_context": wiki_context}


def node_decide_retrieval(state: QueryState) -> dict:
    """
    Decide whether the wiki context is sufficient or if ChromaDB chunks
    are also needed. Sets an internal routing flag via a sentinel field.

    Wiki-only is sufficient for: factual queries where the wiki has
    direct coverage AND the context is non-empty.
    ChromaDB is needed for: exploratory, comparative, survey, or when
    wiki context is sparse.
    """
    query_type   = state.get("query_type", "factual")
    wiki_context = state.get("wiki_context", "")

    # Heuristic: wiki is sufficient if it has substantial content AND
    # the query is simple factual
    wiki_sufficient = (
        query_type == "factual"
        and len(wiki_context) > 400
    )
    # Store decision in state as a flag node_synthesize_answer can read
    return {"cache_hit": state.get("cache_hit", False),
            "_wiki_sufficient": wiki_sufficient}  # type: ignore[misc]


def node_hybrid_search(state: QueryState) -> dict:
    """
    Run hybrid BM25 + semantic search over ChromaDB.
    Uses HyDE for exploratory queries (generates hypothetical doc first).
    """
    sub_queries  = state.get("sub_queries", [state.get("query", "")])
    year_filter  = state.get("year_filter")
    source_filter = state.get("source_filter")
    query_type   = state.get("query_type", "factual")

    all_chunks: list[dict] = []

    for sq in sub_queries[:3]:  # limit to 3 sub-queries to avoid too many calls
        try:
            chunks = hybrid_search(
                query=sq,
                top_k=Config.RETRIEVAL_K,
                year_filter=year_filter,
                source_filter=source_filter,
                use_hyde=(Config.USE_HYDE and query_type == "exploratory"),
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.warning("hybrid_search failed for sub-query '%s': %s", sq[:40], exc)

    # Deduplicate by chunk_id
    seen: set[str] = set()
    unique_chunks: list[dict] = []
    for c in all_chunks:
        cid = c.get("chunk_id", c.get("text", "")[:50])
        if cid not in seen:
            seen.add(cid)
            unique_chunks.append(c)

    return {"retrieved_chunks": unique_chunks[:Config.RETRIEVAL_K]}


def node_rerank(state: QueryState) -> dict:
    """Cross-encoder rerank retrieved chunks."""
    query  = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {"reranked_chunks": []}

    try:
        reranked = rerank_chunks(query, chunks, top_k=Config.RERANK_TOP_K)
        return {"reranked_chunks": reranked}
    except Exception as exc:
        logger.warning("rerank failed: %s", exc)
        # Fall back to top-K unreranked
        return {"reranked_chunks": chunks[:Config.RERANK_TOP_K]}


def node_synthesize_answer(state: QueryState) -> dict:
    """Generate a grounded answer using wiki context + retrieved chunks."""
    query        = state.get("query", "")
    wiki_context = state.get("wiki_context", "")
    chunks       = state.get("reranked_chunks") or state.get("retrieved_chunks") or []

    try:
        answer = synthesize_answer(query, wiki_context, chunks)
    except Exception as exc:
        logger.error("synthesize_answer failed: %s", exc)
        answer = "Unable to generate an answer at this time."

    # Build sources list from chunks + wiki pages
    sources: list[dict] = []
    for c in chunks:
        sources.append({
            "paper_id": c.get("paper_id", ""),
            "title":    c.get("title", ""),
            "year":     c.get("year", 0),
            "source":   c.get("source", ""),
        })
    # Add wiki pages as sources
    for page_path in state.get("wiki_pages_read", []):
        sources.append({"page_path": page_path, "type": "wiki"})

    return {
        "answer":  answer,
        "sources": sources,
    }


def node_self_critique(state: QueryState) -> dict:
    """Evaluate whether the answer is grounded in the provided sources."""
    query   = state.get("query", "")
    answer  = state.get("answer", "")
    sources = state.get("sources", [])

    if not answer:
        return {"is_grounded": False, "confidence": 0.0}

    try:
        result = self_critique(query, answer, sources)
        return {
            "is_grounded": result.get("is_grounded", True),
            "confidence":  result.get("confidence", 0.5),
        }
    except Exception as exc:
        logger.warning("self_critique failed: %s", exc)
        return {"is_grounded": True, "confidence": 0.5}


def node_refine_answer(state: QueryState) -> dict:
    """
    Retry synthesis with a stronger grounding instruction.
    Called once when self_critique returns is_grounded=False.
    """
    query        = state.get("query", "")
    wiki_context = state.get("wiki_context", "")
    chunks       = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    issues       = state.get("sources", [])  # re-use sources field for issue hints

    # Add explicit grounding instruction to the query
    grounded_query = (
        f"{query}\n\n"
        "IMPORTANT: Base your answer ONLY on the provided sources. "
        "If you cannot find sufficient evidence, explicitly state so."
    )

    try:
        answer = synthesize_answer(grounded_query, wiki_context, chunks)
    except Exception as exc:
        logger.error("refine_answer failed: %s", exc)
        answer = state.get("answer", "Unable to generate a grounded answer.")

    return {
        "answer":      answer,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def node_decide_file_back(state: QueryState) -> dict:
    """Decide whether to write this answer to the wiki."""
    if not Config.WIKI_WRITE_BACK:
        return {"should_file_back": False}

    query  = state.get("query", "")
    answer = state.get("answer", "")

    try:
        worth = is_worth_filing(query, answer)
    except Exception as exc:
        logger.warning("is_worth_filing failed: %s", exc)
        worth = False

    return {"should_file_back": worth}


def node_write_synthesis_page(state: QueryState) -> dict:
    """Write a synthesis page to wiki/synthesis/query-answers/."""
    query      = state.get("query", "")
    answer     = state.get("answer", "")
    sources    = state.get("sources", [])

    today = date.today().isoformat()
    slug  = _slugify(query)[:50]
    page_path = f"synthesis/query-answers/{today}-{slug}.md"

    wiki_sources  = [s["page_path"] for s in sources if s.get("type") == "wiki"]
    chunk_sources = [s["paper_id"]  for s in sources if s.get("paper_id")]

    try:
        content = make_synthesis_page(
            query=query,
            answer=answer,
            sources_wiki=wiki_sources,
            sources_chromadb=chunk_sources,
        )
        write_wiki_page(page_path, content, reason=f"query-filed: {query[:60]}")
        update_wiki_index(page_path, query[:80], "synthesis")
        append_wiki_log(
            operation="query-filed",
            title=query[:60],
            details=(
                f"Confidence: {'high' if state.get('confidence', 0) > 0.7 else 'medium'} | "
                f"Wiki pages read: {len(wiki_sources)} | "
                f"Chunks used: {len(chunk_sources)}"
            ),
        )
        return {"filed_page_path": page_path}
    except Exception as exc:
        logger.warning("write_synthesis_page failed: %s", exc)
        return {"filed_page_path": None}


def node_save_cache(state: QueryState) -> dict:
    """Persist the query-answer pair to the SQLite cache."""
    if not Config.USE_PERSISTENT_CACHE:
        return {}

    query      = state.get("query", "")
    answer     = state.get("answer", "")
    sources    = state.get("sources", [])
    confidence = state.get("confidence", 0.5)

    if not query or not answer:
        return {}

    try:
        cache = get_query_cache()
        cache.put(query, answer, sources, confidence)
    except Exception as exc:
        logger.warning("cache.put failed: %s", exc)

    return {}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_cache(state: QueryState) -> str:
    """Cache hit → END, miss → classify_query."""
    if state.get("cache_hit"):
        return "end"
    return "classify_query"


def _route_after_decide_retrieval(state: QueryState) -> str:
    """Wiki sufficient → synthesize directly, else → hybrid_search."""
    if state.get("_wiki_sufficient"):
        return "synthesize_answer"
    return "hybrid_search"


def _route_after_self_critique(state: QueryState) -> str:
    """
    Grounded → decide_file_back
    Not grounded + retry_count < 1 → refine_answer
    Not grounded + retry_count >= 1 → decide_file_back (proceed with caveat)
    """
    is_grounded = state.get("is_grounded", True)
    retry_count = state.get("retry_count", 0)

    if is_grounded or retry_count >= 1:
        return "decide_file_back"
    return "refine_answer"


def _route_after_decide_file_back(state: QueryState) -> str:
    """should_file_back → write_synthesis_page, else → save_cache."""
    if state.get("should_file_back"):
        return "write_synthesis_page"
    return "save_cache"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_query_graph() -> Any:
    """
    Build and compile the Query Agent LangGraph StateGraph.

    Graph topology:
      check_cache ──(hit)──────────────────────────────→ END
           │ (miss)
      classify_query
           │
      read_wiki_index
           │
      read_wiki_pages
           │
      decide_retrieval ──(wiki ok)──→ synthesize_answer
           │ (need chunks)                    ↑
      hybrid_search                           │
           │                                 │
         rerank ──────────────────────────────┘
           │
      synthesize_answer
           │
      self_critique ──(grounded or retried)──→ decide_file_back
           │ (not grounded, retry=0)                  │
      refine_answer ───────────────────────────────────┘
           │                      │
      decide_file_back            │ (should file)
           │ (no file)            ↓
      save_cache        write_synthesis_page
           │                      │
          END               save_cache → END
    """
    graph = StateGraph(QueryState)

    # Register nodes
    graph.add_node("check_cache",          node_check_cache)
    graph.add_node("classify_query",       node_classify_query)
    graph.add_node("read_wiki_index",      node_read_wiki_index)
    graph.add_node("read_wiki_pages",      node_read_wiki_pages)
    graph.add_node("decide_retrieval",     node_decide_retrieval)
    graph.add_node("hybrid_search",        node_hybrid_search)
    graph.add_node("rerank",               node_rerank)
    graph.add_node("synthesize_answer",    node_synthesize_answer)
    graph.add_node("self_critique",        node_self_critique)
    graph.add_node("refine_answer",        node_refine_answer)
    graph.add_node("decide_file_back",     node_decide_file_back)
    graph.add_node("write_synthesis_page", node_write_synthesis_page)
    graph.add_node("save_cache",           node_save_cache)

    # Entry point
    graph.set_entry_point("check_cache")

    # Cache hit → END, miss → pipeline
    graph.add_conditional_edges(
        "check_cache",
        _route_after_cache,
        {"end": END, "classify_query": "classify_query"},
    )

    # Linear pipeline: classify → index → pages → decide
    graph.add_edge("classify_query",   "read_wiki_index")
    graph.add_edge("read_wiki_index",  "read_wiki_pages")
    graph.add_edge("read_wiki_pages",  "decide_retrieval")

    # Branch: wiki sufficient → synthesize, else → search
    graph.add_conditional_edges(
        "decide_retrieval",
        _route_after_decide_retrieval,
        {"synthesize_answer": "synthesize_answer", "hybrid_search": "hybrid_search"},
    )

    # Retrieval path
    graph.add_edge("hybrid_search",     "rerank")
    graph.add_edge("rerank",            "synthesize_answer")

    # Synthesis → critique → optional refinement
    graph.add_edge("synthesize_answer", "self_critique")

    graph.add_conditional_edges(
        "self_critique",
        _route_after_self_critique,
        {"decide_file_back": "decide_file_back", "refine_answer": "refine_answer"},
    )

    # Refine loops back into synthesis chain
    graph.add_edge("refine_answer",     "synthesize_answer")

    # File-back branch
    graph.add_conditional_edges(
        "decide_file_back",
        _route_after_decide_file_back,
        {"write_synthesis_page": "write_synthesis_page", "save_cache": "save_cache"},
    )

    graph.add_edge("write_synthesis_page", "save_cache")
    graph.add_edge("save_cache",           END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_query(
    query: str,
    year_filter: int | None = None,
    source_filter: str | None = None,
) -> QueryState:
    """
    Run the full query pipeline for a plain-text question.

    Returns the final QueryState containing:
      answer:           the synthesized answer
      sources:          list of source dicts (chunks + wiki pages)
      confidence:       0.0–1.0 grounding confidence
      is_grounded:      bool
      cache_hit:        bool
      filed_page_path:  path to wiki synthesis page if filed, else None
    """
    app = build_query_graph()
    initial: dict = {"query": query}
    if year_filter is not None:
        initial["year_filter"] = year_filter
    if source_filter is not None:
        initial["source_filter"] = source_filter

    final_state = app.invoke(initial)
    logger.info(
        "Query complete: grounded=%s confidence=%.2f cache_hit=%s filed=%s",
        final_state.get("is_grounded"),
        final_state.get("confidence", 0.0),
        final_state.get("cache_hit"),
        final_state.get("filed_page_path"),
    )
    return final_state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60]
