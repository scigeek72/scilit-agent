"""
agents/frontier_agent.py — LangGraph Frontier Agent (Phase 11).

A reactive agent (never scheduled) that synthesises across the entire wiki
to surface unexplored territory: methodological gaps (technique X not applied
to domain Y) and conceptual gaps (question Z raised repeatedly but never
addressed). Operates on the wiki alone — the KG layer is optional.

Trigger examples:
  "What's unexplored in CRISPR delivery mechanisms?"
  "What methodological gaps exist in this corpus?"
  "What open questions has nobody worked on?"

Nodes:
  classify_focus            → methodological | conceptual | both
  read_wiki_index           → find all relevant wiki pages
  aggregate_open_questions  → cluster open questions from papers/ pages
  find_method_domain_gaps   → cross-reference methods/ and concepts/
  find_temporal_dropouts    → scan log.md + paper dates
  find_contradiction_clusters → scan debates/ pages
  find_cross_domain         → scan concepts/ cross-domain notes
  query_kg_gaps             → OPTIONAL: only if Config.USE_KG=True
  read_wiki_context         → read relevant concept/method/debate pages
  synthesize_report         → LLM generates structured gap report
  file_report               → wiki/synthesis/frontier-{date}.md
  append_log                → wiki/log.md

Conditional edges:
  classify_focus → sets query_focus; all gap-finding nodes check it and
  skip irrelevant work based on focus (runs all for "both").

  After gap nodes:
    USE_KG=True  → query_kg_gaps → read_wiki_context
    USE_KG=False → read_wiki_context

Usage:
    from agents.frontier_agent import run_frontier
    result = run_frontier("What methodological gaps exist in this corpus?")
    print(result["filed_page_path"])
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any

from langgraph.graph import END, StateGraph

from agents.state import FrontierState
from config import Config
from tools.wiki_tools import (
    append_wiki_log,
    list_wiki_pages,
    read_wiki_index,
    read_wiki_page,
    update_wiki_index,
    write_wiki_page,
)

logger = logging.getLogger(__name__)

# Regex helpers
_YEAR_RE          = re.compile(r"\b(20\d{2}|19\d{2})\b")
_WIKI_LINK_RE     = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]")
_FRONTMATTER_RE   = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_FRONTMATTER_YEAR = re.compile(r"^year:\s*(\d{4})", re.MULTILINE)
_OPEN_Q_SECTION   = re.compile(
    r"## Open Questions Raised\s*\n(.*?)(?=^##|\Z)", re.DOTALL | re.MULTILINE
)
_CROSS_DOMAIN_SECTION = re.compile(
    r"## Cross-domain Notes\s*\n(.*?)(?=^##|\Z)", re.DOTALL | re.MULTILINE
)
_DEBATE_STATUS_RE = re.compile(r"^status:\s*(open|resolved|superseded)", re.MULTILINE)

# LLM call helper
def _call_llm(prompt: str, llm, max_tokens: int = 800) -> str | None:
    if llm is None:
        try:
            from llm_provider import get_llm
            llm = get_llm(temperature=0.2)
        except Exception as exc:
            logger.warning("Could not load LLM: %s", exc)
            return None
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke(
            [HumanMessage(content=prompt)],
            config={"max_tokens": max_tokens},
        )
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Node: classify_focus
# ---------------------------------------------------------------------------

def node_classify_focus(state: FrontierState) -> dict:
    """
    Classify the user's frontier query as methodological, conceptual, or both.

    Heuristics (LLM-free):
      - "method", "technique", "approach", "applied", "benchmark"    → methodological
      - "open question", "challenge", "unexplored", "nobody", "gap"  → conceptual
      - Anything else, or both signals present                        → both
    """
    query = (state.get("query") or "").lower()

    method_signals    = {"method", "technique", "approach", "benchmark",
                         "applied", "apply", "algorithm", "model", "architecture"}
    conceptual_signals = {"open question", "challenge", "unexplored", "nobody",
                          "conceptual", "theoretical", "hypothesis", "gap",
                          "unanswered", "unresolved", "what nobody"}

    has_method      = any(s in query for s in method_signals)
    has_conceptual  = any(s in query for s in conceptual_signals)

    if has_method and not has_conceptual:
        focus = "methodological"
    elif has_conceptual and not has_method:
        focus = "conceptual"
    else:
        focus = "both"

    logger.info("Frontier: query_focus = %s", focus)
    return {"query_focus": focus}


# ---------------------------------------------------------------------------
# Node: read_wiki_index
# ---------------------------------------------------------------------------

def node_read_wiki_index(state: FrontierState) -> dict:
    """Read wiki/index.md to get an overview of available pages."""
    index_content = read_wiki_index()
    # Store in wiki_context dict for later use
    ctx = state.get("wiki_context") or {}
    ctx["index"] = index_content
    return {"wiki_context": ctx}


# ---------------------------------------------------------------------------
# Node: aggregate_open_questions
# ---------------------------------------------------------------------------

def node_aggregate_open_questions(state: FrontierState) -> dict:
    """
    Read all papers/ pages and cluster open questions from each.

    Groups similar questions by keyword overlap. Returns ranked list with
    how many papers raise each question cluster.
    """
    focus = state.get("query_focus", "both")
    if focus == "methodological":
        return {"open_questions": []}

    raw_questions: list[dict] = []  # {question, paper_id, year}

    for page_path in list_wiki_pages("papers"):
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            year = 0
            yr_m = _FRONTMATTER_YEAR.search(content)
            if yr_m:
                year = int(yr_m.group(1))

            oq_m = _OPEN_Q_SECTION.search(content)
            if not oq_m:
                continue
            section = oq_m.group(1)
            for line in section.splitlines():
                line = line.strip().lstrip("-*").strip()
                if len(line) > 15:
                    raw_questions.append({
                        "question": line,
                        "paper_path": page_path,
                        "year": year,
                    })
        except Exception as exc:
            logger.debug("aggregate_open_questions: error reading %s: %s", page_path, exc)

    if not raw_questions:
        return {"open_questions": []}

    # Cluster by keyword overlap (lightweight, no LLM needed)
    clusters: dict[str, list[dict]] = defaultdict(list)
    for item in raw_questions:
        words = set(re.findall(r"[a-z]{4,}", item["question"].lower()))
        stop  = {"that", "this", "with", "from", "have", "been", "what", "when",
                 "will", "does", "such", "more", "they", "their", "which"}
        words -= stop
        key = " ".join(sorted(words)[:4])  # cheap fingerprint
        clusters[key].append(item)

    # Merge tiny clusters (single occurrences) under an "other" bucket
    clustered: list[dict] = []
    for key, items in sorted(clusters.items(), key=lambda x: -len(x[1])):
        if len(items) == 0:
            continue
        representative = max(items, key=lambda x: len(x["question"]))
        clustered.append({
            "question":   representative["question"],
            "count":      len(items),
            "papers":     [i["paper_path"] for i in items],
            "years":      sorted({i["year"] for i in items if i["year"]}),
            "confidence": "high" if len(items) >= 3 else ("medium" if len(items) == 2 else "low"),
        })

    # Cap at 15 to keep the report readable
    clustered.sort(key=lambda x: -x["count"])
    logger.info("Frontier: found %d open question clusters", len(clustered))
    return {"open_questions": clustered[:15]}


# ---------------------------------------------------------------------------
# Node: find_method_domain_gaps
# ---------------------------------------------------------------------------

def node_find_method_domain_gaps(state: FrontierState) -> dict:
    """
    Cross-reference methods/ and concepts/ pages to find methods not yet
    applied to domains that have papers in the corpus.

    Logic:
      1. Parse methods/ pages → extract domains from `domains:` frontmatter
      2. Parse concepts/ pages → extract what domains they appear in
      3. Compare: if a method appears in domain A but no concept page for
         domain B exists, and papers from domain B are indexed, flag it.
    """
    focus = state.get("query_focus", "both")
    if focus == "conceptual":
        return {"method_domain_gaps": []}

    # Collect methods and their documented domains
    method_domains: dict[str, list[str]] = {}
    for page_path in list_wiki_pages("methods"):
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            fm_m = _FRONTMATTER_RE.search(content)
            if fm_m:
                fm = fm_m.group(1)
                d_m = re.search(r"^domains:\s*\[(.+?)\]", fm, re.MULTILINE)
                if d_m:
                    domains = [d.strip().strip('"').strip("'") for d in d_m.group(1).split(",")]
                    method_name = page_path.replace("methods/", "").replace(".md", "").replace("-", " ")
                    method_domains[method_name] = domains
        except Exception:
            pass

    # Collect source domains present in corpus from papers/ frontmatter
    corpus_sources: set[str] = set()
    for page_path in list_wiki_pages("papers"):
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            fm_m = _FRONTMATTER_RE.search(content)
            if fm_m:
                src_m = re.search(r"^source:\s*(\S+)", fm_m.group(1), re.MULTILINE)
                if src_m:
                    corpus_sources.add(src_m.group(1).strip())
                tag_m = re.search(r"^tags:\s*\[(.+?)\]", fm_m.group(1), re.MULTILINE)
                if tag_m:
                    for t in tag_m.group(1).split(","):
                        corpus_sources.add(t.strip().strip('"').strip("'"))
        except Exception:
            pass

    # Broad domain mapping from sources to domain labels
    source_to_domain = {
        "pubmed": "medicine", "biorxiv": "biology", "medrxiv": "medicine",
        "arxiv": "cs", "semantic_scholar": "cross-domain", "local": "local",
        "biology": "biology", "medicine": "medicine", "cs": "cs",
    }
    corpus_domains = {source_to_domain.get(s, s) for s in corpus_sources} - {"cross-domain", "local"}

    gaps: list[dict] = []
    for method, documented_domains in method_domains.items():
        for corpus_domain in corpus_domains:
            if corpus_domain not in documented_domains:
                gaps.append({
                    "method":            method,
                    "documented_domains": documented_domains,
                    "missing_domain":    corpus_domain,
                    "confidence":        "medium",
                })

    logger.info("Frontier: found %d method-domain gaps", len(gaps))
    return {"method_domain_gaps": gaps[:20]}


# ---------------------------------------------------------------------------
# Node: find_temporal_dropouts
# ---------------------------------------------------------------------------

def node_find_temporal_dropouts(state: FrontierState) -> dict:
    """
    Find research directions that were active in the corpus then went quiet.

    Method:
      1. Gather paper years from papers/ frontmatter
      2. Cluster papers by their concept tags
      3. If a concept cluster has no paper newer than 2 years ago → temporal dropout
    """
    focus = state.get("query_focus", "both")
    if focus == "methodological":
        return {"temporal_dropouts": []}

    current_year = date.today().year
    quiet_threshold = current_year - 2

    # concept_slug → list of years
    concept_years: dict[str, list[int]] = defaultdict(list)

    for page_path in list_wiki_pages("papers"):
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            yr_m = _FRONTMATTER_YEAR.search(content)
            year = int(yr_m.group(1)) if yr_m else 0

            # Extract concept links from Key Concepts section
            kc_m = re.search(r"## Key Concepts\s*\n(.*?)(?=^##|\Z)", content, re.DOTALL | re.MULTILINE)
            if kc_m:
                for link_m in _WIKI_LINK_RE.finditer(kc_m.group(1)):
                    target = link_m.group(1).strip()
                    if target.startswith("concepts/"):
                        slug = target.replace("concepts/", "").replace(".md", "")
                        if year:
                            concept_years[slug].append(year)
        except Exception:
            pass

    dropouts: list[dict] = []
    for concept, years in concept_years.items():
        if not years:
            continue
        max_year = max(years)
        if max_year <= quiet_threshold:
            dropouts.append({
                "concept":     concept,
                "last_year":   max_year,
                "paper_count": len(years),
                "years":       sorted(set(years)),
                "confidence":  "high" if len(years) >= 3 else "medium",
            })

    dropouts.sort(key=lambda x: -x["paper_count"])
    logger.info("Frontier: found %d temporal dropouts", len(dropouts))
    return {"temporal_dropouts": dropouts[:10]}


# ---------------------------------------------------------------------------
# Node: find_contradiction_clusters
# ---------------------------------------------------------------------------

def node_find_contradiction_clusters(state: FrontierState) -> dict:
    """
    Cluster open debates/ pages by shared concept links or title keywords.

    A cluster of related open debates points at a deeper unresolved
    methodological or conceptual question.
    """
    debate_pages = list_wiki_pages("debates")
    open_debates: list[dict] = []

    for page_path in debate_pages:
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            status_m = _DEBATE_STATUS_RE.search(content)
            if status_m and status_m.group(1) != "open":
                continue  # skip resolved

            # Extract title from page
            title_m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = title_m.group(1).strip() if title_m else page_path

            # Find concept links referenced in this debate
            concepts = [
                m.group(1).replace("concepts/", "").replace(".md", "")
                for m in _WIKI_LINK_RE.finditer(content)
                if m.group(1).startswith("concepts/")
            ]

            open_debates.append({
                "page_path": page_path,
                "title":     title,
                "concepts":  list(set(concepts)),
            })
        except Exception:
            pass

    # Cluster debates by shared concepts
    clusters: list[dict] = []
    used: set[str] = set()

    for i, d in enumerate(open_debates):
        if d["page_path"] in used:
            continue
        cluster = [d]
        used.add(d["page_path"])
        for j, other in enumerate(open_debates):
            if other["page_path"] in used:
                continue
            shared = set(d["concepts"]) & set(other["concepts"])
            if shared:
                cluster.append(other)
                used.add(other["page_path"])

        if len(cluster) >= 1:
            all_concepts = [c for item in cluster for c in item["concepts"]]
            clusters.append({
                "debates":           [item["page_path"] for item in cluster],
                "titles":            [item["title"] for item in cluster],
                "shared_concepts":   list(set(all_concepts)),
                "size":              len(cluster),
                "confidence":        "high" if len(cluster) >= 3 else ("medium" if len(cluster) == 2 else "low"),
            })

    clusters.sort(key=lambda x: -x["size"])
    logger.info("Frontier: found %d contradiction clusters from %d open debates",
                len(clusters), len(open_debates))
    return {"contradiction_clusters": clusters[:10]}


# ---------------------------------------------------------------------------
# Node: find_cross_domain
# ---------------------------------------------------------------------------

def node_find_cross_domain(state: FrontierState) -> dict:
    """
    Scan concepts/ pages for Cross-domain Notes sections.

    A non-empty Cross-domain Notes section in a concept page means the
    concept bridges multiple fields — flag it as a potential transfer opportunity.
    """
    focus = state.get("query_focus", "both")
    if focus == "conceptual":
        return {"cross_domain_opportunities": []}

    opportunities: list[dict] = []

    for page_path in list_wiki_pages("concepts"):
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            cd_m = _CROSS_DOMAIN_SECTION.search(content)
            if not cd_m:
                continue
            notes = cd_m.group(1).strip()
            if len(notes) < 30:
                continue  # placeholder only

            concept = page_path.replace("concepts/", "").replace(".md", "").replace("-", " ")

            # Extract domains mentioned in the notes
            mentioned_domains: list[str] = []
            for domain in ("cs", "biology", "medicine", "physics", "chemistry",
                           "clinical", "genomics", "nlp", "computer vision"):
                if domain in notes.lower():
                    mentioned_domains.append(domain)

            if len(mentioned_domains) >= 2:
                opportunities.append({
                    "concept":           concept,
                    "page_path":         page_path,
                    "domains":           mentioned_domains,
                    "notes_excerpt":     notes[:200],
                    "confidence":        "high" if len(mentioned_domains) >= 3 else "medium",
                })
        except Exception:
            pass

    logger.info("Frontier: found %d cross-domain opportunities", len(opportunities))
    return {"cross_domain_opportunities": opportunities[:10]}


# ---------------------------------------------------------------------------
# Node: query_kg_gaps (optional — only if USE_KG=True)
# ---------------------------------------------------------------------------

def node_query_kg_gaps(state: FrontierState) -> dict:
    """
    Query the Knowledge Graph layer for NOT_YET_APPLIED_TO edges.
    Only runs when Config.USE_KG=True (Phase 12 feature).
    """
    if not Config.USE_KG:
        return {"kg_gap_edges": []}

    try:
        from tools.kg_tools import find_gap_edges  # type: ignore[import]
        edges = find_gap_edges()
        logger.info("Frontier: KG returned %d gap edges", len(edges))
        return {"kg_gap_edges": edges[:20]}
    except ImportError:
        logger.debug("kg_tools not available — skipping KG gap query")
        return {"kg_gap_edges": []}


# ---------------------------------------------------------------------------
# Node: read_wiki_context
# ---------------------------------------------------------------------------

def node_read_wiki_context(state: FrontierState) -> dict:
    """
    Read the most relevant concept, method, and debate pages to provide
    narrative context for the synthesis step.

    Selects pages referenced in the gap findings (up to 8 pages total).
    """
    ctx = state.get("wiki_context") or {}

    pages_to_read: list[str] = []

    # Pages from contradiction clusters
    for cluster in (state.get("contradiction_clusters") or [])[:3]:
        pages_to_read.extend(cluster.get("debates", [])[:2])

    # Pages from cross-domain opportunities
    for opp in (state.get("cross_domain_opportunities") or [])[:3]:
        pages_to_read.append(opp.get("page_path", ""))

    # Pages from method-domain gaps
    for gap in (state.get("method_domain_gaps") or [])[:3]:
        method_slug = gap.get("method", "").lower().replace(" ", "-")
        pages_to_read.append(f"methods/{method_slug}.md")

    # Deduplicate and cap
    seen: set[str] = set()
    unique_pages: list[str] = []
    for p in pages_to_read:
        if p and p not in seen:
            seen.add(p)
            unique_pages.append(p)

    wiki_pages_content: dict[str, str] = {}
    for page_path in unique_pages[:8]:
        try:
            content = read_wiki_page(page_path)
            if content:
                wiki_pages_content[page_path] = content[:500]  # excerpt only
        except Exception:
            pass

    ctx["pages"] = wiki_pages_content
    return {"wiki_context": ctx}


# ---------------------------------------------------------------------------
# Node: synthesize_report
# ---------------------------------------------------------------------------

def node_synthesize_report(state: FrontierState, llm=None) -> dict:
    """
    Generate a structured frontier gap report using the LLM.

    If the LLM is unavailable, falls back to a structured plain-text report
    assembled directly from the gap data.
    """
    query       = state.get("query", "")
    focus       = state.get("query_focus", "both")
    oqs         = state.get("open_questions", [])
    meth_gaps   = state.get("method_domain_gaps", [])
    temporal    = state.get("temporal_dropouts", [])
    clusters    = state.get("contradiction_clusters", [])
    cross       = state.get("cross_domain_opportunities", [])
    kg_edges    = state.get("kg_gap_edges", [])

    # Build a compact data digest for the LLM
    digest_parts: list[str] = [f"User question: {query}\nQuery focus: {focus}\n"]

    if oqs:
        digest_parts.append("## Open Questions (from papers):")
        for q in oqs[:5]:
            digest_parts.append(f'- [{q["count"]} papers] {q["question"]} (confidence: {q["confidence"]})')

    if meth_gaps:
        digest_parts.append("\n## Method-Domain Gaps:")
        for g in meth_gaps[:5]:
            digest_parts.append(
                f'- Method "{g["method"]}" applied to {g["documented_domains"]} '
                f'but NOT to {g["missing_domain"]} (confidence: {g["confidence"]})'
            )

    if temporal:
        digest_parts.append("\n## Temporal Dropouts (went quiet):")
        for t in temporal[:5]:
            digest_parts.append(
                f'- Concept "{t["concept"]}": last paper {t["last_year"]}, '
                f'{t["paper_count"]} papers total (confidence: {t["confidence"]})'
            )

    if clusters:
        digest_parts.append("\n## Contradiction Clusters:")
        for c in clusters[:4]:
            digest_parts.append(
                f'- {c["size"]} open debates share concepts: {", ".join(c["shared_concepts"][:3])} '
                f'(confidence: {c["confidence"]})'
            )

    if cross:
        digest_parts.append("\n## Cross-Domain Opportunities:")
        for o in cross[:4]:
            digest_parts.append(
                f'- Concept "{o["concept"]}" bridges {", ".join(o["domains"])} '
                f'(confidence: {o["confidence"]})'
            )

    if kg_edges:
        digest_parts.append("\n## KG Gap Edges:")
        for e in kg_edges[:5]:
            digest_parts.append(f'- {e}')

    digest = "\n".join(digest_parts)

    prompt = f"""You are a scientific research analyst. Using the gap analysis data below,
write a Frontier Gap Report for a researcher studying: {Config.TOPIC_NAME}

{digest}

Write a structured markdown report with these sections:
1. ## Methodological Gaps — methods not applied to certain domains
2. ## Conceptual Gaps — open questions raised but unaddressed
3. ## Cross-Domain Opportunities — concepts bridging multiple fields
4. ## Temporal Dropouts — research directions that went quiet
5. ## Suggested Next Searches — 3-5 specific search queries to fill gaps

For each gap, include:
- A descriptive heading
- Which papers/concepts support this gap (reference [[wiki-links]])
- A confidence level: high | medium | low
- A brief narrative (2-3 sentences) explaining why the gap matters

Important caveats to include at the end:
- All gaps are relative to THIS corpus, not all of science
- Some gaps may be unexplored for good reasons (technically infeasible, niche)
- Human judgment required to assess feasibility and significance
- Never assert a gap SHOULD be filled, only that it EXISTS in this corpus"""

    today = date.today().isoformat()

    response = _call_llm(prompt, llm, max_tokens=1200)

    if response:
        report = response.strip()
    else:
        # Fallback: build report directly from structured data without LLM
        report = _build_fallback_report(query, focus, oqs, meth_gaps, temporal, clusters, cross)

    return {"report": report}


def _build_fallback_report(
    query: str,
    focus: str,
    oqs: list[dict],
    meth_gaps: list[dict],
    temporal: list[dict],
    clusters: list[dict],
    cross: list[dict],
) -> str:
    """Assemble a structured report from gap data without calling the LLM."""

    def _items(items: list[dict], fmt) -> str:
        if not items:
            return "_None detected._\n"
        return "\n".join(fmt(i) for i in items) + "\n"

    parts = [f"## Methodological Gaps\n"]
    parts.append(_items(meth_gaps, lambda g:
        f'### "{g["method"]}" not applied to {g["missing_domain"]}\n'
        f'- Documented in: {", ".join(g["documented_domains"])}\n'
        f'- Confidence: {g["confidence"]}\n'))

    parts.append(f"\n## Conceptual Gaps\n")
    parts.append(_items(oqs, lambda q:
        f'### "{q["question"][:80]}"\n'
        f'- Raised in {q["count"]} paper(s) · Confidence: {q["confidence"]}\n'))

    parts.append(f"\n## Cross-Domain Opportunities\n")
    parts.append(_items(cross, lambda o:
        f'### "{o["concept"]}" bridges {", ".join(o["domains"])}\n'
        f'- {o["notes_excerpt"][:120]}\n'
        f'- Confidence: {o["confidence"]}\n'))

    parts.append(f"\n## Temporal Dropouts\n")
    parts.append(_items(temporal, lambda t:
        f'### "{t["concept"]}" — last paper {t["last_year"]}\n'
        f'- {t["paper_count"]} paper(s) in corpus · Confidence: {t["confidence"]}\n'))

    parts.append(f"\n## Suggested Next Searches\n")
    suggestions: list[str] = []
    for g in meth_gaps[:2]:
        suggestions.append(f'"{g["method"]} {g["missing_domain"]}"')
    for q in oqs[:2]:
        words = q["question"].split()[:6]
        suggestions.append(f'"{"  ".join(words)}"')
    for t in temporal[:1]:
        suggestions.append(f'"recent {t["concept"]}"')
    if suggestions:
        parts.append("\n".join(f"- {s}" for s in suggestions[:5]))
    else:
        parts.append("_No specific suggestions generated._")

    parts.append(f"\n\n---\n*Gaps are relative to this corpus only. Human judgment required.*")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Node: file_report
# ---------------------------------------------------------------------------

def node_file_report(state: FrontierState) -> dict:
    """Write the frontier report to wiki/synthesis/frontier-{date}.md."""
    today      = date.today().isoformat()
    query      = state.get("query", "")
    focus      = state.get("query_focus", "both")
    report     = state.get("report", "")
    kg_used    = Config.USE_KG and bool(state.get("kg_gap_edges"))

    n_papers = len(list_wiki_pages("papers"))
    slug     = re.sub(r"[^a-z0-9]+", "-", query.lower())[:40].strip("-")
    page_path = f"synthesis/frontier-{today}-{slug}.md"

    full_page = f"""---
query: "{query}"
date: {today}
focus: {focus}
papers_analyzed: {n_papers}
kg_used: {str(kg_used).lower()}
---

# Research Frontier Report — {Config.TOPIC_NAME}

**Query**: {query}
**Date**: {today} | **Focus**: {focus} | **Papers analysed**: {n_papers}

{report}

---
*Generated by Frontier Agent. Re-run after ingesting new papers for updated gaps.*
"""

    try:
        write_wiki_page(page_path, full_page, reason=f"frontier report {today}")
        update_wiki_index(page_path, f"Frontier report {today}: {query[:60]}", "synthesis")
        logger.info("Frontier report filed: %s", page_path)
        return {"filed_page_path": page_path}
    except Exception as exc:
        logger.error("file_report failed: %s", exc)
        return {"filed_page_path": None}


# ---------------------------------------------------------------------------
# Node: append_log
# ---------------------------------------------------------------------------

def node_append_log(state: FrontierState) -> dict:
    """Append a frontier entry to wiki/log.md."""
    query   = state.get("query", "")
    oqs     = state.get("open_questions", [])
    gaps    = state.get("method_domain_gaps", [])
    cross   = state.get("cross_domain_opportunities", [])
    filed   = state.get("filed_page_path", "")

    details = (
        f"Open question clusters: {len(oqs)} | "
        f"Method-domain gaps: {len(gaps)} | "
        f"Cross-domain opportunities: {len(cross)} | "
        f"Report: {filed}"
    )
    try:
        append_wiki_log(
            operation="frontier",
            title=query[:80],
            details=details,
        )
    except Exception as exc:
        logger.warning("append_log failed: %s", exc)
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_kg(state: FrontierState) -> str:
    """After gap-finding nodes, check whether to query KG or go straight to context."""
    if Config.USE_KG:
        return "query_kg_gaps"
    return "read_wiki_context"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_frontier_graph() -> Any:
    """
    Build and compile the Frontier Agent LangGraph StateGraph.

    Topology:
      classify_focus
           │
      read_wiki_index
           │
      aggregate_open_questions  ──┐
      find_method_domain_gaps   ──┤  (run all, nodes self-filter by focus)
      find_temporal_dropouts    ──┤
      find_contradiction_clusters─┤
      find_cross_domain         ──┘
           │
           ▼ (conditional)
      [query_kg_gaps]  ←── only if USE_KG=True
           │
      read_wiki_context
           │
      synthesize_report
           │
      file_report
           │
      append_log
           │
          END
    """
    graph = StateGraph(FrontierState)

    graph.add_node("classify_focus",              node_classify_focus)
    graph.add_node("read_wiki_index",             node_read_wiki_index)
    graph.add_node("aggregate_open_questions",    node_aggregate_open_questions)
    graph.add_node("find_method_domain_gaps",     node_find_method_domain_gaps)
    graph.add_node("find_temporal_dropouts",      node_find_temporal_dropouts)
    graph.add_node("find_contradiction_clusters", node_find_contradiction_clusters)
    graph.add_node("find_cross_domain",           node_find_cross_domain)
    graph.add_node("query_kg_gaps",               node_query_kg_gaps)
    graph.add_node("read_wiki_context",           node_read_wiki_context)
    graph.add_node("synthesize_report",           node_synthesize_report)
    graph.add_node("file_report",                 node_file_report)
    graph.add_node("append_log",                  node_append_log)

    graph.set_entry_point("classify_focus")

    # Linear backbone through gap-finding nodes
    graph.add_edge("classify_focus",              "read_wiki_index")
    graph.add_edge("read_wiki_index",             "aggregate_open_questions")
    graph.add_edge("aggregate_open_questions",    "find_method_domain_gaps")
    graph.add_edge("find_method_domain_gaps",     "find_temporal_dropouts")
    graph.add_edge("find_temporal_dropouts",      "find_contradiction_clusters")
    graph.add_edge("find_contradiction_clusters", "find_cross_domain")

    # Conditional: KG or straight to context
    graph.add_conditional_edges(
        "find_cross_domain",
        _route_after_kg,
        {
            "query_kg_gaps":   "query_kg_gaps",
            "read_wiki_context": "read_wiki_context",
        },
    )

    graph.add_edge("query_kg_gaps",    "read_wiki_context")
    graph.add_edge("read_wiki_context", "synthesize_report")
    graph.add_edge("synthesize_report", "file_report")
    graph.add_edge("file_report",       "append_log")
    graph.add_edge("append_log",        END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_frontier(query: str) -> FrontierState:
    """
    Run the full Frontier Agent pipeline and return the final FrontierState.

    Args:
        query: Plain-text gap question from the user.

    Returns a dict with:
      query_focus:               'methodological' | 'conceptual' | 'both'
      open_questions:            list of clustered open question dicts
      method_domain_gaps:        list of method × domain gap dicts
      temporal_dropouts:         list of concept dropout dicts
      contradiction_clusters:    list of debate cluster dicts
      cross_domain_opportunities: list of cross-domain opportunity dicts
      kg_gap_edges:              list of KG gap edges (empty if USE_KG=False)
      report:                    the narrative report string
      filed_page_path:           wiki path of the filed report, or None
    """
    app = build_frontier_graph()
    final_state = app.invoke({"query": query})
    logger.info(
        "Frontier complete: focus=%s oqs=%d meth_gaps=%d temporal=%d filed=%s",
        final_state.get("query_focus"),
        len(final_state.get("open_questions", [])),
        len(final_state.get("method_domain_gaps", [])),
        len(final_state.get("temporal_dropouts", [])),
        final_state.get("filed_page_path"),
    )
    return final_state
