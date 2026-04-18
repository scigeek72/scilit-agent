"""
agents/lint_agent.py — LangGraph Lint Agent.

Scans the wiki for health issues and writes a structured lint report.
Triggered by the user or on a schedule (APScheduler).

Nodes:
  scan_all_pages       — build inbound-link graph over wiki/
  find_orphans         — pages with zero inbound links
  find_contradictions  — unresolved debate pages
  find_stale_claims    — claims not reflected in newest papers
  find_missing_pages   — [[wiki-links]] to concept pages that don't exist yet
  find_gaps            — under-represented topics → suggest next searches
  write_lint_report    — wiki/synthesis/lint-{date}.md
  append_log           — wiki/log.md

Usage:
    from agents.lint_agent import build_lint_graph, run_lint
    result = run_lint()
    print(result["report_path"])

Scheduler integration:
    from agents.lint_agent import start_lint_scheduler
    start_lint_scheduler()   # runs every Config.WIKI_LINT_INTERVAL_DAYS days
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

from langgraph.graph import END, StateGraph

from agents.state import LintState
from config import Config
from tools.wiki_tools import (
    append_wiki_log,
    build_link_graph,
    list_wiki_pages,
    read_wiki_page,
    update_wiki_index,
    write_wiki_page,
)

logger = logging.getLogger(__name__)

# Minimum inbound links for a page to be considered "connected"
_ORPHAN_INBOUND_THRESHOLD = 0

# How many years back before a claim is considered potentially stale
_STALE_YEARS_THRESHOLD = 3

# Regex to find [[wiki-links]] in page content
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]")

# Regex to find year mentions in claims (e.g. "(2019)")
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")

# Regex to extract paper year from frontmatter
_FRONTMATTER_YEAR_RE = re.compile(r"^year:\s*(\d{4})", re.MULTILINE)

# Regex to find debate pages with status: open
_DEBATE_STATUS_RE = re.compile(r"^status:\s*(open|resolved|superseded)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def node_scan_all_pages(state: LintState) -> dict:
    """Build an inbound-link graph over the entire wiki directory."""
    try:
        link_graph = build_link_graph()
        logger.info("Lint: scanned %d wiki pages", len(link_graph))
        return {"link_graph": link_graph}
    except Exception as exc:
        logger.error("scan_all_pages failed: %s", exc)
        return {"link_graph": {}}


def node_find_orphans(state: LintState) -> dict:
    """
    Find wiki pages with zero inbound links (no other page links to them).

    Excludes: index.md, log.md, and synthesis/ pages — these are
    structural files not expected to be linked by other pages.
    """
    link_graph = state.get("link_graph", {})

    _exempt_prefixes = ("synthesis/", "index", "log")

    orphans = [
        page_path
        for page_path, inbound in link_graph.items()
        if len(inbound) <= _ORPHAN_INBOUND_THRESHOLD
        and not any(page_path.startswith(p) or page_path in (p + ".md") for p in _exempt_prefixes)
    ]

    logger.info("Lint: found %d orphan pages", len(orphans))
    return {"orphans": sorted(orphans)}


def node_find_contradictions(state: LintState) -> dict:
    """
    Find unresolved debate pages (status: open in frontmatter).

    These are wiki/debates/*.md pages where the debate has not been
    marked 'resolved' or 'superseded'.
    """
    debate_pages = list_wiki_pages("debates")
    unresolved: list[str] = []

    for page_path in debate_pages:
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            status_m = _DEBATE_STATUS_RE.search(content)
            if not status_m or status_m.group(1) == "open":
                unresolved.append(page_path)
        except Exception as exc:
            logger.warning("find_contradictions: error reading %s: %s", page_path, exc)

    logger.info("Lint: found %d unresolved debates", len(unresolved))
    return {"contradictions": sorted(unresolved)}


def node_find_stale_claims(state: LintState) -> dict:
    """
    Find claims in concept/method pages that may be stale.

    A claim is considered potentially stale if:
    - It cites a year older than (current_year - _STALE_YEARS_THRESHOLD)
    - AND a newer paper on the same concept has been ingested since

    Looks at wiki/concepts/*.md and wiki/methods/*.md bullet points
    containing year citations.
    """
    current_year = date.today().year
    cutoff_year  = current_year - _STALE_YEARS_THRESHOLD

    # Find the newest paper year in the wiki
    paper_pages = list_wiki_pages("papers")
    newest_paper_year = 0
    for page_path in paper_pages:
        try:
            content = read_wiki_page(page_path)
            m = _FRONTMATTER_YEAR_RE.search(content)
            if m:
                y = int(m.group(1))
                if y > newest_paper_year:
                    newest_paper_year = y
        except Exception:
            pass

    if newest_paper_year == 0:
        return {"stale_claims": []}

    stale: list[str] = []
    for category in ("concepts", "methods"):
        for page_path in list_wiki_pages(category):  # type: ignore[arg-type]
            try:
                content = read_wiki_page(page_path)
                if not content:
                    continue
                # Find bullet-point claims with old year citations
                for line in content.splitlines():
                    if not line.strip().startswith(("-", "*")):
                        continue
                    year_matches = _YEAR_RE.findall(line)
                    for yr_str in year_matches:
                        yr = int(yr_str)
                        if yr <= cutoff_year and newest_paper_year > yr + 1:
                            stale.append(
                                f"{page_path}: claim from {yr} may be superseded "
                                f"(newest paper: {newest_paper_year}): {line.strip()[:80]}"
                            )
                            break  # one stale flag per line is enough
            except Exception as exc:
                logger.warning("find_stale_claims: error reading %s: %s", page_path, exc)

    logger.info("Lint: found %d potentially stale claims", len(stale))
    return {"stale_claims": stale[:50]}  # cap at 50 to keep report readable


def node_find_missing_pages(state: LintState) -> dict:
    """
    Find [[wiki-links]] to concept pages that are referenced in papers/
    but for which no wiki/concepts/*.md page exists.

    These are the concepts most urgently needing their own page.
    """
    existing_concept_pages = set(list_wiki_pages("concepts"))
    # Normalise: strip "concepts/" prefix and ".md" suffix for matching
    existing_slugs = {
        p.replace("concepts/", "").replace(".md", "")
        for p in existing_concept_pages
    }

    referenced: dict[str, int] = {}  # slug → reference count

    for page_path in list_wiki_pages("papers"):
        try:
            content = read_wiki_page(page_path)
            for m in _WIKI_LINK_RE.finditer(content):
                target = m.group(1).strip()
                # Only care about concepts/ links
                if target.startswith("concepts/"):
                    slug = target.replace("concepts/", "").replace(".md", "")
                    if slug not in existing_slugs:
                        referenced[slug] = referenced.get(slug, 0) + 1
        except Exception as exc:
            logger.warning("find_missing_pages: error reading %s: %s", page_path, exc)

    # Sort by reference count descending — most-referenced missing pages first
    missing = [
        f"concepts/{slug}.md (referenced {count}x)"
        for slug, count in sorted(referenced.items(), key=lambda x: -x[1])
    ]

    logger.info("Lint: found %d missing concept pages", len(missing))
    return {"missing_concept_pages": missing[:20]}


def node_find_gaps(state: LintState) -> dict:
    """
    Suggest next search queries to fill gaps in the corpus.

    Gap signals:
    1. Concepts with only 1 paper in their Key Papers table
    2. Methods with no entries in their Comparison Table
    3. Debate pages that are 'open' with no recent (< 2 years) papers
    4. Concepts referenced in papers/ but missing their own page (from missing_concept_pages)
    """
    gaps: list[str] = []
    current_year = date.today().year

    # Signal 1: thin concept pages (paper_count == 1 or only one row in Key Papers)
    for page_path in list_wiki_pages("concepts"):
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            # Count table rows in Key Papers section
            kp_m = re.search(r"## Key Papers\s*\n(.*?)(?=^##|\Z)", content, re.DOTALL | re.MULTILINE)
            if kp_m:
                rows = [l for l in kp_m.group(1).splitlines() if l.strip().startswith("|") and "---" not in l and "Paper" not in l]
                if len(rows) <= 1:
                    concept = page_path.replace("concepts/", "").replace(".md", "").replace("-", " ")
                    gaps.append(f'Search for more papers on "{concept}" — only {len(rows)} paper(s) indexed')
        except Exception:
            pass

    # Signal 2: method pages with empty comparison table
    for page_path in list_wiki_pages("methods"):
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            ct_m = re.search(r"## Comparison Table\s*\n(.*?)(?=^##|\Z)", content, re.DOTALL | re.MULTILINE)
            if ct_m:
                rows = [l for l in ct_m.group(1).splitlines() if l.strip().startswith("|") and "---" not in l and "Approach" not in l]
                if len(rows) == 0:
                    method = page_path.replace("methods/", "").replace(".md", "").replace("-", " ")
                    gaps.append(f'Search for benchmark comparisons of "{method}"')
        except Exception:
            pass

    # Signal 3: open debates with no recent papers
    for page_path in (state.get("contradictions") or []):
        try:
            content = read_wiki_page(page_path)
            if not content:
                continue
            years_in_page = [int(y) for y in _YEAR_RE.findall(content) if int(y) >= 2000]
            if years_in_page and max(years_in_page) < current_year - 2:
                topic = page_path.replace("debates/", "").replace(".md", "").replace("-", " ")
                gaps.append(f'Search for recent resolution of debate: "{topic}"')
        except Exception:
            pass

    # Signal 4: most-referenced missing concept pages
    for entry in (state.get("missing_concept_pages") or [])[:5]:
        slug = entry.split(".md")[0].replace("concepts/", "").replace("-", " ")
        gaps.append(f'Search to build concept page: "{slug}"')

    logger.info("Lint: found %d gap suggestions", len(gaps))
    return {"gaps": gaps[:20]}


def node_write_lint_report(state: LintState) -> dict:
    """
    Write a structured lint report to wiki/synthesis/lint-{date}.md.

    Report sections:
    - Summary stats
    - Orphaned pages
    - Unresolved debates
    - Potentially stale claims
    - Missing concept pages
    - Suggested next searches
    """
    today        = date.today().isoformat()
    page_path    = f"synthesis/lint-{today}.md"

    orphans       = state.get("orphans", [])
    contradictions = state.get("contradictions", [])
    stale_claims  = state.get("stale_claims", [])
    missing_pages = state.get("missing_concept_pages", [])
    gaps          = state.get("gaps", [])

    def _bullet_list(items: list[str], max_items: int = 20) -> str:
        if not items:
            return "_None found._\n"
        return "\n".join(f"- {item}" for item in items[:max_items]) + "\n"

    report = f"""---
lint_date: {today}
orphans: {len(orphans)}
unresolved_debates: {len(contradictions)}
stale_claims: {len(stale_claims)}
missing_concept_pages: {len(missing_pages)}
gap_suggestions: {len(gaps)}
---

# Wiki Lint Report — {Config.TOPIC_NAME}

**Date**: {today} | **Topic**: {Config.TOPIC_NAME}

## Summary

| Check | Count |
|---|---|
| Orphaned pages | {len(orphans)} |
| Unresolved debates | {len(contradictions)} |
| Potentially stale claims | {len(stale_claims)} |
| Missing concept pages | {len(missing_pages)} |
| Gap suggestions | {len(gaps)} |

## Orphaned Pages
Pages with no inbound links from other wiki pages.
These may need to be cross-referenced or deleted.

{_bullet_list(orphans)}

## Unresolved Debates
Open debate pages that have not been marked resolved or superseded.

{_bullet_list(contradictions)}

## Potentially Stale Claims
Claims in concept/method pages that cite older years while newer papers exist.

{_bullet_list(stale_claims, max_items=20)}

## Missing Concept Pages
Concepts referenced via [[wiki-links]] in papers/ but lacking their own page.

{_bullet_list(missing_pages)}

## Suggested Next Searches
Queries to run next to fill knowledge gaps in this corpus.

{_bullet_list(gaps)}

---
_Generated by Lint Agent. Run `lint` again after ingesting new papers._
"""

    try:
        write_wiki_page(page_path, report, reason=f"lint report {today}")
        update_wiki_index(page_path, f"Lint report {today}", "synthesis")
        logger.info("Lint report written: %s", page_path)
        return {"report_path": page_path}
    except Exception as exc:
        logger.error("write_lint_report failed: %s", exc)
        return {"report_path": None}


def node_append_log(state: LintState) -> dict:
    """Append a lint entry to wiki/log.md."""
    orphans        = state.get("orphans", [])
    contradictions = state.get("contradictions", [])
    missing_pages  = state.get("missing_concept_pages", [])

    details = (
        f"Orphans: {len(orphans)} | "
        f"Contradictions: {len(contradictions)} | "
        f"Missing concept pages: {len(missing_pages)}"
    )

    try:
        append_wiki_log(
            operation="lint",
            title="Weekly health check",
            details=details,
        )
    except Exception as exc:
        logger.warning("append_log failed: %s", exc)

    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_lint_graph() -> Any:
    """
    Build and compile the Lint Agent LangGraph StateGraph.

    Graph topology (linear — all nodes run in sequence):
      scan_all_pages
           │
      find_orphans
           │
      find_contradictions
           │
      find_stale_claims
           │
      find_missing_pages
           │
      find_gaps
           │
      write_lint_report
           │
      append_log
           │
          END
    """
    graph = StateGraph(LintState)

    graph.add_node("scan_all_pages",      node_scan_all_pages)
    graph.add_node("find_orphans",        node_find_orphans)
    graph.add_node("find_contradictions", node_find_contradictions)
    graph.add_node("find_stale_claims",   node_find_stale_claims)
    graph.add_node("find_missing_pages",  node_find_missing_pages)
    graph.add_node("find_gaps",           node_find_gaps)
    graph.add_node("write_lint_report",   node_write_lint_report)
    graph.add_node("append_log",          node_append_log)

    graph.set_entry_point("scan_all_pages")

    graph.add_edge("scan_all_pages",      "find_orphans")
    graph.add_edge("find_orphans",        "find_contradictions")
    graph.add_edge("find_contradictions", "find_stale_claims")
    graph.add_edge("find_stale_claims",   "find_missing_pages")
    graph.add_edge("find_missing_pages",  "find_gaps")
    graph.add_edge("find_gaps",           "write_lint_report")
    graph.add_edge("write_lint_report",   "append_log")
    graph.add_edge("append_log",          END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_lint() -> LintState:
    """
    Run the full lint pipeline and return the final LintState.

    Returns a dict with:
      orphans:               list of pages with no inbound links
      contradictions:        list of unresolved debate pages
      stale_claims:          list of potentially stale claim strings
      missing_concept_pages: list of missing concept page paths
      gaps:                  list of suggested next search queries
      report_path:           wiki path of the lint report, or None
    """
    app = build_lint_graph()
    final_state = app.invoke({})
    logger.info(
        "Lint complete: orphans=%d contradictions=%d missing=%d gaps=%d report=%s",
        len(final_state.get("orphans", [])),
        len(final_state.get("contradictions", [])),
        len(final_state.get("missing_concept_pages", [])),
        len(final_state.get("gaps", [])),
        final_state.get("report_path"),
    )
    return final_state


# ---------------------------------------------------------------------------
# APScheduler integration
# ---------------------------------------------------------------------------

def start_lint_scheduler() -> Any:
    """
    Start a background APScheduler job that runs the Lint Agent every
    Config.WIKI_LINT_INTERVAL_DAYS days.

    Returns the scheduler instance (call .shutdown() to stop it).
    Safe to call multiple times — only one scheduler is started.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        logger.warning("APScheduler not installed — lint scheduler disabled. "
                       "Install with: pip install apscheduler")
        return None

    interval_days = Config.WIKI_LINT_INTERVAL_DAYS

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=_run_lint_safe,
        trigger="interval",
        days=interval_days,
        id="lint_agent",
        replace_existing=True,
        next_run_time=None,  # don't run immediately on start
    )
    scheduler.start()
    logger.info("Lint scheduler started: runs every %d days", interval_days)
    return scheduler


def _run_lint_safe() -> None:
    """Wrapper that catches all exceptions so the scheduler never crashes."""
    try:
        result = run_lint()
        logger.info("Scheduled lint complete: report=%s", result.get("report_path"))
    except Exception as exc:
        logger.error("Scheduled lint failed: %s", exc)
