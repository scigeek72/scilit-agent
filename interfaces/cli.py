"""
interfaces/cli.py — Command-line interface for scilit-agent.

Commands:
  ingest  <query>    Search all sources and ingest papers into wiki + index.
  query   <question> Answer a question from the knowledge base.
  lint               Run wiki health check and write a lint report.
  status             Show service status (Grobid, marker, ChromaDB, etc.).
  rebuild            Wipe ChromaDB, BM25 index, and wiki, then start fresh.

Usage:
  python -m interfaces.cli ingest "attention mechanisms transformers"
  python -m interfaces.cli query "What are the key challenges in CRISPR delivery?"
  python -m interfaces.cli lint
  python -m interfaces.cli status
  python -m interfaces.cli rebuild --confirm
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Allow running as `python -m interfaces.cli` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from agents.ingest_agent import run_ingest
from agents.query_agent import run_query
from agents.lint_agent import run_lint
from agents.frontier_agent import run_frontier
import services

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cli")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingest pipeline for a search query."""
    query = " ".join(args.query)
    print(f"Ingesting papers for: '{query}'")
    print("Searching all sources...\n")

    result = run_ingest(query)

    n_papers  = len(result.get("papers_processed", []))
    n_chunks  = result.get("chunks_indexed", 0)
    n_wiki    = sum(1 for p in result.get("papers_processed", []))
    errors    = result.get("errors", [])

    print(f"✓ Papers processed : {n_papers}")
    print(f"✓ Chunks indexed   : {n_chunks}")
    if errors:
        print(f"⚠ Non-fatal errors : {len(errors)}")
        if args.verbose:
            for e in errors:
                print(f"  • {e}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Answer a question from the knowledge base."""
    question = " ".join(args.question)
    print(f"Query: {question}\n")

    result = run_query(
        question,
        year_filter=args.year,
        source_filter=args.source,
    )

    answer     = result.get("answer", "No answer generated.")
    confidence = result.get("confidence", 0.0)
    cache_hit  = result.get("cache_hit", False)
    filed      = result.get("filed_page_path")
    sources    = result.get("sources", [])

    print("─" * 60)
    print(answer)
    print("─" * 60)

    tags: list[str] = []
    if cache_hit:
        tags.append("cache hit")
    if not result.get("is_grounded", True):
        tags.append("⚠ low confidence")
    conf_pct = f"{confidence * 100:.0f}%"
    tag_str  = f"  [{', '.join(tags)}]" if tags else ""
    print(f"\nConfidence: {conf_pct}{tag_str}")

    if filed:
        print(f"Filed to  : {filed}")

    if args.verbose and sources:
        print("\nSources:")
        seen: set[str] = set()
        for s in sources:
            pid = s.get("paper_id", s.get("page_path", ""))
            if pid and pid not in seen:
                seen.add(pid)
                title = s.get("title", pid)
                year  = s.get("year", "")
                print(f"  • {title}" + (f" ({year})" if year else ""))

    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    """Run the wiki lint agent."""
    print("Running wiki lint check...")
    result = run_lint()

    orphans        = result.get("orphans", [])
    contradictions = result.get("contradictions", [])
    stale          = result.get("stale_claims", [])
    missing        = result.get("missing_concept_pages", [])
    gaps           = result.get("gaps", [])
    report         = result.get("report_path")

    print(f"\n{'─'*40}")
    print(f"  Orphaned pages       : {len(orphans)}")
    print(f"  Unresolved debates   : {len(contradictions)}")
    print(f"  Potentially stale    : {len(stale)}")
    print(f"  Missing concept pages: {len(missing)}")
    print(f"  Gap suggestions      : {len(gaps)}")
    print(f"{'─'*40}")

    if gaps:
        print("\nSuggested next searches:")
        for g in gaps[:5]:
            print(f"  • {g}")

    if report:
        print(f"\nFull report: {report}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show service and dependency status."""
    services.print_service_status()
    print()

    # Show wiki stats
    wiki_dir = Config.wiki_dir()
    if wiki_dir.exists():
        paper_count   = len(list((wiki_dir / "papers").glob("*.md")))   if (wiki_dir / "papers").exists()   else 0
        concept_count = len(list((wiki_dir / "concepts").glob("*.md"))) if (wiki_dir / "concepts").exists() else 0
        method_count  = len(list((wiki_dir / "methods").glob("*.md")))  if (wiki_dir / "methods").exists()  else 0
        debate_count  = len(list((wiki_dir / "debates").glob("*.md")))  if (wiki_dir / "debates").exists()  else 0
        print(f"Wiki ({Config.TOPIC_NAME}):")
        print(f"  Papers   : {paper_count}")
        print(f"  Concepts : {concept_count}")
        print(f"  Methods  : {method_count}")
        print(f"  Debates  : {debate_count}")
    else:
        print("Wiki: not initialised (run 'ingest' first)")

    # Show vector DB stats
    vdb_dir = Config.vector_db_dir()
    if vdb_dir.exists():
        print(f"\nVector DB: {vdb_dir}")
    else:
        print("\nVector DB: not initialised")

    return 0


def cmd_frontier(args: argparse.Namespace) -> int:
    """Surface unexplored methodological and conceptual gaps in the wiki corpus."""
    query = " ".join(args.query)
    print(f"Frontier analysis: '{query}'\n")
    print("Scanning wiki for research gaps...\n")

    result = run_frontier(query)

    focus   = result.get("query_focus", "both")
    oqs     = result.get("open_questions", [])
    gaps    = result.get("method_domain_gaps", [])
    temporal = result.get("temporal_dropouts", [])
    cross   = result.get("cross_domain_opportunities", [])
    report  = result.get("report", "")
    filed   = result.get("filed_page_path")

    print(f"Focus: {focus}")
    print(f"Open question clusters  : {len(oqs)}")
    print(f"Method-domain gaps      : {len(gaps)}")
    print(f"Temporal dropouts       : {len(temporal)}")
    print(f"Cross-domain opportunities: {len(cross)}")
    print()

    if report:
        print("─" * 60)
        print(report[:2000])
        if len(report) > 2000:
            print(f"\n… ({len(report) - 2000} more characters — see full report below)")
        print("─" * 60)

    if filed:
        print(f"\nFull report filed to: {filed}")

    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Wipe ChromaDB, BM25 index, and wiki, then re-initialise directories."""
    if not args.confirm:
        print("This will DELETE all indexed data and the wiki.")
        print("Re-run with --confirm to proceed.")
        return 1

    targets = [
        Config.vector_db_dir(),
        Config.bm25_index_dir(),
        Config.wiki_dir(),
        Config.cache_db_path().parent,
    ]

    for path in targets:
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed: {path}")

    Config.ensure_dirs()
    _init_wiki()
    print("\n✓ Rebuild complete. Run 'ingest' to re-populate.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scilit-agent",
        description="Agentic scientific literature research assistant.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed output (errors, sources, etc.)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ingest
    p_ingest = sub.add_parser("ingest", help="Search sources and ingest papers.")
    p_ingest.add_argument("query", nargs="+", help="Plain-text search query.")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="Answer a question from the knowledge base.")
    p_query.add_argument("question", nargs="+", help="Plain-text question.")
    p_query.add_argument("--year",   type=int, default=None, metavar="YEAR",
                         help="Restrict retrieval to papers from this year or later.")
    p_query.add_argument("--source", type=str, default=None,
                         metavar="SOURCE",
                         help="Restrict retrieval to a single source (e.g. arxiv).")
    p_query.set_defaults(func=cmd_query)

    # lint
    p_lint = sub.add_parser("lint", help="Run wiki health check.")
    p_lint.set_defaults(func=cmd_lint)

    # status
    p_status = sub.add_parser("status", help="Show service and wiki status.")
    p_status.set_defaults(func=cmd_status)

    # frontier
    p_frontier = sub.add_parser(
        "frontier",
        help="Surface unexplored gaps in the corpus.",
        description=(
            "Analyse the wiki to find methodological gaps (technique X not applied to domain Y), "
            "conceptual gaps (open questions never addressed), temporal dropouts, and "
            "cross-domain opportunities."
        ),
    )
    p_frontier.add_argument(
        "query", nargs="+",
        help='Plain-text gap question, e.g. "What methodological gaps exist?"',
    )
    p_frontier.set_defaults(func=cmd_frontier)

    # rebuild
    p_rebuild = sub.add_parser("rebuild", help="Wipe all data and start fresh.")
    p_rebuild.add_argument("--confirm", action="store_true",
                           help="Required to actually perform the wipe.")
    p_rebuild.set_defaults(func=cmd_rebuild)

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_wiki() -> None:
    """Write empty index.md and log.md so the wiki is valid after a rebuild."""
    from datetime import date
    wiki_dir = Config.wiki_dir()
    today    = date.today().isoformat()

    index = wiki_dir / "index.md"
    if not index.exists():
        index.write_text(
            f"# Wiki Index — {Config.TOPIC_NAME}\n\n"
            f"Last updated: {today} | Papers: 0 | Total pages: 0\n\n"
            "## Papers (0)\n| Page | Title | Year | Source | Tags |\n|---|---|---|---|---|\n\n"
            "## Concepts (0)\n| Page | Summary | Domains |\n|---|---|---|\n\n"
            "## Methods (0)\n| Page | Summary | Domains |\n|---|---|---|\n\n"
            "## Debates (0)\n| Page | Status | Domains |\n|---|---|---|\n\n"
            "## Synthesis (0)\n| Page | Query | Date |\n|---|---|---|\n",
            encoding="utf-8",
        )

    log = wiki_dir / "log.md"
    if not log.exists():
        log.write_text(
            f"# Wiki Log — {Config.TOPIC_NAME}\n\n"
            f"## [{today}] init | Wiki initialised\n"
            "Directories created. Ready for ingest.\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
