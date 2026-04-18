"""
validate.py — End-to-end smoke test for Phases 1–3.

Runs with real APIs (arXiv, Semantic Scholar) but keeps traffic minimal:
  - max 3 papers from arXiv only
  - downloads ONE PDF
  - parses it with the router (PyMuPDF if Grobid/Nougat not running)

Usage:
  python validate.py              # full run
  python validate.py --no-parse  # skip PDF download + parse (fastest)
  python validate.py --source semantic_scholar  # test a specific source

Exit code 0 = all checks passed, non-zero = a check failed.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

# ── make sure project root is on path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m  {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m  {msg}")


def _info(msg: str) -> None:
    print(f"     {msg}")


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Services (Grobid + Nougat)
# ─────────────────────────────────────────────────────────────────────────────

def check_services() -> None:
    """Print service status. Not a pass/fail check — informational only."""
    _section("Services — Grobid & Nougat")
    from services import grobid_status, marker_status, ensure_grobid

    g = grobid_status()
    m = marker_status()

    if g["healthy"]:
        _ok("Grobid is running and healthy")
    else:
        _info("Grobid not running — attempting auto-start ...")
        started = ensure_grobid(auto_start=True)
        if started:
            _ok("Grobid auto-started successfully")
        else:
            if not g["docker_available"]:
                _info("Docker not found. Install Docker Desktop to enable Grobid.")
                _info("  https://www.docker.com/products/docker-desktop/")
            else:
                _info("Docker is available but Grobid failed to start.")
                _info(f"  Manual start: docker run --rm -p 8070:8070 {g['image']}")
            _info("Parser will use PyMuPDF fallback (no action required).")

    if m["available"]:
        _ok("marker is installed (no daemon needed — runs per-PDF)")
    else:
        _info("marker not installed. For math-heavy papers run:")
        _info("  pip install marker-pdf")
        _info("Parser will use PyMuPDF fallback for math papers until then.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

def check_config() -> bool:
    _section("Phase 1 — Config")
    from config import Config

    warnings = Config.validate()
    Config.ensure_dirs()

    for line in Config.status().splitlines():
        _info(line)

    if warnings:
        for w in warnings:
            _fail(f"Config warning: {w}")
        return False

    _ok("Config loaded, all dirs created, no warnings")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Source connectors (live API calls — small max_results)
# ─────────────────────────────────────────────────────────────────────────────

def check_source(source_name: str, query: str = "transformer attention mechanism",
                 max_results: int = 3) -> list:
    from sources.federation import get_connector

    _section(f"Phase 1/2 — Source: {source_name}")
    try:
        connector = get_connector(source_name)
    except ValueError as exc:
        _fail(f"Connector not registered: {exc}")
        return []

    start = time.time()
    results = connector.search(query, max_results=max_results)
    elapsed = time.time() - start

    if not results:
        _fail(f"No results returned from {source_name} ({elapsed:.1f}s)")
        return []

    _ok(f"{len(results)} paper(s) returned in {elapsed:.1f}s")
    for p in results:
        doi_str = f"  DOI: {p.doi}" if p.doi else "  DOI: —"
        oa_str  = "OA" if p.is_open_access else "paywalled"
        pdf_str = "PDF ✓" if p.pdf_url else "PDF ✗"
        _info(f"[{p.paper_id}]  {p.title[:70]}")
        _info(f"     {p.year}  {oa_str}  {pdf_str}  {doi_str}")

    # Structural checks
    failures = []
    for p in results:
        if not p.paper_id:
            failures.append("paper_id missing")
        if not p.title:
            failures.append(f"title missing for {p.paper_id}")
        if p.year < 1900 or p.year > 2030:
            failures.append(f"implausible year {p.year} for {p.paper_id}")
        if p.source != source_name and not (
            source_name == "semantic_scholar" and p.source == "arxiv"
        ):
            failures.append(f"unexpected source '{p.source}' for connector '{source_name}'")

    if failures:
        for f in failures:
            _fail(f)
        return []

    _ok("All returned papers pass structural checks")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Federation (deduplication with two sources)
# ─────────────────────────────────────────────────────────────────────────────

def check_federation(query: str = "transformer attention") -> bool:
    _section("Phase 1 — Federation + Deduplication")
    from config import Config
    from sources.federation import federated_search

    original_sources = Config.ACTIVE_SOURCES
    original_max = Config.SOURCE_MAX_RESULTS.copy()

    # Restrict to arXiv + Semantic Scholar, tiny batch
    Config.ACTIVE_SOURCES = ["arxiv", "semantic_scholar"]
    Config.SOURCE_MAX_RESULTS["arxiv"] = 3
    Config.SOURCE_MAX_RESULTS["semantic_scholar"] = 3

    start = time.time()
    results = federated_search(query, max_total=10)
    elapsed = time.time() - start

    Config.ACTIVE_SOURCES = original_sources
    Config.SOURCE_MAX_RESULTS = original_max

    if not results:
        _fail(f"federated_search returned nothing ({elapsed:.1f}s)")
        return False

    # Check for duplicate paper_ids
    ids = [p.paper_id for p in results]
    if len(ids) != len(set(ids)):
        _fail("Duplicate paper_ids in federation output — deduplication broken")
        return False

    _ok(f"{len(results)} unique papers after dedup in {elapsed:.1f}s")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDF download
# ─────────────────────────────────────────────────────────────────────────────

def check_pdf_download(papers: list) -> tuple[str | None, dict | None]:
    _section("Phase 1 — PDF Download")
    from config import Config

    oa_papers = [p for p in papers if p.is_open_access and p.pdf_url]
    if not oa_papers:
        _info("No open-access papers with PDF URLs available — skipping download check")
        return None, None

    target = oa_papers[0]
    output_dir = str(Config.raw_pdf_dir())
    _info(f"Downloading: {target.title[:70]}")
    _info(f"URL: {target.pdf_url}")

    from sources.federation import get_connector
    connector = get_connector(target.source if target.source != "arxiv" else "arxiv")

    start = time.time()
    pdf_path = connector.download_pdf(target, output_dir)
    elapsed = time.time() - start

    if not pdf_path:
        _fail(f"download_pdf returned None ({elapsed:.1f}s)")
        return None, None

    if not Path(pdf_path).exists():
        _fail(f"PDF path returned but file not on disk: {pdf_path}")
        return None, None

    size_kb = Path(pdf_path).stat().st_size // 1024
    if size_kb < 10:
        _fail(f"Downloaded file suspiciously small: {size_kb} KB")
        return None, None

    _ok(f"PDF saved to {pdf_path} ({size_kb} KB, {elapsed:.1f}s)")
    return pdf_path, target.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parser router
# ─────────────────────────────────────────────────────────────────────────────

def check_parser(pdf_path: str, metadata: dict) -> bool:
    _section("Phase 3 — Parser Router")
    from parsers.router import route_and_parse
    from parsers.grobid_parser import GrobidParser
    from parsers.marker_parser import MarkerParser

    grobid_up = GrobidParser().is_available()
    marker_up = MarkerParser().is_available()
    _info(f"Grobid available: {grobid_up}  (docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0)")
    _info(f"marker available: {marker_up}  (pip install marker-pdf)")

    start = time.time()
    result = route_and_parse(pdf_path, metadata)
    elapsed = time.time() - start

    required_keys = {
        "paper_id", "title", "authors", "abstract", "year", "source",
        "sections", "references", "figures", "tables", "equations",
        "parser_used", "math_fraction", "is_abstract_only",
    }
    missing = required_keys - set(result.keys())
    if missing:
        _fail(f"ParsedPaper missing keys: {missing}")
        return False

    _ok(f"Parsed with '{result['parser_used']}' in {elapsed:.1f}s")
    _info(f"  math_fraction : {result['math_fraction']:.3f}")
    _info(f"  sections      : {len(result['sections'])}")
    _info(f"  references    : {len(result['references'])}")
    _info(f"  figures       : {len(result['figures'])}")
    _info(f"  tables        : {len(result['tables'])}")
    _info(f"  equations     : {len(result['equations'])}")

    if not result["title"] and not metadata.get("title"):
        _fail("title missing from both ParsedPaper and metadata")
        return False

    if result["math_fraction"] < 0 or result["math_fraction"] > 1:
        _fail(f"math_fraction out of range: {result['math_fraction']}")
        return False

    # Show first section snippet
    if result["sections"]:
        s = result["sections"][0]
        snippet = textwrap.shorten(s["text"], width=120, placeholder="…")
        _info(f"  first section : [{s['heading']}] {snippet}")

    _ok("ParsedPaper schema valid")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 6. Abstract-only path
# ─────────────────────────────────────────────────────────────────────────────

def check_abstract_only() -> bool:
    _section("Phase 3 — Abstract-Only Paper Path")
    from parsers.router import route_and_parse

    meta = {
        "paper_id": "pubmed:99999999",
        "title": "A Paywalled Paper",
        "authors": ["Doe, J."],
        "abstract": "This paper is behind a paywall and has no PDF.",
        "year": 2024,
        "source": "pubmed",
        "is_abstract_only": True,
    }
    result = route_and_parse(None, meta)

    if not result["is_abstract_only"]:
        _fail("is_abstract_only not set in result")
        return False
    if result["sections"]:
        _fail("sections should be empty for abstract-only paper")
        return False
    if result["abstract"] != meta["abstract"]:
        _fail("abstract not preserved from metadata")
        return False

    _ok("Abstract-only path: metadata preserved, no sections parsed")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="scilit-agent Phase 1-3 validation")
    parser.add_argument("--no-parse", action="store_true",
                        help="Skip PDF download and parsing (fastest mode)")
    parser.add_argument("--source", default="arxiv",
                        choices=["arxiv", "semantic_scholar", "pubmed",
                                 "biorxiv", "medrxiv"],
                        help="Which source to test for live search")
    parser.add_argument("--query", default="transformer attention mechanism",
                        help="Search query to use")
    args = parser.parse_args()

    passed: list[str] = []
    failed: list[str] = []

    def run(name: str, fn, *a, **kw) -> bool:
        ok = fn(*a, **kw)
        (passed if ok else failed).append(name)
        return ok

    # Always run these
    check_services()   # informational only — never blocks the run
    run("Config", check_config)
    papers = check_source(args.source, query=args.query, max_results=3)
    (passed if papers else failed).append(f"Source:{args.source}")
    run("Federation", check_federation, args.query)
    run("Abstract-only", check_abstract_only)

    # PDF download + parse (skippable)
    if not args.no_parse and papers:
        pdf_path, metadata = check_pdf_download(papers)
        passed.append("PDF download") if pdf_path else failed.append("PDF download")

        if pdf_path and metadata:
            run("Parser", check_parser, pdf_path, metadata)
    else:
        _section("PDF download + parse — SKIPPED (--no-parse)")

    # Summary
    print(f"\n{'═'*60}")
    print(f"  RESULT: {len(passed)} passed, {len(failed)} failed")
    if passed:
        for name in passed:
            print(f"    \033[32m✓\033[0m {name}")
    if failed:
        for name in failed:
            print(f"    \033[31m✗\033[0m {name}")
    print(f"{'═'*60}\n")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
