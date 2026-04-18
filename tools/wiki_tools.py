"""
tools/wiki_tools.py — Read/write operations for the LLM-maintained wiki.

Implements the Karpathy LLM-Wiki pattern:
  Instead of re-deriving knowledge from raw chunks on every query, the system
  maintains a persistent wiki of LLM-written, cross-referenced markdown files
  that compounds with every ingest and every good query answer.

Invariants enforced here (never violate):
  1. log.md is APPEND-ONLY — write_wiki_page refuses to touch it.
  2. index.md is ALWAYS updated after any wiki write (caller responsibility,
     but update_wiki_index() is provided to make it easy).
  3. All writes are scoped to wiki/ — never data/ or source files.
  4. [[wiki-links]] are the cross-reference format throughout.
  5. Each log entry starts with ## [{ISO date}] for grep-ability.

Wiki directory layout:
  wiki/
    index.md               — master catalog, updated on every ingest
    log.md                 — append-only chronological record
    papers/                — one .md per ingested paper
    concepts/              — cross-paper concept synthesis
    methods/               — method comparison pages
    debates/               — contradictions and open questions
    authors/               — research group / author pages (optional)
    synthesis/
      query-answers/       — filed-back query answers
      frontier-*.md        — frontier agent reports
      lint-*.md            — lint reports
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from config import Config

logger = logging.getLogger(__name__)

# Categories understood by index.md and list_wiki_pages()
WikiCategory = Literal["papers", "concepts", "methods", "debates", "authors", "synthesis"]

# Protected files — write_wiki_page() refuses to overwrite these directly
_PROTECTED_FILES = {"log.md"}

# Header patterns for index.md sections
_INDEX_SECTION_RE = re.compile(r"^## (Papers|Concepts|Methods|Debates|Synthesis)\s*\((\d+)\)", re.MULTILINE)
_INDEX_STATS_RE   = re.compile(r"(Last updated: )[\d-]+( \| Papers: )(\d+)( \| Total pages: )(\d+)")

# Log entry format (must start with ## [{ISO date}] for grep-ability)
_LOG_ENTRY_DATE_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}\]", re.MULTILINE)


# ---------------------------------------------------------------------------
# Core read / write
# ---------------------------------------------------------------------------

def read_wiki_index() -> str:
    """
    Read wiki/index.md — the master catalog of all wiki pages.

    Always call this first when answering a query to discover which
    wiki pages are relevant before doing ChromaDB retrieval.
    Returns the full index as a string.
    """
    path = Config.wiki_dir() / "index.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_wiki_page(page_path: str) -> str:
    """
    Read a wiki page by path relative to wiki/.
    e.g. 'concepts/attention-mechanism.md'
         'papers/arxiv-2301.12345.md'

    Returns empty string if the page does not exist.
    """
    full_path = Config.wiki_dir() / page_path
    if not full_path.exists():
        logger.debug("Wiki page not found: %s", full_path)
        return ""
    return full_path.read_text(encoding="utf-8")


def write_wiki_page(page_path: str, content: str, reason: str) -> None:
    """
    Write or overwrite a wiki page. page_path is relative to wiki/.
    NEVER write to raw sources or data/. ONLY write to wiki/.

    reason is recorded in log.md automatically after every successful write.

    Raises ValueError if page_path is a protected file (log.md).
    After writing, always call update_wiki_index() to keep the catalog fresh.
    """
    # Guard: refuse to overwrite protected files
    base = Path(page_path).name
    if base in _PROTECTED_FILES:
        raise ValueError(
            f"Cannot write to protected wiki file '{page_path}'. "
            f"Use append_wiki_log() for log.md."
        )

    # Guard: refuse to escape the wiki directory
    full_path = (Config.wiki_dir() / page_path).resolve()
    wiki_root = Config.wiki_dir().resolve()
    if not str(full_path).startswith(str(wiki_root)):
        raise ValueError(
            f"write_wiki_page: path '{page_path}' escapes wiki directory."
        )

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    logger.info("Wiki page written: %s (%s)", page_path, reason)


def update_wiki_index(page_path: str, summary: str, category: WikiCategory) -> None:
    """
    Add or update an entry in wiki/index.md.

    Call after every write_wiki_page() to keep the master catalog current.
    This is the mechanism by which the wiki compounds — every new page
    immediately becomes discoverable in the index.

    page_path: relative to wiki/ (e.g. 'papers/arxiv-2301.12345.md')
    summary:   one-line description shown in the index table
    category:  'papers' | 'concepts' | 'methods' | 'debates' | 'authors' | 'synthesis'
    """
    index_path = Config.wiki_dir() / "index.md"
    if not index_path.exists():
        _init_index()

    content = index_path.read_text(encoding="utf-8")

    # Derive the wiki-link and a short display title from the path
    stem = Path(page_path).stem
    wiki_link = f"[[{page_path.rstrip('.md')}]]"

    # Determine which section to update and build the new row
    section_header, row = _build_index_row(page_path, wiki_link, summary, category)

    content = _upsert_index_row(content, section_header, wiki_link, row, category)
    content = _update_index_stats(content)

    index_path.write_text(content, encoding="utf-8")
    logger.debug("Wiki index updated: %s (%s)", page_path, category)


def append_wiki_log(operation: str, title: str, details: str) -> None:
    """
    Append an entry to wiki/log.md (append-only, never edits existing entries).

    operation: 'ingest' | 'query-filed' | 'lint' | 'update' | 'init'
    title:     short description (paper_id, query text, etc.)
    details:   one or more lines of detail (parser, chunk count, confidence, etc.)

    Format: ## [ISO date] operation | title
    """
    log_path = Config.wiki_dir() / "log.md"
    if not log_path.exists():
        _init_log()

    today = date.today().isoformat()
    entry = f"\n## [{today}] {operation} | {title}\n{details}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

    logger.debug("Wiki log appended: [%s] %s | %s", today, operation, title)


def list_wiki_pages(category: WikiCategory | None = None) -> list[str]:
    """
    List all wiki page paths relative to wiki/, optionally filtered by category.

    category: 'papers' | 'concepts' | 'methods' | 'debates' | 'authors' | 'synthesis'
              If None, returns all pages (excluding index.md and log.md).

    Returns paths sorted alphabetically, e.g.:
      ['papers/arxiv-2301.12345.md', 'concepts/attention-mechanism.md']
    """
    wiki_root = Config.wiki_dir()
    if not wiki_root.exists():
        return []

    if category:
        search_root = wiki_root / category
        if not search_root.exists():
            return []
        paths = sorted(search_root.rglob("*.md"))
    else:
        paths = sorted(p for p in wiki_root.rglob("*.md")
                       if p.name not in {"index.md", "log.md"})

    return [str(p.relative_to(wiki_root)) for p in paths]


def search_wiki(query: str, top_k: int = 5) -> list[dict]:
    """
    BM25 keyword search over all wiki markdown files.

    Returns a list of {page_path, snippet, score} dicts sorted by relevance.
    Use when index.md alone is insufficient to find relevant pages — e.g.
    for exploratory queries where you need to scan concept and method pages.

    Always call read_wiki_index() first; use search_wiki() for deeper digs.
    """
    pages = list_wiki_pages()
    if not pages:
        return []

    wiki_root = Config.wiki_dir()
    docs: list[tuple[str, str]] = []   # (page_path, text)
    for page_path in pages:
        text = (wiki_root / page_path).read_text(encoding="utf-8", errors="replace")
        docs.append((page_path, text))

    if not docs:
        return []

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed — wiki search unavailable")
        return []

    def tokenise(text: str) -> list[str]:
        return re.split(r"[^a-z0-9]+", text.lower())

    corpus = [tokenise(text) for _, text in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenise(query))

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, (page_path, text) in ranked[:top_k]:
        if score <= 0:
            break
        snippet = _extract_snippet(text, query)
        results.append({
            "page_path": page_path,
            "snippet":   snippet,
            "score":     float(score),
        })

    return results


# ---------------------------------------------------------------------------
# Page template helpers (used by the Ingest Agent in Phase 6)
# ---------------------------------------------------------------------------

def make_paper_page(
    paper_id: str,
    title: str,
    authors: list[str],
    year: int,
    source: str,
    venue: str,
    parser_used: str,
    math_fraction: float,
    is_abstract_only: bool,
    tags: list[str],
    summary: str,
    pdf_url: str | None = None,
) -> str:
    """
    Generate the initial wiki page for a paper (LLM fills in sections later).

    Returns a markdown string with correct frontmatter and section stubs.
    The Ingest Agent calls this to create the page skeleton, then uses the
    LLM to fill in Summary, Key Contributions, Methods, etc.
    """
    today = date.today().isoformat()
    authors_yaml = ", ".join(authors) if authors else ""
    tags_yaml = ", ".join(tags) if tags else ""
    abstract_warning = (
        "\n> ⚠️ ABSTRACT ONLY — PDF unavailable. "
        f"Drop PDF at data/{Config.topic_slug()}/raw/local_drop/ to index full text.\n"
        if is_abstract_only else ""
    )
    source_link = f"[{source}]({pdf_url})" if pdf_url else source
    author_str = " | ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")

    return f"""---
paper_id: {paper_id}
title: "{title}"
authors: [{authors_yaml}]
year: {year}
source: {source}
venue: {venue}
parser_used: {parser_used}
math_fraction: {math_fraction:.3f}
is_abstract_only: {str(is_abstract_only).lower()}
tags: [{tags_yaml}]
date_ingested: {today}
---

# {title}

**Authors**: {author_str} | **Year**: {year} | **Source**: {source_link}
{abstract_warning}
## Summary
{summary}

## Key Contributions
- _To be filled by Ingest Agent_

## Methods
_To be filled by Ingest Agent_

## Results & Claims
- _To be filled by Ingest Agent_

## Limitations
_To be filled by Ingest Agent_

## Related Papers
- Builds on:
- Contradicts:
- Extended by:

## Key Concepts


## Open Questions Raised
-
"""


def make_concept_page(concept: str, tags: list[str] | None = None) -> str:
    """
    Generate a stub wiki page for a concept. The Ingest Agent fills in
    the Definition and Key Papers sections after calling this.
    """
    today = date.today().isoformat()
    tags_yaml = ", ".join(tags or [])
    return f"""---
concept: "{concept}"
tags: [{tags_yaml}]
paper_count: 0
sources: []
last_updated: {today}
---

# {concept}

## Definition
_To be filled by Ingest Agent_

## How it works
_To be filled by Ingest Agent_

## Key Papers
| Paper | Contribution | Year | Source |
|---|---|---|---|

## Variants & Related Concepts

## Cross-domain Notes

## Open Debates
"""


def make_method_page(method: str, domains: list[str] | None = None) -> str:
    """Generate a stub wiki page for a method."""
    today = date.today().isoformat()
    domains_yaml = ", ".join(domains or [])
    return f"""---
method: "{method}"
domains: [{domains_yaml}]
last_updated: {today}
---

# {method}

## Summary
_To be filled by Ingest Agent_

## Comparison Table
| Approach | Paper | Domain | Strengths | Weaknesses | Benchmark |
|---|---|---|---|---|---|

## When to use
_To be filled by Ingest Agent_
"""


def make_debate_page(topic: str, domains: list[str] | None = None) -> str:
    """Generate a stub wiki page for an open debate."""
    today = date.today().isoformat()
    domains_yaml = ", ".join(domains or [])
    return f"""---
debate: "{topic}"
status: open
domains: [{domains_yaml}]
last_updated: {today}
---

# {topic} — Open Debate

## The Question
_To be filled by Ingest Agent_

## Position A
_Claim_ — supported by

## Position B
_Competing claim_ — supported by

## Resolution
_Open. Evidence that would resolve it:_
"""


def make_synthesis_page(
    query: str,
    answer: str,
    sources_wiki: list[str],
    sources_chromadb: list[str],
) -> str:
    """Generate a filed-back query-answer synthesis page."""
    today = date.today().isoformat()
    title = _query_to_title(query)
    wiki_sources = "\n".join(f"- [[{p}]]" for p in sources_wiki) or "_none_"
    db_sources = "\n".join(f"- {p}" for p in sources_chromadb) or "_none_"

    return f"""---
query: "{query}"
date: {today}
sources_wiki: [{", ".join(sources_wiki)}]
sources_chromadb: [{", ".join(sources_chromadb)}]
---

# {title}

{answer}

## Sources

### Wiki pages consulted
{wiki_sources}

### ChromaDB papers cited
{db_sources}
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_index() -> None:
    """Create a fresh index.md."""
    today = date.today().isoformat()
    content = f"""# Wiki Index — {Config.TOPIC_NAME}

Last updated: {today} | Papers: 0 | Total pages: 0
Sources active: {", ".join(Config.ACTIVE_SOURCES)}

## Papers (0)
| Page | Title | Year | Source | Tags |
|---|---|---|---|---|

## Concepts (0)
| Page | Summary | Domains |
|---|---|---|

## Methods (0)
| Page | Summary | Domains |
|---|---|---|

## Debates (0)
| Page | Status | Domains |
|---|---|---|

## Synthesis (0)
| Page | Query | Date |
|---|---|---|
"""
    index_path = Config.wiki_dir() / "index.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(content, encoding="utf-8")


def _init_log() -> None:
    """Create a fresh log.md."""
    today = date.today().isoformat()
    content = f"""# Wiki Log — {Config.TOPIC_NAME}

<!-- Append-only. Never edit or delete existing entries. -->
<!-- Format: ## [ISO date] operation | title -->

## [{today}] init | Wiki initialized
Topic: {Config.TOPIC_NAME} | Sources: {", ".join(Config.ACTIVE_SOURCES)}
"""
    log_path = Config.wiki_dir() / "log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


def _build_index_row(
    page_path: str,
    wiki_link: str,
    summary: str,
    category: WikiCategory,
) -> tuple[str, str]:
    """Return (section_header_text, table_row) for the given category."""
    today = date.today().isoformat()
    stem = Path(page_path).stem

    if category == "papers":
        # Extract year and source from path stem if possible (e.g. arxiv-2301-12345)
        parts = stem.split("-", 1)
        source = parts[0] if parts else ""
        title_short = summary[:60] if summary else stem
        return "Papers", f"| {wiki_link} | {title_short} | — | {source} | |"

    if category == "concepts":
        return "Concepts", f"| {wiki_link} | {summary[:80]} | — |"

    if category == "methods":
        return "Methods", f"| {wiki_link} | {summary[:80]} | — |"

    if category == "debates":
        return "Debates", f"| {wiki_link} | open | — |"

    # synthesis, authors, or anything else
    query_short = summary[:60] if summary else stem
    return "Synthesis", f"| {wiki_link} | {query_short} | {today} |"


def _upsert_index_row(
    content: str,
    section_header: str,
    wiki_link: str,
    new_row: str,
    category: WikiCategory,
) -> str:
    """
    Insert or replace the row for wiki_link in the named section.

    If the link already exists in that section, replace it (idempotent).
    Otherwise append it before the next section header or end of file.
    """
    # Find the section
    section_re = re.compile(
        rf"(## {section_header}\s*\(\d+\)\n(?:.*\n)*?)"
        rf"(?=\n## |\Z)",
        re.MULTILINE,
    )
    m = section_re.search(content)
    if not m:
        # Section not found — append at end
        content += f"\n{new_row}\n"
        return content

    section_text = m.group(0)
    escaped = re.escape(wiki_link)

    # Check if row already exists (update) or add new
    existing_row_re = re.compile(rf"^\|.*{escaped}.*\|.*$", re.MULTILINE)
    if existing_row_re.search(section_text):
        new_section = existing_row_re.sub(new_row, section_text)
    else:
        # Append before next section or end
        new_section = section_text.rstrip("\n") + "\n" + new_row + "\n"

    return content[:m.start()] + new_section + content[m.end():]


def _update_index_stats(content: str) -> str:
    """Update the 'Last updated / Papers / Total pages' stat line."""
    today = date.today().isoformat()

    # Count rows in each section
    paper_rows   = len(re.findall(r"^\| \[\[papers/",   content, re.MULTILINE))
    concept_rows = len(re.findall(r"^\| \[\[concepts/", content, re.MULTILINE))
    method_rows  = len(re.findall(r"^\| \[\[methods/",  content, re.MULTILINE))
    debate_rows  = len(re.findall(r"^\| \[\[debates/",  content, re.MULTILINE))
    synth_rows   = len(re.findall(r"^\| \[\[synthesis/",content, re.MULTILINE))
    author_rows  = len(re.findall(r"^\| \[\[authors/",  content, re.MULTILINE))
    total = paper_rows + concept_rows + method_rows + debate_rows + synth_rows + author_rows

    # Update stats line
    content = re.sub(
        r"Last updated: [\d-]+ \| Papers: \d+ \| Total pages: \d+",
        f"Last updated: {today} | Papers: {paper_rows} | Total pages: {total}",
        content,
    )

    # Update section headers with counts
    def _replace_count(m: re.Match) -> str:
        name = m.group(1)
        counts = {
            "Papers": paper_rows, "Concepts": concept_rows,
            "Methods": method_rows, "Debates": debate_rows,
            "Synthesis": synth_rows,
        }
        return f"## {name} ({counts.get(name, 0)})"

    content = re.sub(r"## (Papers|Concepts|Methods|Debates|Synthesis)\s*\(\d+\)", _replace_count, content)
    return content


def _extract_snippet(text: str, query: str, max_chars: int = 200) -> str:
    """
    Find the most relevant line/paragraph in text for a query,
    return a short snippet.
    """
    query_words = set(re.split(r"\W+", query.lower()))
    best_line = ""
    best_hits = 0

    for line in text.splitlines():
        if not line.strip() or line.startswith("---") or line.startswith("|"):
            continue
        hits = sum(1 for w in query_words if w and w in line.lower())
        if hits > best_hits:
            best_hits = hits
            best_line = line.strip()

    if not best_line:
        # Fall back to first non-frontmatter line
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("---") and not stripped.startswith("#"):
                best_line = stripped
                break

    return best_line[:max_chars] + ("…" if len(best_line) > max_chars else "")


def _query_to_title(query: str) -> str:
    """Convert a query string to a human-readable title (title case, max 80 chars)."""
    title = query.strip().rstrip("?").title()
    return title[:80]


# ---------------------------------------------------------------------------
# Inbound-link graph (used by Lint Agent in Phase 8)
# ---------------------------------------------------------------------------

def build_link_graph() -> dict[str, list[str]]:
    """
    Build an inbound-link graph over all wiki pages.

    Returns {page_path: [pages_that_link_to_it]}.
    Used by the Lint Agent to find orphaned pages (zero inbound links).
    """
    wiki_root = Config.wiki_dir()
    pages = list_wiki_pages()

    # Map: page_path → set of page_paths that link to it
    inbound: dict[str, list[str]] = {p: [] for p in pages}

    _wiki_link_re = re.compile(r"\[\[([^\]]+)\]\]")

    for page_path in pages:
        text = (wiki_root / page_path).read_text(encoding="utf-8", errors="replace")
        for m in _wiki_link_re.finditer(text):
            target = m.group(1).strip()
            # Normalise: ensure .md extension
            if not target.endswith(".md"):
                target = target + ".md"
            if target in inbound:
                inbound[target].append(page_path)

    return inbound


def extract_open_questions() -> list[dict]:
    """
    Collect all 'Open Questions Raised' bullets from papers/ pages.

    Returns list of {paper_id, question} dicts.
    Used by the Frontier Agent (Phase 11) to cluster unaddressed questions.
    """
    results = []
    _oq_re = re.compile(r"^## Open Questions Raised\s*\n((?:- .+\n?)+)", re.MULTILINE)
    _bullet_re = re.compile(r"^- (.+)$", re.MULTILINE)
    _paper_id_re = re.compile(r"^paper_id:\s*(.+)$", re.MULTILINE)

    for page_path in list_wiki_pages("papers"):
        text = read_wiki_page(page_path)
        pid_m = _paper_id_re.search(text)
        paper_id = pid_m.group(1).strip() if pid_m else page_path

        oq_m = _oq_re.search(text)
        if oq_m:
            for q in _bullet_re.finditer(oq_m.group(1)):
                q_text = q.group(1).strip()
                if q_text and q_text != "_":
                    results.append({"paper_id": paper_id, "question": q_text})

    return results
