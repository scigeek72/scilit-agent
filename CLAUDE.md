# CLAUDE.md — scilit-agent

This file serves two purposes simultaneously:
1. **Build spec** — tells Claude Code how to construct this system from scratch
2. **Runtime schema** — tells the running agent how to maintain the wiki, ingest papers,
   answer queries, and keep the knowledge base healthy

Read the entire file before writing any code or touching any files.

---

## Project Overview

**scilit-agent** is a source-agnostic, agentic scientific literature research assistant.
The user types a plain text query. The system searches across all configured open-access
sources in parallel, parses the retrieved papers intelligently, builds a persistent
LLM-maintained wiki, and answers questions by synthesizing compiled knowledge — not by
re-deriving it from raw chunks on every query.

The system combines three ideas into one coherent architecture:

1. **Federated source search**: A single query fans out across arXiv, PubMed, bioRxiv,
   medRxiv, Semantic Scholar, and a local PDF drop folder. The user never names a source.
   The system handles routing, deduplication, and open-access resolution invisibly.

2. **Agentic orchestration via LangGraph**: The LLM drives all pipeline decisions
   dynamically. No hardcoded if/else routing. Three agents — Ingest, Query, Lint —
   each implemented as a LangGraph StateGraph.

3. **LLM-Wiki pattern** (Karpathy, 2026): Instead of re-deriving knowledge from raw
   chunks on every query, the system maintains a persistent wiki of LLM-written,
   cross-referenced markdown files that compounds with every ingest and every good
   query answer. ChromaDB handles exact retrieval; the wiki handles compiled synthesis.

**Parser routing**: Grobid (structure-heavy papers) and marker (math-heavy papers),
with the agent choosing per paper based on content classification.

**LLM providers**: Provider-agnostic. Supports OpenAI, Anthropic, and LM Studio (local).
Switch via config — no code changes required.

**Target users**: Researchers in any scientific domain — CS, biology, medicine, physics,
climate science, etc. Topic-agnostic: change one line in config.py to switch domains.

**Language**: Python 3.10+

---

## The Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER INPUT — plain text query                                       │
│  e.g. "attention mechanisms in transformer models"                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SOURCE FEDERATION LAYER                                             │
│  Single query fans out to all active sources in parallel             │
│                                                                      │
│   arXiv   PubMed   bioRxiv   medRxiv   Semantic Scholar   local_pdf │
│      └────────┴────────┴────────┴───────────┴────────────┘          │
│                              │                                       │
│                   Deduplicate by DOI / title                         │
│                   Open-access resolution                             │
│                   Unified ranked PaperMetadata list                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PARSER ROUTER                                                       │
│  classify paper → math_fraction >= threshold?                        │
│       YES → marker   (LaTeX / equations)                             │
│       NO  → Grobid   (sections / references / tables)               │
│       FALLBACK → PyMuPDF (plain text, if both unavailable)           │
│  Output: unified ParsedPaper dict — parser-agnostic downstream       │
└─────────────────────────────────────────────────────────────────────┘
                    │                         │
          ┌─────────┘                         └──────────┐
          ▼                                              ▼
┌─────────────────────┐               ┌──────────────────────────────┐
│  ChromaDB + BM25    │               │  LLM WIKI  (wiki/)           │
│  Semantic chunks    │               │  Compiled, cross-referenced  │
│  for exact retrieval│               │  markdown knowledge base     │
│  and quotes         │               │  that compounds over time    │
└─────────────────────┘               └──────────────────────────────┘
          │                                              │
          └────────────────────┬─────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LANGGRAPH QUERY AGENT                                               │
│  reads wiki/index.md → routes to wiki and/or ChromaDB               │
│  synthesizes answer → files good answers back into wiki              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
scilit-agent/
│
├── CLAUDE.md                          ← this file (build spec + runtime schema)
├── config.py                          ← topic, sources, LLM provider, all toggles
├── .env.example
├── .gitignore
├── requirements.txt
│
├── sources/                           ← SOURCE FEDERATION LAYER
│   ├── __init__.py
│   ├── base.py                        ← SourceConnector ABC + PaperMetadata dataclass
│   ├── federation.py                  ← federated_search(): fans out, deduplicates
│   ├── arxiv_source.py
│   ├── pubmed_source.py
│   ├── biorxiv_source.py
│   ├── medrxiv_source.py
│   ├── semantic_scholar_source.py     ← universal fallback, aggregates many sources
│   └── local_pdf_source.py            ← user drops PDFs into watched folder
│
├── parsers/
│   ├── __init__.py
│   ├── base.py                        ← Parser ABC
│   ├── router.py                      ← classify paper → choose parser
│   ├── grobid_parser.py               ← TEI XML → ParsedPaper
│   ├── marker_parser.py               ← markdown → ParsedPaper
│   └── pymupdf_parser.py              ← plain text fallback → ParsedPaper
│
├── retrieval/
│   ├── __init__.py
│   ├── embeddings.py                  ← provider-agnostic embedding wrapper
│   ├── vector_store.py                ← ChromaDB wrapper
│   ├── bm25_index.py                  ← BM25 index (rank-bm25)
│   ├── hybrid_search.py               ← RRF fusion of BM25 + semantic
│   ├── reranker.py                    ← cross-encoder ms-marco-MiniLM
│   └── hyde.py                        ← hypothetical document embedding
│
├── tools/
│   ├── __init__.py
│   ├── source_tools.py                ← federated_search, fetch_metadata, download_pdf
│   ├── parse_tools.py                 ← run_parser (router-aware)
│   ├── retrieval_tools.py             ← search_vector_db, hybrid_search, rerank
│   ├── wiki_tools.py                  ← read/write wiki pages, index, log, search
│   ├── citation_tools.py              ← get_references, get_cited_by (Semantic Scholar)
│   └── llm_tools.py                   ← summarize, decompose, self_critique, is_worth_filing
│
├── agents/
│   ├── __init__.py
│   ├── state.py                       ← IngestState, QueryState, LintState TypedDicts
│   ├── ingest_agent.py                ← LangGraph: search → parse → index → wiki
│   ├── query_agent.py                 ← LangGraph: classify → route → answer → file-back
│   └── lint_agent.py                  ← LangGraph: health-check wiki
│
├── llm_provider.py                    ← get_llm() and get_embeddings() factories
│
├── wiki/                              ← LLM-maintained knowledge base
│   ├── index.md                       ← master catalog, updated on every ingest
│   ├── log.md                         ← append-only chronological record
│   ├── papers/                        ← one .md per ingested paper
│   ├── concepts/                      ← cross-paper concept synthesis
│   ├── methods/                       ← method comparison pages
│   ├── debates/                       ← contradictions and open questions
│   ├── authors/                       ← research group / author pages (optional)
│   └── synthesis/
│       └── query-answers/             ← filed-back query answers
│
├── data/
│   └── {topic_slug}/
│       ├── raw/
│       │   ├── pdfs/                  ← downloaded PDFs (immutable)
│       │   └── local_drop/            ← user drops PDFs here manually
│       ├── parsed/                    ← Grobid/marker/PyMuPDF JSON output
│       ├── vector_db/                 ← ChromaDB
│       ├── bm25_index/                ← serialized BM25 index
│       ├── cache/
│       │   └── query_cache.db         ← SQLite persistent query cache
│       └── metadata/
│           └── papers.json            ← unified PaperMetadata for all papers
│
├── interfaces/
│   ├── cli.py
│   ├── gradio_interface.py
│   └── desktop_app.py                 ← PyQt6 (Tkinter fallback)
│
└── tests/
    ├── test_sources.py
    ├── test_parsers.py
    ├── test_tools.py
    ├── test_agents.py
    └── fixtures/
        └── sample_pdfs/
```

---

## Configuration (config.py)

```python
# config.py — the only file users need to edit

# ── Topic ──────────────────────────────────────────────────────────────
TOPIC_NAME: str = "Transformer Models"
SEARCH_QUERY: str = "attention mechanisms transformer models"
# Plain text. The federation layer submits this to each source's search API.
# No source-specific syntax required from the user.

# ── Sources ─────────────────────────────────────────────────────────────
# Enable/disable sources. Order determines deduplication priority.
ACTIVE_SOURCES: list[str] = [
    "arxiv",
    "biorxiv",
    "medrxiv",
    "pubmed",
    "semantic_scholar",   # universal fallback — searches across many publishers
    "local_pdf",          # always active; watches data/{topic}/raw/local_drop/
]

SOURCE_MAX_RESULTS: dict[str, int] = {
    "arxiv":            100,
    "biorxiv":           50,
    "medrxiv":           30,
    "pubmed":            50,
    "semantic_scholar":  50,
    "local_pdf":        999,   # no limit — ingest everything in the drop folder
}

# ── LLM Provider ────────────────────────────────────────────────────────
# "openai" | "anthropic" | "lmstudio"
LLM_PROVIDER: str = "openai"
OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
ANTHROPIC_CHAT_MODEL: str = "claude-sonnet-4-20250514"
LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
LM_STUDIO_MODEL: str = "local-model"

# ── Embeddings ──────────────────────────────────────────────────────────
# "openai" | "local"
EMBEDDING_PROVIDER: str = "local"
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

# ── Parser ──────────────────────────────────────────────────────────────
PARSER_AUTO_ROUTE: bool = True
DEFAULT_PARSER: str = "grobid"        # fallback if routing disabled
GROBID_URL: str = "http://localhost:8070"
MATH_THRESHOLD: float = 0.3           # fraction of math tokens → use marker

# ── Retrieval ───────────────────────────────────────────────────────────
BM25_WEIGHT: float = 0.4
SEMANTIC_WEIGHT: float = 0.6
RETRIEVAL_K: int = 20
RERANK_TOP_K: int = 5
USE_HYDE: bool = True
USE_HYBRID_SEARCH: bool = True
USE_RERANKER: bool = True
USE_MULTIHOP: bool = True

# ── Wiki ────────────────────────────────────────────────────────────────
WIKI_DIR: str = "wiki"
WIKI_WRITE_BACK: bool = True
WIKI_LINT_INTERVAL_DAYS: int = 7

# ── Cache ───────────────────────────────────────────────────────────────
USE_PERSISTENT_CACHE: bool = True
CACHE_MAX_SIZE: int = 500

# ── Build ───────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50
```

---

## Source Federation Layer

### SourceConnector ABC (sources/base.py)

Every source implements this interface. The rest of the system only ever calls these
methods — it never imports a specific source connector directly.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class PaperMetadata:
    paper_id: str        # source-namespaced: "arxiv:2301.12345", "pubmed:38291847"
    title: str
    authors: list[str]
    abstract: str
    year: int
    source: str          # "arxiv" | "pubmed" | "biorxiv" | "medrxiv" |
                         # "semantic_scholar" | "local"
    pdf_url: str | None  # None if paywalled or unavailable
    doi: str | None
    venue: str | None
    tags: list[str]
    is_open_access: bool


class SourceConnector(ABC):

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        """Search this source. query is plain text — no source-specific syntax."""

    @abstractmethod
    def fetch_metadata(self, paper_id: str) -> PaperMetadata:
        """Fetch full metadata for a known paper_id."""

    @abstractmethod
    def download_pdf(self, metadata: PaperMetadata, output_dir: str) -> str | None:
        """
        Download PDF to output_dir. Returns local file path, or None if unavailable.
        Never raises on download failure — return None and log the reason.
        """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique string identifier for this source."""
```

### Federated Search (sources/federation.py)

```python
def federated_search(
    query: str,
    max_total: int = 200,
) -> list[PaperMetadata]:
    """
    Submit query to all active sources in parallel.
    Deduplicate by DOI (primary) then by title similarity (fallback).
    Return a unified ranked list. Caller never knows which source each
    paper came from — that information is in PaperMetadata.source.
    """
```

**Deduplication logic** (in priority order):
1. Exact DOI match → keep the copy from the highest-priority source
2. Title similarity >= 0.92 (fuzzy match) → keep highest-priority source copy
3. No match → keep both (different papers)

**Open-access resolution**:
- For each paper, check `pdf_url` and `is_open_access`
- If paywalled: log as `access: abstract_only`, index abstract + metadata only
- Unpaywall API can be queried as a fallback to find open-access versions of
  paywalled papers (free, no key required for low volume)

```
Paper found but paywalled:
  → Index abstract + metadata in ChromaDB (labeled as abstract_only)
  → Create stub wiki/papers/{id}.md with abstract only, marked ⚠️ ABSTRACT ONLY
  → Log: "PDF unavailable — drop file at data/{topic}/raw/local_drop/{id}.pdf
          to index full text"
```

### Source implementations

| File | API used | Notes |
|---|---|---|
| `arxiv_source.py` | arxiv Python library | Always open access |
| `pubmed_source.py` | NCBI Entrez (Biopython) | Metadata free; PDFs via PMC |
| `biorxiv_source.py` | bioRxiv REST API | Always open access (preprints) |
| `medrxiv_source.py` | medRxiv REST API | Always open access (preprints) |
| `semantic_scholar_source.py` | Semantic Scholar API | Aggregates many publishers; often finds OA PDFs for paywalled papers |
| `local_pdf_source.py` | filesystem watch | User drops PDFs into local_drop/; no API needed |

**Semantic Scholar** is the recommended universal fallback because it indexes papers
from Nature, Springer, Elsevier, IEEE, ACM, and many others, and frequently has
open-access PDF links even when the publisher does not. Always include it in
ACTIVE_SOURCES.

---

## Parser Router

### Decision logic

```
Input: PDF path + PaperMetadata (source, tags)
       │
       ├─ math_fraction >= MATH_THRESHOLD (0.3)?
       │       → marker (LaTeX / equations)
       │
       ├─ math_fraction < MATH_THRESHOLD?
       │       → Grobid (structured scientific prose, references, tables)
       │
       ├─ Grobid or marker unavailable?
       │       → PyMuPDF fallback (plain text extraction)
       │
       └─ DEFAULT_PARSER from config (if PARSER_AUTO_ROUTE = False)
```

### Unified output schema (ParsedPaper)

Both Grobid and marker return this same dict. Downstream code is parser-agnostic.

```python
ParsedPaper = {
    "paper_id":     str,              # source-namespaced
    "title":        str,
    "authors":      list[str],
    "abstract":     str,
    "year":         int,
    "source":       str,
    "sections": [
        {"heading": str, "text": str, "level": int}
    ],
    "references": [
        {"ref_id": str, "title": str, "authors": list[str], "year": int}
    ],
    "figures":  [{"caption": str, "label": str}],
    "tables":   [{"caption": str, "label": str}],
    "equations": [str],               # LaTeX strings; marker only, empty list otherwise
    "parser_used":    "grobid" | "marker" | "pymupdf",
    "math_fraction":  float,
    "is_abstract_only": bool,         # True if PDF was unavailable
}
```

### Parser setup

**Grobid** — runs as a local Docker container:
```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```
Call `POST /api/processFulltextDocument` with PDF as multipart/form-data.
Parse TEI XML response with lxml.

**marker** — install via pip:
```bash
pip install marker-pdf
marker_single {pdf_path} {output_dir}
```
Output is `.md` (Markdown). Parse sections by `#` / `##` / `###` headings.
Works on Intel Mac, Apple Silicon (MPS), Linux (CPU/CUDA), and Windows.
No minimum PyTorch version constraint — portable across all platforms.

**PyMuPDF** — always available, no external service needed. Use as fallback only.

---

## LLM Provider Abstraction (llm_provider.py)

All agents and tools must use these factory functions. Never instantiate LLM clients
directly anywhere else in the codebase.

```python
def get_llm(temperature: float = 0.0):
    """Return a LangChain-compatible chat model based on config."""
    provider = Config.LLM_PROVIDER
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=Config.OPENAI_CHAT_MODEL, temperature=temperature)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=Config.ANTHROPIC_CHAT_MODEL, temperature=temperature)
    elif provider == "lmstudio":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=Config.LM_STUDIO_BASE_URL,
            api_key="lm-studio",
            model=Config.LM_STUDIO_MODEL,
            temperature=temperature,
        )

def get_embeddings():
    """Return a LangChain-compatible embeddings model based on config."""
    if Config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=Config.OPENAI_EMBEDDING_MODEL)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
```

---

## Tools Registry

All tools are plain Python functions with type annotations and docstrings.
The docstring is what the LLM reads to decide when to call the tool.
Every tool must be independently testable without a running LLM.

### source_tools.py

```python
def federated_search(query: str, max_results: int = 200) -> list[dict]:
    """
    Search all configured sources for papers matching the plain-text query.
    Returns a deduplicated, ranked list of PaperMetadata dicts.
    Use this as the entry point for any new paper discovery.
    The user never needs to specify which source to search.
    """

def fetch_paper_metadata(paper_id: str) -> dict:
    """
    Fetch full metadata for a known paper by its namespaced ID
    (e.g. 'arxiv:2301.12345', 'pubmed:38291847').
    Routes to the correct source connector automatically.
    """

def download_pdf(metadata: dict, output_dir: str) -> str | None:
    """
    Download the PDF for a paper. Returns local file path, or None if unavailable
    (paywalled or no PDF link). Never raises — logs failure and returns None.
    If None: paper will be indexed as abstract_only.
    """
```

### wiki_tools.py

```python
def read_wiki_index() -> str:
    """
    Read wiki/index.md — the master catalog of all wiki pages.
    Always call this first when answering a query.
    """

def read_wiki_page(page_path: str) -> str:
    """
    Read a wiki page by path relative to wiki/.
    e.g. 'concepts/attention-mechanism.md'
    """

def write_wiki_page(page_path: str, content: str, reason: str) -> None:
    """
    Write or overwrite a wiki page. path is relative to wiki/.
    NEVER write to raw sources or data/. ONLY write to wiki/.
    reason is recorded in log.md.
    """

def update_wiki_index(page_path: str, summary: str, category: str) -> None:
    """
    Add or update an entry in wiki/index.md.
    Call after every write_wiki_page.
    category: 'paper' | 'concept' | 'method' | 'debate' | 'author' | 'synthesis'
    """

def append_wiki_log(operation: str, title: str, details: str) -> None:
    """
    Append an entry to wiki/log.md (append-only, never edit existing entries).
    operation: 'ingest' | 'query-filed' | 'lint' | 'update'
    Format: ## [{ISO date}] {operation} | {title}
    """

def search_wiki(query: str, top_k: int = 5) -> list[dict]:
    """
    BM25 keyword search over all wiki markdown files.
    Returns list of {page_path, snippet, score}.
    Use when index.md alone is insufficient to find relevant pages.
    """

def list_wiki_pages(category: str | None = None) -> list[str]:
    """
    List all wiki page paths, optionally filtered by subdirectory.
    category: 'papers' | 'concepts' | 'methods' | 'debates' | 'authors' | 'synthesis'
    """
```

### retrieval_tools.py

```python
def search_vector_db(query: str, top_k: int = 20,
                     year_filter: int | None = None,
                     source_filter: str | None = None) -> list[dict]:
    """
    Semantic search over ChromaDB chunks.
    source_filter: optionally restrict to 'arxiv' | 'pubmed' | 'biorxiv' etc.
    Use for exact quotes or specific claims. Returns chunk_text, paper_id,
    title, authors, year, source, score.
    """

def hybrid_search(query: str, top_k: int = 20,
                  year_filter: int | None = None,
                  source_filter: str | None = None) -> list[dict]:
    """
    BM25 + semantic search fused with Reciprocal Rank Fusion.
    Preferred over pure semantic for technical terms, gene names, drug names.
    """

def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder rerank (ms-marco-MiniLM-L-6-v2).
    Always call after retrieval, before passing chunks to the LLM.
    """
```

### citation_tools.py

```python
def get_references(paper_id: str) -> list[dict]:
    """
    Papers cited by this paper. Uses Semantic Scholar API.
    Works for any source (arxiv, pubmed, doi-based).
    Returns list of {title, authors, year, paper_id, doi}.
    """

def get_cited_by(paper_id: str, limit: int = 20) -> list[dict]:
    """
    Papers that cite this paper. Uses Semantic Scholar API.
    Use to find follow-up work and assess impact.
    """
```

### llm_tools.py

```python
def generate_hyde_document(query: str) -> str:
    """
    Generate a hypothetical paper excerpt that would answer the query.
    Use before retrieval for exploratory or vague queries.
    """

def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex query into 2-4 simpler sub-queries.
    Use when query contains: compare, difference, vs, survey, review, explain.
    """

def self_critique(query: str, answer: str, sources: list[dict]) -> dict:
    """
    Evaluate whether the answer is grounded in the provided sources.
    Returns {is_grounded: bool, confidence: float, issues: list[str]}.
    """

def is_worth_filing(query: str, answer: str) -> bool:
    """
    Decide whether this query-answer pair should be written back to the wiki.
    Returns True for novel synthesis, comparisons, cross-paper analyses.
    Returns False for simple factual lookups or trivial answers.
    """
```

---

## LangGraph Agents

### Agent 1: Ingest Agent (agents/ingest_agent.py)

```
State: IngestState
  query: str                      # original search query
  candidates: list[PaperMetadata] # from federated_search
  paper: PaperMetadata | None     # current paper being processed
  pdf_path: str | None
  parsed_paper: ParsedPaper | None
  is_abstract_only: bool
  chunks: list[str]
  wiki_pages_written: list[str]
  errors: list[str]

Nodes:
  federated_search     → fan out to all active sources
  deduplicate          → remove duplicates by DOI / title
  resolve_open_access  → check pdf_url, try Unpaywall fallback
  classify_paper       → math_fraction → choose parser
  parse_grobid         → TEI XML → ParsedPaper
  parse_marker         → markdown → ParsedPaper
  parse_pymupdf        → plain text fallback → ParsedPaper
  chunk_and_embed      → split, embed, write to ChromaDB + BM25
  write_paper_page     → wiki/papers/{source}-{id}.md
  update_concept_pages → update or create wiki/concepts/*.md
  update_method_pages  → update or create wiki/methods/*.md
  flag_contradictions  → update wiki/debates/*.md if claims conflict
  update_index         → wiki/index.md
  write_log            → wiki/log.md

Conditional edges:
  resolve_open_access:
    pdf available     → classify_paper
    abstract only     → chunk_and_embed (abstract text only)

  classify_paper:
    math_fraction >= threshold  → parse_marker
    math_fraction <  threshold  → parse_grobid

  parse_grobid / parse_marker:
    success           → chunk_and_embed
    service error     → parse_pymupdf (fallback)
```

### Agent 2: Query Agent (agents/query_agent.py)

```
State: QueryState
  query: str
  year_filter: int | None
  source_filter: str | None
  query_type: str | None     # factual | comparative | exploratory | survey
  sub_queries: list[str]
  wiki_pages_read: list[str]
  retrieved_chunks: list[dict]
  answer: str | None
  sources: list[dict]
  is_grounded: bool
  retry_count: int
  should_file_back: bool
  filed_page_path: str | None

Nodes:
  check_cache          → SQLite hit? return immediately
  classify_query       → determine query_type
  read_wiki_index      → always: find relevant wiki pages
  read_wiki_pages      → read relevant pages
  decide_retrieval     → wiki sufficient? or need ChromaDB chunks?
  generate_hyde        → if exploratory
  decompose_query      → if comparative or survey
  hybrid_search        → ChromaDB + BM25
  rerank               → cross-encoder
  synthesize_answer    → LLM generates answer
  self_critique        → verify grounding
  refine_answer        → retry once if not grounded
  decide_file_back     → is_worth_filing?
  write_synthesis_page → wiki/synthesis/query-answers/{date}-{slug}.md
  save_cache           → SQLite

Conditional edges:
  decide_retrieval:
    wiki sufficient          → synthesize_answer
    need chunks              → hybrid_search

  self_critique:
    grounded                 → decide_file_back
    not grounded, retry < 1  → refine_answer
    not grounded, retry >= 1 → decide_file_back (proceed with caveat)
```

### Agent 3: Lint Agent (agents/lint_agent.py)

```
Nodes:
  scan_all_pages        → build inbound-link graph over wiki/
  find_orphans          → pages with zero inbound links
  find_contradictions   → unresolved flags in wiki/debates/
  find_stale_claims     → claims not reflected in newest papers
  find_missing_pages    → concepts in papers/ without own concept page
  find_gaps             → under-represented topics, suggest next searches
  write_lint_report     → wiki/synthesis/lint-{date}.md
  append_log            → wiki/log.md

Triggered: on schedule (WIKI_LINT_INTERVAL_DAYS) or on user command.
```

---

## Wiki Schema (Runtime Instructions)

**This section governs the running agent's behavior, not just construction.**

### Page formats

#### wiki/papers/{source}-{id}.md

```markdown
---
paper_id: arxiv:2301.12345
title: "{title}"
authors: [{author1}, {author2}]
year: {year}
source: arxiv | pubmed | biorxiv | medrxiv | local
venue: {journal or arXiv}
parser_used: grobid | marker | pymupdf
math_fraction: {float}
is_abstract_only: false
tags: [{tag1}, {tag2}]
date_ingested: {ISO date}
---

# {Title}

**Authors**: ... | **Year**: ... | **Source**: [{source}]({url})

> ⚠️ ABSTRACT ONLY — PDF unavailable. Drop PDF at data/{topic}/raw/local_drop/
> to index full text.        ← include only if is_abstract_only: true

## Summary
{2–4 sentence plain-language summary}

## Key Contributions
- ...

## Methods
{Reference [[concepts/]] and [[methods/]] pages inline}

## Results & Claims
- {claim} (confidence: high | medium | low)
- {claim} ← ⚠️ contradicted by [[papers/arxiv-2402.67890]]

## Limitations
...

## Related Papers
- Builds on: [[papers/...]]
- Contradicts: [[papers/...]]
- Extended by: [[papers/...]]

## Key Concepts
[[concepts/{concept1}]], [[concepts/{concept2}]]

## Open Questions Raised
- ...
```

#### wiki/concepts/{concept-name}.md

```markdown
---
concept: "{concept name}"
tags: []
paper_count: {N}
sources: [arxiv, pubmed]
last_updated: {ISO date}
---

# {Concept Name}

## Definition
...

## How it works
...

## Key Papers
| Paper | Contribution | Year | Source |
|---|---|---|---|
| [[papers/arxiv-2301.12345]] | introduced concept | 2023 | arXiv |
| [[papers/pubmed-38291847]]  | applied in biology | 2024 | PubMed |

## Variants & Related Concepts
- [[concepts/{related}]] — {one-line distinction}

## Cross-domain Notes
{If the concept appears in both CS and biology/medicine papers, note
differences in terminology and application here.}

## Open Debates
[[debates/{topic}]]
```

#### wiki/methods/{method-name}.md

```markdown
---
method: "{method name}"
domains: [cs, biology, medicine]
last_updated: {ISO date}
---

# {Method Name}

## Summary
...

## Comparison Table
| Approach | Paper | Domain | Strengths | Weaknesses | Benchmark |
|---|---|---|---|---|---|

## When to use
...
```

#### wiki/debates/{topic}.md

```markdown
---
debate: "{topic}"
status: open | resolved | superseded
domains: [cs, biology]
last_updated: {ISO date}
---

# {Topic} — Open Debate

## The Question
...

## Position A
{Claim} — supported by [[papers/...]]

## Position B
{Competing claim} — supported by [[papers/...]]

## Resolution
{If resolved: how. If open: what evidence would resolve it.}
```

#### wiki/synthesis/query-answers/{date}-{slug}.md

```markdown
---
query: "{original query}"
date: {ISO date}
sources_wiki: [{page_path}, ...]
sources_chromadb: [{paper_id}, ...]
---

# {Descriptive title}

{The filed-back answer, cleaned up for future reference}

## Sources
...
```

### index.md format

```markdown
# Wiki Index — {TOPIC_NAME}

Last updated: {ISO date} | Papers: {N} | Total pages: {N}
Sources active: arxiv, pubmed, biorxiv, semantic_scholar, local_pdf

## Papers ({N})
| Page | Title | Year | Source | Tags |
|---|---|---|---|---|
| [[papers/arxiv-2301.12345]] | Schema Linking... | 2023 | arXiv | schema-linking |
| [[papers/pubmed-38291847]]  | BERT for NER...   | 2024 | PubMed | ner, bert |

## Concepts ({N})
| Page | Summary | Domains |
|---|---|---|

## Methods ({N})
| Page | Summary | Domains |
|---|---|---|

## Debates ({N})
| Page | Status | Domains |
|---|---|---|

## Synthesis ({N})
| Page | Query | Date |
|---|---|---|
```

### log.md format

Each entry must start with `## [{ISO date}]` for grep-ability:

```markdown
# Wiki Log — {TOPIC_NAME}

## [2026-04-08] ingest | arxiv:2301.12345 — Schema Linking for Text-to-SQL
Parser: grobid | Chunks: 47 | Wiki pages updated: 8 | Access: full

## [2026-04-08] ingest | pubmed:38291847 — BERT for NER in Clinical Text
Parser: grobid | Chunks: 0 | Wiki pages updated: 1 | Access: abstract_only
Note: PDF unavailable — stub page created

## [2026-04-08] query-filed | Compare BERT vs GPT for schema linking
Confidence: high | Wiki pages read: 4 | Chunks used: 5

## [2026-04-08] lint | Weekly health check
Orphans: 2 | Contradictions: 1 | Missing concept pages: 3
```

---

## Operations Reference (for running agent)

### Ingest new papers

```
User: "Find papers about CRISPR gene editing"

Agent:
1. federated_search("CRISPR gene editing") → PaperMetadata list
2. For each paper:
   a. download_pdf(metadata, output_dir)
      → if None: mark is_abstract_only, continue
   b. classify_paper(pdf_path) → parser choice
   c. parse_{parser}(pdf_path) → ParsedPaper
   d. chunk_and_embed(parsed_paper) → ChromaDB + BM25
   e. write_wiki_page("papers/{source}-{id}.md", ...)
   f. update relevant concepts/, methods/, debates/ pages
   g. update_wiki_index(...)
   h. append_wiki_log("ingest", ...)
```

### Answer a query

```
User: "What are the main challenges in CRISPR delivery mechanisms?"

Agent:
1. check_cache(query) → miss
2. classify_query → "exploratory"
3. read_wiki_index() → find relevant pages
4. read_wiki_page("concepts/crispr-delivery.md")
5. decide_retrieval → need ChromaDB chunks too
6. generate_hyde(query) → hypothetical excerpt
7. hybrid_search(hyde_doc) → rerank_chunks(...)
8. synthesize_answer(query, wiki_pages, chunks)
9. self_critique(...) → is_grounded: true
10. save_cache(query, answer)
11. is_worth_filing → true
    write_wiki_page("synthesis/query-answers/2026-04-08-crispr-delivery.md")
    append_wiki_log("query-filed", ...)
12. Return answer with citations
```

### Local PDF drop

```
User drops a Nature paper PDF into data/{topic}/raw/local_drop/

local_pdf_source detects new file →
  extract title/authors from PDF metadata or first page →
  create PaperMetadata (source: "local", is_open_access: true) →
  hand off to ingest pipeline (parser router → wiki → ChromaDB)
```

### Run lint

```
User: "Run lint" (or triggered by scheduler)

Agent:
1. scan_all_pages() → inbound-link graph
2. find_orphans(), find_contradictions(), find_stale_claims()
3. find_missing_pages(), find_gaps()
4. write_wiki_page("synthesis/lint-{date}.md", report)
5. append_wiki_log("lint", "Weekly health check", summary)
6. Return report to user with suggested next searches
```

---

## Implementation Phases

Complete and test each phase before moving to the next.

### Phase 1 — Foundation
- [ ] `config.py`
- [ ] `llm_provider.py` — get_llm() and get_embeddings() factories
- [ ] `sources/base.py` — SourceConnector ABC + PaperMetadata dataclass
- [ ] `sources/arxiv_source.py`
- [ ] `sources/semantic_scholar_source.py` — universal fallback, implement early
- [ ] `sources/local_pdf_source.py` — filesystem watcher
- [ ] `sources/federation.py` — federated_search() with deduplication
- [ ] Unit tests for federation and deduplication logic

### Phase 2 — Remaining Sources
- [ ] `sources/pubmed_source.py` (Biopython Entrez)
- [ ] `sources/biorxiv_source.py`
- [ ] `sources/medrxiv_source.py`
- [ ] Open-access resolution + Unpaywall fallback in federation.py
- [ ] Unit tests for each source connector

### Phase 3 — Parsers
- [ ] `parsers/base.py` — Parser ABC
- [ ] `parsers/grobid_parser.py` — TEI XML → ParsedPaper
- [ ] `parsers/marker_parser.py` — markdown → ParsedPaper
- [ ] `parsers/pymupdf_parser.py` — fallback
- [ ] `parsers/router.py` — classify paper, choose parser, handle fallback
- [ ] Unit tests for all parsers against fixture PDFs

### Phase 4 — Retrieval Layer
- [ ] `retrieval/vector_store.py` — ChromaDB wrapper
- [ ] `retrieval/bm25_index.py` — with serialization
- [ ] `retrieval/embeddings.py` — provider-agnostic
- [ ] `retrieval/hybrid_search.py` — RRF fusion
- [ ] `retrieval/reranker.py` — cross-encoder
- [ ] `retrieval/hyde.py`
- [ ] `tools/retrieval_tools.py`

### Phase 5 — Wiki Layer
- [ ] Initialize wiki/ directory + empty index.md + log.md
- [ ] `tools/wiki_tools.py` — all read/write operations
- [ ] BM25 search over wiki markdown (wiki_tools.search_wiki)
- [ ] Unit tests (no LLM required — pure file I/O)

### Phase 6 — Ingest Agent
- [ ] `agents/state.py` — all TypedDicts
- [ ] `tools/source_tools.py`
- [ ] `tools/parse_tools.py`
- [ ] `tools/citation_tools.py`
- [ ] `tools/llm_tools.py`
- [ ] `agents/ingest_agent.py` — full LangGraph graph
- [ ] End-to-end test: ingest one arXiv paper, verify wiki pages written
- [ ] End-to-end test: ingest abstract-only paper, verify stub page + log

### Phase 7 — Query Agent
- [ ] `agents/query_agent.py` — full LangGraph graph
- [ ] SQLite persistent cache
- [ ] End-to-end test: query → answer → file-back → verify wiki updated

### Phase 8 — Lint Agent
- [ ] `agents/lint_agent.py`
- [ ] Link graph construction over wiki/
- [ ] APScheduler integration
- [ ] End-to-end test: lint on small corpus

### Phase 9 — Interfaces
- [ ] `interfaces/cli.py`
- [ ] `interfaces/gradio_interface.py`
- [ ] `interfaces/desktop_app.py` (PyQt6 + Tkinter fallback)

### Phase 10 — README and Polish
- [ ] README.md: Grobid Docker setup, marker pip install, API keys, quickstart
- [ ] `--rebuild` flag to wipe ChromaDB + wiki and start fresh
- [ ] Verify all three LLM providers work end-to-end

### Phase 11 — Frontier Agent (RESERVED — implement after Phase 10)

**Implement after Phases 1–10 are working. The Frontier Agent runs on the
wiki alone — it does NOT require the KG layer (Phase 12). The KG is an
optional enhancement that improves gap detection if available, but the
Frontier Agent is fully functional without it.**

#### Purpose
A reactive agent — triggered by the user, never on a schedule — that
synthesizes across the entire wiki to surface unexplored territory:
methodological gaps (technique X not applied to domain Y) and conceptual
gaps (question Z raised repeatedly but never addressed by any method).
The LLM cannot discover genuinely novel hypotheses, but it can map the
*shape of what's missing* from the compiled knowledge base and hand that
map to the user, who supplies the scientific judgment.

#### Trigger
```
User: "What's unexplored in CRISPR delivery mechanisms?"
User: "What methodological gaps exist in this corpus?"
User: "What open questions has nobody worked on?"
```

#### How it works without KG (wiki-only mode, default)

The Frontier Agent reads the wiki directly — no graph traversal needed:

- **Aggregate open questions**: every `papers/{id}.md` has an
  "Open Questions Raised" section written at ingest time. The agent reads
  all of them, clusters similar ones, and ranks by how many independent
  papers raise the same question without anyone addressing it.
- **Detect method-domain gaps**: reads `methods/` and `concepts/` pages
  together. If method X is documented and applied in domain A and B but
  not C — and domain C papers exist in the corpus — that is a gap.
- **Find temporal dropouts**: reads `log.md` and paper dates to find
  research directions active in one period then absent in recent papers.
- **Surface contradiction clusters**: reads `debates/` pages to find
  clusters of related contradictions pointing at a deeper unresolved
  methodological question.
- **Cross-domain opportunities**: reads `concepts/` pages for the
  "Cross-domain Notes" field written during ingest — flags concepts
  established in one domain but absent in another.

#### LangGraph graph (agents/frontier_agent.py)

```
State: FrontierState
  query: str                        # user's gap question
  query_focus: str                  # 'methodological' | 'conceptual' | 'both'
  open_questions: list[str]         # from papers/ "Open Questions" sections
  method_domain_gaps: list[dict]    # from methods/ + concepts/ cross-reference
  temporal_dropouts: list[dict]     # from log.md + paper dates
  contradiction_clusters: list      # from debates/ pages
  cross_domain_opportunities: list  # from concepts/ cross-domain notes
  kg_gap_edges: list[dict]          # OPTIONAL: populated only if KG available
  wiki_context: dict                # relevant wiki pages read for narrative
  report: str | None
  filed_page_path: str | None

Nodes:
  classify_focus           → methodological | conceptual | both
  read_wiki_index          → find relevant wiki pages
  aggregate_open_questions → cluster open questions across papers/
  find_method_domain_gaps  → cross-reference methods/ and concepts/
  find_temporal_dropouts   → scan log.md and paper dates
  find_contradiction_clusters → scan debates/ pages
  find_cross_domain        → scan concepts/ cross-domain notes
  query_kg_gaps            → OPTIONAL: traverse KG if USE_KG=True
  read_wiki_context        → read concept/method/debate pages for narrative
  synthesize_report        → LLM generates structured gap report
  file_report              → wiki/synthesis/frontier-{date}.md
  append_log               → wiki/log.md

Conditional edges:
  classify_focus:
    methodological → find_method_domain_gaps + find_cross_domain
    conceptual     → aggregate_open_questions + find_temporal_dropouts
    both           → all gap-finding nodes

  after all gap-finding nodes:
    USE_KG=True    → query_kg_gaps → read_wiki_context
    USE_KG=False   → read_wiki_context (skip kg node)
```

#### Output format (wiki/synthesis/frontier-{date}.md)

```markdown
---
query: "{user's question}"
date: {ISO date}
focus: methodological | conceptual | both
papers_analyzed: {N}
kg_used: true | false
---

# Research Frontier Report — {topic}

## Methodological Gaps
### {Method} has not been applied to {Domain/Disease}
- Method documented in: [[papers/...]], [[papers/...]]
- Domain documented in: [[papers/...]]
- Why this gap is plausible: {narrative from wiki context}
- What would be needed to bridge it: {narrative}
- Confidence: high | medium | low

## Conceptual Gaps
### "{Open Question}" — raised but unaddressed
- Raised in: [[papers/...]], [[papers/...]] ({N} papers)
- No method currently targets this question in corpus
- Related concepts: [[concepts/...]]
- Confidence: high | medium | low

## Cross-Domain Opportunities
### {Concept} established in {Domain A}, absent in {Domain B}
- ...

## Temporal Dropouts
### {Research direction} active {year range}, then quiet
- Last paper: [[papers/...]] ({year})
- Possible reasons: {narrative}
- Worth revisiting because: {narrative}

## Suggested Next Searches
- "{search query 1}" — to fill gap X
- "{search query 2}" — to fill gap Y

---
*This report reflects the current corpus only. Gaps may be actively
researched in unpublished or unindexed work. Human judgment required
to assess feasibility and significance.*
```

#### Caveats to always include in the Frontier Agent system prompt
- Surface gaps in *this corpus*, not in all of science
- Flag when a gap may be unexplored for good reasons (technically
  infeasible, already addressed under a different name, or simply niche)
- Never assert that a gap *should* be filled — only that it *exists*
- Confidence-score every gap: high (many papers converge) | medium | low

---

### Phase 12 — Knowledge Graph Layer (RESERVED — optional enhancement)

**Entirely optional. The Frontier Agent (Phase 11) works without this.
Implement only if wiki-only gap detection proves insufficient after real
use. Do not attempt to finalize the schema before ingesting 50–100 papers
— the right node and edge types only become clear from real data.**

#### Motivation
The wiki layer is human-readable synthesis. The KG layer makes relationships
machine-traversable, enabling systematic graph traversal for gap detection
rather than LLM reading of prose pages. When the KG is available, the
Frontier Agent's `query_kg_gaps` node activates and supplements (does not
replace) the wiki-based gap finding.

#### Planned node types (validate against real corpus before finalizing)
```
Method, Concept, Disease, Organism, Benchmark,
Paper, Author, OpenQuestion
```

#### Planned edge types (validate against real corpus before finalizing)
```
APPLIED_TO, TESTED_IN, ACHIEVES, CONTRADICTS,
BLOCKS, PROPOSES, RAISES, CITES,
NOT_YET_APPLIED_TO   ← primary gap-detection primitive
```

#### Technology decision (deferred)
Prefer `NetworkX` (pure Python, no external DB, serializable to JSON) for
corpora under ~500 papers. Migrate to `Neo4j` (AuraDB free tier) if the
graph exceeds NetworkX's in-memory limits or Cypher queries become
necessary. Use `neo4j-graphrag` Python package for extraction — do NOT
use the Neo4j llm-graph-builder web app.

#### Integration points (when ready)
- Ingest Agent: add optional `extract_kg_entities` node after
  `chunk_and_embed`, running on the same ParsedPaper dict
- Frontier Agent: `query_kg_gaps` node already stubbed in Phase 11 —
  activate by setting `USE_KG = True` in config.py
- New module: `kg/` alongside `wiki/` and `retrieval/`
- New tool: `tools/kg_tools.py` — `query_kg`, `find_gap_edges`,
  `get_node_neighbors`
- New config flag: `USE_KG: bool = False` (default off)

---

## Key Invariants (never violate these)

1. **Raw sources are immutable.** Never write to `data/{topic}/raw/`. Read only.
2. **The wiki is LLM-owned.** The agent writes all wiki pages. Humans read.
3. **The user never names a source.** All source routing is internal to the
   federation layer. Queries are always plain text.
4. **Parser output schema is stable.** Both parsers return the same ParsedPaper
   dict. Downstream code must not depend on parser-specific fields.
5. **get_llm() and get_embeddings() are the only LLM entry points.**
   Never instantiate ChatOpenAI, ChatAnthropic, etc. directly in agent or tool code.
6. **Abstract-only papers are first-class citizens.** They get a stub wiki page,
   are indexed by abstract in ChromaDB, and are clearly marked. Never silently skip them.
7. **Write-back is gated by is_worth_filing().** Never write trivial answers to wiki.
8. **log.md is append-only.** Never edit or delete existing entries.
9. **index.md is always updated after any wiki write.**
10. **All tools are independently testable.** No tool requires a running LLM to unit test.

---

## Dependencies (requirements.txt)

```
# Core
python-dotenv>=1.0.0
pydantic>=2.0.0
tqdm>=4.66.0
numpy>=1.26.0

# LLM providers
langchain>=0.2.0
langchain-core>=0.2.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-huggingface>=0.0.3
openai>=1.30.0
anthropic>=0.25.0

# LangGraph
langgraph>=0.1.0

# Sources
arxiv>=2.1.0
biopython>=1.83          # PubMed / Entrez
requests>=2.31.0         # bioRxiv, medRxiv, Semantic Scholar, Grobid REST
semanticscholar>=0.8.0
watchdog>=4.0.0          # local_pdf_source filesystem watcher

# Parsers
PyMuPDF>=1.24.0          # fallback parser
marker-pdf>=0.2.6
lxml>=5.0.0              # TEI XML for Grobid
thefuzz>=0.22.0          # fuzzy title deduplication in federation

# Retrieval
chromadb>=0.5.0
sentence-transformers>=3.0.0
rank-bm25>=0.2.2

# Scheduler
APScheduler>=3.10.0

# Interfaces
gradio>=4.0.0
PyQt6>=6.7.0

# Dev / test
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

---

## Environment Variables (.env.example)

```
# LLM — set whichever provider you use
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Sources — all optional; system degrades gracefully without them
SEMANTIC_SCHOLAR_API_KEY=      # free tier works without key (rate-limited)
NCBI_API_KEY=                  # PubMed; free, increases rate limit

# No key needed for: arXiv, bioRxiv, medRxiv, Unpaywall, local_pdf
```

---

## Notes for Claude Code

- Read this entire file before writing any code.
- Implement Phase 1 first. Do not proceed to Phase 2 until Phase 1 tests pass.
- Ask before making any architectural decision not covered here.
- If Grobid or marker are unavailable, fall back to PyMuPDF silently and log it.
- The first working configuration to verify end-to-end is:
  LLM_PROVIDER=openai, ACTIVE_SOURCES=["arxiv", "semantic_scholar", "local_pdf"]
- Anthropic and LM Studio provider support should be verified in Phase 10.
- Prioritize correctness and test coverage over feature completeness.
  A working Phase 1–5 is more valuable than a broken Phase 1–10.
