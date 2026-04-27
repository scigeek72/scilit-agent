# scilit-agent

Source-agnostic, agentic scientific literature research assistant.

Type a plain-text query. The system searches arXiv, PubMed, bioRxiv, medRxiv, Semantic Scholar, and your local PDF folder in parallel, parses retrieved papers intelligently, and maintains a persistent LLM-written wiki that compounds with every ingest and every query. Answers are synthesised from the compiled wiki — not re-derived from raw chunks on every call.

---

## Why this project exists

One of my work projects involved fine-tuning an LLM for text-to-SQL — part of a
broader set of applied ML work at my company. The volume of relevant papers has
grown so fast that keeping up became genuinely unmanageable — not just
time-consuming, but structurally broken. Downloading PDFs, skimming abstracts,
deciding what was worth a full read, extracting the methods and results that might
actually be useful, and then somehow remembering how it all connected — none of
that scaled.

I started with a bare-bones script to automate the download-and-summarise loop.
That turned into this: a system that ingests papers from every major open-access
source, parses them properly (equations and all), writes LLM-maintained synthesis
pages that cross-reference each other, and answers questions from that compiled
knowledge rather than re-reading raw PDFs every time.

This grew from a real, daily frustration — not from curiosity about the tech. The
tech is in service of one goal: spend less time managing papers and more time
thinking about what they actually mean.

---

## Architecture overview

```
User query
    │
    ▼
Source Federation (arxiv, pubmed, biorxiv, medrxiv, semantic_scholar, local_pdf)
    │  deduplicate → open-access resolve
    ▼
Parser Router  →  Grobid (structure-heavy)  |  marker (math-heavy)  |  PyMuPDF (fallback)
    │
    ├──► ChromaDB + BM25  (exact retrieval / quotes)
    │
    └──► LLM Wiki  wiki/  (compiled synthesis, compounds over time)
                │
                ▼
         LangGraph Query Agent  →  answer  →  file-back to wiki
```

Three LangGraph agents:
- **Ingest** — search → parse → index → wiki
- **Query** — classify → route → answer → file-back
- **Lint** — weekly health check of the wiki
- **Frontier** — surface unexplored methodological and conceptual gaps in the corpus

---

## Requirements

- Python 3.10+
- Docker (for Grobid — optional, PyMuPDF fallback always available)
- An API key for at least one LLM provider (OpenAI, Anthropic, or a local LM Studio server)

---

## Getting started (full walkthrough)

Follow these steps in order. Each step builds on the previous one.

### Step 1 — Clone and install

```bash
git clone <repo>
cd scilit-agent
pip install -r requirements.txt
```

### Step 2 — Set your API key

```bash
cp .env.example .env
```

Open `.env` and fill in the key for whichever LLM provider you want to use:

```env
# Pick one:
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# (LM Studio needs no key — see Configuration section below)
```

### Step 3 — Choose your topic and LLM provider

Open `config.py` and set these two lines:

```python
TOPIC_NAME: str = "Transformer Models"        # ← change to your topic
SEARCH_QUERY: str = "attention mechanisms transformer models"  # ← matching query

LLM_PROVIDER: str = "openai"                  # ← "openai" | "anthropic" | "lmstudio"
```

Everything else can stay at its default for a first run.

### Step 4 — (Optional) Start Grobid for better paper parsing

Skip this step if you don't have Docker — PyMuPDF will be used as a fallback automatically.

```bash
docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

### Step 5 — Check that services are ready

```bash
python -m interfaces.cli status
```

You should see Grobid ✅ (or ❌ with PyMuPDF fallback noted), your LLM config, and an empty wiki.

### Step 6 — Ingest papers

```bash
python -m interfaces.cli ingest "attention mechanisms transformer models"
```

This fans out to all configured sources, downloads open-access PDFs, parses them, builds the ChromaDB index, and writes wiki pages. Let it run — it may take a few minutes depending on how many papers are found.

### Step 7 — Ask a question

```bash
python -m interfaces.cli query "What are the key ideas behind self-attention?"
```

### Step 8 — Explore research gaps (optional)

```bash
python -m interfaces.cli frontier "What methodological gaps exist in this corpus?"
```

### Step 9 — Run a health check on the wiki (optional)

```bash
python -m interfaces.cli lint
```

### Step 10 — Use the web UI instead (optional)

```bash
python -m interfaces.gradio_interface
# open http://localhost:7860
```

All commands above are also available as tabs in the Gradio UI.

---

## Installation

### Grobid (structured-paper parser)

Grobid runs as a local Docker container. It handles section extraction, reference parsing, and table detection for structure-heavy papers.

```bash
docker pull lfoppiano/grobid:0.8.0
docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

Verify it is running:
```bash
curl http://localhost:8070/api/isalive
# → true
```

If Grobid is unavailable, the system automatically falls back to PyMuPDF (plain-text extraction). No action needed — the fallback is silent.

### marker (math-heavy paper parser)

marker handles LaTeX equations and math-dense papers.

```bash
pip install marker-pdf
```

marker works on Intel Mac, Apple Silicon (MPS), Linux (CPU/CUDA), and Windows. No minimum PyTorch version constraint.

If marker is unavailable, the system falls back to Grobid or PyMuPDF. No action needed.

---

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

### API keys

```env
# Set the key for whichever LLM provider you use
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional — free tier works without a key (rate-limited)
SEMANTIC_SCHOLAR_API_KEY=

# Optional — increases PubMed rate limit
NCBI_API_KEY=
```

No key is required for arXiv, bioRxiv, medRxiv, Unpaywall, or local PDFs.

### LLM provider

Edit `config.py` to select your provider:

```python
# "openai" | "anthropic" | "lmstudio"
LLM_PROVIDER: str = "openai"
OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
ANTHROPIC_CHAT_MODEL: str = "claude-sonnet-4-20250514"
LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
LM_STUDIO_MODEL: str = "local-model"
```

**OpenAI** — set `LLM_PROVIDER = "openai"` and `OPENAI_API_KEY` in `.env`.

**Anthropic** — set `LLM_PROVIDER = "anthropic"` and `ANTHROPIC_API_KEY` in `.env`.

**LM Studio (local)** — start LM Studio, load a model, start the local server (default port 1234), then set `LLM_PROVIDER = "lmstudio"`. No API key required.

### Embeddings

```python
# "openai" | "local"
EMBEDDING_PROVIDER: str = "local"
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"       # downloaded automatically
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
```

`"local"` uses a HuggingFace model downloaded on first use (~1.3 GB for `bge-large`). `"openai"` requires `OPENAI_API_KEY`.

### Topic

The only line most users need to change:

```python
TOPIC_NAME: str = "Transformer Models"
SEARCH_QUERY: str = "attention mechanisms transformer models"
```

Change these to any scientific domain. The system is topic-agnostic.

---

## Quickstart

### 1. Check service status

```bash
python -m interfaces.cli status
```

### 2. Ingest papers

```bash
python -m interfaces.cli ingest "attention mechanisms transformer models"
```

The system fans out across all configured sources, downloads open-access PDFs, parses them, embeds chunks into ChromaDB, and writes wiki pages.

### 3. Ask a question

```bash
python -m interfaces.cli query "What are the main challenges in scaling transformer models?"
```

### 4. Ask with filters

```bash
# Restrict to papers from 2023 or later
python -m interfaces.cli query "What is flash attention?" --year 2023

# Restrict to a single source
python -m interfaces.cli query "BERT vs GPT for classification" --source arxiv
```

### 5. Run lint (wiki health check)

```bash
python -m interfaces.cli lint
```

Reports orphaned pages, unresolved debates, stale claims, missing concept pages, and suggested next searches.

### 6. Explore research gaps (Frontier Agent)

```bash
python -m interfaces.cli frontier "What methodological gaps exist in this corpus?"
python -m interfaces.cli frontier "What open questions has nobody worked on?"
python -m interfaces.cli frontier "What's unexplored in attention mechanisms?"
```

The Frontier Agent reads the entire wiki and surfaces:
- Methods applied in domain A but not domain B
- Open questions raised repeatedly but never addressed
- Research directions that went quiet
- Cross-domain concept transplant opportunities

### 7. Verbose output

```bash
python -m interfaces.cli -v query "Compare BERT and GPT"
python -m interfaces.cli -v ingest "CRISPR gene editing"
```

### 8. Wipe and rebuild from scratch

```bash
python -m interfaces.cli rebuild --confirm
```

This deletes the ChromaDB vector store, BM25 index, wiki, and query cache, then re-initialises empty directories. Run `ingest` afterwards to re-populate.

---

## Web UI (Gradio)

```bash
python -m interfaces.gradio_interface
# → open http://localhost:7860
```

Tabs: **Query**, **Ingest**, **Lint**, **Frontier**, **Status**.

---

## Desktop app (PyQt6)

```bash
python -m interfaces.desktop_app
```

Falls back to Tkinter if PyQt6 is not installed.

---

## Drop a local PDF

Place any PDF in `data/{topic_slug}/raw/local_drop/`. On the next `ingest` run the system detects it, extracts metadata, and processes it through the full pipeline (parser → wiki → ChromaDB). No API key needed — the file is treated as open access.

For the default topic:
```bash
cp my_paper.pdf data/transformer_models/raw/local_drop/
python -m interfaces.cli ingest "attention mechanisms"
```

---

## Wiki layout

```
wiki/
├── index.md              ← master catalog, updated on every ingest
├── log.md                ← append-only chronological record
├── papers/               ← one .md per ingested paper
├── concepts/             ← cross-paper concept synthesis
├── methods/              ← method comparison pages
├── debates/              ← contradictions and open questions
├── authors/              ← research group pages (optional)
└── synthesis/
    ├── query-answers/    ← filed-back query answers
    ├── frontier-*.md     ← frontier gap reports
    └── lint-*.md         ← lint reports
```

The wiki is entirely LLM-maintained. Never edit pages manually — run `ingest` or `query` to update them.

---

## LLM provider verification

To confirm all three providers load correctly without making LLM calls:

```python
from config import Config
from llm_provider import get_llm, get_embeddings

# OpenAI
Config.LLM_PROVIDER = "openai"
llm = get_llm()
print(type(llm).__name__)   # ChatOpenAI

# Anthropic
Config.LLM_PROVIDER = "anthropic"
llm = get_llm()
print(type(llm).__name__)   # ChatAnthropic

# LM Studio
Config.LLM_PROVIDER = "lmstudio"
llm = get_llm()
print(type(llm).__name__)   # ChatOpenAI (pointed at localhost:1234)

# Embeddings
Config.EMBEDDING_PROVIDER = "local"
emb = get_embeddings()
print(type(emb).__name__)   # HuggingFaceEmbeddings
```

---

## Running tests

```bash
pytest tests/ -v
```

Key test files:
- `tests/test_sources.py` — federation and deduplication logic
- `tests/test_parsers.py` — parser output schema
- `tests/test_wiki_tools.py` — wiki read/write operations
- `tests/test_agents.py` — ingest agent end-to-end
- `tests/test_query_agent.py` — query agent end-to-end
- `tests/test_lint_agent.py` — lint agent
- `tests/test_frontier_agent.py` — frontier agent

All tools are independently testable without a running LLM (pass a mock as the `llm` parameter).

---

## Key invariants

1. **Raw sources are immutable.** Never write to `data/{topic}/raw/`.
2. **The wiki is LLM-owned.** The agent writes all wiki pages. Humans read.
3. **The user never names a source.** All routing is internal to the federation layer.
4. **Parser output schema is stable.** Both Grobid and marker return the same `ParsedPaper` dict.
5. **`get_llm()` and `get_embeddings()` are the only LLM entry points.**
6. **Abstract-only papers are first-class citizens.** They get a stub wiki page.
7. **Write-back is gated by `is_worth_filing()`.** Trivial answers are not stored.
8. **`log.md` is append-only.** Never edit or delete existing entries.

---

## Configuration reference

All settings live in `config.py`. The table below covers the most commonly changed ones.

| Setting | Default | Description |
|---|---|---|
| `TOPIC_NAME` | `"Transformer Models"` | Display name for this research topic |
| `SEARCH_QUERY` | `"attention mechanisms..."` | Default search query for new ingests |
| `ACTIVE_SOURCES` | all six | Which sources to search |
| `LLM_PROVIDER` | `"openai"` | `"openai"` \| `"anthropic"` \| `"lmstudio"` |
| `EMBEDDING_PROVIDER` | `"local"` | `"openai"` \| `"local"` |
| `PARSER_AUTO_ROUTE` | `True` | Auto-choose Grobid vs marker per paper |
| `MATH_THRESHOLD` | `0.3` | Math token fraction → use marker |
| `USE_HYBRID_SEARCH` | `True` | BM25 + semantic fusion |
| `USE_HYDE` | `True` | Hypothetical document for retrieval |
| `USE_RERANKER` | `True` | Cross-encoder rerank |
| `WIKI_WRITE_BACK` | `True` | File novel answers back to wiki |
| `WIKI_LINT_INTERVAL_DAYS` | `7` | Auto-lint interval |
| `USE_KG` | `False` | Knowledge graph layer (Phase 12, off by default) |

---

## Acknowledgements

The persistent wiki at the heart of this system — where the LLM writes and
cross-references its own markdown knowledge base that compounds over time — is
directly inspired by Andrej Karpathy's **LLM-wiki** idea. Rather than
re-deriving answers from raw chunks on every query, the system accumulates
compiled synthesis pages that grow more useful with each ingest and each
answered question. Credit for that core architectural insight goes to him.
