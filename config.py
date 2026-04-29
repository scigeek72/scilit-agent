"""
config.py — the only file users need to edit to change topics, sources,
LLM provider, and retrieval behaviour. All other modules import from here.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# .env loading — search priority-ordered locations
# ---------------------------------------------------------------------------

def _load_env() -> str:
    """Load the first .env file found in priority order. Return its path."""
    candidates = [
        Path(__file__).parent / ".env",           # project-local (highest priority)
        Path.home() / ".env",                     # user home
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path)
            return str(path)
    load_dotenv()   # shell environment fallback
    return ""


_ENV_PATH = _load_env()


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

class Config:

    # ── Topic ────────────────────────────────────────────────────────────
    TOPIC_NAME: str = "Transformer Models"
    SEARCH_QUERY: str = "attention mechanisms transformer models"

    # ── Sources ──────────────────────────────────────────────────────────
    ACTIVE_SOURCES: list[str] = [
        "arxiv",
        "biorxiv",
        "medrxiv",
        "pubmed",
        "semantic_scholar",
        "local_pdf",
    ]

    SOURCE_MAX_RESULTS: dict[str, int] = {
        "arxiv":            100,
        "biorxiv":           50,
        "medrxiv":           30,
        "pubmed":            50,
        "semantic_scholar":  50,
        "local_pdf":        999,
    }

    # ── LLM Provider ─────────────────────────────────────────────────────
    # "openai" | "anthropic" | "lmstudio"
    LLM_PROVIDER: str = "openai"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_CHAT_MODEL: str = "claude-sonnet-4-20250514"
    LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LM_STUDIO_MODEL: str = "local-model"

    # ── Embeddings ───────────────────────────────────────────────────────
    # "openai" | "local"
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ── Parser ───────────────────────────────────────────────────────────
    PARSER_AUTO_ROUTE: bool = True
    DEFAULT_PARSER: str = "grobid"
    GROBID_URL: str = "http://localhost:8070"
    MATH_THRESHOLD: float = 0.3

    # ── Retrieval ────────────────────────────────────────────────────────
    BM25_WEIGHT: float = 0.4
    SEMANTIC_WEIGHT: float = 0.6
    RETRIEVAL_K: int = 20
    RERANK_TOP_K: int = 5
    USE_HYDE: bool = True
    USE_HYBRID_SEARCH: bool = True
    USE_RERANKER: bool = True
    USE_MULTIHOP: bool = True

    # ── Wiki ─────────────────────────────────────────────────────────────
    WIKI_DIR: str = "wiki"
    WIKI_WRITE_BACK: bool = True
    WIKI_LINT_INTERVAL_DAYS: int = 7

    # ── Cache ────────────────────────────────────────────────────────────
    USE_PERSISTENT_CACHE: bool = True
    CACHE_MAX_SIZE: int = 500

    # ── Build ────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # ── Knowledge Graph (Phase 12 — off by default) ───────────────────────
    USE_KG: bool = False

    # ── API Keys (read from environment) ─────────────────────────────────
    @classmethod
    def openai_api_key(cls) -> str:
        return os.environ.get("OPENAI_API_KEY", "")

    @classmethod
    def anthropic_api_key(cls) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "")

    @classmethod
    def semantic_scholar_api_key(cls) -> str:
        return os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    @classmethod
    def ncbi_api_key(cls) -> str:
        return os.environ.get("NCBI_API_KEY", "")

    # ── Derived paths ─────────────────────────────────────────────────────
    @classmethod
    def topic_slug(cls) -> str:
        slug = cls.TOPIC_NAME.lower()
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
        return slug

    @classmethod
    def data_dir(cls) -> Path:
        return Path("data") / cls.topic_slug()

    @classmethod
    def raw_pdf_dir(cls) -> Path:
        return cls.data_dir() / "raw" / "pdfs"

    @classmethod
    def local_drop_dir(cls) -> Path:
        return cls.data_dir() / "raw" / "local_drop"

    @classmethod
    def parsed_dir(cls) -> Path:
        return cls.data_dir() / "parsed"

    @classmethod
    def vector_db_dir(cls) -> Path:
        return cls.data_dir() / "vector_db"

    @classmethod
    def bm25_index_dir(cls) -> Path:
        return cls.data_dir() / "bm25_index"

    @classmethod
    def cache_db_path(cls) -> Path:
        return cls.data_dir() / "cache" / "query_cache.db"

    @classmethod
    def metadata_path(cls) -> Path:
        return cls.data_dir() / "metadata" / "papers.json"

    @classmethod
    def wiki_dir(cls) -> Path:
        return Path(cls.WIKI_DIR) / cls.topic_slug()

    # ── Directory initialisation ──────────────────────────────────────────
    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all required directories if they do not already exist."""
        dirs = [
            cls.raw_pdf_dir(),
            cls.local_drop_dir(),
            cls.parsed_dir(),
            cls.vector_db_dir(),
            cls.bm25_index_dir(),
            cls.cache_db_path().parent,
            cls.metadata_path().parent,
            cls.wiki_dir() / "papers",
            cls.wiki_dir() / "concepts",
            cls.wiki_dir() / "methods",
            cls.wiki_dir() / "debates",
            cls.wiki_dir() / "authors",
            cls.wiki_dir() / "synthesis" / "query-answers",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ── Validation ────────────────────────────────────────────────────────
    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of configuration warnings (not fatal errors)."""
        warnings: list[str] = []
        if cls.LLM_PROVIDER == "openai" and not cls.openai_api_key():
            warnings.append("OPENAI_API_KEY not set but LLM_PROVIDER='openai'")
        if cls.LLM_PROVIDER == "anthropic" and not cls.anthropic_api_key():
            warnings.append("ANTHROPIC_API_KEY not set but LLM_PROVIDER='anthropic'")
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.openai_api_key():
            warnings.append("OPENAI_API_KEY not set but EMBEDDING_PROVIDER='openai'")
        return warnings

    # ── Status report ─────────────────────────────────────────────────────
    @classmethod
    def status(cls) -> str:
        lines = [
            f"Topic      : {cls.TOPIC_NAME}",
            f"Slug       : {cls.topic_slug()}",
            f"LLM        : {cls.LLM_PROVIDER} / {cls.OPENAI_CHAT_MODEL if cls.LLM_PROVIDER == 'openai' else cls.ANTHROPIC_CHAT_MODEL if cls.LLM_PROVIDER == 'anthropic' else cls.LM_STUDIO_MODEL}",
            f"Embeddings : {cls.EMBEDDING_PROVIDER} / {cls.EMBEDDING_MODEL if cls.EMBEDDING_PROVIDER == 'local' else cls.OPENAI_EMBEDDING_MODEL}",
            f"Sources    : {', '.join(cls.ACTIVE_SOURCES)}",
            f"Parser     : auto={cls.PARSER_AUTO_ROUTE}, default={cls.DEFAULT_PARSER}",
            f"Retrieval  : hybrid={cls.USE_HYBRID_SEARCH}, hyde={cls.USE_HYDE}, rerank={cls.USE_RERANKER}",
            f"Wiki       : {cls.WIKI_DIR}, write_back={cls.WIKI_WRITE_BACK}",
        ]
        warnings = cls.validate()
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)
