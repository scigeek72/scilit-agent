"""
retrieval/query_cache.py — SQLite persistent query cache.

Caches (query, answer, sources) so repeated queries return instantly
without re-running the full retrieval + synthesis pipeline.

Cache key: SHA-256 of the normalised query string.
TTL: entries older than CACHE_TTL_DAYS are evicted on access.

Usage:
    cache = get_query_cache()
    hit = cache.get("What is attention?")
    if hit:
        return hit["answer"]
    ...
    cache.put("What is attention?", answer, sources)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)

# Entries older than this are considered stale and evicted
CACHE_TTL_DAYS: int = 30

# Singleton
_cache_instance: Optional["QueryCache"] = None


class QueryCache:
    """
    SQLite-backed query cache.

    Thread-safe for single-process use (SQLite WAL mode).
    Each entry stores: query_hash, query_text, answer, sources_json,
    confidence, date_cached.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[dict]:
        """
        Look up a query in the cache.

        Returns {answer, sources, confidence, date_cached} or None on miss.
        Evicts the entry if it is older than CACHE_TTL_DAYS.
        """
        key = self._hash(query)
        cutoff = (date.today() - timedelta(days=CACHE_TTL_DAYS)).isoformat()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT answer, sources_json, confidence, date_cached "
                "FROM query_cache WHERE query_hash = ? AND date_cached >= ?",
                (key, cutoff),
            ).fetchone()

        if row is None:
            return None

        try:
            sources = json.loads(row[1]) if row[1] else []
        except json.JSONDecodeError:
            sources = []

        return {
            "answer":      row[0],
            "sources":     sources,
            "confidence":  float(row[2]) if row[2] is not None else 0.5,
            "date_cached": row[3],
        }

    def put(
        self,
        query: str,
        answer: str,
        sources: list[dict],
        confidence: float = 0.5,
    ) -> None:
        """
        Store a query-answer pair. Overwrites any existing entry for the query.
        Evicts oldest entries if the cache exceeds Config.CACHE_MAX_SIZE.
        """
        key  = self._hash(query)
        today = date.today().isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_cache "
                "(query_hash, query_text, answer, sources_json, confidence, date_cached) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, query[:500], answer, json.dumps(sources), confidence, today),
            )
            # Evict oldest entries beyond max size
            max_size = Config.CACHE_MAX_SIZE
            conn.execute(
                "DELETE FROM query_cache WHERE query_hash IN ("
                "  SELECT query_hash FROM query_cache "
                "  ORDER BY date_cached ASC "
                "  LIMIT MAX(0, (SELECT COUNT(*) FROM query_cache) - ?)"
                ")",
                (max_size,),
            )
        logger.debug("Cache PUT: %s…", query[:60])

    def invalidate(self, query: str) -> None:
        """Remove a specific query from the cache."""
        key = self._hash(query)
        with self._connect() as conn:
            conn.execute("DELETE FROM query_cache WHERE query_hash = ?", (key,))

    def clear(self) -> None:
        """Remove all cache entries."""
        with self._connect() as conn:
            conn.execute("DELETE FROM query_cache")
        logger.info("Query cache cleared")

    def size(self) -> int:
        """Return the number of entries currently in the cache."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash  TEXT PRIMARY KEY,
                    query_text  TEXT,
                    answer      TEXT,
                    sources_json TEXT,
                    confidence  REAL,
                    date_cached TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_date ON query_cache(date_cached)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _hash(query: str) -> str:
        normalised = " ".join(query.lower().split())
        return hashlib.sha256(normalised.encode()).hexdigest()


def get_query_cache() -> QueryCache:
    """Return the singleton QueryCache, creating it if needed."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache(Config.cache_db_path())
    return _cache_instance
