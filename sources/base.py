"""
sources/base.py — SourceConnector ABC and PaperMetadata dataclass.

Every source connector implements SourceConnector. The rest of the system
only ever calls these methods — it never imports a specific connector directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PaperMetadata:
    """
    Source-agnostic representation of a single paper's metadata.

    paper_id uses source-namespaced format:
        "arxiv:2301.12345"
        "pubmed:38291847"
        "biorxiv:10.1101/2024.01.01.123456"
        "semantic_scholar:abc123"
        "local:{filename_stem}"
    """
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    source: str             # "arxiv" | "pubmed" | "biorxiv" | "medrxiv" |
                            # "semantic_scholar" | "local"
    pdf_url: str | None     # None if paywalled or unavailable
    doi: str | None
    venue: str | None       # journal name or arXiv category
    tags: list[str] = field(default_factory=list)
    is_open_access: bool = True

    def __post_init__(self) -> None:
        # Normalise: strip whitespace from string fields
        self.title = self.title.strip()
        self.abstract = self.abstract.strip()
        self.authors = [a.strip() for a in self.authors]
        if self.doi:
            self.doi = self.doi.strip().lower()
            # Normalise "https://doi.org/10.xxxx" → "10.xxxx"
            if self.doi.startswith("https://doi.org/"):
                self.doi = self.doi[len("https://doi.org/"):]
            elif self.doi.startswith("http://dx.doi.org/"):
                self.doi = self.doi[len("http://dx.doi.org/"):]

    # ------------------------------------------------------------------
    # Convenience helpers (no business logic — pure data access)
    # ------------------------------------------------------------------

    def short_id(self) -> str:
        """Return the ID without the source prefix, e.g. '2301.12345'."""
        if ":" in self.paper_id:
            return self.paper_id.split(":", 1)[1]
        return self.paper_id

    def wiki_filename(self) -> str:
        """
        Return a filesystem-safe wiki filename stem.
        e.g. 'arxiv-2301.12345', 'pubmed-38291847'
        """
        return self.paper_id.replace(":", "-").replace("/", "_")

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "source": self.source,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "venue": self.venue,
            "tags": self.tags,
            "is_open_access": self.is_open_access,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperMetadata":
        return cls(
            paper_id=data["paper_id"],
            title=data.get("title", ""),
            authors=data.get("authors", []),
            abstract=data.get("abstract", ""),
            year=data.get("year", 0),
            source=data.get("source", ""),
            pdf_url=data.get("pdf_url"),
            doi=data.get("doi"),
            venue=data.get("venue"),
            tags=data.get("tags", []),
            is_open_access=data.get("is_open_access", True),
        )


class SourceConnector(ABC):
    """
    Abstract base class for all source connectors.

    Downstream code only ever calls these three methods.
    No source-specific logic leaks beyond this interface.
    """

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        """
        Search this source for papers matching the plain-text query.
        Never apply source-specific query syntax here — keep it plain text.
        Return an empty list (never raise) on API failure.
        """

    @abstractmethod
    def fetch_metadata(self, paper_id: str) -> PaperMetadata:
        """
        Fetch full metadata for a known paper_id (source-namespaced).
        Raise ValueError if the paper_id is not owned by this source.
        Raise LookupError if the paper is not found.
        """

    @abstractmethod
    def download_pdf(self, metadata: PaperMetadata, output_dir: str) -> str | None:
        """
        Download the paper PDF to output_dir.
        Return the local file path on success, or None if unavailable.
        Never raise on download failure — log the reason and return None.
        """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique string identifier: 'arxiv', 'pubmed', 'biorxiv', etc."""
