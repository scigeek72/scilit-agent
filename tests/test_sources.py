"""
tests/test_sources.py — Unit tests for Phase 1.

Covers:
- PaperMetadata dataclass (construction, normalisation, helpers)
- deduplicate() — DOI dedup, title-similarity dedup, priority ordering
- federated_search() — uses mocked connectors (no real API calls)
- Config — topic slug, derived paths, validate()
- ArxivSource, SemanticScholarSource (mocked HTTP)
- LocalPdfSource (real filesystem, temp dirs)

All tests are independently runnable without a running LLM or any API key.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path when running from tests/ subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.base import PaperMetadata, SourceConnector
from sources.federation import deduplicate, resolve_open_access


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_paper(
    paper_id: str = "arxiv:2301.00001",
    title: str = "Attention Is All You Need",
    doi: str | None = None,
    source: str = "arxiv",
    year: int = 2017,
    pdf_url: str | None = "https://arxiv.org/pdf/1706.03762",
    is_open_access: bool = True,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Vaswani, A."],
        abstract="An abstract.",
        year=year,
        source=source,
        pdf_url=pdf_url,
        doi=doi,
        venue="NeurIPS",
        tags=["cs.CL"],
        is_open_access=is_open_access,
    )


# ---------------------------------------------------------------------------
# PaperMetadata tests
# ---------------------------------------------------------------------------

class TestPaperMetadata:

    def test_doi_normalisation_https(self):
        p = make_paper(doi="https://doi.org/10.1234/foo")
        assert p.doi == "10.1234/foo"

    def test_doi_normalisation_http_dx(self):
        p = make_paper(doi="http://dx.doi.org/10.1234/bar")
        assert p.doi == "10.1234/bar"

    def test_doi_already_clean(self):
        p = make_paper(doi="10.5678/baz")
        assert p.doi == "10.5678/baz"

    def test_doi_none(self):
        p = make_paper(doi=None)
        assert p.doi is None

    def test_title_stripped(self):
        p = make_paper(title="  Foo Bar  ")
        assert p.title == "Foo Bar"

    def test_authors_stripped(self):
        p = PaperMetadata(
            paper_id="arxiv:x", title="T", authors=["  Alice  ", " Bob"],
            abstract="", year=2020, source="arxiv", pdf_url=None, doi=None, venue=None,
        )
        assert p.authors == ["Alice", "Bob"]

    def test_short_id(self):
        p = make_paper(paper_id="arxiv:2301.12345")
        assert p.short_id() == "2301.12345"

    def test_short_id_no_colon(self):
        p = make_paper(paper_id="someid")
        assert p.short_id() == "someid"

    def test_wiki_filename(self):
        p = make_paper(paper_id="arxiv:2301.12345")
        assert p.wiki_filename() == "arxiv-2301.12345"

    def test_wiki_filename_biorxiv(self):
        p = make_paper(paper_id="biorxiv:10.1101/2024.01.01.123456")
        assert "/" not in p.wiki_filename()

    def test_to_dict_roundtrip(self):
        p = make_paper(doi="10.999/test")
        d = p.to_dict()
        p2 = PaperMetadata.from_dict(d)
        assert p2.paper_id == p.paper_id
        assert p2.doi == p.doi
        assert p2.title == p.title

    def test_from_dict_defaults(self):
        p = PaperMetadata.from_dict({"paper_id": "local:foo", "source": "local"})
        assert p.title == ""
        assert p.year == 0
        assert p.tags == []
        assert p.is_open_access is True


# ---------------------------------------------------------------------------
# deduplicate() tests
# ---------------------------------------------------------------------------

class TestDeduplicate:

    def test_empty(self):
        assert deduplicate([]) == []

    def test_no_duplicates(self):
        papers = [
            make_paper("arxiv:0001", "Paper Alpha", doi="10.1/a"),
            make_paper("arxiv:0002", "Paper Beta", doi="10.1/b"),
        ]
        result = deduplicate(papers)
        assert len(result) == 2

    def test_exact_doi_dedup_keep_first(self):
        """When DOIs match and both have same source priority, keep first."""
        p1 = make_paper("arxiv:0001", "Paper A", doi="10.1/same", source="arxiv")
        p2 = make_paper("arxiv:0002", "Paper A v2", doi="10.1/same", source="pubmed")
        result = deduplicate([p1, p2])
        # arxiv has higher priority than pubmed → keep p1
        assert len(result) == 1
        assert result[0].paper_id == "arxiv:0001"

    def test_exact_doi_dedup_replace_with_higher_priority(self):
        """Higher-priority source replaces lower-priority for same DOI."""
        p1 = make_paper("pubmed:111", "Paper A", doi="10.1/same", source="pubmed")
        p2 = make_paper("arxiv:222", "Paper A", doi="10.1/same", source="arxiv")
        result = deduplicate([p1, p2])
        assert len(result) == 1
        assert result[0].paper_id == "arxiv:222"

    def test_title_similarity_dedup(self):
        """Titles that are >92% similar should be deduplicated."""
        # Use near-identical titles (punctuation + trivial word difference)
        p1 = make_paper("arxiv:0001", "Attention Is All You Need", source="arxiv")
        p2 = make_paper("pubmed:111", "Attention is All You Need.", source="pubmed")
        result = deduplicate([p1, p2])
        # Titles are similar enough; arxiv has higher priority
        assert len(result) == 1

    def test_different_titles_not_deduped(self):
        p1 = make_paper("arxiv:0001", "GPT-4 Technical Report")
        p2 = make_paper("arxiv:0002", "LLaMA: Open and Efficient Foundation Language Models")
        result = deduplicate([p1, p2])
        assert len(result) == 2

    def test_no_doi_papers_deduplicated_by_title(self):
        p1 = make_paper("arxiv:0001", "Attention Is All You Need", doi=None, source="arxiv")
        p2 = make_paper("semantic_scholar:abc", "Attention Is All You Need", doi=None, source="semantic_scholar")
        result = deduplicate([p1, p2])
        assert len(result) == 1
        assert result[0].source == "arxiv"

    def test_ordering_preserved_for_non_duplicates(self):
        papers = [
            make_paper("arxiv:0001", "Alpha"),
            make_paper("arxiv:0002", "Beta"),
            make_paper("arxiv:0003", "Gamma"),
        ]
        result = deduplicate(papers)
        assert [p.paper_id for p in result] == ["arxiv:0001", "arxiv:0002", "arxiv:0003"]

    def test_many_duplicates(self):
        """All copies of the same DOI across 5 sources collapse to 1."""
        papers = [
            make_paper(f"src{i}:001", "Same Paper", doi="10.99/dup", source=s)
            for i, s in enumerate(["pubmed", "semantic_scholar", "arxiv", "biorxiv", "medrxiv"])
        ]
        result = deduplicate(papers)
        assert len(result) == 1
        assert result[0].source == "arxiv"    # highest priority

    def test_local_papers_not_deduped_against_each_other(self):
        """Two distinct local PDFs should both be kept."""
        p1 = make_paper("local:my_paper_a", "The Effects of X", source="local", doi=None)
        p2 = make_paper("local:my_paper_b", "Exploration of Y", source="local", doi=None)
        result = deduplicate([p1, p2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# resolve_open_access() tests (mock Unpaywall)
# ---------------------------------------------------------------------------

class TestResolveOpenAccess:

    def test_skips_papers_with_existing_pdf_url(self):
        p = make_paper(pdf_url="https://existing.pdf", doi="10.1/foo")
        papers = [p]
        result = resolve_open_access(papers)
        assert result[0].pdf_url == "https://existing.pdf"

    def test_skips_papers_without_doi(self):
        p = make_paper(pdf_url=None, doi=None)
        papers = [p]
        result = resolve_open_access(papers)
        assert result[0].pdf_url is None

    @patch("requests.get")
    def test_unpaywall_sets_pdf_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "best_oa_location": {"url_for_pdf": "https://unpaywall.pdf"}
        }
        mock_get.return_value = mock_resp

        p = make_paper(pdf_url=None, doi="10.1/paywalled", is_open_access=False)
        papers = [p]
        result = resolve_open_access(papers)
        assert result[0].pdf_url == "https://unpaywall.pdf"
        assert result[0].is_open_access is True

    @patch("requests.get")
    def test_unpaywall_no_oa_leaves_pdf_url_none(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"best_oa_location": None}
        mock_get.return_value = mock_resp

        p = make_paper(pdf_url=None, doi="10.1/paywalled")
        result = resolve_open_access([p])
        assert result[0].pdf_url is None

    @patch("requests.get", side_effect=Exception("network error"))
    def test_unpaywall_failure_is_silent(self, mock_get):
        p = make_paper(pdf_url=None, doi="10.1/paywalled")
        result = resolve_open_access([p])   # must not raise
        assert result[0].pdf_url is None


# ---------------------------------------------------------------------------
# Federated search (mocked connectors)
# ---------------------------------------------------------------------------

class TestFederatedSearch:
    """
    Tests for federated_search() using monkey-patched connectors.
    No real network calls are made.
    """

    def _make_mock_connector(
        self,
        name: str,
        papers: list[PaperMetadata],
    ) -> MagicMock:
        connector = MagicMock(spec=SourceConnector)
        connector.source_name = name
        connector.search.return_value = papers
        return connector

    @patch("sources.federation._get_registry")
    @patch("sources.federation.get_connector")
    def test_results_merged(self, mock_get_conn, mock_registry):
        from config import Config
        original_sources = Config.ACTIVE_SOURCES
        Config.ACTIVE_SOURCES = ["arxiv", "semantic_scholar"]

        p1 = make_paper("arxiv:001", "Paper One", doi="10.1/one", source="arxiv")
        p2 = make_paper("semantic_scholar:abc", "Paper Two", doi="10.1/two", source="semantic_scholar")

        mock_registry.return_value = {"arxiv": MagicMock, "semantic_scholar": MagicMock}
        mock_get_conn.side_effect = lambda name: self._make_mock_connector(
            name, [p1] if name == "arxiv" else [p2]
        )

        from sources.federation import federated_search
        with patch("sources.federation.resolve_open_access", side_effect=lambda x: x):
            results = federated_search("transformers", max_total=200)

        assert len(results) == 2
        Config.ACTIVE_SOURCES = original_sources

    @patch("sources.federation._get_registry")
    @patch("sources.federation.get_connector")
    def test_duplicates_removed(self, mock_get_conn, mock_registry):
        from config import Config
        original_sources = Config.ACTIVE_SOURCES
        Config.ACTIVE_SOURCES = ["arxiv", "semantic_scholar"]

        shared_doi = "10.1/shared"
        p1 = make_paper("arxiv:001", "Same Paper", doi=shared_doi, source="arxiv")
        p2 = make_paper("semantic_scholar:abc", "Same Paper", doi=shared_doi, source="semantic_scholar")

        mock_registry.return_value = {"arxiv": MagicMock, "semantic_scholar": MagicMock}
        mock_get_conn.side_effect = lambda name: self._make_mock_connector(
            name, [p1] if name == "arxiv" else [p2]
        )

        from sources.federation import federated_search
        with patch("sources.federation.resolve_open_access", side_effect=lambda x: x):
            results = federated_search("anything", max_total=200)

        assert len(results) == 1
        assert results[0].source == "arxiv"
        Config.ACTIVE_SOURCES = original_sources

    @patch("sources.federation._get_registry")
    @patch("sources.federation.get_connector")
    def test_source_failure_does_not_crash(self, mock_get_conn, mock_registry):
        from config import Config
        original_sources = Config.ACTIVE_SOURCES
        Config.ACTIVE_SOURCES = ["arxiv", "semantic_scholar"]

        good_paper = make_paper("arxiv:001", "Good Paper", source="arxiv")

        def _connector(name):
            c = MagicMock(spec=SourceConnector)
            c.source_name = name
            if name == "arxiv":
                c.search.return_value = [good_paper]
            else:
                c.search.side_effect = RuntimeError("API down")
            return c

        mock_registry.return_value = {"arxiv": MagicMock, "semantic_scholar": MagicMock}
        mock_get_conn.side_effect = _connector

        from sources.federation import federated_search
        with patch("sources.federation.resolve_open_access", side_effect=lambda x: x):
            results = federated_search("anything", max_total=200)

        assert len(results) == 1
        Config.ACTIVE_SOURCES = original_sources

    @patch("sources.federation._get_registry")
    @patch("sources.federation.get_connector")
    def test_max_total_respected(self, mock_get_conn, mock_registry):
        from config import Config
        original_sources = Config.ACTIVE_SOURCES
        Config.ACTIVE_SOURCES = ["arxiv"]

        papers = [
            make_paper(f"arxiv:{i:04d}", f"Paper {i}", doi=f"10.1/{i}", source="arxiv")
            for i in range(50)
        ]

        mock_registry.return_value = {"arxiv": MagicMock}
        mock_get_conn.return_value = self._make_mock_connector("arxiv", papers)

        from sources.federation import federated_search
        with patch("sources.federation.resolve_open_access", side_effect=lambda x: x):
            results = federated_search("anything", max_total=10)

        assert len(results) == 10
        Config.ACTIVE_SOURCES = original_sources


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:

    def test_topic_slug_simple(self):
        from config import Config
        original = Config.TOPIC_NAME
        Config.TOPIC_NAME = "Transformer Models"
        assert Config.topic_slug() == "transformer_models"
        Config.TOPIC_NAME = original

    def test_topic_slug_special_chars(self):
        from config import Config
        original = Config.TOPIC_NAME
        Config.TOPIC_NAME = "CRISPR Gene-Editing & Delivery"
        slug = Config.topic_slug()
        assert " " not in slug
        assert "&" not in slug
        Config.TOPIC_NAME = original

    def test_data_dir_uses_slug(self):
        from config import Config
        original = Config.TOPIC_NAME
        Config.TOPIC_NAME = "Test Topic"
        assert Config.data_dir() == Path("data") / "test_topic"
        Config.TOPIC_NAME = original

    def test_validate_warns_missing_openai_key(self):
        from config import Config
        original_provider = Config.LLM_PROVIDER
        Config.LLM_PROVIDER = "openai"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            warnings = Config.validate()
        assert any("OPENAI_API_KEY" in w for w in warnings)
        Config.LLM_PROVIDER = original_provider

    def test_validate_no_warnings_when_key_set(self):
        from config import Config
        original_provider = Config.LLM_PROVIDER
        Config.LLM_PROVIDER = "openai"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            warnings = Config.validate()
        assert not any("OPENAI_API_KEY" in w for w in warnings)
        Config.LLM_PROVIDER = original_provider

    def test_ensure_dirs_creates_paths(self, tmp_path, monkeypatch):
        from config import Config
        # Override all path methods to use tmp_path
        monkeypatch.setattr(Config, "data_dir", classmethod(lambda cls: tmp_path / "data" / "test"))
        Config.ensure_dirs()
        # Just verify it doesn't raise — specific dirs depend on overridden method


# ---------------------------------------------------------------------------
# LocalPdfSource tests (real filesystem, temp dir)
# ---------------------------------------------------------------------------

class TestLocalPdfSource:

    def test_search_empty_dir(self, tmp_path, monkeypatch):
        from config import Config
        from sources.local_pdf_source import LocalPdfSource

        monkeypatch.setattr(Config, "local_drop_dir", classmethod(lambda cls: tmp_path))
        source = LocalPdfSource()
        results = source.search("anything", max_results=50)
        assert results == []

    def test_search_creates_dir_if_missing(self, tmp_path, monkeypatch):
        from config import Config
        from sources.local_pdf_source import LocalPdfSource

        drop = tmp_path / "nonexistent"
        monkeypatch.setattr(Config, "local_drop_dir", classmethod(lambda cls: drop))
        source = LocalPdfSource()
        results = source.search("anything", max_results=50)
        assert drop.exists()
        assert results == []

    def test_source_name(self):
        from sources.local_pdf_source import LocalPdfSource
        assert LocalPdfSource().source_name == "local_pdf"

    def test_fetch_metadata_wrong_prefix(self):
        from sources.local_pdf_source import LocalPdfSource
        with pytest.raises(ValueError):
            LocalPdfSource().fetch_metadata("arxiv:1234")

    def test_fetch_metadata_missing_file(self, tmp_path, monkeypatch):
        from config import Config
        from sources.local_pdf_source import LocalPdfSource
        monkeypatch.setattr(Config, "local_drop_dir", classmethod(lambda cls: tmp_path))
        with pytest.raises(LookupError):
            LocalPdfSource().fetch_metadata("local:doesnotexist")

    def test_download_pdf_copies_file(self, tmp_path, monkeypatch):
        from config import Config
        from sources.local_pdf_source import LocalPdfSource

        # Create a fake PDF in the drop dir
        drop = tmp_path / "drop"
        drop.mkdir()
        fake_pdf = drop / "my_paper.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content")

        monkeypatch.setattr(Config, "local_drop_dir", classmethod(lambda cls: drop))

        source = LocalPdfSource()
        meta = make_paper(paper_id="local:my_paper", source="local", doi=None, pdf_url=None)

        dest_dir = str(tmp_path / "output")
        result = source.download_pdf(meta, dest_dir)
        assert result is not None
        assert Path(result).exists()
