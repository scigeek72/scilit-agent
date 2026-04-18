"""
tests/test_sources_phase2.py — Unit tests for Phase 2 source connectors.

Covers:
- PubMedSource  — mocked Entrez calls
- BiorxivSource — mocked REST API responses
- MedrxivSource — mocked REST API responses
- federation.py registry now includes all Phase 2 sources

All tests run without network access or API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _biorxiv_item(
    doi="10.1101/2024.01.01.123456",
    title="CRISPR Delivery Mechanisms",
    abstract="We study delivery of CRISPR.",
    authors="Smith, J.; Lee, K.",
    date="2024-01-01",
    version="1",
    category="genomics",
    server="biorxiv",
) -> dict:
    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "date": date,
        "version": version,
        "category": category,
    }


def _pubmed_article(
    pmid="38291847",
    title="BERT for NER in Clinical Text",
    abstract="An abstract about BERT.",
    authors=None,
    year="2024",
    doi="10.1234/test",
    pmc_id=None,
) -> dict:
    """Build a minimal structure mimicking Biopython's Entrez XML parse output."""
    if authors is None:
        authors = [{"LastName": "Smith", "ForeName": "John", "Initials": "J"}]

    class _AttrStr(str):
        """String with .attributes dict (mimics Biopython's StringElement)."""
        def __new__(cls, val, id_type=""):
            obj = super().__new__(cls, val)
            obj.attributes = {"IdType": id_type}
            return obj

    id_list = [_AttrStr(doi, "doi")]
    if pmc_id:
        id_list.append(_AttrStr(f"PMC{pmc_id}", "pmc"))

    return {
        "MedlineCitation": {
            "PMID": pmid,
            "Article": {
                "ArticleTitle": title,
                "Abstract": {"AbstractText": [abstract]},
                "AuthorList": authors,
                "Journal": {
                    "Title": "Nature",
                    "ISOAbbreviation": "Nature",
                    "JournalIssue": {
                        "PubDate": {"Year": year}
                    },
                },
                "ArticleDate": [],
            },
            "MeshHeadingList": [],
        },
        "PubmedData": {
            "ArticleIdList": id_list,
        },
    }


# ---------------------------------------------------------------------------
# BiorxivSource tests
# ---------------------------------------------------------------------------

class TestBiorxivSource:

    def _make_api_response(self, items, total=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "messages": [{"total": total or len(items), "status": "ok"}],
            "collection": items,
        }
        resp.raise_for_status = MagicMock()
        return resp

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_search_returns_matching_papers(self, mock_get):
        from sources.biorxiv_source import BiorxivSource

        item = _biorxiv_item(title="Attention mechanism in transformers")
        mock_get.return_value = self._make_api_response([item], total=1)

        source = BiorxivSource()
        results = source.search("attention transformer", max_results=10)

        assert len(results) == 1
        assert results[0].source == "biorxiv"
        assert results[0].paper_id == "biorxiv:10.1101/2024.01.01.123456"
        assert results[0].is_open_access is True

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_search_filters_non_matching(self, mock_get):
        from sources.biorxiv_source import BiorxivSource

        item = _biorxiv_item(title="Unrelated paper about geology", abstract="Rocks and stuff.")
        mock_get.return_value = self._make_api_response([item], total=1)

        source = BiorxivSource()
        results = source.search("attention transformer", max_results=10)
        assert results == []

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_search_api_failure_returns_empty(self, mock_get):
        from sources.biorxiv_source import BiorxivSource
        mock_get.side_effect = Exception("network error")

        source = BiorxivSource()
        results = source.search("anything", max_results=10)
        assert results == []

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_pdf_url_constructed_correctly(self, mock_get):
        from sources.biorxiv_source import BiorxivSource

        item = _biorxiv_item(doi="10.1101/2024.05.01.999999", version="2")
        mock_get.return_value = self._make_api_response([item], total=1)

        source = BiorxivSource()
        results = source.search("crispr delivery", max_results=10)

        assert results[0].pdf_url == "https://www.biorxiv.org/content/10.1101/2024.05.01.999999v2.full.pdf"

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_fetch_metadata(self, mock_get):
        from sources.biorxiv_source import BiorxivSource

        item = _biorxiv_item()
        mock_get.return_value = self._make_api_response([item])

        source = BiorxivSource()
        meta = source.fetch_metadata("biorxiv:10.1101/2024.01.01.123456")
        assert meta.title == "CRISPR Delivery Mechanisms"
        assert meta.doi == "10.1101/2024.01.01.123456"

    def test_fetch_metadata_wrong_prefix(self):
        from sources.biorxiv_source import BiorxivSource
        with pytest.raises(ValueError):
            BiorxivSource().fetch_metadata("arxiv:2301.12345")

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_fetch_metadata_not_found(self, mock_get):
        from sources.biorxiv_source import BiorxivSource
        mock_get.return_value = self._make_api_response([])
        with pytest.raises(LookupError):
            BiorxivSource().fetch_metadata("biorxiv:10.1101/notexist")

    def test_source_name(self):
        from sources.biorxiv_source import BiorxivSource
        assert BiorxivSource().source_name == "biorxiv"

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_year_parsed_from_date(self, mock_get):
        from sources.biorxiv_source import BiorxivSource

        item = _biorxiv_item(date="2023-07-15")
        mock_get.return_value = self._make_api_response([item], total=1)

        source = BiorxivSource()
        results = source.search("crispr delivery genomics", max_results=10)
        assert results[0].year == 2023

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_authors_parsed_semicolon_separated(self, mock_get):
        from sources.biorxiv_source import BiorxivSource

        item = _biorxiv_item(authors="Smith, J.; Lee, K.; Wang, X.")
        mock_get.return_value = self._make_api_response([item], total=1)

        source = BiorxivSource()
        results = source.search("crispr delivery genomics", max_results=10)
        assert results[0].authors == ["Smith, J.", "Lee, K.", "Wang, X."]

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_download_pdf_no_url(self, mock_get, tmp_path):
        from sources.biorxiv_source import BiorxivSource
        from sources.base import PaperMetadata

        meta = PaperMetadata(
            paper_id="biorxiv:10.1101/x", title="T", authors=[], abstract="",
            year=2024, source="biorxiv", pdf_url=None, doi=None, venue=None,
        )
        result = BiorxivSource().download_pdf(meta, str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# MedrxivSource tests
# ---------------------------------------------------------------------------

class TestMedrxivSource:

    def _make_api_response(self, items, total=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "messages": [{"total": total or len(items), "status": "ok"}],
            "collection": items,
        }
        resp.raise_for_status = MagicMock()
        return resp

    def test_source_name(self):
        from sources.medrxiv_source import MedrxivSource
        assert MedrxivSource().source_name == "medrxiv"

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_search_uses_medrxiv_server(self, mock_get):
        from sources.medrxiv_source import MedrxivSource

        item = _biorxiv_item(
            doi="10.1101/2024.02.01.000001",
            title="COVID vaccine efficacy",
            abstract="We study COVID vaccines.",
            category="infectious diseases",
        )
        mock_get.return_value = self._make_api_response([item], total=1)

        source = MedrxivSource()
        results = source.search("COVID vaccine", max_results=10)

        assert len(results) == 1
        assert results[0].source == "medrxiv"
        assert results[0].paper_id.startswith("medrxiv:")
        assert "www.medrxiv.org" in results[0].pdf_url

    @patch("sources._biorxiv_medrxiv_base.requests.get")
    def test_fetch_metadata_wrong_prefix(self, mock_get):
        from sources.medrxiv_source import MedrxivSource
        with pytest.raises(ValueError):
            MedrxivSource().fetch_metadata("biorxiv:10.1101/x")


# ---------------------------------------------------------------------------
# PubMedSource tests
# ---------------------------------------------------------------------------

class TestPubMedSource:

    def _mock_entrez(self, articles):
        """Return a mock Entrez module that yields the given article list."""
        entrez = MagicMock()

        # esearch returns pmids
        search_handle = MagicMock()
        entrez.esearch.return_value = search_handle
        entrez.read.side_effect = [
            # First call: esearch result
            {"IdList": [a["MedlineCitation"]["PMID"] for a in articles]},
            # Second call: efetch result
            {"PubmedArticle": articles},
        ]
        efetch_handle = MagicMock()
        entrez.efetch.return_value = efetch_handle

        return entrez

    def test_source_name(self):
        from sources.pubmed_source import PubMedSource
        assert PubMedSource().source_name == "pubmed"

    @patch("sources.pubmed_source._entrez")
    def test_search_returns_papers(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource

        article = _pubmed_article()
        mock_entrez_fn.return_value = self._mock_entrez([article])

        source = PubMedSource()
        results = source.search("BERT clinical NLP", max_results=10)

        assert len(results) == 1
        assert results[0].paper_id == "pubmed:38291847"
        assert results[0].source == "pubmed"
        assert results[0].title == "BERT for NER in Clinical Text"

    @patch("sources.pubmed_source._entrez")
    def test_search_no_results(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource

        entrez = MagicMock()
        entrez.esearch.return_value = MagicMock()
        entrez.read.return_value = {"IdList": []}
        mock_entrez_fn.return_value = entrez

        results = PubMedSource().search("nothing matches", max_results=10)
        assert results == []

    @patch("sources.pubmed_source._entrez")
    def test_search_api_failure_returns_empty(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource
        mock_entrez_fn.return_value = MagicMock(
            esearch=MagicMock(side_effect=Exception("API down"))
        )
        results = PubMedSource().search("anything", max_results=10)
        assert results == []

    @patch("sources.pubmed_source._entrez")
    def test_doi_extracted(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource

        article = _pubmed_article(doi="10.1038/s41586-024-00001-1")
        mock_entrez_fn.return_value = self._mock_entrez([article])

        results = PubMedSource().search("test", max_results=10)
        assert results[0].doi == "10.1038/s41586-024-00001-1"

    @patch("sources.pubmed_source._entrez")
    def test_pmc_paper_has_pdf_url_and_is_oa(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource

        article = _pubmed_article(pmc_id="9876543")
        mock_entrez_fn.return_value = self._mock_entrez([article])

        results = PubMedSource().search("test", max_results=10)
        assert results[0].is_open_access is True
        assert "PMC9876543" in results[0].pdf_url

    @patch("sources.pubmed_source._entrez")
    def test_non_pmc_paper_has_no_pdf_url(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource

        article = _pubmed_article(pmc_id=None)
        mock_entrez_fn.return_value = self._mock_entrez([article])

        results = PubMedSource().search("test", max_results=10)
        assert results[0].pdf_url is None
        assert results[0].is_open_access is False

    @patch("sources.pubmed_source._entrez")
    def test_year_extracted(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource

        article = _pubmed_article(year="2023")
        mock_entrez_fn.return_value = self._mock_entrez([article])

        results = PubMedSource().search("test", max_results=10)
        assert results[0].year == 2023

    @patch("sources.pubmed_source._entrez")
    def test_fetch_metadata_wrong_prefix(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource
        with pytest.raises(ValueError):
            PubMedSource().fetch_metadata("arxiv:2301.12345")

    @patch("sources.pubmed_source._entrez")
    def test_fetch_metadata_not_found(self, mock_entrez_fn):
        from sources.pubmed_source import PubMedSource
        entrez = MagicMock()
        entrez.efetch.return_value = MagicMock()
        entrez.read.return_value = {"PubmedArticle": []}
        mock_entrez_fn.return_value = entrez

        with pytest.raises(LookupError):
            PubMedSource().fetch_metadata("pubmed:99999999")


# ---------------------------------------------------------------------------
# Federation registry includes Phase 2 connectors
# ---------------------------------------------------------------------------

class TestFederationRegistryPhase2:

    def test_registry_includes_pubmed(self):
        from sources.federation import _build_registry
        registry = _build_registry()
        assert "pubmed" in registry

    def test_registry_includes_biorxiv(self):
        from sources.federation import _build_registry
        registry = _build_registry()
        assert "biorxiv" in registry

    def test_registry_includes_medrxiv(self):
        from sources.federation import _build_registry
        registry = _build_registry()
        assert "medrxiv" in registry

    def test_get_connector_pubmed(self):
        from sources.federation import get_connector
        from sources.pubmed_source import PubMedSource
        assert isinstance(get_connector("pubmed"), PubMedSource)

    def test_get_connector_biorxiv(self):
        from sources.federation import get_connector
        from sources.biorxiv_source import BiorxivSource
        assert isinstance(get_connector("biorxiv"), BiorxivSource)

    def test_get_connector_medrxiv(self):
        from sources.federation import get_connector
        from sources.medrxiv_source import MedrxivSource
        assert isinstance(get_connector("medrxiv"), MedrxivSource)
