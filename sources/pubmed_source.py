"""
sources/pubmed_source.py — PubMed source connector.

Uses Biopython's Entrez API.
- Metadata is always free.
- PDFs are available only for PubMed Central (PMC) open-access papers.
  For papers without a PMC ID, pdf_url=None and is_open_access=False.
- NCBI_API_KEY in .env raises the rate limit from 3 → 10 requests/sec.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from sources.base import PaperMetadata, SourceConnector
from config import Config

logger = logging.getLogger(__name__)

# Biopython Entrez is imported lazily so the module loads without biopython
# installed (other sources still work).
_ENTREZ_EMAIL = "scilit-agent@example.com"
_PMC_PDF_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _entrez():
    """Return the Entrez module, raising ImportError with a helpful message."""
    try:
        from Bio import Entrez
        return Entrez
    except ImportError as exc:
        raise ImportError(
            "biopython is required for PubMed. Install it with: "
            "pip install biopython"
        ) from exc


class PubMedSource(SourceConnector):

    _REQUEST_DELAY: float = 0.34   # ~3 req/s; drops to 0.1 with API key

    def __init__(self) -> None:
        self._api_key = Config.ncbi_api_key()
        if self._api_key:
            self._REQUEST_DELAY = 0.1

    @property
    def source_name(self) -> str:
        return "pubmed"

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int) -> list[PaperMetadata]:
        """Search PubMed for papers matching the plain-text query."""
        Entrez = _entrez()
        Entrez.email = _ENTREZ_EMAIL
        if self._api_key:
            Entrez.api_key = self._api_key

        results: list[PaperMetadata] = []
        try:
            # Step 1: esearch — get PMIDs
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
                usehistory="y",
            )
            search_record = Entrez.read(handle)
            handle.close()

            pmids = search_record.get("IdList", [])
            if not pmids:
                return []

            time.sleep(self._REQUEST_DELAY)

            # Step 2: efetch in batches of 200
            batch_size = 200
            for i in range(0, len(pmids), batch_size):
                batch = pmids[i : i + batch_size]
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch),
                    rettype="xml",
                    retmode="xml",
                )
                records = Entrez.read(handle)
                handle.close()

                for article in records.get("PubmedArticle", []):
                    try:
                        metadata = self._parse_article(article)
                        results.append(metadata)
                    except Exception as exc:
                        logger.debug("Could not parse PubMed article: %s", exc)

                time.sleep(self._REQUEST_DELAY)

        except Exception as exc:
            logger.warning("PubMed search failed: %s", exc)

        return results

    # ------------------------------------------------------------------
    # fetch_metadata
    # ------------------------------------------------------------------

    def fetch_metadata(self, paper_id: str) -> PaperMetadata:
        """Fetch metadata for 'pubmed:{pmid}'."""
        if not paper_id.startswith("pubmed:"):
            raise ValueError(f"paper_id '{paper_id}' does not belong to pubmed source")
        pmid = paper_id[len("pubmed:"):]

        Entrez = _entrez()
        Entrez.email = _ENTREZ_EMAIL
        if self._api_key:
            Entrez.api_key = self._api_key

        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        articles = records.get("PubmedArticle", [])
        if not articles:
            raise LookupError(f"PubMed article not found: {pmid}")
        return self._parse_article(articles[0])

    # ------------------------------------------------------------------
    # download_pdf
    # ------------------------------------------------------------------

    def download_pdf(self, metadata: PaperMetadata, output_dir: str) -> str | None:
        """
        Download PDF from PMC if available.
        Returns local path, or None if the paper is not in PMC or is paywalled.
        """
        if not metadata.pdf_url:
            logger.info(
                "No PMC PDF for %s — abstract-only indexing", metadata.paper_id
            )
            return None

        output_path = Path(output_dir) / f"{metadata.wiki_filename()}.pdf"
        if output_path.exists():
            return str(output_path)

        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            resp = requests.get(metadata.pdf_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(output_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
            return str(output_path)
        except Exception as exc:
            logger.warning("PMC PDF download failed for %s: %s", metadata.paper_id, exc)
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_article(self, article: dict) -> PaperMetadata:
        medline = article.get("MedlineCitation", {})
        pmid = str(medline.get("PMID", ""))
        article_data = medline.get("Article", {})

        # Title
        title = str(article_data.get("ArticleTitle", "")).strip()

        # Authors
        authors: list[str] = []
        author_list = article_data.get("AuthorList", [])
        for a in author_list:
            last = a.get("LastName", "")
            fore = a.get("ForeName", "") or a.get("Initials", "")
            name = f"{last}, {fore}".strip(", ")
            if name:
                authors.append(name)

        # Abstract
        abstract_obj = article_data.get("Abstract", {})
        abstract_texts = abstract_obj.get("AbstractText", [])
        if isinstance(abstract_texts, list):
            abstract = " ".join(str(t) for t in abstract_texts).strip()
        else:
            abstract = str(abstract_texts).strip()

        # Year
        journal = article_data.get("Journal", {})
        pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
        year = int(str(pub_date.get("Year", 0) or 0))
        if not year:
            # Fallback: ArticleDate
            article_dates = article_data.get("ArticleDate", [])
            for d in article_dates:
                y = int(str(d.get("Year", 0) or 0))
                if y:
                    year = y
                    break

        # DOI
        doi: str | None = None
        id_list = article.get("PubmedData", {}).get("ArticleIdList", [])
        pmc_id: str | None = None
        for aid in id_list:
            id_type = str(aid.attributes.get("IdType", ""))
            val = str(aid)
            if id_type == "doi":
                doi = val
            elif id_type == "pmc":
                pmc_id = val.replace("PMC", "")

        # PDF URL (PMC only)
        pdf_url: str | None = None
        is_oa = False
        if pmc_id:
            pdf_url = _PMC_PDF_BASE.format(pmc_id=f"PMC{pmc_id}")
            is_oa = True

        # Venue
        venue = journal.get("Title") or journal.get("ISOAbbreviation")

        # Tags — MeSH headings
        mesh_list = medline.get("MeshHeadingList", [])
        tags: list[str] = []
        for mesh in mesh_list:
            descriptor = mesh.get("DescriptorName")
            if descriptor:
                tags.append(str(descriptor))

        return PaperMetadata(
            paper_id=f"pubmed:{pmid}",
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            source="pubmed",
            pdf_url=pdf_url,
            doi=doi,
            venue=venue,
            tags=tags[:10],   # cap to avoid bloat
            is_open_access=is_oa,
        )
