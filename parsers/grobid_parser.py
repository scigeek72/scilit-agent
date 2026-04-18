"""
parsers/grobid_parser.py — Grobid TEI XML → ParsedPaper.

Grobid runs as a local Docker container:
  docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0

Sends the PDF to POST /api/processFulltextDocument and parses the
TEI XML response with lxml.

Falls back gracefully: if the service is down, is_available() returns
False and the router will use PyMuPDF instead.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import requests
from lxml import etree

from config import Config
from parsers.base import (
    FigureDict,
    ParsedPaper,
    Parser,
    ReferenceDict,
    SectionDict,
    TableDict,
    empty_parsed_paper,
)
from parsers.pymupdf_parser import _compute_math_fraction

logger = logging.getLogger(__name__)

# TEI namespace
_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}


def _t(tag: str) -> str:
    """Return fully-qualified TEI tag name."""
    return f"{{{_TEI_NS}}}{tag}"


class GrobidParser(Parser):

    _HEALTH_ENDPOINT = "/api/isalive"
    _PROCESS_ENDPOINT = "/api/processFulltextDocument"
    _TIMEOUT_HEALTH = 3
    _TIMEOUT_PROCESS = 120

    @property
    def parser_name(self) -> str:
        return "grobid"

    def is_available(self) -> bool:
        """Ping the Grobid health endpoint."""
        try:
            resp = requests.get(
                Config.GROBID_URL.rstrip("/") + self._HEALTH_ENDPOINT,
                timeout=self._TIMEOUT_HEALTH,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str, metadata: dict) -> ParsedPaper:
        result = empty_parsed_paper(
            paper_id=metadata.get("paper_id", ""),
            title=metadata.get("title", ""),
            authors=metadata.get("authors", []),
            abstract=metadata.get("abstract", ""),
            year=metadata.get("year", 0),
            source=metadata.get("source", ""),
            parser_used="grobid",
            is_abstract_only=metadata.get("is_abstract_only", False),
        )

        if result["is_abstract_only"] or not pdf_path:
            return result

        tei_xml = self._call_grobid(pdf_path)
        if tei_xml is None:
            logger.warning("Grobid returned no XML for %s", pdf_path)
            return result

        try:
            root = etree.fromstring(tei_xml.encode("utf-8") if isinstance(tei_xml, str) else tei_xml)
        except etree.XMLSyntaxError as exc:
            logger.warning("Could not parse Grobid TEI XML: %s", exc)
            return result

        result.update(self._parse_tei(root, metadata))
        return result

    # ------------------------------------------------------------------
    # Grobid API call
    # ------------------------------------------------------------------

    def _call_grobid(self, pdf_path: str) -> str | None:
        url = Config.GROBID_URL.rstrip("/") + self._PROCESS_ENDPOINT
        try:
            with open(pdf_path, "rb") as fh:
                resp = requests.post(
                    url,
                    files={"input": (Path(pdf_path).name, fh, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "consolidateCitations": "0",  # faster; citations still extracted
                        "includeRawCitations": "0",
                        "includeRawAffiliations": "0",
                        "teiCoordinates": "0",
                    },
                    timeout=self._TIMEOUT_PROCESS,
                )
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.warning("Grobid API call failed for %s: %s", pdf_path, exc)
            return None

    # ------------------------------------------------------------------
    # TEI XML → ParsedPaper fields
    # ------------------------------------------------------------------

    def _parse_tei(self, root: etree._Element, metadata: dict) -> dict:
        header = root.find(f".//{_t('teiHeader')}")
        body = root.find(f".//{_t('body')}")
        back = root.find(f".//{_t('back')}")

        out: dict = {}

        # Title — prefer TEI over metadata fallback
        title_el = root.find(f".//{_t('titleStmt')}/{_t('title')}")
        if title_el is not None and title_el.text:
            out["title"] = title_el.text.strip()

        # Authors
        authors: list[str] = []
        for author_el in root.findall(f".//{_t('author')}"):
            persname = author_el.find(f".//{_t('persName')}")
            if persname is not None:
                forename = persname.findtext(f"{_t('forename')}", default="").strip()
                surname = persname.findtext(f"{_t('surname')}", default="").strip()
                name = f"{surname}, {forename}".strip(", ")
                if name:
                    authors.append(name)
        if authors:
            out["authors"] = authors

        # Abstract
        abstract_el = root.find(f".//{_t('abstract')}")
        if abstract_el is not None:
            out["abstract"] = _inner_text(abstract_el).strip()

        # Year — build XPath with attribute predicate outside f-string
        # to avoid Python 3.12 f-string bracket parsing ambiguity
        _date_tag = _t("date")
        date_el = root.find(f".//{_date_tag}[@when]")
        if date_el is None:
            date_el = root.find(f".//{_date_tag}")
        if date_el is not None:
            when = date_el.get("when", "")
            m = re.search(r"\b(\d{4})\b", when)
            if m:
                out["year"] = int(m.group(1))

        # Sections
        sections: list[SectionDict] = []
        if body is not None:
            for div in body.findall(f"{_t('div')}"):
                head = div.find(f"{_t('head')}")
                heading = head.text.strip() if head is not None and head.text else ""
                # Determine level from @n attribute
                n_attr = head.get("n", "") if head is not None else ""
                level = len(n_attr.split(".")) if n_attr else 1
                text_parts: list[str] = []
                for p in div.findall(f".//{_t('p')}"):
                    text_parts.append(_inner_text(p))
                sections.append(
                    SectionDict(
                        heading=heading,
                        text="\n".join(text_parts).strip(),
                        level=level,
                    )
                )
        out["sections"] = sections

        # Figures
        figures: list[FigureDict] = []
        for fig in root.findall(f".//{_t('figure')}"):
            if fig.get("type") == "table":
                continue
            label = fig.findtext(f"{_t('label')}", default="").strip()
            desc = fig.find(f"{_t('figDesc')}")
            caption = _inner_text(desc).strip() if desc is not None else ""
            if caption or label:
                figures.append(FigureDict(label=label, caption=caption))
        out["figures"] = figures

        # Tables
        tables: list[TableDict] = []
        for fig in root.findall(f".//{_t('figure')}[@type='table']"):
            label = fig.findtext(f"{_t('label')}", default="").strip()
            head = fig.find(f"{_t('head')}")
            caption = _inner_text(head).strip() if head is not None else ""
            if caption or label:
                tables.append(TableDict(label=label, caption=caption))
        out["tables"] = tables

        # References
        references: list[ReferenceDict] = []
        if back is not None:
            for i, bib in enumerate(
                back.findall(f".//{_t('biblStruct')}"), start=1
            ):
                _title_tag = _t("title")
                ref_title_el = bib.find(f".//{_title_tag}[@level='a']")
                if ref_title_el is None:
                    ref_title_el = bib.find(f".//{_title_tag}")
                ref_title = (
                    ref_title_el.text.strip()
                    if ref_title_el is not None and ref_title_el.text
                    else ""
                )
                ref_authors: list[str] = []
                for a in bib.findall(f".//{_t('author')}"):
                    pn = a.find(f".//{_t('persName')}")
                    if pn is not None:
                        fn = pn.findtext(f"{_t('forename')}", default="").strip()
                        sn = pn.findtext(f"{_t('surname')}", default="").strip()
                        name = f"{sn}, {fn}".strip(", ")
                        if name:
                            ref_authors.append(name)
                date = bib.find(f".//{_t('date')}")
                ref_year = 0
                if date is not None:
                    when = date.get("when", "")
                    m = re.search(r"\b(\d{4})\b", when)
                    if m:
                        ref_year = int(m.group(1))

                _xml_id_attr = "{http://www.w3.org/XML/1998/namespace}id"
                xml_id = bib.get(_xml_id_attr) or f"b{i}"
                references.append(
                    ReferenceDict(
                        ref_id=xml_id or f"b{i}",
                        title=ref_title,
                        authors=ref_authors,
                        year=ref_year,
                    )
                )
                if len(references) >= 150:
                    break
        out["references"] = references

        # Math fraction — estimate from full TEI text
        full_text = _inner_text(root)
        out["math_fraction"] = _compute_math_fraction(full_text)
        out["equations"] = []    # Grobid doesn't extract LaTeX equations
        out["parser_used"] = "grobid"

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inner_text(element: etree._Element) -> str:
    """Return all inner text of an element, joining tail text too."""
    return "".join(element.itertext())
