"""
retrieval/chunker.py — Split a ParsedPaper into overlapping text chunks.

Chunks are the unit stored in ChromaDB and the BM25 index.  Each chunk
carries metadata so search results can be traced back to the source paper
and section.

Chunk size is measured in words (tokens are approximated as whitespace-split
words).  Config.CHUNK_SIZE and Config.CHUNK_OVERLAP control the sizes.
"""

from __future__ import annotations

import re
from typing import TypedDict

from config import Config


class Chunk(TypedDict):
    chunk_id: str           # "{paper_id}_{index}" — globally unique
    text: str
    paper_id: str
    title: str
    authors: str            # pipe-joined for ChromaDB metadata compatibility
    year: int
    source: str
    chunk_index: int
    section_heading: str


def chunk_parsed_paper(parsed_paper: dict) -> list[Chunk]:
    """
    Split a ParsedPaper dict into overlapping Chunk dicts.

    Strategy:
      1. Abstract → one or more chunks (prefix with "Abstract: ")
      2. Each section → split into overlapping word-window chunks
      3. Chunks shorter than 20 words are merged with the previous chunk
         (avoids indexing stub headings with no useful content)
    """
    paper_id = parsed_paper.get("paper_id", "")
    title = parsed_paper.get("title", "")
    authors = " | ".join(parsed_paper.get("authors", []))
    year = parsed_paper.get("year", 0)
    source = parsed_paper.get("source", "")

    base_meta = dict(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        source=source,
    )

    raw_segments: list[tuple[str, str]] = []   # (heading, text)

    # Abstract
    abstract = parsed_paper.get("abstract", "").strip()
    if abstract:
        raw_segments.append(("Abstract", abstract))

    # Sections
    for sec in parsed_paper.get("sections", []):
        text = sec.get("text", "").strip()
        heading = sec.get("heading", "")
        if text:
            raw_segments.append((heading, text))

    # If paper is abstract-only and abstract was empty, nothing to chunk
    if not raw_segments:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0

    for heading, text in raw_segments:
        windows = _sliding_window(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        for window_text in windows:
            chunk_id = f"{paper_id}_{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=window_text,
                    section_heading=heading,
                    chunk_index=chunk_index,
                    **base_meta,
                )
            )
            chunk_index += 1

    return chunks


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping word windows.

    Returns at least one window even if text is shorter than size.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= size:
        return [text]

    step = max(1, size - overlap)
    windows: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        windows.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return windows
