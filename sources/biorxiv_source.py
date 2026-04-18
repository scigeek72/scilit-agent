"""
sources/biorxiv_source.py — bioRxiv source connector.

Uses the bioRxiv public REST API (no API key required).
All bioRxiv preprints are open access with a direct PDF URL.

API docs: https://api.biorxiv.org/
Endpoint used: /details/biorxiv/{interval}/{cursor}
Search is implemented via the /publisher endpoint for topic-based
queries, with a keyword-filter fallback for plain-text matching.
"""

from __future__ import annotations

import logging
import time

import requests

from sources.base import PaperMetadata, SourceConnector
from sources._biorxiv_medrxiv_base import BiorxivMedrxivBase

logger = logging.getLogger(__name__)


class BiorxivSource(BiorxivMedrxivBase):

    _SERVER = "biorxiv"

    @property
    def source_name(self) -> str:
        return "biorxiv"
