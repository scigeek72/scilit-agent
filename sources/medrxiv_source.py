"""
sources/medrxiv_source.py — medRxiv source connector.

Uses the medRxiv public REST API (same structure as bioRxiv, no key required).
All medRxiv preprints are open access with a direct PDF URL.
"""

from __future__ import annotations

from sources._biorxiv_medrxiv_base import BiorxivMedrxivBase


class MedrxivSource(BiorxivMedrxivBase):

    _SERVER = "medrxiv"

    @property
    def source_name(self) -> str:
        return "medrxiv"
