"""sources — source connector package."""
from sources.base import PaperMetadata, SourceConnector
from sources.federation import federated_search, get_connector, deduplicate

__all__ = [
    "PaperMetadata",
    "SourceConnector",
    "federated_search",
    "get_connector",
    "deduplicate",
]
