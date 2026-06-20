import hashlib
from abc import ABC, abstractmethod
from ingestion.models import IngestedDocument

class BaseSourceAdapter(ABC):
    """Abstract base class representing a document ingestion source adapter."""
    
    @abstractmethod
    async def ingest(self, source_uri: str, **kwargs) -> IngestedDocument:
        """Ingest document from source_uri and return unified IngestedDocument envelope."""
        pass

    def compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 hash in hex format for binary data."""
        return hashlib.sha256(data).hexdigest()

    def safe_decode(self, data: bytes) -> str:
        """Safely decode binary data to UTF-8 text, replacing unrecognized characters."""
        return data.decode("utf-8", errors="replace")
