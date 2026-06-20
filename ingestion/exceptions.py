"""Exceptions for Ingestion module."""

class IngestionError(Exception):
    """Base exception class for Ingestion module."""
    pass

class DuplicateDocumentError(IngestionError):
    """Raised when a document checksum/URI already exists in deduplication cache."""
    def __init__(self, doc_id: str, message: str = "Duplicate document detected"):
        self.doc_id = doc_id
        super().__init__(f"{message}: {doc_id}")

class AdapterError(IngestionError):
    """Raised when an adapter fails to process or extract text from source."""
    pass

class SignatureVerificationError(IngestionError):
    """Raised when HMAC signature verification fails on webhook endpoints."""
    pass

class RateLimitExceededError(IngestionError):
    """Raised when external API returns 429 and retries have run out."""
    pass

class FileSizeExceededError(IngestionError):
    """Raised when file size exceeds limits and cannot be streamed or parsed."""
    pass

class ValidationError(IngestionError):
    """Raised when document schema or quality checks fail validation."""
    pass

class QualityGateError(ValidationError):
    """Raised when document content quality does not meet required thresholds."""
    pass

