from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

SourceType = Literal["pdf", "image", "api", "stream", "database"]

class IngestedDocument(BaseModel):
    """Unified schema representing a document ingested into the IDIP platform."""
    
    doc_id: str = Field(description="UUID4 unique identifier for the document")
    ingestion_ts: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when ingestion occurred")
    source_type: SourceType = Field(description="Source of the document ingestion")
    source_uri: str = Field(description="Unique URI/path to source document")
    raw_text: str = Field(description="Normalized extracted text content")
    raw_bytes: Optional[bytes] = Field(default=None, description="Raw binary content of the document (if available)")
    byte_size: int = Field(description="Size of the raw document in bytes")
    checksum: str = Field(description="SHA-256 hex checksum of raw_bytes or raw_text")
    language: str = Field(default="en", description="Detected language code (ISO 639-1)")
    mime_type: str = Field(description="MIME type of the source document")
    page_count: Optional[int] = Field(default=None, description="Number of pages if PDF/Image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific extra metadata attributes")
