"""Pydantic request and response schemas for the serving API layer."""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Pydantic model representing incoming RAG search queries."""
    query: str = Field(..., description="The user query text")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata key-value search filter conditions")
    top_k: int = Field(default=5, description="Number of results to retrieve and rerank")
    include_citations: bool = Field(default=True, description="Flag indicating if reference citations are included")

class IngestResponse(BaseModel):
    """Pydantic response model returned on successful document ingestion staging."""
    doc_id: str = Field(..., description="UUID identifier assigned to the document")
    status: str = Field(..., description="Document ingestion state (e.g. queued, processing)")
    ingestion_ts: datetime = Field(..., description="Ingestion processing start timestamp")
    estimated_processing_time_s: float = Field(..., description="Estimated background parse duration in seconds")

class DocumentDetailResponse(BaseModel):
    """Pydantic response model returning comprehensive document metadata."""
    doc_id: str = Field(..., description="Unique UUID document ID")
    status: str = Field(..., description="Current document state (e.g. queued, ingested, processed)")
    ingestion_ts: datetime = Field(..., description="Ingestion timestamp")
    source_type: str = Field(..., description="Source type category (e.g. pdf, image, api)")
    source_uri: str = Field(..., description="URI of the original document source")
    byte_size: int = Field(..., description="Byte size of the raw document file")
    checksum: str = Field(..., description="SHA-256 hex checksum verification signature")
    language: str = Field(..., description="Detected language code (e.g. en)")
    mime_type: str = Field(..., description="MIME type header descriptor")
    page_count: Optional[int] = Field(None, description="Extracted count of pages in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific extra fields")
    extracted_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted named entity items")

class ChunkDetail(BaseModel):
    """Pydantic model describing parsed document text chunks."""
    chunk_id: str = Field(..., description="UUID identifier of the chunk")
    doc_id: str = Field(..., description="UUID of the parent document")
    chunk_index: int = Field(..., description="Index order of this chunk in the document")
    text: str = Field(..., description="Raw text segment content")
    token_count: int = Field(..., description="Token count length of chunk text")
    char_start: int = Field(..., description="Start character offset in parent raw text")
    char_end: int = Field(..., description="End character offset in parent raw text")
    page_number: Optional[int] = Field(None, description="Page index where chunk text occurs")
    section_heading: Optional[str] = Field(None, description="Heading label associated with segment")
    chunk_strategy: str = Field(..., description="Chunking strategy code (e.g. fixed, semantic)")
    embedding: Optional[List[float]] = Field(None, description="Generated dense vector representation float array")

class PaginatedChunksResponse(BaseModel):
    """Pydantic response envelope containing paginated text chunks."""
    chunks: List[ChunkDetail] = Field(..., description="List of chunk items")
    total: int = Field(..., description="Total count of text chunks associated with the document")
    page: int = Field(..., description="Current page number index")
    limit: int = Field(..., description="Max limit items per page requested")

class HealthResponse(BaseModel):
    """Pydantic response model returning system health details."""
    status: str = Field(..., description="Overall health state (e.g. healthy, degraded)")
    version: str = Field(..., description="API software application release version")
    model_versions: Dict[str, str] = Field(..., description="Tracking version values for model weights loaded")
    uptime_s: float = Field(..., description="Application execution duration uptime in seconds")
    dependencies: Dict[str, str] = Field(..., description="Active connection statuses (e.g. database, redis)")

class ErrorDetailResponse(BaseModel):
    """Pydantic model representing structured API error messages."""
    error_code: str = Field(..., description="Categorized exception code string")
    message: str = Field(..., description="Human-readable exception details description")
    doc_id: Optional[str] = Field(None, description="Target document ID associated with error if any")
    request_id: str = Field(..., description="Injected request ID header tracking identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Log timestamp matching exception instance")


class DocumentListItem(BaseModel):
    """Pydantic model describing a document row item in the paginated list."""
    doc_id: str = Field(..., description="Unique UUID assigned to document")
    filename: str = Field(..., description="The original filename of the document")
    source_type: str = Field(..., description="Source type category (e.g. pdf, image, json)")
    status: str = Field(..., description="Ingestion processing state (e.g. queued, processing, completed, failed)")
    doc_type: Optional[str] = Field(None, description="Classified document type (e.g. invoice, contract, report)")
    ingestion_ts: datetime = Field(..., description="Timestamp when document was ingested")


class PaginatedDocumentsResponse(BaseModel):
    """Pydantic response envelope containing paginated list of documents."""
    documents: List[DocumentListItem] = Field(..., description="List of document items")
    total: int = Field(..., description="Total count of documents matching the filters")
    page: int = Field(..., description="Current page index")
    limit: int = Field(..., description="Limit of items per page")


class BulkDeleteRequest(BaseModel):
    """Pydantic request model for bulk deleting documents."""
    doc_ids: List[str] = Field(..., description="List of document IDs to delete")

