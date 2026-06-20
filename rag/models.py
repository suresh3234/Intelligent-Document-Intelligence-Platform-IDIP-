from typing import List, Dict, Any
from pydantic import BaseModel, Field
from rag.vector_store import SearchResult

class Citation(BaseModel):
    """Represents a specific source document attribution mapping."""
    doc_id: str = Field(description="Full UUID doc_id of the cited document")
    doc_id_short: str = Field(description="First 8 characters of doc_id used inside generation citations")
    source_uri: str = Field(description="URI of the original document source")
    text_snippet: str = Field(description="Exact reference text block citation snippet")

class RAGResponse(BaseModel):
    """Standard RAG response envelope enclosing citation listings and score metrics."""
    answer: str = Field(description="Generated answer from the LLM model containing document tags")
    citations: List[Citation] = Field(default_factory=list, description="Associated details for all document tags")
    confidence: float = Field(description="Computed confidence score of the generated answer")
    reranked_chunks: List[SearchResult] = Field(default_factory=list, description="Candidate search results after cross-encoder filtering")
    hyde_queries: List[str] = Field(default_factory=list, description="Hypothetical answers generated during HyDE query expansion")
    total_latency_ms: float = Field(description="End-to-end execution latency in milliseconds")
    low_confidence: bool = Field(description="Boolean flag set to True if answer confidence is below threshold")
