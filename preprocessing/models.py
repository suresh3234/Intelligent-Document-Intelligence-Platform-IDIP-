from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class TextChunk(BaseModel):
    """Represents an individual text slice/chunk produced by the chunking pipelines."""
    
    chunk_id: str = Field(description="UUID4 unique identifier for the chunk")
    doc_id: str = Field(description="UUID4 parent document identifier")
    chunk_index: int = Field(description="Sequential position index of chunk in document")
    text: str = Field(description="Extracted clean text content of the chunk")
    token_count: int = Field(description="Number of tokens within text (determined by tokenizer)")
    char_start: int = Field(description="Inclusive character start offset in parent document text")
    char_end: int = Field(description="Exclusive character end offset in parent document text")
    page_number: Optional[int] = Field(default=None, description="Physical page number of source document if known")
    section_heading: Optional[str] = Field(default=None, description="Heading of the document section containing this chunk")
    chunk_strategy: str = Field(description="Chunking strategy used: 'fixed', 'semantic', or 'sentence_window'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata flags (e.g. is_table, base_sentence, context_window)")
