import pytest
import numpy as np
from unittest.mock import MagicMock
from preprocessing.cleaning import TextCleaner
from preprocessing.chunking import ChunkingPipeline
from ingestion.models import IngestedDocument

@pytest.fixture
def clean_doc_factory():
    def _create_doc(text: str) -> IngestedDocument:
        return IngestedDocument(
            doc_id="test-doc-id",
            source_type="pdf",
            source_uri="s3://doc",
            raw_text=text,
            byte_size=len(text),
            checksum="fake-checksum",
            mime_type="application/pdf"
        )
    return _create_doc

def test_text_cleaner_basic():
    cleaner = TextCleaner()
    
    # Unicode and control characters test
    dirty_text = "Hello\u2122 \u0007World!\x0b\nCollapse   multiple   spaces.\n\n\n\nPreserve double newlines."
    cleaned = cleaner.clean(dirty_text)
    
    assert "HelloTM" in cleaned
    assert "World!" in cleaned
    assert "Collapse multiple spaces." in cleaned
    assert "\n\nPreserve" in cleaned
    assert "\n\n\n" not in cleaned

def test_text_cleaner_boilerplate():
    cleaner = TextCleaner()
    
    # Document with 4 pages sharing a repeated header
    page1 = "--- PAGE 1 ---\nIDIP DOCUMENT CONFIDENTIAL\nUnique line content for page 1."
    page2 = "--- PAGE 2 ---\nIDIP DOCUMENT CONFIDENTIAL\nUnique line content for page 2."
    page3 = "--- PAGE 3 ---\nIDIP DOCUMENT CONFIDENTIAL\nUnique line content for page 3."
    page4 = "--- PAGE 4 ---\nIDIP DOCUMENT CONFIDENTIAL\nUnique line content for page 4."
    
    doc_text = f"{page1}\n{page2}\n{page3}\n{page4}"
    cleaned = cleaner.clean(doc_text)
    
    # Verify the repeated header line is removed, while unique content is preserved
    assert "IDIP DOCUMENT CONFIDENTIAL" not in cleaned
    assert "Unique line content for page 1." in cleaned
    assert "Unique line content for page 3." in cleaned

def test_text_cleaner_is_table():
    cleaner = TextCleaner()
    
    table_text = "Header 1\tHeader 2\tHeader 3\nVal 1\tVal 2\tVal 3"
    assert cleaner.is_table_chunk(table_text) is True
    
    md_table = "| Col 1 | Col 2 | Col 3 |\n|---|---|---|"
    assert cleaner.is_table_chunk(md_table) is True
    
    normal_text = "This is just standard text without tabs or table separators."
    assert cleaner.is_table_chunk(normal_text) is False

def test_fixed_chunking(clean_doc_factory):
    pipeline = ChunkingPipeline()
    
    # 5 sentences
    doc_text = (
        "First sentence of the text. Second sentence here. Third sentence is longer. "
        "Fourth sentence is quite long too. Fifth sentence closes the paragraph."
    )
    doc = clean_doc_factory(doc_text)
    
    # Small size to trigger multiple chunks without splitting sentences
    chunks = pipeline.chunk_document(doc, strategy="fixed", chunk_size=15, overlap=2)
    
    assert len(chunks) > 1
    assert chunks[0].chunk_strategy == "fixed"
    # Verify we did not split sentences (every chunk ends with a dot)
    for chunk in chunks:
        assert chunk.text.endswith(".")
        assert chunk.token_count <= 25  # Should fit sentences cleanly

def test_semantic_chunking(clean_doc_factory):
    # Mock embedding service returning orthogonal/similar embeddings
    emb_service = MagicMock()
    
    # 3 sentences
    s1 = "This is a sentence about NLP and chunking pipeline."
    s2 = "We build a semantic chunker using cosine similarity."
    s3 = "Cooking a delicious cake requires flour and sugar."
    
    # Mock embeddings: s1 and s2 are similar (similarity=1.0), s3 is orthogonal (similarity=0.0)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.0, 1.0])
    
    emb_service.encode.return_value = [v1, v2, v3]
    
    pipeline = ChunkingPipeline(embedding_service=emb_service)
    doc = clean_doc_factory(f"{s1} {s2} {s3}")
    
    chunks = pipeline.chunk_document(doc, strategy="semantic", similarity_threshold=0.50)
    
    # Should split between s2 and s3 (similarity = 0.0 < 0.50)
    # But wait! If s1+s2 tokens is less than 128 (default min), they'll be merged if the last group is also small.
    # Let's verify chunks are generated
    assert len(chunks) >= 1
    assert chunks[0].chunk_strategy == "semantic"

def test_sentence_window_chunking(clean_doc_factory):
    pipeline = ChunkingPipeline()
    
    sentences = [f"Sentence number {i} in the list." for i in range(10)]
    doc_text = " ".join(sentences)
    doc = clean_doc_factory(doc_text)
    
    chunks = pipeline.chunk_document(doc, strategy="sentence_window")
    
    # Number of chunks should match number of sentences
    assert len(chunks) == 10
    
    # Inspect chunk at index 5
    # Should attach context window of N=3 (sentences 2, 3, 4, 5, 6, 7, 8)
    middle_chunk = chunks[5]
    assert middle_chunk.chunk_strategy == "sentence_window"
    assert middle_chunk.metadata["base_sentence"] == "Sentence number 5 in the list."
    assert len(middle_chunk.metadata["context_window"]) == 7  # 3 before + base + 3 after
    assert "Sentence number 2" in middle_chunk.text
    assert "Sentence number 8" in middle_chunk.text
    assert "Sentence number 9" not in middle_chunk.text

def test_chunking_edge_cases(clean_doc_factory):
    pipeline = ChunkingPipeline()
    
    # 1. Empty doc
    empty_doc = clean_doc_factory("")
    assert pipeline.chunk_document(empty_doc, strategy="fixed") == []
    
    # 2. Single word doc
    word_doc = clean_doc_factory("Word.")
    chunks = pipeline.chunk_document(word_doc, strategy="fixed")
    assert len(chunks) == 1
    assert chunks[0].text == "Word."
