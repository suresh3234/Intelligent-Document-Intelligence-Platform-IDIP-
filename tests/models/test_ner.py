"""Unit tests for the Named Entity Recognition (NER) service."""
import pytest
from unittest.mock import MagicMock, patch
from models.ner.service import NERService, EntityResult
from models.ner.exceptions import NERInferenceError

def test_ner_regex_extraction():
    """Verify that domain-specific regex entities are successfully parsed."""
    service = NERService()
    text = "Please check invoice INV-987654 and contract CON-112233-44 on 2026-06-18 with value $12500.50."
    
    entities = service._extract_regex_entities(text, doc_id="doc1", chunk_id="chunk1")
    
    labels = [e.label for e in entities]
    assert "INVOICE_NO" in labels
    assert "CONTRACT_ID" in labels
    assert "DATE" in labels
    assert "MONEY" in labels
    
    # Assert specific matched values
    inv_entity = next(e for e in entities if e.label == "INVOICE_NO")
    assert inv_entity.text == "INV-987654"
    assert inv_entity.doc_id == "doc1"
    assert inv_entity.confidence == 1.0

def test_ner_bert_token_merging():
    """Verify subword prefix merging and B-/I- token sequence alignment."""
    service = NERService()
    text = "Mr. Devarsh was working at Google today."
    
    # Mock output from HuggingFace pipeline containing split subword tokens
    mock_tokens = [
        {"word": "Dev", "entity": "B-PER", "start": 4, "end": 7, "score": 0.98},
        {"word": "##ar", "entity": "I-PER", "start": 7, "end": 9, "score": 0.95},
        {"word": "##sh", "entity": "I-PER", "start": 9, "end": 11, "score": 0.92},
        {"word": "Google", "entity": "B-ORG", "start": 27, "end": 33, "score": 0.99}
    ]
    
    merged = service._merge_bert_tokens(mock_tokens, text, doc_id="doc1")
    
    assert len(merged) == 2
    per = next(e for e in merged if e.label == "PERSON")
    assert per.text == "Devarsh"
    assert per.start == 4
    assert per.end == 11
    # Average score
    assert per.confidence == pytest.approx(0.95, abs=0.01)
    
    org = next(e for e in merged if e.label == "ORG")
    assert org.text == "Google"
    assert org.confidence == 0.99

def test_ner_overlapping_spans_resolution():
    """Verify overlapping span resolution keeps highest confidence entity."""
    service = NERService()
    
    # Overlapping spans:
    # Span 1: "INV-987654" (INVOICE_NO) from regex (conf=1.0)
    # Span 2: "INV" (ORGANIZATION) from transformer (conf=0.85) - overlap!
    e1 = EntityResult(text="INV-987654", label="INVOICE_NO", start=10, end=20, confidence=1.0)
    e2 = EntityResult(text="INV", label="ORG", start=10, end=13, confidence=0.85)
    
    resolved = service._resolve_overlapping_spans([e1, e2])
    # The higher confidence e1 should win
    assert len(resolved) == 1
    assert resolved[0].text == "INV-987654"
    assert resolved[0].label == "INVOICE_NO"
    
    # Span 3: "Google Inc." (conf=0.9) vs Span 4: "Google" (conf=0.95) - overlap!
    e3 = EntityResult(text="Google Inc.", label="ORG", start=30, end=41, confidence=0.90)
    e4 = EntityResult(text="Google", label="ORG", start=30, end=36, confidence=0.95)
    
    resolved2 = service._resolve_overlapping_spans([e3, e4])
    assert len(resolved2) == 1
    assert resolved2[0].text == "Google"
    assert resolved2[0].confidence == 0.95

@patch("transformers.pipeline")
def test_ner_service_extract_flow(mock_pipeline_cls):
    """Verify full extract_entities flow mapping with mocks."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [
        {"word": "Alice", "entity": "B-PER", "start": 0, "end": 5, "score": 0.99}
    ]
    mock_pipeline_cls.return_value = mock_pipeline
    
    service = NERService()
    text = "Alice paid invoice INV-123456."
    
    results = service.extract_entities(text, doc_id="doc_test")
    
    # Expected: "Alice" (PERSON) + "INV-123456" (INVOICE_NO)
    assert len(results) == 2
    names = [r.text for r in results]
    assert "Alice" in names
    assert "INV-123456" in names
