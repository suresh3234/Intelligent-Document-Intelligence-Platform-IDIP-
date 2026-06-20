"""Unit tests for the Guardrail Checker module."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from models.guardrails import GuardrailChecker, PIIDetectedError, GuardrailValidationError, IDIPResponse
from models.ner.service import EntityResult

def test_guardrail_schema_validation():
    """Verify output schema validation accepts correct inputs and rejects malformed fields."""
    checker = GuardrailChecker()
    
    # Valid input
    resp = checker.validate_schema(
        answer="Valid response text.",
        confidence=0.85,
        citations=[{"doc_id": "uuid1", "text_snippet": "fact"}],
        metadata={"run": "success"}
    )
    assert isinstance(resp, IDIPResponse)
    assert resp.answer == "Valid response text."
    assert resp.confidence == 0.85
    assert len(resp.citations) == 1

    # Invalid input: answer missing (TypeError/ValidationError)
    with pytest.raises(GuardrailValidationError):
        checker.validate_schema(
            answer=None,  # type: ignore
            confidence=0.85
        )

def test_guardrail_pii_redaction():
    """Verify that PII patterns are successfully redacted in text."""
    checker = GuardrailChecker(pii_action="redact")
    
    text = "Please write to test@example.com or call 555-019-2834. My SSN is 000-12-3456."
    redacted = checker.scan_pii(text)
    
    assert "test@example.com" not in redacted
    assert "555-019-2834" not in redacted
    assert "000-12-3456" not in redacted
    
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_SSN]" in redacted

def test_guardrail_pii_raise_exception():
    """Verify that PII scanner raises PIIDetectedError when config action is 'raise'."""
    checker = GuardrailChecker(pii_action="raise")
    
    text = "Reach me at 555-019-2834."
    with pytest.raises(PIIDetectedError):
        checker.scan_pii(text)

def test_guardrail_pii_ner_redaction():
    """Verify that NER entities are redacted by the PII scanner."""
    mock_ner = MagicMock()
    # Mock entity detection
    mock_ner.extract_entities.return_value = [
        EntityResult(text="Alice", label="PERSON", start=10, end=15, confidence=0.99)
    ]
    
    checker = GuardrailChecker(ner_service=mock_ner, pii_action="redact")
    
    text = "The user is Alice today."
    redacted = checker.scan_pii(text)
    
    assert "Alice" not in redacted
    assert "[REDACTED_PERSON]" in redacted

def test_guardrail_hallucination_detection():
    """Verify hallucination checker penalizes confidence when NLI contradiction exceeds threshold."""
    checker = GuardrailChecker()
    
    # Mock NLI CrossEncoder model
    mock_model = MagicMock()
    # Mock model predict output logits for 1 pair
    # Logits index: 0=entailment, 1=neutral, 2=contradiction
    # Case A: Low contradiction
    mock_model.predict.return_value = np.array([[2.0, 1.0, -2.0]])  # contradiction is low
    
    # Assign model config mock for id2label
    mock_config = MagicMock()
    mock_config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    mock_model.model.config = mock_config
    
    checker.nli_model = mock_model
    
    conf_a, flag_a = checker.check_hallucination(
        answer="Claim",
        retrieved_chunks=["Fact"],
        confidence=0.9
    )
    # Contradiction probability is very low, no change
    assert conf_a == 0.9
    assert not flag_a

    # Case B: High contradiction (> 0.7)
    # Logits where index 2 (contradiction) is extremely high
    mock_model.predict.return_value = np.array([[-2.0, -2.0, 5.0]])
    
    conf_b, flag_b = checker.check_hallucination(
        answer="Contradicting claim",
        retrieved_chunks=["Fact"],
        confidence=0.85
    )
    # Contradiction is high, flag is True and confidence drops by 0.3
    assert flag_b
    assert conf_b == pytest.approx(0.55, abs=0.01)
