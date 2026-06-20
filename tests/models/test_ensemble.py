"""Unit tests for the Ensemble Router module."""
import pytest
from models.ensemble import EnsembleRouter, EnsembleResult
from models.ner.service import EntityResult

def test_ensemble_classification_weighted_blend():
    """Verify ensembled classification class calculation based on mock probabilities."""
    weights = {"llm": 0.5, "ner": 0.25, "classifier": 0.25}
    router = EnsembleRouter(ensemble_weights=weights)

    # Class order is: invoice, contract, report, email, form, receipt, legal, other
    # We will pass probability dicts directly
    classifier_probs = {"invoice": 0.8, "contract": 0.1, "report": 0.0, "email": 0.0, "form": 0.0, "receipt": 0.0, "legal": 0.0, "other": 0.1}
    llm_probs = {"invoice": 0.2, "contract": 0.7, "report": 0.0, "email": 0.0, "form": 0.0, "receipt": 0.0, "legal": 0.0, "other": 0.1}
    ner_probs = {"invoice": 0.1, "contract": 0.8, "report": 0.0, "email": 0.0, "form": 0.0, "receipt": 0.0, "legal": 0.0, "other": 0.1}

    # Blended scores:
    # invoice: 0.8 * 0.25 (classifier) + 0.2 * 0.5 (llm) + 0.1 * 0.25 (ner) = 0.2 + 0.1 + 0.025 = 0.325
    # contract: 0.1 * 0.25 (classifier) + 0.7 * 0.5 (llm) + 0.8 * 0.25 (ner) = 0.025 + 0.35 + 0.2 = 0.575
    # other: 0.1 * 0.25 + 0.1 * 0.5 + 0.1 * 0.25 = 0.1
    
    result = router.route_and_ensemble(
        classifier_probs=classifier_probs,
        llm_probs=llm_probs,
        ner_probs=ner_probs
    )

    assert result.final_class == "contract"
    assert result.confidence == pytest.approx(0.575, abs=0.001)
    assert not result.is_ambiguous

def test_ensemble_normalization_partial_models():
    """Verify weight normalization when only a subset of models contribute."""
    # Weights configuration includes ner, but we only supply classifier and vision
    weights = {"classifier": 0.6, "vision": 0.4, "ner": 0.5}
    router = EnsembleRouter(ensemble_weights=weights)

    classifier_probs = {"invoice": 0.9, "contract": 0.0, "report": 0.0, "email": 0.0, "form": 0.0, "receipt": 0.0, "legal": 0.0, "other": 0.1}
    vision_probs = {"invoice": 0.1, "contract": 0.8, "report": 0.0, "email": 0.0, "form": 0.0, "receipt": 0.0, "legal": 0.0, "other": 0.1}

    # Active weights normalized:
    # classifier: 0.6 / (0.6 + 0.4) = 0.6
    # vision: 0.4 / (0.6 + 0.4) = 0.4
    # invoice blended: 0.9 * 0.6 + 0.1 * 0.4 = 0.54 + 0.04 = 0.58
    # contract blended: 0.0 * 0.6 + 0.8 * 0.4 = 0.32
    
    result = router.route_and_ensemble(
        classifier_probs=classifier_probs,
        vision_probs=vision_probs
    )

    assert result.final_class == "invoice"
    assert result.confidence == pytest.approx(0.58, abs=0.001)

def test_ensemble_conflict_ambiguity():
    """Verify the is_ambiguous flag sets when top-2 ensembled classes are within 0.1."""
    router = EnsembleRouter(ensemble_weights={"classifier": 0.5, "llm": 0.5})

    classifier_probs = {"invoice": 0.51, "contract": 0.49, "report": 0.0, "email": 0.0, "form": 0.0, "receipt": 0.0, "legal": 0.0, "other": 0.0}
    llm_probs = {"invoice": 0.49, "contract": 0.51, "report": 0.0, "email": 0.0, "form": 0.0, "receipt": 0.0, "legal": 0.0, "other": 0.0}

    # Blended:
    # invoice = 0.51 * 0.5 + 0.49 * 0.5 = 0.5
    # contract = 0.49 * 0.5 + 0.51 * 0.5 = 0.5
    # Delta is 0.0 (<= 0.1), should flag as ambiguous

    result = router.route_and_ensemble(
        classifier_probs=classifier_probs,
        llm_probs=llm_probs
    )
    assert result.is_ambiguous
    assert result.final_class in ("invoice", "contract") # equal score tie

def test_ensemble_entity_merging():
    """Verify entity lists are correctly merged from NER predictions."""
    router = EnsembleRouter()
    
    e1 = EntityResult(text="CON-123456", label="CONTRACT_ID", start=0, end=10, confidence=1.0)
    e2 = EntityResult(text="INV-789012", label="INVOICE_NO", start=15, end=25, confidence=1.0)
    
    result = router.route_and_ensemble(
        classifier_probs={"other": 1.0},
        ner_result=[e1, e2]
    )

    assert len(result.entity_list) == 2
    texts = [e.text for e in result.entity_list]
    assert "CON-123456" in texts
    assert "INV-789012" in texts
