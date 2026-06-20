"""Guardrail validation layer implementation for IDIP."""
import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field

from config import settings
from models.ner.service import EntityResult

logger = logging.getLogger("idip.models.guardrails")

class PIIDetectedError(ValueError):
    """Exception raised when personally identifiable information (PII) is detected in the response."""
    pass

class GuardrailValidationError(ValueError):
    """Exception raised when a response fails output schema validation."""
    pass

class IDIPResponse(BaseModel):
    """Standardized production API response schema envelope."""
    answer: str = Field(..., description="The ensembled or synthesized answer text")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="References and attributions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution logging details")

class GuardrailChecker:
    """
    Guardrail validation layer.
    Applies hallucination penalties via pre-trained NLI models, scans and redacts/rejects PII,
    and enforces Pydantic schema schemas before outputs reach the serving boundary.
    """

    def __init__(
        self,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        ner_service: Optional[Any] = None,
        pii_action: str = "redact"  # "redact" or "raise"
    ):
        self.nli_model_name = nli_model_name
        self.ner_service = ner_service
        self.pii_action = pii_action
        self.nli_model = None

        # Regex patterns for various PII types
        self.pii_patterns = {
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
            "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "CREDIT_CARD": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
            "PASSPORT": re.compile(r"\b[A-Z0-9]{9}\b")
        }

    def load_nli(self) -> None:
        """Lazily loads the NLI CrossEncoder model to optimize startup time."""
        if self.nli_model is not None:
            return

        try:
            logger.info(f"Loading NLI cross-encoder model: {self.nli_model_name}...")
            from sentence_transformers import CrossEncoder
            self.nli_model = CrossEncoder(self.nli_model_name)
            logger.info("NLI model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed loading NLI model: {e}")
            # Do not crash, log warning; fallback checks can be performed or mockable
            self.nli_model = None

    def check_hallucination(
        self,
        answer: str,
        retrieved_chunks: List[str],
        confidence: float
    ) -> tuple[float, bool]:
        """
        Runs DeBERTa-v3 NLI checking.
        Computes maximum contradiction probability across chunks.
        If it exceeds 0.7, flags a hallucination warning and penalizes confidence by 0.3.
        """
        if not retrieved_chunks or not answer.strip():
            return confidence, False

        self.load_nli()
        if not self.nli_model:
            logger.warning("NLI model is unavailable. Skipping hallucination check.")
            return confidence, False

        try:
            # Build pairs (premise: fact, hypothesis: model claim)
            pairs = [[chunk, answer] for chunk in retrieved_chunks]
            
            # Predict logits
            logits_list = self.nli_model.predict(pairs)
            
            # CrossEncoder can return a 1D array if only one pair is provided
            if len(pairs) == 1 and logits_list.ndim == 1:
                logits_list = np.expand_dims(logits_list, axis=0)

            # Retrieve contradiction label index from model config or default to 2
            id2label = getattr(self.nli_model.model, "config", {}).get("id2label", {})
            contradiction_idx = 2
            for idx, label in id2label.items():
                if "contradict" in label.lower():
                    contradiction_idx = int(idx)
                    break

            max_contradiction = 0.0
            for logits in logits_list:
                # Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                
                contra_prob = float(probs[contradiction_idx])
                if contra_prob > max_contradiction:
                    max_contradiction = contra_prob

            logger.info(f"NLI Hallucination check completed. Max contradiction score: {max_contradiction:.4f}")

            is_hallucination = max_contradiction > 0.7
            penalized_confidence = confidence
            if is_hallucination:
                penalized_confidence = max(0.0, confidence - 0.3)
                logger.warning(
                    f"Hallucination detected (score {max_contradiction:.4f}). "
                    f"Confidence score penalized: {confidence} -> {penalized_confidence:.4f}"
                )

            return float(round(penalized_confidence, 4)), is_hallucination

        except Exception as e:
            logger.error(f"Error during hallucination validation: {e}")
            return confidence, False

    def scan_pii(self, text: str) -> str:
        """
        Scans text for emails, phone numbers, SSNs, credit cards, and passports.
        Optionally runs NER service to identify names or locations.
        Redacts PII blocks or raises PIIDetectedError based on configuration.
        """
        pii_found = False
        redacted_text = text

        # 1. Regex PII scanning
        for label, pattern in self.pii_patterns.items():
            matches = list(pattern.finditer(redacted_text))
            if matches:
                pii_found = True
                if self.pii_action == "raise":
                    raise PIIDetectedError(f"PII violation: found pattern matching {label} in text.")
                
                # Replace matches from end to start to maintain index locations
                for match in reversed(matches):
                    redacted_text = (
                        redacted_text[:match.start()] + 
                        f"[REDACTED_{label}]" + 
                        redacted_text[match.end():]
                    )

        # 2. NER PII scanning (PERSON names, etc.)
        if self.ner_service:
            try:
                ner_entities = self.ner_service.extract_entities(redacted_text)
                # Filter for person/location names
                target_ents = [
                    e for e in ner_entities 
                    if e.label in ("PERSON", "LOCATION")
                ]
                if target_ents:
                    pii_found = True
                    if self.pii_action == "raise":
                        raise PIIDetectedError("PII violation: found PERSON or LOCATION entity in text.")
                    
                    # Sort by start index descending
                    sorted_ents = sorted(target_ents, key=lambda e: e.start, reverse=True)
                    for ent in sorted_ents:
                        redacted_text = (
                            redacted_text[:ent.start] + 
                            f"[REDACTED_{ent.label}]" + 
                            redacted_text[ent.end:]
                        )
            except Exception as e:
                logger.error(f"Failed to scan NER entities for PII check: {e}")

        if pii_found and self.pii_action == "raise":
            raise PIIDetectedError("PII detected in text response.")

        return redacted_text

    def validate_schema(
        self,
        answer: str,
        confidence: float,
        citations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IDIPResponse:
        """
        Validates response attributes against the production IDIPResponse Pydantic model.
        Throws GuardrailValidationError if inputs are malformed.
        """
        try:
            return IDIPResponse(
                answer=answer,
                confidence=confidence,
                citations=citations or [],
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Response failed output schema validation: {e}")
            raise GuardrailValidationError(f"Invalid schema format: {e}") from e
