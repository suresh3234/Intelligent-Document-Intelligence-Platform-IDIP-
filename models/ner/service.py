"""NER Service implementation for IDIP."""
import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from config import settings
from models.ner.config import TRANSFORMER_ENTITY_MAP, DEFAULT_REGEX_PATTERNS
from models.ner.exceptions import NERInferenceError

logger = logging.getLogger("idip.models.ner.service")

class EntityResult(BaseModel):
    """Pydantic model representing an extracted named entity."""
    text: str = Field(..., description="The exact matched entity text")
    label: str = Field(..., description="The category/label of the entity")
    start: int = Field(..., description="Start character offset")
    end: int = Field(..., description="End character offset (exclusive)")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    doc_id: Optional[str] = Field(None, description="Associated document ID")
    chunk_id: Optional[str] = Field(None, description="Associated chunk ID")

class NERService:
    """
    Named Entity Recognition (NER) Service.
    Uses pre-trained transformer model (bert-base-NER) for base entities
    and a configurable regex matching pipeline for domain-specific entities.
    """

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        custom_regex_patterns: Optional[Dict[str, str]] = None
    ):
        self.model_name = model_name
        self.pipeline = None
        
        # Merge default patterns with custom configured regex patterns
        self.regex_patterns = dict(DEFAULT_REGEX_PATTERNS)
        if custom_regex_patterns:
            self.regex_patterns.update(custom_regex_patterns)

    def load_model(self) -> None:
        """Lazily loads the transformer NER pipeline to optimize startup time."""
        if self.pipeline is not None:
            return

        try:
            logger.info(f"Loading transformer NER model: {self.model_name}...")
            from transformers import pipeline
            # Load with no aggregation first so we can demonstrate custom B-/I- and subword merging logic
            self.pipeline = pipeline("ner", model=self.model_name, aggregation_strategy="none")
            logger.info("Transformer NER model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise NERInferenceError(f"Could not load HuggingFace model '{self.model_name}': {e}") from e

    def extract_entities(
        self,
        text: str,
        doc_id: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> List[EntityResult]:
        """
        Extracts base named entities (via BERT) and domain entities (via regex),
        merges word pieces, and resolves overlapping spans.
        """
        if not text.strip():
            return []

        # 1. Extract base entities using BERT model
        transformer_entities = self._extract_transformer_entities(text, doc_id, chunk_id)

        # 2. Extract domain entities using Regex matchers
        regex_entities = self._extract_regex_entities(text, doc_id, chunk_id)

        # 3. Combine both lists
        combined_entities = transformer_entities + regex_entities

        # 4. Resolve overlapping spans (keep highest confidence)
        resolved_entities = self._resolve_overlapping_spans(combined_entities)

        return resolved_entities

    def _extract_transformer_entities(
        self,
        text: str,
        doc_id: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> List[EntityResult]:
        """Queries the transformer pipeline and applies B-/I- & subword merging."""
        self.load_model()
        if not self.pipeline:
            return []

        try:
            # Perform prediction
            raw_predictions = self.pipeline(text)
            return self._merge_bert_tokens(raw_predictions, text, doc_id, chunk_id)
        except Exception as e:
            logger.error(f"Error during transformer entity extraction: {e}")
            raise NERInferenceError(f"Transformer inference failure: {e}") from e

    def _merge_bert_tokens(
        self,
        tokens: List[Dict[str, Any]],
        text: str,
        doc_id: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> List[EntityResult]:
        """
        Merges subword tokens (prefixed with '##') and merges contiguous B- and I- tags
        into single complete EntityResult instances.
        """
        if not tokens:
            return []

        merged_entities: List[EntityResult] = []
        active_entity: Optional[Dict[str, Any]] = None

        for tok in tokens:
            word = tok.get("word", "")
            entity_label = tok.get("entity", "O")
            start = tok.get("start", 0)
            end = tok.get("end", 0)
            score = float(tok.get("score", 0.0))

            if entity_label == "O":
                if active_entity:
                    merged_entities.append(self._create_entity_result(active_entity, text, doc_id, chunk_id))
                    active_entity = None
                continue

            # Parse B- or I- label structures (e.g. B-PER -> tag=PER, prefix=B)
            parts = entity_label.split("-")
            prefix = parts[0]
            tag = parts[1] if len(parts) > 1 else parts[0]
            mapped_label = TRANSFORMER_ENTITY_MAP.get(tag, tag)

            # Check if this token belongs to the active entity
            is_subword = word.startswith("##")
            is_contiguous = active_entity and (start == active_entity["end"] or (start == active_entity["end"] + 1 and text[active_entity["end"]] == " "))
            is_same_tag = active_entity and active_entity["label"] == mapped_label

            if active_entity and (is_subword or (prefix == "I" and is_contiguous and is_same_tag)):
                # Extend active entity
                clean_word = word[2:] if is_subword else word
                # Add space if there is one in the source text
                if start == active_entity["end"] + 1 and text[active_entity["end"]] == " ":
                    active_entity["text"] += " " + clean_word
                else:
                    active_entity["text"] += clean_word
                active_entity["end"] = end
                active_entity["scores"].append(score)
            else:
                # Close active entity if there is one
                if active_entity:
                    merged_entities.append(self._create_entity_result(active_entity, text, doc_id, chunk_id))
                
                # Start new entity
                clean_word = word[2:] if is_subword else word
                active_entity = {
                    "text": clean_word,
                    "label": mapped_label,
                    "start": start,
                    "end": end,
                    "scores": [score]
                }

        # Close final active entity
        if active_entity:
            merged_entities.append(self._create_entity_result(active_entity, text, doc_id, chunk_id))

        return merged_entities

    def _create_entity_result(
        self,
        entity_dict: Dict[str, Any],
        text: str,
        doc_id: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> EntityResult:
        """Helper to compute mean confidence score and instantiate EntityResult."""
        mean_score = sum(entity_dict["scores"]) / len(entity_dict["scores"])
        # Re-fetch exact text from source to avoid space mismatches
        matched_text = text[entity_dict["start"]:entity_dict["end"]]
        return EntityResult(
            text=matched_text,
            label=entity_dict["label"],
            start=entity_dict["start"],
            end=entity_dict["end"],
            confidence=float(round(mean_score, 4)),
            doc_id=doc_id,
            chunk_id=chunk_id
        )

    def _extract_regex_entities(
        self,
        text: str,
        doc_id: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> List[EntityResult]:
        """Extracts domain-specific entities using predefined and custom regex matchers."""
        results: List[EntityResult] = []
        for label, pattern in self.regex_patterns.items():
            for match in re.finditer(pattern, text):
                results.append(
                    EntityResult(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,  # Rule matches have base confidence 1.0
                        doc_id=doc_id,
                        chunk_id=chunk_id
                    )
                )
        return results

    def _resolve_overlapping_spans(self, entities: List[EntityResult]) -> List[EntityResult]:
        """
        Iterates over entities, detecting overlaps.
        When overlaps are found, retains the prediction with the highest confidence score.
        """
        if not entities:
            return []

        # Sort entities by start offset, then by length descending
        sorted_entities = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        resolved: List[EntityResult] = []

        for ent in sorted_entities:
            overlap_detected = False
            for idx, existing in enumerate(resolved):
                # Check for character span overlap: max(start1, start2) < min(end1, end2)
                if max(ent.start, existing.start) < min(ent.end, existing.end):
                    overlap_detected = True
                    # Compare confidence scores to select winner
                    if ent.confidence > existing.confidence:
                        # Replace existing with current entity
                        resolved[idx] = ent
                    elif ent.confidence == existing.confidence:
                        # Keep the longer entity if confidence is equal
                        ent_len = ent.end - ent.start
                        ex_len = existing.end - existing.start
                        if ent_len > ex_len:
                            resolved[idx] = ent
                    break
            
            if not overlap_detected:
                resolved.append(ent)

        # Re-sort final list by start character
        return sorted(resolved, key=lambda e: e.start)
