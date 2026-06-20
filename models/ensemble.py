"""Ensemble Router implementation for IDIP."""
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from config import settings
from models.ner.service import EntityResult
from models.classifier.config import CLASSIFIER_CLASSES

logger = logging.getLogger("idip.models.ensemble")

class EnsembleResult(BaseModel):
    """Pydantic model representing the consolidated ensemble output."""
    final_answer: Optional[str] = Field(None, description="The final synthesized text answer")
    final_class: str = Field(..., description="The ensembled/voted classification label")
    entity_list: List[EntityResult] = Field(default_factory=list, description="Consolidated list of extracted named entities")
    confidence: float = Field(..., description="The ensembled confidence score of the voted class")
    model_contributions: Dict[str, Any] = Field(..., description="Details of model contributions and probabilities")
    is_ambiguous: bool = Field(False, description="Flagged as True if top-2 ensembled classes are within 0.1 probability")

class EnsembleRouter:
    """
    Ensemble Router.
    Collects outputs from different inference layers, performs weighted probability voting,
    and returns a consolidated EnsembleResult containing the voted class, confidence,
    merged entities, and execution details.
    """

    def __init__(self, ensemble_weights: Optional[Dict[str, float]] = None):
        # Fetch default ensemble weights from global settings or config
        self.weights = ensemble_weights or settings.ENSEMBLE_WEIGHTS
        logger.info(f"EnsembleRouter initialized with weights: {self.weights}")

    def _one_hot(self, class_name: str) -> Dict[str, float]:
        """Creates a one-hot probability map for a specific class label."""
        target = class_name.lower().strip()
        if target not in CLASSIFIER_CLASSES:
            target = "other"
        return {c: (1.0 if c == target else 0.0) for c in CLASSIFIER_CLASSES}

    def _to_probs(self, result: Any) -> Optional[Dict[str, float]]:
        """Parses various result types into a standard class probabilities dictionary."""
        if result is None:
            return None

        # 1. If already a matching dict format
        if isinstance(result, dict):
            if all(k in CLASSIFIER_CLASSES for k in result.keys()):
                return {k: float(v) for k, v in result.items()}
            if "class_probabilities" in result:
                return {k: float(v) for k, v in result["class_probabilities"].items()}
            if "predicted_class" in result:
                return self._one_hot(result["predicted_class"])
            return None

        # 2. If Pydantic model
        if hasattr(result, "class_probabilities"):
            probs = getattr(result, "class_probabilities")
            if isinstance(probs, dict):
                return {k: float(v) for k, v in probs.items()}
        if hasattr(result, "predicted_class"):
            pred_class = getattr(result, "predicted_class")
            if isinstance(pred_class, str):
                return self._one_hot(pred_class)

        # 3. If raw string (e.g. parsed from LLM text output)
        if isinstance(result, str):
            val = result.strip().lower()
            for c in CLASSIFIER_CLASSES:
                if c in val:
                    return self._one_hot(c)
            return self._one_hot("other")

        return None

    def route_and_ensemble(
        self,
        classifier_result: Optional[Any] = None,
        vision_result: Optional[Any] = None,
        llm_result: Optional[Any] = None,
        ner_result: Optional[Any] = None,
        # Allow passing probability distributions directly
        classifier_probs: Optional[Dict[str, float]] = None,
        vision_probs: Optional[Dict[str, float]] = None,
        llm_probs: Optional[Dict[str, float]] = None,
        ner_probs: Optional[Dict[str, float]] = None,
        final_answer: Optional[str] = None
    ) -> EnsembleResult:
        """
        Blends class probabilities across models using configured weights,
        checks for class prediction ambiguity, merges entity lists, and returns EnsembleResult.
        """
        # Resolve probability maps
        p_classifier = classifier_probs or self._to_probs(classifier_result)
        p_vision = vision_probs or self._to_probs(vision_result)
        p_llm = llm_probs or self._to_probs(llm_result)
        p_ner = ner_probs or self._to_probs(ner_result)

        contributed_probs = {
            "classifier": p_classifier,
            "vision": p_vision,
            "llm": p_llm,
            "ner": p_ner
        }

        # Filter out models that did not contribute
        active_models = {k: v for k, v in contributed_probs.items() if v is not None}
        
        # Calculate active weights normalized to sum to 1.0
        active_weights = {}
        for model in active_models:
            active_weights[model] = self.weights.get(model, 1.0 / len(active_models))

        total_weight = sum(active_weights.values())
        if total_weight > 0:
            active_weights = {k: v / total_weight for k, v in active_weights.items()}
        else:
            # Fallback to equal weighting if no weights are defined or total weight is zero
            active_weights = {k: 1.0 / len(active_models) for k in active_models}

        # Compute ensembled probabilities per class
        blended_probs: Dict[str, float] = {c: 0.0 for c in CLASSIFIER_CLASSES}
        for cls in CLASSIFIER_CLASSES:
            score = 0.0
            for model, proba in active_models.items():
                score += proba.get(cls, 0.0) * active_weights[model]
            blended_probs[cls] = float(round(score, 4))

        # Determine top classes
        sorted_classes = sorted(blended_probs.items(), key=lambda x: x[1], reverse=True)
        final_class = sorted_classes[0][0]
        confidence = sorted_classes[0][1]

        # Ambiguity resolution: check if top-2 classes are within 0.1 probability
        is_ambiguous = False
        if len(sorted_classes) > 1:
            diff = sorted_classes[0][1] - sorted_classes[1][1]
            if diff <= 0.1:
                is_ambiguous = True
                logger.warning(
                    f"Classification is ambiguous. Top-2 classes: "
                    f"{sorted_classes[0]} vs {sorted_classes[1]} (delta {diff:.4f})"
                )

        # Consolidate entities
        entity_list: List[EntityResult] = []
        
        def add_entities(entities_data: Any) -> None:
            if not entities_data:
                return
            if isinstance(entities_data, list):
                for ent in entities_data:
                    if isinstance(ent, EntityResult):
                        entity_list.append(ent)
                    elif isinstance(ent, dict):
                        try:
                            entity_list.append(EntityResult(**ent))
                        except Exception as e:
                            logger.warning(f"Failed to parse entity dict: {ent}, error: {e}")
            elif hasattr(entities_data, "regions"): # layoutLMv3 region texts
                regions = getattr(entities_data, "regions")
                if isinstance(regions, list):
                    for r in regions:
                        # Convert detected regions into entities if they have text and label
                        if hasattr(r, "text") and getattr(r, "text") and hasattr(r, "label"):
                            entity_list.append(
                                EntityResult(
                                    text=getattr(r, "text"),
                                    label=getattr(r, "label").upper(),
                                    start=0,
                                    end=len(getattr(r, "text")),
                                    confidence=getattr(r, "confidence", 1.0)
                                )
                            )

        # Retrieve entities from ner_result or vision_result
        add_entities(ner_result)
        if not ner_result and vision_result:
            add_entities(vision_result)

        # Extract answer from LLM output if not explicitly provided
        resolved_answer = final_answer
        if not resolved_answer and llm_result:
            if isinstance(llm_result, str):
                resolved_answer = llm_result
            elif hasattr(llm_result, "answer"):
                resolved_answer = getattr(llm_result, "answer")

        model_contributions = {
            "probabilities": {m: {k: float(round(v, 4)) for k, v in probs.items()} for m, probs in active_models.items()},
            "weights": {m: float(round(w, 4)) for m, w in active_weights.items()},
            "blended_probabilities": blended_probs
        }

        return EnsembleResult(
            final_answer=resolved_answer,
            final_class=final_class,
            entity_list=entity_list,
            confidence=float(round(confidence, 4)),
            model_contributions=model_contributions,
            is_ambiguous=is_ambiguous
        )
