"""Configuration settings for Document Classifier module."""
from typing import List, Dict

CLASSIFIER_CLASSES: List[str] = [
    "invoice",
    "contract",
    "report",
    "email",
    "form",
    "receipt",
    "legal",
    "other"
]

DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "xgboost": 0.4,
    "bert": 0.6
}

# Temperature scaling value for class probability calibration
DEFAULT_CALIBRATION_TEMPERATURE: float = 1.0
