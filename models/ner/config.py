"""Configuration settings for Named Entity Recognition (NER) module."""
from typing import Dict, Any

# Map transformer-native labels to unified target outputs
TRANSFORMER_ENTITY_MAP: Dict[str, str] = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION"
}

# Default regular expression extractors for domain-specific entities
DEFAULT_REGEX_PATTERNS: Dict[str, str] = {
    "CONTRACT_ID": r"\bCON-\d{6}-\d{2}\b",
    "INVOICE_NO": r"\bINV-\d{6,8}\b",
    "MONEY": r"\$\d+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s?(?:USD|EUR|GBP|dollars|cents)\b",
    "DATE": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4}\b"
}
