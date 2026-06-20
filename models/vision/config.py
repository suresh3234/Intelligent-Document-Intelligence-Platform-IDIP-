"""Configuration settings for Vision module."""
VISION_MODEL_NAME: str = "microsoft/layoutlmv3-base"

# Region classification labels
REGION_LABELS = [
    "text",
    "title",
    "table",
    "figure",
    "list",
    "header",
    "footer",
    "signature",
    "other"
]
