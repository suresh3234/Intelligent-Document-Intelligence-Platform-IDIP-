"""Custom exceptions for the Vision module."""
from models.exceptions import ModelsError

class VisionError(ModelsError):
    """Base exception class for Vision module."""
    pass

class VisionInferenceError(VisionError):
    """Raised when LayoutLMv3 or image inference fails."""
    pass
