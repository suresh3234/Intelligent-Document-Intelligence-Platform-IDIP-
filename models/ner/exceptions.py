"""Custom exceptions for the NER module."""
from models.exceptions import ModelsError

class NERError(ModelsError):
    """Base exception for NER module."""
    pass

class NERInferenceError(NERError):
    """Raised when NER inference fails."""
    pass

class InvalidEntityConfigError(NERError):
    """Raised when custom entity patterns or configuration are invalid."""
    pass
