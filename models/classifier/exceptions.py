"""Custom exceptions for the Document Classifier module."""
from models.exceptions import ModelsError

class ClassifierError(ModelsError):
    """Base exception class for Document Classifier module."""
    pass

class ClassifierTrainingError(ClassifierError):
    """Raised when document classifier training fails."""
    pass

class ClassifierInferenceError(ClassifierError):
    """Raised when document classifier inference fails."""
    pass
