"""Exceptions for Serving module."""

class ServingError(Exception):
    """Base exception class for Serving module."""
    pass

class ModelTimeoutError(ServingError):
    """Exception raised when a model inference call times out."""
    pass
