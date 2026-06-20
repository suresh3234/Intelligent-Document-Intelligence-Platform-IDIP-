"""FastAPI dependency injection provider for IDIP serving layer."""
from typing import Generator, Any
from fastapi import Request
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import settings
from rag.vector_store import VectorStoreInterface
from models.ner.service import NERService
from models.classifier.service import DocumentClassifier
from models.vision.service import VisionDocumentAnalyzer
from models.llm.inference import LLMInferenceService
from models.ensemble import EnsembleRouter
from models.guardrails import GuardrailChecker

# Create Database Session Maker
db_url = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)
if "sqlite" in settings.POSTGRES_DB or settings.ENVIRONMENT == "development":
    # Fallback to local SQLite database if Postgres is not set or during local development/test
    db_url = "sqlite:///./idip_metadata.db"

engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session() -> Generator[Session, None, None]:
    """Yields a database session instance, closing after completion."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

_redis_clients = {}

def get_redis_client() -> redis.Redis:
    """Returns Redis connection pool client."""
    global _redis_clients
    # Bypass cache if redis.Redis is mocked in tests
    import unittest.mock
    if isinstance(redis.Redis, (unittest.mock.Mock, unittest.mock.MagicMock)):
        return redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
    key = (settings.REDIS_HOST, settings.REDIS_PORT)
    if key not in _redis_clients:
        _redis_clients[key] = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
    return _redis_clients[key]

def get_vector_store(request: Request) -> VectorStoreInterface:
    """Gets the vector database interface singleton from app state."""
    return request.app.state.vector_store

def get_ner_service(request: Request) -> NERService:
    """Gets the NER service singleton from app state."""
    return request.app.state.ner_service

def get_classifier_service(request: Request) -> DocumentClassifier:
    """Gets the Document Classifier singleton from app state."""
    return request.app.state.classifier_service

def get_vision_analyzer(request: Request) -> VisionDocumentAnalyzer:
    """Gets the Vision Document Analyzer singleton from app state."""
    return request.app.state.vision_analyzer

def get_llm_service(request: Request) -> LLMInferenceService:
    """Gets the LLM causal inference singleton from app state."""
    return request.app.state.llm_service

def get_ensemble_router(request: Request) -> EnsembleRouter:
    """Gets the Ensemble Router singleton from app state."""
    return request.app.state.ensemble_router

def get_guardrail_checker(request: Request) -> GuardrailChecker:
    """Gets the Guardrail Checker singleton from app state."""
    return request.app.state.guardrail_checker
