"""Integration tests for the IDIP serving API layer."""
import json
import base64
import hmac
import hashlib
import time
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

from serving.api import app
from serving.schemas import IngestResponse
from ingestion.exceptions import DuplicateDocumentError
from models.guardrails import PIIDetectedError
from serving.exceptions import ModelTimeoutError

# Helper to construct a valid signed JWT for testing AuthMiddleware
def create_mock_jwt(api_key: str, secret: str = "idip_secret_key_1234567890", expired: bool = False) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    exp_time = time.time() - 3600 if expired else time.time() + 3600
    payload = {"api_key": api_key, "exp": exp_time}
    
    def b64_url_encode(d: dict) -> str:
        s = json.dumps(d)
        return base64.urlsafe_b64encode(s.encode("utf-8")).decode("utf-8").rstrip("=")
        
    h_b64 = b64_url_encode(header)
    p_b64 = b64_url_encode(payload)
    
    signing_input = f"{h_b64}.{p_b64}".encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    
    return f"{h_b64}.{p_b64}.{sig_b64}"

@pytest.fixture
def valid_auth_headers():
    token = create_mock_jwt("test_client_key")
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture(autouse=True)
def setup_app_state():
    app.state.ner_service = MagicMock()
    app.state.classifier_service = MagicMock()
    app.state.vision_analyzer = MagicMock()
    app.state.llm_service = MagicMock()
    app.state.embedding_service = MagicMock()
    app.state.embedding_service.encode_query.return_value = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    app.state.vector_store = MagicMock()
    app.state.ensemble_router = MagicMock()
    app.state.guardrail_checker = MagicMock()

@pytest.mark.asyncio
async def test_health_check():
    """Verify that GET /v1/health is exempt from Auth and checks connection status."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/v1/health")
        
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "database" in data["dependencies"]
    assert "redis" in data["dependencies"]

@pytest.mark.asyncio
async def test_auth_middleware_missing_token():
    """Verify endpoint rejects requests with missing Authorization header."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/query", json={"query": "hello"})
        
    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"

@pytest.mark.asyncio
async def test_auth_middleware_invalid_token():
    """Verify endpoint rejects requests with invalid or expired signatures."""
    # Expired token
    expired_token = create_mock_jwt("test_client", expired=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {expired_token}"},
            json={"query": "hello"}
        )
    assert response.status_code == 401

    # Malformed token signature
    invalid_token = create_mock_jwt("test_client", secret="wrong_secret_key")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/query",
            headers={"Authorization": f"Bearer {invalid_token}"},
            json={"query": "hello"}
        )
    assert response.status_code == 401

@pytest.mark.asyncio
@patch("serving.api.ingest_document_task.delay")
async def test_document_ingestion_success(mock_celery_delay, valid_auth_headers):
    """Verify successful multipart document ingestion routes to background queues."""
    mock_celery_delay.return_value = MagicMock()
    
    file_payload = ("test_doc.pdf", b"Dummy PDF file content structure.", "application/pdf")
    form_data = {
        "metadata": '{"source_uri": "s3://source/doc.pdf", "language": "en"}'
    }
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/documents/ingest",
            headers=valid_auth_headers,
            files={"file": file_payload},
            data=form_data
        )
        
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert data["status"] == "queued"
    mock_celery_delay.assert_called_once()

@pytest.mark.asyncio
async def test_document_ingestion_size_limit(valid_auth_headers):
    """Verify that file size exceeding 50MB limit is rejected."""
    file_payload = ("huge.pdf", b"small content", "application/pdf")
    form_data = {"metadata": '{"source_uri": "s3://source/doc.pdf"}'}

    with patch("starlette.datastructures.UploadFile.read", new_callable=AsyncMock) as mock_read:
        mock_bytes = MagicMock(spec=bytes)
        mock_bytes.__len__.return_value = int(51 * 1024 * 1024)
        mock_read.return_value = mock_bytes

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/documents/ingest",
                headers=valid_auth_headers,
                files={"file": file_payload},
                data=form_data
            )
        
    assert response.status_code == 422
    assert "File size exceeds" in response.json()["message"]

@pytest.mark.asyncio
@patch("serving.dependencies.get_redis_client")
async def test_rate_limit_blocking(mock_redis_func, valid_auth_headers):
    """Verify that Redis rate limit counts exceeding threshold returns HTTP 429."""
    mock_redis = MagicMock()
    # Mock redis pipeline card output representing more than 100 requests in sliding window
    mock_redis.pipeline.return_value.execute.return_value = [None, 105]
    mock_redis_func.return_value = mock_redis
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/query",
            headers=valid_auth_headers,
            json={"query": "test query"}
        )
        
    assert response.status_code == 429
    assert response.json()["error_code"] == "RATE_LIMIT_EXCEEDED"

@pytest.mark.asyncio
@patch("serving.dependencies.get_redis_client")
async def test_semantic_cache_hit(mock_redis_func, valid_auth_headers):
    """Verify that semantic cache hit returns cached response and sets correct header."""
    mock_redis = MagicMock()
    # Mock matching cached query embeddings with similarity delta score >= 0.95
    # Since query vector is L2 normalized, similarity = dot product
    # Mock query_vector = [1, 0], cached_vector = [1, 0]
    cached_query = "Who is the EKS cluster sponsor?"
    
    # Mock redis getall/hget returns
    mock_redis.hgetall.return_value = {cached_query: ",".join(["1.0"] + ["0.0"] * 1023)}
    mock_redis.hget.return_value = '{"answer": "Mocked cached RAG answer response.", "confidence": 0.99, "citations": []}'
    
    # Setup rate limit mock to avoid ValueError unpacking exception
    mock_pipe = MagicMock()
    mock_pipe.execute.return_value = (None, 0)
    mock_redis.pipeline.return_value = mock_pipe
    
    mock_redis_func.return_value = mock_redis

    # Mock embedding encode to return matching vector
    app.state.embedding_service = MagicMock()
    app.state.embedding_service.encode_query.return_value = np.array([1.0] + [0.0] * 1023, dtype=np.float32)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/query",
            headers=valid_auth_headers,
            json={"query": "Who is the EKS cluster sponsor?"}
        )

    assert response.status_code == 200
    assert response.headers.get("X-Cache-Lookup") == "HIT"
    assert response.json()["answer"] == "Mocked cached RAG answer response."

@pytest.mark.asyncio
@patch("serving.dependencies.get_redis_client")
async def test_custom_error_handlers(mock_redis_func, valid_auth_headers):
    """Verify that domain exceptions route correctly to designated HTTP status codes."""
    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = {}
    mock_pipe = MagicMock()
    mock_pipe.execute.return_value = (None, 0)
    mock_redis.pipeline.return_value = mock_pipe
    mock_redis_func.return_value = mock_redis
    
    app.state.embedding_service = MagicMock()
    app.state.embedding_service.encode_query.return_value = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    
    # 1. DuplicateDocumentError -> 409
    with patch("serving.api.RAGPipeline.execute", side_effect=DuplicateDocumentError("Doc already exists")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/v1/query", headers=valid_auth_headers, json={"query": "trigger duplicate"})
        assert response.status_code == 409
        assert response.json()["error_code"] == "DUPLICATE_DOCUMENT"

    # 2. PIIDetectedError -> 451
    with patch("serving.api.RAGPipeline.execute", side_effect=PIIDetectedError("SSN leak found")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/v1/query", headers=valid_auth_headers, json={"query": "trigger pii"})
        assert response.status_code == 451
        assert response.json()["error_code"] == "PII_DETECTED"

    # 3. ModelTimeoutError -> 504
    with patch("serving.api.RAGPipeline.execute", side_effect=ModelTimeoutError("Transformer timed out")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/v1/query", headers=valid_auth_headers, json={"query": "trigger timeout"})
        assert response.status_code == 504
        assert response.json()["error_code"] == "MODEL_TIMEOUT"

@pytest.mark.asyncio
async def test_paginated_chunks_endpoint(valid_auth_headers):
    """Verify chunks pagination limit constraints and structure output."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(
            "/v1/documents/doc_sample_uuid/chunks?page=1&limit=2",
            headers=valid_auth_headers
        )
    assert response.status_code == 200
    data = response.json()
    assert "chunks" in data
    assert len(data["chunks"]) == 2
    assert data["total"] == 5
    assert data["page"] == 1


@pytest.mark.asyncio
@patch("serving.tasks.trigger_retraining_task.delay")
async def test_admin_retrain_endpoint(mock_retrain_delay):
    """Verify that POST /v1/admin/retrain enqueues retraining when authorized."""
    mock_task = MagicMock()
    mock_task.id = "mock_retrain_task_id"
    mock_retrain_delay.return_value = mock_task

    # 1. Unauthorized - missing admin key
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/admin/retrain?trigger_source=api_call")
    assert response.status_code == 401

    # 2. Unauthorized - invalid admin key
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/admin/retrain?trigger_source=api_call",
            headers={"X-Admin-Key": "wrong-key"}
        )
    assert response.status_code == 401

    # 3. Authorized - valid admin key
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/admin/retrain?trigger_source=api_call",
            headers={"X-Admin-Key": "super-admin-secret-key"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "enqueued"
    assert data["task_id"] == "mock_retrain_task_id"
    assert data["trigger_source"] == "api_call"
    mock_retrain_delay.assert_called_once_with("api_call")
