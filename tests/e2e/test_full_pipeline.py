import os
import time
import json
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
from sqlalchemy import text

from serving.api import app
from serving.worker import celery_app
from serving.dependencies import engine, get_redis_client
from monitoring.drift import DriftDetector
from rag.vector_store import WeaviateBackend
from config import settings

# Configure Celery to execute tasks synchronously for integration testing
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True

def create_mock_jwt(api_key: str, secret: str = "idip_secret_key_1234567890", expired: bool = False) -> str:
    import base64
    import hmac
    import hashlib
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
def auth_headers():
    # Bypass timing/auth JWT middleware by using mock secret or mocking validation
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
    app.state.guardrail_checker.check_hallucination.return_value = (0.95, False)
    app.state.guardrail_checker.scan_pii.side_effect = lambda x: x

@pytest.mark.asyncio
async def test_happy_path_ingest_to_query(
    postgres_service, redis_service, weaviate_service, s3_client, auth_headers
):
    """Test 1: Happy path - Ingestion of an invoice PDF to query retrieval."""
    # 1. Initialize Weaviate backend onto the app state
    app.state.vector_store = WeaviateBackend(url=weaviate_service)
    
    # Mock model router outputs to simulate PDF text, classifications, and entities
    app.state.classifier_service.predict.return_value = MagicMock(predicted_class="invoice", confidence=0.98)
    app.state.ner_service.extract_entities.return_value = [{"text": "$1500.00", "label": "MONEY", "start": 10, "end": 18}]
    
    # Mock LLM generation for final answer
    mock_llm_response = {
        "answer": "The total amount on this invoice is $1500.00.",
        "confidence": 0.95,
        "citations": [{"chunk_id": "c-0", "text": "Total: $1500.00"}]
    }
    app.state.llm_service.generate.return_value = mock_llm_response
    
    # 2. Ingest document
    file_payload = ("invoice.pdf", b"Total amount is $1500.00. Paid via bank transfer.", "application/pdf")
    form_data = {
        "metadata": '{"source_uri": "s3://idip-raw-data/invoice.pdf", "language": "en"}'
    }
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        ingest_res = await ac.post(
            "/v1/documents/ingest",
            headers=auth_headers,
            files={"file": file_payload},
            data=form_data
        )
        
    assert ingest_res.status_code == 200
    doc_id = ingest_res.json()["doc_id"]
    assert doc_id is not None
    
    # 3. Poll document status (with eager Celery, it should complete immediately)
    status = "queued"
    for _ in range(6):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            status_res = await ac.get(f"/v1/documents/{doc_id}", headers=auth_headers)
        assert status_res.status_code == 200
        status = status_res.json()["status"]
        if status == "completed":
            break
        time.sleep(1)
        
    assert status == "completed"
    
    # 4. Query the ingested document
    from rag.models import Citation
    with patch("serving.api.RAGPipeline") as mock_rag_pipeline_class:
        mock_pipeline_instance = MagicMock()
        mock_res = MagicMock()
        mock_res.answer = "The total amount on this invoice is $1500.00."
        mock_res.confidence = 0.95
        mock_res.citations = [Citation(
            doc_id="doc-0",
            doc_id_short="doc-0",
            source_uri="s3://idip-raw-data/invoice.pdf",
            text_snippet="Total: $1500.00"
        )]
        mock_res.reranked_chunks = []
        mock_pipeline_instance.execute.return_value = mock_res
        mock_rag_pipeline_class.return_value = mock_pipeline_instance
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            query_res = await ac.post(
                "/v1/query",
                headers=auth_headers,
                json={"query": "What is the total amount on this invoice?"}
            )
        
    assert query_res.status_code == 200
    data = query_res.json()
    assert "$1500.00" in data["answer"]
    assert data["confidence"] > 0.7
    assert len(data["citations"]) > 0

@pytest.mark.asyncio
async def test_duplicate_document_rejection(
    postgres_service, redis_service, weaviate_service, s3_client, auth_headers
):
    """Test 2: Re-ingesting the same document must return 409 Conflict (DuplicateDocumentError)."""
    file_payload = ("duplicate.pdf", b"Unique file content block.", "application/pdf")
    form_data = {
        "metadata": '{"source_uri": "s3://idip-raw-data/duplicate.pdf", "language": "en"}'
    }
    
    # First Ingest
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res1 = await ac.post(
            "/v1/documents/ingest",
            headers=auth_headers,
            files={"file": file_payload},
            data=form_data
        )
    assert res1.status_code == 200
    
    # Second Ingest (Same file, triggers Redis deduplication lock)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res2 = await ac.post(
            "/v1/documents/ingest",
            headers=auth_headers,
            files={"file": file_payload},
            data=form_data
        )
    # The deduplication logic throws DuplicateDocumentError, which routes to 409 Conflict
    assert res2.status_code == 409
    assert res2.json()["error_code"] == "DUPLICATE_DOCUMENT"

@pytest.mark.asyncio
async def test_semantic_cache_hit(
    postgres_service, redis_service, weaviate_service, auth_headers
):
    """Test 3: Semantic Cache hit for paraphrased queries."""
    get_redis_client().flushall()
    q1 = "Who is the EKS cluster lead sponsor?"
    q2 = "Who leading the EKS sponsor cluster?"
    
    # Mock Embeddings to return similar vectors (cosine similarity >= 0.95)
    mock_vector = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    app.state.embedding_service.encode_query.return_value = mock_vector
    
    # Mock LLM generation
    app.state.llm_service.generate.return_value = {
        "answer": "The sponsor is cloud division team.",
        "confidence": 0.95,
        "citations": []
    }
    
    # 1. First query (Misses cache, runs LLM, caches result)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res1 = await ac.post(
            "/v1/query",
            headers=auth_headers,
            json={"query": q1}
        )
    assert res1.status_code == 200
    assert res1.headers.get("X-Cache-Lookup") != "HIT"
    
    # 2. Second equivalent query (Hits cache, returns instantly)
    start_time = time.time()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res2 = await ac.post(
            "/v1/query",
            headers=auth_headers,
            json={"query": q2}
        )
    latency = (time.time() - start_time) * 1000  # in ms
    
    assert res2.status_code == 200
    assert res2.headers.get("X-Cache-Lookup") == "HIT"
    assert latency < 50.0  # Must be fast under 50ms

@pytest.mark.asyncio
async def test_guardrail_triggers(
    postgres_service, redis_service, weaviate_service, auth_headers
):
    """Test 4: Verify that queries triggering PII exceptions return PII_DETECTED error code."""
    # Mock Guardrail Checker to throw PII exception
    from models.guardrails import PIIDetectedError
    app.state.guardrail_checker.check_input.side_effect = PIIDetectedError("Input contains SSN")
    
    # Return orthogonal vector to ensure a cache miss
    app.state.embedding_service.encode_query.return_value = np.array([0.0, 1.0] + [0.0] * 1022, dtype=np.float32)
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/v1/query",
            headers=auth_headers,
            json={"query": "My SSN is 000-12-3456"}
        )
        
    assert res.status_code == 451
    assert res.json()["error_code"] == "PII_DETECTED"
    app.state.guardrail_checker.check_input.side_effect = None  # Reset mock

@pytest.mark.asyncio
@patch("serving.tasks.trigger_retraining_task.delay")
async def test_model_drift_simulation(
    mock_retrain, postgres_service, redis_service, weaviate_service
):
    """Test 5: Feed 1000 drifted embedding elements and verify retraining triggers."""
    detector = DriftDetector()
    
    np.random.seed(42)
    # Generate reference distribution (normal)
    ref_data = {
        "embeddings": np.random.normal(0, 1, (20, 16)).tolist(),
        "text_lengths": [100, 150, 200] * 5,
        "languages": ["en", "en", "fr"] * 5,
        "predicted_classes": ["invoice", "contract"] * 5,
        "confidence_scores": [0.85, 0.90] * 5,
        "entity_rates": [2, 4] * 5
    }
    
    # Generate heavily drifted current distribution (shifted mean)
    cur_data = {
        "embeddings": np.random.normal(4.5, 1, (20, 16)).tolist(),
        "text_lengths": [1200, 1500, 1800] * 5,
        "languages": ["es", "zh", "hi"] * 5,
        "predicted_classes": ["report", "report"] * 5,
        "confidence_scores": [0.30, 0.25] * 5,
        "entity_rates": [15, 20] * 5
    }
    
    report = detector.evaluate_drift(ref_data, cur_data)
    
    assert report["drift_detected"] is True
    # Ensure retraining scheduler Celery task was enqueued
    mock_retrain.assert_called_once()

@pytest.mark.asyncio
async def test_resilience_chaos_redis(
    postgres_service, redis_service, weaviate_service, auth_headers
):
    """Test 6: Resilience - Redis failure mid-request returns 504 within timeout, recovery check."""
    # 1. Store host details
    orig_host = settings.REDIS_HOST
    orig_port = settings.REDIS_PORT
    
    # Mock LLM generation
    app.state.llm_service.generate.return_value = {
        "answer": "Resilient output.",
        "confidence": 0.95,
        "citations": []
    }
    app.state.embedding_service.encode_query.return_value = np.array([0.0, 0.0, 1.0] + [0.0] * 1021, dtype=np.float32)
    
    with patch("serving.api.RAGPipeline") as mock_rag_pipeline_class:
        mock_pipeline_instance = MagicMock()
        mock_res = MagicMock()
        mock_res.answer = "Resilient output."
        mock_res.confidence = 0.95
        mock_res.citations = []
        mock_res.reranked_chunks = []
        mock_pipeline_instance.execute.return_value = mock_res
        mock_rag_pipeline_class.return_value = mock_pipeline_instance

        # 2. Kill connection (point settings to dead port simulating timeout/disconnect)
        settings.REDIS_HOST = "127.0.0.1"
        settings.REDIS_PORT = 9999  # Non-existent redis port
        
        # Verify that query triggers a timeout fallback/handled 504 or equivalent HTTP error
        # (FastAPI middleware handles Redis connection failures gracefully by raising timeout/504)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.post(
                "/v1/query",
                headers=auth_headers,
                json={"query": "Test query with dead Redis"}
            )
            
        assert res.status_code in (504, 503, 408)  # Gateway Timeout / Service Unavailable / Request Timeout
        
        # 3. Restore connection (recovery)
        settings.REDIS_HOST = orig_host
        settings.REDIS_PORT = orig_port
        
        # System should recover immediately without restart
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res_recovered = await ac.post(
                "/v1/query",
                headers=auth_headers,
                json={"query": "Test query after restoring Redis"}
            )
            
        assert res_recovered.status_code == 200
        assert res_recovered.json()["answer"] == "Resilient output."
