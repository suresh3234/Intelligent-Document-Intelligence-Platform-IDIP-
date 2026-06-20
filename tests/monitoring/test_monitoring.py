import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from monitoring.drift import DriftDetector, calculate_psi
from monitoring.metrics import (
    idip_documents_ingested_total,
    idip_queries_total,
    idip_model_confidence_score,
    idip_drift_score
)
from monitoring.llm_eval import LLMEvaluator
from monitoring.tracing import trace_span, setup_tracing
from serving.api import app
from serving.dependencies import engine

@pytest.fixture(autouse=True)
def setup_test_env():
    """Sets testing environment variables."""
    os.environ["IDIP_TESTING"] = "true"
    yield
    os.environ.pop("IDIP_TESTING", None)

# ==================== DATA DRIFT TESTS ====================

def test_calculate_psi():
    """Verify that PSI behaves logically on matching vs shifted arrays."""
    np.random.seed(42)
    # Similar arrays
    exp = np.random.normal(0, 1, 100)
    act = np.random.normal(0, 1, 100)
    psi_low = calculate_psi(exp, act)
    assert psi_low < 0.3

    # Shifted array
    act_shifted = np.random.normal(1.5, 1, 100)
    psi_high = calculate_psi(exp, act_shifted)
    assert psi_high > 0.5


@patch("monitoring.drift.engine")
@patch("serving.tasks.trigger_retraining_task.delay")
def test_drift_detector_alerting(mock_celery, mock_engine):
    """Verify DriftDetector calculates all metrics, flags alerts, and queues retraining tasks."""
    detector = DriftDetector()
    
    np.random.seed(42)
    
    # 1. Reference Data
    ref_data = {
        "embeddings": np.random.normal(0, 1, (20, 16)).tolist(),
        "text_lengths": [100, 150, 200, 180, 220] * 4,
        "languages": ["en", "en", "de", "en", "fr"] * 4,
        "predicted_classes": ["invoice", "contract", "invoice", "email"] * 5,
        "confidence_scores": [0.85, 0.90, 0.78, 0.95] * 5,
        "entity_rates": [2, 5, 1, 3] * 5
    }

    # 2. Current Data (High Shift to guarantee alert triggers)
    cur_data = {
        # Drastically shifted embeddings
        "embeddings": np.random.normal(3.0, 1, (20, 16)).tolist(),
        # Shifted text lengths
        "text_lengths": [800, 1200, 950, 1100, 1000] * 4,
        "languages": ["es", "es", "zh", "zh", "hi"] * 4,
        # Shifted class predictions
        "predicted_classes": ["report", "report", "other", "other"] * 5,
        "confidence_scores": [0.35, 0.40, 0.50, 0.25] * 5,
        "entity_rates": [12, 15, 20, 18] * 5
    }

    report = detector.evaluate_drift(ref_data, cur_data)

    assert report["drift_detected"] is True
    assert "embeddings" in report["affected_features"]
    assert "text_lengths" in report["affected_features"]
    assert "predicted_classes" in report["affected_features"]
    
    # Check metrics are populated
    scores = report["drift_scores"]
    assert scores["embedding_psi"] > 0.0
    assert scores["text_length_ks_p"] < 0.05
    assert scores["confidence_scores_js"] > 0.1
    
    # Ensure Celery task was enqueued
    mock_celery.assert_called_once()

# ==================== PROMETHEUS METRICS TESTS ====================

def test_prom_metrics_recording():
    """Verify Counter and Gauge increments/sets track telemetry correctly."""
    # Test Counter
    c_start = idip_documents_ingested_total.labels(source_type="pdf", status="queued")._value.get()
    idip_documents_ingested_total.labels(source_type="pdf", status="queued").inc(3)
    c_end = idip_documents_ingested_total.labels(source_type="pdf", status="queued")._value.get()
    assert c_end == c_start + 3

    # Test Gauge
    idip_model_confidence_score.labels(model_name="test_gauge").set(0.92)
    assert idip_model_confidence_score.labels(model_name="test_gauge")._value.get() == 0.92

# ==================== LLM EVALUATION TESTS ====================

def test_llm_eval_weekly():
    """Verify LLMEvaluator runs evaluations, returns scores, and logs to SQLite/Postgres."""
    evaluator = LLMEvaluator()
    
    qa_samples = [
        {
            "question": "Dummy question?",
            "contexts": ["Dummy context text block."],
            "answer": "Dummy answer.",
            "ground_truth": "Dummy ground truth."
        }
    ]

    scores = evaluator.evaluate_dataset(qa_samples, eval_source="test_suite")

    assert "faithfulness" in scores
    assert "answer_relevancy" in scores
    assert scores["faithfulness"] > 0.0
    assert scores["answer_relevancy"] <= 1.0

    # Query local SQLite/Postgres DB to check entry was logged
    with engine.connect() as conn:
        res = conn.execute(text("SELECT * FROM llm_eval_reports WHERE eval_source = 'test_suite'"))
        rows = res.fetchall()
        
    assert len(rows) > 0
    assert rows[0][2] == "test_suite"  # Column 2 is eval_source

# ==================== TRACING TESTS ====================

def test_otel_tracing():
    """Verify trace_span context manages active OpenTelemetry spans and logs custom attributes."""
    setup_tracing("test-tracer-service")
    
    with trace_span("retrieve", {"doc_id": "test_doc_123", "query_id": "test_query_456"}) as span:
        assert span.is_recording() is True
        # Otel handles context setting transparently

# ==================== ADMIN EVALUATE API ENDPOINT ====================

@pytest.mark.asyncio
async def test_admin_evaluate_endpoint():
    """Verify administrative POST /v1/admin/evaluate executes evaluations with authentication check."""
    # 1. Unauthorized (Missing admin key)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post("/v1/admin/evaluate")
    assert res.status_code == 401

    # 2. Authorized (Valid admin key)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/v1/admin/evaluate?eval_source=api_test",
            headers={"X-Admin-Key": "super-admin-secret-key"}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["eval_source"] == "api_test"
    assert "faithfulness" in data["metrics"]
