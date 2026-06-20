import os
import pytest
from datetime import datetime, timedelta
from sqlalchemy import text
from unittest.mock import patch, MagicMock

from mlops.ab_testing import ABTestingController, ABTestConfig, ABTestResult
import mlops.ab_testing

@pytest.fixture(autouse=True)
def setup_test_env():
    """Sets testing environment variables."""
    os.environ["IDIP_TESTING"] = "true"
    yield
    os.environ.pop("IDIP_TESTING", None)

@pytest.fixture
def clean_db():
    """Truncates A/B testing tables before each test."""
    queries = [
        "DELETE FROM ab_test_configs",
        "DELETE FROM ab_test_shadow_logs",
        "DELETE FROM ab_test_live_metrics",
        "DELETE FROM ab_test_results"
    ]
    with mlops.ab_testing.engine.begin() as conn:
        for q in queries:
            try:
                conn.execute(text(q))
            except Exception:
                pass
    yield

def test_create_and_get_experiment(clean_db):
    """Verify creating a new config, checking defaults, and retrieving the active experiment config."""
    controller = ABTestingController()
    
    # 1. Create shadow experiment
    config = controller.create_experiment(
        experiment_id="exp-shadow-01",
        model_a_version="v1.0.0",
        model_b_version="v1.1.0",
        mode="shadow",
        shadow_duration_hours=12
    )
    
    assert config.experiment_id == "exp-shadow-01"
    assert config.mode == "shadow"
    assert config.split_ratio == 0.0
    assert config.shadow_duration_hours == 12
    assert config.status == "running"
    
    # 2. Retrieve active config
    active = controller.get_active_config()
    assert active is not None
    assert active.experiment_id == "exp-shadow-01"
    assert active.model_a_version == "v1.0.0"
    assert active.model_b_version == "v1.1.0"
    
    # 3. Create live split experiment to overwrite active config
    config_live = controller.create_experiment(
        experiment_id="exp-live-02",
        model_a_version="v1.0.0",
        model_b_version="v1.1.0",
        mode="ab",
        split_ratio=0.10
    )
    
    active_live = controller.get_active_config()
    assert active_live is not None
    assert active_live.experiment_id == "exp-live-02"
    assert active_live.mode == "ab"
    assert active_live.split_ratio == 0.10

def test_route_request_deterministic(clean_db):
    """Verify that routing partitions traffic deterministically based on request_id hashes."""
    controller = ABTestingController()
    
    # 1. Shadow mode - all variant routings must default to production A
    controller.create_experiment("exp-shadow", "v1.0", "v1.1", mode="shadow")
    
    for i in range(20):
        res = controller.route_request(f"req-{i}")
        assert res["variant"] == "A"
        
    # 2. A/B Mode with 10% split
    controller.create_experiment("exp-ab", "v1.0", "v1.1", mode="ab", split_ratio=0.10)
    
    routes = []
    for i in range(100):
        res = controller.route_request(f"req-id-hash-string-{i}")
        routes.append(res["variant"])
        
    assert "A" in routes
    # At 10% ratio, some requests should be routed to B
    assert "B" in routes
    
    # Confirm deterministic hashing: routing same request ID again returns the exact same variant
    for i in range(20):
        req_id = f"test-req-id-{i}"
        v1 = controller.route_request(req_id)["variant"]
        v2 = controller.route_request(req_id)["variant"]
        assert v1 == v2

@patch("monitoring.metrics.idip_ab_latency_seconds.labels")
@patch("monitoring.metrics.idip_ab_response_quality.labels")
def test_shadow_mode_evaluation_promotion(mock_quality_labels, mock_latency_labels, clean_db):
    """Verify shadow metrics comparison and promotion logic based on ROUGE-L, confidence, and latency constraints."""
    mock_metric = MagicMock()
    mock_quality_labels.return_value = mock_metric
    mock_latency_labels.return_value = mock_metric
    
    controller = ABTestingController()
    exp_id = "exp-shadow-promote"
    controller.create_experiment(exp_id, "v1.0", "v1.1", mode="shadow")
    
    # Log requests where Model B outperforms or matches Model A
    # Model B latency (0.08s) <= Model A latency (0.10s)
    # Model B confidence (0.92) >= Model A confidence (0.90)
    # Model B response matches Model A closely (high ROUGE-L)
    for i in range(10):
        controller.log_shadow_request(
            experiment_id=exp_id,
            request_id=f"r-{i}",
            model_a_response="EKS cluster sponsor is the cloud infrastructure engineering unit.",
            model_b_response="EKS cluster sponsor is the cloud infrastructure engineering unit.",
            latency_a=0.10,
            latency_b=0.08,
            confidence_a=0.90,
            confidence_b=0.92
        )
        
    result = controller.evaluate_shadow_experiment(exp_id)
    assert result is not None
    assert result.winner == "B"
    assert result.metric_deltas["avg_rouge_l"] == 1.0  # Exact match
    assert result.metric_deltas["avg_confidence_diff"] > 0
    assert result.metric_deltas["p50_latency_diff"] < 0  # B is faster
    
    # Verify DB update of status
    with mlops.ab_testing.engine.connect() as conn:
        res = conn.execute(text("SELECT status FROM ab_test_configs WHERE experiment_id = :id"), {"id": exp_id})
        assert res.fetchone()[0] == "promoted"

@patch("monitoring.metrics.idip_ab_latency_seconds.labels")
@patch("monitoring.metrics.idip_ab_response_quality.labels")
def test_shadow_mode_evaluation_failed_promotion(mock_quality_labels, mock_latency_labels, clean_db):
    """Verify shadow promotion is skipped if Model B fails latency or quality constraints."""
    mock_metric = MagicMock()
    mock_quality_labels.return_value = mock_metric
    mock_latency_labels.return_value = mock_metric
    
    controller = ABTestingController()
    exp_id = "exp-shadow-fail"
    controller.create_experiment(exp_id, "v1.0", "v1.1", mode="shadow")
    
    # Model B is slower than Model A (0.15s > 0.10s)
    for i in range(10):
        controller.log_shadow_request(
            experiment_id=exp_id,
            request_id=f"r-{i}",
            model_a_response="Sponsor is cloud unit.",
            model_b_response="Sponsor is cloud unit.",
            latency_a=0.10,
            latency_b=0.15,
            confidence_a=0.90,
            confidence_b=0.92
        )
        
    result = controller.evaluate_shadow_experiment(exp_id)
    assert result is not None
    assert result.winner == "A"  # Promotion rejected because B is slower
    
    with mlops.ab_testing.engine.connect() as conn:
        res = conn.execute(text("SELECT status FROM ab_test_configs WHERE experiment_id = :id"), {"id": exp_id})
        assert res.fetchone()[0] == "completed"

def test_live_experiment_auto_rollback(clean_db):
    """Verify live experiments automatically trigger rollback when Model B error rate exceeds threshold."""
    controller = ABTestingController()
    exp_id = "exp-live-rollback"
    
    controller.create_experiment(exp_id, "v1.0", "v1.1", mode="ab", split_ratio=0.20)
    
    # Log 20 requests for Model A (0 errors)
    for i in range(20):
        controller.log_live_request(exp_id, variant="A", latency=0.1, error=False, cache_hit=False)
        
    # Log 20 requests for Model B (10 errors -> 50% error rate, significantly > 1.5 * A)
    for i in range(20):
        # 10 errors
        is_err = i < 10
        controller.log_live_request(exp_id, variant="B", latency=0.1, error=is_err, cache_hit=False)
        
    result = controller.evaluate_live_experiment(exp_id)
    assert result is not None
    assert result.winner == "A"
    assert result.rollback_count == 1
    
    # Verify split ratio is reset to 0.0 and status is rolled_back
    active = controller.get_active_config()
    assert active is None # No active experiment config (status='running') is found
    
    with mlops.ab_testing.engine.connect() as conn:
        res = conn.execute(text("SELECT split_ratio, status FROM ab_test_configs WHERE experiment_id = :id"), {"id": exp_id})
        row = res.fetchone()
        assert row[0] == 0.0
        assert row[1] == "rolled_back"

def test_live_experiment_z_test_promotion(clean_db):
    """Verify A/B z-test promotion logic triggers when Model B has statistically significant higher success rate."""
    controller = ABTestingController()
    exp_id = "exp-live-promote"
    
    controller.create_experiment(exp_id, "v1.0", "v1.1", mode="ab", split_ratio=0.50)
    
    # Model A: 100 trials, 15 errors (85% success rate)
    for i in range(100):
        is_err = i < 15
        controller.log_live_request(exp_id, variant="A", latency=0.1, error=is_err, cache_hit=False)
        
    # Model B: 100 trials, 1 error (99% success rate)
    for i in range(100):
        is_err = i < 1
        controller.log_live_request(exp_id, variant="B", latency=0.1, error=is_err, cache_hit=False)
        
    result = controller.evaluate_live_experiment(exp_id)
    assert result is not None
    assert result.winner == "B"
    assert result.metric_deltas["p_value"] < 0.05
    
    with mlops.ab_testing.engine.connect() as conn:
        res = conn.execute(text("SELECT status FROM ab_test_configs WHERE experiment_id = :id"), {"id": exp_id})
        assert res.fetchone()[0] == "promoted"

def test_split_ratio_ramping_and_limits(clean_db):
    """Verify split ratio increments by 0.05 after 2 hours and stops at limit constraints."""
    controller = ABTestingController()
    exp_id = "exp-ramp"
    
    # 1. Start with 0.10 split
    controller.create_experiment(exp_id, "v1.0", "v1.1", mode="ab", split_ratio=0.10)
    
    # Log some dummy requests to avoid regression checks halting it
    controller.log_live_request(exp_id, variant="A", latency=0.1, error=False, cache_hit=False)
    controller.log_live_request(exp_id, variant="B", latency=0.1, error=False, cache_hit=False)
    
    # Simulate time elapsed: update start_time and last_update_time to 3 hours ago
    three_hours_ago = datetime.utcnow() - timedelta(hours=3)
    with mlops.ab_testing.engine.begin() as conn:
        conn.execute(
            text("UPDATE ab_test_configs SET last_update_time = :t WHERE experiment_id = :id"),
            {"t": three_hours_ago, "id": exp_id}
        )
        
    # Ramping update
    new_ratio = controller.update_split_ratio(exp_id)
    assert new_ratio == pytest.approx(0.15, abs=1e-5)
    
    # 2. Test cap limit at 0.50
    with mlops.ab_testing.engine.begin() as conn:
        conn.execute(
            text("UPDATE ab_test_configs SET split_ratio = 0.48, last_update_time = :t WHERE experiment_id = :id"),
            {"t": three_hours_ago, "id": exp_id}
        )
        
    new_ratio_cap = controller.update_split_ratio(exp_id)
    assert new_ratio_cap == 0.50
