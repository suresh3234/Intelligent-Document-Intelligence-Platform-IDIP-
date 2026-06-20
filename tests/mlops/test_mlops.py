import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from mlops.tracking import ExperimentTracker, compare_runs
from mlops.data_versioning import track_dataset, auto_version_on_batch
from mlops.retraining import RetrainingScheduler
from config.settings import Settings

@pytest.fixture(autouse=True)
def setup_test_env():
    """Configures the execution environment variables for unit testing mocks."""
    os.environ["IDIP_TESTING"] = "true"
    yield
    os.environ.pop("IDIP_TESTING", None)

# ==================== EXPERIMENT TRACKER TESTS ====================

@patch("mlops.tracking.mlflow")
def test_experiment_tracker_context(mock_mlflow):
    """Verifies that ExperimentTracker context manager configures tracking and handles active runs."""
    mock_run = MagicMock()
    mock_mlflow.start_run.return_value = mock_run

    with ExperimentTracker(experiment_name="test-experiment", tracking_uri="http://localhost:5000") as tracker:
        tracker.log_params({"epochs": 5})
        tracker.log_metrics({"loss": 0.05})
        tracker.log_artifact("dummy_path")

    mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called_once_with({"epochs": 5})
    mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.05}, step=None)
    mock_mlflow.log_artifact.assert_called_once_with("dummy_path", artifact_path=None)


@patch("mlops.tracking.mlflow")
@patch("mlops.tracking.MlflowClient")
def test_model_registry_transition_success(mock_client_class, mock_mlflow):
    """Verifies model registration and auto-promotion to Staging when metrics pass thresholds."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    # Mock return values for latest versions query
    mock_version = MagicMock()
    mock_version.version = "2"
    mock_client.get_latest_versions.return_value = [mock_version]

    metrics = {"classifier_f1": 0.89, "rag_rouge_l": 0.45, "hallucination_rate": 0.02}

    tracker = ExperimentTracker(experiment_name="test-experiment")
    # Log model version with passing metrics
    version = tracker.log_model_version(
        model_name="test-model",
        model_object=lambda x: x,
        metrics=metrics
    )

    assert version == "2"
    # Verify that transition stage was triggered
    mock_client.transition_model_version_stage.assert_called_once_with(
        name="test-model",
        version="2",
        stage="Staging"
    )
    # Check tags were applied
    mock_client.set_model_version_tag.assert_any_call("test-model", "2", "environment", "development")


@patch("mlops.tracking.mlflow")
@patch("mlops.tracking.MlflowClient")
def test_model_registry_transition_fails_threshold(mock_client_class, mock_mlflow):
    """Verifies that model is not promoted to Staging if any metric fails to satisfy thresholds."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    mock_version = MagicMock()
    mock_version.version = "3"
    mock_client.get_latest_versions.return_value = [mock_version]

    # Metrics where classifier_f1 fails the threshold (> 0.85)
    metrics = {"classifier_f1": 0.81, "rag_rouge_l": 0.45, "hallucination_rate": 0.02}

    tracker = ExperimentTracker(experiment_name="test-experiment")
    version = tracker.log_model_version(
        model_name="test-model",
        model_object=lambda x: x,
        metrics=metrics
    )

    assert version == "3"
    # Ensure transition was NOT called
    mock_client.transition_model_version_stage.assert_not_called()


@patch("mlops.tracking.MlflowClient")
def test_run_comparison(mock_client_class):
    """Verifies compare_runs collects metrics and correctly flags the best run for each metric."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Mock runs data
    run1 = MagicMock()
    run1.info.run_id = "run1"
    run1.info.run_name = "Alpha"
    run1.data.metrics = {"f1": 0.82, "loss": 0.12}

    run2 = MagicMock()
    run2.info.run_id = "run2"
    run2.info.run_name = "Beta"
    run2.data.metrics = {"f1": 0.88, "loss": 0.08}

    mock_client.get_run.side_effect = lambda run_id: run2 if run_id == "run2" else run1

    df = compare_runs(["run1", "run2"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # f1 and loss metrics
    
    # Check that f1 best run is Beta (higher is better)
    f1_row = df[df["Metric"] == "f1"].iloc[0]
    assert "Beta" in f1_row["Best Run"]

    # Check that loss best run is Beta (lower is better)
    loss_row = df[df["Metric"] == "loss"].iloc[0]
    assert "Beta" in loss_row["Best Run"]

# ==================== DVC DATA VERSIONING TESTS ====================

@patch("mlops.data_versioning._run_dvc_command")
def test_dvc_dataset_tracking(mock_run_dvc):
    """Verifies that dataset tracking initializes, configures remote and adds path under DVC."""
    mock_run_dvc.return_value = MagicMock(stdout="idip_remote s3://dummy")

    with patch("os.path.exists", return_value=True):
        version = track_dataset(dataset_dir="data/raw", version="v1.1.0")

    assert "v1.1.0" in version
    # Check command sequence calls
    mock_run_dvc.assert_any_call(["add", "data/raw"])
    mock_run_dvc.assert_any_call(["push"])


@patch("mlops.data_versioning.track_dataset")
def test_auto_versioning_trigger(mock_track_dataset):
    """Verifies auto versioning triggers DVC updates only when the batch size exceeds 1000 items."""
    # Under limit
    res1 = auto_version_on_batch(500)
    assert res1 is None
    mock_track_dataset.assert_not_called()

    # Over limit
    auto_version_on_batch(1200)
    mock_track_dataset.assert_called_once_with("data/raw")

# ==================== RETRAINING SCHEDULER TESTS ====================

@patch("mlops.retraining.ExperimentTracker")
@patch("mlops.retraining.MlflowClient")
@patch("mlops.retraining.httpx.post")
def test_retraining_triggers_and_pipeline(mock_post, mock_client_class, mock_tracker_class):
    """Verifies retraining triggers on drift, evaluates model, updates stage, and alerts Slack."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    mock_tracker = MagicMock()
    mock_tracker_class.return_value = mock_tracker
    mock_tracker.log_model_version.return_value = "5"

    scheduler = RetrainingScheduler()

    # 1. Test Drift check trigger
    # Settings default DRIFT_ALERT_THRESHOLD is 0.15
    with patch.object(scheduler, "trigger_retraining") as mock_trigger:
        # Below threshold
        assert scheduler.check_drift(0.10) is False
        mock_trigger.assert_not_called()

        # Above threshold
        assert scheduler.check_drift(0.18) is True
        mock_trigger.assert_called_once_with(trigger_source="drift_alert_score_0.18")

    # 2. Test Labeled data check trigger
    with patch.object(scheduler, "trigger_retraining") as mock_trigger:
        # Below limit
        assert scheduler.check_labeled_data(300) is False
        mock_trigger.assert_not_called()

        # Above limit
        assert scheduler.check_labeled_data(550) is True
        mock_trigger.assert_called_once_with(trigger_source="labeled_data_accumulation_550")

    # 3. Test Full pipeline execution
    result = scheduler.trigger_retraining(trigger_source="unit_test")

    assert result["status"] == "completed"
    assert result["promoted_to_staging"] is True
    assert result["promoted_to_production"] is True
    assert result["new_model_version"] == "5"

    # Ensure stage transitions were executed
    mock_client.transition_model_version_stage.assert_called_with(
        name="idip-llm-model",
        version="5",
        stage="Production"
    )
