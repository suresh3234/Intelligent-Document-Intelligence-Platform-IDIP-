import os
import logging
import subprocess
from typing import Dict, Any, List, Optional
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from config.settings import Settings

logger = logging.getLogger("idip.mlops.tracking")

def _get_git_sha() -> str:
    """Helper to fetch the current Git commit SHA, returning 'unknown' if not in a Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"

def _get_environment() -> str:
    """Helper to fetch the environment setting."""
    try:
        settings = Settings()
        return settings.ENVIRONMENT
    except Exception:
        return "development"

class IDIPModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper to expose any custom Python model object as a PyFunc model."""
    def __init__(self, model: Any):
        self.model = model

    def predict(self, context: Any, model_input: Any) -> Any:
        if hasattr(self.model, "predict"):
            return self.model.predict(model_input)
        elif hasattr(self.model, "generate"):
            return self.model.generate(model_input)
        elif callable(self.model):
            return self.model(model_input)
        else:
            raise ValueError("Model object is not callable or lacks predict/generate methods.")

class ExperimentTracker:
    """MLflow-based experiment tracking system for IDIP."""
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run = None

    def __enter__(self) -> "ExperimentTracker":
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run()
        self.run.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.run:
            self.run.__exit__(exc_type, exc_val, exc_tb)
            self.run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs training/configuration parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs evaluations or training metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Logs a local file or directory as an MLflow artifact."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model_version(
        self,
        model_name: str,
        model_object: Any,
        metrics: Dict[str, float],
        signature: Optional[Any] = None,
        sample_input: Optional[Any] = None,
        sample_output: Optional[Any] = None,
        training_dataset_version: str = "unknown",
        base_model: str = "unknown"
    ) -> str:
        """Logs a model version, registers it, tags it, and evaluates it for Staging promotion."""
        client = MlflowClient()

        # 1. Infer signature if sample dataset details are provided
        if signature is None and sample_input is not None and sample_output is not None:
            try:
                signature = infer_signature(sample_input, sample_output)
            except Exception as e:
                logger.warning(f"Failed to infer model signature: {e}")

        # 2. Log model based on flavor or generic PyFunc fallback
        module_name = type(model_object).__module__
        
        if "sklearn" in module_name:
            mlflow.sklearn.log_model(
                model_object,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name
            )
        elif "xgboost" in module_name:
            mlflow.xgboost.log_model(
                model_object,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name
            )
        else:
            wrapper = IDIPModelWrapper(model_object)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapper,
                signature=signature,
                registered_model_name=model_name
            )

        # 3. Retrieve model registration details
        # Wait, if registered_model_name was passed, the client can fetch the version
        try:
            versions = client.get_latest_versions(model_name)
            version = versions[0].version if versions else "1"
        except Exception:
            # Fallback manually registering model
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
            model_uri = f"runs:/{run_id}/model"
            model_details = mlflow.register_model(model_uri, model_name)
            version = model_details.version

        # 4. Set MLflow version tags
        git_sha = _get_git_sha()
        env = _get_environment()
        
        client.set_model_version_tag(model_name, version, "git_sha", git_sha)
        client.set_model_version_tag(model_name, version, "training_dataset_version", training_dataset_version)
        client.set_model_version_tag(model_name, version, "base_model", base_model)
        client.set_model_version_tag(model_name, version, "environment", env)

        # 5. Check transition thresholds
        # classifier F1 > 0.85
        # rag ROUGE-L > 0.40
        # hallucination_rate < 0.05
        is_eligible = True
        
        # Check F1
        f1_keys = ["classifier_f1", "f1"]
        f1_val = next((metrics[k] for k in f1_keys if k in metrics), None)
        if f1_val is not None and f1_val <= 0.85:
            is_eligible = False

        # Check ROUGE-L
        rouge_keys = ["rag_rouge_l", "rouge_l", "ROUGE-L"]
        rouge_val = next((metrics[k] for k in rouge_keys if k in metrics), None)
        if rouge_val is not None and rouge_val <= 0.40:
            is_eligible = False

        # Check Hallucination Rate
        h_val = metrics.get("hallucination_rate", None)
        if h_val is not None and h_val >= 0.05:
            is_eligible = False

        # Auto-promote to Staging if checked metrics passed and at least one threshold was set
        has_checks = f1_val is not None or rouge_val is not None or h_val is not None
        if has_checks and is_eligible:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            logger.info(f"Model {model_name} version {version} automatically transitioned to Staging.")
        else:
            logger.info(f"Model {model_name} version {version} not eligible for Staging.")

        return version

def compare_runs(run_ids: List[str]) -> pd.DataFrame:
    """Compares metrics across multiple runs and identifies the best run for each metric."""
    client = MlflowClient()
    runs_data = []
    all_metric_keys = set()

    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            run_metrics = run.data.metrics
            all_metric_keys.update(run_metrics.keys())
            runs_data.append({
                "run_id": run_id,
                "run_name": run.info.run_name or run_id[:8],
                "metrics": run_metrics
            })
        except Exception as e:
            logger.error(f"Failed to fetch run {run_id}: {e}")

    rows = []
    for metric in sorted(all_metric_keys):
        row = {"Metric": metric}
        best_val = None
        best_run_col = None
        
        # Lower is better for error/loss/hallucination metrics
        is_lower_better = any(x in metric.lower() for x in ["loss", "error", "hallucination"])

        for run_dict in runs_data:
            val = run_dict["metrics"].get(metric, None)
            col_name = f"{run_dict['run_name']} ({run_dict['run_id'][:8]})"
            row[col_name] = val

            if val is not None:
                if best_val is None:
                    best_val = val
                    best_run_col = col_name
                else:
                    if is_lower_better:
                        if val < best_val:
                            best_val = val
                            best_run_col = col_name
                    else:
                        if val > best_val:
                            best_val = val
                            best_run_col = col_name

        row["Best Run"] = best_run_col if best_run_col else "N/A"
        rows.append(row)

    return pd.DataFrame(rows)
