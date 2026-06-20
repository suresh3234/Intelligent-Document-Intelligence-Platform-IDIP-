import os
import json
import logging
import math
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import text

from config.settings import Settings
from serving.dependencies import engine
from models.llm.eval import LLMEvaluator

logger = logging.getLogger("idip.mlops.ab_testing")

# --- Schemas ---

class ABTestConfig:
    """Configuration schema for an A/B test or shadow deployment."""
    def __init__(
        self,
        experiment_id: str,
        model_a_version: str,
        model_b_version: str,
        split_ratio: float,
        start_time: datetime,
        mode: str,
        status: str,
        shadow_duration_hours: int = 24,
        last_update_time: Optional[datetime] = None
    ):
        self.experiment_id = experiment_id
        self.model_a_version = model_a_version
        self.model_b_version = model_b_version
        self.split_ratio = split_ratio
        self.start_time = start_time
        self.mode = mode
        self.status = status
        self.shadow_duration_hours = shadow_duration_hours
        self.last_update_time = last_update_time or start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model_a_version": self.model_a_version,
            "model_b_version": self.model_b_version,
            "split_ratio": self.split_ratio,
            "start_time": self.start_time.isoformat(),
            "mode": self.mode,
            "status": self.status,
            "shadow_duration_hours": self.shadow_duration_hours,
            "last_update_time": self.last_update_time.isoformat()
        }

class ABTestResult:
    """Result summary of an completed/rolled-back A/B test or shadow run."""
    def __init__(
        self,
        experiment_id: str,
        winner: Optional[str],
        confidence: float,
        metric_deltas: Dict[str, Any],
        promoted_at: Optional[datetime] = None,
        rollback_count: int = 0
    ):
        self.experiment_id = experiment_id
        self.winner = winner
        self.confidence = confidence
        self.metric_deltas = metric_deltas
        self.promoted_at = promoted_at
        self.rollback_count = rollback_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "winner": self.winner,
            "confidence": self.confidence,
            "metric_deltas": self.metric_deltas,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "rollback_count": self.rollback_count
        }

# --- Controller ---

class ABTestingController:
    """Orchestrates model shadow routing, A/B live testing split adjustments, statistics significance tests, and rollbacks."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.evaluator = LLMEvaluator()
        self._init_db()

    def _init_db(self) -> None:
        """Bootstraps database schemas for tracking A/B testing configurations and logged outcomes."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS ab_test_configs (
                experiment_id VARCHAR(100) PRIMARY KEY,
                model_a_version VARCHAR(100) NOT NULL,
                model_b_version VARCHAR(100) NOT NULL,
                split_ratio FLOAT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                mode VARCHAR(20) NOT NULL,
                status VARCHAR(20) NOT NULL,
                shadow_duration_hours INT NOT NULL,
                last_update_time TIMESTAMP NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ab_test_shadow_logs (
                id SERIAL PRIMARY KEY,
                experiment_id VARCHAR(100) NOT NULL,
                request_id VARCHAR(100) NOT NULL,
                model_a_response TEXT,
                model_b_response TEXT,
                latency_a FLOAT,
                latency_b FLOAT,
                rouge_l FLOAT,
                confidence_a FLOAT,
                confidence_b FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ab_test_live_metrics (
                id SERIAL PRIMARY KEY,
                experiment_id VARCHAR(100) NOT NULL,
                variant VARCHAR(2) NOT NULL,
                latency FLOAT NOT NULL,
                error BOOLEAN NOT NULL,
                cache_hit BOOLEAN NOT NULL,
                quality_score FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ab_test_results (
                experiment_id VARCHAR(100) PRIMARY KEY,
                winner VARCHAR(100),
                confidence FLOAT,
                metric_deltas TEXT,
                promoted_at TIMESTAMP,
                rollback_count INT DEFAULT 0
            )
            """
        ]
        
        try:
            dialect_name = engine.dialect.name
            with engine.begin() as conn:
                for query in queries:
                    if dialect_name == "sqlite":
                        query = query.replace("id SERIAL PRIMARY KEY", "id INTEGER PRIMARY KEY AUTOINCREMENT")
                    conn.execute(text(query))
            logger.info("Successfully bootstrapped A/B testing and shadow mode database tables.")
        except Exception as e:
            logger.error(f"Failed to bootstrap A/B testing database tables: {e}")

    def create_experiment(
        self,
        experiment_id: str,
        model_a_version: str,
        model_b_version: str,
        mode: str = "shadow",
        shadow_duration_hours: int = 24,
        split_ratio: float = 0.05
    ) -> ABTestConfig:
        """Saves a new experiment config in active state."""
        config = ABTestConfig(
            experiment_id=experiment_id,
            model_a_version=model_a_version,
            model_b_version=model_b_version,
            split_ratio=0.0 if mode == "shadow" else split_ratio,
            start_time=datetime.utcnow(),
            mode=mode,
            status="running",
            shadow_duration_hours=shadow_duration_hours
        )
        
        query = """
        INSERT INTO ab_test_configs (experiment_id, model_a_version, model_b_version, split_ratio, start_time, mode, status, shadow_duration_hours, last_update_time)
        VALUES (:experiment_id, :model_a_version, :model_b_version, :split_ratio, :start_time, :mode, :status, :shadow_duration_hours, :last_update_time)
        ON CONFLICT(experiment_id) DO UPDATE SET
            model_a_version = EXCLUDED.model_a_version,
            model_b_version = EXCLUDED.model_b_version,
            split_ratio = EXCLUDED.split_ratio,
            mode = EXCLUDED.mode,
            status = EXCLUDED.status,
            last_update_time = EXCLUDED.last_update_time
        """
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(query),
                    {
                        "experiment_id": config.experiment_id,
                        "model_a_version": config.model_a_version,
                        "model_b_version": config.model_b_version,
                        "split_ratio": config.split_ratio,
                        "start_time": config.start_time,
                        "mode": config.mode,
                        "status": config.status,
                        "shadow_duration_hours": config.shadow_duration_hours,
                        "last_update_time": config.last_update_time
                    }
                )
            logger.info(f"Experiment {experiment_id} successfully created and registered.")
        except Exception as e:
            logger.error(f"Failed to save experiment config for {experiment_id}: {e}")
            
        return config

    def get_active_config(self) -> Optional[ABTestConfig]:
        """Returns the currently active config (status='running')."""
        query = """
        SELECT experiment_id, model_a_version, model_b_version, split_ratio, start_time, mode, status, shadow_duration_hours, last_update_time
        FROM ab_test_configs
        WHERE status = 'running'
        ORDER BY start_time DESC
        LIMIT 1
        """
        try:
            with engine.connect() as conn:
                res = conn.execute(text(query)).fetchone()
                if res:
                    # Handle SQLite vs Postgres timestamp deserialization
                    start_time = res[4]
                    if isinstance(start_time, str):
                        # clean fractional seconds for iso parsing
                        start_time = datetime.fromisoformat(start_time.split(".")[0])
                    last_update_time = res[8]
                    if isinstance(last_update_time, str):
                        last_update_time = datetime.fromisoformat(last_update_time.split(".")[0])
                        
                    return ABTestConfig(
                        experiment_id=res[0],
                        model_a_version=res[1],
                        model_b_version=res[2],
                        split_ratio=float(res[3]),
                        start_time=start_time,
                        mode=res[5],
                        status=res[6],
                        shadow_duration_hours=int(res[7]),
                        last_update_time=last_update_time
                    )
        except Exception as e:
            logger.error(f"Failed to query active experiment config: {e}")
        return None

    def route_request(self, request_id: str) -> Dict[str, Any]:
        """
        Determines routing variant A (Production) or B (Challenger) based on deterministic hashing.
        Returns variant name and active configuration.
        """
        config = self.get_active_config()
        if not config:
            return {"variant": "A", "config": None}

        if config.mode == "shadow":
            # Shadow mode routes all user responses to production Model A
            return {"variant": "A", "config": config}

        # A/B Live split mode
        hash_val = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
        percentile = (hash_val % 10000) / 10000.0
        variant = "B" if percentile < config.split_ratio else "A"
        return {"variant": variant, "config": config}

    def log_shadow_request(
        self,
        experiment_id: str,
        request_id: str,
        model_a_response: str,
        model_b_response: str,
        latency_a: float,
        latency_b: float,
        confidence_a: float,
        confidence_b: float
    ) -> None:
        """Saves a shadow execution comparison run, calculating ROUGE-L."""
        rouge_l = self.evaluator.compute_rouge_l(model_b_response, model_a_response)
        
        query = """
        INSERT INTO ab_test_shadow_logs (experiment_id, request_id, model_a_response, model_b_response, latency_a, latency_b, rouge_l, confidence_a, confidence_b)
        VALUES (:experiment_id, :request_id, :model_a_response, :model_b_response, :latency_a, :latency_b, :rouge_l, :confidence_a, :confidence_b)
        """
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(query),
                    {
                        "experiment_id": experiment_id,
                        "request_id": request_id,
                        "model_a_response": model_a_response,
                        "model_b_response": model_b_response,
                        "latency_a": latency_a,
                        "latency_b": latency_b,
                        "rouge_l": rouge_l,
                        "confidence_a": confidence_a,
                        "confidence_b": confidence_b
                    }
                )
            # Record prometheus metrics
            from monitoring.metrics import idip_ab_latency_seconds, idip_ab_response_quality
            idip_ab_latency_seconds.labels(experiment_id=experiment_id, variant="A").observe(latency_a)
            idip_ab_latency_seconds.labels(experiment_id=experiment_id, variant="B").observe(latency_b)
            idip_ab_response_quality.labels(experiment_id=experiment_id, variant="B").set(rouge_l)
            
        except Exception as e:
            logger.error(f"Failed to log shadow request comparisons: {e}")

    def log_live_request(
        self,
        experiment_id: str,
        variant: str,
        latency: float,
        error: bool,
        cache_hit: bool,
        quality_score: Optional[float] = None
    ) -> None:
        """Records live experiment metric events."""
        query = """
        INSERT INTO ab_test_live_metrics (experiment_id, variant, latency, error, cache_hit, quality_score)
        VALUES (:experiment_id, :variant, :latency, :error, :cache_hit, :quality_score)
        """
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(query),
                    {
                        "experiment_id": experiment_id,
                        "variant": variant,
                        "latency": latency,
                        "error": error,
                        "cache_hit": cache_hit,
                        "quality_score": quality_score
                    }
                )
            
            # Integrate with monitoring/metrics.py counter triggers
            from monitoring.metrics import (
                idip_ab_queries_total, idip_ab_latency_seconds,
                idip_ab_cache_hits_total, idip_ab_cache_misses_total
            )
            
            status_code = "500" if error else "200"
            idip_ab_queries_total.labels(experiment_id=experiment_id, variant=variant, status=status_code).inc()
            idip_ab_latency_seconds.labels(experiment_id=experiment_id, variant=variant).observe(latency)
            
            if cache_hit:
                idip_ab_cache_hits_total.labels(experiment_id=experiment_id, variant=variant).inc()
            else:
                idip_ab_cache_misses_total.labels(experiment_id=experiment_id, variant=variant).inc()
                
        except Exception as e:
            logger.error(f"Failed to log live experiment request metrics: {e}")

    def evaluate_shadow_experiment(self, experiment_id: str) -> Optional[ABTestResult]:
        """
        Evaluates shadow runs on: ROUGE-L, confidence, and latency p50/p95/p99.
        Promotes challenger Model B if all metrics >= Model A (latencies of B <= A).
        """
        query = """
        SELECT latency_a, latency_b, confidence_a, confidence_b, rouge_l
        FROM ab_test_shadow_logs
        WHERE experiment_id = :experiment_id
        """
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(query), {"experiment_id": experiment_id}).fetchall()
                
            if not rows:
                logger.warning(f"No shadow logs found to evaluate for experiment {experiment_id}.")
                return None
                
            df = pd.DataFrame(rows, columns=["latency_a", "latency_b", "confidence_a", "confidence_b", "rouge_l"])
            
            # Latency calculations
            p50_a, p95_a, p99_a = np.percentile(df["latency_a"], [50, 95, 99])
            p50_b, p95_b, p99_b = np.percentile(df["latency_b"], [50, 95, 99])
            
            avg_conf_a = float(df["confidence_a"].mean())
            avg_conf_b = float(df["confidence_b"].mean())
            avg_rouge = float(df["rouge_l"].mean())
            
            # Decision metrics
            latency_ok = (p50_b <= p50_a) and (p95_b <= p95_a) and (p99_b <= p99_a)
            conf_ok = avg_conf_b >= avg_conf_a
            # Since ROUGE-L represents semantic overlap of B with A (current production baseline),
            # we want B to align well (typically ROUGE-L > 0.40 baseline or >= 0.70 high similarity)
            rouge_ok = avg_rouge >= 0.40 
            
            promoted = latency_ok and conf_ok and rouge_ok
            winner = "B" if promoted else "A"
            
            metric_deltas = {
                "p50_latency_diff": float(p50_b - p50_a),
                "p95_latency_diff": float(p95_b - p95_a),
                "p99_latency_diff": float(p99_b - p99_a),
                "avg_confidence_diff": float(avg_conf_b - avg_conf_a),
                "avg_rouge_l": avg_rouge
            }
            
            result = ABTestResult(
                experiment_id=experiment_id,
                winner=winner,
                confidence=1.0 if promoted else 0.0, # Shadow does not use z-test confidence
                metric_deltas=metric_deltas,
                promoted_at=datetime.utcnow() if promoted else None
            )
            
            self._save_ab_result(result)
            
            # Update status in config table
            new_status = "promoted" if promoted else "completed"
            self._update_experiment_status(experiment_id, new_status)
            
            return result
        except Exception as e:
            logger.error(f"Error during shadow evaluation of {experiment_id}: {e}")
        return None

    def evaluate_live_experiment(self, experiment_id: str) -> Optional[ABTestResult]:
        """
        Evaluates a live A/B experiment. 
        1. Checks auto-rollback trigger (Model B error rate > Model A * 1.5).
        2. Computes two-proportion z-test on success rate. Promotes if p < 0.05.
        """
        # Fetch configurations
        config = self.get_active_config()
        if not config or config.experiment_id != experiment_id:
            logger.warning(f"Experiment {experiment_id} is not the active running configuration.")
            return None
            
        # Get query summary
        query = """
        SELECT variant, latency, error, cache_hit
        FROM ab_test_live_metrics
        WHERE experiment_id = :experiment_id
        """
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(query), {"experiment_id": experiment_id}).fetchall()
                
            if not rows:
                logger.warning(f"No live metrics found to evaluate for experiment {experiment_id}.")
                return None
                
            df = pd.DataFrame(rows, columns=["variant", "latency", "error", "cache_hit"])
            
            # Splits
            df_a = df[df["variant"] == "A"]
            df_b = df[df["variant"] == "B"]
            
            n_a = len(df_a)
            n_b = len(df_b)
            
            if n_a < 10 or n_b < 10:
                logger.info(f"Insufficient samples (A: {n_a}, B: {n_b}) to run live A/B evaluation yet.")
                return None
                
            errors_a = int(df_a["error"].sum())
            errors_b = int(df_b["error"].sum())
            
            err_rate_a = errors_a / n_a
            err_rate_b = errors_b / n_b
            
            # 1. Check Auto-Rollback Threshold
            # If B's error rate is significantly worse than A's by a factor of 1.5, trigger rollback.
            # Add small epsilon to handle zero errors
            if err_rate_b > (err_rate_a * 1.5) and err_rate_b > 0.01:
                logger.warning(f"🚨 AUTO-ROLLBACK TRIGGERED: Model B error rate ({err_rate_b:.4f}) exceeds Model A ({err_rate_a:.4f}) by >1.5x. Reverting.")
                self.rollback_experiment(experiment_id)
                
                result = ABTestResult(
                    experiment_id=experiment_id,
                    winner="A",
                    confidence=1.0,
                    metric_deltas={"error_rate_a": err_rate_a, "error_rate_b": err_rate_b, "reason": "auto_rollback_error_rate"},
                    rollback_count=1
                )
                self._save_ab_result(result)
                return result
                
            # 2. Run two-proportion z-test on Success Rate (success = not error)
            success_a = n_a - errors_a
            success_b = n_b - errors_b
            
            p_a = success_a / n_a
            p_b = success_b / n_b
            
            p_pooled = (success_a + success_b) / (n_a + n_b)
            
            z_stat = 0.0
            p_value = 1.0
            
            if 0.0 < p_pooled < 1.0:
                se = math.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n_a + 1.0 / n_b))
                if se > 0:
                    z_stat = (p_b - p_a) / se
                    # Two-tailed test
                    p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
            
            # Promotion triggers if Model B is significantly better than A (p < 0.05 and success proportion of B > A)
            promoted = (p_value < 0.05) and (p_b > p_a)
            winner = "B" if promoted else ( "A" if p_value < 0.05 and p_b < p_a else None )
            
            metric_deltas = {
                "success_rate_a": p_a,
                "success_rate_b": p_b,
                "latency_avg_a": float(df_a["latency"].mean()),
                "latency_avg_b": float(df_b["latency"].mean()),
                "cache_hit_a": float(df_a["cache_hit"].mean()),
                "cache_hit_b": float(df_b["cache_hit"].mean()),
                "z_statistic": z_stat,
                "p_value": p_value
            }
            
            if winner or promoted:
                new_status = "promoted" if promoted else "completed"
                self._update_experiment_status(experiment_id, new_status)
                
                result = ABTestResult(
                    experiment_id=experiment_id,
                    winner=winner,
                    confidence=float(1.0 - p_value),
                    metric_deltas=metric_deltas
                )
                self._save_ab_result(result)
                return result
                
            return None
            
        except Exception as e:
            logger.error(f"Failed live experiment evaluation for {experiment_id}: {e}")
        return None

    def rollback_experiment(self, experiment_id: str) -> None:
        """Sets the experiment configuration split ratio to 0.0 and transitions status to 'rolled_back'."""
        query = """
        UPDATE ab_test_configs
        SET split_ratio = 0.0, status = 'rolled_back', last_update_time = :now
        WHERE experiment_id = :experiment_id
        """
        try:
            with engine.begin() as conn:
                conn.execute(text(query), {"experiment_id": experiment_id, "now": datetime.utcnow()})
            logger.warning(f"Experiment {experiment_id} successfully rolled back to 100% Model A.")
        except Exception as e:
            logger.error(f"Failed executing database rollback for {experiment_id}: {e}")

    def update_split_ratio(self, experiment_id: str) -> float:
        """
        Increases split ratio by 5% (+0.05) every 2 hours if no regressions are found.
        Cap live split ratio at 0.50 (50% traffic allocation).
        """
        config = self.get_active_config()
        if not config or config.experiment_id != experiment_id or config.mode == "shadow":
            return 0.0
            
        now = datetime.utcnow()
        time_elapsed = now - config.last_update_time
        
        # Verify 2-hour threshold
        if time_elapsed < timedelta(hours=2):
            logger.info(f"Insufficient elapsed time ({time_elapsed.total_seconds() / 3600:.2f}h) since last split adjustment. Skipping ramp-up.")
            return config.split_ratio

        # Check regressions before ramping
        query = """
        SELECT variant, error
        FROM ab_test_live_metrics
        WHERE experiment_id = :experiment_id
        """
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(query), {"experiment_id": experiment_id}).fetchall()
                
            if rows:
                df = pd.DataFrame(rows, columns=["variant", "error"])
                df_a = df[df["variant"] == "A"]
                df_b = df[df["variant"] == "B"]
                
                n_a = len(df_a)
                n_b = len(df_b)
                
                if n_a > 0 and n_b > 0:
                    err_a = df_a["error"].sum() / n_a
                    err_b = df_b["error"].sum() / n_b
                    
                    if err_b > (err_a * 1.5) and err_b > 0.01:
                        logger.warning(f"Regression detected (Model B error {err_b:.4f} > Model A {err_a:.4f} * 1.5). Halting ramp-up and initiating rollback.")
                        self.rollback_experiment(experiment_id)
                        return 0.0

            # Increment split ratio by 0.05 up to a maximum of 0.50
            new_ratio = min(0.50, config.split_ratio + 0.05)
            
            update_query = """
            UPDATE ab_test_configs
            SET split_ratio = :new_ratio, last_update_time = :now
            WHERE experiment_id = :experiment_id
            """
            with engine.begin() as conn:
                conn.execute(text(update_query), {"experiment_id": experiment_id, "new_ratio": new_ratio, "now": now})
                
            logger.info(f"Ramped up split ratio for experiment {experiment_id} to {new_ratio:.2f}.")
            return new_ratio
            
        except Exception as e:
            logger.error(f"Failed ramping up split ratio: {e}")
            
        return config.split_ratio

    def _save_ab_result(self, result: ABTestResult) -> None:
        query = """
        INSERT INTO ab_test_results (experiment_id, winner, confidence, metric_deltas, promoted_at, rollback_count)
        VALUES (:experiment_id, :winner, :confidence, :metric_deltas, :promoted_at, :rollback_count)
        ON CONFLICT(experiment_id) DO UPDATE SET
            winner = EXCLUDED.winner,
            confidence = EXCLUDED.confidence,
            metric_deltas = EXCLUDED.metric_deltas,
            promoted_at = EXCLUDED.promoted_at,
            rollback_count = EXCLUDED.rollback_count
        """
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(query),
                    {
                        "experiment_id": result.experiment_id,
                        "winner": result.winner,
                        "confidence": result.confidence,
                        "metric_deltas": json.dumps(result.metric_deltas),
                        "promoted_at": result.promoted_at,
                        "rollback_count": result.rollback_count
                    }
                )
            logger.info(f"Recorded results for A/B experiment {result.experiment_id} successfully.")
        except Exception as e:
            logger.error(f"Failed to record A/B experiment results: {e}")

    def _update_experiment_status(self, experiment_id: str, status: str) -> None:
        query = """
        UPDATE ab_test_configs
        SET status = :status, last_update_time = :now
        WHERE experiment_id = :experiment_id
        """
        try:
            with engine.begin() as conn:
                conn.execute(text(query), {"experiment_id": experiment_id, "status": status, "now": datetime.utcnow()})
            logger.info(f"Updated experiment {experiment_id} status to '{status}'.")
        except Exception as e:
            logger.error(f"Failed to update experiment status: {e}")
