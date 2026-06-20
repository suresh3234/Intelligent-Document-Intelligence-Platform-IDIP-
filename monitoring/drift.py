import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sqlalchemy import text

from config.settings import Settings
from serving.dependencies import engine

logger = logging.getLogger("idip.monitoring.drift")

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between reference and production windows."""
    # Bin both arrays using the same boundaries
    expected_counts, bins = np.histogram(expected, bins=num_bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Normalize to probabilities
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Add small epsilon to avoid log(0) or division by zero
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0.0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0.0, eps, actual_pct)

    # PSI calculation formula
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

class DriftDetector:
    """Detects feature (input) and concept (output) distribution drift in production document traffic."""

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.db_url = f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
        self._init_database()

    def _init_database(self) -> None:
        """Bootstraps the drift_reports database table."""
        query = """
        CREATE TABLE IF NOT EXISTS drift_reports (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            drift_detected BOOLEAN NOT NULL,
            drift_scores TEXT NOT NULL,
            affected_features TEXT NOT NULL
        )
        """
        try:
            # Fallback for SQLite in local development testing
            dialect_name = engine.dialect.name
            if dialect_name == "sqlite":
                query = query.replace("id SERIAL PRIMARY KEY", "id INTEGER PRIMARY KEY AUTOINCREMENT")
            with engine.begin() as conn:
                conn.execute(text(query))
        except Exception as e:
            logger.error(f"Failed to bootstrap drift_reports table: {e}")

    def evaluate_drift(
        self,
        reference_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compares reference data distributions (e.g. last 7 days) against current window data.
        Returns a DriftReport dictionary containing detections, metrics, and affected keys.
        """
        drift_scores = {}
        affected_features = []
        drift_detected = False

        # --- FEATURE DRIFT ---

        # 1. Embedding Distribution Shift (PCA to 10 dimensions + PSI calculation)
        ref_embs = np.array(reference_data.get("embeddings", []))
        cur_embs = np.array(current_data.get("embeddings", []))
        
        if len(ref_embs) > 10 and len(cur_embs) > 10:
            pca = PCA(n_components=10)
            ref_pca = pca.fit_transform(ref_embs)
            cur_pca = pca.transform(cur_embs)
            
            psi_dims = []
            for d in range(10):
                psi_dims.append(calculate_psi(ref_pca[:, d], cur_pca[:, d]))
            max_emb_psi = float(np.max(psi_dims))
            drift_scores["embedding_psi"] = max_emb_psi
            
            # Alert trigger
            if max_emb_psi > self.settings.DRIFT_ALERT_THRESHOLD:
                affected_features.append("embeddings")
                drift_detected = True
        else:
            drift_scores["embedding_psi"] = 0.0

        # 2. Text Length Distribution Shift (Kolmogorov-Smirnov test)
        ref_len = np.array(reference_data.get("text_lengths", []))
        cur_len = np.array(current_data.get("text_lengths", []))

        if len(ref_len) > 0 and len(cur_len) > 0:
            ks_stat, p_val = ks_2samp(ref_len, cur_len)
            drift_scores["text_length_ks_p"] = float(p_val)
            
            if p_val < 0.05:
                affected_features.append("text_lengths")
                drift_detected = True
        else:
            drift_scores["text_length_ks_p"] = 1.0

        # 3. Language Distribution Shift (Chi-square contingency test)
        ref_lang = reference_data.get("languages", [])
        cur_lang = current_data.get("languages", [])

        if len(ref_lang) > 0 and len(cur_lang) > 0:
            ref_series = pd.Series(ref_lang).value_counts()
            cur_series = pd.Series(cur_lang).value_counts()
            
            # Align categories
            all_langs = list(set(ref_series.index) | set(cur_series.index))
            ref_counts = [ref_series.get(lang, 0) + 1 for lang in all_langs] # laplace add-one smoothing
            cur_counts = [cur_series.get(lang, 0) + 1 for lang in all_langs]
            
            contingency_table = [ref_counts, cur_counts]
            try:
                chi2, chi2_p, dof, ex = chi2_contingency(contingency_table)
                drift_scores["language_chi2_p"] = float(chi2_p)
                if chi2_p < 0.05:
                    affected_features.append("languages")
                    drift_detected = True
            except Exception as e:
                logger.warning(f"Failed to calculate language chi2: {e}")
                drift_scores["language_chi2_p"] = 1.0
        else:
            drift_scores["language_chi2_p"] = 1.0

        # --- CONCEPT DRIFT (OUTPUT DRIFT) ---

        # Helper to compute Jensen-Shannon divergence between discrete categorical counts
        def compute_js_divergence_categorical(ref_list: List[str], cur_list: List[str]) -> float:
            if not ref_list or not cur_list:
                return 0.0
            r_s = pd.Series(ref_list).value_counts(normalize=True)
            c_s = pd.Series(cur_list).value_counts(normalize=True)
            all_cats = list(set(r_s.index) | set(c_s.index))
            p = np.array([r_s.get(cat, 0.0) for cat in all_cats])
            q = np.array([c_s.get(cat, 0.0) for cat in all_cats])
            # Add eps
            p = p / np.sum(p)
            q = q / np.sum(q)
            dist = jensenshannon(p, q)
            return float(dist ** 2) if not np.isnan(dist) else 0.0

        # 4. Predicted Class Distribution Shift
        ref_classes = reference_data.get("predicted_classes", [])
        cur_classes = current_data.get("predicted_classes", [])
        class_js = compute_js_divergence_categorical(ref_classes, cur_classes)
        drift_scores["predicted_classes_js"] = class_js
        if class_js > 0.1:
            affected_features.append("predicted_classes")
            drift_detected = True

        # 5. Confidence Score Distribution Shift
        ref_conf = np.array(reference_data.get("confidence_scores", []))
        cur_conf = np.array(current_data.get("confidence_scores", []))
        if len(ref_conf) > 0 and len(cur_conf) > 0:
            # Bin into 10 bins between 0.0 and 1.0
            bins = np.linspace(0.0, 1.0, 11)
            p_hist, _ = np.histogram(ref_conf, bins=bins, density=True)
            q_hist, _ = np.histogram(cur_conf, bins=bins, density=True)
            p_hist = np.where(p_hist == 0.0, 1e-5, p_hist)
            q_hist = np.where(q_hist == 0.0, 1e-5, q_hist)
            p_hist /= np.sum(p_hist)
            q_hist /= np.sum(q_hist)
            conf_js = jensenshannon(p_hist, q_hist)
            conf_js_val = float(conf_js ** 2) if not np.isnan(conf_js) else 0.0
            drift_scores["confidence_scores_js"] = conf_js_val
            if conf_js_val > 0.1:
                affected_features.append("confidence_scores")
                drift_detected = True
        else:
            drift_scores["confidence_scores_js"] = 0.0

        # 6. Entity Extraction Rate Shift (Average entities per document)
        ref_ent = reference_data.get("entity_rates", [])
        cur_ent = current_data.get("entity_rates", [])
        ent_js = compute_js_divergence_categorical(
            [str(x) for x in ref_ent],
            [str(x) for x in cur_ent]
        )
        drift_scores["entity_extraction_rates_js"] = ent_js
        if ent_js > 0.1:
            affected_features.append("entity_extraction_rates")
            drift_detected = True

        # Compile final DriftReport
        report = {
            "drift_detected": drift_detected,
            "drift_scores": drift_scores,
            "affected_features": affected_features,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Store report in PostgreSQL drift_reports table
        self._save_drift_report(report)

        # Trigger retraining if drift detected
        if drift_detected:
            logger.warning(f"Drift detected in features: {affected_features}. Triggering retraining scheduler task...")
            try:
                from serving.tasks import trigger_retraining_task
                trigger_retraining_task.delay(trigger_source=f"data_drift_alert_{','.join(affected_features)}")
            except Exception as e:
                logger.error(f"Failed to queue retraining task on drift detection: {e}")

        return report

    def _save_drift_report(self, report: Dict[str, Any]) -> None:
        """Stores the parsed DriftReport results inside PostgreSQL database table."""
        query = """
        INSERT INTO drift_reports (drift_detected, drift_scores, affected_features, timestamp)
        VALUES (:drift_detected, :drift_scores, :affected_features, :timestamp)
        """
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(query),
                    {
                        "drift_detected": report["drift_detected"],
                        "drift_scores": json.dumps(report["drift_scores"]),
                        "affected_features": json.dumps(report["affected_features"]),
                        "timestamp": datetime.fromisoformat(report["timestamp"])
                    }
                )
            logger.info("Successfully recorded new drift report to database.")
        except Exception as e:
            logger.error(f"Failed to save drift report to Postgres: {e}")
