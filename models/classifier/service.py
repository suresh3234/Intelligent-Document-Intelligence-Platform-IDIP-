"""Document Classifier service implementation for IDIP."""
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

from config import settings
from ingestion.models import IngestedDocument
from preprocessing.features import compute_all_document_features
from models.classifier.config import CLASSIFIER_CLASSES, DEFAULT_ENSEMBLE_WEIGHTS, DEFAULT_CALIBRATION_TEMPERATURE
from models.classifier.exceptions import ClassifierTrainingError, ClassifierInferenceError

try:
    import xgboost
except ImportError:
    xgboost = None

logger = logging.getLogger("idip.models.classifier.service")

class ClassificationResult(BaseModel):
    """Pydantic model representing document classification output."""
    predicted_class: str = Field(..., description="The predicted category label")
    confidence: float = Field(..., description="The blended probability score of the class")
    class_probabilities: Dict[str, float] = Field(..., description="The probability mapping per class")
    uncertain_classification: bool = Field(..., description="Flag indicating if max confidence is below 0.6")
    model_version: str = Field(..., description="Version identifier of the classifier model")

class TrainingResult(BaseModel):
    """Pydantic model representing training validation metrics."""
    status: str = Field(..., description="Overall training status")
    stage1_accuracy: float = Field(..., description="Stage 1 (XGBoost) validation accuracy")
    stage2_accuracy: float = Field(..., description="Stage 2 (BERT head) validation accuracy")
    model_version: str = Field(..., description="Model version tag")

class DocumentClassifier:
    """
    Document Classifier Service.
    Employs a 2-stage ensemble:
      - Stage 1: XGBoost classifier on document structural features.
      - Stage 2: Pre-trained BERT encoder (CLS token) with a linear classifier head.
    Combines outputs via weighted probability blending and calibrates with temperature scaling.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        ensemble_weights: Optional[Dict[str, float]] = None,
        model_version: str = "1.0.0"
    ):
        self.bert_model_name = bert_model_name
        self.model_version = model_version
        self.ensemble_weights = ensemble_weights or DEFAULT_ENSEMBLE_WEIGHTS
        
        # In-memory model storages (can be persisted to disk)
        self.xgb_model = None
        self.bert_head = None
        self.tokenizer = None
        self.bert_model = None
        self.label_encoder = None

        # Verify weights sum to 1
        total_w = sum(self.ensemble_weights.values())
        if not np.isclose(total_w, 1.0):
            # Normalize weights
            self.ensemble_weights = {k: v / total_w for k, v in self.ensemble_weights.items()}

    def load_bert(self) -> None:
        """Lazily loads the pre-trained BERT tokenizer and model components."""
        if self.tokenizer is not None and self.bert_model is not None:
            return

        try:
            logger.info(f"Loading BERT encoder model: {self.bert_model_name}...")
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
            self.bert_model.eval()  # Freeze weights for feature extraction
            logger.info("BERT encoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise ClassifierInferenceError(f"Could not load BERT model '{self.bert_model_name}': {e}") from e

    def _get_cls_embedding(self, text: str) -> np.ndarray:
        """Tokenizes text and extracts the CLS hidden state vector using the BERT encoder."""
        self.load_bert()
        if not self.tokenizer or not self.bert_model:
            raise ClassifierInferenceError("BERT models not initialized.")

        try:
            import torch
            # Truncate to 512 tokens
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # CLS token is at index 0
                cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            return cls_embedding
        except Exception as e:
            logger.error(f"Error extracting CLS embedding: {e}")
            raise ClassifierInferenceError(f"Embedding extraction failure: {e}") from e

    def _extract_tabular_features(self, doc: IngestedDocument) -> np.ndarray:
        """Translates computed document structural features into a normalized float array for XGBoost."""
        features = compute_all_document_features(doc)
        
        # Map string categoricals to continuous values
        lang_map = {"en": 0.0, "fr": 1.0, "de": 2.0, "es": 3.0, "hi": 4.0, "zh": 5.0}
        type_map = {"invoice": 0.0, "contract": 1.0, "report": 2.0, "email": 3.0, "other": 4.0}
        
        lang_val = lang_map.get(features.get("language", "en"), 6.0)
        type_val = type_map.get(features.get("doc_type_signal", "other"), 4.0)

        feature_vector = [
            float(features.get("page_count", 1)),
            float(features.get("avg_words_per_page", 0.0)),
            1.0 if features.get("has_tables", False) else 0.0,
            1.0 if features.get("has_images", False) else 0.0,
            float(lang_val),
            float(type_val),
            float(features.get("reading_level", 0.0)),
            float(features.get("entity_density", 0.0)),
            float(features.get("text_quality_score", 1.0))
        ]
        return np.array(feature_vector, dtype=np.float32)

    def train(self, docs: List[IngestedDocument], labels: List[str]) -> TrainingResult:
        """
        Trains both XGBoost (structural features) and BERT Linear Head (text embeddings) classifiers
        using the input documents and matching string labels.
        """
        if not docs or not labels or len(docs) != len(labels):
            raise ClassifierTrainingError("Inconsistent training features and labels count.")

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
            
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)

            # 1. Process and train Stage 1 (XGBoost / GBC fallback)
            logger.info("Extracting structural features for Stage 1...")
            X_tab = np.array([self._extract_tabular_features(d) for d in docs])
            
            # Adjust estimators dynamically for tiny train counts
            n_est = min(50, len(docs))
            if xgboost is not None:
                self.xgb_model = xgboost.XGBClassifier(
                    n_estimators=n_est,
                    max_depth=3,
                    learning_rate=0.1,
                    eval_metric="mlogloss"
                )
            else:
                logger.warning("xgboost is not installed. Falling back to sklearn.ensemble.GradientBoostingClassifier.")
                from sklearn.ensemble import GradientBoostingClassifier
                self.xgb_model = GradientBoostingClassifier(
                    n_estimators=n_est,
                    max_depth=3,
                    learning_rate=0.1
                )
                
            self.xgb_model.fit(X_tab, y)
            stage1_acc = float(self.xgb_model.score(X_tab, y))

            # 2. Process and train Stage 2 (BERT head)
            logger.info("Extracting BERT CLS embeddings...")
            X_bert = np.array([self._get_cls_embedding(d.raw_text) for d in docs])
            
            self.bert_head = LogisticRegression(max_iter=1000, C=1.0)
            self.bert_head.fit(X_bert, y)
            stage2_acc = float(self.bert_head.score(X_bert, y))

            logger.info(f"Training completed. XGBoost Acc: {stage1_acc:.4f}, BERT Acc: {stage2_acc:.4f}")
            return TrainingResult(
                status="success",
                stage1_accuracy=stage1_acc,
                stage2_accuracy=stage2_acc,
                model_version=self.model_version
            )
        except Exception as e:
            logger.error(f"Error during classifier training: {e}")
            raise ClassifierTrainingError(f"Classifier fitting failed: {e}") from e

    def predict(
        self,
        doc: IngestedDocument,
        temperature: float = DEFAULT_CALIBRATION_TEMPERATURE
    ) -> ClassificationResult:
        """
        Predicts document category class by blending XGBoost and BERT probability distributions,
        calibrates via temperature scaling, and flags low-confidence predictions.
        """
        # Ensure models are trained or bootstrapped
        if self.xgb_model is None or self.bert_head is None:
            # Fallback/bootstrap models with default weights to prevent crashes if predict called before train
            raise ClassifierInferenceError("Classifier models have not been trained yet. Run train() first.")

        try:
            # 1. Stage 1 probability distribution
            X_tab = self._extract_tabular_features(doc).reshape(1, -1)
            p_xgb = self.xgb_model.predict_proba(X_tab)[0]

            # 2. Stage 2 probability distribution
            X_bert = self._get_cls_embedding(doc.raw_text).reshape(1, -1)
            p_bert = self.bert_head.predict_proba(X_bert)[0]

            # Ensure both distributions align with target classes count
            num_classes = len(CLASSIFIER_CLASSES)
            if self.label_encoder is not None:
                p_xgb_full = np.zeros(num_classes)
                for idx, label in enumerate(self.label_encoder.classes_):
                    global_idx = CLASSIFIER_CLASSES.index(label.lower())
                    p_xgb_full[global_idx] = p_xgb[idx]
                p_xgb = p_xgb_full

                p_bert_full = np.zeros(num_classes)
                for idx, label in enumerate(self.label_encoder.classes_):
                    global_idx = CLASSIFIER_CLASSES.index(label.lower())
                    p_bert_full[global_idx] = p_bert[idx]
                p_bert = p_bert_full
            else:
                if len(p_xgb) < num_classes:
                    p_xgb_full = np.zeros(num_classes)
                    p_xgb_full[:len(p_xgb)] = p_xgb
                    p_xgb = p_xgb_full
                if len(p_bert) < num_classes:
                    p_bert_full = np.zeros(num_classes)
                    p_bert_full[:len(p_bert)] = p_bert
                    p_bert = p_bert_full

            # 3. Blending
            w_xgb = self.ensemble_weights.get("xgboost", 0.4)
            w_bert = self.ensemble_weights.get("bert", 0.6)
            blended_probs = (w_xgb * p_xgb) + (w_bert * p_bert)

            # 4. Temperature Calibration
            calibrated_probs = self._calibrate_probabilities(blended_probs, temperature)

            # 5. Extract results
            max_idx = int(np.argmax(calibrated_probs))
            predicted_label = CLASSIFIER_CLASSES[max_idx]
            confidence_score = float(calibrated_probs[max_idx])

            # Class probability mapping dictionary
            probabilities_dict = {
                CLASSIFIER_CLASSES[i]: float(calibrated_probs[i])
                for i in range(num_classes)
            }

            # Low confidence checking (threshold < 0.6)
            uncertain = confidence_score < 0.6

            return ClassificationResult(
                predicted_class=predicted_label,
                confidence=float(round(confidence_score, 4)),
                class_probabilities={k: float(round(v, 4)) for k, v in probabilities_dict.items()},
                uncertain_classification=uncertain,
                model_version=self.model_version
            )
        except Exception as e:
            logger.error(f"Error during classifier inference: {e}")
            raise ClassifierInferenceError(f"Classifier prediction failure: {e}") from e

    def _calibrate_probabilities(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """Applies softmax temperature scaling calibration over probabilities."""
        if temperature == 1.0:
            return probs

        # Softmax over pseudo-logits log(probs + epsilon)
        eps = 1e-9
        logits = np.log(probs + eps)
        scaled_logits = logits / max(1e-5, temperature)
        
        # Softmax formula
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / np.sum(exp_logits)
