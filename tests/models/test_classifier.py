"""Unit tests for the Document Classifier service."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from ingestion.models import IngestedDocument
from models.classifier.service import DocumentClassifier, ClassificationResult, TrainingResult

def create_dummy_doc(text: str, source_type: str = "pdf") -> IngestedDocument:
    """Helper to instantiate mock IngestedDocument."""
    return IngestedDocument(
        doc_id="dummy-id",
        ingestion_ts=datetime.utcnow(),
        source_type=source_type,
        source_uri="s3://bucket/test.pdf",
        raw_text=text,
        raw_bytes=None,
        byte_size=len(text),
        checksum="dummy-hash",
        language="en",
        mime_type="application/pdf",
        page_count=2,
        metadata={}
    )

def test_classifier_tabular_features_mapping():
    """Verify conversion of document statistics to tabular float array."""
    service = DocumentClassifier()
    doc = create_dummy_doc("This is a simple report document text. It has two pages.", source_type="pdf")
    
    vec = service._extract_tabular_features(doc)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 9
    assert vec[0] == 2.0  # page_count
    assert vec[4] == 0.0  # language map code ("en")
    assert vec[5] == 2.0  # doc_type_signal ("report")

def test_probability_temperature_scaling():
    """Verify that probability distribution is correctly scaled using calibration temperature."""
    service = DocumentClassifier()
    probs = np.array([0.5, 0.3, 0.1, 0.1])
    
    # T = 1.0 does nothing
    p_cal_1 = service._calibrate_probabilities(probs, temperature=1.0)
    np.testing.assert_array_almost_equal(p_cal_1, probs)
    
    # T -> high makes it uniform
    p_cal_high = service._calibrate_probabilities(probs, temperature=100.0)
    assert p_cal_high[0] < 0.3  # Shrunk from 0.5
    assert p_cal_high[2] > 0.2  # Rose from 0.1
    
    # T -> low makes it sharp (one-hot)
    p_cal_low = service._calibrate_probabilities(probs, temperature=0.01)
    assert p_cal_low[0] > 0.99
    assert p_cal_low[1] < 0.01

@patch("models.classifier.service.AutoTokenizer")
@patch("models.classifier.service.AutoModel")
def test_classifier_train_and_predict(mock_bert_model_cls, mock_tokenizer_cls):
    """Verify 2-stage ensemble train and prediction execution pipelines."""
    # Mock Tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    
    # Mock BERT model returning hidden states of shape [1, 512, 768]
    mock_bert = MagicMock()
    mock_output = MagicMock()
    mock_output.last_hidden_state = np.random.randn(1, 512, 768)
    # Convert mock output return value to support pytorch structures if needed
    import torch
    mock_output.last_hidden_state = torch.tensor(mock_output.last_hidden_state)
    mock_bert.return_value = mock_output
    mock_bert_model_cls.from_pretrained.return_value = mock_bert
    
    service = DocumentClassifier()
    
    # Create training docs
    docs = [
        create_dummy_doc("Subject: Project plan details. To: team", source_type="api"),
        create_dummy_doc("Invoice total due: $150.00. Billing details.", source_type="pdf")
    ]
    labels = ["email", "invoice"]
    
    # Fit models
    train_res = service.train(docs, labels)
    
    assert train_res.status == "success"
    assert service.xgb_model is not None
    assert service.bert_head is not None
    
    # Run prediction
    pred_doc = create_dummy_doc("Subject: Important schedule update. Cc: engineer", source_type="api")
    res = service.predict(pred_doc)
    
    assert res.predicted_class in ["invoice", "contract", "report", "email", "form", "receipt", "legal", "other"]
    assert res.confidence >= 0.0
    assert res.confidence <= 1.0
    assert len(res.class_probabilities) == 8
    assert isinstance(res.uncertain_classification, bool)
