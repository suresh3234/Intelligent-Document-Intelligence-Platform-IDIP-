"""Unit tests for the Model Router."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from ingestion.models import IngestedDocument
from models.router import ModelRouter, ModelExecutionTrace
from models.ner.service import EntityResult
from models.classifier.service import ClassificationResult
from models.vision.service import VisionResult, Region

def create_doc_stub(source_type: str) -> IngestedDocument:
    """Helper to instantiate mock IngestedDocument."""
    return IngestedDocument(
        doc_id="test-doc-123",
        ingestion_ts=datetime.utcnow(),
        source_type=source_type,
        source_uri="s3://test/doc",
        raw_text="Test document text for billing total: $100.00.",
        raw_bytes=None,
        byte_size=100,
        checksum="dummy-hash",
        language="en",
        mime_type="application/octet-stream",
        page_count=1,
        metadata={}
    )

@pytest.mark.asyncio
async def test_router_image_source():
    """Verify that image sources execute Vision analyzer first, then NER sequentially."""
    # Mock services
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = [
        EntityResult(text="$100.00", label="MONEY", start=38, end=45, confidence=1.0)
    ]
    
    mock_vision = MagicMock()
    mock_vision.analyze_document.return_value = VisionResult(
        regions=[Region(bbox=[10, 10, 100, 30], label="text", confidence=0.95, text="billing total: $100.00.")],
        extracted_kv={"total": "$100.00"},
        doc_structure="<document></document>"
    )
    
    mock_classifier = MagicMock()
    mock_llm = MagicMock()

    router = ModelRouter(
        ner_service=mock_ner,
        classifier_service=mock_classifier,
        vision_analyzer=mock_vision,
        llm_service=mock_llm
    )
    
    doc = create_doc_stub(source_type="image")
    trace = await router.route_document(doc)
    
    assert isinstance(trace, ModelExecutionTrace)
    assert trace.doc_id == "test-doc-123"
    assert "vision" in trace.models_run
    assert "ner" in trace.models_run
    assert "classifier" not in trace.models_run
    
    # Verify sequential parameters passed correctly
    mock_vision.analyze_document.assert_called_once()
    mock_ner.extract_entities.assert_called_once_with("billing total: $100.00.", "test-doc-123")
    assert trace.latency_ms > 0.0

@pytest.mark.asyncio
async def test_router_pdf_source_parallel():
    """Verify that pdf sources run Classifier and NER in parallel using asyncio.gather."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = []
    
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = ClassificationResult(
        predicted_class="invoice",
        confidence=0.88,
        class_probabilities={"invoice": 0.88, "other": 0.12},
        uncertain_classification=False,
        model_version="1.0.0"
    )
    
    mock_vision = MagicMock()
    mock_llm = MagicMock()

    router = ModelRouter(
        ner_service=mock_ner,
        classifier_service=mock_classifier,
        vision_analyzer=mock_vision,
        llm_service=mock_llm
    )
    
    doc = create_doc_stub(source_type="pdf")
    trace = await router.route_document(doc)
    
    assert "classifier" in trace.models_run
    assert "ner" in trace.models_run
    assert "vision" not in trace.models_run
    
    mock_classifier.predict.assert_called_once_with(doc)
    mock_ner.extract_entities.assert_called_once_with(doc.raw_text, "test-doc-123")

@pytest.mark.asyncio
async def test_router_llm_generation_task():
    """Verify that LLM tasks are executed when run_llm is enabled."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = []
    
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = ClassificationResult(
        predicted_class="other",
        confidence=0.9,
        class_probabilities={"other": 0.9},
        uncertain_classification=False,
        model_version="1.0.0"
    )
    
    mock_vision = MagicMock()
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Generative answer text"

    router = ModelRouter(
        ner_service=mock_ner,
        classifier_service=mock_classifier,
        vision_analyzer=mock_vision,
        llm_service=mock_llm
    )
    
    doc = create_doc_stub(source_type="api")
    trace = await router.route_document(doc, run_llm=True, llm_prompt="Summarize invoice")
    
    assert "llm" in trace.models_run
    mock_llm.generate.assert_called_once_with("Summarize invoice")
    assert trace.execution_details["llm"] == "Generative answer text"
