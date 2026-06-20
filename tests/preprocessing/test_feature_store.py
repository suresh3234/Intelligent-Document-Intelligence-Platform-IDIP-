import datetime
import pytest
from ingestion.models import IngestedDocument
from preprocessing.features import (
    calculate_reading_level,
    calculate_entity_density,
    detect_doc_type_signal,
    compute_all_document_features
)
from preprocessing.feature_store import FeatureStore

@pytest.fixture
def sample_doc():
    """Returns a sample document for feature extraction testing."""
    text = (
        "--- PAGE 1 ---\n"
        "This is an annual report containing findings on the new project.\n"
        "We are analyzing the contract details and agreement terms. "
        "The project has been successful. John Doe from Paris completed the task.\n"
        "--- PAGE 2 ---\n"
        "Section with tabular data:\n"
        "Col 1\tCol 2\tCol 3\n"
        "Val 1\tVal 2\tVal 3\n"
        "End of the report document."
    )
    return IngestedDocument(
        doc_id="doc-feat-777",
        ingestion_ts=datetime.datetime.utcnow(),
        source_type="pdf",
        source_uri="s3://reports/annual.pdf",
        raw_text=text,
        byte_size=len(text),
        checksum="checksum-111",
        mime_type="application/pdf",
        language="en",
        page_count=2
    )

def test_feature_heuristics(sample_doc):
    # Test reading level (Flesch-Kincaid)
    grade = calculate_reading_level(sample_doc.raw_text)
    assert grade > 0.0
    
    # Test entity density
    # "John Doe from Paris completed the task." - proper nouns should be detected
    density = calculate_entity_density(sample_doc.raw_text)
    assert density > 0.0
    assert density < 0.5
    
    # Test document type detection
    # Text contains "annual report" and "report" -> should map to report
    doc_type = detect_doc_type_signal(sample_doc.raw_text)
    assert doc_type == "report"

    # Verify other type matches
    email_text = "Subject: Action required\nTo: team@idip.io\nFrom: leader@idip.io\nMessage body here."
    assert detect_doc_type_signal(email_text) == "email"

    invoice_text = "Invoice Number: INV-001\nTotal Amount Due: $150.00\nPlease pay by 2026-06-30."
    assert detect_doc_type_signal(invoice_text) == "invoice"

    contract_text = "This Confidentiality Agreement is entered into by the parties hereby undersigned."
    assert detect_doc_type_signal(contract_text) == "contract"

def test_compute_all_features(sample_doc):
    features = compute_all_document_features(sample_doc)
    
    assert features["page_count"] == 2
    assert features["avg_words_per_page"] > 10.0
    assert features["has_tables"] is True
    assert features["has_images"] is False
    assert features["language"] == "en"
    assert features["doc_type_signal"] == "report"
    assert features["reading_level"] > 0.0
    assert features["entity_density"] > 0.0
    assert features["text_quality_score"] > 0.6

def test_feature_store_sqlite():
    # Setup FeatureStore using in-memory SQLite database
    store = FeatureStore(db_url="sqlite:///:memory:")
    
    doc_id = "doc-test-999"
    features = {
        "page_count": 5,
        "avg_words_per_page": 250.5,
        "has_tables": True,
        "has_images": False,
        "language": "fr",
        "doc_type_signal": "invoice",
        "reading_level": 8.5,
        "entity_density": 0.0245,
        "text_quality_score": 0.8876
    }
    
    # Test set features
    store.set(doc_id, features)
    
    # Test get features
    retrieved = store.get(doc_id)
    
    assert retrieved == features
    # Assert exact types
    assert isinstance(retrieved["page_count"], int)
    assert isinstance(retrieved["avg_words_per_page"], float)
    assert isinstance(retrieved["has_tables"], bool)
    assert isinstance(retrieved["has_images"], bool)
    assert isinstance(retrieved["language"], str)
    assert isinstance(retrieved["doc_type_signal"], str)
    assert isinstance(retrieved["reading_level"], float)
    assert isinstance(retrieved["entity_density"], float)
    assert isinstance(retrieved["text_quality_score"], float)

    # Test update upsert
    updated_features = dict(features)
    updated_features["page_count"] = 6
    updated_features["avg_words_per_page"] = 280.0
    
    store.set(doc_id, updated_features)
    retrieved_updated = store.get(doc_id)
    
    assert retrieved_updated["page_count"] == 6
    assert retrieved_updated["avg_words_per_page"] == 280.0
    
    # Test non-existent doc
    assert store.get("non-existent-id") == {}
