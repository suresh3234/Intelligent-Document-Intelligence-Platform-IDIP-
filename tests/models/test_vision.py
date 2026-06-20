"""Unit tests for the Multimodal Vision Analyzer service."""
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
from models.vision.service import VisionDocumentAnalyzer, Region, VisionResult

@patch("models.vision.service.pytesseract")
def test_vision_ocr_preprocessor_fallback(mock_tesseract):
    """Verify fallback to Tesseract coordinate parser when OCR inputs are omitted."""
    # Mock image_to_data dict
    mock_tesseract.image_to_data.return_value = {
        "text": ["Invoice", "No:", "INV-100", "", "Total:", "$500"],
        "left": [10, 80, 150, 0, 10, 100],
        "top": [20, 20, 20, 0, 80, 80],
        "width": [50, 20, 80, 0, 60, 50],
        "height": [15, 15, 15, 0, 15, 15]
    }
    
    analyzer = VisionDocumentAnalyzer()
    img = Image.new("RGB", (1000, 1000), color="white")
    
    words, boxes = analyzer._run_ocr_preprocessor(img)
    
    assert len(words) == 5
    assert words[0] == "Invoice"
    assert words[2] == "INV-100"
    
    # Assert coordinates normalized to [0, 1000]
    # left=10, width=50 -> x0=10, x1=60 relative to 1000
    assert boxes[0] == [10, 20, 60, 35]

def test_vision_key_value_parsing():
    """Verify extraction of key-value matches from token layouts."""
    analyzer = VisionDocumentAnalyzer()
    words = ["Date:", "2026-06-18", "Customer:", "John", "Doe", "Total:", "$100.00"]
    boxes = [[0, 0, 0, 0]] * len(words)
    
    kvs = analyzer._parse_key_values(words, boxes)
    assert kvs["date"] == "2026-06-18"
    assert kvs["customer"] == "John Doe"
    assert kvs["total"] == "$100.00"

def test_vision_layout_segmentation():
    """Verify segmentation of table and text region scopes."""
    analyzer = VisionDocumentAnalyzer()
    words = ["Standard", "text", "description.", "Price|Qty|Total", "10.00|1|10.00"]
    boxes = [
        [10, 10, 100, 30],
        [110, 10, 150, 30],
        [160, 10, 250, 30],
        [10, 50, 300, 80],
        [10, 90, 300, 120]
    ]
    
    regions = analyzer._segment_layout_regions(words, boxes, hidden_states=None)
    
    assert len(regions) == 2
    # Verify table label extraction
    table_reg = next(r for r in regions if r.label == "table")
    assert "Price|Qty|Total" in table_reg.text
    
    text_reg = next(r for r in regions if r.label == "text")
    assert "Standard text" in text_reg.text

@patch("models.vision.service.LayoutLMv3Processor")
@patch("models.vision.service.LayoutLMv3Model")
def test_vision_analyzer_flow(mock_model_cls, mock_processor_cls):
    """Verify end-to-end analyze_document execution mapping."""
    mock_processor = MagicMock()
    mock_processor.return_value = {"input_ids": None}  # mock tokenizer outputs
    mock_processor_cls.from_pretrained.return_value = mock_processor
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    # Mock hidden states of shape [1, seq_len, 768]
    import numpy as np
    import torch
    mock_output.last_hidden_state = torch.tensor(np.random.randn(1, 10, 768))
    mock_model.return_value = mock_output
    mock_model_cls.from_pretrained.return_value = mock_model
    
    analyzer = VisionDocumentAnalyzer()
    img = Image.new("RGB", (500, 500), color="white")
    
    # Run analysis with pre-extracted OCR parameters
    res = analyzer.analyze_document(img, ocr_text="Total: $150.00", ocr_boxes=[[10, 10, 100, 30], [110, 10, 200, 30]])
    
    assert isinstance(res, VisionResult)
    assert len(res.regions) > 0
    assert res.extracted_kv["total"] == "$150.00"
    assert "<document>" in res.doc_structure
