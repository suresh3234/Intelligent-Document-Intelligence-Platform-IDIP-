import pytest
from unittest.mock import MagicMock, patch
from ingestion.adapters.pdf import PDFAdapter
from ingestion.exceptions import AdapterError

@pytest.mark.asyncio
async def test_pdf_adapter_ingest_success():
    adapter = PDFAdapter()
    
    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.metadata = {"format": "PDF 1.5", "title": "Test Title"}
    
    mock_page = MagicMock()
    # Mock text block extraction
    mock_page.get_text.return_value = {
        "blocks": [
            {
                "type": 0,
                "bbox": [50.0, 50.0, 200.0, 100.0],
                "lines": [
                    {
                        "spans": [
                            {"size": 16.0, "flags": 2, "text": "Main Heading", "bbox": [50.0, 50.0, 150.0, 70.0]}
                        ]
                    }
                ]
            },
            {
                "type": 0,
                "bbox": [50.0, 110.0, 300.0, 200.0],
                "lines": [
                    {
                        "spans": [
                            {"size": 10.0, "flags": 0, "text": "Standard body paragraph text content.", "bbox": [50.0, 110.0, 300.0, 130.0]}
                        ]
                    }
                ]
            }
        ]
    }
    
    # Mock table extraction
    mock_table = MagicMock()
    mock_table.bbox = [50.0, 210.0, 400.0, 350.0]
    mock_table.extract.return_value = [
        ["Header 1", "Header 2"],
        ["Val 1", "Val 2"]
    ]
    mock_page.find_tables.return_value = [mock_table]
    
    mock_doc.__getitem__.return_value = mock_page
    
    # Patch fitz.open
    with patch("fitz.open", return_value=mock_doc) as mock_open:
        doc = await adapter.ingest("mock.pdf", raw_bytes=b"%PDF-1.5 test raw data")
        
        mock_open.assert_called_once()
        assert doc.source_type == "pdf"
        assert doc.source_uri == "mock.pdf"
        assert doc.page_count == 1
        assert "Main Heading" in doc.raw_text
        assert "Standard body paragraph" in doc.raw_text
        assert "| Header 1 | Header 2 |" in doc.raw_text  # Markdown table string
        
        # Verify metadata extraction
        assert len(doc.metadata["headings"]) == 1
        assert doc.metadata["headings"][0]["text"] == "Main Heading"
        assert len(doc.metadata["tables"]) == 1
        assert doc.metadata["tables"][0]["headers"] == ["Header 1", "Header 2"]
        assert doc.metadata["pdf_metadata"]["title"] == "Test Title"

@pytest.mark.asyncio
async def test_pdf_adapter_invalid_file():
    adapter = PDFAdapter()
    with pytest.raises(AdapterError):
        # Ingestion of non-existent file path
        await adapter.ingest("nonexistent.pdf")
