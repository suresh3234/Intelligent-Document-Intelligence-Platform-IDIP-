import pytest
import os
import json
from unittest.mock import MagicMock, AsyncMock, patch
from ingestion.orchestrator import IngestionOrchestrator
from ingestion.deduplication import DeduplicationService
from ingestion.models import IngestedDocument
from ingestion.exceptions import FileSizeExceededError, DuplicateDocumentError

@pytest.mark.asyncio
async def test_orchestrator_routing():
    # Instantiate orchestrator with mocked adapters
    mock_pdf_adapter = MagicMock()
    mock_pdf_adapter.ingest = AsyncMock(return_value=MagicMock(spec=IngestedDocument, doc_id="pdf-123", byte_size=100))
    
    mock_image_adapter = MagicMock()
    mock_image_adapter.ingest = AsyncMock(return_value=MagicMock(spec=IngestedDocument, doc_id="img-456", byte_size=200))
    
    orchestrator = IngestionOrchestrator(adapters={
        "pdf": mock_pdf_adapter,
        "image": mock_image_adapter
    })
    
    # Test routing to PDF
    pdf_doc = await orchestrator.route_and_ingest("pdf", "sample.pdf")
    mock_pdf_adapter.ingest.assert_called_once_with("sample.pdf")
    assert pdf_doc.doc_id == "pdf-123"
    
    # Test routing to Image
    image_doc = await orchestrator.route_and_ingest("image", "sample.jpg")
    mock_image_adapter.ingest.assert_called_once_with("sample.jpg")
    assert image_doc.doc_id == "img-456"

@pytest.mark.asyncio
async def test_orchestrator_file_size_limit():
    orchestrator = IngestionOrchestrator(max_file_size=100) # 100 bytes limit
    
    # Mock os.path.exists and os.path.getsize
    with patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=500): # Exceeds limit
         
         with pytest.raises(FileSizeExceededError):
             await orchestrator.route_and_ingest("pdf", "large.pdf")

@pytest.mark.asyncio
async def test_deduplication_service(mock_redis_client):
    service = DeduplicationService(redis_client=mock_redis_client)
    
    source_uri = "https://example.com/doc.pdf"
    checksum = "abcde12345checksum"
    doc_id = "uuid-777"
    
    # Scen 1: Unique document
    # redis GET returns None
    mock_redis_client.get.return_value = None
    await service.check_and_register(source_uri, checksum, doc_id)
    
    # Verify set in Redis with TTL
    expected_key = f"idip:dedup:{service.compute_dedup_key(source_uri, checksum)}"
    mock_redis_client.get.assert_called_with(expected_key)
    mock_redis_client.set.assert_called_with(expected_key, doc_id, ex=86400)
    
    # Scen 2: Duplicate document
    # redis GET returns existing doc_id
    mock_redis_client.get.return_value = b"uuid-777"
    
    with pytest.raises(DuplicateDocumentError) as exc_info:
        await service.check_and_register(source_uri, checksum, "uuid-new")
    
    assert exc_info.value.doc_id == "uuid-777"
