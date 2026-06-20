import pytest
import asyncio
import hashlib
import json
import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Callable
from ingestion.models import IngestedDocument
from ingestion.validation import ValidationPipeline
from ingestion.dlq import DLQProducer, DLQConsumer
from ingestion.storage import S3ParquetWriter
from ingestion.exceptions import ValidationError, DuplicateDocumentError, QualityGateError

def _create_clean_doc(text="This is a long and valid standard text string containing more than fifty characters to pass the check.", lang="en") -> IngestedDocument:
    raw_bytes = b"fake-pdf-file-bytes-longer-than-hundred-for-checking" * 3
    checksum = hashlib.sha256(raw_bytes).hexdigest()
    return IngestedDocument(
        doc_id="doc-uuid-abc-123",
        ingestion_ts=datetime.datetime.utcnow(),
        source_type="pdf",
        source_uri="s3://source/doc.pdf",
        raw_text=text,
        raw_bytes=raw_bytes,
        byte_size=len(raw_bytes),
        checksum=checksum,
        language=lang,
        mime_type="application/pdf"
    )

@pytest.mark.asyncio
async def test_validation_pipeline_pass(mock_redis_client):
    # Setup
    mock_redis_client.get = AsyncMock(return_value=None)
    pipeline = ValidationPipeline(redis_client=mock_redis_client)
    doc = _create_clean_doc()
    
    # Mock langdetect.detect_langs to return 99% probability
    mock_lang = MagicMock()
    mock_lang.lang = "en"
    mock_lang.prob = 0.99
    
    with patch("langdetect.detect_langs", return_value=[mock_lang]):
        await pipeline.validate(doc)
        
    assert doc.language == "en"
    assert doc.metadata.get("uncertain_language") is None
    assert doc.metadata.get("quality_score") is not None

@pytest.mark.asyncio
async def test_validation_pipeline_schema_fail_length(mock_redis_client):
    pipeline = ValidationPipeline(redis_client=mock_redis_client)
    doc = _create_clean_doc(text="Short text.")  # Less than 50 chars
    
    with pytest.raises(ValidationError) as exc_info:
        await pipeline.validate(doc)
    assert "Schema check failed: raw_text length" in str(exc_info.value)

@pytest.mark.asyncio
async def test_validation_pipeline_uncertain_language(mock_redis_client):
    pipeline = ValidationPipeline(redis_client=mock_redis_client)
    doc = _create_clean_doc()
    
    mock_lang = MagicMock()
    mock_lang.lang = "en"
    mock_lang.prob = 0.75  # Low confidence
    
    with patch("langdetect.detect_langs", return_value=[mock_lang]):
        await pipeline.validate(doc)
        
    assert doc.metadata.get("uncertain_language") is True

@pytest.mark.asyncio
async def test_validation_pipeline_language_not_allowed(mock_redis_client):
    pipeline = ValidationPipeline(redis_client=mock_redis_client)
    doc = _create_clean_doc()
    
    mock_lang = MagicMock()
    mock_lang.lang = "jp"  # Japanese, not allowed
    mock_lang.prob = 0.90
    
    with patch("langdetect.detect_langs", return_value=[mock_lang]):
        with pytest.raises(ValidationError) as exc_info:
            await pipeline.validate(doc)
        assert "not in allowed list" in str(exc_info.value)

@pytest.mark.asyncio
async def test_validation_pipeline_quality_gate_ratio(mock_redis_client):
    pipeline = ValidationPipeline(redis_client=mock_redis_client)
    # Text with low alpha ratio (numbers and symbols only)
    doc = _create_clean_doc(text="1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 !!!###")
    
    mock_lang = MagicMock()
    mock_lang.lang = "en"
    mock_lang.prob = 0.99
    
    with patch("langdetect.detect_langs", return_value=[mock_lang]):
        with pytest.raises(QualityGateError) as exc_info:
            await pipeline.validate(doc)
        assert "low text quality ratio" in str(exc_info.value)

@pytest.mark.asyncio
async def test_dlq_producer_publish():
    class DummyProducer:
        def send(self, topic, value):
            pass
    mock_producer = MagicMock(spec=DummyProducer)
    dlq = DLQProducer(kafka_producer=mock_producer, topic="failed.topic")
    
    doc = _create_clean_doc()
    await dlq.publish_failure(doc, "Failed quality check")
    
    mock_producer.send.assert_called_once()
    args = mock_producer.send.call_args[0]
    assert args[0] == "failed.topic"
    payload = json.loads(args[1].decode("utf-8"))
    assert payload["failure_reason"] == "Failed quality check"
    assert payload["document"]["doc_id"] == "doc-uuid-abc-123"

@pytest.mark.asyncio
async def test_dlq_consumer_retry():
    mock_counter = MagicMock()
    mock_retry_cb = AsyncMock()
    
    consumer = DLQConsumer(prometheus_counter=mock_counter, retry_callback=mock_retry_cb)
    
    payload = {
        "failure_reason": "Low quality",
        "failed_at": "2026-06-18T06:00:00",
        "document": {
            "doc_id": "test-id",
            "source_uri": "s3://doc",
            "source_type": "pdf",
            # Inject metadata test_mode to skip 300s sleep
            "metadata": {"test_mode": True}
        }
    }
    
    msg_bytes = json.dumps(payload).encode("utf-8")
    await consumer.handle_dlq_message(msg_bytes)
    
    # Verify counter and logging alerts occurred
    mock_counter.inc.assert_called_once()
    
    # Wait for the async retry task to execute (delay = 0.1s in test_mode)
    await asyncio.sleep(0.2)
    mock_retry_cb.assert_called_once_with(payload["document"])

@pytest.mark.asyncio
async def test_s3_parquet_writer(mock_s3_client):
    writer = S3ParquetWriter(s3_client=mock_s3_client, bucket_name="dest-bucket")
    doc = _create_clean_doc()
    
    s3_path = await writer.write_document(doc)
    
    # Verify s3 partitioned key
    ts = doc.ingestion_ts
    year = ts.strftime("%Y")
    month = ts.strftime("%m")
    day = ts.strftime("%d")
    expected_key = f"raw/{year}/{month}/{day}/pdf/doc-uuid-abc-123.parquet"
    
    assert s3_path == f"s3://dest-bucket/{expected_key}"
    mock_s3_client.put_object.assert_called_once()
    call_kwargs = mock_s3_client.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "dest-bucket"
    assert call_kwargs["Key"] == expected_key
