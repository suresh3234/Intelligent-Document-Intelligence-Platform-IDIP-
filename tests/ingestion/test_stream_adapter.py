import pytest
import json
from unittest.mock import MagicMock, patch
from ingestion.adapters.stream import KafkaStreamAdapter
from ingestion.exceptions import AdapterError

@pytest.mark.asyncio
async def test_stream_adapter_ingest_single():
    # Test single-ingest fallback behavior
    adapter = KafkaStreamAdapter(kafka_config={}, schema_registry_client=None)
    doc = await adapter.ingest("kafka://topic/0/123", raw_bytes=b"raw stream data payload")
    
    assert doc.source_type == "stream"
    assert doc.raw_text == "raw stream data payload"
    assert doc.byte_size == len(b"raw stream data payload")

@pytest.mark.asyncio
async def test_stream_adapter_consume_micro_batch(mock_s3_client, mock_kafka_message):
    kafka_config = {"bootstrap.servers": "localhost:9092"}
    
    # Instantiate adapter with S3 mock client
    adapter = KafkaStreamAdapter(
        kafka_config=kafka_config,
        schema_registry_client=None,
        s3_client=mock_s3_client,
        bucket_name="test-bucket"
    )
    
    # Construct mock messages
    msg1 = mock_kafka_message(b'{"doc_id": "1", "text": "value 1"}', topic="raw.docs", partition=0, offset=100)
    msg2 = mock_kafka_message(b'{"doc_id": "2", "text": "value 2"}', topic="raw.docs", partition=0, offset=101)
    
    # Mock confluent_kafka Consumer
    mock_consumer_instance = MagicMock()
    mock_consumer_instance.poll.side_effect = [msg1, msg2, None]
    
    with patch("ingestion.adapters.stream.Consumer", return_value=mock_consumer_instance):
        docs = await adapter.consume_micro_batch(max_messages=10)
        
        assert len(docs) == 2
        assert docs[0].doc_id is not None
        assert docs[0].metadata["topic"] == "raw.docs"
        assert docs[0].metadata["offset"] == 100
        
        # Verify micro-batch was pushed to S3
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["ContentType"] == "application/json"
        
        # Verify JSON batch payload includes parsed contents
        batch_json = json.loads(call_kwargs["Body"].decode("utf-8"))
        assert len(batch_json) == 2
        assert batch_json[0]["payload"]["doc_id"] == "1"
        assert batch_json[1]["payload"]["doc_id"] == "2"
        
        # S3 path key partition validation
        assert call_kwargs["Key"].startswith("raw/")
