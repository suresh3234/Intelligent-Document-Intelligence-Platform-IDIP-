import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_redis_client():
    """Returns a mock async Redis client."""
    client = MagicMock()
    # Mock GET and SET methods as coroutines
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    return client

@pytest.fixture
def mock_s3_client():
    """Returns a mock S3 client."""
    client = MagicMock()
    client.put_object = MagicMock(return_value={"ResponseMetadata": {"HTTPStatusCode": 200}})
    return client

@pytest.fixture
def mock_schema_registry():
    """Returns a mock Schema Registry client."""
    client = MagicMock()
    client.decode = MagicMock(return_value={"id": 42, "content": "mocked schema message"})
    return client

@pytest.fixture
def mock_kafka_message():
    """Helper to construct mock Kafka messages."""
    def _create_message(value: bytes, topic: str = "raw.documents", partition: int = 0, offset: int = 100, error_code=None):
        msg = MagicMock()
        msg.value.return_value = value
        msg.topic.return_value = topic
        msg.partition.return_value = partition
        msg.offset.return_value = offset
        if error_code:
            err = MagicMock()
            err.code.return_value = error_code
            msg.error.return_value = err
        else:
            msg.error.return_value = None
        return msg
    return _create_message
