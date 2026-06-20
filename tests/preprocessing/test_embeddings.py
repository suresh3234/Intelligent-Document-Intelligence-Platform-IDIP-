import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from preprocessing.models import TextChunk
from preprocessing.embeddings import EmbeddingService

@pytest.fixture
def mock_redis():
    """Returns a mock Redis client."""
    client = MagicMock()
    client.get = MagicMock(return_value=None)
    client.set = MagicMock(return_value=True)
    return client

@pytest.fixture
def dummy_chunks():
    """Generates 5 dummy TextChunk objects for testing."""
    return [
        TextChunk(
            chunk_id=f"c-{i}",
            doc_id="doc-123",
            chunk_index=i,
            text=f"This is chunk number {i} text for embedding test.",
            token_count=10,
            char_start=0,
            char_end=50,
            chunk_strategy="fixed"
        )
        for i in range(5)
    ]

@patch("preprocessing.embeddings.SentenceTransformer")
def test_embedding_service_basic(mock_transformer_cls, mock_redis, dummy_chunks):
    # Setup mock transformer
    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.ones((5, 1024), dtype=np.float32)
    mock_transformer_cls.return_value = mock_instance
    
    # Initialize EmbeddingService
    service = EmbeddingService(redis_client=mock_redis)
    
    # Encode batch
    embeddings = service.encode_batch(dummy_chunks)
    
    # Assertions
    assert len(embeddings) == 5
    assert embeddings[0].shape == (1024,)
    
    # Verify cached GET was checked for each chunk
    assert mock_redis.get.call_count == 5
    # Verify cached SET was set for each chunk
    assert mock_redis.set.call_count == 5

@patch("preprocessing.embeddings.SentenceTransformer")
def test_embedding_service_cache_hit(mock_transformer_cls, mock_redis, dummy_chunks):
    # Setup mock transformer - should NOT be called if all are cache hits
    mock_instance = MagicMock()
    mock_transformer_cls.return_value = mock_instance
    
    # Setup Redis to return a pre-cached embedding for first chunk, and miss on the rest
    cached_vector = np.full((1024,), 0.5, dtype=np.float32)
    
    # Simple get side_effect: hit for first, miss for others
    def mock_redis_get(key):
        if key == "emb:c-0":
            return json.dumps(cached_vector.tolist()).encode("utf-8")
        return None
    mock_redis.get.side_effect = mock_redis_get
    
    # Setup mock model encode to return vectors for the remaining 4 chunks
    mock_instance.encode.return_value = np.ones((4, 1024), dtype=np.float32)
    
    service = EmbeddingService(redis_client=mock_redis)
    embeddings = service.encode_batch(dummy_chunks)
    
    # Verify first embedding is the cached one
    assert np.allclose(embeddings[0], cached_vector)
    # Verify other embeddings are from the model
    assert np.allclose(embeddings[1], np.ones((1024,)))
    
    # Verify model encode was called with 4 texts
    mock_instance.encode.assert_called_once()
    args, kwargs = mock_instance.encode.call_args
    assert len(args[0]) == 4
    assert "c-0" not in dummy_chunks[1].text
    
    # Redis set should only have been called 4 times (for the misses)
    assert mock_redis.set.call_count == 4

@patch("preprocessing.embeddings.SentenceTransformer")
def test_encode_query_prefix(mock_transformer_cls):
    mock_instance = MagicMock()
    mock_instance.encode.return_value = [np.ones((1024,), dtype=np.float32)]
    mock_transformer_cls.return_value = mock_instance
    
    service = EmbeddingService()
    _ = service.encode_query("my test query")
    
    # Verify BGE prefix was prepended
    mock_instance.encode.assert_called_once()
    args, kwargs = mock_instance.encode.call_args
    assert args[0] == ["Represent this sentence: my test query"]

@patch("preprocessing.embeddings.SentenceTransformer")
def test_embeddings_integration_100_chunks(mock_transformer_cls):
    """
    Integration test:
    - Generate 100 chunks.
    - Verify output shape = (1024,) for each chunk.
    - Verify all output embeddings are L2-normalized (norm = 1.0).
    """
    mock_instance = MagicMock()
    
    # Mock encode method to return L2 normalized random vectors
    def mock_encode_impl(sentences, batch_size=32, **kwargs):
        num = len(sentences)
        vectors = np.random.randn(num, 1024).astype(np.float32)
        # Apply L2 normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms
        return normalized

    mock_instance.encode.side_effect = mock_encode_impl
    mock_transformer_cls.return_value = mock_instance
    
    service = EmbeddingService(redis_client=None)
    
    # 1. Generate 100 chunks
    chunks = [
        TextChunk(
            chunk_id=f"c-{i}",
            doc_id="doc-integration",
            chunk_index=i,
            text=f"Integration test chunk sentence {i}. Some descriptive text.",
            token_count=12,
            char_start=i * 50,
            char_end=i * 50 + 50,
            chunk_strategy="fixed"
        )
        for i in range(100)
    ]
    
    # 2. Batch encode chunks
    embeddings = service.encode_batch(chunks)
    
    # 3. Verify assertions
    assert len(embeddings) == 100
    
    for emb in embeddings:
        # Verify shape
        assert emb.shape == (1024,)
        # Verify type is float32
        assert emb.dtype == np.float32
        # Verify L2-norm is 1.0
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-5) == 1.0
