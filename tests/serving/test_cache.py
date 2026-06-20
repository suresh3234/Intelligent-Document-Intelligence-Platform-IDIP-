"""Unit tests for the IDIP SemanticCache implementation."""
import json
import base64
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from serving.cache import SemanticCache, idip_cache_hits_total, idip_cache_misses_total, idip_cache_hit_rate
from rag.models import RAGResponse, Citation

@pytest.fixture
def mock_redis():
    """Fixture providing a mocked Redis client."""
    client = MagicMock()
    # Mock return values
    client.keys.return_value = []
    client.get.return_value = None
    client.setex.return_value = True
    client.set.return_value = True
    client.ttl.return_value = 1800
    return client

@pytest.fixture
def dummy_response():
    """Fixture returning a sample RAGResponse."""
    return RAGResponse(
        answer="This is a test RAG answer response.",
        citations=[
            Citation(
                doc_id="doc-uuid-abc-12345",
                doc_id_short="doc-uuid",
                source_uri="s3://source/test.pdf",
                text_snippet="matching reference text block."
            )
        ],
        confidence=0.92,
        total_latency_ms=120.5,
        low_confidence=False
    )

def test_semantic_cache_set(mock_redis, dummy_response):
    """Verify that set correctly generates a base64 key and stores serialized payload."""
    cache = SemanticCache(redis_client=mock_redis, cache_ttl=3600)
    
    query = "What is the RAG context threshold?"
    query_emb = np.ones(1024, dtype=np.float32)
    # Ensure normalized
    query_emb /= np.linalg.norm(query_emb)
    
    cache.set(query, query_emb, dummy_response)
    
    # Verify Redis setex was called
    mock_redis.setex.assert_called_once()
    
    # Extract args
    args, kwargs = mock_redis.setex.call_args
    key, ttl, value_str = args
    
    assert key.startswith("cache:")
    assert ttl == 3600
    
    # Parse payload
    payload = json.loads(value_str)
    assert payload["query"] == query
    assert len(payload["embedding"]) == 1024
    assert payload["hit_count"] == 0
    assert "created_at" in payload
    
    # Parse response_json
    resp_dict = json.loads(payload["response_json"])
    assert resp_dict["answer"] == dummy_response.answer
    assert resp_dict["confidence"] == dummy_response.confidence

def test_semantic_cache_get_hit(mock_redis, dummy_response):
    """Verify that get retrieves cached response on exact/high similarity matches."""
    cache = SemanticCache(redis_client=mock_redis)
    
    query = "What is the RAG context threshold?"
    query_emb = np.zeros(1024, dtype=np.float32)
    query_emb[0] = 1.0  # Normalized vector
    
    # Mock cached entry
    cached_key = "cache:somebase64string"
    cached_payload = {
        "query": query,
        "embedding": query_emb.tolist(),
        "response_json": json.dumps(dummy_response.model_dump()),
        "hit_count": 0,
        "created_at": time.time()
    }
    
    mock_redis.keys.return_value = [cached_key]
    mock_redis.get.return_value = json.dumps(cached_payload)
    mock_redis.ttl.return_value = 2500
    
    # Get cache hit
    hit_res = cache.get(query_emb)
    
    assert hit_res is not None
    assert hit_res.answer == dummy_response.answer
    assert hit_res.confidence == dummy_response.confidence
    
    # Verify hit_count was updated and written back
    mock_redis.get.assert_called_with(cached_key)
    mock_redis.setex.assert_called_once()
    write_args = mock_redis.setex.call_args[0]
    assert write_args[0] == cached_key
    assert write_args[1] == 2500
    updated_payload = json.loads(write_args[2])
    assert updated_payload["hit_count"] == 1

def test_semantic_cache_get_miss(mock_redis, dummy_response):
    """Verify that get returns None on low similarity matches."""
    cache = SemanticCache(redis_client=mock_redis)
    
    query = "What is the RAG context threshold?"
    # Vector 1: [1, 0, 0, ...]
    query_emb = np.zeros(1024, dtype=np.float32)
    query_emb[0] = 1.0
    
    # Cached Vector 2: [0, 1, 0, ...] (Cosine Similarity = 0.0)
    cached_emb = np.zeros(1024, dtype=np.float32)
    cached_emb[1] = 1.0
    
    cached_key = "cache:anotherbase64string"
    cached_payload = {
        "query": query,
        "embedding": cached_emb.tolist(),
        "response_json": json.dumps(dummy_response.model_dump()),
        "hit_count": 0,
        "created_at": time.time()
    }
    
    mock_redis.keys.return_value = [cached_key]
    mock_redis.get.return_value = json.dumps(cached_payload)
    
    # Get cache miss
    res = cache.get(query_emb)
    
    assert res is None
    # No updates written back
    mock_redis.setex.assert_not_called()
    mock_redis.set.assert_not_called()

def test_semantic_cache_metrics_update(mock_redis, dummy_response):
    """Verify that hit/miss updates Prometheus counters and hit rate gauge."""
    # Reset metrics state
    idip_cache_hits_total._value.set(0)
    idip_cache_misses_total._value.set(0)
    idip_cache_hit_rate.set(0.0)
    
    cache = SemanticCache(redis_client=mock_redis)
    
    # 1. Miss scenario
    mock_redis.keys.return_value = []
    res = cache.get(np.ones(1024, dtype=np.float32))
    assert res is None
    assert idip_cache_misses_total._value.get() == 1.0
    assert idip_cache_hits_total._value.get() == 0.0
    assert idip_cache_hit_rate._value.get() == 0.0
    
    # 2. Hit scenario
    query_emb = np.zeros(1024, dtype=np.float32)
    query_emb[0] = 1.0
    cached_payload = {
        "query": "test query",
        "embedding": query_emb.tolist(),
        "response_json": json.dumps(dummy_response.model_dump()),
        "hit_count": 0,
        "created_at": time.time()
    }
    mock_redis.keys.return_value = ["cache:key1"]
    mock_redis.get.return_value = json.dumps(cached_payload)
    
    res = cache.get(query_emb)
    assert res is not None
    assert idip_cache_misses_total._value.get() == 1.0
    assert idip_cache_hits_total._value.get() == 1.0
    # Rolling rate over last 5-min window: 1 hit out of 2 requests = 50%
    assert idip_cache_hit_rate._value.get() == 0.5
