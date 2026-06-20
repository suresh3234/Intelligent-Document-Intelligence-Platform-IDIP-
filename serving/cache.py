"""Semantic cache implementation for RAG query results in IDIP."""
import base64
import json
import time
import logging
import collections
import threading
from typing import Optional, Any
import numpy as np
from prometheus_client import Counter, Gauge
from config import settings
from rag.models import RAGResponse

logger = logging.getLogger("idip.serving.cache")

# Define Prometheus metrics
idip_cache_hits_total = Counter(
    "idip_cache_hits_total",
    "Total number of semantic cache hits"
)
idip_cache_misses_total = Counter(
    "idip_cache_misses_total",
    "Total number of semantic cache misses"
)
idip_cache_hit_rate = Gauge(
    "idip_cache_hit_rate",
    "Rolling 5-minute semantic cache hit rate (ratio of hits to total requests)"
)

class HitRateTracker:
    """Tracks cache hits/misses in a rolling window to compute hit rate."""
    
    def __init__(self, window_seconds: float = 300.0):
        self.window_seconds = window_seconds
        self.history = collections.deque()  # items: (timestamp, is_hit)
        self.lock = threading.Lock()

    def record(self, is_hit: bool):
        """Records a new event and updates the hit rate gauge."""
        now = time.time()
        with self.lock:
            self.history.append((now, is_hit))
            self._cleanup(now)
            
            hits = sum(1 for _, hit in self.history if hit)
            total = len(self.history)
            rate = hits / total if total > 0 else 0.0
            idip_cache_hit_rate.set(rate)

    def _cleanup(self, now: float):
        """Removes entries older than the window length."""
        cutoff = now - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()


class SemanticCache:
    """
    Semantic Cache Layer using Redis.
    Uses vector cosine similarity thresholding (>= 0.95) to cache and serve
    RAG answers based on query semantic proximity.
    """
    
    def __init__(self, redis_client: Any, cache_ttl: Optional[int] = None):
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl or getattr(settings, "CACHE_TTL", 3600)
        self.tracker = HitRateTracker()

    def _generate_key(self, query_embedding: np.ndarray) -> str:
        """Generates a cache key based on query_embedding[:32] bytes."""
        # query_embedding[:32] takes the first 32 elements
        sub_vec = query_embedding[:32].astype(np.float32)
        vec_bytes = sub_vec.tobytes()
        b64_str = base64.b64encode(vec_bytes).decode("utf-8")
        return f"cache:{b64_str}"

    def set(self, query: str, query_embedding: np.ndarray, response: RAGResponse) -> None:
        """Caches a RAG response associated with query and embedding."""
        if self.redis_client is None:
            return

        key = self._generate_key(query_embedding)
        
        # Serialize response to dictionary
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        else:
            response_dict = response

        payload = {
            "query": query,
            "embedding": query_embedding.tolist(),
            "response_json": json.dumps(response_dict),
            "hit_count": 0,
            "created_at": time.time()
        }

        try:
            self.redis_client.setex(
                key,
                self.cache_ttl,
                json.dumps(payload)
            )
            logger.info(f"Successfully cached query response in Redis under key: {key[:30]}...")
        except Exception as e:
            logger.warning(f"Failed to cache response in Redis: {e}")

    def get(self, query_embedding: np.ndarray) -> Optional[RAGResponse]:
        """Looks up semantically similar cached response from Redis."""
        if self.redis_client is None:
            idip_cache_misses_total.inc()
            self.tracker.record(is_hit=False)
            return None

        try:
            # Retrieve all cache keys
            keys = self.redis_client.keys("cache:*")
        except Exception as e:
            logger.warning(f"Failed to scan keys in Redis: {e}")
            idip_cache_misses_total.inc()
            self.tracker.record(is_hit=False)
            return None

        max_similarity = -1.0
        best_key = None
        best_payload = None

        for key in keys:
            try:
                val = self.redis_client.get(key)
                if not val:
                    continue
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                
                payload = json.loads(val)
                cached_emb = np.array(payload["embedding"], dtype=np.float32)
                
                # Calculate Cosine Similarity
                dot = np.dot(query_embedding, cached_emb)
                norm1 = np.linalg.norm(query_embedding)
                norm2 = np.linalg.norm(cached_emb)
                similarity = float(dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_key = key
                    best_payload = payload
            except Exception as e:
                logger.warning(f"Failed to process cached embedding for key {key}: {e}")

        if max_similarity >= 0.95 and best_key is not None and best_payload is not None:
            logger.info(f"Semantic Cache Hit! Similarity: {max_similarity:.4f}")
            # Update hit_count and write back
            try:
                best_payload["hit_count"] = best_payload.get("hit_count", 0) + 1
                ttl = self.redis_client.ttl(best_key)
                if ttl > 0:
                    self.redis_client.setex(best_key, ttl, json.dumps(best_payload))
                else:
                    self.redis_client.set(best_key, json.dumps(best_payload))
            except Exception as e:
                logger.warning(f"Failed to update hit count in cache: {e}")

            idip_cache_hits_total.inc()
            self.tracker.record(is_hit=True)
            
            # Reconstruct response object
            response_dict = json.loads(best_payload["response_json"])
            return RAGResponse(**response_dict)

        logger.info("Semantic Cache Miss.")
        idip_cache_misses_total.inc()
        self.tracker.record(is_hit=False)
        return None
