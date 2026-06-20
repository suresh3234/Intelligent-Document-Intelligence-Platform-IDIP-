import json
import logging
from typing import List, Optional, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from preprocessing.models import TextChunk

logger = logging.getLogger("idip.preprocessing.embeddings")

class EmbeddingService:
    """
    Embedding service for generating dense vectors from text chunks.
    Uses BAAI/bge-large-en-v1.5 and provides Redis-based caching.
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        model_name: str = "BAAI/bge-large-en-v1.5",
        cache_ttl: int = 86400
    ):
        self.redis_client = redis_client
        self.model_name = model_name
        self.cache_ttl = cache_ttl
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily instantiates the SentenceTransformer model to optimize memory and startup time."""
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device '{device}'...")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def encode_batch(self, chunks: List[TextChunk]) -> List[np.ndarray]:
        """
        Encodes a list of TextChunks into L2-normalized dense embeddings.
        Checks Redis cache first using 'emb:{chunk_id}'. Only performs inference
        for cache misses, and caches newly computed embeddings.
        """
        if not chunks:
            return []

        results = [None] * len(chunks)
        miss_indices = []
        miss_texts = []

        # 1. Check Cache
        if self.redis_client is not None:
            for i, chunk in enumerate(chunks):
                cache_key = f"emb:{chunk.chunk_id}"
                try:
                    cached_val = self.redis_client.get(cache_key)
                    if cached_val:
                        # Decode and deserialize
                        if isinstance(cached_val, bytes):
                            cached_val = cached_val.decode("utf-8")
                        vector = np.array(json.loads(cached_val), dtype=np.float32)
                        results[i] = vector
                        logger.info(f"Embedding cache hit for chunk_id: {chunk.chunk_id}")
                    else:
                        miss_indices.append(i)
                        miss_texts.append(chunk.text)
                except Exception as e:
                    logger.warning(f"Failed to query Redis cache for key {cache_key}: {e}. Falling back to inference.")
                    miss_indices.append(i)
                    miss_texts.append(chunk.text)
        else:
            # If no redis client, all are misses
            miss_indices = list(range(len(chunks)))
            miss_texts = [chunk.text for chunk in chunks]

        # 2. Perform Inference for Misses
        if miss_texts:
            # Batch encode on device (cuda/cpu) with batch_size=32 and L2 normalization
            logger.info(f"Performing batch inference for {len(miss_texts)} embedding cache misses...")
            embeddings = self.model.encode(
                miss_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )

            # 3. Save to Cache & Populate Results
            for idx, emb_val in zip(miss_indices, embeddings, strict=True):
                # Ensure float32 representation
                emb_array = np.array(emb_val, dtype=np.float32)
                results[idx] = emb_array
                
                if self.redis_client is not None:
                    chunk = chunks[idx]
                    cache_key = f"emb:{chunk.chunk_id}"
                    try:
                        self.redis_client.set(
                            cache_key,
                            json.dumps(emb_array.tolist()),
                            ex=self.cache_ttl
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write embedding cache to Redis for key {cache_key}: {e}")

        # Ensure all elements were populated
        for i, res in enumerate(results):
            if res is None:
                raise RuntimeError(f"Embedding encoding failed to populate index {i} in results list.")

        return results

    def encode_single(self, text: str) -> np.ndarray:
        """Encodes a single text string and returns the L2-normalized embedding."""
        embeddings = self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(embeddings[0], dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a query string with the required BGE retrieval prefix.
        Returns the L2-normalized embedding.
        """
        prefixed_query = f"Represent this sentence: {query}"
        return self.encode_single(prefixed_query)
