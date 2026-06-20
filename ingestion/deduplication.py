import hashlib
from typing import Optional, Any
from ingestion.exceptions import DuplicateDocumentError

class DeduplicationService:
    """Service to detect duplicate document submissions using Redis key lookups."""
    
    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.redis_set_name = "idip:dedup"

    def compute_dedup_key(self, source_uri: str, checksum: str) -> str:
        """Computes a unique SHA-256 hash representation of the document source + contents."""
        val = f"{source_uri}:{checksum}".encode("utf-8")
        return hashlib.sha256(val).hexdigest()

    async def check_and_register(self, source_uri: str, checksum: str, doc_id: str, ttl: int = 86400) -> None:
        """
        Validates if document has already been processed. 
        If exists, raises DuplicateDocumentError.
        If new, registers the key into Redis with the given TTL.
        """
        dedup_key = self.compute_dedup_key(source_uri, checksum)
        redis_key = f"{self.redis_set_name}:{dedup_key}"
        
        try:
            # Query Redis to see if we have seen this dedup key
            import inspect
            res = self.redis_client.get(redis_key)
            if inspect.isawaitable(res):
                existing_doc_id = await res
            else:
                existing_doc_id = res

            if existing_doc_id:
                # Decoded if needed
                if isinstance(existing_doc_id, bytes):
                    existing_doc_id = existing_doc_id.decode("utf-8")
                raise DuplicateDocumentError(
                    doc_id=existing_doc_id, 
                    message=f"Duplicate document detected (same URI and checksum) matching existing doc_id"
                )
            
            # If new, register key in Redis with TTL
            res_set = self.redis_client.set(redis_key, doc_id, ex=ttl)
            if inspect.isawaitable(res_set):
                await res_set
        except Exception as e:
            if isinstance(e, DuplicateDocumentError):
                raise
            # Fall open in case of Redis connection failures in production, log error
            pass
