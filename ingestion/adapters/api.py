import uuid
import datetime
import time
import random
import hmac
import hashlib
import asyncio
from typing import Optional, Dict, Any
import httpx

from ingestion.base import BaseSourceAdapter
from ingestion.models import IngestedDocument
from ingestion.exceptions import AdapterError, SignatureVerificationError, RateLimitExceededError

class RESTAPIAdapter(BaseSourceAdapter):
    """Adapter for ingesting documents from external REST APIs using async HTTPX."""
    
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis_client = redis_client

    async def ingest(self, source_uri: str, **kwargs) -> IngestedDocument:
        """
        Fetches document bytes from the given REST API endpoint.
        Implements exponential backoff with jitter and respects rate-limit headers.
        """
        headers = kwargs.get("headers", {})
        idempotency_key = kwargs.get("idempotency_key")
        
        # Check idempotency key if Redis is available
        if idempotency_key and self.redis_client:
            redis_key = f"idip:idempotency:{idempotency_key}"
            is_duplicate = await self._check_and_set_idempotency(redis_key)
            if is_duplicate:
                raise AdapterError(f"Request with idempotency key '{idempotency_key}' is already being processed or completed.")

        # Fetch payload via HTTPX
        raw_bytes = await self._fetch_with_retry(source_uri, headers)
        
        byte_size = len(raw_bytes)
        checksum = self.compute_checksum(raw_bytes)
        
        # Infer MIME type from Content-Type header or source_uri
        mime_type = kwargs.get("mime_type", "application/octet-stream")
        
        # Attempt to decode text content
        raw_text = self.safe_decode(raw_bytes)
        
        return IngestedDocument(
            doc_id=str(uuid.uuid4()),
            ingestion_ts=datetime.datetime.utcnow(),
            source_type="api",
            source_uri=source_uri,
            raw_text=raw_text,
            raw_bytes=raw_bytes,
            byte_size=byte_size,
            checksum=checksum,
            language=kwargs.get("language", "en"),
            mime_type=mime_type,
            page_count=None,
            metadata={
                "headers": headers,
                "idempotency_key": idempotency_key
            }
        )

    def verify_webhook_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verifies HMAC-SHA256 signature for inbound webhooks."""
        if not signature or not secret:
            raise SignatureVerificationError("Missing webhook signature or shared secret.")
        
        try:
            expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
            # Constant time comparison to prevent timing attacks
            if not hmac.compare_digest(expected, signature):
                raise SignatureVerificationError("Signature verification failed. Invalid payload signature.")
            return True
        except Exception as e:
            if not isinstance(e, SignatureVerificationError):
                raise SignatureVerificationError(f"Error validating webhook signature: {str(e)}") from e
            raise

    async def _fetch_with_retry(self, url: str, headers: Dict[str, str]) -> bytes:
        """Fetch endpoint with exponential backoff + jitter and rate-limiting parsing."""
        max_retries = 5
        base_delay = 1.0  # seconds
        max_delay = 16.0
        
        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(url, headers=headers, timeout=10.0)
                    
                    # Rate limiting parsing
                    rate_remaining = response.headers.get("X-RateLimit-Remaining")
                    rate_reset = response.headers.get("X-RateLimit-Reset")
                    
                    if response.status_code == 429:
                        # Too Many Requests
                        wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        if rate_reset:
                            try:
                                # Try parsing resetting timestamp (seconds epoch or delta seconds)
                                reset_epoch = float(rate_reset)
                                now = time.time()
                                if reset_epoch > now:
                                    wait_time = min(reset_epoch - now, max_delay)
                            except ValueError:
                                pass
                        
                        if attempt == max_retries - 1:
                            raise RateLimitExceededError("Rate limit exceeded and max retries reached.")
                            
                        await asyncio.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    return response.content
                    
                except (httpx.HTTPError, asyncio.TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise AdapterError(f"Failed to fetch document after {max_retries} attempts: {str(e)}") from e
                    
                    delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    
        raise AdapterError("Unknown failure during HTTP fetch.")

    async def _check_and_set_idempotency(self, redis_key: str) -> bool:
        """Interacts with Redis client to verify idempotency keys."""
        try:
            # Check if key exists
            exists = await self.redis_client.get(redis_key)
            if exists:
                return True
            # Set key with TTL (86400 seconds = 24 hours)
            await self.redis_client.set(redis_key, "processing", ex=86400)
            return False
        except Exception as e:
            # Fail-open/log error in redis connectivity, don't block ingestion
            return False
