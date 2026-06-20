import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock
from ingestion.adapters.api import RESTAPIAdapter
from ingestion.exceptions import AdapterError, SignatureVerificationError, RateLimitExceededError

@pytest.mark.asyncio
@respx.mock
async def test_api_adapter_fetch_success(mock_redis_client):
    adapter = RESTAPIAdapter(redis_client=mock_redis_client)
    url = "https://api.document-source.com/v1/docs/1"
    
    # Mocking standard HTTP response
    respx.get(url).respond(
        status_code=200, 
        content=b"Sample downloaded document data content",
        headers={"Content-Type": "text/plain"}
    )
    
    doc = await adapter.ingest(
        source_uri=url, 
        idempotency_key="unique-key-123", 
        mime_type="text/plain"
    )
    
    assert doc.source_type == "api"
    assert doc.source_uri == url
    assert doc.raw_text == "Sample downloaded document data content"
    assert doc.mime_type == "text/plain"
    
    # Verify idempotency was checked and registered
    mock_redis_client.get.assert_called_once_with("idip:idempotency:unique-key-123")
    mock_redis_client.set.assert_called_once_with("idip:idempotency:unique-key-123", "processing", ex=86400)

@pytest.mark.asyncio
@respx.mock
async def test_api_adapter_rate_limit_backoff():
    adapter = RESTAPIAdapter()
    url = "https://api.document-source.com/v1/docs/2"
    
    # Setup mock routing: first call is 429, second is 200
    route = respx.get(url)
    route.side_effect = [
        httpx.Response(429, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1"}),
        httpx.Response(200, content=b"Successful fetch after rate limit")
    ]
    
    # Speed up sleep call by patching asyncio.sleep
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        doc = await adapter.ingest(url)
        assert doc.raw_text == "Successful fetch after rate limit"
        assert mock_sleep.call_count == 1

@pytest.mark.asyncio
async def test_webhook_signature_verification():
    adapter = RESTAPIAdapter()
    payload = b'{"event": "document_created", "id": "123"}'
    secret = "webhook-signing-secret"
    
    # Valid signature check
    import hmac, hashlib
    valid_sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert adapter.verify_webhook_signature(payload, valid_sig, secret) is True
    
    # Invalid signature check
    with pytest.raises(SignatureVerificationError):
        adapter.verify_webhook_signature(payload, "invalid_sig", secret)

# Helper patch import
from unittest.mock import patch
