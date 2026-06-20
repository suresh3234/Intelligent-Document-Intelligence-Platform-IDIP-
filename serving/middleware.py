"""FastAPI middle-tier pipeline (middlewares) for IDIP serving layer."""
import uuid
import time
import logging
import hmac
import hashlib
import base64
import json
from typing import Callable, Dict, Any, Optional
import numpy as np

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

logger = logging.getLogger("idip.serving.middleware")

def decode_jwt_payload(token: str, secret: str) -> Dict[str, Any]:
    """
    Decodes and verifies a JWT token using HMAC-SHA256.
    Avoids third-party dependencies by using standard Python cryptography primitives.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token structure. Token must have 3 parts.")

    header_b64, payload_b64, signature_b64 = parts

    def b64_url_decode(s: str) -> bytes:
        # Add padding to base64 string if necessary
        padded = s + "=" * (4 - len(s) % 4)
        return base64.urlsafe_b64decode(padded.encode())

    # Verify signature
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig_bytes = b64_url_decode(signature_b64)
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()

    if not hmac.compare_digest(sig_bytes, expected_sig):
        raise ValueError("JWT signature verification failed.")

    # Parse payload
    payload = json.loads(b64_url_decode(payload_b64).decode("utf-8"))

    # Expiry verification
    if "exp" in payload:
        if time.time() > float(payload["exp"]):
            raise ValueError("JWT token has expired.")

    return payload

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware injecting a unique X-Request-ID identifier header into requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = req_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware measuring endpoint latency and appending X-Response-Time headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000
        
        response.headers["X-Response-Time"] = f"{duration:.2f}ms"
        # Log latency metric
        logger.info(f"API Request: {request.method} {request.url.path} completed in {duration:.2f}ms")
        return response

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware checking JWT token authorization (HMAC-SHA256 signature matching)."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exempt health check, metrics, retraining, evaluate, static dashboard, and SSE status paths from auth
        if (request.url.path in ("/", "/index.html", "/v1/health", "/v1/metrics", "/metrics", "/docs", "/openapi.json", "/v1/admin/retrain", "/v1/admin/evaluate") or 
            (request.url.path.startswith("/v1/documents/") and request.url.path.endswith("/status"))):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error_code": "UNAUTHORIZED", "message": "Missing or malformed Authorization header."}
            )

        token = auth_header.split(" ")[1]
        jwt_secret = getattr(settings, "JWT_SECRET", "idip_secret_key_1234567890")

        try:
            payload = decode_jwt_payload(token, jwt_secret)
            # Inject api_key/client_id claim to request state for rate limiter mapping
            request.state.api_key = payload.get("api_key", payload.get("sub", "anonymous"))
        except Exception as e:
            logger.warning(f"Authentication failed: {e}")
            return JSONResponse(
                status_code=401,
                content={"error_code": "UNAUTHORIZED", "message": f"Invalid token signature: {e}"}
            )

        return await call_next(request)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing a sliding window rate limit of 100 requests per minute per api_key."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exempt health, metrics, static dashboard, and SSE status paths
        if (request.url.path in ("/", "/index.html", "/v1/health", "/v1/metrics", "/metrics", "/docs", "/openapi.json") or
            (request.url.path.startswith("/v1/documents/") and request.url.path.endswith("/status"))):
            return await call_next(request)

        # Retrieve api_key mapped by AuthMiddleware
        api_key = getattr(request.state, "api_key", "anonymous")
        
        # Access redis client
        try:
            from serving.dependencies import get_redis_client
            r = get_redis_client()
        except Exception as e:
            logger.error(f"Redis not available for rate limiting: {e}. Bypassing rate limit.")
            return await call_next(request)

        # Sliding window check
        now = time.time()
        window_start = now - 60.0
        zset_key = f"rate_limit:{api_key}"

        try:
            pipe = r.pipeline()
            # Clear old timestamps
            pipe.zremrangebyscore(zset_key, 0, window_start)
            # Count remaining requests
            pipe.zcard(zset_key)
            _, current_count = pipe.execute()

            if current_count >= 100:
                logger.warning(f"Rate limit exceeded for api_key: {api_key}. Request count: {current_count}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Limit is 100 requests per minute."
                    }
                )

            # Record this request timestamp
            pipe = r.pipeline()
            pipe.zadd(zset_key, {str(now): now})
            pipe.expire(zset_key, 60)
            pipe.execute()
        except Exception as e:
            logger.error(f"Rate limiting sliding window execution failed: {e}")
            return JSONResponse(
                status_code=504,
                content={
                    "error_code": "GATEWAY_TIMEOUT",
                    "message": f"Gateway Timeout: Redis connection failed: {e}"
                }
            )

        return await call_next(request)

class SemanticCacheMiddleware(BaseHTTPMiddleware):
    """Middleware checking semantic cache hits in Redis before RAG pipeline execution."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache RAG query path
        if request.method != "POST" or request.url.path != "/v1/query":
            return await call_next(request)

        # Intercept body content safely
        body = await request.body()
        try:
            data = json.loads(body.decode("utf-8"))
            query = data.get("query", "")
        except Exception:
            # Fallback if request body is empty/malformed
            return await call_next(request)

        if not query:
            return await call_next(request)

        # Fetch redis connection
        try:
            from serving.dependencies import get_redis_client
            r = get_redis_client()
        except Exception:
            return await call_next(request)

        # Generate query embedding
        try:
            embedding_service = request.app.state.embedding_service
            # L2 normalized vector
            query_vector = embedding_service.encode_query(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding in SemanticCache: {e}")
            return await call_next(request)

        # Query semantic cache matches
        cache_hit_found = False
        cached_response_bytes = None

        try:
            # We store embeddings in a Redis hash 'semantic_cache:embeddings' -> field: query_text, val: serialized numpy array
            # And responses in a Redis hash 'semantic_cache:responses' -> field: query_text, val: json string
            embeddings_dict = r.hgetall("semantic_cache:embeddings")
            
            for cached_query, serialized_vec_str in embeddings_dict.items():
                cached_vector = np.array(list(map(float, serialized_vec_str.split(","))), dtype=np.float32)
                # Compute cosine similarity (since vectors are L2 normalized, similarity = dot product)
                similarity = np.dot(query_vector, cached_vector)
                
                if similarity >= 0.95:
                    logger.info(f"Semantic Cache Hit! Similarity {similarity:.4f} for query: '{query}' -> '{cached_query}'")
                    cached_response_str = r.hget("semantic_cache:responses", cached_query)
                    if cached_response_str:
                        cache_hit_found = True
                        cached_response_bytes = cached_response_str.encode("utf-8")
                        break
        except Exception as e:
            logger.error(f"Semantic Cache retrieval failed: {e}")

        if cache_hit_found and cached_response_bytes:
            return Response(
                content=cached_response_bytes,
                media_type="application/json",
                headers={
                    "X-Cache-Lookup": "HIT",
                    "X-Request-ID": getattr(request.state, "request_id", "")
                }
            )

        # Modify request state so downstream API knows it is a cache miss
        request.state.query_vector = query_vector
        request.state.query_text = query

        # Process request downstream
        response = await call_next(request)

        # Cache response if successful (HTTP 200) and it was a cache miss
        if response.status_code == 200:
            # We must reconstruct the response body stream to read it
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            try:
                # Save to cache
                serialized_vector = ",".join(map(str, query_vector.tolist()))
                r.hset("semantic_cache:embeddings", query, serialized_vector)
                r.hset("semantic_cache:responses", query, response_body.decode("utf-8"))
                # Set TTL key to clean up cache hash over time
                r.expire("semantic_cache:embeddings", settings.CACHE_TTL)
                r.expire("semantic_cache:responses", settings.CACHE_TTL)
            except Exception as e:
                logger.error(f"Failed to cache response in Redis: {e}")

            # Return a new response because we consumed the body iterator
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

        return response
