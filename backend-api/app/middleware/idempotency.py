"""Idempotency Middleware - Request deduplication for POST endpoints.

Provides:
- Idempotency key validation and handling
- Response caching for duplicate requests
- Configurable TTL for idempotency keys
- In-progress request locking

Usage:
    In main.py:
        from app.middleware.idempotency import IdempotencyMiddleware
        app.add_middleware(IdempotencyMiddleware)

    In client requests:
        headers: {"Idempotency-Key": "unique-request-id"}
"""

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Cached response for idempotent requests."""

    body: bytes
    status_code: int
    headers: dict[str, str]
    timestamp: float


@dataclass
class IdempotencyLock:
    """Lock for in-progress requests."""

    started_at: float
    request_hash: str


class IdempotencyStore:
    """In-memory storage for idempotency keys.

    For production, replace with Redis-backed implementation.
    """

    def __init__(self, ttl: int = 86400, cleanup_interval: int = 3600) -> None:
        self._responses: dict[str, CachedResponse] = {}
        self._locks: dict[str, IdempotencyLock] = {}
        self._ttl = ttl
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def get_response(self, key: str) -> CachedResponse | None:
        """Get cached response by idempotency key."""
        self._maybe_cleanup()
        cached = self._responses.get(key)
        if cached and (time.time() - cached.timestamp) < self._ttl:
            return cached
        if cached:
            # Expired, remove it
            del self._responses[key]
        return None

    def set_response(
        self,
        key: str,
        body: bytes,
        status_code: int,
        headers: dict[str, str],
    ) -> None:
        """Cache a response for an idempotency key."""
        self._responses[key] = CachedResponse(
            body=body,
            status_code=status_code,
            headers=headers,
            timestamp=time.time(),
        )
        # Remove lock since request completed
        self._locks.pop(key, None)

    def acquire_lock(self, key: str, request_hash: str) -> bool:
        """Acquire a lock for an in-progress request.

        Returns True if lock acquired, False if request already in progress.
        """
        existing = self._locks.get(key)
        if existing:
            # Check if lock is stale (> 60 seconds)
            if time.time() - existing.started_at > 60:
                logger.warning(f"Idempotency lock expired for key: {key}")
            elif existing.request_hash != request_hash:
                # Different request with same key - conflict
                return False

        self._locks[key] = IdempotencyLock(
            started_at=time.time(),
            request_hash=request_hash,
        )
        return True

    def release_lock(self, key: str) -> None:
        """Release a lock."""
        self._locks.pop(key, None)

    def is_locked(self, key: str) -> bool:
        """Check if a key is locked."""
        lock = self._locks.get(key)
        return bool(lock and time.time() - lock.started_at < 60)

    def _maybe_cleanup(self) -> None:
        """Periodic cleanup of expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        expired_keys = []

        for key, cached in self._responses.items():
            if now - cached.timestamp >= self._ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._responses[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired idempotency keys")

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        return {
            "cached_responses": len(self._responses),
            "active_locks": len(self._locks),
            "ttl_seconds": self._ttl,
        }


# Global store instance
_idempotency_store = IdempotencyStore()


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Middleware for handling idempotent POST/PUT/PATCH requests.

    When a request includes an Idempotency-Key header:
    1. If we have a cached response for this key, return it
    2. If request is in progress, return 409 Conflict
    3. Otherwise, process request and cache the response
    """

    IDEMPOTENCY_HEADER = "Idempotency-Key"
    TTL_HEADER = "Idempotency-TTL"
    DEFAULT_TTL = 86400  # 24 hours

    # Endpoints to apply idempotency to
    IDEMPOTENT_PATHS: ClassVar[set[str]] = {
        "/api/v1/generate",
        "/api/v1/transform",
        "/api/v1/execute",
        "/api/v1/generation/jailbreak/generate",
        "/api/v1/session/create",
    }

    def __init__(
        self,
        app,
        store: IdempotencyStore | None = None,
        enabled_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._store = store or _idempotency_store
        self._enabled_paths = enabled_paths or self.IDEMPOTENT_PATHS

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only apply to mutation methods
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        # Check if path is enabled for idempotency
        if request.url.path not in self._enabled_paths:
            return await call_next(request)

        # Get idempotency key from header
        idempotency_key = request.headers.get(self.IDEMPOTENCY_HEADER)
        if not idempotency_key:
            # No key provided, process normally
            return await call_next(request)

        # Validate key format (max 255 chars, alphanumeric + dashes)
        if len(idempotency_key) > 255:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_IDEMPOTENCY_KEY",
                    "message": "Idempotency key must be at most 255 characters",
                },
            )

        # Create cache key with path to avoid collisions
        cache_key = f"{request.url.path}:{idempotency_key}"

        # Check for cached response
        cached = self._store.get_response(cache_key)
        if cached:
            logger.debug(f"Returning cached response for idempotency key: {idempotency_key}")
            return Response(
                content=cached.body,
                status_code=cached.status_code,
                headers={
                    **cached.headers,
                    "X-Idempotent-Cached": "true",
                    "X-Idempotency-Key": idempotency_key,
                },
            )

        # Check if request is in progress
        if self._store.is_locked(cache_key):
            return JSONResponse(
                status_code=409,
                content={
                    "error": "IDEMPOTENT_REQUEST_IN_PROGRESS",
                    "message": "A request with this idempotency key is already being processed",
                },
                headers={"X-Idempotency-Key": idempotency_key},
            )

        # Create request hash for conflict detection
        body = await request.body()
        request_hash = hashlib.sha256(body).hexdigest()

        # Try to acquire lock
        if not self._store.acquire_lock(cache_key, request_hash):
            return JSONResponse(
                status_code=409,
                content={
                    "error": "IDEMPOTENCY_KEY_CONFLICT",
                    "message": "This idempotency key was used with a different request body",
                },
                headers={"X-Idempotency-Key": idempotency_key},
            )

        try:
            # Process request
            # We need to create a new request with the body since we consumed it
            from starlette.requests import Request as StarletteRequest

            # Create scope copy
            scope = dict(request.scope)

            # Create a receive function that returns the body we already read
            async def receive():
                return {"type": "http.request", "body": body}

            # Create new request with the body
            new_request = StarletteRequest(scope, receive)

            response = await call_next(new_request)

            # Cache successful responses (2xx status codes)
            if 200 <= response.status_code < 300:
                # Read response body
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk

                # Cache the response
                response_headers = dict(response.headers)
                self._store.set_response(
                    cache_key,
                    response_body,
                    response.status_code,
                    response_headers,
                )

                # Return new response with body
                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers={
                        **response_headers,
                        "X-Idempotency-Key": idempotency_key,
                        "X-Idempotent-Cached": "false",
                    },
                )
            # Don't cache error responses, but release lock
            self._store.release_lock(cache_key)

            return response

        except Exception:
            # Release lock on error
            self._store.release_lock(cache_key)
            raise


def get_idempotency_store() -> IdempotencyStore:
    """Get the global idempotency store."""
    return _idempotency_store
