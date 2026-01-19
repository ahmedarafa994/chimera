"""
Rate Limiting Module
Provides distributed rate limiting with Redis and local fallback

Features:
- Sliding window rate limiting
- Role-based rate limits
- Per-endpoint rate limits
- Cost-weighted requests
- Burst protection
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limit Configuration
# =============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit"""

    requests_per_minute: int = 100
    burst_size: int = 20
    cost_weight: int = 1  # How many "requests" this counts as


class RateLimitTier(str, Enum):
    """Rate limit tiers by user role"""

    ADMIN = "admin"
    OPERATOR = "operator"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    API_CLIENT = "api_client"
    ANONYMOUS = "anonymous"


# Default rate limits by tier
DEFAULT_RATE_LIMITS: dict[RateLimitTier, RateLimitConfig] = {
    RateLimitTier.ADMIN: RateLimitConfig(requests_per_minute=1000, burst_size=100),
    RateLimitTier.OPERATOR: RateLimitConfig(requests_per_minute=500, burst_size=50),
    RateLimitTier.DEVELOPER: RateLimitConfig(requests_per_minute=200, burst_size=30),
    RateLimitTier.VIEWER: RateLimitConfig(requests_per_minute=100, burst_size=20),
    RateLimitTier.API_CLIENT: RateLimitConfig(requests_per_minute=300, burst_size=40),
    RateLimitTier.ANONYMOUS: RateLimitConfig(
        requests_per_minute=300, burst_size=50
    ),  # Increased for dev
}

# Endpoint-specific rate limits (path pattern -> config)
ENDPOINT_RATE_LIMITS: dict[str, RateLimitConfig] = {
    "/api/v1/enhance": RateLimitConfig(requests_per_minute=30, cost_weight=3),
    "/api/v1/jailbreak": RateLimitConfig(requests_per_minute=10, cost_weight=5),
    "/api/v1/transform": RateLimitConfig(requests_per_minute=60, cost_weight=2),
    "/api/v1/batch": RateLimitConfig(requests_per_minute=5, cost_weight=10),
    "/health": RateLimitConfig(requests_per_minute=1000, cost_weight=0),
    "/api/v1/health": RateLimitConfig(requests_per_minute=1000, cost_weight=0),
    "/api/v1/models": RateLimitConfig(requests_per_minute=1000, cost_weight=0),
    "/api/v1/providers": RateLimitConfig(requests_per_minute=1000, cost_weight=0),
    "/api/v1/techniques": RateLimitConfig(requests_per_minute=1000, cost_weight=0),
    "/api/v1/session": RateLimitConfig(requests_per_minute=500, cost_weight=0),
    "/api/v1/gptfuzz": RateLimitConfig(requests_per_minute=100, cost_weight=1),
    "/api/v1/intent-aware": RateLimitConfig(requests_per_minute=100, cost_weight=1),
    "/api/v1/generation": RateLimitConfig(requests_per_minute=100, cost_weight=1),
    "/api/v1/metrics": RateLimitConfig(requests_per_minute=500, cost_weight=0),
}


# =============================================================================
# Rate Limiter Interface
# =============================================================================


class RateLimiter:
    """Base rate limiter interface"""

    def check_rate_limit(self, key: str, config: RateLimitConfig) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is within rate limit.

        Returns:
            Tuple of (allowed, info_dict)
            info_dict contains: remaining, reset_at, retry_after
        """
        raise NotImplementedError


# =============================================================================
# Redis Rate Limiter (Production)
# =============================================================================


class RedisRateLimiter(RateLimiter):
    """
    Redis-based sliding window rate limiter.

    Uses Redis sorted sets for accurate sliding window implementation.
    """

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import redis

                self._client = redis.from_url(self.redis_url, decode_responses=True)
                self._client.ping()
                logger.info(f"Redis rate limiter connected: {self.redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to local limiter.")
                return None
        return self._client

    def check_rate_limit(self, key: str, config: RateLimitConfig) -> tuple[bool, dict[str, Any]]:
        if not self.client:
            # Fallback to allowing request if Redis unavailable
            return True, {"remaining": -1, "reset_at": 0, "retry_after": 0}

        now = time.time()
        window_start = now - 60  # 1-minute window

        try:
            pipe = self.client.pipeline()

            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current requests in window
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {f"{now}": now})

            # Set expiry on the key
            pipe.expire(key, 120)

            results = pipe.execute()

            # Validate pipeline results
            if not isinstance(results, list) or len(results) < 2:
                logger.error(f"Invalid Redis pipeline results: {results}")
                return True, {"remaining": -1, "reset_at": 0, "retry_after": 0}

            current_count = results[1]
            if not isinstance(current_count, int):
                logger.error(f"Invalid current_count type: {type(current_count)}")
                return True, {"remaining": -1, "reset_at": 0, "retry_after": 0}

            # Calculate effective limit (accounting for cost weight)
            effective_limit = config.requests_per_minute
            remaining = max(0, effective_limit - current_count - config.cost_weight)

            # Check if allowed
            allowed = current_count < effective_limit

            # Calculate reset time
            oldest_entry = self.client.zrange(key, 0, 0, withscores=True)
            reset_at = int(oldest_entry[0][1]) + 60 if oldest_entry else int(now) + 60

            retry_after = 0 if allowed else max(0, reset_at - int(now))

            return allowed, {
                "remaining": remaining,
                "reset_at": reset_at,
                "retry_after": retry_after,
                "limit": effective_limit,
            }
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open to avoid blocking legitimate requests
            return True, {"remaining": -1, "reset_at": 0, "retry_after": 0}


# =============================================================================
# Local Rate Limiter (Development/Fallback)
# =============================================================================


class LocalRateLimiter(RateLimiter):
    """
    In-memory sliding window rate limiter.

    For single-instance deployments or development.
    WARNING: Not suitable for multi-worker production environments.
    HIGH-004 FIX: Added asyncio.Lock for thread safety.
    """

    def __init__(self):
        self._windows: dict[str, list] = {}
        self._cleanup_threshold = 1000
        self._max_keys = 10000
        self._lock = asyncio.Lock()  # HIGH-004: Thread safety for concurrent access

    def _cleanup_old_entries(self):
        """Remove expired entries to prevent memory bloat"""
        now = time.time()
        window_start = now - 60

        # Prune old timestamps
        keys_to_delete = []
        for key, timestamps in self._windows.items():
            self._windows[key] = [ts for ts in timestamps if ts > window_start]
            if not self._windows[key]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._windows[key]

        # Hard limit on keys if still too many
        if len(self._windows) > self._max_keys:
            # Remove oldest keys (simple heuristic since dict is ordered in recent Py)
            # Convert to list to avoid runtime error during iteration
            keys = list(self._windows.keys())
            excess = len(self._windows) - self._max_keys
            for k in keys[:excess]:
                del self._windows[k]

    async def check_rate_limit(
        self, key: str, config: RateLimitConfig
    ) -> tuple[bool, dict[str, Any]]:
        async with self._lock:
            now = time.time()
            window_start = now - 60

            # Periodic cleanup
            if sum(len(v) for v in self._windows.values()) > self._cleanup_threshold:
                self._cleanup_old_entries()

            # Initialize or get window
            if key not in self._windows:
                self._windows[key] = []

            # Remove old entries
            self._windows[key] = [ts for ts in self._windows[key] if ts > window_start]

            current_count = len(self._windows[key])
            effective_limit = config.requests_per_minute

            # Check if allowed
            allowed = current_count < effective_limit

            if allowed:
                # Add current request (with cost weight)
                for _ in range(config.cost_weight):
                    self._windows[key].append(now)

            remaining = max(0, effective_limit - current_count - config.cost_weight)

            # Calculate reset time
            reset_at = int(self._windows[key][0]) + 60 if self._windows[key] else int(now) + 60

            retry_after = 0 if allowed else max(0, reset_at - int(now))

            return allowed, {
                "remaining": remaining,
                "reset_at": reset_at,
                "retry_after": retry_after,
                "limit": effective_limit,
            }


# =============================================================================
# Rate Limiter Factory
# =============================================================================

_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the configured rate limiter instance"""
    global _rate_limiter

    if _rate_limiter is None:
        redis_url = os.getenv("REDIS_URL")

        if redis_url:
            try:
                _rate_limiter = RedisRateLimiter(redis_url)
                # Test connection
                _rate_limiter.check_rate_limit("test", RateLimitConfig())
                logger.info("Using Redis rate limiter (Distributed)")
            except Exception as e:
                logger.error(f"Redis rate limiter failed: {e}")
                # In production, we might want to raise here, but for now fallback with warning
                logger.warning("Falling back to LocalRateLimiter (Non-distributed)")
                _rate_limiter = LocalRateLimiter()
        else:
            logger.warning(
                "REDIS_URL not set. Using LocalRateLimiter (Non-distributed). Not recommended for production."
            )
            _rate_limiter = LocalRateLimiter()

    return _rate_limiter


# =============================================================================
# FastAPI Integration
# =============================================================================


def get_rate_limit_key(request: Request, user_id: str | None = None) -> str:
    """Generate rate limit key from request"""
    if user_id:
        return f"ratelimit:user:{user_id}"

    # Fall back to IP address for anonymous users
    client_ip = request.client.host if request.client else "unknown"
    return f"ratelimit:ip:{client_ip}"


def get_endpoint_config(path: str) -> RateLimitConfig | None:
    """Get endpoint-specific rate limit config"""
    for pattern, config in ENDPOINT_RATE_LIMITS.items():
        if path.startswith(pattern):
            return config
    return None


async def check_rate_limit(
    request: Request, user_id: str | None = None, tier: RateLimitTier = RateLimitTier.ANONYMOUS
):
    """
    FastAPI dependency for rate limiting.

    Usage:
        @app.get("/api/endpoint")
        async def endpoint(
            rate_limit: None = Depends(
                lambda req: check_rate_limit(req, tier=RateLimitTier.DEVELOPER)
            )
        ):
            ...
    """
    limiter = get_rate_limiter()
    key = get_rate_limit_key(request, user_id)

    # Get config (endpoint-specific or tier-based)
    endpoint_config = get_endpoint_config(request.url.path)
    tier_config = DEFAULT_RATE_LIMITS.get(tier, DEFAULT_RATE_LIMITS[RateLimitTier.ANONYMOUS])

    # Use stricter of the two
    if endpoint_config:
        config = RateLimitConfig(
            requests_per_minute=min(
                endpoint_config.requests_per_minute, tier_config.requests_per_minute
            ),
            burst_size=min(endpoint_config.burst_size, tier_config.burst_size),
            cost_weight=endpoint_config.cost_weight,
        )
    else:
        config = tier_config

    # Check rate limit - handle both sync and async limiters
    result = limiter.check_rate_limit(key, config)
    if asyncio.iscoroutine(result):
        allowed, info = await result
    else:
        allowed, info = result

    # Add rate limit headers to response
    request.state.rate_limit_info = info

    if not allowed:
        # Log rate limit event
        from app.core.audit import AuditAction, audit_log

        audit_log(
            action=AuditAction.SECURITY_RATE_LIMIT,
            user_id=user_id,
            resource=request.url.path,
            details={"limit": info["limit"], "retry_after": info["retry_after"]},
            ip_address=request.client.host if request.client else None,
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info["reset_at"]),
                "Retry-After": str(info["retry_after"]),
            },
        )


# =============================================================================
# Rate Limit Middleware
# =============================================================================


class RateLimitMiddleware:
    """
    Middleware for automatic rate limiting on all requests.

    Usage:
        app.add_middleware(RateLimitMiddleware)
    """

    def __init__(self, app, exclude_paths: list | None = None):
        self.app = app
        # Expanded exclude paths for development and common endpoints
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/v1/health",
            "/api/v1/providers",
            "/api/v1/techniques",
            "/api/v1/models",
            "/api/v1/session",
            "/api/v1/connection",
            "/ws",  # WebSocket endpoints
        ]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request

        request = Request(scope, receive)

        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            await self.app(scope, receive, send)
            return

        # Check rate limit
        try:
            await check_rate_limit(request)
        except HTTPException as e:
            from starlette.responses import JSONResponse

            response = JSONResponse(
                status_code=e.status_code, content={"detail": e.detail}, headers=e.headers
            )
            await response(scope, receive, send)
            return

        # Add rate limit headers to response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                if hasattr(request.state, "rate_limit_info"):
                    info = request.state.rate_limit_info
                    headers[b"X-RateLimit-Limit"] = str(info["limit"]).encode()
                    headers[b"X-RateLimit-Remaining"] = str(info["remaining"]).encode()
                    headers[b"X-RateLimit-Reset"] = str(info["reset_at"]).encode()
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)


# =============================================================================
# Decorator for rate limiting
# =============================================================================


def rate_limit(
    requests_per_minute: int = 60, cost_weight: int = 1, key_func: Callable | None = None
):
    """
    Decorator for rate limiting individual endpoints.

    Usage:
        @app.get("/api/expensive")
        @rate_limit(requests_per_minute=10, cost_weight=5)
        async def expensive_endpoint():
            ...
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            limiter = get_rate_limiter()

            # Get key
            key = key_func(request) if key_func else get_rate_limit_key(request)

            config = RateLimitConfig(
                requests_per_minute=requests_per_minute, cost_weight=cost_weight
            )

            # Check rate limit - handle both sync and async limiters
            result = limiter.check_rate_limit(key, config)
            if asyncio.iscoroutine(result):
                allowed, info = await result
            else:
                allowed, info = result

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(info["retry_after"])},
                )

            return await func(*args, request=request, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Security Headers Middleware
# =============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response
