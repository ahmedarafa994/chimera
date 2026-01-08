"""
Authentication Rate Limiting Middleware

Provides stricter rate limiting for authentication endpoints to prevent:
- Brute force attacks
- Credential stuffing
- API key enumeration

SEC-001: Implements rate limiting for auth endpoints as recommended in security audit.
"""

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.logging import logger


@dataclass
class AuthRateLimitConfig:
    """Configuration for authentication rate limiting."""

    # Login/auth attempts
    auth_requests_per_minute: int = 10
    auth_burst_size: int = 3

    # Failed auth attempts (stricter)
    failed_auth_requests_per_minute: int = 5
    failed_auth_lockout_minutes: int = 15

    # API key validation
    api_key_validation_per_minute: int = 20

    # Session creation
    session_creation_per_minute: int = 30


# Default configuration
DEFAULT_AUTH_RATE_LIMIT_CONFIG = AuthRateLimitConfig()


class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting authentication-related endpoints.

    Implements stricter rate limits for:
    - /api/v1/session (session creation)
    - /api/v1/auth/* (authentication endpoints)
    - /api/v1/connection/test (connection testing)
    - Any endpoint that validates API keys
    """

    # Auth-related path patterns
    AUTH_PATHS: ClassVar[list[str]] = [
        "/api/v1/session",
        "/api/v1/auth",
        "/api/v1/connection/test",
        "/api/v1/connection/mode",
        "/api/v1/providers/select",
    ]

    def __init__(
        self,
        app,
        config: AuthRateLimitConfig | None = None,
        on_rate_limit: Callable[[str, str], None] | None = None,
    ):
        """
        Initialize auth rate limiting middleware.

        Args:
            app: FastAPI application
            config: Rate limit configuration
            on_rate_limit: Optional callback when rate limit is hit
        """
        super().__init__(app)
        self.config = config or DEFAULT_AUTH_RATE_LIMIT_CONFIG
        self.on_rate_limit = on_rate_limit

        # Track requests per IP per endpoint type
        self.auth_requests: dict[str, deque] = defaultdict(lambda: deque())
        self.failed_auth_requests: dict[str, deque] = defaultdict(lambda: deque())
        self.api_key_validations: dict[str, deque] = defaultdict(lambda: deque())
        self.session_creations: dict[str, deque] = defaultdict(lambda: deque())

        # Track lockouts
        self.lockouts: dict[str, float] = {}

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        client_ip = self._get_client_ip(request)

        # Check if this is an auth-related path
        if not self._is_auth_path(path):
            return await call_next(request)

        # Check for lockout
        if self._is_locked_out(client_ip):
            lockout_remaining = self._get_lockout_remaining(client_ip)
            logger.warning(f"AUTH_RATE_LIMIT: Locked out IP {client_ip} attempted access to {path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Too many failed authentication attempts. Please try again later.",
                    "retry_after_seconds": int(lockout_remaining),
                    "lockout_type": "failed_auth",
                },
                headers={"Retry-After": str(int(lockout_remaining))},
            )

        # Apply appropriate rate limit based on endpoint
        rate_limit_result = self._check_rate_limit(client_ip, path, request.method)

        if rate_limit_result["limited"]:
            logger.warning(
                f"AUTH_RATE_LIMIT: {rate_limit_result['limit_type']} exceeded for "
                f"IP {client_ip} on {path}"
            )

            if self.on_rate_limit:
                self.on_rate_limit(client_ip, path)

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded for {rate_limit_result['limit_type']}",
                    "retry_after_seconds": rate_limit_result["retry_after"],
                    "limit_type": rate_limit_result["limit_type"],
                },
                headers={"Retry-After": str(rate_limit_result["retry_after"])},
            )

        # Process request
        response = await call_next(request)

        # Track failed auth attempts for lockout
        if response.status_code in (401, 403):
            self._record_failed_auth(client_ip)

        return response

    def _is_auth_path(self, path: str) -> bool:
        """Check if path is an authentication-related endpoint."""
        return any(path.startswith(auth_path) for auth_path in self.AUTH_PATHS)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _check_rate_limit(self, client_ip: str, path: str, method: str) -> dict:
        """Check rate limit for the given request."""
        now = time.time()

        # Determine which rate limit to apply
        if "/session" in path and method == "POST":
            requests = self.session_creations[client_ip]
            limit = self.config.session_creation_per_minute
            limit_type = "session_creation"
        elif "/providers/select" in path:
            requests = self.api_key_validations[client_ip]
            limit = self.config.api_key_validation_per_minute
            limit_type = "api_key_validation"
        else:
            requests = self.auth_requests[client_ip]
            limit = self.config.auth_requests_per_minute
            limit_type = "auth_request"

        # Clean old requests (older than 1 minute)
        while requests and requests[0] < now - 60:
            requests.popleft()

        # Check if limit exceeded
        if len(requests) >= limit:
            return {
                "limited": True,
                "limit_type": limit_type,
                "retry_after": 60 - int(now - requests[0]) if requests else 60,
            }

        # Record this request
        requests.append(now)

        return {"limited": False, "limit_type": limit_type, "retry_after": 0}

    def _record_failed_auth(self, client_ip: str):
        """Record a failed authentication attempt."""
        now = time.time()
        requests = self.failed_auth_requests[client_ip]

        # Clean old failed attempts
        while requests and requests[0] < now - 60:
            requests.popleft()

        requests.append(now)

        # Check if lockout threshold reached
        if len(requests) >= self.config.failed_auth_requests_per_minute:
            lockout_until = now + (self.config.failed_auth_lockout_minutes * 60)
            self.lockouts[client_ip] = lockout_until
            logger.warning(
                f"AUTH_RATE_LIMIT: IP {client_ip} locked out for "
                f"{self.config.failed_auth_lockout_minutes} minutes due to "
                f"{len(requests)} failed auth attempts"
            )

    def _is_locked_out(self, client_ip: str) -> bool:
        """Check if client IP is currently locked out."""
        if client_ip not in self.lockouts:
            return False

        if time.time() >= self.lockouts[client_ip]:
            # Lockout expired, remove it
            del self.lockouts[client_ip]
            # Also clear failed auth history
            self.failed_auth_requests[client_ip].clear()
            return False

        return True

    def _get_lockout_remaining(self, client_ip: str) -> float:
        """Get remaining lockout time in seconds."""
        if client_ip not in self.lockouts:
            return 0
        return max(0, self.lockouts[client_ip] - time.time())

    def get_stats(self) -> dict:
        """Get rate limiting statistics."""
        return {
            "active_lockouts": len(self.lockouts),
            "tracked_ips": {
                "auth_requests": len(self.auth_requests),
                "failed_auth": len(self.failed_auth_requests),
                "api_key_validations": len(self.api_key_validations),
                "session_creations": len(self.session_creations),
            },
            "config": {
                "auth_requests_per_minute": self.config.auth_requests_per_minute,
                "failed_auth_lockout_minutes": self.config.failed_auth_lockout_minutes,
            },
        }


# Dependency for getting rate limit info in endpoints
def get_auth_rate_limiter() -> AuthRateLimitMiddleware | None:
    """Get the auth rate limiter instance if available."""
    # This would be set during app startup
    return getattr(get_auth_rate_limiter, "_instance", None)


def set_auth_rate_limiter(instance: AuthRateLimitMiddleware):
    """Set the auth rate limiter instance."""
    get_auth_rate_limiter._instance = instance
