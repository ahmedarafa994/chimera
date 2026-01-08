"""
Middleware module for the backend API.
Provides authentication, logging, error handling, and other cross-cutting concerns using FastAPI/Starlette.
"""

import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import config

logger = logging.getLogger(__name__)

# API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Dependency to verify API key.
    """
    if not config.security.enable_api_key_auth:
        return

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header in your request.",
        )

    if not config.security.validate_api_key(api_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

    return api_key


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log incoming requests and outgoing responses.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger("app.middleware.logging")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for health checks if needed, but keeping it for now
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Add request ID to request state so endpoints can access it
        request.state.request_id = request_id

        # Log request
        if config.security.enable_request_logging:
            self.logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_host": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown"),
                },
            )

        try:
            response = await call_next(request)

            # Add headers
            response.headers["X-Request-ID"] = request_id

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            if config.security.enable_request_logging:
                self.logger.info(
                    f"Request completed: {response.status_code}",
                    extra={
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                    },
                )

            return response

        except Exception as e:
            # Log exception
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"Request failed: {e!s}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration_ms": duration_ms,
                },
                exc_info=True,
            )
            raise e


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter middleware.
    """

    def __init__(self, app: ASGIApp, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # IP -> [(timestamp, count)]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for non-API routes if needed
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        identifier = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean and count
        if identifier in self.requests:
            self.requests[identifier] = [
                (ts, count)
                for ts, count in self.requests[identifier]
                if current_time - ts < self.window_seconds
            ]
        else:
            self.requests[identifier] = []

        request_count = sum(count for _, count in self.requests[identifier])

        if request_count >= self.max_requests:
            # Rate limit exceeded
            return Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + self.window_seconds)),
                },
            )

        # Increment count
        self.requests[identifier].append((current_time, 1))

        # Process request
        response = await call_next(request)

        # Add headers
        remaining = max(0, self.max_requests - request_count - 1)
        reset_time = int(current_time + self.window_seconds)

        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def setup_middleware(app: FastAPI):
    """
    Set up all middleware for the FastAPI application.
    """
    # Security Headers (Inner - executed last in request flow, first in response)
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate Limiting
    app.add_middleware(
        RateLimitMiddleware, max_requests=config.security.rate_limit_per_minute, window_seconds=60
    )

    # Request Logging (Outer - executed first)
    app.add_middleware(RequestLoggingMiddleware)
