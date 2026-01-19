"""
Request Logging Middleware

Provides:
- Automatic request/response logging
- Request ID generation and propagation
- Performance timing
- Error tracking integration
"""

import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.structured_logging import (
    clear_request_context,
    error_tracker,
    generate_request_id,
    logger,
    set_request_context,
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request logging.

    Features:
    - Generates unique request IDs
    - Logs request/response details
    - Tracks performance metrics
    - Captures errors with context

    PERF-026: Optimized to reduce duplicate logging and skip known 404 patterns.
    """

    # PERF-027: Known non-existent paths that generate high 404 traffic
    KNOWN_404_PREFIXES = frozenset(
        [
            "/v1/chat/",
            "/v1/messages",
            "/v1/models",
            "/v1/api/event_logging",
            "/v1/v1/",  # Double-prefixed paths from misconfigured clients
        ]
    )

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list[str] | None = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        slow_request_threshold_ms: float = 5000,
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.slow_request_threshold_ms = slow_request_threshold_ms
        # PERF-028: Track 404 counts to reduce log spam
        self._404_counts: dict[str, int] = {}
        self._404_log_interval = 100  # Log summary every N requests

    def _is_known_404_pattern(self, path: str) -> bool:
        """Check if path matches known 404 patterns (PERF-027)."""
        return any(path.startswith(prefix) for prefix in self.KNOWN_404_PREFIXES)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip logging for excluded paths
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # PERF-027: Fast-path for known 404 patterns - skip detailed logging
        is_known_404 = self._is_known_404_pattern(path)

        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or generate_request_id()

        # Extract user/session info from headers
        user_id = request.headers.get("X-User-ID", "")
        session_id = request.headers.get("X-Session-ID", "")

        # Set request context for logging
        set_request_context(request_id=request_id, user_id=user_id, session_id=session_id)

        # Start timing
        start_time = time.perf_counter()

        # PERF-029: Only build detailed request data if we're going to log it
        request_data = None

        # PERF-030: Skip "Request started" log for known 404 patterns
        if not is_known_404:
            request_data = {
                "method": request.method,
                "path": path,
                "query_params": dict(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("User-Agent", ""),
                "content_type": request.headers.get("Content-Type", ""),
                "content_length": request.headers.get("Content-Length", "0"),
            }

            logger.info(
                f"Request started: {request.method} {path}",
                extra={"extra_data": {"request": request_data}},
            )

        # Process request
        response = None
        error = None

        try:
            response = await call_next(request)
            return response
        except Exception as e:
            error = e
            # Build request_data if not already built (for error logging)
            if request_data is None:
                request_data = {
                    "method": request.method,
                    "path": path,
                    "client_ip": self._get_client_ip(request),
                }
            # Capture exception with context
            error_tracker.capture_exception(
                exception=e,
                context={
                    "request": request_data,
                    "request_id": request_id,
                },
                tags={
                    "endpoint": path,
                    "method": request.method,
                },
            )
            raise
        finally:
            # Calculate elapsed time
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # PERF-028: Rate-limit logging for known 404 patterns
            status_code = response.status_code if response else 500
            should_log = True

            if is_known_404 and status_code == 404:
                # Track 404 counts and only log periodically
                self._404_counts[path] = self._404_counts.get(path, 0) + 1
                count = self._404_counts[path]

                # Log first occurrence and then every N occurrences
                if count == 1:
                    should_log = True
                elif count % self._404_log_interval == 0:
                    # Log summary instead of individual request
                    logger.warning(
                        f"Known 404 pattern: {request.method} {path} - {count} occurrences",
                        extra={
                            "extra_data": {"path": path, "count": count, "request_id": request_id}
                        },
                    )
                    should_log = False
                else:
                    should_log = False

            if should_log:
                # Build request_data if not already built
                if request_data is None:
                    request_data = {
                        "method": request.method,
                        "path": path,
                        "client_ip": self._get_client_ip(request),
                    }

                # Build response data
                response_data = {
                    "status_code": status_code,
                    "elapsed_ms": round(elapsed_ms, 2),
                }

                # Log completion
                log_data = {
                    "request": request_data,
                    "response": response_data,
                    "request_id": request_id,
                }

                if error:
                    log_data["error"] = str(error)
                    logger.error(
                        f"Request failed: {request.method} {path} - {error}",
                        extra={"extra_data": log_data},
                    )
                elif elapsed_ms > self.slow_request_threshold_ms:
                    logger.warning(
                        f"Slow request: {request.method} {path} took {elapsed_ms:.2f}ms",
                        extra={"extra_data": log_data},
                    )
                else:
                    logger.info(
                        f"Request completed: {request.method} {path} - {status_code} in {elapsed_ms:.2f}ms",
                        extra={"extra_data": log_data},
                    )

            # Add request ID to response headers
            if response:
                response.headers["X-Request-ID"] = request_id

            # Clear request context
            clear_request_context()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client
        if request.client:
            return request.client.host

        return "unknown"


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.

    Tracks:
    - Request counts by endpoint
    - Response times
    - Error rates
    - Status code distribution
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._request_count = 0
        self._error_count = 0
        self._total_response_time_ms = 0.0
        self._status_codes: dict[int, int] = {}
        self._endpoint_metrics: dict[str, dict] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            self._error_count += 1
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._request_count += 1
            self._total_response_time_ms += elapsed_ms

            # Track status codes
            self._status_codes[status_code] = self._status_codes.get(status_code, 0) + 1

            # Track per-endpoint metrics
            endpoint = f"{request.method} {request.url.path}"
            if endpoint not in self._endpoint_metrics:
                self._endpoint_metrics[endpoint] = {
                    "count": 0,
                    "total_time_ms": 0.0,
                    "errors": 0,
                }

            self._endpoint_metrics[endpoint]["count"] += 1
            self._endpoint_metrics[endpoint]["total_time_ms"] += elapsed_ms
            if status_code >= 400:
                self._endpoint_metrics[endpoint]["errors"] += 1

        return response

    def get_metrics(self) -> dict:
        """Get current metrics."""
        avg_response_time = (
            self._total_response_time_ms / self._request_count if self._request_count > 0 else 0
        )

        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": self._error_count / self._request_count if self._request_count > 0 else 0,
            "avg_response_time_ms": round(avg_response_time, 2),
            "status_codes": self._status_codes,
            "endpoints": {
                endpoint: {
                    "count": data["count"],
                    "avg_time_ms": (
                        round(data["total_time_ms"] / data["count"], 2) if data["count"] > 0 else 0
                    ),
                    "error_rate": data["errors"] / data["count"] if data["count"] > 0 else 0,
                }
                for endpoint, data in self._endpoint_metrics.items()
            },
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self._request_count = 0
        self._error_count = 0
        self._total_response_time_ms = 0.0
        self._status_codes.clear()
        self._endpoint_metrics.clear()


# Global metrics middleware instance for access from endpoints
metrics_middleware: MetricsMiddleware | None = None


def get_metrics_middleware() -> MetricsMiddleware | None:
    """Get the global metrics middleware instance."""
    return metrics_middleware


def set_metrics_middleware(middleware: MetricsMiddleware):
    """Set the global metrics middleware instance."""
    global metrics_middleware
    metrics_middleware = middleware
