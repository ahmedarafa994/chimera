"""Provider Rate Limiting Middleware.

Config-driven rate limiting per provider that:
1. Gets provider from request state (set by ProviderSelectionMiddleware)
2. Checks rate limits from configuration
3. Tracks request counts per provider per client
4. Returns 429 if limit exceeded
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


@dataclass
class RateLimitWindow:
    """Represents a rate limit tracking window."""

    count: int = 0
    window_start: float = 0.0
    tokens_used: int = 0


@dataclass
class RateLimitEntry:
    """Rate limit tracking entry for a client/provider combination."""

    minute_window: RateLimitWindow = field(default_factory=RateLimitWindow)
    day_window: RateLimitWindow = field(default_factory=RateLimitWindow)
    last_request_time: float = 0.0


class RateLimitTracker:
    """Track rate limits per provider per client.

    Thread-safe implementation using locks for concurrent access.
    Supports multiple time windows (per-minute, per-day).
    """

    def __init__(
        self,
        cleanup_interval: int = 300,
        max_entries: int = 10000,
    ) -> None:
        """Initialize rate limit tracker.

        Args:
            cleanup_interval: Seconds between cleanup of stale entries
            max_entries: Maximum number of entries to track

        """
        self._entries: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._lock = Lock()
        self._cleanup_interval = cleanup_interval
        self._max_entries = max_entries
        self._last_cleanup = time.time()
        self._config_manager = None

    def _get_config_manager(self):
        """Lazily get the AI config manager."""
        if self._config_manager is None:
            try:
                from app.core.ai_config_manager import get_ai_config_manager

                self._config_manager = get_ai_config_manager()
            except ImportError:
                logger.warning("AIConfigManager not available for rate limiting")
        return self._config_manager

    def _get_key(self, provider: str, client_id: str) -> str:
        """Generate cache key for provider/client combination."""
        return f"{provider}:{client_id}"

    def _get_rate_limits(self, provider: str) -> dict[str, Any]:
        """Get rate limits for a provider from configuration.

        Args:
            provider: Provider ID

        Returns:
            Dict with rate limit configuration

        """
        config_manager = self._get_config_manager()
        if not config_manager or not config_manager.is_loaded():
            # Default limits when no config available
            return {
                "requests_per_minute": 60,
                "tokens_per_minute": 100000,
                "requests_per_day": None,
            }

        config = config_manager.get_config()
        provider_config = config.get_provider(provider)

        if not provider_config:
            return {
                "requests_per_minute": 60,
                "tokens_per_minute": 100000,
                "requests_per_day": None,
            }

        rate_limits = provider_config.rate_limits
        return {
            "requests_per_minute": rate_limits.requests_per_minute,
            "tokens_per_minute": rate_limits.tokens_per_minute,
            "requests_per_day": rate_limits.requests_per_day,
        }

    def check_limit(
        self,
        provider: str,
        client_id: str,
    ) -> dict[str, Any]:
        """Check if request is within rate limits.

        Args:
            provider: Provider ID
            client_id: Client identifier

        Returns:
            Dict with:
                - allowed: bool indicating if request is allowed
                - reason: str reason if not allowed
                - retry_after: int seconds to wait before retry
                - remaining: dict with remaining quota

        """
        key = self._get_key(provider, client_id)
        current_time = time.time()
        limits = self._get_rate_limits(provider)

        with self._lock:
            # Periodic cleanup
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup_stale_entries(current_time)
                self._last_cleanup = current_time

            entry = self._entries[key]

            # Check per-minute limit
            minute_limit = limits.get("requests_per_minute", 60)
            minute_window_start = current_time - (current_time % 60)

            if entry.minute_window.window_start != minute_window_start:
                # New minute window
                entry.minute_window = RateLimitWindow(
                    count=0,
                    window_start=minute_window_start,
                    tokens_used=0,
                )

            if entry.minute_window.count >= minute_limit:
                retry_after = int(60 - (current_time % 60))
                return {
                    "allowed": False,
                    "reason": f"Rate limit exceeded: {minute_limit}/minute",
                    "retry_after": retry_after,
                    "remaining": self._get_remaining_dict(entry, limits, current_time),
                }

            # Check per-day limit if configured
            day_limit = limits.get("requests_per_day")
            if day_limit:
                day_window_start = current_time - (current_time % 86400)

                if entry.day_window.window_start != day_window_start:
                    # New day window
                    entry.day_window = RateLimitWindow(
                        count=0,
                        window_start=day_window_start,
                        tokens_used=0,
                    )

                if entry.day_window.count >= day_limit:
                    retry_after = int(86400 - (current_time % 86400))
                    return {
                        "allowed": False,
                        "reason": f"Daily limit exceeded: {day_limit}/day",
                        "retry_after": retry_after,
                        "remaining": self._get_remaining_dict(entry, limits, current_time),
                    }

            return {
                "allowed": True,
                "remaining": self._get_remaining_dict(entry, limits, current_time),
            }

    def record_request(
        self,
        provider: str,
        client_id: str,
        tokens: int = 0,
    ) -> None:
        """Record a request for rate tracking.

        Args:
            provider: Provider ID
            client_id: Client identifier
            tokens: Number of tokens used (optional)

        """
        key = self._get_key(provider, client_id)
        current_time = time.time()

        with self._lock:
            entry = self._entries[key]
            entry.last_request_time = current_time

            # Update minute window
            minute_window_start = current_time - (current_time % 60)
            if entry.minute_window.window_start != minute_window_start:
                entry.minute_window = RateLimitWindow(
                    count=1,
                    window_start=minute_window_start,
                    tokens_used=tokens,
                )
            else:
                entry.minute_window.count += 1
                entry.minute_window.tokens_used += tokens

            # Update day window
            day_window_start = current_time - (current_time % 86400)
            if entry.day_window.window_start != day_window_start:
                entry.day_window = RateLimitWindow(
                    count=1,
                    window_start=day_window_start,
                    tokens_used=tokens,
                )
            else:
                entry.day_window.count += 1
                entry.day_window.tokens_used += tokens

    def get_remaining(
        self,
        provider: str,
        client_id: str,
    ) -> dict[str, Any]:
        """Get remaining quota for provider.

        Args:
            provider: Provider ID
            client_id: Client identifier

        Returns:
            Dict with remaining quota information

        """
        key = self._get_key(provider, client_id)
        current_time = time.time()
        limits = self._get_rate_limits(provider)

        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return {
                    "requests_remaining_minute": limits.get("requests_per_minute", 60),
                    "tokens_remaining_minute": limits.get("tokens_per_minute", 100000),
                    "requests_remaining_day": limits.get("requests_per_day"),
                    "reset_at_minute": int(current_time - (current_time % 60) + 60),
                    "reset_at_day": int(current_time - (current_time % 86400) + 86400),
                }

            return self._get_remaining_dict(entry, limits, current_time)

    def _get_remaining_dict(
        self,
        entry: RateLimitEntry,
        limits: dict[str, Any],
        current_time: float,
    ) -> dict[str, Any]:
        """Build remaining quota dictionary."""
        minute_limit = limits.get("requests_per_minute", 60)
        token_limit = limits.get("tokens_per_minute", 100000)
        day_limit = limits.get("requests_per_day")

        minute_window_start = current_time - (current_time % 60)
        day_window_start = current_time - (current_time % 86400)

        # Calculate minute remaining
        minute_used = 0
        minute_tokens = 0
        if entry.minute_window.window_start == minute_window_start:
            minute_used = entry.minute_window.count
            minute_tokens = entry.minute_window.tokens_used

        # Calculate day remaining
        day_used = 0
        if day_limit and entry.day_window.window_start == day_window_start:
            day_used = entry.day_window.count

        return {
            "requests_remaining_minute": max(0, minute_limit - minute_used),
            "tokens_remaining_minute": max(0, token_limit - minute_tokens),
            "requests_remaining_day": (max(0, day_limit - day_used) if day_limit else None),
            "reset_at_minute": int(minute_window_start + 60),
            "reset_at_day": int(day_window_start + 86400) if day_limit else None,
        }

    def _cleanup_stale_entries(self, current_time: float) -> None:
        """Remove stale entries to prevent memory growth."""
        stale_threshold = current_time - 3600  # 1 hour

        keys_to_remove = [
            key for key, entry in self._entries.items() if entry.last_request_time < stale_threshold
        ]

        for key in keys_to_remove:
            del self._entries[key]

        # If still too many entries, remove oldest
        if len(self._entries) > self._max_entries:
            sorted_entries = sorted(
                self._entries.items(),
                key=lambda x: x[1].last_request_time,
            )
            for key, _ in sorted_entries[: len(sorted_entries) - self._max_entries]:
                del self._entries[key]

        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} stale rate limit entries")

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "last_cleanup": datetime.fromtimestamp(self._last_cleanup).isoformat(),
            }

    def reset(self, provider: str | None = None) -> None:
        """Reset rate limit tracking.

        Args:
            provider: Optional provider to reset (resets all if None)

        """
        with self._lock:
            if provider:
                keys_to_remove = [key for key in self._entries if key.startswith(f"{provider}:")]
                for key in keys_to_remove:
                    del self._entries[key]
            else:
                self._entries.clear()


# Global rate limit tracker instance
_rate_limit_tracker: RateLimitTracker | None = None


def get_rate_limit_tracker() -> RateLimitTracker:
    """Get the global rate limit tracker instance."""
    global _rate_limit_tracker
    if _rate_limit_tracker is None:
        _rate_limit_tracker = RateLimitTracker()
    return _rate_limit_tracker


class ProviderRateLimitMiddleware(BaseHTTPMiddleware):
    """Config-driven rate limiting per provider middleware.

    This middleware:
    1. Gets provider from request state (set by ProviderSelectionMiddleware)
    2. Checks rate limits from configuration
    3. Tracks request counts per provider per client
    4. Returns 429 if limit exceeded

    Rate limit headers are added to all responses:
    - X-RateLimit-Limit: Maximum requests per minute
    - X-RateLimit-Remaining: Remaining requests in current window
    - X-RateLimit-Reset: Unix timestamp when window resets
    """

    # Paths to skip rate limiting
    SKIP_PATHS = [
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
        "/api/v1/health",
        "/api/v1/auth",  # Exclude auth
    ]

    def __init__(
        self,
        app,
        *,
        enabled: bool = True,
        tracker: RateLimitTracker | None = None,
    ) -> None:
        """Initialize rate limit middleware.

        Args:
            app: ASGI application
            enabled: Whether rate limiting is enabled
            tracker: Optional custom rate limit tracker

        """
        super().__init__(app)
        self._enabled = enabled
        self._tracker = tracker or get_rate_limit_tracker()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with rate limit checking.

        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain

        Returns:
            Response from next handler or 429 error

        """
        if not self._enabled:
            return await call_next(request)

        path = request.url.path

        # Skip rate limiting for certain paths
        if self._should_skip(path):
            return await call_next(request)

        try:
            # Get provider from request state
            provider = self._get_provider_from_request(request)
            if not provider:
                # No provider context - skip rate limiting
                return await call_next(request)

            # Get client identifier
            client_id = self._get_client_id(request)

            # Check rate limit
            check_result = self._tracker.check_limit(provider, client_id)

            if not check_result.get("allowed", True):
                return self._create_rate_limit_response(
                    check_result,
                    provider,
                )

            # Record the request
            self._tracker.record_request(provider, client_id)

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            remaining = check_result.get("remaining", {})
            self._add_rate_limit_headers(response, remaining)

            return response

        except Exception as e:
            logger.exception(f"Rate limit middleware error: {e}")
            # Continue without rate limiting on error
            return await call_next(request)

    def _should_skip(self, path: str) -> bool:
        """Check if path should skip rate limiting."""
        return any(path.startswith(skip_path) for skip_path in self.SKIP_PATHS)

    def _get_provider_from_request(self, request: Request) -> str | None:
        """Get provider from request state."""
        # Try provider context first
        context = getattr(request.state, "provider_context", None)
        if context:
            return context.provider

        # Fall back to validated_provider
        return getattr(request.state, "validated_provider", None)

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key[:8]}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        if request.client:
            return f"ip:{request.client.host}"

        return "unknown"

    def _add_rate_limit_headers(
        self,
        response: Response,
        remaining: dict[str, Any],
    ) -> None:
        """Add rate limit headers to response."""
        if "requests_remaining_minute" in remaining:
            limit = remaining.get("requests_remaining_minute", 0)
            # Get limit from remaining + used
            response.headers["X-RateLimit-Remaining"] = str(limit)

        if "reset_at_minute" in remaining:
            response.headers["X-RateLimit-Reset"] = str(remaining["reset_at_minute"])

    def _create_rate_limit_response(
        self,
        check_result: dict[str, Any],
        provider: str,
    ) -> JSONResponse:
        """Create 429 rate limit exceeded response."""
        retry_after = check_result.get("retry_after", 60)
        remaining = check_result.get("remaining", {})

        response = JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": check_result.get("reason", "Too many requests"),
                "provider": provider,
                "retry_after": retry_after,
                "remaining": remaining,
            },
        )

        response.headers["Retry-After"] = str(retry_after)
        if "reset_at_minute" in remaining:
            response.headers["X-RateLimit-Reset"] = str(remaining["reset_at_minute"])

        return response


def create_rate_limit_middleware(
    enabled: bool = True,
    tracker: RateLimitTracker | None = None,
) -> type:
    """Factory function to create rate limit middleware with custom configuration.

    Args:
        enabled: Whether rate limiting is enabled
        tracker: Optional custom rate limit tracker

    Returns:
        Configured ProviderRateLimitMiddleware class

    Example:
        from fastapi import FastAPI
        from app.middleware.rate_limit_middleware import (
            create_rate_limit_middleware
        )

        app = FastAPI()
        app.add_middleware(
            create_rate_limit_middleware(enabled=True)
        )

    """

    class ConfiguredRateLimitMiddleware(ProviderRateLimitMiddleware):
        def __init__(self, app) -> None:
            super().__init__(
                app,
                enabled=enabled,
                tracker=tracker,
            )

    return ConfiguredRateLimitMiddleware
