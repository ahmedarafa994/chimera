"""Per-Model Rate Limiting Service.

Provides rate limiting on a per-provider and per-model basis with
support for different user tiers and graceful degradation.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a tier/provider/model combination."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 100000
    tokens_per_day: int = 1000000
    concurrent_requests: int = 5


@dataclass
class RateLimitWindow:
    """Sliding window for rate limiting."""

    window_start: datetime
    request_count: int = 0
    token_count: int = 0


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining_requests: int
    remaining_tokens: int
    reset_at: datetime
    retry_after_seconds: int | None = None
    limit_type: str | None = None  # 'requests', 'tokens', 'concurrent'


# Default rate limits by tier
TIER_LIMITS: dict[str, dict[str, RateLimitConfig]] = {
    "free": {
        "default": RateLimitConfig(
            requests_per_minute=20,
            requests_per_hour=200,
            tokens_per_minute=50000,
            tokens_per_day=500000,
            concurrent_requests=2,
        ),
        "gemini": RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=300,
            tokens_per_minute=60000,
            tokens_per_day=600000,
            concurrent_requests=3,
        ),
        "deepseek": RateLimitConfig(
            requests_per_minute=20,
            requests_per_hour=200,
            tokens_per_minute=40000,
            tokens_per_day=400000,
            concurrent_requests=2,
        ),
    },
    "pro": {
        "default": RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            tokens_per_minute=150000,
            tokens_per_day=2000000,
            concurrent_requests=5,
        ),
        "gemini": RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=2000,
            tokens_per_minute=200000,
            tokens_per_day=3000000,
            concurrent_requests=10,
        ),
        "deepseek": RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            tokens_per_minute=100000,
            tokens_per_day=1500000,
            concurrent_requests=5,
        ),
    },
    "enterprise": {
        "default": RateLimitConfig(
            requests_per_minute=200,
            requests_per_hour=5000,
            tokens_per_minute=500000,
            tokens_per_day=10000000,
            concurrent_requests=20,
        ),
        "gemini": RateLimitConfig(
            requests_per_minute=300,
            requests_per_hour=10000,
            tokens_per_minute=1000000,
            tokens_per_day=20000000,
            concurrent_requests=50,
        ),
        "deepseek": RateLimitConfig(
            requests_per_minute=200,
            requests_per_hour=5000,
            tokens_per_minute=500000,
            tokens_per_day=10000000,
            concurrent_requests=20,
        ),
    },
}


class ModelRateLimiter:
    """Per-model rate limiter with sliding window algorithm.

    Features:
    - Per-provider and per-model rate limits
    - User tier-based limits (free, pro, enterprise)
    - Request count and token count limits
    - Concurrent request limiting
    - Graceful degradation with fallback suggestions
    """

    def __init__(self) -> None:
        # In-memory storage (use Redis in production)
        # Structure: {user_id: {provider: {model: {window_type: RateLimitWindow}}}}
        self._windows: dict[str, dict[str, dict[str, dict[str, RateLimitWindow]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict)),
        )

        # Concurrent request tracking: {user_id: {provider: count}}
        self._concurrent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def get_limit_config(
        self,
        tier: str,
        provider: str,
        model: str | None = None,
    ) -> RateLimitConfig:
        """Get rate limit configuration for a tier/provider combination."""
        tier_config = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

        # Try provider-specific config, fall back to default
        return tier_config.get(provider, tier_config["default"])

    async def check_rate_limit(
        self,
        user_id: str,
        provider: str,
        model: str,
        tier: str = "free",
        estimated_tokens: int = 0,
    ) -> RateLimitResult:
        """Check if a request is allowed under rate limits.

        Args:
            user_id: User identifier
            provider: Provider name (gemini, deepseek)
            model: Model name
            tier: User tier (free, pro, enterprise)
            estimated_tokens: Estimated tokens for this request

        Returns:
            RateLimitResult with allowed status and limit info

        """
        async with self._lock:
            config = self.get_limit_config(tier, provider, model)
            now = datetime.utcnow()

            # Check concurrent requests
            concurrent = self._concurrent[user_id][provider]
            if concurrent >= config.concurrent_requests:
                return RateLimitResult(
                    allowed=False,
                    remaining_requests=0,
                    remaining_tokens=0,
                    reset_at=now + timedelta(seconds=30),
                    retry_after_seconds=30,
                    limit_type="concurrent",
                )

            # Get or create windows
            user_windows = self._windows[user_id][provider][model]

            # Check minute window
            minute_window = self._get_or_create_window(
                user_windows,
                "minute",
                now,
                timedelta(minutes=1),
            )

            if minute_window.request_count >= config.requests_per_minute:
                reset_at = minute_window.window_start + timedelta(minutes=1)
                return RateLimitResult(
                    allowed=False,
                    remaining_requests=0,
                    remaining_tokens=max(0, config.tokens_per_minute - minute_window.token_count),
                    reset_at=reset_at,
                    retry_after_seconds=int((reset_at - now).total_seconds()),
                    limit_type="requests_per_minute",
                )

            # Check token limit
            if minute_window.token_count + estimated_tokens > config.tokens_per_minute:
                reset_at = minute_window.window_start + timedelta(minutes=1)
                return RateLimitResult(
                    allowed=False,
                    remaining_requests=config.requests_per_minute - minute_window.request_count,
                    remaining_tokens=0,
                    reset_at=reset_at,
                    retry_after_seconds=int((reset_at - now).total_seconds()),
                    limit_type="tokens_per_minute",
                )

            # Check hourly window
            hour_window = self._get_or_create_window(user_windows, "hour", now, timedelta(hours=1))

            if hour_window.request_count >= config.requests_per_hour:
                reset_at = hour_window.window_start + timedelta(hours=1)
                return RateLimitResult(
                    allowed=False,
                    remaining_requests=0,
                    remaining_tokens=max(0, config.tokens_per_minute - minute_window.token_count),
                    reset_at=reset_at,
                    retry_after_seconds=int((reset_at - now).total_seconds()),
                    limit_type="requests_per_hour",
                )

            # All checks passed
            return RateLimitResult(
                allowed=True,
                remaining_requests=config.requests_per_minute - minute_window.request_count - 1,
                remaining_tokens=config.tokens_per_minute
                - minute_window.token_count
                - estimated_tokens,
                reset_at=minute_window.window_start + timedelta(minutes=1),
            )

    async def record_request(
        self,
        user_id: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
    ) -> None:
        """Record a completed request for rate limiting."""
        async with self._lock:
            now = datetime.utcnow()
            user_windows = self._windows[user_id][provider][model]

            # Update minute window
            minute_window = self._get_or_create_window(
                user_windows,
                "minute",
                now,
                timedelta(minutes=1),
            )
            minute_window.request_count += 1
            minute_window.token_count += tokens_used

            # Update hour window
            hour_window = self._get_or_create_window(user_windows, "hour", now, timedelta(hours=1))
            hour_window.request_count += 1
            hour_window.token_count += tokens_used

            # Update day window
            day_window = self._get_or_create_window(user_windows, "day", now, timedelta(days=1))
            day_window.request_count += 1
            day_window.token_count += tokens_used

    async def acquire_concurrent_slot(
        self,
        user_id: str,
        provider: str,
    ) -> bool:
        """Acquire a concurrent request slot."""
        async with self._lock:
            # For simplicity, we don't check limits here - check_rate_limit does that
            self._concurrent[user_id][provider] += 1
            return True

    async def release_concurrent_slot(
        self,
        user_id: str,
        provider: str,
    ) -> None:
        """Release a concurrent request slot."""
        async with self._lock:
            if self._concurrent[user_id][provider] > 0:
                self._concurrent[user_id][provider] -= 1

    def _get_or_create_window(
        self,
        windows: dict[str, RateLimitWindow],
        window_type: str,
        now: datetime,
        window_size: timedelta,
    ) -> RateLimitWindow:
        """Get existing window or create new one if expired."""
        window = windows.get(window_type)

        if window is None or now - window.window_start >= window_size:
            # Create new window
            window = RateLimitWindow(window_start=now)
            windows[window_type] = window

        return window

    def get_rate_limit_headers(self, result: RateLimitResult, provider: str) -> dict[str, str]:
        """Generate rate limit headers for HTTP response."""
        headers = {
            "X-RateLimit-Remaining-Requests": str(result.remaining_requests),
            "X-RateLimit-Remaining-Tokens": str(result.remaining_tokens),
            "X-RateLimit-Reset": result.reset_at.isoformat(),
            "X-RateLimit-Provider": provider,
        }

        if not result.allowed:
            headers["X-RateLimit-Limit-Type"] = result.limit_type or "unknown"
            if result.retry_after_seconds:
                headers["Retry-After"] = str(result.retry_after_seconds)

        return headers

    async def get_user_usage_stats(
        self,
        user_id: str,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Get usage statistics for a user."""
        async with self._lock:
            stats = {
                "user_id": user_id,
                "providers": {},
            }

            user_data = self._windows.get(user_id, {})

            for prov, models in user_data.items():
                if provider and prov != provider:
                    continue

                provider_stats = {
                    "models": {},
                    "concurrent_requests": self._concurrent[user_id].get(prov, 0),
                }

                for model, windows in models.items():
                    model_stats = {}
                    for window_type, window in windows.items():
                        model_stats[window_type] = {
                            "request_count": window.request_count,
                            "token_count": window.token_count,
                            "window_start": window.window_start.isoformat(),
                        }
                    provider_stats["models"][model] = model_stats

                stats["providers"][prov] = provider_stats

            return stats

    def suggest_fallback_provider(
        self,
        current_provider: str,
        user_id: str,
        tier: str = "free",
    ) -> str | None:
        """Suggest a fallback provider when rate limited."""
        fallback_order = {
            "gemini": ["deepseek"],
            "deepseek": ["gemini"],
        }

        fallbacks = fallback_order.get(current_provider, [])

        for fallback in fallbacks:
            # Check if fallback has capacity (simplified check)
            config = self.get_limit_config(tier, fallback)
            concurrent = self._concurrent[user_id].get(fallback, 0)

            if concurrent < config.concurrent_requests:
                return fallback

        return None


# Global singleton instance
model_rate_limiter = ModelRateLimiter()


def get_model_rate_limiter() -> ModelRateLimiter:
    """Get the model rate limiter singleton."""
    return model_rate_limiter
