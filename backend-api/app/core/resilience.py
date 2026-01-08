"""
Resilience utilities for LLM provider calls.

Provides:
- Retry with exponential backoff
- Rate limit tracking and header parsing
- Timeout handling
"""

import asyncio
import contextlib
import functools
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, ParamSpec, TypeVar

from app.core.logging import logger

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (
        Exception,
    )  # Customize per use case
    retryable_status_codes: frozenset[int] = frozenset({429, 500, 502, 503, 504})


@dataclass
class RateLimitInfo:
    """Parsed rate limit information from provider headers."""

    limit: int | None = None
    remaining: int | None = None
    reset_at: datetime | None = None
    retry_after: float | None = None


@dataclass
class RateLimitTracker:
    """Tracks rate limit state for a provider."""

    provider: str
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    current_requests: int = 0
    current_tokens: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    last_limit_info: RateLimitInfo | None = None

    def reset_if_window_expired(self) -> None:
        """Reset counters if the minute window has passed."""
        now = datetime.now()
        if now - self.window_start >= timedelta(minutes=1):
            self.current_requests = 0
            self.current_tokens = 0
            self.window_start = now

    def can_make_request(self, estimated_tokens: int = 0) -> bool:
        """Check if a request can be made within rate limits."""
        self.reset_if_window_expired()

        if self.current_requests >= self.requests_per_minute:
            return False
        if estimated_tokens > 0:
            if self.current_tokens + estimated_tokens > self.tokens_per_minute:
                return False
        return True

    def record_request(self, tokens_used: int = 0) -> None:
        """Record a completed request."""
        self.reset_if_window_expired()
        self.current_requests += 1
        self.current_tokens += tokens_used

    def wait_time(self) -> float:
        """Calculate seconds to wait before next request."""
        if self.last_limit_info and self.last_limit_info.retry_after:
            return self.last_limit_info.retry_after

        self.reset_if_window_expired()

        if self.current_requests >= self.requests_per_minute:
            elapsed = (datetime.now() - self.window_start).total_seconds()
            return max(0, 60 - elapsed)

        return 0


class RateLimitRegistry:
    """Registry for provider rate limit trackers."""

    _trackers: dict[str, RateLimitTracker] = {}

    @classmethod
    def get_tracker(cls, provider: str) -> RateLimitTracker:
        """Get or create a rate limit tracker for a provider."""
        if provider not in cls._trackers:
            cls._trackers[provider] = RateLimitTracker(provider=provider)
        return cls._trackers[provider]

    @classmethod
    def update_from_headers(cls, provider: str, headers: dict[str, str]) -> None:
        """Update rate limit info from response headers."""
        tracker = cls.get_tracker(provider)
        tracker.last_limit_info = parse_rate_limit_headers(headers)


def parse_rate_limit_headers(headers: dict[str, str]) -> RateLimitInfo:
    """
    Parse rate limit headers from provider responses.

    Supports common header formats:
    - X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
    - RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset
    - Retry-After
    """
    info = RateLimitInfo()

    # Try common header variations
    for prefix in ["X-RateLimit-", "RateLimit-", "x-ratelimit-", "ratelimit-"]:
        if f"{prefix}Limit" in headers or f"{prefix}limit" in headers:
            limit_key = f"{prefix}Limit" if f"{prefix}Limit" in headers else f"{prefix}limit"
            with contextlib.suppress(ValueError, TypeError):
                info.limit = int(headers[limit_key])

        if f"{prefix}Remaining" in headers or f"{prefix}remaining" in headers:
            rem_key = (
                f"{prefix}Remaining" if f"{prefix}Remaining" in headers else f"{prefix}remaining"
            )
            with contextlib.suppress(ValueError, TypeError):
                info.remaining = int(headers[rem_key])

        if f"{prefix}Reset" in headers or f"{prefix}reset" in headers:
            reset_key = f"{prefix}Reset" if f"{prefix}Reset" in headers else f"{prefix}reset"
            try:
                # Could be timestamp or seconds
                reset_val = headers[reset_key]
                if len(reset_val) > 10:  # Likely timestamp
                    info.reset_at = datetime.fromtimestamp(float(reset_val))
                else:
                    info.reset_at = datetime.now() + timedelta(seconds=float(reset_val))
            except (ValueError, TypeError):
                pass

    # Retry-After header
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        with contextlib.suppress(ValueError, TypeError):
            info.retry_after = float(retry_after)

    return info


def calculate_backoff(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate backoff delay with exponential growth and optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    delay = config.base_delay * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter up to 25% of delay
        jitter = delay * 0.25 * random.random()
        delay += jitter

    return delay


def retry_with_backoff(
    config: RetryConfig | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async functions with exponential backoff retry.

    Usage:
        @retry_with_backoff(RetryConfig(max_attempts=3))
        async def call_api():
            ...

    Args:
        config: Retry configuration (uses defaults if None)

    Returns:
        Decorated async function with retry behavior
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    # Check if this is the last attempt
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise

                    # Calculate backoff delay
                    delay = calculate_backoff(attempt, config)

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)

            # Should not reach here, but raise if we do
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


async def with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    operation_name: str = "operation",
) -> T:
    """
    Execute a coroutine with a timeout.

    Args:
        coro: The coroutine to execute
        timeout_seconds: Maximum time to wait
        operation_name: Name for logging purposes

    Returns:
        Result of the coroutine

    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except TimeoutError:
        logger.error(f"{operation_name} timed out after {timeout_seconds}s")
        raise


class ProviderRateLimiter:
    """
    Rate limiter that respects provider-specific limits.

    Usage:
        limiter = ProviderRateLimiter("openai", rpm=60, tpm=100000)
        async with limiter.acquire(estimated_tokens=1000):
            response = await call_api()
            limiter.record_tokens(response.usage.total_tokens)
    """

    def __init__(
        self,
        provider: str,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ):
        self.tracker = RateLimitTracker(
            provider=provider,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        )
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0) -> "ProviderRateLimiter":
        """Acquire permission to make a request, waiting if necessary."""
        async with self._lock:
            while not self.tracker.can_make_request(estimated_tokens):
                wait = self.tracker.wait_time()
                if wait > 0:
                    logger.info(f"Rate limited for {self.tracker.provider}, waiting {wait:.2f}s")
                    await asyncio.sleep(wait)
                self.tracker.reset_if_window_expired()
        return self

    def record_tokens(self, tokens: int) -> None:
        """Record tokens used after a request."""
        self.tracker.record_request(tokens)

    async def __aenter__(self) -> "ProviderRateLimiter":
        """Context manager entry - acquire permission."""
        return await self.acquire()

    async def __aexit__(self, *args: Any) -> None:
        """Context manager exit."""
        pass
