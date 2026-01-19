"""
Centralized Retry Logic with Exponential Backoff

Story 1.2: Direct API Integration
Provides configurable retry mechanisms for LLM provider API calls.

Features:
- Exponential backoff with jitter
- Configurable retry counts and delays
- Provider-specific retry policies
- Rate limit handling with backoff
- Timeout handling
"""

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        is_rate_limit: bool = False,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.is_rate_limit = is_rate_limit


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class BackoffStrategy(Enum):
    """Backoff strategy types."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.25
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL

    # Specific error handling
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True

    # Rate limit specific settings
    respect_retry_after: bool = True
    rate_limit_max_wait: float = 120.0

    # Exception types to retry
    retryable_exceptions: tuple = field(
        default_factory=lambda: (
            TimeoutError,
            ConnectionError,
            asyncio.TimeoutError,
        )
    )


# Provider-specific default configurations
PROVIDER_RETRY_CONFIGS: dict[str, RetryConfig] = {
    "google": RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
    ),
    "gemini": RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
    ),
    "openai": RetryConfig(
        max_retries=5,
        initial_delay=1.0,
        max_delay=60.0,
        backoff_multiplier=2.0,
    ),
    "anthropic": RetryConfig(
        max_retries=4,
        initial_delay=1.5,
        max_delay=45.0,
        backoff_multiplier=2.0,
    ),
    "deepseek": RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
    ),
    "qwen": RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
    ),
    "cursor": RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
    ),
}


def get_provider_retry_config(provider: str) -> RetryConfig:
    """Get retry configuration for a specific provider."""
    return PROVIDER_RETRY_CONFIGS.get(provider.lower(), RetryConfig())


class RetryHandler:
    """
    Centralized retry handler with exponential backoff.

    Provides consistent retry behavior across all LLM provider API calls
    with support for different backoff strategies and rate limit handling.
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self._retry_stats: dict[str, int] = {
            "total_retries": 0,
            "successful_retries": 0,
            "exhausted_retries": 0,
            "rate_limit_retries": 0,
        }

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given attempt using configured strategy.

        Args:
            attempt: The current retry attempt number (0-indexed).

        Returns:
            float: The delay in seconds before the next retry.
        """
        if self.config.strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.backoff_multiplier**attempt)
        elif self.config.strategy == BackoffStrategy.LINEAR:
            delay = self.config.initial_delay * (attempt + 1)
        elif self.config.strategy == BackoffStrategy.FIBONACCI:
            delay = self._fibonacci_delay(attempt)
        else:  # CONSTANT
            delay = self.config.initial_delay

        # Apply max delay cap
        delay = min(delay, self.config.max_delay)

        # Apply jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Ensure minimum delay

        return delay

    def _fibonacci_delay(self, attempt: int) -> float:
        """Calculate Fibonacci-based delay."""
        if attempt <= 0:
            return self.config.initial_delay

        a, b = 1, 1
        for _ in range(attempt):
            a, b = b, a + b

        return self.config.initial_delay * a

    def is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error should trigger a retry.

        Args:
            error: The exception that occurred.

        Returns:
            bool: True if the error is retryable.
        """
        # Check for RetryableError
        if isinstance(error, RetryableError):
            if error.is_rate_limit:
                return self.config.retry_on_rate_limit
            return True

        # Check for timeout errors
        if isinstance(error, TimeoutError | asyncio.TimeoutError):
            return self.config.retry_on_timeout

        # Check for connection errors
        if isinstance(error, ConnectionError):
            return self.config.retry_on_server_error

        # Check for configured retryable exceptions
        if isinstance(error, self.config.retryable_exceptions):
            return True

        # Check for HTTP status code errors
        error_str = str(error).lower()
        if "rate limit" in error_str or "429" in error_str:
            return self.config.retry_on_rate_limit
        if "500" in error_str or "502" in error_str or "503" in error_str:
            return self.config.retry_on_server_error
        if "timeout" in error_str:
            return self.config.retry_on_timeout

        return False

    def get_retry_after(self, error: Exception) -> float | None:
        """
        Extract retry-after value from an error if present.

        Args:
            error: The exception that occurred.

        Returns:
            Optional[float]: The suggested retry delay, or None.
        """
        if isinstance(error, RetryableError) and error.retry_after:
            return min(error.retry_after, self.config.rate_limit_max_wait)

        # Try to extract from error message
        error_str = str(error)
        if "retry-after" in error_str.lower():
            # Try to parse retry-after value
            import re

            match = re.search(r"retry-after[:\s]+(\d+)", error_str, re.IGNORECASE)
            if match:
                return min(float(match.group(1)), self.config.rate_limit_max_wait)

        return None

    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        provider: str | None = None,
        **kwargs,
    ) -> Any:
        """
        Execute an async function with retry logic.

        Args:
            func: The async function to execute.
            *args: Positional arguments to pass to the function.
            provider: Optional provider name for logging.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.

        Raises:
            RetryExhaustedError: When all retry attempts are exhausted.
        """
        last_error: Exception | None = None
        provider_name = provider or "unknown"

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)

                if attempt > 0:
                    self._retry_stats["successful_retries"] += 1
                    logger.info(f"[{provider_name}] Request succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                last_error = e

                if not self.is_retryable_error(e):
                    logger.warning(f"[{provider_name}] Non-retryable error: {e!s}")
                    raise

                if attempt >= self.config.max_retries:
                    self._retry_stats["exhausted_retries"] += 1
                    logger.error(
                        f"[{provider_name}] All {self.config.max_retries} "
                        f"retry attempts exhausted. Last error: {e!s}"
                    )
                    break

                self._retry_stats["total_retries"] += 1

                # Determine delay
                retry_after = self.get_retry_after(e)
                if retry_after and self.config.respect_retry_after:
                    delay = retry_after
                    self._retry_stats["rate_limit_retries"] += 1
                    logger.info(
                        f"[{provider_name}] Rate limited. "
                        f"Waiting {delay:.1f}s (retry-after header)"
                    )
                else:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"[{provider_name}] Attempt {attempt + 1} failed: {e!s}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                await asyncio.sleep(delay)

        raise RetryExhaustedError(
            message=f"All {self.config.max_retries} retry attempts exhausted",
            attempts=self.config.max_retries,
            last_error=last_error,
        )

    def get_stats(self) -> dict[str, int]:
        """Get retry statistics."""
        return self._retry_stats.copy()

    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self._retry_stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "exhausted_retries": 0,
            "rate_limit_retries": 0,
        }


def with_retry(
    config: RetryConfig | None = None,
    provider: str | None = None,
):
    """
    Decorator to add retry logic to async functions.

    Args:
        config: Optional RetryConfig to customize behavior.
        provider: Optional provider name for logging and config lookup.

    Returns:
        Decorated async function with retry logic.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get config from provider defaults if not specified
            retry_config = config
            if retry_config is None and provider:
                retry_config = get_provider_retry_config(provider)

            handler = RetryHandler(retry_config)
            return await handler.execute_with_retry(func, *args, provider=provider, **kwargs)

        return wrapper

    return decorator


# Global retry handler instance for shared usage
_global_retry_handler: RetryHandler | None = None


def get_retry_handler(config: RetryConfig | None = None) -> RetryHandler:
    """Get or create the global retry handler."""
    global _global_retry_handler

    if _global_retry_handler is None or config is not None:
        _global_retry_handler = RetryHandler(config)

    return _global_retry_handler
