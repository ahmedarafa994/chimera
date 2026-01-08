"""
Unified Circuit Breaker Pattern Implementation

HIGH-001 FIX: Consolidated circuit breaker implementation that can be used
across all Chimera components (backend-api, chimera-orchestrator, chimera-agent).

This module provides:
- Thread-safe circuit breaker with async support
- Configurable thresholds and timeouts
- Registry for managing multiple breakers
- Decorator for easy function wrapping
- Metrics and monitoring support

Usage:
    from app.core.shared import circuit_breaker, CircuitState

    # As a decorator
    @circuit_breaker("gemini", failure_threshold=3, recovery_timeout=60)
    async def call_gemini_api(prompt: str):
        ...

    # As a class instance
    breaker = CircuitBreaker(CircuitBreakerConfig(
        name="openai",
        failure_threshold=5,
        recovery_timeout=30
    ))

    if breaker.can_execute():
        try:
            result = await call_api()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, ClassVar, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states following the standard pattern."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker instance."""

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2  # Successes needed in half-open to close
    timeout: float | None = None  # PERF-001 FIX: Operation timeout in seconds

    def __post_init__(self):
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.recovery_timeout < 0:
            raise ValueError("recovery_timeout must be non-negative")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")
        if self.timeout is not None and self.timeout < 0:
            raise ValueError("timeout must be non-negative")


@dataclass
class CircuitBreakerMetrics:
    """Metrics for monitoring circuit breaker behavior."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    last_state_change: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_transitions": self.state_transitions,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "last_state_change": self.last_state_change,
            "success_rate": (
                self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
            ),
        }


class CircuitBreaker:
    """
    Thread-safe circuit breaker implementation with async support.

    The circuit breaker pattern prevents cascading failures by failing fast
    when a service is unhealthy, allowing it time to recover.

    State transitions:
        CLOSED -> OPEN: When failure_threshold is reached
        OPEN -> HALF_OPEN: After recovery_timeout expires
        HALF_OPEN -> CLOSED: After success_threshold successes
        HALF_OPEN -> OPEN: On any failure
    """

    def __init__(self, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker with configuration.

        Args:
            config: CircuitBreakerConfig instance with settings
        """
        self.config = config
        self.name = config.name

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

        # Metrics
        self._metrics = CircuitBreakerMetrics()

        # Thread safety
        self._lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self._metrics

    def _get_async_lock(self) -> asyncio.Lock:
        """Get or create async lock (lazy initialization for thread safety)."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        self._metrics.state_transitions += 1
        self._metrics.last_state_change = time.time()

        logger.info(
            f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}"
        )

    def _check_recovery(self) -> bool:
        """Check if recovery timeout has passed and transition if needed."""
        if self._state != CircuitState.OPEN:
            return True

        if self._metrics.last_failure_time is None:
            return True

        elapsed = time.time() - self._metrics.last_failure_time
        if elapsed >= self.config.recovery_timeout:
            self._transition_to(CircuitState.HALF_OPEN)
            self._success_count = 0
            self._half_open_calls = 0
            return True

        return False

    def can_execute(self) -> bool:
        """
        Check if a call can be executed.

        Returns:
            True if the call should proceed, False if it should be rejected
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._check_recovery():
                    # Transitioned to HALF_OPEN
                    self._half_open_calls += 1
                    return True
                self._metrics.rejected_calls += 1
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                self._metrics.rejected_calls += 1
                return False

            return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Decay failure count on success (sliding window behavior)
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = time.time()
            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition_to(CircuitState.OPEN)
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._metrics = CircuitBreakerMetrics()
            logger.info(f"Circuit breaker '{self.name}' reset to initial state")

    def get_status(self) -> dict[str, Any]:
        """Get current status for monitoring."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "half_open_max_calls": self.config.half_open_max_calls,
                },
                "metrics": self._metrics.to_dict(),
            }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access to circuit breakers across the application,
    with thread-safe creation and retrieval.
    """

    _breakers: ClassVar[dict[str, CircuitBreaker]] = {}
    _configs: ClassVar[dict[str, CircuitBreakerConfig]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(cls, config: CircuitBreakerConfig) -> CircuitBreaker:
        """
        Register a new circuit breaker with configuration.

        Args:
            config: Configuration for the circuit breaker

        Returns:
            The created or existing CircuitBreaker instance
        """
        with cls._lock:
            if config.name in cls._breakers:
                logger.warning(
                    f"Circuit breaker '{config.name}' already registered, "
                    "returning existing instance"
                )
                return cls._breakers[config.name]

            breaker = CircuitBreaker(config)
            cls._breakers[config.name] = breaker
            cls._configs[config.name] = config
            logger.info(f"Registered circuit breaker: {config.name}")
            return breaker

    @classmethod
    def get(
        cls,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker by name.

        Args:
            name: Unique identifier for the circuit breaker
            failure_threshold: Failures before opening (used if creating new)
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max calls in half-open state

        Returns:
            CircuitBreaker instance
        """
        with cls._lock:
            if name not in cls._breakers:
                config = CircuitBreakerConfig(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    half_open_max_calls=half_open_max_calls,
                )
                cls._breakers[name] = CircuitBreaker(config)
                cls._configs[name] = config
            return cls._breakers[name]

    @classmethod
    def reset(cls, name: str) -> None:
        """Reset a circuit breaker to initial state."""
        with cls._lock:
            if name in cls._breakers:
                cls._breakers[name].reset()

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers."""
        with cls._lock:
            for breaker in cls._breakers.values():
                breaker.reset()

    @classmethod
    def get_all_status(cls) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers for monitoring."""
        with cls._lock:
            return {name: breaker.get_status() for name, breaker in cls._breakers.items()}

    @classmethod
    def get_status(cls, name: str) -> dict[str, Any] | None:
        """Get status for a single circuit breaker."""
        with cls._lock:
            breaker = cls._breakers.get(name)
            if not breaker:
                return None
            return breaker.get_status()

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a circuit breaker from the registry."""
        with cls._lock:
            if name in cls._breakers:
                del cls._breakers[name]
                cls._configs.pop(name, None)
                return True
            return False


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open and rejecting calls."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker '{name}' is open. Retry after {retry_after:.1f}s")


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_max_calls: int = 3,
    timeout: float | None = None,  # PERF-001 FIX: Add timeout parameter
    exceptions: tuple = (Exception,),
):
    """
    Circuit breaker decorator for async functions.

    PERF-001 FIX: Added timeout support to prevent hanging requests.

    Args:
        name: Unique identifier for this circuit (e.g., "gemini", "openai")
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        half_open_max_calls: Max concurrent calls in half-open state
        timeout: Operation timeout in seconds (None = no timeout, use TimeoutConfig)
        exceptions: Tuple of exception types that count as failures

    Example:
        from app.core.timeouts import TimeoutConfig

        @circuit_breaker("gemini", timeout=TimeoutConfig.get_timeout("llm"))
        async def call_gemini(prompt):
            return await gemini_client.generate(prompt)

    Raises:
        CircuitBreakerOpen: When circuit is open and call is rejected
        asyncio.TimeoutError: When operation times out
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            breaker = CircuitBreakerRegistry.get(
                name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
            )

            if not breaker.can_execute():
                # Calculate retry_after
                if breaker.metrics.last_failure_time:
                    elapsed = time.time() - breaker.metrics.last_failure_time
                    retry_after = max(0, recovery_timeout - elapsed)
                else:
                    retry_after = recovery_timeout
                raise CircuitBreakerOpen(name, retry_after)

            try:
                # PERF-001 FIX: Apply timeout if configured
                if timeout is not None:
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except TimeoutError:
                # PERF-001 FIX: Count timeouts as circuit breaker failures
                breaker.record_failure()
                logger.warning(
                    f"Circuit breaker '{name}': operation timed out after {timeout}s"
                )
                raise
            except exceptions:
                breaker.record_failure()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            breaker = CircuitBreakerRegistry.get(
                name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
            )

            if not breaker.can_execute():
                if breaker.metrics.last_failure_time:
                    elapsed = time.time() - breaker.metrics.last_failure_time
                    retry_after = max(0, recovery_timeout - elapsed)
                else:
                    retry_after = recovery_timeout
                raise CircuitBreakerOpen(name, retry_after)

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except exceptions:
                breaker.record_failure()
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def is_circuit_open(name: str) -> bool:
    """
    Check if a circuit breaker is currently open.

    Convenience function for use with retry libraries like tenacity.

    Args:
        name: Name of the circuit breaker to check

    Returns:
        True if circuit is open, False otherwise
    """
    breaker = CircuitBreakerRegistry.get(name)
    return breaker.state == CircuitState.OPEN
