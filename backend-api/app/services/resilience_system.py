"""
Advanced Circuit Breakers and Bulkheads for System Resilience.

This module provides comprehensive resilience patterns:
- Advanced Circuit Breaker with adaptive thresholds
- Bulkhead isolation for resource protection
- Rate limiting and throttling
- Retry mechanisms with backoff strategies
- Timeout management and deadline propagation
- Graceful degradation and fallback handling
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.logging import logger

# =====================================================
# Core Resilience Patterns and Models
# =====================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategy types."""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI = "fibonacci"


class ThrottleStrategy(Enum):
    """Throttling strategy types."""

    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Basic configuration
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    success_threshold: int = 3  # Half-open to closed transition

    # Adaptive configuration
    adaptive_threshold: bool = True
    min_request_threshold: int = 20
    error_rate_threshold: float = 0.5  # 50% error rate

    # Timing configuration
    request_timeout_seconds: float = 30.0
    slow_call_threshold_seconds: float = 10.0
    slow_call_rate_threshold: float = 0.5

    # Monitoring
    metrics_window_seconds: float = 300.0  # 5 minutes
    enable_metrics: bool = True


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""

    # Thread pool configuration
    max_concurrent_calls: int = 100
    queue_capacity: int = 200
    keep_alive_seconds: float = 60.0

    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: float = 80.0

    # Monitoring
    enable_monitoring: bool = True
    monitor_interval_seconds: float = 30.0


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""

    # Basic retry settings
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0

    # Strategy configuration
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    # Conditional retry
    retryable_exceptions: set[str] = field(
        default_factory=lambda: {"TimeoutError", "ConnectionError", "HTTPError"}
    )
    non_retryable_exceptions: set[str] = field(
        default_factory=lambda: {"AuthenticationError", "AuthorizationError", "ValidationError"}
    )


@dataclass
class ThrottleConfig:
    """Configuration for rate limiting and throttling."""

    # Rate limiting
    requests_per_second: float = 100.0
    burst_capacity: int = 200

    # Strategy
    strategy: ThrottleStrategy = ThrottleStrategy.TOKEN_BUCKET

    # Timing
    window_size_seconds: float = 60.0
    refill_interval_seconds: float = 1.0


# =====================================================
# Advanced Circuit Breaker Implementation
# =====================================================


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, name: str, state: CircuitState, retry_after: float):
        self.name = name
        self.state = state
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker '{name}' is {state.value}, retry after {retry_after}s")


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker monitoring."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    rejected_requests: int = 0

    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    error_rate: float = 0.0

    state_transitions: dict[str, int] = field(
        default_factory=lambda: {
            "closed_to_open": 0,
            "open_to_half_open": 0,
            "half_open_to_closed": 0,
            "half_open_to_open": 0,
        }
    )

    last_failure_time: float = 0.0
    last_success_time: float = 0.0


class AdvancedCircuitBreaker:
    """
    Advanced circuit breaker with adaptive thresholds and comprehensive metrics.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config

        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.consecutive_failures = 0
        self.consecutive_successes = 0

        # Metrics tracking
        self.metrics = CircuitMetrics()
        self._response_times: deque = deque(maxlen=1000)
        self._request_history: deque = deque(maxlen=self.config.min_request_threshold * 2)

        # Locks for thread safety
        self._state_lock = asyncio.Lock()
        self._metrics_lock = asyncio.Lock()

        # Active requests tracking
        self._active_requests: set[str] = set()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function call through circuit breaker."""
        request_id = f"{time.time()}_{id(func)}"

        # Check if circuit allows request
        async with self._state_lock:
            if not await self._can_proceed():
                self.metrics.rejected_requests += 1
                raise CircuitBreakerException(self.name, self.state, self._get_retry_after())

        # Track active request
        self._active_requests.add(request_id)
        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.config.request_timeout_seconds,
            )

            # Record success
            response_time_ms = (time.time() - start_time) * 1000
            await self._record_success(response_time_ms)

            return result

        except TimeoutError:
            # Record timeout
            response_time_ms = (time.time() - start_time) * 1000
            await self._record_timeout(response_time_ms)
            raise

        except Exception as e:
            # Record failure
            response_time_ms = (time.time() - start_time) * 1000
            await self._record_failure(response_time_ms, e)
            raise

        finally:
            # Cleanup
            self._active_requests.discard(request_id)

    async def _can_proceed(self) -> bool:
        """Check if circuit breaker allows request to proceed."""
        current_time = time.time()

        if self.state == CircuitState.CLOSED:
            return True

        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.config.recovery_timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.consecutive_successes = 0
                self.metrics.state_transitions["open_to_half_open"] += 1
                logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
                return True
            return False

        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests to test recovery
            return len(self._active_requests) == 0

        return False

    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the actual function call."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    async def _record_success(self, response_time_ms: float) -> None:
        """Record successful request."""
        async with self._metrics_lock:
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = time.time()

            self._response_times.append(response_time_ms)
            self._request_history.append(("success", time.time(), response_time_ms))

            # Update response time metrics
            self._update_response_time_metrics()

            # Handle state transitions
            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to_closed()
            else:
                self.consecutive_failures = 0

    async def _record_failure(self, response_time_ms: float, exception: Exception) -> None:
        """Record failed request."""
        async with self._metrics_lock:
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = time.time()
            self.last_failure_time = time.time()

            self._response_times.append(response_time_ms)
            self._request_history.append(("failure", time.time(), response_time_ms))

            # Update response time and error rate metrics
            self._update_response_time_metrics()
            self._update_error_rate()

            # Handle state transitions
            if self.state == CircuitState.CLOSED:
                self.consecutive_failures += 1
                if await self._should_trip():
                    await self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()

    async def _record_timeout(self, response_time_ms: float) -> None:
        """Record timeout request."""
        async with self._metrics_lock:
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.timeout_requests += 1
            self.metrics.failed_requests += 1  # Timeouts are failures

            self._response_times.append(response_time_ms)
            self._request_history.append(("timeout", time.time(), response_time_ms))

            # Update metrics
            self._update_response_time_metrics()
            self._update_error_rate()

            # Handle state transitions (same as failure)
            if self.state == CircuitState.CLOSED:
                self.consecutive_failures += 1
                if await self._should_trip():
                    await self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()

    async def _should_trip(self) -> bool:
        """Determine if circuit breaker should trip to OPEN state."""
        # Check consecutive failures threshold
        if self.consecutive_failures >= self.config.failure_threshold:
            return True

        # Adaptive threshold based on error rate
        if (
            self.config.adaptive_threshold
            and len(self._request_history) >= self.config.min_request_threshold
        ) and self.metrics.error_rate >= self.config.error_rate_threshold:
            return True

        # Check slow call rate
        return self._get_slow_call_rate() >= self.config.slow_call_rate_threshold

    def _get_slow_call_rate(self) -> float:
        """Calculate the rate of slow calls."""
        if not self._response_times:
            return 0.0

        slow_calls = sum(
            1 for rt in self._response_times if rt >= self.config.slow_call_threshold_seconds * 1000
        )
        return slow_calls / len(self._response_times)

    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        async with self._state_lock:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()

            if old_state == CircuitState.CLOSED:
                self.metrics.state_transitions["closed_to_open"] += 1
            elif old_state == CircuitState.HALF_OPEN:
                self.metrics.state_transitions["half_open_to_open"] += 1

        logger.warning(f"Circuit breaker '{self.name}' tripped to OPEN state")

    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        async with self._state_lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.metrics.state_transitions["half_open_to_closed"] += 1

        logger.info(f"Circuit breaker '{self.name}' recovered to CLOSED state")

    def _update_response_time_metrics(self) -> None:
        """Update response time metrics."""
        if not self._response_times:
            return

        response_times = list(self._response_times)
        self.metrics.avg_response_time_ms = sum(response_times) / len(response_times)

        # Calculate P95
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        self.metrics.p95_response_time_ms = sorted_times[p95_index] if sorted_times else 0.0

    def _update_error_rate(self) -> None:
        """Update error rate metrics."""
        if self.metrics.total_requests == 0:
            self.metrics.error_rate = 0.0
        else:
            self.metrics.error_rate = (
                self.metrics.failed_requests + self.metrics.timeout_requests
            ) / self.metrics.total_requests

    def _get_retry_after(self) -> float:
        """Get retry after time for rejected requests."""
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            return max(0, self.config.recovery_timeout_seconds - elapsed)
        return 0.0

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "active_requests": len(self._active_requests),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "timeout_requests": self.metrics.timeout_requests,
                "rejected_requests": self.metrics.rejected_requests,
                "avg_response_time_ms": self.metrics.avg_response_time_ms,
                "p95_response_time_ms": self.metrics.p95_response_time_ms,
                "error_rate": self.metrics.error_rate,
                "slow_call_rate": self._get_slow_call_rate(),
            },
            "state_transitions": self.metrics.state_transitions,
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time,
        }


# =====================================================
# Bulkhead Implementation for Resource Isolation
# =====================================================


@dataclass
class BulkheadMetrics:
    """Metrics for bulkhead monitoring."""

    active_requests: int = 0
    queued_requests: int = 0
    completed_requests: int = 0
    rejected_requests: int = 0
    timeout_requests: int = 0

    avg_processing_time_ms: float = 0.0
    queue_wait_time_ms: float = 0.0
    resource_utilization: float = 0.0


class BulkheadIsolation:
    """
    Bulkhead pattern implementation for resource isolation.
    """

    def __init__(self, name: str, config: BulkheadConfig):
        self.name = name
        self.config = config

        # Request management
        self._semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_capacity)

        # Metrics
        self.metrics = BulkheadMetrics()
        self._processing_times: deque = deque(maxlen=1000)

        # Resource monitoring
        self._memory_usage_mb = 0.0
        self._cpu_usage_percent = 0.0

        # Background tasks
        self._monitor_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start bulkhead monitoring."""
        if self.config.enable_monitoring:
            self._is_running = True
            self._monitor_task = asyncio.create_task(self._monitor_resources())

    async def stop(self) -> None:
        """Stop bulkhead monitoring."""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead isolation."""
        # Check resource limits before queuing
        if not await self._check_resource_limits():
            self.metrics.rejected_requests += 1
            raise RuntimeError(f"Bulkhead '{self.name}' resource limits exceeded")

        # Try to add to queue
        try:
            request_data = (func, args, kwargs, time.time())
            self._request_queue.put_nowait(request_data)
            self.metrics.queued_requests += 1
        except asyncio.QueueFull:
            self.metrics.rejected_requests += 1
            raise RuntimeError(f"Bulkhead '{self.name}' queue is full")

        # Process request
        return await self._process_request()

    async def _process_request(self) -> Any:
        """Process queued request with resource isolation."""
        # Get request from queue
        func, args, kwargs, queue_time = await self._request_queue.get()
        (time.time() - queue_time) * 1000

        self.metrics.queued_requests -= 1

        # Acquire semaphore for concurrent execution limit
        async with self._semaphore:
            self.metrics.active_requests += 1
            start_time = time.time()

            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, *args, **kwargs)

                # Record success metrics
                processing_time_ms = (time.time() - start_time) * 1000
                self._processing_times.append(processing_time_ms)
                self.metrics.completed_requests += 1

                # Update average processing time
                if self._processing_times:
                    self.metrics.avg_processing_time_ms = sum(self._processing_times) / len(
                        self._processing_times
                    )

                return result

            except TimeoutError:
                self.metrics.timeout_requests += 1
                raise
            except Exception:
                # Other exceptions are propagated but counted
                raise
            finally:
                self.metrics.active_requests -= 1

    async def _check_resource_limits(self) -> bool:
        """Check if resource limits allow new requests."""
        # Check memory limit
        if self._memory_usage_mb > self.config.max_memory_mb:
            logger.warning(
                f"Bulkhead '{self.name}' memory limit exceeded: {self._memory_usage_mb}MB"
            )
            return False

        # Check CPU limit
        if self._cpu_usage_percent > self.config.max_cpu_percent:
            logger.warning(f"Bulkhead '{self.name}' CPU limit exceeded: {self._cpu_usage_percent}%")
            return False

        return True

    async def _monitor_resources(self) -> None:
        """Background resource monitoring."""
        import psutil

        while self._is_running:
            try:
                # Monitor system resources
                process = psutil.Process()
                self._memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self._cpu_usage_percent = process.cpu_percent()

                # Calculate resource utilization
                memory_utilization = self._memory_usage_mb / self.config.max_memory_mb
                cpu_utilization = self._cpu_usage_percent / self.config.max_cpu_percent
                self.metrics.resource_utilization = max(memory_utilization, cpu_utilization)

                await asyncio.sleep(self.config.monitor_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bulkhead resource monitoring error: {e}")
                await asyncio.sleep(self.config.monitor_interval_seconds)

    def get_metrics(self) -> dict[str, Any]:
        """Get bulkhead metrics."""
        return {
            "name": self.name,
            "active_requests": self.metrics.active_requests,
            "queued_requests": self.metrics.queued_requests,
            "completed_requests": self.metrics.completed_requests,
            "rejected_requests": self.metrics.rejected_requests,
            "timeout_requests": self.metrics.timeout_requests,
            "avg_processing_time_ms": self.metrics.avg_processing_time_ms,
            "queue_utilization": self._request_queue.qsize() / self.config.queue_capacity,
            "semaphore_utilization": (self.config.max_concurrent_calls - self._semaphore._value)
            / self.config.max_concurrent_calls,
            "resource_utilization": self.metrics.resource_utilization,
            "memory_usage_mb": self._memory_usage_mb,
            "cpu_usage_percent": self._cpu_usage_percent,
        }


# =====================================================
# Rate Limiting and Throttling
# =====================================================


class TokenBucketThrottle:
    """Token bucket rate limiting implementation."""

    def __init__(self, config: ThrottleConfig):
        self.config = config

        # Token bucket state
        self._tokens = config.burst_capacity
        self._last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens from bucket."""
        async with self._lock:
            # Refill tokens based on time elapsed
            now = time.time()
            time_elapsed = now - self._last_refill
            tokens_to_add = time_elapsed * self.config.requests_per_second

            self._tokens = min(self.config.burst_capacity, self._tokens + tokens_to_add)
            self._last_refill = now

            # Check if enough tokens available
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for tokens."""
        if self._tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self._tokens
        return tokens_needed / self.config.requests_per_second


# =====================================================
# Retry Mechanism with Backoff Strategies
# =====================================================


class RetryHandler:
    """Advanced retry handler with multiple backoff strategies."""

    def __init__(self, config: RetryConfig):
        self.config = config

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, *args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.debug(f"Non-retryable exception: {type(e).__name__}")
                    raise

                # Don't retry on last attempt
                if attempt == self.config.max_retries:
                    break

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.debug(f"Retry attempt {attempt + 1} after {delay:.2f}s delay")
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"All {self.config.max_retries} retries exhausted")
        raise last_exception

    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        exception_name = type(exception).__name__

        # Check non-retryable exceptions first
        if exception_name in self.config.non_retryable_exceptions:
            return False

        # Check explicitly retryable exceptions
        if exception_name in self.config.retryable_exceptions:
            return True

        # Default behavior for common retryable exceptions
        retryable_types = {
            "TimeoutError",
            "ConnectionError",
            "HTTPError",
            "ServiceUnavailableError",
            "TooManyRequestsError",
        }

        return exception_name in retryable_types

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.initial_delay_seconds

        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay_seconds * (self.config.backoff_multiplier**attempt)

        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay_seconds * (attempt + 1)

        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.initial_delay_seconds * self._fibonacci(attempt + 1)

        else:
            delay = self.config.initial_delay_seconds

        # Apply jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_factor
            import random

            delay += random.uniform(-jitter_amount, jitter_amount)

        # Ensure delay doesn't exceed maximum
        return min(delay, self.config.max_delay_seconds)

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


# =====================================================
# Integrated Resilience Manager
# =====================================================


class ResilienceManager:
    """
    Integrated resilience manager combining all patterns.
    """

    def __init__(self):
        # Component registries
        self._circuit_breakers: dict[str, AdvancedCircuitBreaker] = {}
        self._bulkheads: dict[str, BulkheadIsolation] = {}
        self._throttles: dict[str, TokenBucketThrottle] = {}
        self._retry_handlers: dict[str, RetryHandler] = {}

        # Global metrics
        self._global_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_trips": 0,
            "bulkhead_rejections": 0,
            "throttle_rejections": 0,
        }

    def register_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig
    ) -> AdvancedCircuitBreaker:
        """Register a circuit breaker."""
        circuit_breaker = AdvancedCircuitBreaker(name, config)
        self._circuit_breakers[name] = circuit_breaker
        logger.info(f"Registered circuit breaker: {name}")
        return circuit_breaker

    def register_bulkhead(self, name: str, config: BulkheadConfig) -> BulkheadIsolation:
        """Register a bulkhead."""
        bulkhead = BulkheadIsolation(name, config)
        self._bulkheads[name] = bulkhead
        logger.info(f"Registered bulkhead: {name}")
        return bulkhead

    def register_throttle(self, name: str, config: ThrottleConfig) -> TokenBucketThrottle:
        """Register a throttle."""
        throttle = TokenBucketThrottle(config)
        self._throttles[name] = throttle
        logger.info(f"Registered throttle: {name}")
        return throttle

    def register_retry_handler(self, name: str, config: RetryConfig) -> RetryHandler:
        """Register a retry handler."""
        retry_handler = RetryHandler(config)
        self._retry_handlers[name] = retry_handler
        logger.info(f"Registered retry handler: {name}")
        return retry_handler

    async def execute_with_resilience(
        self,
        func: Callable,
        circuit_breaker: str | None = None,
        bulkhead: str | None = None,
        throttle: str | None = None,
        retry_handler: str | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with specified resilience patterns."""
        self._global_metrics["total_requests"] += 1

        try:
            # Apply throttling first
            if throttle and throttle in self._throttles:
                throttle_instance = self._throttles[throttle]
                if not await throttle_instance.acquire():
                    self._global_metrics["throttle_rejections"] += 1
                    raise RuntimeError(f"Request throttled by '{throttle}'")

            # Define execution function
            async def execute():
                # Apply bulkhead isolation
                if bulkhead and bulkhead in self._bulkheads:
                    bulkhead_instance = self._bulkheads[bulkhead]
                    return await bulkhead_instance.execute(func, *args, **kwargs)
                else:
                    # Direct execution
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, func, *args, **kwargs)

            # Apply circuit breaker protection
            if circuit_breaker and circuit_breaker in self._circuit_breakers:
                cb_instance = self._circuit_breakers[circuit_breaker]

                # Apply retry handler if specified
                if retry_handler and retry_handler in self._retry_handlers:
                    retry_instance = self._retry_handlers[retry_handler]
                    result = await retry_instance.execute_with_retry(cb_instance.call, execute)
                else:
                    result = await cb_instance.call(execute)
            else:
                # Apply retry handler directly if no circuit breaker
                if retry_handler and retry_handler in self._retry_handlers:
                    retry_instance = self._retry_handlers[retry_handler]
                    result = await retry_instance.execute_with_retry(execute)
                else:
                    result = await execute()

            self._global_metrics["successful_requests"] += 1
            return result

        except CircuitBreakerException:
            self._global_metrics["circuit_breaker_trips"] += 1
            self._global_metrics["failed_requests"] += 1
            raise
        except RuntimeError as e:
            if "bulkhead" in str(e).lower():
                self._global_metrics["bulkhead_rejections"] += 1
            self._global_metrics["failed_requests"] += 1
            raise
        except Exception:
            self._global_metrics["failed_requests"] += 1
            raise

    async def start_all(self) -> None:
        """Start all registered bulkheads."""
        for bulkhead in self._bulkheads.values():
            await bulkhead.start()

    async def stop_all(self) -> None:
        """Stop all registered bulkheads."""
        for bulkhead in self._bulkheads.values():
            await bulkhead.stop()

    def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics from all components."""
        return {
            "global_metrics": self._global_metrics,
            "circuit_breakers": {
                name: cb.get_metrics() for name, cb in self._circuit_breakers.items()
            },
            "bulkheads": {
                name: bulkhead.get_metrics() for name, bulkhead in self._bulkheads.items()
            },
            "registered_components": {
                "circuit_breakers": list(self._circuit_breakers.keys()),
                "bulkheads": list(self._bulkheads.keys()),
                "throttles": list(self._throttles.keys()),
                "retry_handlers": list(self._retry_handlers.keys()),
            },
        }


# Global resilience manager instance
resilience_manager = ResilienceManager()
