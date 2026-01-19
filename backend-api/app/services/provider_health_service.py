# =============================================================================
# Chimera - Provider Health Monitoring Service
# =============================================================================
# Background health monitoring with configurable intervals and alerting.
# Tracks health metrics, historical data, and detects status transitions.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# Subtask 2.2: Create Provider Health Monitoring Service
# =============================================================================

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from app.domain.health_models import (
    AlertSeverity,
    HealthAlert,
    HealthHistoryEntry,
    LatencyMetrics,
    ProviderHealthMetrics,
    ProviderHealthStatus,
    ProviderUptimeMetrics,
    RequestMetrics,
    UptimeWindow,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# Default health check configuration
DEFAULT_CHECK_INTERVAL_SECONDS = 60
DEFAULT_CHECK_TIMEOUT_SECONDS = 15
DEFAULT_HISTORY_RETENTION = 1000
DEFAULT_LATENCY_SAMPLE_SIZE = 100

# Threshold defaults
DEFAULT_WARNING_ERROR_RATE = 10.0  # 10% error rate
DEFAULT_CRITICAL_ERROR_RATE = 25.0  # 25% error rate
DEFAULT_WARNING_LATENCY_MS = 2000  # 2 seconds
DEFAULT_CRITICAL_LATENCY_MS = 5000  # 5 seconds
DEFAULT_CONSECUTIVE_FAILURES_DEGRADED = 3
DEFAULT_CONSECUTIVE_FAILURES_DOWN = 5

# Alert configuration
DEFAULT_ALERT_COOLDOWN_SECONDS = 300  # 5 minutes
MAX_ALERTS_RETAINED = 200


# =============================================================================
# Provider Health Configuration
# =============================================================================


class ProviderHealthConfig:
    """
    Configuration for health monitoring of a specific provider.

    Allows per-provider customization of check intervals, thresholds, and alerting.
    """

    def __init__(
        self,
        provider_id: str,
        check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
        check_timeout_seconds: int = DEFAULT_CHECK_TIMEOUT_SECONDS,
        warning_error_rate: float = DEFAULT_WARNING_ERROR_RATE,
        critical_error_rate: float = DEFAULT_CRITICAL_ERROR_RATE,
        warning_latency_ms: float = DEFAULT_WARNING_LATENCY_MS,
        critical_latency_ms: float = DEFAULT_CRITICAL_LATENCY_MS,
        consecutive_failures_degraded: int = DEFAULT_CONSECUTIVE_FAILURES_DEGRADED,
        consecutive_failures_down: int = DEFAULT_CONSECUTIVE_FAILURES_DOWN,
        enabled: bool = True,
    ):
        self.provider_id = provider_id
        self.check_interval_seconds = check_interval_seconds
        self.check_timeout_seconds = check_timeout_seconds
        self.warning_error_rate = warning_error_rate
        self.critical_error_rate = critical_error_rate
        self.warning_latency_ms = warning_latency_ms
        self.critical_latency_ms = critical_latency_ms
        self.consecutive_failures_degraded = consecutive_failures_degraded
        self.consecutive_failures_down = consecutive_failures_down
        self.enabled = enabled

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "provider_id": self.provider_id,
            "check_interval_seconds": self.check_interval_seconds,
            "check_timeout_seconds": self.check_timeout_seconds,
            "thresholds": {
                "warning_error_rate": self.warning_error_rate,
                "critical_error_rate": self.critical_error_rate,
                "warning_latency_ms": self.warning_latency_ms,
                "critical_latency_ms": self.critical_latency_ms,
                "consecutive_failures_degraded": self.consecutive_failures_degraded,
                "consecutive_failures_down": self.consecutive_failures_down,
            },
            "enabled": self.enabled,
        }


# =============================================================================
# Provider Health State
# =============================================================================


class ProviderHealthState:
    """
    Maintains health state for a single provider.

    Tracks metrics, history, latency samples, and calculates rolling statistics.
    """

    def __init__(self, provider_id: str, provider_name: str):
        self.provider_id = provider_id
        self.provider_name = provider_name

        # Current status
        self.status = ProviderHealthStatus.UNKNOWN
        self.status_since: datetime | None = None

        # Request metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.timeout_requests = 0
        self.rate_limited_requests = 0

        # Consecutive tracking
        self.consecutive_failures = 0
        self.consecutive_successes = 0

        # Timing
        self.last_check_at: datetime | None = None
        self.last_success_at: datetime | None = None
        self.last_failure_at: datetime | None = None
        self.last_status_change_at: datetime | None = None

        # Latency samples for percentile calculation
        self._latency_samples: deque[float] = deque(maxlen=DEFAULT_LATENCY_SAMPLE_SIZE)

        # Circuit breaker state
        self.circuit_breaker_state = "closed"

        # Uptime tracking
        self._uptime_samples: deque[tuple[float, bool]] = deque(
            maxlen=3600
        )  # 1 hour at 1s resolution

        # Last calculated metrics
        self._last_calculated_at: float = 0

    def record_check(
        self,
        success: bool,
        latency_ms: float,
        is_timeout: bool = False,
        is_rate_limited: bool = False,
        error_message: str | None = None,
    ) -> None:
        """
        Record the result of a health check.

        Args:
            success: Whether the check succeeded
            latency_ms: Response time in milliseconds
            is_timeout: Whether the check timed out
            is_rate_limited: Whether the check was rate limited
            error_message: Error message if failed
        """
        now = datetime.utcnow()
        self.last_check_at = now
        self.total_requests += 1

        if success:
            self.successful_requests += 1
            self.last_success_at = now
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self._latency_samples.append(latency_ms)
        else:
            self.failed_requests += 1
            self.last_failure_at = now
            self.consecutive_failures += 1
            self.consecutive_successes = 0

            if is_timeout:
                self.timeout_requests += 1
            if is_rate_limited:
                self.rate_limited_requests += 1

        # Record for uptime calculation
        self._uptime_samples.append((time.time(), success))

    def calculate_latency_metrics(self) -> LatencyMetrics:
        """Calculate latency percentiles from samples."""
        if not self._latency_samples:
            return LatencyMetrics()

        sorted_latencies = sorted(self._latency_samples)
        n = len(sorted_latencies)

        return LatencyMetrics(
            avg_ms=sum(sorted_latencies) / n,
            p50_ms=sorted_latencies[int(n * 0.5)] if n > 0 else 0.0,
            p95_ms=sorted_latencies[min(int(n * 0.95), n - 1)] if n > 0 else 0.0,
            p99_ms=sorted_latencies[min(int(n * 0.99), n - 1)] if n > 0 else 0.0,
            min_ms=min(sorted_latencies) if sorted_latencies else 0.0,
            max_ms=max(sorted_latencies) if sorted_latencies else 0.0,
        )

    def calculate_request_metrics(self) -> RequestMetrics:
        """Calculate request statistics."""
        return RequestMetrics(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            timeout_requests=self.timeout_requests,
            rate_limited_requests=self.rate_limited_requests,
        )

    def calculate_error_rate(self) -> float:
        """Calculate current error rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    def calculate_uptime_percent(self, window_seconds: int = 3600) -> float:
        """
        Calculate uptime percentage over a time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Uptime percentage (0-100)
        """
        if not self._uptime_samples:
            return 100.0

        now = time.time()
        cutoff = now - window_seconds

        # Filter samples within window
        window_samples = [(ts, success) for ts, success in self._uptime_samples if ts >= cutoff]

        if not window_samples:
            return 100.0

        successful = sum(1 for _, success in window_samples if success)
        return (successful / len(window_samples)) * 100

    def get_current_latency(self) -> float:
        """Get the most recent latency value."""
        if self._latency_samples:
            return self._latency_samples[-1]
        return 0.0

    def to_metrics(
        self, check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS
    ) -> ProviderHealthMetrics:
        """
        Convert current state to ProviderHealthMetrics model.

        Args:
            check_interval_seconds: Health check interval for metadata

        Returns:
            ProviderHealthMetrics instance
        """
        latency_metrics = self.calculate_latency_metrics()
        request_metrics = self.calculate_request_metrics()
        error_rate = self.calculate_error_rate()
        uptime = self.calculate_uptime_percent()

        return ProviderHealthMetrics(
            provider_id=self.provider_id,
            provider_name=self.provider_name,
            status=self.status,
            latency_ms=self.get_current_latency(),
            latency_metrics=latency_metrics,
            error_rate=error_rate,
            request_metrics=request_metrics,
            uptime_percent=uptime,
            availability=uptime,
            consecutive_failures=self.consecutive_failures,
            consecutive_successes=self.consecutive_successes,
            last_check_at=self.last_check_at,
            last_success_at=self.last_success_at,
            last_failure_at=self.last_failure_at,
            last_status_change_at=self.last_status_change_at,
            circuit_breaker_state=self.circuit_breaker_state,
            check_interval_seconds=check_interval_seconds,
        )


# =============================================================================
# Provider Health Monitoring Service
# =============================================================================


class ProviderHealthService:
    """
    Background health monitoring service with configurable intervals and alerting.

    Features:
    - Background health check tasks for all configured providers
    - Per-provider configurable check intervals and thresholds
    - Historical health data storage for analysis
    - Rolling averages for latency and error rates
    - Status transition detection (operational → degraded → down)
    - Alert generation with cooldown
    - Callback registration for alerts and status changes

    Usage:
        from app.services.provider_health_service import get_provider_health_service

        # Get service instance
        health_service = get_provider_health_service()

        # Start monitoring
        await health_service.start()

        # Get health dashboard data
        dashboard = await health_service.get_health_dashboard()

        # Get specific provider health
        metrics = await health_service.get_provider_health("openai")

        # Stop monitoring
        await health_service.stop()
    """

    _instance: "ProviderHealthService | None" = None

    def __new__(cls) -> "ProviderHealthService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        # Service state
        self._running = False
        self._health_check_task: asyncio.Task | None = None

        # Provider configurations
        self._provider_configs: dict[str, ProviderHealthConfig] = {}

        # Provider health states
        self._provider_states: dict[str, ProviderHealthState] = {}

        # Health history
        self._health_history: deque[HealthHistoryEntry] = deque(maxlen=DEFAULT_HISTORY_RETENTION)

        # Alerts
        self._alerts: deque[HealthAlert] = deque(maxlen=MAX_ALERTS_RETAINED)
        self._alert_cooldown: dict[str, float] = {}  # provider_id -> last alert timestamp

        # Callbacks
        self._alert_callbacks: list[Callable[[HealthAlert], None]] = []
        self._status_change_callbacks: list[
            Callable[[str, ProviderHealthStatus, ProviderHealthStatus], None]
        ] = []

        # Async lock
        self._lock = asyncio.Lock()

        self._initialized = True

        logger.info("ProviderHealthService initialized")

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure_provider(
        self,
        provider_id: str,
        provider_name: str | None = None,
        check_interval_seconds: int | None = None,
        check_timeout_seconds: int | None = None,
        warning_error_rate: float | None = None,
        critical_error_rate: float | None = None,
        warning_latency_ms: float | None = None,
        critical_latency_ms: float | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Configure health monitoring for a provider.

        Args:
            provider_id: Provider identifier
            provider_name: Display name (defaults to provider_id.title())
            check_interval_seconds: Health check interval
            check_timeout_seconds: Timeout for health checks
            warning_error_rate: Error rate threshold for warnings
            critical_error_rate: Error rate threshold for critical
            warning_latency_ms: Latency threshold for warnings
            critical_latency_ms: Latency threshold for critical
            enabled: Whether monitoring is enabled
        """
        display_name = provider_name or provider_id.title()

        # Create or update config
        config = ProviderHealthConfig(
            provider_id=provider_id,
            check_interval_seconds=check_interval_seconds or DEFAULT_CHECK_INTERVAL_SECONDS,
            check_timeout_seconds=check_timeout_seconds or DEFAULT_CHECK_TIMEOUT_SECONDS,
            warning_error_rate=warning_error_rate or DEFAULT_WARNING_ERROR_RATE,
            critical_error_rate=critical_error_rate or DEFAULT_CRITICAL_ERROR_RATE,
            warning_latency_ms=warning_latency_ms or DEFAULT_WARNING_LATENCY_MS,
            critical_latency_ms=critical_latency_ms or DEFAULT_CRITICAL_LATENCY_MS,
            enabled=enabled,
        )
        self._provider_configs[provider_id] = config

        # Create state if not exists
        if provider_id not in self._provider_states:
            self._provider_states[provider_id] = ProviderHealthState(
                provider_id=provider_id,
                provider_name=display_name,
            )

        logger.info(f"Configured health monitoring for provider: {provider_id}")

    def get_provider_config(self, provider_id: str) -> ProviderHealthConfig | None:
        """Get configuration for a provider."""
        return self._provider_configs.get(provider_id)

    def update_provider_config(
        self,
        provider_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Update configuration for a provider.

        Args:
            provider_id: Provider identifier
            **kwargs: Configuration fields to update

        Returns:
            True if updated, False if provider not found
        """
        config = self._provider_configs.get(provider_id)
        if not config:
            return False

        for key, value in kwargs.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return True

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the health monitoring service."""
        if self._running:
            logger.warning("ProviderHealthService is already running")
            return

        self._running = True
        logger.info("Starting ProviderHealthService")

        # Initialize providers from LLM service if not configured
        await self._initialize_providers()

        # Start background health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"ProviderHealthService started with {len(self._provider_configs)} providers")

    async def stop(self) -> None:
        """Stop the health monitoring service."""
        if not self._running:
            logger.warning("ProviderHealthService is not running")
            return

        self._running = False
        logger.info("Stopping ProviderHealthService")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        logger.info("ProviderHealthService stopped")

    async def _initialize_providers(self) -> None:
        """Initialize provider configurations from LLM service."""
        try:
            from app.services.llm_service import llm_service

            providers = llm_service._providers

            for provider_name in providers:
                if provider_name not in self._provider_configs:
                    self.configure_provider(
                        provider_id=provider_name,
                        provider_name=provider_name.title(),
                    )

            logger.info(f"Initialized {len(providers)} providers from LLM service")

        except Exception as e:
            logger.warning(f"Failed to initialize providers from LLM service: {e}")

    # =========================================================================
    # Health Check Loop
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Background loop that runs health checks at configured intervals."""
        # Track last check time per provider
        last_check_times: dict[str, float] = {}

        while self._running:
            try:
                now = time.time()

                # Check each provider at its configured interval
                tasks = []
                for provider_id, config in self._provider_configs.items():
                    if not config.enabled:
                        continue

                    last_check = last_check_times.get(provider_id, 0)
                    if now - last_check >= config.check_interval_seconds:
                        tasks.append(self._check_provider_health(provider_id))
                        last_check_times[provider_id] = now

                # Run health checks concurrently
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Sleep for minimum check interval
                await asyncio.sleep(1)  # Check every second for scheduling

            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error

    async def _check_provider_health(self, provider_id: str) -> dict[str, Any]:
        """
        Perform health check for a single provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Health check result dictionary
        """
        config = self._provider_configs.get(provider_id)
        if not config:
            return {"provider": provider_id, "success": False, "error": "Provider not configured"}

        state = self._provider_states.get(provider_id)
        if not state:
            return {"provider": provider_id, "success": False, "error": "Provider state not found"}

        start_time = time.perf_counter()
        success = False
        error_message = None
        is_timeout = False
        is_rate_limited = False

        try:
            # Check circuit breaker state first
            try:
                from app.core.shared.circuit_breaker import CircuitBreakerRegistry, CircuitState

                cb_status = CircuitBreakerRegistry.get_status(provider_id)
                if cb_status:
                    cb_state = cb_status.get("state")
                    if cb_state == CircuitState.OPEN or cb_state == "open":
                        state.circuit_breaker_state = "open"
                        # Record as failure due to circuit breaker
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        state.record_check(
                            success=False,
                            latency_ms=latency_ms,
                            error_message="Circuit breaker open",
                        )
                        await self._process_health_result(
                            provider_id, False, latency_ms, "Circuit breaker open"
                        )
                        return {
                            "provider": provider_id,
                            "success": False,
                            "latency_ms": latency_ms,
                            "error": "Circuit breaker open",
                            "skipped": True,
                        }
                    elif cb_state == CircuitState.HALF_OPEN or cb_state == "half_open":
                        state.circuit_breaker_state = "half_open"
                    else:
                        state.circuit_breaker_state = "closed"
            except ImportError:
                pass  # Circuit breaker not available

            # Perform lightweight health check
            from app.domain.models import GenerationConfig, LLMProviderType, PromptRequest
            from app.services.llm_service import llm_service

            provider = llm_service.get_provider(provider_id)
            if not provider:
                error_message = f"Provider {provider_id} not registered"
            else:
                # Create minimal health check request
                health_request = PromptRequest(
                    prompt="ping",  # Minimal prompt
                    provider=LLMProviderType(provider_id),
                    config=GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=10,
                    ),
                )

                # Execute with timeout
                response = await asyncio.wait_for(
                    provider.generate(health_request),
                    timeout=config.check_timeout_seconds,
                )

                if response and response.text:
                    success = True
                else:
                    error_message = "Empty response from provider"

        except TimeoutError:
            error_message = f"Health check timeout after {config.check_timeout_seconds}s"
            is_timeout = True
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "quota" in error_str:
                is_rate_limited = True
            error_message = str(e)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record check result
        state.record_check(
            success=success,
            latency_ms=latency_ms,
            is_timeout=is_timeout,
            is_rate_limited=is_rate_limited,
            error_message=error_message,
        )

        # Process result (status transitions, alerts, history)
        await self._process_health_result(provider_id, success, latency_ms, error_message)

        return {
            "provider": provider_id,
            "success": success,
            "latency_ms": latency_ms,
            "error": error_message,
        }

    async def _process_health_result(
        self,
        provider_id: str,
        success: bool,
        latency_ms: float,
        error_message: str | None,
    ) -> None:
        """
        Process health check result, update status, and trigger alerts.

        Args:
            provider_id: Provider identifier
            success: Whether check succeeded
            latency_ms: Response latency
            error_message: Error message if failed
        """
        config = self._provider_configs.get(provider_id)
        state = self._provider_states.get(provider_id)

        if not config or not state:
            return

        # Store old status for change detection
        old_status = state.status

        # Determine new status
        new_status = self._determine_status(config, state)

        # Check for status transition
        if old_status != new_status:
            state.status = new_status
            state.last_status_change_at = datetime.utcnow()
            state.status_since = state.last_status_change_at

            logger.info(
                f"Provider {provider_id} status changed: {old_status.value} -> {new_status.value}"
            )

            # Trigger status change callbacks
            for callback in self._status_change_callbacks:
                try:
                    callback(provider_id, old_status, new_status)
                except Exception as e:
                    logger.error(f"Error in status change callback: {e}")

            # Generate status transition alert
            await self._generate_status_transition_alert(provider_id, old_status, new_status, state)
        else:
            state.status = new_status

        # Check for threshold-based alerts
        await self._check_threshold_alerts(provider_id, config, state)

        # Record history entry
        self._record_history(
            provider_id=provider_id,
            status=new_status,
            success=success,
            latency_ms=latency_ms,
            error_message=error_message,
        )

    def _determine_status(
        self,
        config: ProviderHealthConfig,
        state: ProviderHealthState,
    ) -> ProviderHealthStatus:
        """
        Determine provider health status based on metrics.

        Status logic:
        - DOWN: Consecutive failures >= threshold OR error rate >= critical
        - DEGRADED: Consecutive failures >= warning threshold OR error rate >= warning OR high latency
        - OPERATIONAL: Otherwise
        """
        error_rate = state.calculate_error_rate()
        state.get_current_latency()
        latency_metrics = state.calculate_latency_metrics()

        # Check for DOWN status
        if state.consecutive_failures >= config.consecutive_failures_down:
            return ProviderHealthStatus.DOWN
        if error_rate >= config.critical_error_rate:
            return ProviderHealthStatus.DOWN

        # Check for DEGRADED status
        if state.consecutive_failures >= config.consecutive_failures_degraded:
            return ProviderHealthStatus.DEGRADED
        if error_rate >= config.warning_error_rate:
            return ProviderHealthStatus.DEGRADED
        if latency_metrics.p95_ms >= config.critical_latency_ms:
            return ProviderHealthStatus.DEGRADED
        if latency_metrics.p95_ms >= config.warning_latency_ms:
            return ProviderHealthStatus.DEGRADED

        # Default to OPERATIONAL
        return ProviderHealthStatus.OPERATIONAL

    async def _generate_status_transition_alert(
        self,
        provider_id: str,
        old_status: ProviderHealthStatus,
        new_status: ProviderHealthStatus,
        state: ProviderHealthState,
    ) -> None:
        """Generate an alert for status transition."""
        now = datetime.utcnow()

        # Determine severity based on transition
        if new_status == ProviderHealthStatus.DOWN:
            severity = AlertSeverity.CRITICAL
            title = f"Provider {provider_id} is DOWN"
        elif new_status == ProviderHealthStatus.DEGRADED:
            severity = AlertSeverity.WARNING
            title = f"Provider {provider_id} is DEGRADED"
        elif new_status == ProviderHealthStatus.OPERATIONAL and old_status in (
            ProviderHealthStatus.DOWN,
            ProviderHealthStatus.DEGRADED,
        ):
            severity = AlertSeverity.INFO
            title = f"Provider {provider_id} is RECOVERED"
        else:
            return  # No alert needed

        alert = HealthAlert(
            alert_id=f"status_{provider_id}_{int(now.timestamp())}_{uuid.uuid4().hex[:8]}",
            provider_id=provider_id,
            severity=severity,
            title=title,
            message=f"Provider {provider_id} status changed from {old_status.value} to {new_status.value}",
            error_rate=state.calculate_error_rate(),
            latency_ms=state.get_current_latency(),
            uptime_percent=state.calculate_uptime_percent(),
            previous_status=old_status,
            current_status=new_status,
            triggered_at=now,
        )

        await self._emit_alert(alert)

    async def _check_threshold_alerts(
        self,
        provider_id: str,
        config: ProviderHealthConfig,
        state: ProviderHealthState,
    ) -> None:
        """Check thresholds and generate alerts if exceeded."""
        now = time.time()

        # Check cooldown
        last_alert = self._alert_cooldown.get(provider_id, 0)
        if now - last_alert < DEFAULT_ALERT_COOLDOWN_SECONDS:
            return

        error_rate = state.calculate_error_rate()
        latency_metrics = state.calculate_latency_metrics()
        alert = None

        # Check critical error rate
        if error_rate >= config.critical_error_rate:
            alert = HealthAlert(
                alert_id=f"error_rate_{provider_id}_{int(now)}_{uuid.uuid4().hex[:8]}",
                provider_id=provider_id,
                severity=AlertSeverity.CRITICAL,
                title=f"Critical error rate on {provider_id}",
                message=f"Error rate ({error_rate:.1f}%) exceeds critical threshold ({config.critical_error_rate}%)",
                error_rate=error_rate,
                latency_ms=latency_metrics.p95_ms,
                uptime_percent=state.calculate_uptime_percent(),
                current_status=state.status,
                triggered_at=datetime.utcnow(),
            )
        # Check warning error rate
        elif error_rate >= config.warning_error_rate:
            alert = HealthAlert(
                alert_id=f"error_rate_{provider_id}_{int(now)}_{uuid.uuid4().hex[:8]}",
                provider_id=provider_id,
                severity=AlertSeverity.WARNING,
                title=f"High error rate on {provider_id}",
                message=f"Error rate ({error_rate:.1f}%) exceeds warning threshold ({config.warning_error_rate}%)",
                error_rate=error_rate,
                latency_ms=latency_metrics.p95_ms,
                uptime_percent=state.calculate_uptime_percent(),
                current_status=state.status,
                triggered_at=datetime.utcnow(),
            )
        # Check critical latency
        elif latency_metrics.p95_ms >= config.critical_latency_ms:
            alert = HealthAlert(
                alert_id=f"latency_{provider_id}_{int(now)}_{uuid.uuid4().hex[:8]}",
                provider_id=provider_id,
                severity=AlertSeverity.WARNING,
                title=f"High latency on {provider_id}",
                message=f"P95 latency ({latency_metrics.p95_ms:.0f}ms) exceeds threshold ({config.critical_latency_ms}ms)",
                error_rate=error_rate,
                latency_ms=latency_metrics.p95_ms,
                uptime_percent=state.calculate_uptime_percent(),
                current_status=state.status,
                triggered_at=datetime.utcnow(),
            )

        if alert:
            await self._emit_alert(alert)
            self._alert_cooldown[provider_id] = now

    async def _emit_alert(self, alert: HealthAlert) -> None:
        """Emit an alert and notify callbacks."""
        self._alerts.append(alert)

        logger.warning(f"Health alert: [{alert.severity.value}] {alert.title} - {alert.message}")

        # Trigger alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _record_history(
        self,
        provider_id: str,
        status: ProviderHealthStatus,
        success: bool,
        latency_ms: float,
        error_message: str | None,
    ) -> None:
        """Record a health history entry."""
        state = self._provider_states.get(provider_id)
        if not state:
            return

        entry = HealthHistoryEntry(
            id=f"hist_{provider_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            provider_id=provider_id,
            timestamp=datetime.utcnow(),
            status=status,
            latency_ms=latency_ms,
            error_rate=state.calculate_error_rate(),
            uptime_percent=state.calculate_uptime_percent(),
            success=success,
            response_time_ms=latency_ms,
            error_message=error_message,
            check_type="automatic",
        )

        self._health_history.append(entry)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def register_alert_callback(self, callback: Callable[[HealthAlert], None]) -> None:
        """Register a callback to be invoked when alerts are triggered."""
        self._alert_callbacks.append(callback)

    def register_status_change_callback(
        self, callback: Callable[[str, ProviderHealthStatus, ProviderHealthStatus], None]
    ) -> None:
        """Register a callback to be invoked when provider status changes."""
        self._status_change_callbacks.append(callback)

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_health_dashboard(self) -> dict[str, Any]:
        """
        Get comprehensive health dashboard data.

        Returns:
            Dashboard dictionary with all providers, summary, and alerts
        """
        providers: dict[str, dict[str, Any]] = {}

        for provider_id, state in self._provider_states.items():
            config = self._provider_configs.get(provider_id)
            interval = config.check_interval_seconds if config else DEFAULT_CHECK_INTERVAL_SECONDS
            providers[provider_id] = state.to_metrics(interval).to_dict()

        # Calculate summary
        total = len(self._provider_states)
        operational = sum(
            1
            for s in self._provider_states.values()
            if s.status == ProviderHealthStatus.OPERATIONAL
        )
        degraded = sum(
            1 for s in self._provider_states.values() if s.status == ProviderHealthStatus.DEGRADED
        )
        down = sum(
            1 for s in self._provider_states.values() if s.status == ProviderHealthStatus.DOWN
        )
        unknown = sum(
            1 for s in self._provider_states.values() if s.status == ProviderHealthStatus.UNKNOWN
        )

        # Get recent alerts
        recent_alerts = list(self._alerts)[-20:]

        return {
            "providers": providers,
            "summary": {
                "total_providers": total,
                "operational": operational,
                "degraded": degraded,
                "down": down,
                "unknown": unknown,
                "overall_status": self._get_overall_status(),
            },
            "alerts": [alert.to_dict() for alert in recent_alerts],
            "monitoring": {
                "running": self._running,
                "providers_configured": len(self._provider_configs),
                "history_entries": len(self._health_history),
            },
            "updated_at": datetime.utcnow().isoformat(),
        }

    def _get_overall_status(self) -> str:
        """Determine overall system health status."""
        if not self._provider_states:
            return "unknown"

        statuses = [s.status for s in self._provider_states.values()]

        if any(s == ProviderHealthStatus.DOWN for s in statuses):
            return "critical"
        elif any(s == ProviderHealthStatus.DEGRADED for s in statuses):
            return "degraded"
        elif all(s == ProviderHealthStatus.OPERATIONAL for s in statuses):
            return "healthy"
        else:
            return "unknown"

    async def get_provider_health(self, provider_id: str) -> dict[str, Any] | None:
        """
        Get health metrics for a specific provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Provider health metrics dictionary or None if not found
        """
        state = self._provider_states.get(provider_id)
        if not state:
            return None

        config = self._provider_configs.get(provider_id)
        interval = config.check_interval_seconds if config else DEFAULT_CHECK_INTERVAL_SECONDS

        return state.to_metrics(interval).to_dict()

    async def get_health_history(
        self,
        provider_id: str | None = None,
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get health history entries.

        Args:
            provider_id: Filter by provider (None = all)
            limit: Maximum entries to return
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of health history entries
        """
        history = list(self._health_history)

        # Filter by provider
        if provider_id:
            history = [h for h in history if h.provider_id == provider_id]

        # Filter by time range
        if start_time:
            history = [h for h in history if h.timestamp >= start_time]
        if end_time:
            history = [h for h in history if h.timestamp <= end_time]

        # Sort by timestamp descending
        history.sort(key=lambda h: h.timestamp, reverse=True)

        # Limit results
        history = history[:limit]

        return [h.to_dict() for h in history]

    async def get_alerts(
        self,
        provider_id: str | None = None,
        severity: AlertSeverity | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get health alerts.

        Args:
            provider_id: Filter by provider
            severity: Filter by severity
            active_only: Only return active alerts
            limit: Maximum alerts to return

        Returns:
            List of alert dictionaries
        """
        alerts = list(self._alerts)

        if provider_id:
            alerts = [a for a in alerts if a.provider_id == provider_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if active_only:
            alerts = [a for a in alerts if a.is_active]

        # Sort by timestamp descending
        alerts.sort(key=lambda a: a.triggered_at, reverse=True)

        return [a.to_dict() for a in alerts[:limit]]

    async def get_provider_uptime(self, provider_id: str) -> ProviderUptimeMetrics | None:
        """
        Get detailed uptime metrics for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            ProviderUptimeMetrics or None if not found
        """
        state = self._provider_states.get(provider_id)
        if not state:
            return None

        now = datetime.utcnow()

        # Calculate uptime for different windows
        def _create_uptime_window(window_seconds: int) -> UptimeWindow:
            uptime_pct = state.calculate_uptime_percent(window_seconds)
            uptime_secs = (uptime_pct / 100) * window_seconds
            downtime_secs = window_seconds - uptime_secs

            return UptimeWindow(
                window_start=now - timedelta(seconds=window_seconds),
                window_end=now,
                window_seconds=window_seconds,
                uptime_seconds=uptime_secs,
                downtime_seconds=downtime_secs,
                uptime_percent=uptime_pct,
            )

        return ProviderUptimeMetrics(
            provider_id=provider_id,
            provider_name=state.provider_name,
            current_status=state.status,
            status_since=state.status_since,
            last_hour=_create_uptime_window(3600),
            last_24_hours=_create_uptime_window(86400),
            last_7_days=_create_uptime_window(604800),
            last_30_days=_create_uptime_window(2592000),
            all_time_uptime_percent=state.calculate_uptime_percent(2592000),  # 30 days
            total_incidents=sum(
                1
                for a in self._alerts
                if a.provider_id == provider_id and a.severity == AlertSeverity.CRITICAL
            ),
        )

    async def check_now(self, provider_id: str | None = None) -> dict[str, Any]:
        """
        Trigger an immediate health check.

        Args:
            provider_id: Specific provider to check (None = all)

        Returns:
            Health check results
        """
        if provider_id:
            if provider_id not in self._provider_configs:
                return {"error": f"Provider {provider_id} not configured"}
            result = await self._check_provider_health(provider_id)
            return {"provider": provider_id, "result": result}
        else:
            # Check all providers
            results = {}
            tasks = [
                self._check_provider_health(pid)
                for pid in self._provider_configs
                if self._provider_configs[pid].enabled
            ]
            check_results = await asyncio.gather(*tasks, return_exceptions=True)

            for pid, result in zip(self._provider_configs.keys(), check_results, strict=False):
                if isinstance(result, Exception):
                    results[pid] = {"success": False, "error": str(result)}
                else:
                    results[pid] = result

            return {"providers": results}

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier
            acknowledged_by: User/system acknowledging the alert

        Returns:
            True if acknowledged, False if not found
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.is_acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    async def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            True if resolved, False if not found
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.is_active = False
                alert.resolved_at = datetime.utcnow()
                return True
        return False

    # =========================================================================
    # External Recording
    # =========================================================================

    async def record_external_check(
        self,
        provider_id: str,
        success: bool,
        latency_ms: float,
        is_timeout: bool = False,
        is_rate_limited: bool = False,
        error_message: str | None = None,
    ) -> None:
        """
        Record an external health check result (e.g., from actual LLM requests).

        This allows integrating health tracking with actual request metrics,
        not just health check probes.

        Args:
            provider_id: Provider identifier
            success: Whether the request succeeded
            latency_ms: Request latency
            is_timeout: Whether it was a timeout
            is_rate_limited: Whether it was rate limited
            error_message: Error message if failed
        """
        state = self._provider_states.get(provider_id)
        if not state:
            # Auto-configure if not exists
            self.configure_provider(provider_id)
            state = self._provider_states.get(provider_id)

        if state:
            state.record_check(
                success=success,
                latency_ms=latency_ms,
                is_timeout=is_timeout,
                is_rate_limited=is_rate_limited,
                error_message=error_message,
            )

            # Process result for status updates and alerts
            await self._process_health_result(provider_id, success, latency_ms, error_message)


# =============================================================================
# Global Singleton
# =============================================================================

_provider_health_service: ProviderHealthService | None = None


def get_provider_health_service() -> ProviderHealthService:
    """Get the global Provider Health Service instance."""
    global _provider_health_service
    if _provider_health_service is None:
        _provider_health_service = ProviderHealthService()
    return _provider_health_service
