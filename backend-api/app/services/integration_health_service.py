"""Integration Health Service for Provider Health Monitoring.

STORY-1.4: Provider Health Monitoring

This service provides comprehensive health monitoring for all registered LLM providers:
- Tracks health metrics (uptime, latency, error rates)
- Runs periodic health checks at configurable intervals
- Integrates with circuit breaker pattern for automatic failover
- Exposes health status via `/health/integration` endpoint
- Maintains health history for trend analysis
- Triggers alerts on health degradation

Configuration:
- HEALTH_CHECK_INTERVAL_SECONDS: Health check interval (default: 30)
- HEALTH_CHECK_TIMEOUT_SECONDS: Timeout for individual health checks (default: 10)
- HEALTH_HISTORY_RETENTION: Number of history records to keep (default: 1000)
- HEALTH_DEGRADATION_WARNING_THRESHOLD: Error rate % for warnings (default: 50)
- HEALTH_DEGRADATION_CRITICAL_THRESHOLD: Error rate % for critical (default: 80)
- HEALTH_LATENCY_WARNING_MS: Latency threshold for warnings (default: 2000)
"""

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.core.config import settings
from app.core.shared.circuit_breaker import CircuitBreakerRegistry, CircuitState
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Health alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ProviderHealthMetrics:
    """Health metrics for a single provider."""

    provider_name: str
    status: HealthStatus = HealthStatus.UNKNOWN

    # Latency metrics (milliseconds)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_avg: float = 0.0

    # Error metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0  # Percentage

    # Availability metrics
    uptime_percent: float = 100.0
    last_check_time: float = field(default_factory=time.time)
    last_success_time: float | None = None
    last_failure_time: float | None = None

    # Consecutive failures
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Circuit breaker state
    circuit_state: CircuitState = CircuitState.CLOSED

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "latency_ms": {
                "p50": round(self.latency_p50, 2),
                "p95": round(self.latency_p95, 2),
                "p99": round(self.latency_p99, 2),
                "avg": round(self.latency_avg, 2),
            },
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "error_rate_percent": round(self.error_rate, 2),
            },
            "availability": {
                "uptime_percent": round(self.uptime_percent, 2),
                "last_check": datetime.fromtimestamp(self.last_check_time).isoformat(),
                "last_success": (
                    datetime.fromtimestamp(self.last_success_time).isoformat()
                    if self.last_success_time
                    else None
                ),
                "last_failure": (
                    datetime.fromtimestamp(self.last_failure_time).isoformat()
                    if self.last_failure_time
                    else None
                ),
            },
            "consecutive": {
                "failures": self.consecutive_failures,
                "successes": self.consecutive_successes,
            },
            "circuit_breaker": {
                "state": self.circuit_state.value,
            },
        }


@dataclass
class HealthHistoryEntry:
    """Single health history entry."""

    timestamp: float
    provider_name: str
    status: HealthStatus
    latency_ms: float
    success: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert history entry to dictionary."""
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "provider_name": self.provider_name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
        }


@dataclass
class HealthAlert:
    """Health alert for degradation detection."""

    alert_id: str
    provider_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    error_rate: float | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "provider_name": self.provider_name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "metrics": {
                "error_rate_percent": (
                    round(self.error_rate, 2) if self.error_rate is not None else None
                ),
                "latency_ms": round(self.latency_ms, 2) if self.latency_ms is not None else None,
            },
        }


class IntegrationHealthService:
    """Integration Health Service for provider health monitoring.

    This service runs periodic health checks on all registered providers,
    tracks metrics, maintains history, and triggers alerts on degradation.

    Usage:
        from app.services.integration_health_service import get_health_service

        # Start health monitoring (typically in lifespan.py)
        health_service = get_health_service()
        await health_service.start()

        # Get current health status
        status = await health_service.get_health_status()

        # Stop health monitoring
        await health_service.stop()
    """

    def __init__(
        self,
        check_interval_seconds: int = 30,
        check_timeout_seconds: int = 10,
        history_retention: int = 1000,
    ) -> None:
        """Initialize the Integration Health Service.

        Args:
            check_interval_seconds: Interval between health checks (default: 30)
            check_timeout_seconds: Timeout for individual health checks (default: 10)
            history_retention: Number of history records to keep (default: 1000)

        """
        # Configuration
        self._check_interval = check_interval_seconds
        self._check_timeout = check_timeout_seconds
        self._history_retention = history_retention

        # Health thresholds
        self._warning_error_rate = getattr(settings, "HEALTH_DEGRADATION_WARNING_THRESHOLD", 50)
        self._critical_error_rate = getattr(settings, "HEALTH_DEGRADATION_CRITICAL_THRESHOLD", 80)
        self._warning_latency_ms = getattr(settings, "HEALTH_LATENCY_WARNING_MS", 2000)

        # State
        self._running = False
        self._health_check_task: asyncio.Task | None = None

        # Provider health metrics (provider_name -> ProviderHealthMetrics)
        self._provider_metrics: dict[str, ProviderHealthMetrics] = {}

        # Health history (deque for efficient pops from left)
        self._health_history: deque[HealthHistoryEntry] = deque(maxlen=history_retention)

        # Latency samples for percentile calculation (provider_name -> list of latencies)
        self._latency_samples: dict[str, list[float]] = {}

        # Alerts
        self._alerts: deque[HealthAlert] = deque(maxlen=100)  # Keep last 100 alerts
        self._alert_cooldown: dict[str, float] = {}  # provider_name -> last alert time
        self._alert_cooldown_seconds = 300  # 5 minutes cooldown

        # Alert callbacks
        self._alert_callbacks: list[Callable[[HealthAlert], None]] = []

        # Health check callbacks (called on health status changes)
        self._health_change_callbacks: list[Callable[[str, HealthStatus, HealthStatus], None]] = []

    def register_alert_callback(self, callback: Callable[[HealthAlert], None]) -> None:
        """Register a callback to be invoked when alerts are triggered."""
        self._alert_callbacks.append(callback)

    def register_health_change_callback(
        self,
        callback: Callable[[str, HealthStatus, HealthStatus], None],
    ) -> None:
        """Register a callback to be invoked when provider health status changes."""
        self._health_change_callbacks.append(callback)

    async def start(self) -> None:
        """Start the health monitoring service."""
        if self._running:
            logger.warning("Integration health service is already running")
            return

        self._running = True
        logger.info("Starting Integration Health Service")

        # Initialize metrics for registered providers
        await self._initialize_provider_metrics()

        # Start background health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Integration Health Service started (check_interval={self._check_interval}s)")

    async def stop(self) -> None:
        """Stop the health monitoring service."""
        if not self._running:
            logger.warning("Integration health service is not running")
            return

        self._running = False
        logger.info("Stopping Integration Health Service")

        # Cancel background health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        logger.info("Integration Health Service stopped")

    async def _initialize_provider_metrics(self) -> None:
        """Initialize health metrics for all registered providers."""
        # Get registered providers from llm_service
        providers = llm_service._providers

        for provider_name in providers:
            if provider_name not in self._provider_metrics:
                self._provider_metrics[provider_name] = ProviderHealthMetrics(
                    provider_name=provider_name,
                )
                self._latency_samples[provider_name] = []

        logger.info(f"Initialized health metrics for {len(providers)} providers")

    async def _health_check_loop(self) -> None:
        """Background loop that runs health checks at configured intervals."""
        while self._running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}", exc_info=True)
                # Continue running despite errors

    async def _run_health_checks(self) -> None:
        """Run health checks for all registered providers."""
        providers = list(llm_service._providers.keys())

        if not providers:
            logger.debug("No providers registered for health checks")
            return

        # Run health checks concurrently
        results = await asyncio.gather(
            *[self._check_provider_health(provider) for provider in providers],
            return_exceptions=True,
        )

        # Process results
        for provider, result in zip(providers, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Health check error for {provider}: {result}")
                await self._record_health_check(
                    provider_name=provider,
                    success=False,
                    latency_ms=0.0,
                    error=str(result),
                )

    async def _check_provider_health(self, provider_name: str) -> dict[str, Any]:
        """Perform health check for a single provider.

        Health check strategy: Make a lightweight API call to verify provider connectivity.
        - Success: Response within timeout with valid response format
        - Failure: Timeout, error response, or exception
        """
        # Check circuit breaker state BEFORE making API call
        try:
            cb_status = CircuitBreakerRegistry.get_status(provider_name)
            if cb_status:
                state = cb_status.get("state")
                if state in (CircuitState.OPEN, "open"):
                    # Skip health check for open circuit breakers (quota exceeded)
                    return {
                        "provider": provider_name,
                        "success": False,
                        "latency_ms": 0.0,
                        "error": "Circuit breaker open - skipping health check",
                        "skipped": True,
                    }
        except Exception:
            pass  # If circuit breaker check fails, continue with normal health check

        start_time = time.perf_counter()
        success = False
        error = None

        try:
            # Get provider instance
            provider = llm_service.get_provider(provider_name)

            # Perform lightweight health check
            # For now, we'll use a simple ping-like check with minimal prompt
            from app.domain.models import GenerationConfig, LLMProviderType, PromptRequest

            # Create a minimal health check request
            health_request = PromptRequest(
                prompt="ping",  # Minimal prompt
                provider=LLMProviderType(provider_name),
                config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=10,  # Minimal output
                ),
            )

            # Execute health check with timeout
            response = await asyncio.wait_for(
                provider.generate(health_request),
                timeout=self._check_timeout,
            )

            # Check response validity
            if response and response.text:
                success = True
            else:
                error = "Empty response from provider"

        except TimeoutError:
            error = f"Health check timeout after {self._check_timeout}s"
        except Exception as e:
            error = str(e)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record health check
        await self._record_health_check(
            provider_name=provider_name,
            success=success,
            latency_ms=latency_ms,
            error=error,
        )

        return {
            "provider": provider_name,
            "success": success,
            "latency_ms": latency_ms,
            "error": error,
        }

    async def _record_health_check(
        self,
        provider_name: str,
        success: bool,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """Record the results of a health check.

        Updates provider metrics, history, and triggers alerts if needed.
        """
        now = time.time()

        # Get or create provider metrics
        if provider_name not in self._provider_metrics:
            self._provider_metrics[provider_name] = ProviderHealthMetrics(
                provider_name=provider_name,
            )
        if provider_name not in self._latency_samples:
            self._latency_samples[provider_name] = []

        metrics = self._provider_metrics[provider_name]

        # Store old status for change detection
        old_status = metrics.status

        # Update metrics
        metrics.total_requests += 1
        metrics.last_check_time = now

        if success:
            metrics.successful_requests += 1
            metrics.last_success_time = now
            metrics.consecutive_successes += 1
            metrics.consecutive_failures = 0

            # Record latency sample
            self._latency_samples[provider_name].append(latency_ms)

            # Keep only last 100 samples for percentile calculation
            if len(self._latency_samples[provider_name]) > 100:
                self._latency_samples[provider_name].pop(0)
        else:
            metrics.failed_requests += 1
            metrics.last_failure_time = now
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0

        # Calculate error rate
        if metrics.total_requests > 0:
            metrics.error_rate = (metrics.failed_requests / metrics.total_requests) * 100

        # Calculate uptime
        if metrics.total_requests > 0:
            metrics.uptime_percent = (metrics.successful_requests / metrics.total_requests) * 100

        # Calculate latency percentiles
        if self._latency_samples[provider_name]:
            sorted_latencies = sorted(self._latency_samples[provider_name])
            n = len(sorted_latencies)
            metrics.latency_p50 = sorted_latencies[int(n * 0.5)]
            metrics.latency_p95 = sorted_latencies[int(n * 0.95)]
            metrics.latency_p99 = sorted_latencies[int(n * 0.99)]
            metrics.latency_avg = sum(sorted_latencies) / n

        # Get circuit breaker state
        cb_status = CircuitBreakerRegistry.get_status(provider_name)
        if cb_status:
            state = cb_status.get("state")
            if isinstance(state, CircuitState):
                metrics.circuit_state = state
            elif isinstance(state, str):
                try:
                    metrics.circuit_state = CircuitState(state)
                except ValueError:
                    metrics.circuit_state = CircuitState.CLOSED

        # Determine health status
        new_status = self._determine_health_status(metrics)
        metrics.status = new_status

        # Record history
        status_for_history = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY
        self._health_history.append(
            HealthHistoryEntry(
                timestamp=now,
                provider_name=provider_name,
                status=status_for_history,
                latency_ms=latency_ms,
                success=success,
            ),
        )

        # Check for health status changes
        if old_status != new_status:
            logger.info(
                f"Provider {provider_name} health status changed: {old_status.value} -> {new_status.value}",
            )
            # Trigger health change callbacks
            for callback in self._health_change_callbacks:
                try:
                    callback(provider_name, old_status, new_status)
                except Exception as e:
                    logger.exception(f"Error in health change callback: {e}")

        # Check for degradation and trigger alerts
        await self._check_for_degradation(provider_name, metrics)

    def _determine_health_status(self, metrics: ProviderHealthMetrics) -> HealthStatus:
        """Determine health status based on metrics.

        Status determination:
        - HEALTHY: Error rate < warning threshold AND latency < warning threshold
        - DEGRADED: Error rate between warning and critical OR latency >= warning threshold
        - UNHEALTHY: Error rate >= critical threshold
        """
        if metrics.error_rate >= self._critical_error_rate:
            return HealthStatus.UNHEALTHY
        if (
            metrics.error_rate >= self._warning_error_rate
            or metrics.latency_p95 >= self._warning_latency_ms
        ):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    async def _check_for_degradation(
        self,
        provider_name: str,
        metrics: ProviderHealthMetrics,
    ) -> None:
        """Check for health degradation and trigger alerts if needed."""
        now = time.time()

        # Check alert cooldown
        last_alert_time = self._alert_cooldown.get(provider_name, 0)
        if now - last_alert_time < self._alert_cooldown_seconds:
            return

        alert = None

        # Check for critical degradation
        if metrics.error_rate >= self._critical_error_rate:
            alert = HealthAlert(
                alert_id=f"{provider_name}-critical-{int(now)}",
                provider_name=provider_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Provider {provider_name} is critically unhealthy (error rate: {metrics.error_rate:.1f}%)",
                timestamp=now,
                error_rate=metrics.error_rate,
                latency_ms=metrics.latency_p95,
            )
        # Check for warning degradation
        elif metrics.error_rate >= self._warning_error_rate:
            alert = HealthAlert(
                alert_id=f"{provider_name}-warning-{int(now)}",
                provider_name=provider_name,
                severity=AlertSeverity.WARNING,
                message=f"Provider {provider_name} is degraded (error rate: {metrics.error_rate:.1f}%)",
                timestamp=now,
                error_rate=metrics.error_rate,
                latency_ms=metrics.latency_p95,
            )
        # Check for high latency
        elif metrics.latency_p95 >= self._warning_latency_ms:
            alert = HealthAlert(
                alert_id=f"{provider_name}-latency-{int(now)}",
                provider_name=provider_name,
                severity=AlertSeverity.WARNING,
                message=f"Provider {provider_name} has high latency (p95: {metrics.latency_p95:.0f}ms)",
                timestamp=now,
                error_rate=metrics.error_rate,
                latency_ms=metrics.latency_p95,
            )

        if alert:
            self._alerts.append(alert)
            self._alert_cooldown[provider_name] = now
            logger.warning(f"Health alert triggered: {alert.severity.value} - {alert.message}")

            # Trigger alert callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.exception(f"Error in alert callback: {e}")

    async def get_health_status(self) -> dict[str, Any]:
        """Get current health status for all providers.

        Returns a dictionary with:
        - providers: Health metrics for each provider
        - summary: Overall health summary
        - alerts: Recent alerts
        """
        # Build provider health status
        providers_health = {}
        for provider_name, metrics in self._provider_metrics.items():
            providers_health[provider_name] = metrics.to_dict()

        # Calculate summary
        healthy_count = sum(
            1 for m in self._provider_metrics.values() if m.status == HealthStatus.HEALTHY
        )
        degraded_count = sum(
            1 for m in self._provider_metrics.values() if m.status == HealthStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for m in self._provider_metrics.values() if m.status == HealthStatus.UNHEALTHY
        )

        # Get recent alerts (last 10)
        recent_alerts = list(self._alerts)[-10:]

        return {
            "providers": providers_health,
            "summary": {
                "total_providers": len(self._provider_metrics),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "overall_status": self._get_overall_status(),
            },
            "alerts": [alert.to_dict() for alert in recent_alerts],
            "monitoring": {
                "check_interval_seconds": self._check_interval,
                "check_timeout_seconds": self._check_timeout,
                "running": self._running,
            },
        }

    def _get_overall_status(self) -> str:
        """Determine overall system health status."""
        if not self._provider_metrics:
            return "unknown"

        statuses = [m.status for m in self._provider_metrics.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return "unhealthy"
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return "degraded"
        return "healthy"

    async def get_provider_health(self, provider_name: str) -> dict[str, Any] | None:
        """Get health status for a specific provider."""
        metrics = self._provider_metrics.get(provider_name)
        if not metrics:
            return None
        return metrics.to_dict()

    async def get_health_history(
        self,
        provider_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get health history.

        Args:
            provider_name: Filter by provider (None = all providers)
            limit: Maximum number of entries to return

        Returns:
            List of health history entries

        """
        history = list(self._health_history)

        # Filter by provider if specified
        if provider_name:
            history = [h for h in history if h.provider_name == provider_name]

        # Sort by timestamp descending and limit
        history = sorted(history, key=lambda h: h.timestamp, reverse=True)[:limit]

        return [h.to_dict() for h in history]

    async def get_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent health alerts."""
        alerts = list(self._alerts)[-limit:]
        return [alert.to_dict() for alert in alerts]

    async def get_service_dependency_graph(self) -> dict[str, Any]:
        """Get service dependency graph.

        Returns a graph showing provider dependencies and health status.
        """
        nodes = []
        edges = []

        # Add provider nodes
        for provider_name, metrics in self._provider_metrics.items():
            nodes.append(
                {
                    "id": provider_name,
                    "type": "provider",
                    "status": metrics.status.value,
                    "label": provider_name,
                    "metrics": {
                        "error_rate": round(metrics.error_rate, 2),
                        "latency_p95_ms": round(metrics.latency_p95, 2),
                        "uptime_percent": round(metrics.uptime_percent, 2),
                    },
                },
            )

        # Add edges (dependencies based on failover chains)
        failover_chains = llm_service._DEFAULT_FAILOVER_CHAIN
        for primary, fallbacks in failover_chains.items():
            if primary in self._provider_metrics:
                for fallback in fallbacks:
                    if fallback in self._provider_metrics:
                        edges.append(
                            {
                                "from": primary,
                                "to": fallback,
                                "type": "failover",
                                "label": "failover",
                            },
                        )

        return {
            "nodes": nodes,
            "edges": edges,
        }

    async def check_now(self) -> dict[str, Any]:
        """Trigger an immediate health check for all providers.

        Returns the health status after running checks.
        """
        logger.info("Triggering immediate health check")
        await self._run_health_checks()
        return await self.get_health_status()


# Global singleton instance
_health_service: IntegrationHealthService | None = None


def get_health_service() -> IntegrationHealthService:
    """Get the global Integration Health Service instance."""
    global _health_service
    if _health_service is None:
        # Load configuration from settings
        check_interval = getattr(settings, "HEALTH_CHECK_INTERVAL_SECONDS", 30)
        check_timeout = getattr(settings, "HEALTH_CHECK_TIMEOUT_SECONDS", 10)
        history_retention = getattr(settings, "HEALTH_HISTORY_RETENTION", 1000)

        _health_service = IntegrationHealthService(
            check_interval_seconds=check_interval,
            check_timeout_seconds=check_timeout,
            history_retention=history_retention,
        )
    return _health_service
