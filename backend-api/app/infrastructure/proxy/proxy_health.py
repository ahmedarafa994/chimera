"""ProxyHealthMonitor for continuous proxy server health monitoring.

STORY-1.3: Provides background health monitoring and status tracking
for the AIClient-2-API proxy server.

Features:
- Background health check loop
- Configurable check intervals
- Health history tracking
- Automatic mode switching on failure
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.infrastructure.proxy.proxy_client import ProxyClient, ProxyHealthStatus, get_proxy_client


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    timestamp: float
    is_healthy: bool
    latency_ms: float
    error: str | None = None


@dataclass
class ProxyHealthMetrics:
    """Aggregated health metrics over time."""

    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    uptime_percent: float = 0.0
    last_failure: float | None = None
    last_success: float | None = None


class ProxyHealthMonitor:
    """Monitors the health of the proxy server with background checks.

    Provides continuous monitoring, health history, and metrics
    for the AIClient-2-API proxy server.
    """

    # Default configuration
    DEFAULT_CHECK_INTERVAL = 30  # seconds
    DEFAULT_HISTORY_SIZE = 100
    UNHEALTHY_THRESHOLD = 3  # consecutive failures before unhealthy

    def __init__(
        self,
        proxy_client: ProxyClient | None = None,
        check_interval: int | None = None,
        history_size: int | None = None,
    ) -> None:
        """Initialize the health monitor.

        Args:
            proxy_client: ProxyClient instance to monitor
            check_interval: Interval between health checks (seconds)
            history_size: Maximum number of results to keep in history

        """
        self._client = proxy_client or get_proxy_client()
        self._check_interval = (
            check_interval
            or settings.PROXY_MODE_HEALTH_CHECK_INTERVAL
            or self.DEFAULT_CHECK_INTERVAL
        )
        self._history_size = history_size or self.DEFAULT_HISTORY_SIZE

        self._history: list[HealthCheckResult] = []
        self._metrics = ProxyHealthMetrics()
        self._is_running = False
        self._task: asyncio.Task | None = None
        self._consecutive_failures = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"ProxyHealthMonitor initialized: "
            f"interval={self._check_interval}s, history_size={self._history_size}",
        )

    @property
    def is_healthy(self) -> bool:
        """Check if proxy is currently healthy."""
        return self._client.is_healthy

    @property
    def is_running(self) -> bool:
        """Check if monitoring is active."""
        return self._is_running

    @property
    def metrics(self) -> ProxyHealthMetrics:
        """Get current health metrics."""
        return self._metrics

    @property
    def history(self) -> list[HealthCheckResult]:
        """Get health check history (most recent first)."""
        return list(reversed(self._history))

    async def start(self) -> None:
        """Start the background health monitoring loop."""
        if self._is_running:
            logger.warning("ProxyHealthMonitor already running")
            return

        self._is_running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("ProxyHealthMonitor started")

    async def stop(self) -> None:
        """Stop the background health monitoring loop."""
        if not self._is_running:
            return

        self._is_running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        logger.info("ProxyHealthMonitor stopped")

    async def check_now(self) -> ProxyHealthStatus:
        """Perform an immediate health check."""
        return await self._perform_check()

    async def _monitoring_loop(self) -> None:
        """Background loop that performs periodic health checks."""
        while self._is_running:
            try:
                await self._perform_check()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self._check_interval)

    async def _perform_check(self) -> ProxyHealthStatus:
        """Perform a single health check and update metrics."""
        start_time = time.time()
        status = await self._client.check_health()
        latency_ms = (time.time() - start_time) * 1000

        result = HealthCheckResult(
            timestamp=start_time,
            is_healthy=status.is_healthy,
            latency_ms=latency_ms,
            error=status.error,
        )

        async with self._lock:
            self._update_history(result)
            self._update_metrics(result)
            self._update_consecutive_failures(result)

        return status

    def _update_history(self, result: HealthCheckResult) -> None:
        """Add result to history, maintaining size limit."""
        self._history.append(result)
        while len(self._history) > self._history_size:
            self._history.pop(0)

    def _update_metrics(self, result: HealthCheckResult) -> None:
        """Update aggregated metrics with new result."""
        self._metrics.total_checks += 1

        if result.is_healthy:
            self._metrics.successful_checks += 1
            self._metrics.last_success = result.timestamp

            # Update latency stats
            self._metrics.max_latency_ms = max(
                self._metrics.max_latency_ms,
                result.latency_ms,
            )
            self._metrics.min_latency_ms = min(
                self._metrics.min_latency_ms,
                result.latency_ms,
            )

            # Running average
            n = self._metrics.successful_checks
            prev_avg = self._metrics.avg_latency_ms
            self._metrics.avg_latency_ms = (prev_avg * (n - 1) + result.latency_ms) / n
        else:
            self._metrics.failed_checks += 1
            self._metrics.last_failure = result.timestamp

        # Calculate uptime percentage
        total = self._metrics.total_checks
        if total > 0:
            self._metrics.uptime_percent = (self._metrics.successful_checks / total) * 100

    def _update_consecutive_failures(self, result: HealthCheckResult) -> None:
        """Track consecutive failures for health status."""
        if result.is_healthy:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

            if self._consecutive_failures >= self.UNHEALTHY_THRESHOLD:
                logger.warning(
                    f"Proxy unhealthy: {self._consecutive_failures} consecutive failures",
                )

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive health status report."""
        client_stats = self._client.get_stats()

        return {
            "is_healthy": self.is_healthy,
            "monitoring_active": self._is_running,
            "check_interval_seconds": self._check_interval,
            "consecutive_failures": self._consecutive_failures,
            "unhealthy_threshold": self.UNHEALTHY_THRESHOLD,
            "metrics": {
                "total_checks": self._metrics.total_checks,
                "successful_checks": self._metrics.successful_checks,
                "failed_checks": self._metrics.failed_checks,
                "uptime_percent": round(self._metrics.uptime_percent, 2),
                "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
                "max_latency_ms": round(self._metrics.max_latency_ms, 2),
                "min_latency_ms": (
                    round(self._metrics.min_latency_ms, 2)
                    if self._metrics.min_latency_ms != float("inf")
                    else None
                ),
                "last_success": self._metrics.last_success,
                "last_failure": self._metrics.last_failure,
            },
            "client": client_stats,
            "history_size": len(self._history),
            "recent_checks": [
                {
                    "timestamp": r.timestamp,
                    "is_healthy": r.is_healthy,
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                }
                for r in self.history[:10]
            ],
        }

    def reset_metrics(self) -> None:
        """Reset all metrics and history."""
        self._history.clear()
        self._metrics = ProxyHealthMetrics()
        self._consecutive_failures = 0
        logger.info("ProxyHealthMonitor metrics reset")


# Global health monitor instance
_health_monitor: ProxyHealthMonitor | None = None


def get_health_monitor() -> ProxyHealthMonitor:
    """Get or create the global ProxyHealthMonitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ProxyHealthMonitor()
    return _health_monitor


async def start_health_monitoring() -> None:
    """Start the global health monitor."""
    monitor = get_health_monitor()
    await monitor.start()


async def stop_health_monitoring() -> None:
    """Stop the global health monitor."""
    global _health_monitor
    if _health_monitor is not None:
        await _health_monitor.stop()
        _health_monitor = None
