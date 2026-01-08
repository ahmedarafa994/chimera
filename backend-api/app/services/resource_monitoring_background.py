"""
Resource Monitoring Background Tasks

Provides background task management for continuous resource monitoring,
periodic alerts, and data persistence.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

from app.services.resource_exhaustion_service import (
    AlertLevel,
    ResourceAlert,
    get_resource_exhaustion_service,
)

logger = logging.getLogger("chimera.resource_monitoring")


class ResourceMonitoringManager:
    """
    Manages background monitoring tasks for resource exhaustion tracking.

    Features:
    - Periodic health checks
    - Budget alert monitoring
    - Historical data aggregation
    - Automatic cleanup of stale sessions
    """

    # Configuration constants
    BUDGET_CHECK_INTERVAL_SECONDS = 60
    SESSION_CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes
    METRICS_AGGREGATION_INTERVAL_SECONDS = 3600  # 1 hour
    STALE_SESSION_THRESHOLD_HOURS = 24

    def __init__(self):
        """Initialize the monitoring manager."""
        self._running = False
        self._tasks: dict[str, asyncio.Task] = {}
        self._service = get_resource_exhaustion_service()
        self._startup_time: datetime | None = None
        self._alerts_triggered: int = 0
        self._sessions_cleaned: int = 0
        logger.info("ResourceMonitoringManager initialized")

    @property
    def is_running(self) -> bool:
        """Check if the manager is currently running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Get the uptime in seconds since start."""
        if self._startup_time is None:
            return 0.0
        return (datetime.utcnow() - self._startup_time).total_seconds()

    def get_status(self) -> dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Dictionary with status information
        """
        return {
            "running": self._running,
            "startup_time": (
                self._startup_time.isoformat() if self._startup_time else None
            ),
            "uptime_seconds": self.uptime_seconds,
            "active_tasks": list(self._tasks.keys()),
            "alerts_triggered": self._alerts_triggered,
            "sessions_cleaned": self._sessions_cleaned,
        }

    async def start(self) -> None:
        """Start all background monitoring tasks."""
        if self._running:
            logger.warning("Monitoring manager already running")
            return

        self._running = True
        self._startup_time = datetime.utcnow()

        # Register alert callback
        self._service.register_alert_callback(self._handle_alert)

        # Start background tasks
        self._tasks["budget_monitor"] = asyncio.create_task(
            self._budget_monitor_task(),
            name="budget_monitor",
        )
        self._tasks["session_cleanup"] = asyncio.create_task(
            self._session_cleanup_task(),
            name="session_cleanup",
        )
        self._tasks["metrics_aggregation"] = asyncio.create_task(
            self._metrics_aggregation_task(),
            name="metrics_aggregation",
        )

        logger.info(
            f"ResourceMonitoringManager started with {len(self._tasks)} tasks"
        )

    async def stop(self) -> None:
        """Stop all background monitoring tasks."""
        if not self._running:
            logger.warning("Monitoring manager not running")
            return

        self._running = False

        # Cancel all tasks
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Task '{name}' cancelled")

        self._tasks.clear()
        logger.info("ResourceMonitoringManager stopped")

    async def _budget_monitor_task(self) -> None:
        """Monitor budget thresholds and trigger alerts."""
        logger.info("Budget monitor task started")

        while self._running:
            try:
                # Get budget status
                budget_status = self._service.get_budget_status()

                # Log summary at debug level
                logger.debug(
                    f"Budget check: tokens={budget_status['tokens']['percent_used']:.1f}%, "
                    f"cost={budget_status['cost_usd']['percent_used']:.1f}%, "
                    f"attacks={budget_status['attacks']['percent_used']:.1f}%"
                )

                # Check for threshold breaches
                if budget_status["any_threshold_exceeded"]:
                    logger.warning(
                        f"Budget threshold exceeded: "
                        f"tokens={budget_status['tokens']['percent_used']:.1f}%, "
                        f"cost=${budget_status['cost_usd']['consumed']:.4f}"
                    )

                if budget_status["any_budget_exceeded"]:
                    logger.error(
                        f"BUDGET EXCEEDED: "
                        f"tokens={budget_status['tokens']['percent_used']:.1f}%, "
                        f"cost=${budget_status['cost_usd']['consumed']:.4f}"
                    )

                await asyncio.sleep(self.BUDGET_CHECK_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                logger.info("Budget monitor task cancelled")
                break
            except Exception as e:
                logger.error(f"Budget monitor task error: {e}")
                await asyncio.sleep(self.BUDGET_CHECK_INTERVAL_SECONDS)

    async def _session_cleanup_task(self) -> None:
        """Clean up stale/abandoned sessions."""
        logger.info("Session cleanup task started")

        while self._running:
            try:
                # Get all active sessions
                active_sessions = self._service.get_all_active_sessions()
                now = datetime.utcnow()
                threshold = timedelta(hours=self.STALE_SESSION_THRESHOLD_HOURS)

                cleaned_count = 0
                for session in active_sessions:
                    session_age = now - session.started_at

                    # Check if session is stale
                    if session_age > threshold:
                        logger.info(
                            f"Cleaning up stale session: {session.session_id} "
                            f"(age: {session_age.total_seconds() / 3600:.1f}h)"
                        )
                        self._service.end_session(session.session_id)
                        cleaned_count += 1

                if cleaned_count > 0:
                    self._sessions_cleaned += cleaned_count
                    logger.info(f"Cleaned up {cleaned_count} stale sessions")

                await asyncio.sleep(self.SESSION_CLEANUP_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Session cleanup task error: {e}")
                await asyncio.sleep(self.SESSION_CLEANUP_INTERVAL_SECONDS)

    async def _metrics_aggregation_task(self) -> None:
        """Aggregate metrics for historical analysis."""
        logger.info("Metrics aggregation task started")

        while self._running:
            try:
                # Get hourly summary
                hourly_summary = self._service.get_hourly_summary()

                # Log aggregated metrics
                logger.info(
                    f"Hourly metrics: "
                    f"tokens={hourly_summary['tokens_consumed']}, "
                    f"cost=${hourly_summary['cost_usd']:.4f}, "
                    f"attacks={hourly_summary['attacks_executed']}, "
                    f"projected_hourly_cost=${hourly_summary['projected_cost_per_hour']:.4f}"
                )

                # Get cost breakdown
                cost_breakdown = self._service.get_cost_breakdown()
                if cost_breakdown.get("cost_by_model"):
                    for model, cost in cost_breakdown["cost_by_model"].items():
                        logger.debug(f"  Model {model}: ${cost:.4f}")

                # Get active session count
                active_count = len(self._service.get_all_active_sessions())
                logger.debug(f"Active sessions: {active_count}")

                await asyncio.sleep(self.METRICS_AGGREGATION_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                logger.info("Metrics aggregation task cancelled")
                break
            except Exception as e:
                logger.error(f"Metrics aggregation task error: {e}")
                await asyncio.sleep(self.METRICS_AGGREGATION_INTERVAL_SECONDS)

    def _handle_alert(self, alert: ResourceAlert) -> None:
        """
        Handle triggered alerts (logging, notifications, etc.).

        Args:
            alert: The triggered ResourceAlert
        """
        self._alerts_triggered += 1

        # Log based on severity
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(
                f"CRITICAL ALERT [{alert.metric.value}]: {alert.message} "
                f"(current={alert.current_value}, threshold={alert.threshold_value})"
            )
        elif alert.level == AlertLevel.WARNING:
            logger.warning(
                f"WARNING ALERT [{alert.metric.value}]: {alert.message} "
                f"(current={alert.current_value}, threshold={alert.threshold_value})"
            )
        else:
            logger.info(
                f"INFO ALERT [{alert.metric.value}]: {alert.message} "
                f"(current={alert.current_value}, threshold={alert.threshold_value})"
            )

        # Additional alert handling could be added here:
        # - Send to notification service
        # - Write to alert log file
        # - Trigger webhook
        # - Send email/Slack notification


# Global manager instance
_manager: ResourceMonitoringManager | None = None


def get_monitoring_manager() -> ResourceMonitoringManager:
    """Get or create the monitoring manager singleton."""
    global _manager
    if _manager is None:
        _manager = ResourceMonitoringManager()
    return _manager


@asynccontextmanager
async def resource_monitoring_context():
    """
    Context manager for resource monitoring lifecycle.

    Usage:
        async with resource_monitoring_context() as manager:
            # Manager is running
            status = manager.get_status()
        # Manager is stopped
    """
    manager = get_monitoring_manager()
    await manager.start()
    try:
        yield manager
    finally:
        await manager.stop()


async def start_monitoring() -> ResourceMonitoringManager:
    """
    Start the resource monitoring manager.

    Returns:
        The started ResourceMonitoringManager instance
    """
    manager = get_monitoring_manager()
    await manager.start()
    return manager


async def stop_monitoring() -> None:
    """Stop the resource monitoring manager."""
    manager = get_monitoring_manager()
    await manager.stop()


# FastAPI lifespan integration helper
@asynccontextmanager
async def lifespan_monitoring():
    """
    FastAPI lifespan context manager for automatic startup/shutdown.

    Usage in main.py:
        from app.services.resource_monitoring_background import lifespan_monitoring

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with lifespan_monitoring():
                yield

        app = FastAPI(lifespan=lifespan)
    """
    logger.info("Starting resource monitoring as part of application lifespan")
    manager = get_monitoring_manager()
    await manager.start()
    try:
        yield
    finally:
        logger.info("Stopping resource monitoring as part of application shutdown")
        await manager.stop()
