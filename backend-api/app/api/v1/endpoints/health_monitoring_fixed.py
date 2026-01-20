"""Endpoint Health Monitoring and Alerting System - FIXED VERSION.

This module provides:
1. Real-time endpoint health monitoring
2. Performance metrics collection
3. Alerting for unhealthy endpoints
4. Health dashboard with detailed status
5. Automatic recovery detection
"""

import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Data Models
# =============================================================================


class EndpointMetrics(BaseModel):
    """Metrics for a single endpoint."""

    endpoint: str
    method: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: str | None = None
    last_success_time: str | None = None
    last_failure_time: str | None = None
    current_status: str = "unknown"  # healthy, degraded, unhealthy
    error_rate: float = 0.0
    availability: float = 0.0


class EndpointHealth(BaseModel):
    """Health status of an endpoint."""

    endpoint: str
    method: str
    status: str = Field(description="healthy, degraded, or unhealthy")
    response_time_ms: float | None = None
    last_checked: str
    error_message: str | None = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0


class AlertRule(BaseModel):
    """Alert rule configuration."""

    name: str
    condition: str  # error_rate, response_time, availability
    threshold: float
    duration_minutes: int = 5
    enabled: bool = True


class Alert(BaseModel):
    """Generated alert."""

    id: str
    rule_name: str
    endpoint: str
    message: str
    severity: str = Field(description="low, medium, high, critical")
    triggered_at: str
    resolved_at: str | None = None
    status: str = "active"  # active, resolved, acknowledged


class HealthDashboard(BaseModel):
    """Complete health dashboard."""

    overall_status: str
    total_endpoints: int
    healthy_endpoints: int
    degraded_endpoints: int
    unhealthy_endpoints: int
    active_alerts: int
    endpoints: list[EndpointHealth]
    recent_alerts: list[Alert]
    metrics_summary: dict[str, Any]
    timestamp: str


# =============================================================================
# Health Monitor
# =============================================================================


class EndpointHealthMonitor:
    """Monitors endpoint health and generates alerts."""

    def __init__(self) -> None:
        self.metrics = {}
        self.health_status = {}
        self.alerts = {}
        self.alert_rules = {}
        self.request_history = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_enabled = True
        self.check_interval = 30  # seconds

        # Initialize default alert rules
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate",
                threshold=0.1,  # 10% error rate
                duration_minutes=5,
            ),
            AlertRule(
                name="slow_response_time",
                condition="response_time",
                threshold=5000.0,  # 5 seconds
                duration_minutes=3,
            ),
            AlertRule(
                name="low_availability",
                condition="availability",
                threshold=0.95,  # 95% availability
                duration_minutes=10,
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.name] = rule

    def record_request(
        self, endpoint: str, method: str, success: bool, response_time: float
    ) -> None:
        """Record a request for monitoring."""
        if not self.monitoring_enabled:
            return

        key = f"{method}:{endpoint}"
        timestamp = datetime.utcnow()

        # Initialize metrics if not exists
        if key not in self.metrics:
            self.metrics[key] = EndpointMetrics(
                endpoint=endpoint,
                method=method,
            )

        # Update metrics
        metrics = self.metrics[key]
        metrics.total_requests += 1
        metrics.last_request_time = timestamp.isoformat()

        if success:
            metrics.successful_requests += 1
            metrics.last_success_time = timestamp.isoformat()
        else:
            metrics.failed_requests += 1
            metrics.last_failure_time = timestamp.isoformat()

        # Update response time (moving average)
        if metrics.total_requests == 1:
            metrics.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            metrics.average_response_time = (
                alpha * response_time + (1 - alpha) * metrics.average_response_time
            )

        # Calculate error rate
        metrics.error_rate = metrics.failed_requests / metrics.total_requests

        # Calculate availability (last 100 requests)
        self.request_history[key].append(
            {
                "timestamp": timestamp,
                "success": success,
                "response_time": response_time,
            },
        )

        recent_requests = list(self.request_history[key])[-100:]
        if recent_requests:
            successful_recent = sum(1 for req in recent_requests if req["success"])
            metrics.availability = successful_recent / len(recent_requests)

        # Update current status
        metrics.current_status = self._calculate_endpoint_status(metrics)

        # Check for alerts
        self._check_alerts(key, metrics)

    def _calculate_endpoint_status(self, metrics: EndpointMetrics) -> str:
        """Calculate endpoint status based on metrics."""
        # Unhealthy conditions
        if metrics.error_rate > 0.2 or metrics.availability < 0.8:
            return "unhealthy"

        # Degraded conditions
        if (
            metrics.error_rate > 0.05
            or metrics.availability < 0.95
            or metrics.average_response_time > 2000
        ):
            return "degraded"

        return "healthy"

    def _check_alerts(self, endpoint_key: str, metrics: EndpointMetrics) -> None:
        """Check if any alert rules are triggered."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            alert_key = f"{rule_name}:{endpoint_key}"
            should_alert = False

            # Check condition
            if (
                (rule.condition == "error_rate" and metrics.error_rate > rule.threshold)
                or (
                    rule.condition == "response_time"
                    and metrics.average_response_time > rule.threshold
                )
                or (rule.condition == "availability" and metrics.availability < rule.threshold)
            ):
                should_alert = True

            # Generate or resolve alert
            if should_alert and alert_key not in self.alerts:
                self._generate_alert(rule, endpoint_key, metrics)
            elif not should_alert and alert_key in self.alerts:
                self._resolve_alert(alert_key)

    def _generate_alert(self, rule: AlertRule, endpoint_key: str, metrics: EndpointMetrics) -> None:
        """Generate a new alert."""
        alert_id = f"alert_{int(time.time())}_{len(self.alerts)}"
        endpoint = metrics.endpoint
        method = metrics.method

        # Determine severity
        severity = "medium"
        if rule.condition == "error_rate":
            severity = "high" if metrics.error_rate > 0.3 else "medium"
        elif rule.condition == "response_time":
            severity = "high" if metrics.average_response_time > 10000 else "medium"
        elif rule.condition == "availability":
            severity = "critical" if metrics.availability < 0.5 else "high"

        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            endpoint=f"{method} {endpoint}",
            message=self._generate_alert_message(rule, metrics),
            severity=severity,
            triggered_at=datetime.utcnow().isoformat(),
            status="active",
        )

        alert_key = f"{rule.name}:{endpoint_key}"
        self.alerts[alert_key] = alert

        logger.warning(f"Alert generated: {alert.message}")

    def _resolve_alert(self, alert_key: str) -> None:
        """Resolve an active alert."""
        if alert_key in self.alerts:
            alert = self.alerts[alert_key]
            alert.status = "resolved"
            alert.resolved_at = datetime.utcnow().isoformat()
            logger.info(f"Alert resolved: {alert.message}")

    def _generate_alert_message(self, rule: AlertRule, metrics: EndpointMetrics) -> str:
        """Generate alert message."""
        if rule.condition == "error_rate":
            return f"High error rate detected: {metrics.error_rate:.2%} (threshold: {rule.threshold:.2%})"
        if rule.condition == "response_time":
            return f"Slow response time: {metrics.average_response_time:.0f}ms (threshold: {rule.threshold:.0f}ms)"
        if rule.condition == "availability":
            return f"Low availability: {metrics.availability:.2%} (threshold: {rule.threshold:.2%})"

        return f"Alert condition '{rule.condition}' triggered"

    async def health_check_endpoint(self, endpoint: str, method: str = "GET") -> EndpointHealth:
        """Perform health check for specific endpoint."""
        key = f"{method}:{endpoint}"
        start_time = time.time()

        try:
            # Simulate health check (in production, make actual HTTP request)
            import os

            import aiohttp

            base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
            url = f"{base_url}{endpoint}"

            timeout = aiohttp.ClientTimeout(total=10)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.request(method, url) as response,
            ):
                response_time = (time.time() - start_time) * 1000
                success = 200 <= response.status < 400

                # Record the health check
                self.record_request(endpoint, method, success, response_time)

                metrics = self.metrics.get(key)
                consecutive_failures = 0

                if metrics:
                    # Calculate consecutive failures
                    recent_requests = list(self.request_history[key])[-10:]
                    for req in reversed(recent_requests):
                        if not req["success"]:
                            consecutive_failures += 1
                        else:
                            break

                return EndpointHealth(
                    endpoint=endpoint,
                    method=method,
                    status="healthy" if success else "unhealthy",
                    response_time_ms=response_time,
                    last_checked=datetime.utcnow().isoformat(),
                    consecutive_failures=consecutive_failures,
                    uptime_percentage=metrics.availability * 100 if metrics else 0.0,
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.record_request(endpoint, method, False, response_time)

            logger.exception(f"Health check failed for {method} {endpoint}: {e}")

            return EndpointHealth(
                endpoint=endpoint,
                method=method,
                status="unhealthy",
                response_time_ms=response_time,
                last_checked=datetime.utcnow().isoformat(),
                error_message=str(e),
                consecutive_failures=1,
                uptime_percentage=0.0,
            )

    def get_dashboard(self) -> HealthDashboard:
        """Get complete health dashboard."""
        endpoints = []
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0

        # Convert metrics to endpoint health
        for key, metrics in self.metrics.items():
            method, endpoint = key.split(":", 1)

            # Calculate consecutive failures
            consecutive_failures = 0
            recent_requests = list(self.request_history[key])[-10:]
            for req in reversed(recent_requests):
                if not req["success"]:
                    consecutive_failures += 1
                else:
                    break

            endpoint_health = EndpointHealth(
                endpoint=endpoint,
                method=method,
                status=metrics.current_status,
                response_time_ms=metrics.average_response_time,
                last_checked=datetime.utcnow().isoformat(),
                consecutive_failures=consecutive_failures,
                uptime_percentage=metrics.availability * 100,
            )

            endpoints.append(endpoint_health)

            # Count statuses
            if endpoint_health.status == "healthy":
                healthy_count += 1
            elif endpoint_health.status == "degraded":
                degraded_count += 1
            else:
                unhealthy_count += 1

        # Overall status
        total_endpoints = len(endpoints)
        if total_endpoints == 0:
            overall_status = "unknown"
        elif unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # Active alerts
        active_alerts = [alert for alert in self.alerts.values() if alert.status == "active"]
        recent_alerts = sorted(
            self.alerts.values(),
            key=lambda x: x.triggered_at,
            reverse=True,
        )[:10]

        # Metrics summary
        total_requests = sum(m.total_requests for m in self.metrics.values())
        total_errors = sum(m.failed_requests for m in self.metrics.values())
        avg_error_rate = (total_errors / total_requests) if total_requests > 0 else 0
        avg_response_time = (
            sum(m.average_response_time for m in self.metrics.values()) / len(self.metrics)
            if self.metrics
            else 0
        )

        return HealthDashboard(
            overall_status=overall_status,
            total_endpoints=total_endpoints,
            healthy_endpoints=healthy_count,
            degraded_endpoints=degraded_count,
            unhealthy_endpoints=unhealthy_count,
            active_alerts=len(active_alerts),
            endpoints=endpoints,
            recent_alerts=recent_alerts,
            metrics_summary={
                "total_requests": total_requests,
                "total_errors": total_errors,
                "average_error_rate": avg_error_rate,
                "average_response_time": avg_response_time,
            },
            timestamp=datetime.utcnow().isoformat(),
        )


# Global health monitor instance
health_monitor = EndpointHealthMonitor()


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/health/dashboard", response_model=HealthDashboard, tags=["health", "monitoring"])
async def get_health_dashboard():
    """Get comprehensive health dashboard with all endpoint statuses."""
    try:
        return health_monitor.get_dashboard()

    except Exception as e:
        logger.exception(f"Failed to generate health dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate health dashboard",
        )


@router.get("/health/endpoints", tags=["health", "monitoring"])
async def list_monitored_endpoints():
    """List all monitored endpoints with their metrics."""
    try:
        endpoints = {}

        for key, metrics in health_monitor.metrics.items():
            method, endpoint = key.split(":", 1)
            endpoint_key = f"{method} {endpoint}"

            endpoints[endpoint_key] = {
                "endpoint": endpoint,
                "method": method,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "error_rate": metrics.error_rate,
                "average_response_time": metrics.average_response_time,
                "availability": metrics.availability,
                "current_status": metrics.current_status,
                "last_request_time": metrics.last_request_time,
            }

        return {
            "endpoints": endpoints,
            "total_count": len(endpoints),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to list endpoints: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve endpoint list",
        )


@router.post(
    "/health/check/{endpoint:path}",
    response_model=EndpointHealth,
    tags=["health", "monitoring"],
)
async def check_endpoint_health(
    endpoint: str,
    method: Annotated[str, Query(description="HTTP method to check")] = "GET",
):
    """Perform real-time health check for a specific endpoint."""
    try:
        # Ensure endpoint starts with /
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        return await health_monitor.health_check_endpoint(endpoint, method.upper())

    except Exception as e:
        logger.exception(f"Health check failed for {method} {endpoint}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {e!s}",
        )


@router.get("/health/alerts", tags=["health", "monitoring"])
async def get_alerts(
    status: Annotated[str | None, Query(description="Filter by alert status")] = None,
    limit: Annotated[int, Query(description="Maximum number of alerts to return")] = 50,
):
    """Get alerts with optional filtering."""
    try:
        alerts = list(health_monitor.alerts.values())

        # Filter by status if provided
        if status:
            alerts = [alert for alert in alerts if alert.status == status]

        # Sort by triggered time (most recent first)
        alerts = sorted(alerts, key=lambda x: x.triggered_at, reverse=True)

        # Apply limit
        alerts = alerts[:limit]

        return {
            "alerts": alerts,
            "total_count": len(health_monitor.alerts),
            "filtered_count": len(alerts),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts",
        )


@router.post("/health/monitoring/enable", tags=["health", "monitoring"])
async def enable_monitoring():
    """Enable endpoint health monitoring."""
    health_monitor.monitoring_enabled = True
    return {"message": "Health monitoring enabled", "enabled": True}


@router.post("/health/monitoring/disable", tags=["health", "monitoring"])
async def disable_monitoring():
    """Disable endpoint health monitoring."""
    health_monitor.monitoring_enabled = False
    return {"message": "Health monitoring disabled", "enabled": False}


@router.get("/health/monitoring/status", tags=["health", "monitoring"])
async def get_monitoring_status():
    """Get monitoring system status."""
    return {
        "enabled": health_monitor.monitoring_enabled,
        "monitored_endpoints": len(health_monitor.metrics),
        "active_alerts": len([a for a in health_monitor.alerts.values() if a.status == "active"]),
        "alert_rules": len(health_monitor.alert_rules),
        "check_interval": health_monitor.check_interval,
        "timestamp": datetime.utcnow().isoformat(),
    }
