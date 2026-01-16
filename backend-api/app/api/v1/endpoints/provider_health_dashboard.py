# =============================================================================
# Chimera - Provider Health Dashboard Endpoints
# =============================================================================
# Comprehensive REST API endpoints for provider health monitoring, quota tracking,
# rate limit visualization, and real-time health metrics.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# Subtask 2.4: Create comprehensive health dashboard API endpoints
# =============================================================================

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import TokenPayload, get_current_user
from app.domain.health_models import (
    AlertSeverity,
    HealthHistoryEntry,
    ProviderHealthDashboardResponse,
    ProviderHealthMetrics,
    ProviderQuotaStatus,
    QuotaDashboardResponse,
    QuotaPeriod,
    RateLimitDashboardResponse,
    RateLimitMetrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/providers/health", tags=["provider-health", "dashboard"])


# =============================================================================
# Response Models
# =============================================================================


class HealthDashboardResponse(BaseModel):
    """Full health dashboard response with all provider data."""

    status: str = Field(..., description="Overall system health status (healthy/degraded/critical/unknown)")
    providers: dict[str, Any] = Field(default_factory=dict, description="Health metrics per provider")
    summary: dict[str, Any] = Field(default_factory=dict, description="Health summary statistics")
    alerts: list[dict[str, Any]] = Field(default_factory=list, description="Active health alerts")
    monitoring: dict[str, Any] = Field(default_factory=dict, description="Monitoring configuration")
    updated_at: str = Field(..., description="Last dashboard update time (ISO format)")


class HealthMetricsResponse(BaseModel):
    """Real-time health metrics response."""

    providers: dict[str, Any] = Field(default_factory=dict, description="Real-time metrics per provider")
    summary: dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    updated_at: str = Field(..., description="Metrics timestamp (ISO format)")


class HealthHistoryResponse(BaseModel):
    """Health history response."""

    provider_id: str | None = Field(default=None, description="Provider filter (if specified)")
    entries: list[dict[str, Any]] = Field(default_factory=list, description="History entries")
    total_count: int = Field(default=0, ge=0, description="Total number of entries")
    start_time: str | None = Field(default=None, description="Query start time")
    end_time: str | None = Field(default=None, description="Query end time")


class QuotaDashboardDataResponse(BaseModel):
    """Quota dashboard data response."""

    providers: dict[str, Any] = Field(default_factory=dict, description="Quota status per provider")
    summary: dict[str, Any] = Field(default_factory=dict, description="Quota summary statistics")
    alerts: list[dict[str, Any]] = Field(default_factory=list, description="Quota-related alerts")
    updated_at: str = Field(..., description="Last update time (ISO format)")


class RateLimitDashboardDataResponse(BaseModel):
    """Rate limit dashboard data response."""

    providers: dict[str, Any] = Field(default_factory=dict, description="Rate limit metrics per provider")
    summary: dict[str, Any] = Field(default_factory=dict, description="Rate limit summary")
    updated_at: str = Field(..., description="Last update time (ISO format)")


class ProviderHealthDetailResponse(BaseModel):
    """Detailed health for a single provider."""

    provider_id: str = Field(..., description="Provider identifier")
    health: dict[str, Any] = Field(default_factory=dict, description="Health metrics")
    uptime: dict[str, Any] = Field(default_factory=dict, description="Uptime metrics")
    quota: dict[str, Any] | None = Field(default=None, description="Quota status")
    rate_limits: dict[str, Any] | None = Field(default=None, description="Rate limit metrics")
    alerts: list[dict[str, Any]] = Field(default_factory=list, description="Provider-specific alerts")
    updated_at: str = Field(..., description="Last update time (ISO format)")


class HealthCheckTriggerResponse(BaseModel):
    """Response for triggered health check."""

    success: bool = Field(..., description="Whether the check was triggered successfully")
    provider: str | None = Field(default=None, description="Provider checked (None = all)")
    result: dict[str, Any] = Field(default_factory=dict, description="Health check results")


class AlertAcknowledgeResponse(BaseModel):
    """Response for alert acknowledgment."""

    success: bool = Field(..., description="Whether the acknowledgment succeeded")
    alert_id: str = Field(..., description="Alert identifier")
    acknowledged_at: str = Field(..., description="Acknowledgment time (ISO format)")


class AlertResolveResponse(BaseModel):
    """Response for alert resolution."""

    success: bool = Field(..., description="Whether the resolution succeeded")
    alert_id: str = Field(..., description="Alert identifier")
    resolved_at: str = Field(..., description="Resolution time (ISO format)")


# =============================================================================
# Service Dependencies
# =============================================================================


def get_health_service():
    """Get the provider health monitoring service instance."""
    from app.services.provider_health_service import get_provider_health_service
    return get_provider_health_service()


def get_quota_service():
    """Get the quota tracking service instance."""
    from app.services.quota_tracking_service import get_quota_tracking_service
    return get_quota_tracking_service()


def get_rate_limiter():
    """Get the model rate limiter instance."""
    try:
        from app.services.model_rate_limiter import ModelRateLimiter
        return ModelRateLimiter()
    except ImportError:
        return None


# =============================================================================
# Dashboard Endpoints
# =============================================================================


@router.get(
    "/dashboard",
    response_model=HealthDashboardResponse,
    summary="Full health dashboard data",
    description="""
Get comprehensive health dashboard data for all configured LLM providers.

Returns:
- Per-provider health metrics (latency, error rates, uptime)
- Overall system health summary
- Active health alerts
- Monitoring configuration status

**Use Cases**:
- Populating the health dashboard UI
- Monitoring all providers at a glance
- Detecting system-wide issues
""",
    responses={
        200: {"description": "Health dashboard data retrieved successfully"},
        401: {"description": "Authentication required"},
        500: {"description": "Internal server error"},
    },
)
async def get_health_dashboard(
    user: TokenPayload = Depends(get_current_user),
) -> HealthDashboardResponse:
    """Get comprehensive health dashboard data."""
    logger.debug(f"Getting health dashboard for user {user.sub}")

    try:
        health_service = get_health_service()
        dashboard_data = await health_service.get_health_dashboard()

        return HealthDashboardResponse(
            status=dashboard_data.get("summary", {}).get("overall_status", "unknown"),
            providers=dashboard_data.get("providers", {}),
            summary=dashboard_data.get("summary", {}),
            alerts=dashboard_data.get("alerts", []),
            monitoring=dashboard_data.get("monitoring", {}),
            updated_at=dashboard_data.get("updated_at", datetime.utcnow().isoformat()),
        )

    except Exception as e:
        logger.error(f"Failed to get health dashboard: {e}", exc_info=True)
        return HealthDashboardResponse(
            status="unknown",
            providers={},
            summary={
                "total_providers": 0,
                "operational": 0,
                "degraded": 0,
                "down": 0,
                "unknown": 0,
                "overall_status": "unknown",
                "error": str(e),
            },
            alerts=[],
            monitoring={"running": False, "error": str(e)},
            updated_at=datetime.utcnow().isoformat(),
        )


@router.get(
    "/metrics",
    response_model=HealthMetricsResponse,
    summary="Real-time health metrics",
    description="""
Get real-time health metrics for all configured providers.

Returns lightweight metrics suitable for frequent polling:
- Current latency values
- Error rates
- Uptime percentages
- Circuit breaker states

**Use Cases**:
- Real-time dashboard updates
- Health status indicators
- Quick provider status checks
""",
    responses={
        200: {"description": "Health metrics retrieved successfully"},
        401: {"description": "Authentication required"},
    },
)
async def get_health_metrics(
    user: TokenPayload = Depends(get_current_user),
) -> HealthMetricsResponse:
    """Get real-time health metrics for all providers."""
    logger.debug(f"Getting health metrics for user {user.sub}")

    try:
        health_service = get_health_service()
        dashboard_data = await health_service.get_health_dashboard()

        # Extract just the metrics for lightweight response
        providers_metrics = {}
        for provider_id, provider_data in dashboard_data.get("providers", {}).items():
            providers_metrics[provider_id] = {
                "status": provider_data.get("status"),
                "latency_ms": provider_data.get("latency", {}).get("current_ms", 0),
                "error_rate": provider_data.get("requests", {}).get("error_rate_percent", 0),
                "uptime_percent": provider_data.get("availability", {}).get("uptime_percent", 100),
                "circuit_breaker": provider_data.get("circuit_breaker", {}).get("state", "unknown"),
                "last_check": provider_data.get("timestamps", {}).get("last_check"),
            }

        summary = dashboard_data.get("summary", {})

        return HealthMetricsResponse(
            providers=providers_metrics,
            summary={
                "total_providers": summary.get("total_providers", 0),
                "operational": summary.get("operational", 0),
                "degraded": summary.get("degraded", 0),
                "down": summary.get("down", 0),
                "overall_status": summary.get("overall_status", "unknown"),
            },
            updated_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get health metrics: {e}", exc_info=True)
        return HealthMetricsResponse(
            providers={},
            summary={"error": str(e)},
            updated_at=datetime.utcnow().isoformat(),
        )


@router.get(
    "/history",
    response_model=HealthHistoryResponse,
    summary="Historical health data",
    description="""
Get historical health performance data for trend analysis.

Query Parameters:
- `provider_id`: Filter by specific provider (optional)
- `limit`: Maximum entries to return (default: 100, max: 1000)
- `start_time`: Start of time range (ISO format, optional)
- `end_time`: End of time range (ISO format, optional)

**Use Cases**:
- Trend analysis and charting
- Historical performance review
- Incident investigation
""",
    responses={
        200: {"description": "Health history retrieved successfully"},
        401: {"description": "Authentication required"},
        400: {"description": "Invalid query parameters"},
    },
)
async def get_health_history(
    provider_id: str | None = Query(None, description="Filter by provider ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
    start_time: str | None = Query(None, description="Start time (ISO format)"),
    end_time: str | None = Query(None, description="End time (ISO format)"),
    user: TokenPayload = Depends(get_current_user),
) -> HealthHistoryResponse:
    """Get historical health performance data."""
    logger.debug(f"Getting health history for user {user.sub}, provider={provider_id}")

    try:
        # Parse time filters
        start_dt = None
        end_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid start_time format: {start_time}",
                )
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid end_time format: {end_time}",
                )

        health_service = get_health_service()
        history = await health_service.get_health_history(
            provider_id=provider_id,
            limit=limit,
            start_time=start_dt,
            end_time=end_dt,
        )

        return HealthHistoryResponse(
            provider_id=provider_id,
            entries=history,
            total_count=len(history),
            start_time=start_time,
            end_time=end_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health history: {e}", exc_info=True)
        return HealthHistoryResponse(
            provider_id=provider_id,
            entries=[],
            total_count=0,
            start_time=start_time,
            end_time=end_time,
        )


@router.get(
    "/{provider_id}",
    response_model=ProviderHealthDetailResponse,
    summary="Provider health details",
    description="""
Get detailed health information for a specific provider.

Returns comprehensive data including:
- Detailed health metrics
- Uptime across multiple time windows
- Current quota status
- Rate limit metrics
- Provider-specific alerts

**Use Cases**:
- Provider detail views
- Troubleshooting specific providers
- Detailed performance analysis
""",
    responses={
        200: {"description": "Provider health details retrieved"},
        401: {"description": "Authentication required"},
        404: {"description": "Provider not found"},
    },
)
async def get_provider_health_detail(
    provider_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> ProviderHealthDetailResponse:
    """Get detailed health information for a specific provider."""
    logger.debug(f"Getting health details for provider {provider_id} by user {user.sub}")

    try:
        health_service = get_health_service()
        quota_service = get_quota_service()

        # Get health metrics
        health_data = await health_service.get_provider_health(provider_id)
        if not health_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider not found: {provider_id}",
            )

        # Get uptime metrics
        uptime_metrics = await health_service.get_provider_uptime(provider_id)
        uptime_data = {}
        if uptime_metrics:
            uptime_data = {
                "current_status": uptime_metrics.current_status.value,
                "status_since": uptime_metrics.status_since.isoformat() if uptime_metrics.status_since else None,
                "last_hour": {
                    "uptime_percent": uptime_metrics.last_hour.uptime_percent if uptime_metrics.last_hour else None,
                    "incident_count": uptime_metrics.last_hour.incident_count if uptime_metrics.last_hour else 0,
                } if uptime_metrics.last_hour else None,
                "last_24_hours": {
                    "uptime_percent": uptime_metrics.last_24_hours.uptime_percent if uptime_metrics.last_24_hours else None,
                    "incident_count": uptime_metrics.last_24_hours.incident_count if uptime_metrics.last_24_hours else 0,
                } if uptime_metrics.last_24_hours else None,
                "last_7_days": {
                    "uptime_percent": uptime_metrics.last_7_days.uptime_percent if uptime_metrics.last_7_days else None,
                    "incident_count": uptime_metrics.last_7_days.incident_count if uptime_metrics.last_7_days else 0,
                } if uptime_metrics.last_7_days else None,
                "last_30_days": {
                    "uptime_percent": uptime_metrics.last_30_days.uptime_percent if uptime_metrics.last_30_days else None,
                    "incident_count": uptime_metrics.last_30_days.incident_count if uptime_metrics.last_30_days else 0,
                } if uptime_metrics.last_30_days else None,
                "all_time_uptime_percent": uptime_metrics.all_time_uptime_percent,
                "total_incidents": uptime_metrics.total_incidents,
            }

        # Get quota status
        quota_data = None
        try:
            quota_status = await quota_service.get_provider_quota(provider_id)
            if quota_status:
                quota_data = quota_status.to_dict()
        except Exception as e:
            logger.warning(f"Failed to get quota for provider {provider_id}: {e}")

        # Get provider-specific alerts
        alerts = await health_service.get_alerts(provider_id=provider_id, limit=20)

        return ProviderHealthDetailResponse(
            provider_id=provider_id,
            health=health_data,
            uptime=uptime_data,
            quota=quota_data,
            rate_limits=None,  # Rate limits are provider-level, not stored per provider yet
            alerts=alerts,
            updated_at=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get provider health details: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provider health details",
        )


# =============================================================================
# Quota Endpoints
# =============================================================================


@router.get(
    "/quota",
    response_model=QuotaDashboardDataResponse,
    summary="Current quota usage",
    description="""
Get current quota usage for all configured providers.

Returns:
- Per-provider quota status (requests, tokens, cost)
- Usage percentages and limits
- Alert thresholds and states
- Summary statistics

Query Parameters:
- `period`: Quota period (daily/monthly, default: daily)

**Use Cases**:
- Quota monitoring dashboard
- Budget tracking
- Cost optimization
""",
    responses={
        200: {"description": "Quota dashboard data retrieved"},
        401: {"description": "Authentication required"},
    },
)
async def get_quota_dashboard(
    period: QuotaPeriod = Query(QuotaPeriod.DAILY, description="Quota period (daily or monthly)"),
    user: TokenPayload = Depends(get_current_user),
) -> QuotaDashboardDataResponse:
    """Get current quota usage for all providers."""
    logger.debug(f"Getting quota dashboard for user {user.sub}, period={period}")

    try:
        quota_service = get_quota_service()
        quota_dashboard = await quota_service.get_quota_dashboard(period)

        # Convert to dict format
        providers_data = {}
        for provider_id, quota_status in quota_dashboard.providers.items():
            providers_data[provider_id] = quota_status.to_dict()

        alerts_data = [alert.to_dict() for alert in quota_dashboard.alerts]

        return QuotaDashboardDataResponse(
            providers=providers_data,
            summary=quota_dashboard.summary,
            alerts=alerts_data,
            updated_at=quota_dashboard.updated_at.isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get quota dashboard: {e}", exc_info=True)
        return QuotaDashboardDataResponse(
            providers={},
            summary={"error": str(e)},
            alerts=[],
            updated_at=datetime.utcnow().isoformat(),
        )


@router.get(
    "/quota/{provider_id}",
    response_model=dict[str, Any],
    summary="Provider quota status",
    description="""
Get quota status for a specific provider.

Returns detailed quota information including:
- Request and token usage
- Cost tracking
- Period information and reset time
- Alert thresholds

**Use Cases**:
- Provider-specific quota monitoring
- Checking remaining quota before requests
""",
    responses={
        200: {"description": "Provider quota status retrieved"},
        401: {"description": "Authentication required"},
        404: {"description": "Provider not found"},
    },
)
async def get_provider_quota(
    provider_id: str,
    period: QuotaPeriod = Query(QuotaPeriod.DAILY, description="Quota period"),
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Get quota status for a specific provider."""
    logger.debug(f"Getting quota for provider {provider_id} by user {user.sub}")

    try:
        quota_service = get_quota_service()
        quota_status = await quota_service.get_provider_quota(provider_id, period)

        if not quota_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Quota not configured for provider: {provider_id}",
            )

        return quota_status.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get provider quota: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provider quota",
        )


@router.get(
    "/quota/{provider_id}/remaining",
    response_model=dict[str, Any],
    summary="Remaining quota",
    description="""
Get remaining quota for a provider.

Returns:
- Remaining requests
- Remaining tokens
- Reset time

**Use Cases**:
- Pre-flight checks before making requests
- Displaying remaining capacity
""",
    responses={
        200: {"description": "Remaining quota retrieved"},
        401: {"description": "Authentication required"},
    },
)
async def get_remaining_quota(
    provider_id: str,
    period: QuotaPeriod = Query(QuotaPeriod.DAILY, description="Quota period"),
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Get remaining quota for a provider."""
    logger.debug(f"Getting remaining quota for provider {provider_id} by user {user.sub}")

    try:
        quota_service = get_quota_service()
        remaining = await quota_service.get_remaining_quota(provider_id, period)

        return {
            "provider_id": provider_id,
            "period": period.value,
            **remaining,
        }

    except Exception as e:
        logger.error(f"Failed to get remaining quota: {e}", exc_info=True)
        return {
            "provider_id": provider_id,
            "period": period.value,
            "remaining_requests": 0,
            "remaining_tokens": 0,
            "reset_at": None,
            "error": str(e),
        }


@router.get(
    "/quota/{provider_id}/history",
    response_model=list[dict[str, Any]],
    summary="Quota usage history",
    description="""
Get historical quota usage data for a provider.

Returns daily or monthly usage summaries for trend analysis.

**Use Cases**:
- Usage trend charts
- Cost analysis over time
""",
    responses={
        200: {"description": "Quota history retrieved"},
        401: {"description": "Authentication required"},
    },
)
async def get_quota_history(
    provider_id: str,
    period: QuotaPeriod = Query(QuotaPeriod.DAILY, description="Period type for history"),
    limit: int = Query(30, ge=1, le=365, description="Maximum entries to return"),
    user: TokenPayload = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Get historical quota usage for a provider."""
    logger.debug(f"Getting quota history for provider {provider_id} by user {user.sub}")

    try:
        quota_service = get_quota_service()
        history = await quota_service.get_usage_history(provider_id, period, limit)
        return history

    except Exception as e:
        logger.error(f"Failed to get quota history: {e}", exc_info=True)
        return []


# =============================================================================
# Rate Limit Endpoints
# =============================================================================


@router.get(
    "/rate-limits",
    response_model=RateLimitDashboardDataResponse,
    summary="Rate limit visualization data",
    description="""
Get rate limit metrics for all providers.

Returns:
- Current requests per minute vs. provider caps
- Token usage per minute
- Rate limit hit history
- Providers currently rate limited

**Use Cases**:
- Rate limit dashboard
- Visualizing rate limit utilization
- Identifying rate limit issues
""",
    responses={
        200: {"description": "Rate limit data retrieved"},
        401: {"description": "Authentication required"},
    },
)
async def get_rate_limits_dashboard(
    user: TokenPayload = Depends(get_current_user),
) -> RateLimitDashboardDataResponse:
    """Get rate limit visualization data for all providers."""
    logger.debug(f"Getting rate limits dashboard for user {user.sub}")

    try:
        health_service = get_health_service()
        dashboard_data = await health_service.get_health_dashboard()

        # Build rate limit data from provider health data
        providers_rate_limits = {}
        currently_rate_limited = 0
        approaching_limit = 0

        for provider_id, provider_data in dashboard_data.get("providers", {}).items():
            requests = provider_data.get("requests", {})
            rate_limited_count = requests.get("rate_limited_requests", 0) if isinstance(requests, dict) else 0

            # Create rate limit metrics from health data
            providers_rate_limits[provider_id] = {
                "provider_id": provider_id,
                "provider_name": provider_data.get("provider_name", provider_id.title()),
                "is_rate_limited": rate_limited_count > 0,
                "rate_limit_hits_last_hour": rate_limited_count,
                "requests_total": requests.get("total", 0) if isinstance(requests, dict) else 0,
                "status": provider_data.get("status", "unknown"),
            }

            if rate_limited_count > 0:
                currently_rate_limited += 1

        return RateLimitDashboardDataResponse(
            providers=providers_rate_limits,
            summary={
                "total_providers": len(providers_rate_limits),
                "currently_rate_limited": currently_rate_limited,
                "approaching_limit": approaching_limit,
            },
            updated_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get rate limits dashboard: {e}", exc_info=True)
        return RateLimitDashboardDataResponse(
            providers={},
            summary={"error": str(e)},
            updated_at=datetime.utcnow().isoformat(),
        )


# =============================================================================
# Alert Management Endpoints
# =============================================================================


@router.get(
    "/alerts",
    response_model=list[dict[str, Any]],
    summary="Health alerts",
    description="""
Get health alerts for all providers.

Query Parameters:
- `provider_id`: Filter by provider (optional)
- `severity`: Filter by severity (info/warning/critical, optional)
- `active_only`: Only return active alerts (default: false)
- `limit`: Maximum alerts to return (default: 50)

**Use Cases**:
- Alert dashboard
- Notification integration
- Incident tracking
""",
    responses={
        200: {"description": "Alerts retrieved"},
        401: {"description": "Authentication required"},
    },
)
async def get_health_alerts(
    provider_id: str | None = Query(None, description="Filter by provider ID"),
    severity: AlertSeverity | None = Query(None, description="Filter by severity"),
    active_only: bool = Query(False, description="Only return active alerts"),
    limit: int = Query(50, ge=1, le=200, description="Maximum alerts to return"),
    user: TokenPayload = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Get health alerts."""
    logger.debug(f"Getting health alerts for user {user.sub}")

    try:
        health_service = get_health_service()
        alerts = await health_service.get_alerts(
            provider_id=provider_id,
            severity=severity,
            active_only=active_only,
            limit=limit,
        )
        return alerts

    except Exception as e:
        logger.error(f"Failed to get health alerts: {e}", exc_info=True)
        return []


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertAcknowledgeResponse,
    summary="Acknowledge alert",
    description="""
Acknowledge a health alert.

Marks the alert as acknowledged but keeps it active until resolved.

**Use Cases**:
- Incident management
- Team collaboration
""",
    responses={
        200: {"description": "Alert acknowledged"},
        401: {"description": "Authentication required"},
        404: {"description": "Alert not found"},
    },
)
async def acknowledge_alert(
    alert_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> AlertAcknowledgeResponse:
    """Acknowledge a health alert."""
    logger.info(f"Acknowledging alert {alert_id} by user {user.sub}")

    try:
        health_service = get_health_service()
        acknowledged_by = user.sub or "system"
        success = await health_service.acknowledge_alert(alert_id, acknowledged_by)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {alert_id}",
            )

        return AlertAcknowledgeResponse(
            success=True,
            alert_id=alert_id,
            acknowledged_at=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert",
        )


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertResolveResponse,
    summary="Resolve alert",
    description="""
Resolve a health alert.

Marks the alert as resolved and inactive.

**Use Cases**:
- Closing resolved incidents
- Alert cleanup
""",
    responses={
        200: {"description": "Alert resolved"},
        401: {"description": "Authentication required"},
        404: {"description": "Alert not found"},
    },
)
async def resolve_alert(
    alert_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> AlertResolveResponse:
    """Resolve a health alert."""
    logger.info(f"Resolving alert {alert_id} by user {user.sub}")

    try:
        health_service = get_health_service()
        success = await health_service.resolve_alert(alert_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {alert_id}",
            )

        return AlertResolveResponse(
            success=True,
            alert_id=alert_id,
            resolved_at=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve alert",
        )


# =============================================================================
# Health Check Trigger Endpoints
# =============================================================================


@router.post(
    "/check",
    response_model=HealthCheckTriggerResponse,
    summary="Trigger health check",
    description="""
Trigger an immediate health check for all providers or a specific provider.

Query Parameters:
- `provider_id`: Specific provider to check (optional, None = all)

**Use Cases**:
- Manual health verification
- Testing after configuration changes
- On-demand monitoring
""",
    responses={
        200: {"description": "Health check triggered"},
        401: {"description": "Authentication required"},
    },
)
async def trigger_health_check(
    provider_id: str | None = Query(None, description="Specific provider to check"),
    user: TokenPayload = Depends(get_current_user),
) -> HealthCheckTriggerResponse:
    """Trigger an immediate health check."""
    logger.info(f"Triggering health check for provider={provider_id} by user {user.sub}")

    try:
        health_service = get_health_service()
        result = await health_service.check_now(provider_id)

        return HealthCheckTriggerResponse(
            success=True,
            provider=provider_id,
            result=result,
        )

    except Exception as e:
        logger.error(f"Failed to trigger health check: {e}", exc_info=True)
        return HealthCheckTriggerResponse(
            success=False,
            provider=provider_id,
            result={"error": str(e)},
        )


# =============================================================================
# Usage Summary Endpoint
# =============================================================================


@router.get(
    "/usage-summary",
    response_model=dict[str, Any],
    summary="Usage summary",
    description="""
Get a summary of usage across all providers.

Returns aggregated usage data including:
- Total requests and tokens
- Cost summary
- Per-provider breakdown

**Use Cases**:
- Executive dashboards
- Cost reporting
- Usage overview
""",
    responses={
        200: {"description": "Usage summary retrieved"},
        401: {"description": "Authentication required"},
    },
)
async def get_usage_summary(
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Get usage summary across all providers."""
    logger.debug(f"Getting usage summary for user {user.sub}")

    try:
        quota_service = get_quota_service()
        summary = await quota_service.get_usage_summary()
        return summary

    except Exception as e:
        logger.error(f"Failed to get usage summary: {e}", exc_info=True)
        return {
            "daily": {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            },
            "by_provider": {},
            "error": str(e),
            "updated_at": datetime.utcnow().isoformat(),
        }
