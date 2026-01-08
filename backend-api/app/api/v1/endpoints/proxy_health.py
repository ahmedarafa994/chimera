"""
Proxy Health API Endpoint.

STORY-1.3: Provides health check endpoints for the proxy server.

Endpoints:
- GET /health/proxy - Get proxy server health status
- POST /health/proxy/check - Trigger immediate health check
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/health/proxy", tags=["proxy-health"])


class ProxyHealthResponse(BaseModel):
    """Response model for proxy health check."""

    is_healthy: bool
    proxy_enabled: bool
    endpoint: str
    connection_state: str
    latency_ms: float | None = None
    consecutive_failures: int = 0
    error: str | None = None
    fallback_enabled: bool = True


class ProxyHealthDetailResponse(BaseModel):
    """Detailed proxy health status response."""

    is_healthy: bool
    monitoring_active: bool
    check_interval_seconds: int
    consecutive_failures: int
    unhealthy_threshold: int
    metrics: dict
    client: dict
    history_size: int
    recent_checks: list


@router.get(
    "",
    response_model=ProxyHealthResponse,
    summary="Get proxy health status",
    description="Returns the current health status of the proxy server.",
)
async def get_proxy_health() -> ProxyHealthResponse:
    """
    Get the current health status of the proxy server.

    Returns basic health information including connection state,
    latency, and error information if any.
    """
    # Check if proxy mode is enabled
    proxy_enabled = settings.PROXY_MODE_ENABLED

    if not proxy_enabled:
        return ProxyHealthResponse(
            is_healthy=False,
            proxy_enabled=False,
            endpoint=settings.PROXY_MODE_ENDPOINT,
            connection_state="disabled",
            error="Proxy mode is not enabled",
            fallback_enabled=settings.PROXY_MODE_FALLBACK_TO_DIRECT,
        )

    try:
        from app.infrastructure.proxy.proxy_client import get_proxy_client

        client = get_proxy_client()
        status = await client.check_health()

        return ProxyHealthResponse(
            is_healthy=status.is_healthy,
            proxy_enabled=True,
            endpoint=client.endpoint,
            connection_state=status.connection_state.value,
            latency_ms=status.latency_ms if status.is_healthy else None,
            consecutive_failures=status.consecutive_failures,
            error=status.error,
            fallback_enabled=settings.PROXY_MODE_FALLBACK_TO_DIRECT,
        )

    except Exception as e:
        logger.error(f"Proxy health check failed: {e}")
        return ProxyHealthResponse(
            is_healthy=False,
            proxy_enabled=True,
            endpoint=settings.PROXY_MODE_ENDPOINT,
            connection_state="error",
            error=str(e),
            fallback_enabled=settings.PROXY_MODE_FALLBACK_TO_DIRECT,
        )


@router.post(
    "/check",
    response_model=ProxyHealthResponse,
    summary="Trigger immediate health check",
    description="Triggers an immediate health check of the proxy server.",
)
async def trigger_health_check() -> ProxyHealthResponse:
    """
    Trigger an immediate health check of the proxy server.

    Unlike the GET endpoint which may return cached status,
    this endpoint always performs a fresh health check.
    """
    if not settings.PROXY_MODE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Proxy mode is not enabled",
        )

    try:
        from app.infrastructure.proxy.proxy_client import get_proxy_client

        client = get_proxy_client()
        health_status = await client.check_health()

        return ProxyHealthResponse(
            is_healthy=health_status.is_healthy,
            proxy_enabled=True,
            endpoint=client.endpoint,
            connection_state=health_status.connection_state.value,
            latency_ms=health_status.latency_ms,
            consecutive_failures=health_status.consecutive_failures,
            error=health_status.error,
            fallback_enabled=settings.PROXY_MODE_FALLBACK_TO_DIRECT,
        )

    except Exception as e:
        logger.error(f"Proxy health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Proxy health check failed: {e}",
        )


@router.get(
    "/detailed",
    response_model=ProxyHealthDetailResponse,
    summary="Get detailed proxy health status",
    description="Returns detailed health metrics including history.",
)
async def get_detailed_proxy_health() -> ProxyHealthDetailResponse:
    """
    Get detailed proxy health status with metrics and history.

    Includes aggregated metrics, recent check history, and
    comprehensive client statistics.
    """
    if not settings.PROXY_MODE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Proxy mode is not enabled",
        )

    try:
        from app.infrastructure.proxy.proxy_health import get_health_monitor

        monitor = get_health_monitor()
        status_data = monitor.get_status()

        return ProxyHealthDetailResponse(
            is_healthy=status_data["is_healthy"],
            monitoring_active=status_data["monitoring_active"],
            check_interval_seconds=status_data["check_interval_seconds"],
            consecutive_failures=status_data["consecutive_failures"],
            unhealthy_threshold=status_data["unhealthy_threshold"],
            metrics=status_data["metrics"],
            client=status_data["client"],
            history_size=status_data["history_size"],
            recent_checks=status_data["recent_checks"],
        )

    except Exception as e:
        logger.error(f"Failed to get detailed proxy health: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get proxy health details: {e}",
        )


@router.get(
    "/stats",
    summary="Get proxy client statistics",
    description="Returns proxy client statistics including request counts.",
)
async def get_proxy_stats() -> dict:
    """
    Get proxy client statistics.

    Returns request counts, error rates, and connection information.
    """
    if not settings.PROXY_MODE_ENABLED:
        return {
            "proxy_enabled": False,
            "message": "Proxy mode is not enabled",
        }

    try:
        from app.infrastructure.proxy.proxy_client import get_proxy_client

        client = get_proxy_client()
        stats = client.get_stats()

        return {
            "proxy_enabled": True,
            **stats,
        }

    except Exception as e:
        logger.error(f"Failed to get proxy stats: {e}")
        return {
            "proxy_enabled": True,
            "error": str(e),
        }
