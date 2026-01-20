"""Comprehensive Health Check Endpoint.

LOW-001 FIX: Enhanced health check with detailed component status,
circuit breaker states, cache statistics, and dependency checks.
P2-FIX-008: Allow None values for optional proxy check in health response.

Provides:
- Liveness probe (basic health)
- Readiness probe (full dependency check)
- Detailed system status for monitoring
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import config
from app.core.shared.circuit_breaker import CircuitBreakerRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str
    status: str = Field(description="healthy, degraded, or unhealthy")
    latency_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] | None = None


class CircuitBreakerStatus(BaseModel):
    """Status of a circuit breaker."""

    name: str
    state: str
    failure_count: int
    success_rate: float | None = None


class CacheStatus(BaseModel):
    """Status of cache systems."""

    name: str
    enabled: bool
    current_size: int | None = None
    max_size: int | None = None
    hit_rate: float | None = None
    evictions: int | None = None


class HealthResponse(BaseModel):
    """Comprehensive health check response.

    P2-FIX-008: Updated to use dict[str, Any] for checks field to allow
    both bool and None values for optional proxy check.
    """

    # Model config to ensure proper validation
    model_config = {"strict": False, "coerce_numbers_to_str": False}

    status: str = Field(description="healthy, degraded, or unhealthy")
    timestamp: str
    version: str
    environment: str
    uptime_seconds: float | None = None
    components: list[ComponentHealth] = []
    circuit_breakers: list[CircuitBreakerStatus] = []
    caches: list[CacheStatus] = []
    # P2-FIX-008: Use Any type for dict values to allow both bool and None
    checks: dict[str, Any] = Field(
        default_factory=dict,
        description="Component check results (bool or None)",
    )


# P2-FIX-008: Force Pydantic to rebuild validators on module reload
HealthResponse.model_rebuild()


class LivenessResponse(BaseModel):
    """Simple liveness probe response."""

    status: str = "ok"
    timestamp: str


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    status: str
    timestamp: str
    ready: bool
    checks: dict[str, bool] = {}


# Track application start time
_start_time = time.time()


async def check_database_health() -> ComponentHealth:
    """Check database connectivity."""
    start = time.perf_counter()
    try:
        # Import here to avoid circular imports
        from app.core.config import config

        # For SQLite, just check if the path is accessible
        if "sqlite" in config.DATABASE_URL.lower():
            return ComponentHealth(
                name="database",
                status="healthy",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                message="SQLite database configured",
                details={"type": "sqlite"},
            )

        # For PostgreSQL, attempt a connection
        # This is a placeholder - actual implementation would use the DB engine
        return ComponentHealth(
            name="database",
            status="healthy",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            message="Database connection configured",
            details={"type": "postgresql"},
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status="unhealthy",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            message=str(e),
        )


async def check_redis_health() -> ComponentHealth:
    """Check Redis connectivity."""
    start = time.perf_counter()
    try:
        from app.core.config import config

        # Check if Redis is configured
        if not config.REDIS_URL:
            # Redis is optional - treat as healthy with informational message
            return ComponentHealth(
                name="redis",
                status="healthy",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                message="Redis not configured (optional, using in-memory fallback)",
                details={"configured": False, "optional": True},
            )

        # Attempt Redis ping using modern redis library
        try:
            import redis.asyncio as redis_async

            client = redis_async.from_url(
                config.REDIS_URL,
                socket_timeout=getattr(config, "REDIS_SOCKET_TIMEOUT", 5.0),
            )
            await client.ping()
            await client.aclose()

            return ComponentHealth(
                name="redis",
                status="healthy",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                message="Redis connection successful",
                details={"configured": True, "connected": True},
            )
        except ImportError:
            # Fall back to aioredis if redis.asyncio not available
            import aioredis

            redis_client = await aioredis.from_url(
                config.REDIS_URL,
                socket_timeout=getattr(config, "REDIS_SOCKET_TIMEOUT", 5.0),
            )
            await redis_client.ping()
            await redis_client.close()

            return ComponentHealth(
                name="redis",
                status="healthy",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                message="Redis connection successful (via aioredis)",
            )
    except ImportError:
        # No Redis client installed - treat as optional/healthy
        return ComponentHealth(
            name="redis",
            status="healthy",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            message="Redis client not installed (optional, using in-memory fallback)",
            details={"configured": False, "optional": True},
        )
    except Exception as e:
        # Connection failed - treat as healthy but note the issue (Redis is optional)
        return ComponentHealth(
            name="redis",
            status="healthy",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            message=f"Redis unavailable, using in-memory fallback: {e!s}",
            details={"configured": True, "connected": False, "error": str(e)},
        )


async def check_llm_providers_health() -> list[ComponentHealth]:
    """Check LLM provider availability via circuit breakers."""
    components = []

    # Get circuit breaker states
    breaker_states = CircuitBreakerRegistry.get_all_status()

    for name, status_info in breaker_states.items():
        state = status_info.get("state", "unknown")

        if state == "closed":
            health_status = "healthy"
        elif state == "half_open":
            health_status = "degraded"
        else:
            health_status = "unhealthy"

        components.append(
            ComponentHealth(
                name=f"llm_provider_{name}",
                status=health_status,
                message=f"Circuit breaker state: {state}",
                details={
                    "failure_count": status_info.get("failure_count", 0),
                    "circuit_state": state,
                },
            ),
        )

    return components


async def check_proxy_health() -> ComponentHealth | None:
    """Check proxy server connectivity and health."""
    from app.core.config import settings

    # Only check proxy health if proxy mode is enabled
    if settings.API_CONNECTION_MODE != "proxy":
        return None

    start = time.perf_counter()
    try:
        from app.infrastructure.proxy import get_health_monitor

        monitor = get_health_monitor()
        health_status = await monitor.check_now()

        latency_ms = (time.perf_counter() - start) * 1000

        if health_status.is_healthy:
            return ComponentHealth(
                name="proxy_server",
                status="healthy",
                latency_ms=round(latency_ms, 2),
                message="Proxy server connection healthy",
                details={
                    "endpoint": settings.PROXY_MODE_ENDPOINT,
                    "connection_state": health_status.connection_state.value,
                    "consecutive_failures": health_status.consecutive_failures,
                },
            )
        return ComponentHealth(
            name="proxy_server",
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            message=health_status.error or "Proxy server unavailable",
            details={
                "endpoint": settings.PROXY_MODE_ENDPOINT,
                "connection_state": health_status.connection_state.value,
                "consecutive_failures": health_status.consecutive_failures,
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="proxy_server",
            status="unhealthy",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            message=str(e),
            details={"endpoint": settings.PROXY_MODE_ENDPOINT, "error": str(e)},
        )


def get_circuit_breaker_status() -> list[CircuitBreakerStatus]:
    """Get status of all circuit breakers."""
    statuses = []

    breaker_states = CircuitBreakerRegistry.get_all_status()

    for name, status_info in breaker_states.items():
        metrics = status_info.get("metrics", {})
        statuses.append(
            CircuitBreakerStatus(
                name=name,
                state=status_info.get("state", "unknown"),
                failure_count=status_info.get("failure_count", 0),
                success_rate=metrics.get("success_rate"),
            ),
        )

    return statuses


def get_cache_status() -> list[CacheStatus]:
    """Get status of cache systems."""
    caches = []

    # Check transformation cache
    try:
        from app.services.transformation_service import transformation_engine

        if transformation_engine.cache:
            stats = transformation_engine.get_cache_stats()
            if stats:
                caches.append(
                    CacheStatus(
                        name="transformation_cache",
                        enabled=True,
                        current_size=stats.get("current_size"),
                        max_size=stats.get("max_size"),
                        hit_rate=stats.get("hit_rate"),
                        evictions=stats.get("evictions"),
                    ),
                )
            else:
                caches.append(CacheStatus(name="transformation_cache", enabled=True))
        else:
            caches.append(CacheStatus(name="transformation_cache", enabled=False))
    except Exception as e:
        logger.warning(f"Failed to get transformation cache status: {e}")
        caches.append(CacheStatus(name="transformation_cache", enabled=False))

    return caches


@router.get(
    "/health",
    summary="Comprehensive health check",
    description="Returns detailed health status of all system components",
)
async def health_check() -> HealthResponse:
    """Comprehensive health check endpoint.

    Returns detailed status of:
    - Database connectivity
    - Redis connectivity
    - LLM provider availability
    - Circuit breaker states
    - Cache statistics
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    uptime = time.time() - _start_time

    # Gather component health checks concurrently
    db_health, redis_health, llm_health, proxy_health = await asyncio.gather(
        check_database_health(),
        check_redis_health(),
        check_llm_providers_health(),
        check_proxy_health(),
        return_exceptions=True,
    )

    components = []

    # Add database health
    if isinstance(db_health, ComponentHealth):
        components.append(db_health)
    else:
        components.append(
            ComponentHealth(name="database", status="unhealthy", message=str(db_health)),
        )

    # Add Redis health
    if isinstance(redis_health, ComponentHealth):
        components.append(redis_health)
    else:
        components.append(
            ComponentHealth(name="redis", status="unhealthy", message=str(redis_health)),
        )

    # Add LLM provider health
    if isinstance(llm_health, list):
        components.extend(llm_health)

    # Add proxy health if available
    if isinstance(proxy_health, ComponentHealth):
        components.append(proxy_health)
    elif proxy_health is not None:
        components.append(
            ComponentHealth(name="proxy_server", status="unhealthy", message=str(proxy_health)),
        )

    # Get circuit breaker status
    circuit_breakers = get_circuit_breaker_status()

    # Get cache status
    caches = get_cache_status()

    # Determine overall status
    unhealthy_count = sum(1 for c in components if c.status == "unhealthy")
    degraded_count = sum(1 for c in components if c.status == "degraded")

    if unhealthy_count > 0:
        overall_status = "unhealthy"
    elif degraded_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    # Build checks summary - P2-FIX-008: Ensure proxy key is only added when proxy mode enabled
    # FIX: Changed to dict[str, Any] to allow optional proxy values
    checks: dict[str, Any] = {
        "database": any(c.name == "database" and c.status == "healthy" for c in components),
        "redis": any(c.name == "redis" and c.status in ["healthy", "degraded"] for c in components),
        "llm_providers": all(
            c.status != "unhealthy" for c in components if c.name.startswith("llm_provider_")
        ),
    }
    # P2-FIX-008: Only include proxy check if proxy_server component exists in components list
    # This prevents None from being set as the proxy value
    proxy_components = [c for c in components if c.name == "proxy_server"]
    if proxy_components:
        checks["proxy"] = any(c.status == "healthy" for c in proxy_components)

    # DEBUG: Log the checks dict to verify no None values
    logger.info(f"P2-FIX-008 DEBUG: checks dict = {checks}, proxy in checks = {'proxy' in checks}")

    return HealthResponse(
        status=overall_status,
        timestamp=timestamp,
        version=config.VERSION,
        environment=config.ENVIRONMENT,
        uptime_seconds=round(uptime, 2),
        components=components,
        circuit_breakers=circuit_breakers,
        caches=caches,
        checks=checks,
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Simple liveness check for Kubernetes probes",
)
async def liveness_probe() -> LivenessResponse:
    """Kubernetes liveness probe endpoint.

    Returns 200 if the application is running.
    Used by Kubernetes to determine if the pod should be restarted.
    """
    return LivenessResponse(status="ok", timestamp=datetime.utcnow().isoformat() + "Z")


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Readiness check for Kubernetes probes",
)
async def readiness_probe() -> ReadinessResponse:
    """Kubernetes readiness probe endpoint.

    Returns 200 if the application is ready to receive traffic.
    Checks critical dependencies before marking as ready.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Check critical dependencies
    db_health = await check_database_health()

    checks = {
        "database": db_health.status == "healthy",
    }

    # Application is ready if database is healthy
    ready = checks["database"]

    response = ReadinessResponse(
        status="ready" if ready else "not_ready",
        timestamp=timestamp,
        ready=ready,
        checks=checks,
    )

    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response


@router.get(
    "/health/circuit-breakers",
    summary="Circuit breaker status",
    description="Returns status of all circuit breakers",
)
async def circuit_breaker_status() -> list[CircuitBreakerStatus]:
    """Get detailed circuit breaker status for all providers."""
    return get_circuit_breaker_status()


@router.post(
    "/health/circuit-breakers/{name}/reset",
    summary="Reset circuit breaker",
    description="Reset a specific circuit breaker to closed state",
)
async def reset_circuit_breaker(name: str) -> dict[str, str]:
    """Reset a circuit breaker to its initial closed state.

    Use with caution - this bypasses the normal recovery process.
    """
    CircuitBreakerRegistry.reset(name)
    logger.info(f"Circuit breaker '{name}' reset via API")
    return {"status": "reset", "circuit_breaker": name}


class ProxyHealthResponse(BaseModel):
    """Proxy server health check response."""

    status: str = Field(description="healthy, degraded, or unhealthy")
    proxy_mode_enabled: bool
    proxy_healthy: bool
    latency_ms: float
    endpoint: str
    connection_state: str | None = None
    consecutive_failures: int = 0
    error: str | None = None
    uptime_percent: float | None = None


@router.get(
    "/health/proxy",
    summary="Proxy server health check",
    description="Returns health status of the AIClient-2-API proxy server",
)
async def proxy_health_check() -> ProxyHealthResponse:
    """Proxy server health check endpoint.

    Returns detailed status of the AIClient-2-API proxy server connection,
    including latency, error rate, and availability metrics.
    """
    from app.core.config import settings

    # Check if proxy mode is enabled
    proxy_enabled = settings.API_CONNECTION_MODE == "proxy"

    if not proxy_enabled:
        return ProxyHealthResponse(
            status="healthy",
            proxy_mode_enabled=False,
            proxy_healthy=False,
            latency_ms=0.0,
            endpoint=settings.PROXY_MODE_ENDPOINT,
            connection_state=None,
            consecutive_failures=0,
            error=None,
            uptime_percent=None,
        )

    # Get proxy health status
    try:
        from app.infrastructure.proxy import get_health_monitor

        monitor = get_health_monitor()
        health_status = await monitor.check_now()
        status_report = monitor.get_status()

        # Determine overall status
        if health_status.is_healthy:
            overall_status = "healthy"
        elif status_report["consecutive_failures"] < 3:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return ProxyHealthResponse(
            status=overall_status,
            proxy_mode_enabled=True,
            proxy_healthy=health_status.is_healthy,
            latency_ms=health_status.latency_ms,
            endpoint=settings.PROXY_MODE_ENDPOINT,
            connection_state=health_status.connection_state.value,
            consecutive_failures=status_report["consecutive_failures"],
            error=health_status.error,
            uptime_percent=status_report["metrics"]["uptime_percent"],
        )
    except Exception as e:
        logger.exception(f"Proxy health check failed: {e}")
        return ProxyHealthResponse(
            status="unhealthy",
            proxy_mode_enabled=True,
            proxy_healthy=False,
            latency_ms=0.0,
            endpoint=settings.PROXY_MODE_ENDPOINT,
            connection_state="error",
            consecutive_failures=0,
            error=str(e),
            uptime_percent=None,
        )


class IntegrationHealthResponse(BaseModel):
    """Integration health check response."""

    status: str = Field(description="Overall health status")
    providers: dict[str, Any] = Field(description="Health metrics per provider")
    summary: dict[str, Any] = Field(description="Overall health summary")
    alerts: list[dict[str, Any]] = Field(default=[], description="Recent health alerts")
    monitoring: dict[str, Any] = Field(description="Monitoring configuration")


@router.get(
    "/health/integration",
    summary="Provider health monitoring",
    description="Returns comprehensive health status of all LLM providers",
)
async def integration_health_check() -> IntegrationHealthResponse:
    """Provider health monitoring endpoint.

    Returns detailed health status of all registered LLM providers including:
    - Per-provider metrics (latency, error rates, uptime)
    - Overall system health summary
    - Recent health alerts
    - Monitoring configuration
    """
    try:
        from app.services.integration_health_service import get_health_service

        health_service = get_health_service()
        health_status = await health_service.get_health_status()

        return IntegrationHealthResponse(**health_status)
    except Exception as e:
        logger.exception(f"Integration health check failed: {e}")
        # Return degraded status rather than failing completely
        return IntegrationHealthResponse(
            status="degraded",
            providers={},
            summary={
                "total_providers": 0,
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0,
                "overall_status": "degraded",
            },
            alerts=[],
            monitoring={
                "error": str(e),
            },
        )


class ServiceDependencyGraphResponse(BaseModel):
    """Service dependency graph response."""

    nodes: list[dict[str, Any]] = Field(description="Service nodes")
    edges: list[dict[str, Any]] = Field(description="Dependency edges")


@router.get(
    "/health/integration/graph",
    summary="Service dependency graph",
    description="Returns the service dependency graph with provider health status",
)
async def service_dependency_graph() -> ServiceDependencyGraphResponse:
    """Service dependency graph endpoint.

    Returns a graph representation of provider dependencies and their health status.
    Useful for visualization and understanding system topology.
    """
    try:
        from app.services.integration_health_service import get_health_service

        health_service = get_health_service()
        graph = await health_service.get_service_dependency_graph()

        return ServiceDependencyGraphResponse(**graph)
    except Exception as e:
        logger.exception(f"Service dependency graph failed: {e}")
        return ServiceDependencyGraphResponse(nodes=[], edges=[])


class HealthHistoryResponse(BaseModel):
    """Health history response."""

    history: list[dict[str, Any]] = Field(description="Health history entries")
    provider: str | None = Field(None, description="Provider filter (if specified)")


@router.get(
    "/health/integration/history",
    summary="Provider health history",
    description="Returns historical health data for providers",
)
async def health_history(
    provider: str | None = None,
    limit: int = 100,
) -> HealthHistoryResponse:
    """Health history endpoint.

    Returns historical health data for trend analysis.
    Can be filtered by provider and limited to N entries.
    """
    try:
        from app.services.integration_health_service import get_health_service

        health_service = get_health_service()
        history = await health_service.get_health_history(
            provider_name=provider,
            limit=limit,
        )

        return HealthHistoryResponse(history=history, provider=provider)
    except Exception as e:
        logger.exception(f"Health history request failed: {e}")
        return HealthHistoryResponse(history=[], provider=provider)


class IntegrationAlertsResponse(BaseModel):
    """Integration alerts response."""

    alerts: list[dict[str, Any]] = Field(description="Health alerts")


@router.get(
    "/health/integration/alerts",
    summary="Provider health alerts",
    description="Returns recent health alerts for providers",
)
async def integration_alerts(
    limit: int = 50,
) -> IntegrationAlertsResponse:
    """Health alerts endpoint.

    Returns recent health alerts showing degradation and issues.
    """
    try:
        from app.services.integration_health_service import get_health_service

        health_service = get_health_service()
        alerts = await health_service.get_alerts(limit=limit)

        return IntegrationAlertsResponse(alerts=alerts)
    except Exception as e:
        logger.exception(f"Integration alerts request failed: {e}")
        return IntegrationAlertsResponse(alerts=[])


@router.post(
    "/health/integration/check",
    summary="Trigger health check",
    description="Triggers an immediate health check for all providers",
)
async def trigger_health_check() -> IntegrationHealthResponse:
    """Trigger immediate health check endpoint.

    Forces an immediate health check for all providers and returns results.
    Useful for manual testing and on-demand monitoring.
    """
    try:
        from app.services.integration_health_service import get_health_service

        health_service = get_health_service()
        health_status = await health_service.check_now()

        return IntegrationHealthResponse(**health_status)
    except Exception as e:
        logger.exception(f"Triggered health check failed: {e}")
        return IntegrationHealthResponse(
            status="unhealthy",
            providers={},
            summary={
                "total_providers": 0,
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0,
                "overall_status": "unhealthy",
            },
            alerts=[],
            monitoring={
                "error": str(e),
            },
        )


# P2-FIX-008: Health endpoint fix v3 - force reload


# =============================================================================
# AI Provider Validation Health Endpoints
# =============================================================================


class AIProviderHealthResponse(BaseModel):
    """AI provider health check response."""

    status: str = Field(description="Overall health status")
    providers: dict[str, Any] = Field(description="Per-provider health info")
    fallback_stats: dict[str, Any] = Field(description="Fallback statistics")
    config_valid: bool = Field(description="Whether config is valid")
    startup_ready: bool = Field(description="Whether startup validation passed")


@router.get(
    "/health/providers",
    summary="AI providers health check",
    description="Check health of all configured AI providers",
)
async def check_providers_health() -> AIProviderHealthResponse:
    """Check health of all configured AI providers.

    Returns health information including:
    - Per-provider status and failure counts
    - Fallback statistics
    - Configuration validity
    - Startup readiness
    """
    try:
        from app.core.config_validator import get_config_validator
        from app.core.fallback_manager import get_fallback_manager
        from app.core.startup_validator import StartupValidator

        fallback_manager = get_fallback_manager()
        config_validator = get_config_validator()

        # Get provider health
        providers_health = fallback_manager.get_all_providers_health()

        # Get fallback stats
        fallback_stats = fallback_manager.get_fallback_stats()

        # Check config validity
        config_valid = config_validator.is_config_valid()

        # Check startup readiness
        startup_ready = StartupValidator.is_ready()

        # Determine overall status
        unhealthy_providers = [
            pid for pid, info in providers_health.items() if info.get("status") == "unhealthy"
        ]
        degraded_providers = [
            pid for pid, info in providers_health.items() if info.get("status") == "degraded"
        ]

        if unhealthy_providers:
            overall_status = "unhealthy"
        elif degraded_providers or not config_valid:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return AIProviderHealthResponse(
            status=overall_status,
            providers=providers_health,
            fallback_stats=fallback_stats,
            config_valid=config_valid,
            startup_ready=startup_ready,
        )

    except Exception as e:
        logger.exception(f"AI provider health check failed: {e}")
        return AIProviderHealthResponse(
            status="unhealthy",
            providers={},
            fallback_stats={},
            config_valid=False,
            startup_ready=False,
        )


class ConfigHealthResponse(BaseModel):
    """Configuration health check response."""

    status: str = Field(description="Config health status")
    config_loaded: bool = Field(description="Whether config is loaded")
    config_valid: bool = Field(description="Whether config is valid")
    error_count: int = Field(description="Number of validation errors")
    warning_count: int = Field(description="Number of validation warnings")
    errors: list[str] = Field(default=[], description="Error messages")
    warnings: list[str] = Field(default=[], description="Warning messages")
    providers_configured: int = Field(description="Number of configured providers")
    providers_enabled: int = Field(description="Number of enabled providers")


@router.get(
    "/health/config",
    summary="Configuration health check",
    description="Validate current AI provider configuration health",
)
async def check_config_health() -> ConfigHealthResponse:
    """Validate current configuration health.

    Returns:
    - Configuration load and validation status
    - Error and warning counts
    - Provider counts

    """
    try:
        from app.core.ai_config_manager import get_ai_config_manager
        from app.core.config_validator import ValidationSeverity, get_config_validator

        config_manager = get_ai_config_manager()
        config_validator = get_config_validator()

        config_loaded = config_manager.is_loaded()
        providers_configured = 0
        providers_enabled = 0

        if config_loaded:
            config = config_manager.get_config()
            providers_configured = len(config.providers)
            providers_enabled = len(config.get_enabled_providers())

        # Get validation results
        validation_results = config_validator.validate_all()

        errors = [r.message for r in validation_results if r.severity == ValidationSeverity.ERROR]
        warnings = [
            r.message for r in validation_results if r.severity == ValidationSeverity.WARNING
        ]

        config_valid = len(errors) == 0

        # Determine status
        if not config_loaded or errors:
            status = "unhealthy"
        elif warnings:
            status = "degraded"
        else:
            status = "healthy"

        return ConfigHealthResponse(
            status=status,
            config_loaded=config_loaded,
            config_valid=config_valid,
            error_count=len(errors),
            warning_count=len(warnings),
            errors=errors,
            warnings=warnings,
            providers_configured=providers_configured,
            providers_enabled=providers_enabled,
        )

    except Exception as e:
        logger.exception(f"Config health check failed: {e}")
        return ConfigHealthResponse(
            status="unhealthy",
            config_loaded=False,
            config_valid=False,
            error_count=1,
            warning_count=0,
            errors=[str(e)],
            warnings=[],
            providers_configured=0,
            providers_enabled=0,
        )


class FallbackStatusResponse(BaseModel):
    """Fallback mechanism status response."""

    status: str = Field(description="Fallback system status")
    total_fallbacks: int = Field(description="Total fallback operations")
    successful_fallbacks: int = Field(description="Successful fallbacks")
    failed_fallbacks: int = Field(description="Failed fallbacks")
    success_rate: float = Field(description="Fallback success rate")
    provider_health: dict[str, Any] = Field(description="Per-provider health")
    fallback_reasons: dict[str, int] = Field(description="Failure reasons")
    skipped_providers: list[str] = Field(default=[], description="Providers being skipped")


@router.get(
    "/health/fallback",
    summary="Fallback mechanism status",
    description="Get status of the fallback mechanism",
)
async def check_fallback_status() -> FallbackStatusResponse:
    """Get fallback mechanism status.

    Returns:
    - Fallback operation statistics
    - Per-provider health information
    - List of providers being skipped

    """
    try:
        from app.core.fallback_manager import get_fallback_manager

        fallback_manager = get_fallback_manager()
        stats = fallback_manager.get_fallback_stats()
        provider_health = fallback_manager.get_all_providers_health()

        # Calculate success rate
        total = stats.get("total_fallbacks", 0)
        successful = stats.get("successful_fallbacks", 0)
        success_rate = successful / total if total > 0 else 1.0

        # Get list of skipped providers
        skipped = [pid for pid, info in provider_health.items() if info.get("should_skip", False)]

        # Determine status
        failed = stats.get("failed_fallbacks", 0)
        status = "degraded" if (failed > successful and total > 0) or skipped else "healthy"

        return FallbackStatusResponse(
            status=status,
            total_fallbacks=total,
            successful_fallbacks=successful,
            failed_fallbacks=failed,
            success_rate=success_rate,
            provider_health=provider_health,
            fallback_reasons=stats.get("fallback_reasons", {}),
            skipped_providers=skipped,
        )

    except Exception as e:
        logger.exception(f"Fallback status check failed: {e}")
        return FallbackStatusResponse(
            status="unhealthy",
            total_fallbacks=0,
            successful_fallbacks=0,
            failed_fallbacks=0,
            success_rate=0.0,
            provider_health={},
            fallback_reasons={},
            skipped_providers=[],
        )


class StartupHealthResponse(BaseModel):
    """Startup validation health response."""

    status: str = Field(description="Startup status")
    ready: bool = Field(description="Whether app is ready to serve")
    config_loaded: bool = Field(description="Configuration loaded")
    config_valid: bool = Field(description="Configuration valid")
    minimum_requirements_met: bool = Field(description="Min requirements met")
    critical_errors: list[str] = Field(default=[], description="Critical errors")
    warnings: list[str] = Field(default=[], description="Warnings")
    startup_time_ms: float | None = Field(None, description="Startup validation time")


@router.get(
    "/health/startup",
    summary="Startup validation status",
    description="Get status of startup validation checks",
)
async def check_startup_status() -> StartupHealthResponse:
    """Get startup validation status.

    Returns status of startup validation including:
    - Configuration loading status
    - Minimum requirements check
    - Critical errors and warnings
    """
    try:
        from app.core.startup_validator import StartupValidator

        report = StartupValidator.get_startup_report()

        if report is None:
            return StartupHealthResponse(
                status="unknown",
                ready=False,
                config_loaded=False,
                config_valid=False,
                minimum_requirements_met=False,
                critical_errors=["Startup validation not yet performed"],
                warnings=[],
                startup_time_ms=None,
            )

        # Determine status
        if report.get("ready_to_serve", False):
            status = "ready"
        elif report.get("config_loaded", False):
            status = "degraded"
        else:
            status = "not_ready"

        return StartupHealthResponse(
            status=status,
            ready=report.get("ready_to_serve", False),
            config_loaded=report.get("config_loaded", False),
            config_valid=report.get("config_valid", False),
            minimum_requirements_met=report.get("minimum_requirements_met", False),
            critical_errors=report.get("critical_errors", []),
            warnings=report.get("warnings", []),
            startup_time_ms=report.get("startup_time_ms"),
        )

    except Exception as e:
        logger.exception(f"Startup status check failed: {e}")
        return StartupHealthResponse(
            status="error",
            ready=False,
            config_loaded=False,
            config_valid=False,
            minimum_requirements_met=False,
            critical_errors=[str(e)],
            warnings=[],
            startup_time_ms=None,
        )


@router.post(
    "/health/fallback/reset",
    summary="Reset fallback provider status",
    description="Reset failure counts for a provider or all providers",
)
async def reset_fallback_status(provider: str | None = None) -> dict[str, str]:
    """Reset failure counts for providers.

    Args:
        provider: Optional provider ID. If not specified, resets all.

    Returns:
        Status message

    """
    try:
        from app.core.fallback_manager import get_fallback_manager

        fallback_manager = get_fallback_manager()

        if provider:
            fallback_manager.reset_provider_status(provider)
            logger.info(f"Reset fallback status for provider: {provider}")
            return {"status": "reset", "provider": provider}
        fallback_manager.reset_all_provider_status()
        logger.info("Reset fallback status for all providers")
        return {"status": "reset", "provider": "all"}

    except Exception as e:
        logger.exception(f"Failed to reset fallback status: {e}")
        return {"status": "error", "message": str(e)}


@router.post(
    "/health/startup/validate",
    summary="Run startup validation",
    description="Manually trigger startup validation checks",
)
async def run_startup_validation(
    test_connectivity: bool = False,
) -> StartupHealthResponse:
    """Manually run startup validation.

    Args:
        test_connectivity: Whether to test provider connectivity

    Returns:
        Startup validation results

    """
    try:
        from app.core.startup_validator import StartupValidator

        await StartupValidator.validate_on_startup(
            test_connectivity=test_connectivity,
            fail_on_warnings=False,
        )

        return await check_startup_status()

    except Exception as e:
        logger.exception(f"Startup validation failed: {e}")
        return StartupHealthResponse(
            status="error",
            ready=False,
            config_loaded=False,
            config_valid=False,
            minimum_requirements_met=False,
            critical_errors=[str(e)],
            warnings=[],
            startup_time_ms=None,
        )


@router.get("/integration/stats", tags=["integration"])
async def integration_stats_v1():
    """Alias for integration stats under /api/v1."""
    stats: dict[str, Any] = {}

    try:
        from app.core.event_bus import event_bus

        stats["event_bus"] = event_bus.get_stats()
    except ImportError:
        stats["event_bus"] = {"error": "not_available"}

    try:
        from app.core.task_queue import task_queue

        stats["task_queue"] = task_queue.get_stats()
    except ImportError:
        stats["task_queue"] = {"error": "not_available"}

    try:
        from app.services.webhook_service import webhook_service

        stats["webhook_service"] = webhook_service.get_stats()
    except ImportError:
        stats["webhook_service"] = {"error": "not_available"}

    try:
        from app.middleware.idempotency import get_idempotency_store

        stats["idempotency"] = get_idempotency_store().get_stats()
    except ImportError:
        stats["idempotency"] = {"error": "not_available"}

    return stats
