"""
Prometheus-Compatible Metrics Endpoint

Provides comprehensive metrics collection for monitoring and observability.
Includes:
- Request latency histograms
- Request counters by endpoint and status
- Circuit breaker status
- Cache hit rates
- LLM provider metrics
- Rate limiting statistics
"""

import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.core.circuit_breaker import CircuitBreakerRegistry
from app.core.logging import logger
from app.services.llm_service import llm_service
from app.services.transformation_service import transformation_engine

router = APIRouter()


# =============================================================================
# Metrics Collection
# =============================================================================


class MetricsCollector:
    """
    Collects and formats metrics in Prometheus exposition format.

    This is a lightweight implementation that doesn't require the
    prometheus_client library, making it suitable for environments
    where that dependency isn't available.
    """

    def __init__(self):
        self._start_time = time.time()
        self._request_counts: dict[str, int] = {}
        self._request_latencies: dict[str, list[float]] = {}
        self._error_counts: dict[str, int] = {}

    def record_request(
        self, endpoint: str, method: str, status_code: int, latency_ms: float
    ) -> None:
        """Record a request for metrics."""
        key = f"{method}:{endpoint}:{status_code}"
        self._request_counts[key] = self._request_counts.get(key, 0) + 1

        latency_key = f"{method}:{endpoint}"
        if latency_key not in self._request_latencies:
            self._request_latencies[latency_key] = []
        self._request_latencies[latency_key].append(latency_ms)

        # Keep only last 1000 latencies per endpoint
        if len(self._request_latencies[latency_key]) > 1000:
            self._request_latencies[latency_key] = self._request_latencies[latency_key][-1000:]

        if status_code >= 400:
            error_key = f"{method}:{endpoint}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self._start_time

    def get_request_stats(self) -> dict[str, Any]:
        """Get request statistics."""
        return {
            "counts": self._request_counts.copy(),
            "error_counts": self._error_counts.copy(),
        }

    def get_latency_percentiles(self, endpoint: str) -> dict[str, float]:
        """Calculate latency percentiles for an endpoint."""
        latencies = self._request_latencies.get(endpoint, [])
        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0, "avg": 0}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "p50": sorted_latencies[int(n * 0.5)] if n > 0 else 0,
            "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            "avg": sum(latencies) / n if n > 0 else 0,
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return metrics_collector


# =============================================================================
# Prometheus Format Helpers
# =============================================================================


def format_prometheus_metric(
    name: str,
    value: float | int,
    metric_type: str = "gauge",
    help_text: str = "",
    labels: dict[str, str] | None = None,
) -> str:
    """Format a single metric in Prometheus exposition format."""
    lines = []

    if help_text:
        lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")

    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        lines.append(f"{name}{{{label_str}}} {value}")
    else:
        lines.append(f"{name} {value}")

    return "\n".join(lines)


def collect_all_metrics() -> str:
    """Collect all metrics and format as Prometheus exposition format."""
    metrics_lines = []

    # ==========================================================================
    # Application Metrics
    # ==========================================================================

    # Uptime
    metrics_lines.append(
        format_prometheus_metric(
            "chimera_uptime_seconds",
            metrics_collector.get_uptime_seconds(),
            "counter",
            "Application uptime in seconds",
        )
    )

    # ==========================================================================
    # Circuit Breaker Metrics
    # ==========================================================================

    circuit_breaker_status = CircuitBreakerRegistry.get_all_status()

    for name, status in circuit_breaker_status.items():
        # Circuit state (0=closed, 1=half_open, 2=open)
        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(status["state"], -1)
        metrics_lines.append(
            format_prometheus_metric(
                "chimera_circuit_breaker_state",
                state_value,
                "gauge",
                "Circuit breaker state (0=closed, 1=half_open, 2=open)",
                {"name": name},
            )
        )

        # Failure count
        metrics_lines.append(
            format_prometheus_metric(
                "chimera_circuit_breaker_failures",
                status["failure_count"],
                "gauge",
                "Current failure count for circuit breaker",
                {"name": name},
            )
        )

        # Metrics from circuit breaker
        cb_metrics = status.get("metrics", {})

        metrics_lines.append(
            format_prometheus_metric(
                "chimera_circuit_breaker_total_calls",
                cb_metrics.get("total_calls", 0),
                "counter",
                "Total calls through circuit breaker",
                {"name": name},
            )
        )

        metrics_lines.append(
            format_prometheus_metric(
                "chimera_circuit_breaker_successful_calls",
                cb_metrics.get("successful_calls", 0),
                "counter",
                "Successful calls through circuit breaker",
                {"name": name},
            )
        )

        metrics_lines.append(
            format_prometheus_metric(
                "chimera_circuit_breaker_failed_calls",
                cb_metrics.get("failed_calls", 0),
                "counter",
                "Failed calls through circuit breaker",
                {"name": name},
            )
        )

        metrics_lines.append(
            format_prometheus_metric(
                "chimera_circuit_breaker_rejected_calls",
                cb_metrics.get("rejected_calls", 0),
                "counter",
                "Rejected calls (circuit open)",
                {"name": name},
            )
        )

    # ==========================================================================
    # LLM Service Metrics
    # ==========================================================================

    try:
        llm_stats = llm_service.get_performance_stats()

        # Cache metrics
        cache_stats = llm_stats.get("cache", {})
        metrics_lines.append(
            format_prometheus_metric(
                "chimera_llm_cache_size",
                cache_stats.get("size", 0),
                "gauge",
                "Current LLM response cache size",
            )
        )

        metrics_lines.append(
            format_prometheus_metric(
                "chimera_llm_cache_hits", cache_stats.get("hits", 0), "counter", "LLM cache hits"
            )
        )

        metrics_lines.append(
            format_prometheus_metric(
                "chimera_llm_cache_misses",
                cache_stats.get("misses", 0),
                "counter",
                "LLM cache misses",
            )
        )

        metrics_lines.append(
            format_prometheus_metric(
                "chimera_llm_cache_hit_rate",
                cache_stats.get("hit_rate", 0),
                "gauge",
                "LLM cache hit rate",
            )
        )

        # Deduplication metrics
        dedup_stats = llm_stats.get("deduplication", {})
        metrics_lines.append(
            format_prometheus_metric(
                "chimera_llm_deduplicated_requests",
                dedup_stats.get("deduplicated_count", 0),
                "counter",
                "Number of deduplicated LLM requests",
            )
        )

        metrics_lines.append(
            format_prometheus_metric(
                "chimera_llm_pending_requests",
                dedup_stats.get("pending_requests", 0),
                "gauge",
                "Number of pending LLM requests",
            )
        )

        # Provider count
        metrics_lines.append(
            format_prometheus_metric(
                "chimera_llm_providers_registered",
                len(llm_stats.get("providers", [])),
                "gauge",
                "Number of registered LLM providers",
            )
        )

    except Exception as e:
        logger.warning(f"Failed to collect LLM metrics: {e}")

    # ==========================================================================
    # Transformation Cache Metrics (Enhanced with PERF-001)
    # ==========================================================================

    try:
        cache_stats = transformation_engine.get_cache_stats()
        cache_metrics = (
            transformation_engine.cache.get_metrics() if transformation_engine.cache else {}
        )

        if cache_stats:
            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_size",
                    cache_stats.get("current_size", 0),
                    "gauge",
                    "Current transformation cache size",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_max_size",
                    cache_stats.get("max_size", 0),
                    "gauge",
                    "Maximum transformation cache size",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_hits",
                    cache_stats.get("hits", 0),
                    "counter",
                    "Transformation cache hits",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_misses",
                    cache_stats.get("misses", 0),
                    "counter",
                    "Transformation cache misses",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_evictions",
                    cache_stats.get("evictions", 0),
                    "counter",
                    "Transformation cache LRU evictions",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_hit_rate",
                    cache_stats.get("hit_rate", 0),
                    "gauge",
                    "Transformation cache hit rate",
                )
            )

        # Enhanced cache metrics (PERF-001)
        if cache_metrics:
            # Health status (0=healthy, 1=degraded, 2=warning, 3=critical)
            health_value = {"healthy": 0, "degraded": 1, "warning": 2, "critical": 3}.get(
                cache_metrics.get("health", "unknown"), -1
            )
            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_health",
                    health_value,
                    "gauge",
                    "Cache health status (0=healthy, 1=degraded, 2=warning, 3=critical)",
                )
            )

            # Utilization percentage
            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_utilization_percent",
                    cache_metrics.get("utilization_percent", 0),
                    "gauge",
                    "Cache utilization percentage",
                )
            )

            # Memory metrics
            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_memory_bytes",
                    cache_metrics.get("estimated_memory_bytes", 0),
                    "gauge",
                    "Estimated cache memory usage in bytes",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_memory_mb",
                    cache_metrics.get("estimated_memory_mb", 0),
                    "gauge",
                    "Estimated cache memory usage in megabytes",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_avg_entry_size_bytes",
                    cache_metrics.get("avg_entry_size_bytes", 0),
                    "gauge",
                    "Average cache entry size in bytes",
                )
            )

            # Event counters
            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_expired",
                    cache_metrics.get("expired", 0),
                    "counter",
                    "Cache entries expired (TTL)",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_size_rejections",
                    cache_metrics.get("size_rejections", 0),
                    "counter",
                    "Cache entries rejected (too large)",
                )
            )

            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_transformation_cache_eviction_rate",
                    cache_metrics.get("eviction_rate", 0),
                    "gauge",
                    "Cache eviction rate (evictions per hit)",
                )
            )
    except Exception as e:
        logger.warning(f"Failed to collect transformation cache metrics: {e}")

    # ==========================================================================
    # Request Metrics (from collector)
    # ==========================================================================

    request_stats = metrics_collector.get_request_stats()

    for key, count in request_stats.get("counts", {}).items():
        parts = key.split(":")
        if len(parts) >= 3:
            method, endpoint, status = parts[0], parts[1], parts[2]
            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_http_requests_total",
                    count,
                    "counter",
                    "Total HTTP requests",
                    {"method": method, "endpoint": endpoint, "status": status},
                )
            )

    for key, count in request_stats.get("error_counts", {}).items():
        parts = key.split(":")
        if len(parts) >= 2:
            method, endpoint = parts[0], parts[1]
            metrics_lines.append(
                format_prometheus_metric(
                    "chimera_http_errors_total",
                    count,
                    "counter",
                    "Total HTTP errors",
                    {"method": method, "endpoint": endpoint},
                )
            )

    return "\n\n".join(metrics_lines) + "\n"


# =============================================================================
# API Endpoints
# =============================================================================


@router.get(
    "/metrics/prometheus",
    response_class=PlainTextResponse,
    tags=["utils"],
    summary="Prometheus metrics endpoint",
    description="Returns metrics in Prometheus exposition format for scraping.",
)
async def get_prometheus_metrics():
    """
    Get all metrics in Prometheus exposition format.

    This endpoint is designed to be scraped by Prometheus or compatible
    monitoring systems. It returns metrics including:

    - Application uptime
    - Circuit breaker status and statistics
    - LLM service cache and deduplication stats
    - Transformation cache statistics
    - HTTP request counts and error rates
    """
    try:
        metrics_output = collect_all_metrics()
        return PlainTextResponse(
            content=metrics_output, media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        return PlainTextResponse(content=f"# Error collecting metrics: {e}\n", status_code=500)


@router.get(
    "/metrics/json",
    tags=["utils"],
    summary="JSON metrics endpoint",
    description="Returns metrics in JSON format for easier consumption by dashboards.",
)
async def get_json_metrics() -> dict[str, Any]:
    """
    Get all metrics in JSON format.

    This endpoint provides the same metrics as /metrics but in JSON format,
    which is easier to consume for custom dashboards and debugging.
    """
    try:
        # Collect all metrics
        metrics = {
            "uptime_seconds": metrics_collector.get_uptime_seconds(),
            "circuit_breakers": CircuitBreakerRegistry.get_all_status(),
            "llm_service": {},
            "transformation_cache": {},
            "request_stats": metrics_collector.get_request_stats(),
        }

        # LLM service stats
        try:
            metrics["llm_service"] = llm_service.get_performance_stats()
        except Exception as e:
            metrics["llm_service"] = {"error": str(e)}

        # Transformation cache stats
        try:
            cache_stats = transformation_engine.get_cache_stats()
            metrics["transformation_cache"] = cache_stats or {}
        except Exception as e:
            metrics["transformation_cache"] = {"error": str(e)}

        return {
            "status": "ok",
            "timestamp": time.time(),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Failed to collect JSON metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


@router.get(
    "/metrics/circuit-breakers",
    tags=["utils"],
    summary="Circuit breaker status",
    description="Returns detailed status of all circuit breakers.",
)
async def get_circuit_breaker_status() -> dict[str, Any]:
    """
    Get detailed status of all circuit breakers.

    Useful for debugging provider availability issues and understanding
    the current state of resilience patterns.
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
        "circuit_breakers": CircuitBreakerRegistry.get_all_status(),
    }


@router.post(
    "/metrics/circuit-breakers/{name}/reset",
    tags=["admin"],
    summary="Reset a circuit breaker",
    description="Reset a specific circuit breaker to closed state.",
)
async def reset_circuit_breaker(name: str) -> dict[str, Any]:
    """
    Reset a circuit breaker to its initial (closed) state.

    Use this to manually recover from a circuit breaker that's stuck open
    after the underlying issue has been resolved.
    """
    try:
        CircuitBreakerRegistry.reset(name)
        return {
            "status": "ok",
            "message": f"Circuit breaker '{name}' reset successfully",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker '{name}': {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


@router.post(
    "/metrics/circuit-breakers/reset-all",
    tags=["admin"],
    summary="Reset all circuit breakers",
    description="Reset all circuit breakers to closed state.",
)
async def reset_all_circuit_breakers() -> dict[str, Any]:
    """
    Reset all circuit breakers to their initial (closed) state.

    Use with caution - this will allow requests to flow to all providers
    regardless of their previous failure state.
    """
    try:
        CircuitBreakerRegistry.reset_all()
        return {
            "status": "ok",
            "message": "All circuit breakers reset successfully",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Failed to reset all circuit breakers: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


# =============================================================================
# Cache Metrics Endpoints (PERF-001 FIX)
# =============================================================================


@router.get(
    "/metrics/cache",
    tags=["utils"],
    summary="Transformation cache metrics",
    description="Returns detailed transformation cache metrics including health, memory usage, and performance statistics.",
)
async def get_cache_metrics() -> dict[str, Any]:
    """
    Get comprehensive transformation cache metrics.

    PERF-001 FIX: Enhanced cache metrics endpoint for monitoring and alerting.

    Returns:
        Dictionary containing:
        - health: Overall cache health status (healthy/degraded/warning/critical)
        - utilization_percent: Cache capacity utilization
        - hit_rate: Cache hit rate (0.0 to 1.0)
        - estimated_memory_mb: Estimated memory usage
        - performance: Request counts and evictions
        - configuration: Cache configuration parameters

    Example:
        GET /api/v1/metrics/cache

        Response:
        {
            "status": "ok",
            "cache_metrics": {
                "health": "healthy",
                "utilization_percent": 25.5,
                "hit_rate": 0.8234,
                "estimated_memory_mb": 12.45,
                "current_size": 255,
                "max_size": 1000,
                ...
            }
        }
    """
    try:
        cache_metrics = (
            transformation_engine.cache.get_metrics() if transformation_engine.cache else {}
        )

        return {
            "status": "ok",
            "timestamp": time.time(),
            "cache_metrics": cache_metrics,
            "cache_enabled": transformation_engine.enable_cache,
        }
    except Exception as e:
        logger.error(f"Failed to collect cache metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


@router.post(
    "/metrics/cache/clear",
    tags=["admin"],
    summary="Clear transformation cache",
    description="Clear all entries from the transformation cache.",
)
async def clear_cache() -> dict[str, Any]:
    """
    Clear all entries from the transformation cache.

    Use this to free memory or reset cache statistics.
    Cache will automatically repopulate as new transformations are performed.
    """
    try:
        if transformation_engine.cache:
            # Get metrics before clearing
            before_metrics = transformation_engine.cache.get_metrics()

            # Clear the cache
            transformation_engine.cache.clear()

            return {
                "status": "ok",
                "message": "Cache cleared successfully",
                "timestamp": time.time(),
                "before": {
                    "size": before_metrics.get("current_size", 0),
                    "memory_mb": before_metrics.get("estimated_memory_mb", 0),
                },
            }
        else:
            return {
                "status": "error",
                "error": "Cache is not enabled",
                "timestamp": time.time(),
            }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


# PERF-001 FIX: Connection Pool Metrics
@router.get(
    "/metrics/connection-pools",
    tags=["utils"],
    summary="Connection pool metrics",
    description="Returns detailed connection pool statistics for all LLM providers.",
)
async def get_connection_pool_metrics() -> dict[str, Any]:
    """
    Get connection pool metrics for all LLM providers.

    PERF-001 FIX: Connection pool monitoring for observability and alerting.
    """
    try:
        from app.core.connection_pool import connection_pool_manager

        pool_stats = connection_pool_manager.get_stats()

        return {
            "status": "ok",
            "timestamp": time.time(),
            "pools": pool_stats,
            "summary": {
                "total_pools": len(pool_stats),
                "total_requests": sum(
                    stats.get("total_requests", 0) for stats in pool_stats.values()
                ),
                "total_successful_requests": sum(
                    stats.get("successful_requests", 0) for stats in pool_stats.values()
                ),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get connection pool metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


@router.post(
    "/metrics/connection-pools/reset",
    tags=["admin"],
    summary="Reset connection pool stats",
    description="Reset connection pool statistics for all providers.",
)
async def reset_connection_pool_stats() -> dict[str, Any]:
    """
    Reset connection pool statistics.

    Useful for baseline measurements after making configuration changes.
    """
    try:
        from app.core.connection_pool import connection_pool_manager

        connection_pool_manager.reset_stats()

        return {
            "status": "ok",
            "message": "Connection pool statistics reset successfully",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Failed to reset connection pool stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


# PERF-001 FIX: Multi-Level Cache Metrics
@router.get(
    "/metrics/multi-level-cache",
    tags=["utils"],
    summary="Multi-level cache metrics",
    description="Returns detailed metrics for L1 (in-memory) and L2 (Redis) cache layers.",
)
async def get_multi_level_cache_metrics() -> dict[str, Any]:
    """
    Get multi-level cache metrics.

    PERF-001 FIX: Provides visibility into both cache layers for monitoring
    cache performance and effectiveness.
    """
    try:
        from app.core.config import settings
        from app.core.redis_cache import get_multi_level_cache

        if not settings.CACHE_ENABLE_L2:
            return {
                "status": "ok",
                "message": "L2 cache is not enabled. Set CACHE_ENABLE_L2=true to enable.",
                "timestamp": time.time(),
                "l2_enabled": False,
            }

        cache = get_multi_level_cache()
        stats = await cache.get_stats()

        return {
            "status": "ok",
            "timestamp": time.time(),
            "l2_enabled": True,
            **stats,
        }
    except Exception as e:
        logger.error(f"Failed to get multi-level cache metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


@router.post(
    "/metrics/multi-level-cache/clear",
    tags=["admin"],
    summary="Clear multi-level cache",
    description="Clear all entries from both L1 and L2 cache.",
)
async def clear_multi_level_cache() -> dict[str, Any]:
    """
    Clear all cache entries from both L1 and L2 cache.

    Useful for freeing memory or resetting cache statistics.
    """
    try:
        from app.core.config import settings
        from app.core.redis_cache import get_multi_level_cache

        if not settings.CACHE_ENABLE_L2:
            return {
                "status": "error",
                "error": "L2 cache is not enabled",
                "timestamp": time.time(),
            }

        cache = get_multi_level_cache()
        await cache.clear()

        return {
            "status": "ok",
            "message": "Multi-level cache cleared successfully",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Failed to clear multi-level cache: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }
