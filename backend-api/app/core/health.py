"""
Enhanced Health Check System

Provides:
- Comprehensive health checks for all services
- Dependency health monitoring
- Readiness and liveness probes
- Detailed health status reporting
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from app.core.config import settings
from app.core.structured_logging import logger


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class OverallHealth:
    """Overall system health status."""

    status: HealthStatus
    version: str
    environment: str
    uptime_seconds: float
    checks: list[HealthCheckResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "version": self.version,
            "environment": self.environment,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "checks": [check.to_dict() for check in self.checks],
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """
    Health check manager for monitoring service dependencies.

    Enhanced with:
    - Service dependency graph
    - Integration health monitoring
    - Event bus, task queue, and webhook service checks
    """

    # Critical services that affect overall health status
    CRITICAL_SERVICES: ClassVar[set[str]] = {"llm_service", "transformation_engine"}

    # Optional services - their failure only causes "degraded" status, not "unhealthy"
    OPTIONAL_SERVICES: ClassVar[set[str]] = {
        "redis",
        "cache",
        "event_bus",
        "task_queue",
        "webhook_service",
        "database",
    }

    def __init__(self):
        self._checks: dict[str, Callable] = {}
        self._start_time = time.time()
        self._last_check_results: dict[str, HealthCheckResult] = {}
        self._dependencies: dict[str, list[str]] = {}

        # Register default checks
        self._register_default_checks()

        # Register service dependencies
        self._register_default_dependencies()

    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("database", self._check_database)
        self.register_check("redis", self._check_redis)
        self.register_check("llm_service", self._check_llm_service)
        self.register_check("transformation_engine", self._check_transformation_engine)
        self.register_check("cache", self._check_cache)
        self.register_check("event_bus", self._check_event_bus)
        self.register_check("task_queue", self._check_task_queue)
        self.register_check("webhook_service", self._check_webhook_service)

    def _register_default_dependencies(self):
        """Register default service dependencies."""
        self._dependencies = {
            "llm_service": ["database"],
            "transformation_engine": ["llm_service", "cache"],
            "task_queue": ["redis"],
            "webhook_service": ["event_bus"],
            "cache": ["redis"],
        }

    def register_dependency(self, service: str, depends_on: list[str]):
        """
        Register service dependencies for health propagation.

        Args:
            service: Service name
            depends_on: List of services this service depends on
        """
        self._dependencies[service] = depends_on

    def register_check(self, name: str, check_func: Callable):
        """
        Register a health check function.

        Args:
            name: Name of the check
            check_func: Async function that returns HealthCheckResult
        """
        self._checks[name] = check_func

    def unregister_check(self, name: str):
        """Unregister a health check."""
        self._checks.pop(name, None)

    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a single health check."""
        if name not in self._checks:
            return HealthCheckResult(
                name=name, status=HealthStatus.UNKNOWN, message=f"Check '{name}' not found"
            )

        start_time = time.perf_counter()

        try:
            check_func = self._checks[name]

            if asyncio.iscoroutinefunction(check_func):
                # Enforce timeout for async checks
                try:
                    result = await asyncio.wait_for(check_func(), timeout=2.0)
                except TimeoutError:
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.DEGRADED,
                        message="Check timed out (limit: 2.0s)",
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                    )
            else:
                result = check_func()

            result.latency_ms = (time.perf_counter() - start_time) * 1000
            self._last_check_results[name] = result
            return result

        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._last_check_results[name] = result
            return result

    async def run_all_checks(self) -> OverallHealth:
        """Run all registered health checks."""
        results = await asyncio.gather(
            *[self.run_check(name) for name in self._checks], return_exceptions=True
        )

        # Process results
        check_results = []
        for i, result in enumerate(results):
            name = list(self._checks.keys())[i]
            if isinstance(result, Exception):
                check_results.append(
                    HealthCheckResult(name=name, status=HealthStatus.UNHEALTHY, message=str(result))
                )
            else:
                check_results.append(result)

        # Determine overall status
        overall_status = self._determine_overall_status(check_results)

        return OverallHealth(
            status=overall_status,
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            uptime_seconds=time.time() - self._start_time,
            checks=check_results,
        )

    def _determine_overall_status(self, results: list[HealthCheckResult]) -> HealthStatus:
        """
        Determine overall health status from individual checks.

        Only critical services can cause UNHEALTHY status.
        Optional services only cause DEGRADED status at worst.
        """
        if not results:
            return HealthStatus.UNKNOWN

        # Separate critical and optional service results
        critical_results = [r for r in results if r.name in self.CRITICAL_SERVICES]
        optional_results = [r for r in results if r.name in self.OPTIONAL_SERVICES]

        # Check critical services first
        critical_statuses = [r.status for r in critical_results]
        if any(s == HealthStatus.UNHEALTHY for s in critical_statuses):
            return HealthStatus.UNHEALTHY

        # If all critical services are healthy, check for degradation
        all_statuses = [r.status for r in results]

        if all(s == HealthStatus.HEALTHY for s in all_statuses):
            return HealthStatus.HEALTHY

        # Any unhealthy optional service or degraded critical service = degraded overall
        if any(s == HealthStatus.UNHEALTHY for s in [r.status for r in optional_results]) or any(
            s == HealthStatus.DEGRADED for s in all_statuses
        ):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    async def liveness_check(self) -> HealthCheckResult:
        """
        Liveness probe - checks if the application is running.
        Used by Kubernetes to determine if the pod should be restarted.
        """
        return HealthCheckResult(
            name="liveness",
            status=HealthStatus.HEALTHY,
            message="Application is running",
            details={
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "version": settings.VERSION,
            },
        )

    async def readiness_check(self) -> HealthCheckResult:
        """
        Readiness probe - checks if the application is ready to serve traffic.
        Used by Kubernetes to determine if the pod should receive traffic.
        """
        # Check critical dependencies
        critical_checks = ["database", "llm_service"]

        results = await asyncio.gather(
            *[self.run_check(name) for name in critical_checks if name in self._checks],
            return_exceptions=True,
        )

        all_healthy = all(
            isinstance(r, HealthCheckResult) and r.status == HealthStatus.HEALTHY for r in results
        )

        if all_healthy:
            return HealthCheckResult(
                name="readiness",
                status=HealthStatus.HEALTHY,
                message="Application is ready to serve traffic",
            )
        else:
            return HealthCheckResult(
                name="readiness",
                status=HealthStatus.UNHEALTHY,
                message="Application is not ready - critical dependencies unavailable",
            )

    def get_last_results(self) -> dict[str, HealthCheckResult]:
        """Get the last check results."""
        return self._last_check_results.copy()

    # Default health check implementations

    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            # For SQLite, just check if we can import and the file exists
            from sqlalchemy import text

            from app.core.database import engine

            # Try a simple query
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={"type": "sqlite"},
            )
        except ImportError:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database not configured (using in-memory)",
                details={"type": "none"},
            )
        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.DEGRADED,
                message=f"Database check skipped: {e}",
                details={"error": str(e)},
            )

    async def _check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        try:
            import socket
            from urllib.parse import urlparse

            # Parse Redis URL to get host and port
            parsed = urlparse(settings.REDIS_URL)
            host = parsed.hostname or "localhost"
            port = parsed.port or 6379

            # Quick TCP check first - fail fast if port is not open
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)  # 100ms timeout for TCP check
            try:
                result = sock.connect_ex((host, port))
                sock.close()
                if result != 0:
                    return HealthCheckResult(
                        name="redis",
                        status=HealthStatus.DEGRADED,
                        message="Redis not running (port not open)",
                        details={"type": "port_closed", "host": host, "port": port},
                    )
            except TimeoutError:
                sock.close()
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message="Redis not reachable (connection timeout)",
                    details={"type": "timeout", "host": host, "port": port},
                )
            except Exception as e:
                sock.close()
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message=f"Redis connection check failed: {e}",
                    details={"type": "error", "error": str(e)},
                )

            # Port is open, now try actual Redis connection
            import redis.asyncio as redis

            client = redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                socket_timeout=0.5,
                socket_connect_timeout=0.3,
            )

            try:
                await asyncio.wait_for(client.ping(), timeout=0.5)
                info = await asyncio.wait_for(client.info("server"), timeout=0.5)
                await client.close()

                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.HEALTHY,
                    message="Redis connection successful",
                    details={
                        "version": info.get("redis_version", "unknown"),
                        "connected_clients": info.get("connected_clients", 0),
                    },
                )
            except TimeoutError:
                await client.close()
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message="Redis unavailable: Timeout connecting to server",
                    details={"error": "Timeout connecting to server"},
                )
            except Exception as e:
                await client.close()
                raise e
        except ImportError:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis client not installed",
                details={"type": "not_installed"},
            )
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "refused" in error_msg.lower():
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message="Redis not running (connection refused)",
                    details={"type": "connection_refused", "url": settings.REDIS_URL},
                )
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.DEGRADED,
                message=f"Redis unavailable: {e}",
                details={"error": str(e)},
            )

    async def _check_llm_service(self) -> HealthCheckResult:
        """Check LLM service availability."""
        try:
            from app.services.llm_service import llm_service

            # DEBUG: Log instance ID to track if same as registration
            logger.info(
                f"health_check: llm_service instance_id={id(llm_service)}, "
                f"_providers_keys={list(llm_service._providers.keys())}"
            )

            providers = llm_service.get_available_providers()

            logger.info(f"health_check: get_available_providers() returned: {providers}")

            if providers:
                return HealthCheckResult(
                    name="llm_service",
                    status=HealthStatus.HEALTHY,
                    message=f"LLM service available with {len(providers)} providers",
                    details={
                        "providers": providers,
                        "default_provider": (
                            str(llm_service.default_provider.value)
                            if llm_service.default_provider
                            else None
                        ),
                    },
                )
            else:
                return HealthCheckResult(
                    name="llm_service",
                    status=HealthStatus.DEGRADED,
                    message="No LLM providers available",
                    details={"providers": []},
                )
        except Exception as e:
            return HealthCheckResult(
                name="llm_service",
                status=HealthStatus.UNHEALTHY,
                message=f"LLM service error: {e}",
                details={"error": str(e)},
            )

    async def _check_transformation_engine(self) -> HealthCheckResult:
        """Check transformation engine availability."""
        try:
            # Check if techniques are loaded
            techniques = settings.transformation.technique_suites

            return HealthCheckResult(
                name="transformation_engine",
                status=HealthStatus.HEALTHY,
                message=f"Transformation engine ready with {len(techniques)} techniques",
                details={
                    "technique_count": len(techniques),
                    "techniques": list(techniques.keys())[:10],  # First 10
                },
            )
        except Exception as e:
            return HealthCheckResult(
                name="transformation_engine",
                status=HealthStatus.UNHEALTHY,
                message=f"Transformation engine error: {e}",
                details={"error": str(e)},
            )

    async def _check_cache(self) -> HealthCheckResult:
        """Check cache availability."""
        try:
            if settings.ENABLE_CACHE:
                return HealthCheckResult(
                    name="cache",
                    status=HealthStatus.HEALTHY,
                    message="Cache enabled",
                    details={
                        "enabled": True,
                        "max_items": settings.CACHE_MAX_MEMORY_ITEMS,
                        "default_ttl": settings.CACHE_DEFAULT_TTL,
                    },
                )
            else:
                return HealthCheckResult(
                    name="cache",
                    status=HealthStatus.DEGRADED,
                    message="Cache disabled",
                    details={"enabled": False},
                )
        except Exception as e:
            return HealthCheckResult(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache error: {e}",
                details={"error": str(e)},
            )

    async def _check_event_bus(self) -> HealthCheckResult:
        """Check event bus availability."""
        try:
            from app.core.event_bus import event_bus

            stats = event_bus.get_stats()

            return HealthCheckResult(
                name="event_bus",
                status=HealthStatus.HEALTHY if stats["running"] else HealthStatus.DEGRADED,
                message=f"Event bus {'running' if stats['running'] else 'stopped'}",
                details={
                    "running": stats["running"],
                    "event_types_subscribed": stats["event_types_subscribed"],
                    "total_handlers": stats["total_handlers"],
                    "events_stored": stats["events_stored"],
                },
            )
        except ImportError:
            return HealthCheckResult(
                name="event_bus",
                status=HealthStatus.DEGRADED,
                message="Event bus module not available",
                details={"type": "not_available"},
            )
        except Exception as e:
            return HealthCheckResult(
                name="event_bus",
                status=HealthStatus.UNHEALTHY,
                message=f"Event bus error: {e}",
                details={"error": str(e)},
            )

    async def _check_task_queue(self) -> HealthCheckResult:
        """Check task queue availability."""
        try:
            from app.core.task_queue import task_queue

            stats = task_queue.get_stats()

            status = HealthStatus.HEALTHY
            if stats["dead_letter_count"] > 10:
                status = HealthStatus.DEGRADED

            return HealthCheckResult(
                name="task_queue",
                status=status,
                message=f"Task queue {'running' if stats['running'] else 'stopped'} with {stats['workers']} workers",
                details={
                    "running": stats["running"],
                    "workers": stats["workers"],
                    "handlers_registered": stats["handlers_registered"],
                    "current_queue_size": stats["current_queue_size"],
                    "dead_letter_count": stats["dead_letter_count"],
                    "total_completed": stats["total_completed"],
                    "total_failed": stats["total_failed"],
                },
            )
        except ImportError:
            return HealthCheckResult(
                name="task_queue",
                status=HealthStatus.DEGRADED,
                message="Task queue module not available",
                details={"type": "not_available"},
            )
        except Exception as e:
            return HealthCheckResult(
                name="task_queue",
                status=HealthStatus.UNHEALTHY,
                message=f"Task queue error: {e}",
                details={"error": str(e)},
            )

    async def _check_webhook_service(self) -> HealthCheckResult:
        """Check webhook service availability."""
        try:
            from app.services.webhook_service import webhook_service

            stats = webhook_service.get_stats()

            status = HealthStatus.HEALTHY
            if stats["failed_deliveries"] > 10:
                status = HealthStatus.DEGRADED

            return HealthCheckResult(
                name="webhook_service",
                status=status,
                message=f"Webhook service with {stats['active_webhooks']} active webhooks",
                details={
                    "total_webhooks": stats["total_webhooks"],
                    "active_webhooks": stats["active_webhooks"],
                    "total_deliveries": stats["total_deliveries"],
                    "successful_deliveries": stats["successful_deliveries"],
                    "failed_deliveries": stats["failed_deliveries"],
                    "pending_deliveries": stats["pending_deliveries"],
                },
            )
        except ImportError:
            return HealthCheckResult(
                name="webhook_service",
                status=HealthStatus.DEGRADED,
                message="Webhook service module not available",
                details={"type": "not_available"},
            )
        except Exception as e:
            return HealthCheckResult(
                name="webhook_service",
                status=HealthStatus.UNHEALTHY,
                message=f"Webhook service error: {e}",
                details={"error": str(e)},
            )

    async def get_dependency_graph(self) -> dict:
        """
        Get service dependency health graph.

        Returns a graph showing services, their health status, and dependencies.
        """
        results = await self.run_all_checks()

        graph = {
            "status": results.status.value,
            "services": {},
            "dependencies": self._dependencies,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        for check in results.checks:
            graph["services"][check.name] = {
                "status": check.status.value,
                "message": check.message,
                "latency_ms": check.latency_ms,
                "depends_on": self._dependencies.get(check.name, []),
            }

        return graph

    def get_dependency_tree(self, service: str, visited: set | None = None) -> dict:
        """
        Get dependency tree for a specific service (recursive).

        Useful for understanding cascading failures.
        """
        if visited is None:
            visited = set()

        if service in visited:
            return {"name": service, "circular": True}

        visited.add(service)

        deps = self._dependencies.get(service, [])
        last_result = self._last_check_results.get(service)

        return {
            "name": service,
            "status": last_result.status.value if last_result else "unknown",
            "dependencies": [self.get_dependency_tree(dep, visited.copy()) for dep in deps],
        }


# Global health checker instance
health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return health_checker
