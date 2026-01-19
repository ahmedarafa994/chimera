"""
Comprehensive Performance Monitoring and Tracking System.

This module provides advanced monitoring capabilities:
- Real-time performance metrics collection
- Resource utilization tracking
- Application Performance Monitoring (APM)
- Custom metrics and alerting
- Performance profiling and analysis
- Health checks and service discovery
"""

import asyncio
import gc
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import psutil
from pydantic import BaseModel, Field

from app.core.logging import logger

# =====================================================
# Metrics and Performance Models
# =====================================================


class MetricType(str):
    """Metric type constants."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricPoint:
    """Single metric data point."""

    name: str
    value: int | float
    timestamp: float
    tags: dict[str, str] = field(default_factory=dict)
    metric_type: str = MetricType.GAUGE


@dataclass
class ServiceHealth:
    """Service health status."""

    service_name: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: float
    uptime_seconds: float
    dependencies_healthy: int
    dependencies_total: int
    last_check: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profiling data."""

    function_name: str
    execution_count: int
    total_time_seconds: float
    avg_time_seconds: float
    max_time_seconds: float
    min_time_seconds: float
    memory_usage_bytes: int
    last_execution: float


class MonitoringConfig(BaseModel):
    """Configuration for monitoring system."""

    # Collection settings
    collection_interval_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    metrics_retention_hours: int = Field(default=24, ge=1, le=168)
    enable_profiling: bool = Field(default=True)
    enable_memory_monitoring: bool = Field(default=True)

    # Storage settings
    metrics_storage_path: str = Field(default="./monitoring/metrics")
    max_metrics_file_size_mb: int = Field(default=100)
    compress_old_metrics: bool = Field(default=True)

    # Alerting settings
    enable_alerting: bool = Field(default=True)
    alert_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time_p95_ms": 5000.0,
            "error_rate_percent": 5.0,
        }
    )

    # Health checks
    health_check_interval_seconds: float = Field(default=60.0)
    health_check_timeout_seconds: float = Field(default=10.0)

    # External integrations
    prometheus_enabled: bool = Field(default=False)
    prometheus_port: int = Field(default=8090)
    webhook_alerts_url: str | None = None


# =====================================================
# Core Monitoring Engine
# =====================================================


class MetricsCollector:
    """
    Advanced metrics collection and aggregation engine.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config

        # Metrics storage
        self._metrics: deque = deque(maxlen=100000)  # In-memory metrics buffer
        self._aggregated_metrics: dict[str, dict[str, Any]] = defaultdict(dict)

        # Performance tracking
        self._performance_profiles: dict[str, PerformanceProfile] = {}
        self._active_timers: dict[str, float] = {}

        # System metrics
        self._system_metrics_history: deque = deque(maxlen=1440)  # 24h at 1min intervals
        self._custom_metrics: dict[str, MetricPoint] = {}

        # Background tasks
        self._collection_task: asyncio.Task | None = None
        self._persistence_task: asyncio.Task | None = None
        self._is_running = False

        # Storage
        self.storage_path = Path(self.config.metrics_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start metrics collection."""
        self._is_running = True

        # Start background tasks
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._persistence_task = asyncio.create_task(self._persistence_loop())

        logger.info("Metrics collector started")

    async def stop(self) -> None:
        """Stop metrics collection."""
        self._is_running = False

        # Cancel tasks
        if self._collection_task:
            self._collection_task.cancel()
        if self._persistence_task:
            self._persistence_task.cancel()

        # Persist remaining metrics
        await self._persist_metrics()

        logger.info("Metrics collector stopped")

    def record_metric(
        self,
        name: str,
        value: int | float,
        metric_type: str = MetricType.GAUGE,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record a custom metric."""
        metric = MetricPoint(
            name=name, value=value, timestamp=time.time(), tags=tags or {}, metric_type=metric_type
        )

        self._metrics.append(metric)
        self._custom_metrics[name] = metric

    def increment_counter(
        self, name: str, value: int = 1, tags: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        if name in self._custom_metrics:
            current_value = self._custom_metrics[name].value
            self.record_metric(name, current_value + value, MetricType.COUNTER, tags)
        else:
            self.record_metric(name, value, MetricType.COUNTER, tags)

    def set_gauge(self, name: str, value: int | float, tags: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def record_timer(
        self, name: str, duration_seconds: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record a timer metric."""
        self.record_metric(name, duration_seconds, MetricType.TIMER, tags)

    def start_timer(self, name: str) -> str:
        """Start a timer for performance measurement."""
        timer_id = f"{name}_{time.time()}"
        self._active_timers[timer_id] = time.time()
        return timer_id

    def stop_timer(self, timer_id: str, tags: dict[str, str] | None = None) -> float:
        """Stop a timer and record the duration."""
        if timer_id not in self._active_timers:
            return 0.0

        start_time = self._active_timers.pop(timer_id)
        duration = time.time() - start_time

        # Extract metric name from timer_id
        metric_name = timer_id.rsplit("_", 1)[0]
        self.record_timer(metric_name, duration, tags)

        return duration

    async def _collect_system_metrics(self) -> dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {}

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)

            metrics["cpu"] = {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2],
            }

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            metrics["memory"] = {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "usage_percent": memory.percent,
                "swap_total_bytes": swap.total,
                "swap_used_bytes": swap.used,
                "swap_usage_percent": swap.percent,
            }

            # Disk metrics
            disk_usage = psutil.disk_usage("/")
            disk_io = psutil.disk_io_counters()

            metrics["disk"] = {
                "total_bytes": disk_usage.total,
                "used_bytes": disk_usage.used,
                "free_bytes": disk_usage.free,
                "usage_percent": (disk_usage.used / disk_usage.total) * 100,
            }

            if disk_io:
                metrics["disk"].update(
                    {
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                        "read_count": disk_io.read_count,
                        "write_count": disk_io.write_count,
                    }
                )

            # Network metrics
            network_io = psutil.net_io_counters()
            if network_io:
                metrics["network"] = {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv,
                    "errin": network_io.errin,
                    "errout": network_io.errout,
                    "dropin": network_io.dropin,
                    "dropout": network_io.dropout,
                }

            # Process metrics
            process = psutil.Process()

            metrics["process"] = {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_rss_bytes": process.memory_info().rss,
                "memory_vms_bytes": process.memory_info().vms,
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, "num_fds") else 0,
                "create_time": process.create_time(),
            }

            # Python-specific metrics
            metrics["python"] = {
                "version": sys.version,
                "gc_counts": gc.get_count(),
                "gc_stats": gc.get_stats() if hasattr(gc, "get_stats") else [],
                "recursion_limit": sys.getrecursionlimit(),
            }

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

        return metrics

    async def _collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self._is_running:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                timestamp = time.time()

                # Store system metrics
                system_metrics["timestamp"] = timestamp
                self._system_metrics_history.append(system_metrics)

                # Record as individual metrics
                for category, metrics in system_metrics.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, int | float):
                                self.set_gauge(f"{category}.{metric_name}", value)

                # Update aggregated metrics
                await self._update_aggregations()

                await asyncio.sleep(self.config.collection_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.config.collection_interval_seconds)

    async def _update_aggregations(self) -> None:
        """Update aggregated metrics for quick access."""
        if not self._metrics:
            return

        # Group metrics by name
        metrics_by_name = defaultdict(list)
        current_time = time.time()
        cutoff_time = current_time - 3600  # Last hour

        for metric in self._metrics:
            if metric.timestamp >= cutoff_time:
                metrics_by_name[metric.name].append(metric)

        # Calculate aggregations
        for name, metric_list in metrics_by_name.items():
            values = [m.value for m in metric_list]

            if values:
                self._aggregated_metrics[name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": self._percentile(values, 50),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99),
                    "last_updated": current_time,
                }

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]

    async def _persistence_loop(self) -> None:
        """Background metrics persistence loop."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Persist every 5 minutes
                await self._persist_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics persistence error: {e}")

    async def _persist_metrics(self) -> None:
        """Persist metrics to storage."""
        if not self._metrics:
            return

        try:
            # Prepare data for persistence
            metrics_data = []
            for metric in self._metrics:
                metrics_data.append(
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "timestamp": metric.timestamp,
                        "tags": metric.tags,
                        "type": metric.metric_type,
                    }
                )

            # Write to file
            timestamp = int(time.time())
            filename = f"metrics_{timestamp}.json"
            filepath = self.storage_path / filename

            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(metrics_data, indent=2))

            logger.debug(f"Persisted {len(metrics_data)} metrics to {filepath}")

            # Clear old metrics from memory
            self._metrics.clear()

        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current metrics summary."""
        current_time = time.time()

        # Get latest system metrics
        latest_system = self._system_metrics_history[-1] if self._system_metrics_history else {}

        # Get aggregated metrics
        recent_aggregations = {}
        for name, agg in self._aggregated_metrics.items():
            if current_time - agg["last_updated"] < 300:  # Last 5 minutes
                recent_aggregations[name] = agg

        return {
            "timestamp": current_time,
            "system_metrics": latest_system,
            "aggregated_metrics": recent_aggregations,
            "custom_metrics": {name: metric.value for name, metric in self._custom_metrics.items()},
            "active_timers": len(self._active_timers),
            "metrics_buffer_size": len(self._metrics),
        }


# =====================================================
# Health Monitoring System
# =====================================================


class HealthMonitor:
    """
    Comprehensive health monitoring for services and dependencies.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config

        # Service registry
        self._services: dict[str, dict[str, Any]] = {}
        self._health_history: deque = deque(maxlen=1000)

        # Background tasks
        self._health_check_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start health monitoring."""
        self._is_running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._is_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
        logger.info("Health monitor stopped")

    def register_service(
        self,
        service_name: str,
        check_url: str,
        check_type: str = "http",
        timeout: float = 10.0,
        expected_status: int = 200,
        dependencies: list[str] | None = None,
    ) -> None:
        """Register a service for health monitoring."""
        self._services[service_name] = {
            "check_url": check_url,
            "check_type": check_type,
            "timeout": timeout,
            "expected_status": expected_status,
            "dependencies": dependencies or [],
            "last_check": 0,
            "status": "unknown",
            "consecutive_failures": 0,
        }

        logger.info(f"Registered service for health monitoring: {service_name}")

    async def check_service_health(self, service_name: str) -> ServiceHealth:
        """Check health of a specific service."""
        if service_name not in self._services:
            return ServiceHealth(
                service_name=service_name,
                status="unknown",
                response_time_ms=0,
                uptime_seconds=0,
                dependencies_healthy=0,
                dependencies_total=0,
                last_check=time.time(),
                details={"error": "Service not registered"},
            )

        service = self._services[service_name]
        start_time = time.time()

        try:
            # Perform health check based on type
            if service["check_type"] == "http":
                health_status = await self._http_health_check(service)
            else:
                health_status = {"status": "unsupported", "details": {}}

            response_time_ms = (time.time() - start_time) * 1000

            # Check dependencies
            dependencies_healthy = 0
            dependencies_total = len(service["dependencies"])

            for dep_service in service["dependencies"]:
                if dep_service in self._services:
                    dep_health = await self.check_service_health(dep_service)
                    if dep_health.status == "healthy":
                        dependencies_healthy += 1

            # Determine overall status
            overall_status = "healthy"
            if health_status["status"] != "healthy":
                overall_status = health_status["status"]
            elif dependencies_healthy < dependencies_total:
                overall_status = "degraded"

            # Update service state
            service["last_check"] = time.time()
            service["status"] = overall_status
            service["consecutive_failures"] = (
                0 if overall_status == "healthy" else service["consecutive_failures"] + 1
            )

            # Create health object
            health = ServiceHealth(
                service_name=service_name,
                status=overall_status,
                response_time_ms=response_time_ms,
                uptime_seconds=time.time() - start_time,  # Placeholder
                dependencies_healthy=dependencies_healthy,
                dependencies_total=dependencies_total,
                last_check=time.time(),
                details=health_status.get("details", {}),
            )

            # Store in history
            self._health_history.append(health)

            return health

        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")

            service["consecutive_failures"] = service["consecutive_failures"] + 1

            return ServiceHealth(
                service_name=service_name,
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                uptime_seconds=0,
                dependencies_healthy=0,
                dependencies_total=len(service["dependencies"]),
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def _http_health_check(self, service: dict[str, Any]) -> dict[str, Any]:
        """Perform HTTP health check."""
        timeout = aiohttp.ClientTimeout(total=service["timeout"])

        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(service["check_url"]) as response,
        ):
            if response.status == service["expected_status"]:
                return {
                    "status": "healthy",
                    "details": {
                        "http_status": response.status,
                        "headers": dict(response.headers),
                    },
                }
            else:
                return {
                    "status": "unhealthy",
                    "details": {
                        "http_status": response.status,
                        "expected_status": service["expected_status"],
                    },
                }

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._is_running:
            try:
                # Check all registered services
                health_checks = []
                for service_name in self._services:
                    health_checks.append(self.check_service_health(service_name))

                if health_checks:
                    await asyncio.gather(*health_checks, return_exceptions=True)

                await asyncio.sleep(self.config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.config.health_check_interval_seconds)

    def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health status."""
        if not self._services:
            return {"status": "unknown", "services": {}}

        service_statuses = {}
        healthy_count = 0
        total_count = len(self._services)

        for service_name, service in self._services.items():
            service_statuses[service_name] = {
                "status": service["status"],
                "last_check": service["last_check"],
                "consecutive_failures": service["consecutive_failures"],
            }

            if service["status"] == "healthy":
                healthy_count += 1

        # Determine overall status
        if healthy_count == total_count:
            overall_status = "healthy"
        elif healthy_count > total_count / 2:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return {
            "status": overall_status,
            "healthy_services": healthy_count,
            "total_services": total_count,
            "services": service_statuses,
            "last_updated": time.time(),
        }


# =====================================================
# Performance Profiler
# =====================================================


class PerformanceProfiler:
    """
    Advanced performance profiling for function-level monitoring.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._profiles: dict[str, PerformanceProfile] = {}
        self._active_calls: dict[str, dict[str, Any]] = {}

    def profile_function(self, func_name: str):
        """Decorator for profiling function performance."""

        def decorator(func):
            if not self.enabled:
                return func

            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_call(func, func_name, *args, **kwargs)

            def sync_wrapper(*args, **kwargs):
                return self._profile_sync_call(func, func_name, *args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    async def _profile_async_call(self, func, func_name: str, *args, **kwargs):
        """Profile async function call."""
        call_id = f"{func_name}_{time.time()}"
        start_time = time.time()
        start_memory = self._get_memory_usage()

        self._active_calls[call_id] = {
            "func_name": func_name,
            "start_time": start_time,
            "start_memory": start_memory,
        }

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            self._complete_profiling(call_id, start_time, start_memory)

    def _profile_sync_call(self, func, func_name: str, *args, **kwargs):
        """Profile synchronous function call."""
        call_id = f"{func_name}_{time.time()}"
        start_time = time.time()
        start_memory = self._get_memory_usage()

        self._active_calls[call_id] = {
            "func_name": func_name,
            "start_time": start_time,
            "start_memory": start_memory,
        }

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self._complete_profiling(call_id, start_time, start_memory)

    def _complete_profiling(self, call_id: str, start_time: float, start_memory: int) -> None:
        """Complete profiling for a function call."""
        if call_id not in self._active_calls:
            return

        call_info = self._active_calls.pop(call_id)
        func_name = call_info["func_name"]

        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - start_memory

        # Update or create profile
        if func_name in self._profiles:
            profile = self._profiles[func_name]
            profile.execution_count += 1
            profile.total_time_seconds += execution_time
            profile.avg_time_seconds = profile.total_time_seconds / profile.execution_count
            profile.max_time_seconds = max(profile.max_time_seconds, execution_time)
            profile.min_time_seconds = min(profile.min_time_seconds, execution_time)
            profile.memory_usage_bytes += memory_usage
            profile.last_execution = time.time()
        else:
            self._profiles[func_name] = PerformanceProfile(
                function_name=func_name,
                execution_count=1,
                total_time_seconds=execution_time,
                avg_time_seconds=execution_time,
                max_time_seconds=execution_time,
                min_time_seconds=execution_time,
                memory_usage_bytes=memory_usage,
                last_execution=time.time(),
            )

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except (psutil.Error, OSError):
            return 0

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "profiles": {},
            "active_calls": len(self._active_calls),
            "total_functions_profiled": len(self._profiles),
            "report_generated": time.time(),
        }

        for func_name, profile in self._profiles.items():
            report["profiles"][func_name] = {
                "execution_count": profile.execution_count,
                "avg_time_seconds": profile.avg_time_seconds,
                "max_time_seconds": profile.max_time_seconds,
                "min_time_seconds": profile.min_time_seconds,
                "total_time_seconds": profile.total_time_seconds,
                "memory_usage_mb": profile.memory_usage_bytes / 1024 / 1024,
                "last_execution": profile.last_execution,
            }

        return report


# =====================================================
# Integrated Monitoring System
# =====================================================


class IntegratedMonitoringSystem:
    """
    Comprehensive monitoring system integrating metrics, health, and profiling.
    """

    def __init__(self, config: MonitoringConfig | None = None):
        self.config = config or MonitoringConfig()

        # Components
        self.metrics_collector = MetricsCollector(self.config)
        self.health_monitor = HealthMonitor(self.config)
        self.profiler = PerformanceProfiler(self.config.enable_profiling)

        # State
        self._is_running = False

    async def start(self) -> None:
        """Start the integrated monitoring system."""
        await self.metrics_collector.start()
        await self.health_monitor.start()

        self._is_running = True
        logger.info("Integrated monitoring system started")

    async def stop(self) -> None:
        """Stop the monitoring system."""
        await self.metrics_collector.stop()
        await self.health_monitor.stop()

        self._is_running = False
        logger.info("Integrated monitoring system stopped")

    def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "monitoring_system": {
                "running": self._is_running,
                "config": self.config.dict(),
            },
            "metrics": self.metrics_collector.get_current_metrics(),
            "health": self.health_monitor.get_overall_health(),
            "performance": self.profiler.get_performance_report(),
            "timestamp": time.time(),
        }

    # Convenience methods
    def record_metric(self, name: str, value: int | float, **kwargs) -> None:
        """Record a metric."""
        self.metrics_collector.record_metric(name, value, **kwargs)

    def increment_counter(self, name: str, value: int = 1, **kwargs) -> None:
        """Increment a counter."""
        self.metrics_collector.increment_counter(name, value, **kwargs)

    def profile(self, func_name: str):
        """Profile function decorator."""
        return self.profiler.profile_function(func_name)

    def register_health_check(self, service_name: str, check_url: str, **kwargs) -> None:
        """Register a health check."""
        self.health_monitor.register_service(service_name, check_url, **kwargs)


# Global monitoring system instance
monitoring_config = MonitoringConfig()
monitoring_system = IntegratedMonitoringSystem(monitoring_config)
