"""
Performance monitoring system for tracking application metrics and bottlenecks.
Provides real-time performance data and analytics for optimization.
"""

import json
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Individual request metric data."""

    timestamp: float
    endpoint: str
    method: str
    status_code: int
    processing_time_ms: float
    memory_usage_mb: float


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    total_requests: int
    requests_per_second: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    memory_usage_mb: float
    cpu_percent: float
    active_connections: int


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system with real-time metrics.
    """

    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.request_history: deque = deque(maxlen=max_history_size)
        self.endpoint_stats: dict[str, list[RequestMetric]] = defaultdict(list)
        self.error_counts: dict[str, int] = defaultdict(int)
        self.performance_cache: dict[str, Any] = {}
        self.cache_ttl = 60  # 1 minute cache TTL
        self.lock = threading.RLock()

        # Performance thresholds for alerts
        self.thresholds = {
            "response_time_warning": 1000,  # 1 second
            "response_time_critical": 5000,  # 5 seconds
            "error_rate_warning": 5.0,  # 5%
            "error_rate_critical": 10.0,  # 10%
            "memory_warning": 500,  # 500 MB
            "memory_critical": 1000,  # 1000 MB
            "cpu_warning": 80,  # 80%
            "cpu_critical": 95,  # 95%
        }

        # Background monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitor_thread.start()

        logger.info("Performance monitor initialized")

    def record_request(
        self, endpoint: str, method: str, status_code: int, processing_time_ms: float
    ):
        """Record a request metric."""
        try:
            current_time = time.time()
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

            metric = RequestMetric(
                timestamp=current_time,
                endpoint=endpoint or "unknown",
                method=method,
                status_code=status_code,
                processing_time_ms=processing_time_ms,
                memory_usage_mb=memory_usage,
            )

            with self.lock:
                self.request_history.append(metric)
                self.endpoint_stats[metric.endpoint].append(metric)

                # Track errors
                if status_code >= 400:
                    self.error_counts[metric.endpoint] += 1

                # Clear performance cache
                self.performance_cache.clear()

        except Exception as e:
            logger.error(f"Error recording request metric: {e}")

    def get_stats(self, time_window_seconds: int = 300) -> PerformanceStats:
        """Get performance statistics for the specified time window."""
        cache_key = f"stats_{time_window_seconds}"

        with self.lock:
            if cache_key in self.performance_cache:
                cached_entry = self.performance_cache[cache_key]
                if time.time() - cached_entry["timestamp"] < self.cache_ttl:
                    return cached_entry["stats"]

        # Calculate fresh statistics
        current_time = time.time()
        cutoff_time = current_time - time_window_seconds

        with self.lock:
            # Filter recent requests
            recent_requests = [req for req in self.request_history if req.timestamp >= cutoff_time]

        if not recent_requests:
            return PerformanceStats(
                total_requests=0,
                requests_per_second=0.0,
                avg_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                error_rate=0.0,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                cpu_percent=psutil.Process().cpu_percent(),
                active_connections=len(recent_requests),
            )

        # Calculate statistics
        processing_times = [req.processing_time_ms for req in recent_requests]
        error_count = sum(1 for req in recent_requests if req.status_code >= 400)

        stats = PerformanceStats(
            total_requests=len(recent_requests),
            requests_per_second=len(recent_requests) / time_window_seconds,
            avg_response_time_ms=statistics.mean(processing_times),
            p95_response_time_ms=statistics.quantiles(processing_times, n=20)[18]
            if len(processing_times) > 20
            else max(processing_times),
            p99_response_time_ms=statistics.quantiles(processing_times, n=100)[98]
            if len(processing_times) > 100
            else max(processing_times),
            error_rate=(error_count / len(recent_requests)) * 100,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_percent=psutil.Process().cpu_percent(),
            active_connections=len(recent_requests),
        )

        # Cache the results
        with self.lock:
            self.performance_cache[cache_key] = {"stats": stats, "timestamp": current_time}

        return stats

    def get_endpoint_stats(self, endpoint: str) -> dict[str, Any]:
        """Get detailed statistics for a specific endpoint."""
        with self.lock:
            endpoint_requests = self.endpoint_stats.get(endpoint, [])

        if not endpoint_requests:
            return {
                "endpoint": endpoint,
                "total_requests": 0,
                "avg_response_time_ms": 0,
                "error_count": 0,
                "error_rate": 0,
                "last_request_time": None,
            }

        processing_times = [req.processing_time_ms for req in endpoint_requests]
        error_count = sum(1 for req in endpoint_requests if req.status_code >= 400)

        return {
            "endpoint": endpoint,
            "total_requests": len(endpoint_requests),
            "avg_response_time_ms": statistics.mean(processing_times),
            "min_response_time_ms": min(processing_times),
            "max_response_time_ms": max(processing_times),
            "p95_response_time_ms": statistics.quantiles(processing_times, n=20)[18]
            if len(processing_times) > 20
            else max(processing_times),
            "error_count": error_count,
            "error_rate": (error_count / len(endpoint_requests)) * 100,
            "last_request_time": max(req.timestamp for req in endpoint_requests),
            "requests_per_minute": len(
                [req for req in endpoint_requests if time.time() - req.timestamp <= 60]
            ),
        }

    def get_system_metrics(self) -> dict[str, Any]:
        """Get current system performance metrics."""
        process = psutil.Process()

        return {
            "timestamp": time.time(),
            "memory": {
                "usage_mb": process.memory_info().rss / 1024 / 1024,
                "usage_percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "total_mb": psutil.virtual_memory().total / 1024 / 1024,
            },
            "cpu": {
                "usage_percent": process.cpu_percent(),
                "system_cpu_percent": psutil.cpu_percent(interval=1),
                "core_count": psutil.cpu_count(),
                "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
            },
            "disk": {
                "usage_percent": psutil.disk_usage("/").percent
                if hasattr(psutil, "disk_usage")
                else None,
                "available_gb": psutil.disk_usage("/").free / 1024 / 1024 / 1024
                if hasattr(psutil, "disk_usage")
                else None,
            },
            "network": psutil.net_io_counters()._asdict()
            if hasattr(psutil, "net_io_counters")
            else None,
            "process": {
                "pid": process.pid,
                "create_time": process.create_time(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()) if hasattr(process, "open_files") else 0,
                "connections": len(process.connections()) if hasattr(process, "connections") else 0,
            },
        }

    def get_alerts(self) -> list[dict[str, Any]]:
        """Check performance thresholds and return alerts."""
        stats = self.get_stats(300)  # Last 5 minutes
        self.get_system_metrics()
        alerts = []

        # Response time alerts
        if stats.avg_response_time_ms > self.thresholds["response_time_critical"]:
            alerts.append(
                {
                    "severity": "critical",
                    "type": "response_time",
                    "message": f"Average response time ({stats.avg_response_time_ms:.1f}ms) exceeds critical threshold",
                    "value": stats.avg_response_time_ms,
                    "threshold": self.thresholds["response_time_critical"],
                }
            )
        elif stats.avg_response_time_ms > self.thresholds["response_time_warning"]:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "response_time",
                    "message": f"Average response time ({stats.avg_response_time_ms:.1f}ms) exceeds warning threshold",
                    "value": stats.avg_response_time_ms,
                    "threshold": self.thresholds["response_time_warning"],
                }
            )

        # Error rate alerts
        if stats.error_rate > self.thresholds["error_rate_critical"]:
            alerts.append(
                {
                    "severity": "critical",
                    "type": "error_rate",
                    "message": f"Error rate ({stats.error_rate:.1f}%) exceeds critical threshold",
                    "value": stats.error_rate,
                    "threshold": self.thresholds["error_rate_critical"],
                }
            )
        elif stats.error_rate > self.thresholds["error_rate_warning"]:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "error_rate",
                    "message": f"Error rate ({stats.error_rate:.1f}%) exceeds warning threshold",
                    "value": stats.error_rate,
                    "threshold": self.thresholds["error_rate_warning"],
                }
            )

        # Memory alerts
        if stats.memory_usage_mb > self.thresholds["memory_critical"]:
            alerts.append(
                {
                    "severity": "critical",
                    "type": "memory",
                    "message": f"Memory usage ({stats.memory_usage_mb:.1f}MB) exceeds critical threshold",
                    "value": stats.memory_usage_mb,
                    "threshold": self.thresholds["memory_critical"],
                }
            )
        elif stats.memory_usage_mb > self.thresholds["memory_warning"]:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "memory",
                    "message": f"Memory usage ({stats.memory_usage_mb:.1f}MB) exceeds warning threshold",
                    "value": stats.memory_usage_mb,
                    "threshold": self.thresholds["memory_warning"],
                }
            )

        # CPU alerts
        if stats.cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append(
                {
                    "severity": "critical",
                    "type": "cpu",
                    "message": f"CPU usage ({stats.cpu_percent:.1f}%) exceeds critical threshold",
                    "value": stats.cpu_percent,
                    "threshold": self.thresholds["cpu_critical"],
                }
            )
        elif stats.cpu_percent > self.thresholds["cpu_warning"]:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "cpu",
                    "message": f"CPU usage ({stats.cpu_percent:.1f}%) exceeds warning threshold",
                    "value": stats.cpu_percent,
                    "threshold": self.thresholds["cpu_warning"],
                }
            )

        return alerts

    def get_performance_report(self) -> dict[str, Any]:
        """Generate a comprehensive performance report."""
        stats = self.get_stats(3600)  # Last hour
        system_metrics = self.get_system_metrics()
        alerts = self.get_alerts()

        # Get endpoint breakdown
        with self.lock:
            endpoint_breakdown = {}
            for endpoint in self.endpoint_stats:
                endpoint_breakdown[endpoint] = self.get_endpoint_stats(endpoint)

        return {
            "timestamp": time.time(),
            "period_hours": 1,
            "summary": asdict(stats),
            "system_metrics": system_metrics,
            "alerts": alerts,
            "endpoints": endpoint_breakdown,
            "performance_history": self._get_performance_trend(),
        }

    def _get_performance_trend(self) -> dict[str, list[float]]:
        """Get performance trend data for the last hour."""
        current_time = time.time()
        trend_data = {
            "timestamps": [],
            "response_times": [],
            "request_rates": [],
            "memory_usage": [],
        }

        # Sample data every 5 minutes for the last hour
        for i in range(12):  # 12 samples = 1 hour
            sample_time = current_time - (i * 300)  # 5 minutes ago

            with self.lock:
                # Get requests in 1-minute window around sample time
                window_requests = [
                    req
                    for req in self.request_history
                    if sample_time - 30 <= req.timestamp <= sample_time + 30
                ]

            if window_requests:
                avg_response_time = statistics.mean(
                    req.processing_time_ms for req in window_requests
                )
                request_rate = len(window_requests) / 60  # requests per second
                avg_memory = statistics.mean(req.memory_usage_mb for req in window_requests)
            else:
                avg_response_time = 0
                request_rate = 0
                avg_memory = 0

            trend_data["timestamps"].append(sample_time)
            trend_data["response_times"].append(avg_response_time)
            trend_data["request_rates"].append(request_rate)
            trend_data["memory_usage"].append(avg_memory)

        # Reverse to have chronological order
        for key in trend_data:
            trend_data[key] = list(reversed(trend_data[key]))

        return trend_data

    def _background_monitoring(self):
        """Background thread for periodic monitoring tasks."""
        while self.monitoring_active:
            try:
                # Check for alerts
                alerts = self.get_alerts()
                if alerts:
                    for alert in alerts:
                        logger.warning(f"Performance alert: {alert['message']}")

                # Clean old data
                self._cleanup_old_data()

                # Log periodic statistics
                stats = self.get_stats(300)  # Last 5 minutes
                logger.debug(
                    f"Performance stats (5m): {stats.total_requests} requests, "
                    f"{stats.avg_response_time_ms:.1f}ms avg, {stats.error_rate:.1f}% errors"
                )

            except Exception as e:
                logger.error(f"Background monitoring error: {e}")

            # Sleep for 1 minute
            time.sleep(60)

    def _cleanup_old_data(self):
        """Clean up old data to prevent memory leaks."""
        cutoff_time = time.time() - 3600  # Keep only last hour

        with self.lock:
            # Clean endpoint stats
            for endpoint in list(self.endpoint_stats.keys()):
                self.endpoint_stats[endpoint] = [
                    req for req in self.endpoint_stats[endpoint] if req.timestamp >= cutoff_time
                ]

                # Remove empty endpoint entries
                if not self.endpoint_stats[endpoint]:
                    del self.endpoint_stats[endpoint]

    def export_metrics(self, format: str = "json") -> str:
        """Export performance metrics in specified format."""
        report = self.get_performance_report()

        if format.lower() == "json":
            return json.dumps(report, indent=2, default=str)
        elif format.lower() == "csv":
            # Simple CSV export of request history
            with self.lock:
                if not self.request_history:
                    return (
                        "timestamp,endpoint,method,status_code,processing_time_ms,memory_usage_mb\n"
                    )

                csv_lines = [
                    "timestamp,endpoint,method,status_code,processing_time_ms,memory_usage_mb"
                ]
                for req in self.request_history:
                    csv_lines.append(
                        f"{req.timestamp},{req.endpoint},{req.method},"
                        f"{req.status_code},{req.processing_time_ms},{req.memory_usage_mb}"
                    )
                return "\n".join(csv_lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def shutdown(self):
        """Shutdown the performance monitor."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitor shutdown complete")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
