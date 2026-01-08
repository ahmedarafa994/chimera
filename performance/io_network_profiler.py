"""
Comprehensive I/O and Network Performance Monitoring
Monitors disk I/O, network latency, API response times, and connection patterns
"""

import asyncio
import json
import os
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiohttp
import psutil
from profiling_config import MetricType, config


@dataclass
class IOMetrics:
    """I/O performance metrics"""
    timestamp: datetime
    disk_io: dict[str, Any]
    network_io: dict[str, Any]
    file_operations: dict[str, int]
    network_connections: int
    io_wait_time: float

@dataclass
class NetworkLatencyTest:
    """Network latency test results"""
    target: str
    test_type: str  # 'ping', 'http', 'tcp'
    latency_ms: float
    success: bool
    timestamp: datetime
    error_message: str | None = None

@dataclass
class APIResponseMetrics:
    """API response time metrics"""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    content_length: int
    timestamp: datetime
    headers: dict[str, str] = field(default_factory=dict)

@dataclass
class IOPerformanceReport:
    """Comprehensive I/O performance report"""
    report_id: str
    timestamp: datetime
    io_metrics: list[IOMetrics]
    network_tests: list[NetworkLatencyTest]
    api_metrics: list[APIResponseMetrics]
    performance_summary: dict[str, Any]
    bottlenecks: list[dict[str, Any]]
    recommendations: list[str]

class IONetworkProfiler:
    """Comprehensive I/O and Network performance profiler"""

    def __init__(self):
        self.monitoring_active = False
        self.io_metrics: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.network_tests: list[NetworkLatencyTest] = []
        self.api_metrics: deque = deque(maxlen=5000)  # Keep last 5000 API calls
        self.monitoring_thread: threading.Thread | None = None

        # Track file operations
        self.file_operations = defaultdict(int)

        # Network connection pool for testing
        self.session: aiohttp.ClientSession | None = None

    async def start_monitoring(self, interval: int = 30) -> None:
        """Start continuous I/O and network monitoring"""
        if not config.is_metric_enabled(MetricType.IO) and not config.is_metric_enabled(MetricType.NETWORK):
            return

        if self.monitoring_active:
            print("I/O and network monitoring already active")
            return

        # Initialize aiohttp session
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )

        self.monitoring_active = True

        # Start monitoring in a separate thread
        self.monitoring_thread = threading.Thread(
            target=self._run_monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()

        print(f"Started I/O and network monitoring with {interval}s interval")

    def _run_monitoring_loop(self, interval: int) -> None:
        """Run monitoring loop in thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._monitoring_loop(interval))
        finally:
            loop.close()

    async def _monitoring_loop(self, interval: int) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect I/O metrics
                await self._collect_io_metrics()

                # Test network latency to key endpoints
                await self._test_network_latency()

                # Test API response times
                await self._test_api_response_times()

                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Error in I/O monitoring loop: {e}")
                await asyncio.sleep(interval)

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        self.monitoring_active = False

        if self.session:
            await self.session.close()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        print("Stopped I/O and network monitoring")

    async def _collect_io_metrics(self) -> None:
        """Collect comprehensive I/O metrics"""
        timestamp = datetime.now(UTC)

        try:
            # Get process I/O stats
            process = psutil.Process()
            process_io = process.io_counters()

            # Get system-wide disk I/O
            disk_io = psutil.disk_io_counters()

            # Get network I/O
            net_io = psutil.net_io_counters()

            # Get network connections
            connections = len(psutil.net_connections())

            # Calculate I/O wait time (Linux-specific)
            io_wait_time = 0.0
            try:
                cpu_times = psutil.cpu_times()
                if hasattr(cpu_times, 'iowait'):
                    io_wait_time = cpu_times.iowait
            except AttributeError:
                pass

            metrics = IOMetrics(
                timestamp=timestamp,
                disk_io={
                    "read_count": disk_io.read_count if disk_io else 0,
                    "write_count": disk_io.write_count if disk_io else 0,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0,
                    "read_time": disk_io.read_time if disk_io else 0,
                    "write_time": disk_io.write_time if disk_io else 0,
                    "process_read_bytes": process_io.read_bytes,
                    "process_write_bytes": process_io.write_bytes
                },
                network_io={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "errin": net_io.errin,
                    "errout": net_io.errout,
                    "dropin": net_io.dropin,
                    "dropout": net_io.dropout
                },
                file_operations=dict(self.file_operations.copy()),
                network_connections=connections,
                io_wait_time=io_wait_time
            )

            self.io_metrics.append(metrics)

        except Exception as e:
            print(f"Error collecting I/O metrics: {e}")

    async def _test_network_latency(self) -> None:
        """Test network latency to various endpoints"""
        if not self.session:
            return

        # Test targets
        test_targets = [
            ("google.com", "ping"),
            ("1.1.1.1", "ping"),
            (config.backend_url, "http"),
            (config.frontend_url, "http")
        ]

        for target, test_type in test_targets:
            try:
                if test_type == "ping":
                    latency = await self._ping_test(target)
                elif test_type == "http":
                    latency = await self._http_test(target)
                else:
                    continue

                test_result = NetworkLatencyTest(
                    target=target,
                    test_type=test_type,
                    latency_ms=latency,
                    success=latency > 0,
                    timestamp=datetime.now(UTC)
                )

                self.network_tests.append(test_result)

                # Keep only last 1000 tests
                if len(self.network_tests) > 1000:
                    self.network_tests = self.network_tests[-1000:]

            except Exception as e:
                error_test = NetworkLatencyTest(
                    target=target,
                    test_type=test_type,
                    latency_ms=0.0,
                    success=False,
                    timestamp=datetime.now(UTC),
                    error_message=str(e)
                )
                self.network_tests.append(error_test)

    async def _ping_test(self, target: str) -> float:
        """Perform ping test"""
        try:
            # Use system ping command
            start_time = time.time()

            if os.name == 'nt':  # Windows
                process = await asyncio.create_subprocess_exec(
                    'ping', '-n', '1', target,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:  # Unix-like
                process = await asyncio.create_subprocess_exec(
                    'ping', '-c', '1', target,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

            _stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            end_time = time.time()

            if process.returncode == 0:
                return (end_time - start_time) * 1000
            else:
                return -1

        except Exception:
            return -1

    async def _http_test(self, target: str) -> float:
        """Perform HTTP latency test"""
        if not self.session:
            return -1

        try:
            start_time = time.time()

            async with self.session.get(target, timeout=aiohttp.ClientTimeout(total=10)) as response:
                await response.read()
                end_time = time.time()

                if 200 <= response.status < 400:
                    return (end_time - start_time) * 1000
                else:
                    return -1

        except Exception:
            return -1

    async def _test_api_response_times(self) -> None:
        """Test API response times for critical endpoints"""
        if not self.session:
            return

        # Test critical API endpoints
        critical_endpoints = [
            ("/health", "GET"),
            ("/api/v1/providers", "GET"),
            ("/api/v1/generate", "POST"),
        ]

        for endpoint, method in critical_endpoints:
            try:
                url = f"{config.backend_url}{endpoint}"
                start_time = time.time()

                if method == "GET":
                    async with self.session.get(url) as response:
                        content = await response.read()
                        response_time = (time.time() - start_time) * 1000

                        metrics = APIResponseMetrics(
                            endpoint=endpoint,
                            method=method,
                            response_time_ms=response_time,
                            status_code=response.status,
                            content_length=len(content),
                            timestamp=datetime.now(UTC),
                            headers=dict(response.headers)
                        )

                elif method == "POST":
                    test_payload = {"prompt": "test", "provider": "mock"}

                    async with self.session.post(url, json=test_payload) as response:
                        content = await response.read()
                        response_time = (time.time() - start_time) * 1000

                        metrics = APIResponseMetrics(
                            endpoint=endpoint,
                            method=method,
                            response_time_ms=response_time,
                            status_code=response.status,
                            content_length=len(content),
                            timestamp=datetime.now(UTC),
                            headers=dict(response.headers)
                        )

                self.api_metrics.append(metrics)

            except Exception:
                # Record failed request
                error_metrics = APIResponseMetrics(
                    endpoint=endpoint,
                    method=method,
                    response_time_ms=-1,
                    status_code=0,
                    content_length=0,
                    timestamp=datetime.now(UTC)
                )
                self.api_metrics.append(error_metrics)

    def track_file_operation(self, operation: str, file_path: str) -> None:
        """Track file operation"""
        self.file_operations[f"{operation}_{os.path.basename(file_path)}"] += 1

    def analyze_io_bottlenecks(self) -> list[dict[str, Any]]:
        """Analyze I/O performance for bottlenecks"""
        bottlenecks = []

        if not self.io_metrics:
            return bottlenecks

        # Analyze disk I/O patterns
        recent_metrics = list(self.io_metrics)[-100:]  # Last 100 measurements

        if len(recent_metrics) >= 2:
            # Calculate I/O rates
            first_metric = recent_metrics[0]
            last_metric = recent_metrics[-1]

            time_diff = (last_metric.timestamp - first_metric.timestamp).total_seconds()

            if time_diff > 0:
                read_rate = (last_metric.disk_io["read_bytes"] - first_metric.disk_io["read_bytes"]) / time_diff
                write_rate = (last_metric.disk_io["write_bytes"] - first_metric.disk_io["write_bytes"]) / time_diff

                # Check for high I/O rates (>100MB/s)
                if read_rate > 100 * 1024 * 1024:
                    bottlenecks.append({
                        "type": "high_disk_read",
                        "severity": "warning",
                        "rate_mbps": read_rate / 1024 / 1024,
                        "description": f"High disk read rate: {read_rate/1024/1024:.1f}MB/s"
                    })

                if write_rate > 100 * 1024 * 1024:
                    bottlenecks.append({
                        "type": "high_disk_write",
                        "severity": "warning",
                        "rate_mbps": write_rate / 1024 / 1024,
                        "description": f"High disk write rate: {write_rate/1024/1024:.1f}MB/s"
                    })

        # Analyze API response times
        if self.api_metrics:
            recent_api_metrics = list(self.api_metrics)[-100:]

            for endpoint in {m.endpoint for m in recent_api_metrics}:
                endpoint_metrics = [m for m in recent_api_metrics if m.endpoint == endpoint and m.response_time_ms > 0]

                if endpoint_metrics:
                    avg_response_time = statistics.mean(m.response_time_ms for m in endpoint_metrics)
                    p95_response_time = statistics.quantiles(
                        [m.response_time_ms for m in endpoint_metrics], n=20
                    )[-1] if len(endpoint_metrics) > 5 else avg_response_time

                    if p95_response_time > 3000:  # 3 seconds
                        bottlenecks.append({
                            "type": "slow_api_endpoint",
                            "severity": "critical" if p95_response_time > 5000 else "warning",
                            "endpoint": endpoint,
                            "avg_response_time_ms": avg_response_time,
                            "p95_response_time_ms": p95_response_time,
                            "description": f"Slow API endpoint: {endpoint} (P95: {p95_response_time:.0f}ms)"
                        })

        # Analyze network latency
        if self.network_tests:
            recent_network_tests = self.network_tests[-50:]

            for target in {t.target for t in recent_network_tests}:
                target_tests = [t for t in recent_network_tests if t.target == target and t.success]

                if target_tests:
                    avg_latency = statistics.mean(t.latency_ms for t in target_tests)

                    if avg_latency > 1000:  # 1 second
                        bottlenecks.append({
                            "type": "high_network_latency",
                            "severity": "warning",
                            "target": target,
                            "avg_latency_ms": avg_latency,
                            "description": f"High network latency to {target}: {avg_latency:.0f}ms"
                        })

        return bottlenecks

    def generate_io_performance_report(self) -> IOPerformanceReport:
        """Generate comprehensive I/O performance report"""
        report_id = f"io_report_{int(time.time())}"
        timestamp = datetime.now(UTC)

        # Analyze bottlenecks
        bottlenecks = self.analyze_io_bottlenecks()

        # Generate performance summary
        performance_summary = {}

        if self.io_metrics:
            recent_metrics = list(self.io_metrics)[-100:]
            performance_summary["avg_io_wait_time"] = statistics.mean(m.io_wait_time for m in recent_metrics)
            performance_summary["avg_network_connections"] = statistics.mean(m.network_connections for m in recent_metrics)

        if self.api_metrics:
            recent_api = list(self.api_metrics)[-100:]
            successful_requests = [m for m in recent_api if m.response_time_ms > 0 and 200 <= m.status_code < 300]

            if successful_requests:
                performance_summary["avg_api_response_time_ms"] = statistics.mean(m.response_time_ms for m in successful_requests)
                performance_summary["api_success_rate"] = len(successful_requests) / len(recent_api)

        if self.network_tests:
            recent_tests = self.network_tests[-100:]
            successful_tests = [t for t in recent_tests if t.success]

            if successful_tests:
                performance_summary["avg_network_latency_ms"] = statistics.mean(t.latency_ms for t in successful_tests)
                performance_summary["network_success_rate"] = len(successful_tests) / len(recent_tests)

        # Generate recommendations
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "high_disk_read":
                recommendations.append("Consider implementing disk read caching")
                recommendations.append("Optimize database queries to reduce disk reads")
            elif bottleneck["type"] == "high_disk_write":
                recommendations.append("Implement write batching to reduce disk writes")
                recommendations.append("Consider using faster storage (SSD) if not already")
            elif bottleneck["type"] == "slow_api_endpoint":
                recommendations.append(f"Optimize {bottleneck['endpoint']} endpoint performance")
                recommendations.append("Add response caching for frequently requested data")
            elif bottleneck["type"] == "high_network_latency":
                recommendations.append(f"Investigate network connectivity to {bottleneck['target']}")
                recommendations.append("Consider using CDN or edge locations")

        if not bottlenecks:
            recommendations.append("I/O performance appears healthy")
            recommendations.append("Continue monitoring for performance degradation")

        report = IOPerformanceReport(
            report_id=report_id,
            timestamp=timestamp,
            io_metrics=list(self.io_metrics),
            network_tests=self.network_tests,
            api_metrics=list(self.api_metrics),
            performance_summary=performance_summary,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )

        # Save report to file
        self._save_io_report(report)

        return report

    def _save_io_report(self, report: IOPerformanceReport) -> None:
        """Save I/O performance report to file"""
        output_path = config.get_output_path(MetricType.IO, f"{report.report_id}.json")

        # Convert to serializable format
        report_dict = {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "performance_summary": report.performance_summary,
            "bottlenecks": report.bottlenecks,
            "recommendations": report.recommendations,
            "metrics_count": len(report.io_metrics),
            "network_tests_count": len(report.network_tests),
            "api_metrics_count": len(report.api_metrics)
        }

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        print(f"I/O performance report saved: {output_path}")

# Global I/O profiler instance
io_profiler = IONetworkProfiler()

# Decorator for monitoring I/O operations
def monitor_io(operation_name: str = "unknown"):
    """Decorator to monitor I/O operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                success = True
            except Exception:
                success = False
                raise
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                # Log slow operations
                if duration_ms > 1000:  # Slower than 1 second
                    print(f"Slow I/O operation '{operation_name}': {duration_ms:.0f}ms (Success: {success})")

            return result
        return wrapper
    return decorator
