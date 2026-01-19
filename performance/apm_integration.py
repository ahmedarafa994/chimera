"""
APM Integration for DataDog, New Relic, and Compatible Platforms
Comprehensive application performance monitoring integration
"""

import json
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# DataDog APM
try:
    import ddtrace
    from ddtrace import tracer
    from ddtrace.contrib.aiohttp import patch as aiohttp_patch
    from ddtrace.contrib.fastapi import patch as fastapi_patch
    from ddtrace.contrib.redis import patch as redis_patch
    from ddtrace.contrib.requests import patch as requests_patch
    from ddtrace.contrib.sqlite3 import patch as sqlite3_patch

    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False

# New Relic APM
try:
    import newrelic.agent
    from newrelic.api.function_trace import function_trace
    from newrelic.api.transaction import current_transaction

    NEW_RELIC_AVAILABLE = True
except ImportError:
    NEW_RELIC_AVAILABLE = False

# Generic APM metrics
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import builtins

from profiling_config import MetricType, config


@dataclass
class APMMetrics:
    """APM-specific metrics"""

    timestamp: datetime
    service_name: str
    trace_id: str | None
    span_id: str | None
    operation_name: str
    duration_ms: float
    error: str | None
    custom_metrics: dict[str, Any]
    tags: dict[str, str]


@dataclass
class APMAlert:
    """APM alert data"""

    alert_id: str
    timestamp: datetime
    severity: str  # critical, warning, info
    metric_name: str
    current_value: float
    threshold_value: float
    service: str
    description: str
    recommendations: list[str]


class DataDogIntegration:
    """DataDog APM integration"""

    def __init__(self):
        self.enabled = config.apm_config["datadog"]["enabled"] and DATADOG_AVAILABLE
        self.service_name = config.apm_config["datadog"]["service_name"]

        if self.enabled:
            self._configure_datadog()

    def _configure_datadog(self) -> None:
        """Configure DataDog tracing"""
        try:
            # Configure tracer
            ddtrace.config.service = self.service_name
            ddtrace.config.env = config.apm_config["datadog"]["env"]
            ddtrace.config.version = config.apm_config["datadog"]["version"]

            # Set sampling rate
            ddtrace.config.trace_sample_rate = config.apm_config["datadog"]["trace_sample_rate"]

            # Enable automatic instrumentation
            fastapi_patch()
            requests_patch()
            aiohttp_patch()
            redis_patch()
            sqlite3_patch()

            # Set global tags
            ddtrace.config.tags = {
                "system": "chimera",
                "component": "ai-optimization",
                "deployment.environment": config.environment.value,
            }

            print("DataDog APM configured")

        except Exception as e:
            print(f"Error configuring DataDog: {e}")
            self.enabled = False

    @contextmanager
    def trace_operation(self, operation_name: str, service: str | None = None, **tags):
        """Trace operation with DataDog"""
        if not self.enabled:
            yield None
            return

        with tracer.trace(operation_name, service=service or self.service_name) as span:
            # Set tags
            for key, value in tags.items():
                span.set_tag(key, value)

            try:
                yield span
            except Exception as e:
                span.error = 1
                span.set_tag("error.message", str(e))
                span.set_tag("error.type", type(e).__name__)
                raise

    def trace_llm_request(
        self, provider: str, model: str, prompt_length: int, response_length: int, latency_ms: float
    ) -> None:
        """Trace LLM request with DataDog"""
        if not self.enabled:
            return

        with self.trace_operation("llm_request", service="llm-service") as span:
            if span:
                span.set_tag("llm.provider", provider)
                span.set_tag("llm.model", model)
                span.set_tag("llm.prompt_length", prompt_length)
                span.set_tag("llm.response_length", response_length)
                span.set_metric("llm.latency", latency_ms)

    def custom_metric(
        self, metric_name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Send custom metric to DataDog"""
        if not self.enabled:
            return

        try:
            from ddtrace.api import DdApi

            dd_api = DdApi()

            metric_tags = [f"{k}:{v}" for k, v in (tags or {}).items()]
            dd_api.increment(metric_name, value, tags=metric_tags)

        except Exception as e:
            print(f"Error sending DataDog metric: {e}")


class NewRelicIntegration:
    """New Relic APM integration"""

    def __init__(self):
        self.enabled = config.apm_config["new_relic"]["enabled"] and NEW_RELIC_AVAILABLE
        self.app_name = config.apm_config["new_relic"]["app_name"]

        if self.enabled:
            self._configure_newrelic()

    def _configure_newrelic(self) -> None:
        """Configure New Relic monitoring"""
        try:
            license_key = config.apm_config["new_relic"]["license_key"]
            if not license_key:
                print("New Relic license key not provided")
                self.enabled = False
                return

            # Initialize New Relic
            settings = newrelic.agent.global_settings()
            settings.license_key = license_key
            settings.app_name = self.app_name
            settings.log_file = "stdout"
            settings.log_level = "info"

            newrelic.agent.initialize()
            print("New Relic APM configured")

        except Exception as e:
            print(f"Error configuring New Relic: {e}")
            self.enabled = False

    @contextmanager
    def trace_operation(self, operation_name: str, category: str = "Function", **attributes):
        """Trace operation with New Relic"""
        if not self.enabled:
            yield None
            return

        @function_trace(name=operation_name, group=category)
        def traced_operation():
            # Add custom attributes
            transaction = current_transaction()
            if transaction:
                for key, value in attributes.items():
                    transaction.add_custom_attribute(key, value)

        try:
            traced_operation()
            yield None
        except Exception:
            transaction = current_transaction()
            if transaction:
                transaction.notice_error()
            raise

    def trace_llm_request(
        self, provider: str, model: str, prompt_length: int, response_length: int, latency_ms: float
    ) -> None:
        """Trace LLM request with New Relic"""
        if not self.enabled:
            return

        @function_trace(name="llm_request", group="AI")
        def llm_trace():
            transaction = current_transaction()
            if transaction:
                transaction.add_custom_attribute("llm.provider", provider)
                transaction.add_custom_attribute("llm.model", model)
                transaction.add_custom_attribute("llm.prompt_length", prompt_length)
                transaction.add_custom_attribute("llm.response_length", response_length)
                transaction.add_custom_attribute("llm.latency_ms", latency_ms)

        llm_trace()

    def custom_metric(self, metric_name: str, value: float, unit: str = "count") -> None:
        """Send custom metric to New Relic"""
        if not self.enabled:
            return

        try:
            newrelic.agent.record_custom_metric(metric_name, value, unit)
        except Exception as e:
            print(f"Error sending New Relic metric: {e}")


class GenericAPMIntegration:
    """Generic APM integration for custom metrics collection"""

    def __init__(self):
        self.metrics: list[APMMetrics] = []
        self.alerts: list[APMAlert] = []
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None

    def start_monitoring(self, interval: int = 60) -> None:
        """Start generic APM monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self.monitoring_thread.start()
        print("Generic APM monitoring started")

    def stop_monitoring(self) -> None:
        """Stop generic APM monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    def _monitoring_loop(self, interval: int) -> None:
        """Main monitoring loop for generic APM"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_performance_thresholds()
                time.sleep(interval)
            except Exception as e:
                print(f"Error in generic APM monitoring: {e}")
                time.sleep(interval)

    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        if not PSUTIL_AVAILABLE:
            return

        datetime.now(UTC)

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric("system.cpu.usage", cpu_percent, {"unit": "percent"})

            # Memory metrics
            memory = psutil.virtual_memory()
            self._record_metric("system.memory.usage", memory.percent, {"unit": "percent"})
            self._record_metric("system.memory.available", memory.available, {"unit": "bytes"})

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._record_metric("system.disk.read_bytes", disk_io.read_bytes, {"unit": "bytes"})
                self._record_metric(
                    "system.disk.write_bytes", disk_io.write_bytes, {"unit": "bytes"}
                )

            # Network I/O metrics
            net_io = psutil.net_io_counters()
            if net_io:
                self._record_metric(
                    "system.network.bytes_sent", net_io.bytes_sent, {"unit": "bytes"}
                )
                self._record_metric(
                    "system.network.bytes_recv", net_io.bytes_recv, {"unit": "bytes"}
                )

        except Exception as e:
            print(f"Error collecting system metrics: {e}")

    def _record_metric(self, metric_name: str, value: float, tags: dict[str, str]) -> None:
        """Record a generic metric"""
        metric = APMMetrics(
            timestamp=datetime.now(UTC),
            service_name="chimera-system",
            trace_id=None,
            span_id=None,
            operation_name="system_metric",
            duration_ms=0,
            error=None,
            custom_metrics={metric_name: value},
            tags=tags,
        )

        self.metrics.append(metric)

        # Keep only recent metrics
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-5000:]

    def _check_performance_thresholds(self) -> None:
        """Check performance thresholds and generate alerts"""
        if not self.metrics:
            return

        recent_metrics = self.metrics[-10:]  # Last 10 metrics

        # Check CPU threshold
        cpu_metrics = [m for m in recent_metrics if "system.cpu.usage" in m.custom_metrics]
        if cpu_metrics:
            avg_cpu = sum(m.custom_metrics["system.cpu.usage"] for m in cpu_metrics) / len(
                cpu_metrics
            )

            if avg_cpu > config.performance_thresholds["cpu"]["critical"]:
                self._create_alert(
                    "high_cpu_usage",
                    avg_cpu,
                    config.performance_thresholds["cpu"]["critical"],
                    "critical",
                    "Average CPU usage is critically high",
                )

    def _create_alert(
        self,
        alert_type: str,
        current_value: float,
        threshold: float,
        severity: str,
        description: str,
    ) -> None:
        """Create a performance alert"""
        alert = APMAlert(
            alert_id=f"{alert_type}_{int(time.time())}",
            timestamp=datetime.now(UTC),
            severity=severity,
            metric_name=alert_type,
            current_value=current_value,
            threshold_value=threshold,
            service="chimera-system",
            description=description,
            recommendations=self._get_alert_recommendations(alert_type),
        )

        self.alerts.append(alert)

        # Print alert
        print(
            f"APM ALERT [{severity.upper()}]: {description} - Current: {current_value:.1f}, Threshold: {threshold:.1f}"
        )

    def _get_alert_recommendations(self, alert_type: str) -> list[str]:
        """Get recommendations for specific alert types"""
        recommendations = {
            "high_cpu_usage": [
                "Check for CPU-intensive processes",
                "Consider optimizing algorithms",
                "Scale horizontally if needed",
            ],
            "high_memory_usage": [
                "Check for memory leaks",
                "Optimize data structures",
                "Consider increasing available memory",
            ],
            "slow_response_time": [
                "Optimize database queries",
                "Implement caching",
                "Review application logic",
            ],
        }

        return recommendations.get(alert_type, ["Investigate the root cause"])


class UnifiedAPMProfiler:
    """Unified APM profiler supporting multiple platforms"""

    def __init__(self):
        self.datadog = DataDogIntegration()
        self.newrelic = NewRelicIntegration()
        self.generic = GenericAPMIntegration()

        # Start generic monitoring if no other APM is enabled
        if not self.datadog.enabled and not self.newrelic.enabled:
            self.generic.start_monitoring()

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Unified operation tracing across all enabled APM platforms"""
        contexts = []

        # DataDog tracing
        if self.datadog.enabled:
            contexts.append(self.datadog.trace_operation(operation_name, **attributes))

        # New Relic tracing
        if self.newrelic.enabled:
            contexts.append(self.newrelic.trace_operation(operation_name, **attributes))

        # Enter all contexts
        active_contexts = []
        for ctx in contexts:
            try:
                active_contexts.append(ctx.__enter__())
            except Exception as e:
                print(f"Error entering APM context: {e}")

        try:
            yield active_contexts[0] if active_contexts else None
        except Exception as e:
            # Let each context handle the exception
            for ctx in contexts:
                with suppress(builtins.BaseException):
                    ctx.__exit__(type(e), e, e.__traceback__)
            raise
        else:
            # Exit all contexts successfully
            for ctx in contexts:
                with suppress(builtins.BaseException):
                    ctx.__exit__(None, None, None)

    def trace_llm_request(
        self, provider: str, model: str, prompt: str, response: str, latency_ms: float
    ) -> None:
        """Trace LLM request across all APM platforms"""
        prompt_length = len(prompt)
        response_length = len(response)

        if self.datadog.enabled:
            self.datadog.trace_llm_request(
                provider, model, prompt_length, response_length, latency_ms
            )

        if self.newrelic.enabled:
            self.newrelic.trace_llm_request(
                provider, model, prompt_length, response_length, latency_ms
            )

        # Generic APM recording
        self.generic._record_metric(
            "llm.request.latency", latency_ms, {"provider": provider, "model": model}
        )

    def trace_transformation(
        self, technique: str, input_length: int, output_length: int, processing_time_ms: float
    ) -> None:
        """Trace prompt transformation"""
        with self.trace_operation("prompt_transformation") as span:
            attributes = {
                "transformation.technique": technique,
                "transformation.input_length": input_length,
                "transformation.output_length": output_length,
                "transformation.processing_time_ms": processing_time_ms,
            }

            # Add attributes to active span if available
            if span and hasattr(span, "set_tag"):
                for key, value in attributes.items():
                    span.set_tag(key, value)

    def custom_metric(
        self, metric_name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Send custom metric to all enabled APM platforms"""
        if self.datadog.enabled:
            self.datadog.custom_metric(metric_name, value, tags)

        if self.newrelic.enabled:
            self.newrelic.custom_metric(metric_name, value)

        # Generic APM
        self.generic._record_metric(metric_name, value, tags or {})

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary from all APM platforms"""
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "apm_platforms": {
                "datadog_enabled": self.datadog.enabled,
                "newrelic_enabled": self.newrelic.enabled,
                "generic_enabled": self.generic.monitoring_active,
            },
            "recent_metrics": len(self.generic.metrics),
            "active_alerts": len(self.generic.alerts),
        }

        # Add recent alerts
        recent_alerts = [
            alert
            for alert in self.generic.alerts
            if (datetime.now(UTC) - alert.timestamp).total_seconds() < 3600
        ]

        summary["recent_alerts"] = [
            {
                "severity": alert.severity,
                "metric": alert.metric_name,
                "description": alert.description,
            }
            for alert in recent_alerts
        ]

        return summary

    def save_apm_report(self) -> str:
        """Save APM report to file"""
        report = {
            "report_id": f"apm_report_{int(time.time())}",
            "timestamp": datetime.now(UTC).isoformat(),
            "performance_summary": self.get_performance_summary(),
            "configuration": {
                "datadog": {"enabled": self.datadog.enabled, "service": self.datadog.service_name},
                "newrelic": {"enabled": self.newrelic.enabled, "app": self.newrelic.app_name},
                "generic": {"enabled": self.generic.monitoring_active},
            },
            "metrics_collected": len(self.generic.metrics),
            "alerts_generated": len(self.generic.alerts),
        }

        output_path = config.get_output_path(
            MetricType.NETWORK, f"apm_report_{int(time.time())}.json"
        )

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"APM report saved: {output_path}")
        return output_path


# Global unified APM profiler instance
apm_profiler = UnifiedAPMProfiler()


# Decorators for APM tracing
def apm_trace(operation_name: str, **attributes):
    """Decorator for APM tracing"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with apm_profiler.trace_operation(operation_name, **attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def apm_trace_async(operation_name: str, **attributes):
    """Decorator for async APM tracing"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            with apm_profiler.trace_operation(operation_name, **attributes):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
