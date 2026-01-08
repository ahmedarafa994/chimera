"""
OpenTelemetry Distributed Tracing Setup
Comprehensive observability with traces, metrics, and logs for Chimera system
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# OpenTelemetry imports
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    print("OpenTelemetry not available - install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")

from profiling_config import MetricType, config


@dataclass
class TraceData:
    """Trace span data"""
    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status: str
    tags: dict[str, Any]
    logs: list[dict[str, Any]]

@dataclass
class DistributedTraceReport:
    """Distributed trace analysis report"""
    report_id: str
    timestamp: datetime
    traces: list[TraceData]
    service_map: dict[str, list[str]]
    latency_analysis: dict[str, Any]
    error_analysis: dict[str, Any]
    bottlenecks: list[dict[str, Any]]
    recommendations: list[str]

class OpenTelemetryProfiler:
    """OpenTelemetry-based distributed tracing and observability"""

    def __init__(self):
        self.tracer_provider: TracerProvider | None = None
        self.tracer = None
        self.meter_provider = None
        self.meter = None
        self.traces: list[TraceData] = []
        self.instrumented = False

        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()
            self._setup_metrics()

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing"""
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": config.apm_config["opentelemetry"]["service_name"],
                "service.version": config.apm_config["opentelemetry"]["service_version"],
                "deployment.environment": config.environment.value
            })

            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self.tracer_provider)

            # Configure exporters
            self._setup_trace_exporters()

            # Get tracer
            self.tracer = trace.get_tracer(__name__)

            print("OpenTelemetry tracing configured")

        except Exception as e:
            print(f"Error setting up OpenTelemetry tracing: {e}")

    def _setup_trace_exporters(self) -> None:
        """Setup trace exporters (OTLP, Jaeger, Console)"""
        if not self.tracer_provider:
            return

        # Console exporter for development
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        self.tracer_provider.add_span_processor(console_processor)

        # OTLP exporter (for Jaeger, DataDog, etc.)
        otlp_endpoint = config.apm_config["opentelemetry"]["endpoint"]
        if otlp_endpoint and otlp_endpoint != "http://localhost:4317":
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    headers=config.apm_config["opentelemetry"]["headers"]
                )
                otlp_processor = BatchSpanProcessor(otlp_exporter)
                self.tracer_provider.add_span_processor(otlp_processor)
                print(f"OTLP exporter configured: {otlp_endpoint}")
            except Exception as e:
                print(f"Error setting up OTLP exporter: {e}")

        # Jaeger exporter (alternative)
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(jaeger_processor)
            print("Jaeger exporter configured")
        except Exception as e:
            print(f"Jaeger exporter not available: {e}")

    def _setup_metrics(self) -> None:
        """Setup OpenTelemetry metrics"""
        try:
            # Create metric reader
            otlp_endpoint = config.apm_config["opentelemetry"]["endpoint"]
            if otlp_endpoint:
                metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint.replace("4317", "4318"))
            else:
                # Use console exporter for development
                from opentelemetry.exporter.console.metrics import ConsoleMetricExporter
                metric_exporter = ConsoleMetricExporter()

            reader = PeriodicExportingMetricReader(
                exporter=metric_exporter,
                export_interval_millis=5000  # Export every 5 seconds
            )

            # Create meter provider
            resource = Resource.create({
                "service.name": config.apm_config["opentelemetry"]["service_name"],
                "service.version": config.apm_config["opentelemetry"]["service_version"]
            })

            self.meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[reader]
            )
            metrics.set_meter_provider(self.meter_provider)

            # Get meter
            self.meter = metrics.get_meter(__name__)

            print("OpenTelemetry metrics configured")

        except Exception as e:
            print(f"Error setting up OpenTelemetry metrics: {e}")

    def instrument_fastapi(self, app) -> None:
        """Instrument FastAPI application"""
        if not OPENTELEMETRY_AVAILABLE or self.instrumented:
            return

        try:
            FastAPIInstrumentor.instrument_app(app)
            print("FastAPI instrumented with OpenTelemetry")
        except Exception as e:
            print(f"Error instrumenting FastAPI: {e}")

    def instrument_libraries(self) -> None:
        """Auto-instrument common libraries"""
        if not OPENTELEMETRY_AVAILABLE or self.instrumented:
            return

        try:
            # Instrument HTTP clients
            RequestsInstrumentor().instrument()
            AioHttpClientInstrumentor().instrument()

            # Instrument databases
            SQLite3Instrumentor().instrument()

            # Instrument Redis
            RedisInstrumentor().instrument()

            self.instrumented = True
            print("Libraries instrumented with OpenTelemetry")

        except Exception as e:
            print(f"Error instrumenting libraries: {e}")

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations"""
        if not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            # Set attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)

            start_time = time.time()

            try:
                yield span
            except Exception as e:
                # Record exception
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                # Add duration attribute
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", duration_ms)

    def trace_llm_request(self, provider: str, model: str, prompt: str,
                         response: str, latency_ms: float) -> None:
        """Trace LLM provider request"""
        if not self.tracer:
            return

        with self.tracer.start_as_current_span("llm_request") as span:
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.response_length", len(response))
            span.set_attribute("llm.latency_ms", latency_ms)

            # Add prompt hash for correlation
            import hashlib
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            span.set_attribute("llm.prompt_hash", prompt_hash)

    def trace_transformation(self, technique: str, input_prompt: str,
                           output_prompt: str, processing_time_ms: float) -> None:
        """Trace prompt transformation operation"""
        if not self.tracer:
            return

        with self.tracer.start_as_current_span("prompt_transformation") as span:
            span.set_attribute("transformation.technique", technique)
            span.set_attribute("transformation.input_length", len(input_prompt))
            span.set_attribute("transformation.output_length", len(output_prompt))
            span.set_attribute("transformation.processing_time_ms", processing_time_ms)

            # Calculate complexity metrics
            complexity_ratio = len(output_prompt) / len(input_prompt) if input_prompt else 1.0
            span.set_attribute("transformation.complexity_ratio", complexity_ratio)

    def trace_jailbreak_attempt(self, technique: str, success: bool,
                              confidence_score: float, detection_bypassed: bool) -> None:
        """Trace jailbreak attempt for research purposes"""
        if not self.tracer:
            return

        with self.tracer.start_as_current_span("jailbreak_attempt") as span:
            span.set_attribute("jailbreak.technique", technique)
            span.set_attribute("jailbreak.success", success)
            span.set_attribute("jailbreak.confidence_score", confidence_score)
            span.set_attribute("jailbreak.detection_bypassed", detection_bypassed)

            # Mark as research activity
            span.set_attribute("research.category", "adversarial_ai")

    def trace_autodan_optimization(self, method: str, iterations: int,
                                 initial_score: float, final_score: float,
                                 optimization_time_ms: float) -> None:
        """Trace AutoDAN optimization process"""
        if not self.tracer:
            return

        with self.tracer.start_as_current_span("autodan_optimization") as span:
            span.set_attribute("autodan.method", method)
            span.set_attribute("autodan.iterations", iterations)
            span.set_attribute("autodan.initial_score", initial_score)
            span.set_attribute("autodan.final_score", final_score)
            span.set_attribute("autodan.score_improvement", final_score - initial_score)
            span.set_attribute("autodan.optimization_time_ms", optimization_time_ms)

    def trace_websocket_connection(self, connection_id: str, client_ip: str,
                                 messages_sent: int, messages_received: int,
                                 connection_duration_ms: float) -> None:
        """Trace WebSocket connection metrics"""
        if not self.tracer:
            return

        with self.tracer.start_as_current_span("websocket_connection") as span:
            span.set_attribute("websocket.connection_id", connection_id)
            span.set_attribute("websocket.client_ip", client_ip)
            span.set_attribute("websocket.messages_sent", messages_sent)
            span.set_attribute("websocket.messages_received", messages_received)
            span.set_attribute("websocket.duration_ms", connection_duration_ms)

    def create_custom_metrics(self) -> dict[str, Any]:
        """Create custom metrics for Chimera-specific operations"""
        if not self.meter:
            return {}

        try:
            # LLM request metrics
            llm_request_counter = self.meter.create_counter(
                "llm_requests_total",
                description="Total number of LLM requests",
                unit="1"
            )

            llm_latency_histogram = self.meter.create_histogram(
                "llm_request_duration_ms",
                description="LLM request latency in milliseconds",
                unit="ms"
            )

            # Transformation metrics
            transformation_counter = self.meter.create_counter(
                "transformations_total",
                description="Total number of prompt transformations",
                unit="1"
            )

            # Jailbreak metrics (for research)
            jailbreak_counter = self.meter.create_counter(
                "jailbreak_attempts_total",
                description="Total number of jailbreak attempts (research)",
                unit="1"
            )

            # WebSocket metrics
            websocket_connections = self.meter.create_up_down_counter(
                "websocket_connections_active",
                description="Number of active WebSocket connections",
                unit="1"
            )

            # System metrics
            cpu_usage_gauge = self.meter.create_observable_gauge(
                "system_cpu_usage_percent",
                callbacks=[self._get_cpu_usage],
                description="System CPU usage percentage",
                unit="%"
            )

            memory_usage_gauge = self.meter.create_observable_gauge(
                "system_memory_usage_bytes",
                callbacks=[self._get_memory_usage],
                description="System memory usage in bytes",
                unit="bytes"
            )

            return {
                "llm_requests": llm_request_counter,
                "llm_latency": llm_latency_histogram,
                "transformations": transformation_counter,
                "jailbreaks": jailbreak_counter,
                "websocket_connections": websocket_connections,
                "cpu_usage": cpu_usage_gauge,
                "memory_usage": memory_usage_gauge
            }

        except Exception as e:
            print(f"Error creating custom metrics: {e}")
            return {}

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except Exception:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes"""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except Exception:
            return 0.0

    def analyze_traces(self, time_window_minutes: int = 60) -> DistributedTraceReport:
        """Analyze distributed traces for performance insights"""
        report_id = f"trace_report_{int(time.time())}"
        timestamp = datetime.now(UTC)

        # Filter traces within time window
        cutoff_time = timestamp.replace(minute=timestamp.minute - time_window_minutes)
        recent_traces = [t for t in self.traces if t.start_time >= cutoff_time]

        # Build service map
        service_map = self._build_service_map(recent_traces)

        # Analyze latency patterns
        latency_analysis = self._analyze_latency_patterns(recent_traces)

        # Analyze errors
        error_analysis = self._analyze_error_patterns(recent_traces)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(recent_traces)

        # Generate recommendations
        recommendations = self._generate_trace_recommendations(
            latency_analysis, error_analysis, bottlenecks
        )

        report = DistributedTraceReport(
            report_id=report_id,
            timestamp=timestamp,
            traces=recent_traces,
            service_map=service_map,
            latency_analysis=latency_analysis,
            error_analysis=error_analysis,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )

        # Save report
        self._save_trace_report(report)

        return report

    def _build_service_map(self, traces: list[TraceData]) -> dict[str, list[str]]:
        """Build service dependency map from traces"""
        service_map = {}

        for trace in traces:
            service_name = trace.tags.get("service.name", "unknown")
            called_services = []

            # Extract called services from span tags
            for key, value in trace.tags.items():
                if key.startswith("downstream.service"):
                    called_services.append(value)

            service_map[service_name] = called_services

        return service_map

    def _analyze_latency_patterns(self, traces: list[TraceData]) -> dict[str, Any]:
        """Analyze latency patterns across operations"""
        operation_latencies = {}

        for trace in traces:
            op_name = trace.operation_name
            if op_name not in operation_latencies:
                operation_latencies[op_name] = []

            operation_latencies[op_name].append(trace.duration_ms)

        # Calculate statistics
        analysis = {}
        for operation, latencies in operation_latencies.items():
            if latencies:
                import statistics
                analysis[operation] = {
                    "count": len(latencies),
                    "avg_latency_ms": statistics.mean(latencies),
                    "median_latency_ms": statistics.median(latencies),
                    "p95_latency_ms": statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 5 else max(latencies),
                    "max_latency_ms": max(latencies),
                    "min_latency_ms": min(latencies)
                }

        return analysis

    def _analyze_error_patterns(self, traces: list[TraceData]) -> dict[str, Any]:
        """Analyze error patterns in traces"""
        error_traces = [t for t in traces if t.status == "ERROR"]

        error_analysis = {
            "total_errors": len(error_traces),
            "error_rate": len(error_traces) / len(traces) if traces else 0,
            "errors_by_operation": {},
            "errors_by_service": {}
        }

        for trace in error_traces:
            # Group by operation
            op_name = trace.operation_name
            error_analysis["errors_by_operation"][op_name] = \
                error_analysis["errors_by_operation"].get(op_name, 0) + 1

            # Group by service
            service_name = trace.tags.get("service.name", "unknown")
            error_analysis["errors_by_service"][service_name] = \
                error_analysis["errors_by_service"].get(service_name, 0) + 1

        return error_analysis

    def _identify_bottlenecks(self, traces: list[TraceData]) -> list[dict[str, Any]]:
        """Identify performance bottlenecks from trace data"""
        bottlenecks = []

        # Analyze operation latencies
        latency_analysis = self._analyze_latency_patterns(traces)

        for operation, stats in latency_analysis.items():
            if stats["p95_latency_ms"] > 5000:  # P95 > 5 seconds
                bottlenecks.append({
                    "type": "high_latency_operation",
                    "operation": operation,
                    "p95_latency_ms": stats["p95_latency_ms"],
                    "severity": "critical" if stats["p95_latency_ms"] > 10000 else "warning"
                })

            if stats["count"] > 100 and stats["avg_latency_ms"] > 1000:  # High volume + slow
                bottlenecks.append({
                    "type": "high_volume_slow_operation",
                    "operation": operation,
                    "request_count": stats["count"],
                    "avg_latency_ms": stats["avg_latency_ms"],
                    "severity": "warning"
                })

        return bottlenecks

    def _generate_trace_recommendations(self, latency_analysis: dict[str, Any],
                                      error_analysis: dict[str, Any],
                                      bottlenecks: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on trace analysis"""
        recommendations = []

        # Latency recommendations
        slow_operations = [op for op, stats in latency_analysis.items()
                         if stats.get("p95_latency_ms", 0) > 2000]

        if slow_operations:
            recommendations.append("Optimize slow operations: " + ", ".join(slow_operations[:3]))

        # Error recommendations
        if error_analysis["error_rate"] > 0.05:  # >5% error rate
            recommendations.append(f"High error rate ({error_analysis['error_rate']:.1%}) - investigate error causes")

        # Bottleneck recommendations
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "high_latency_operation":
                recommendations.append(f"Optimize {bottleneck['operation']} - P95 latency is {bottleneck['p95_latency_ms']:.0f}ms")

        # General recommendations
        recommendations.extend([
            "Implement distributed caching for frequently accessed data",
            "Consider async processing for non-critical operations",
            "Monitor service dependencies for cascading failures",
            "Implement circuit breakers for external service calls"
        ])

        return recommendations

    def _save_trace_report(self, report: DistributedTraceReport) -> None:
        """Save trace analysis report"""
        output_path = config.get_output_path(MetricType.NETWORK, f"{report.report_id}.json")

        report_dict = {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "trace_count": len(report.traces),
            "service_map": report.service_map,
            "latency_analysis": report.latency_analysis,
            "error_analysis": report.error_analysis,
            "bottlenecks": report.bottlenecks,
            "recommendations": report.recommendations
        }

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        print(f"Trace analysis report saved: {output_path}")

# Global OpenTelemetry profiler instance
otel_profiler = OpenTelemetryProfiler()

# Decorators for automatic tracing
def trace_operation(operation_name: str, **attributes):
    """Decorator to trace function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with otel_profiler.trace_operation(operation_name, **attributes):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def trace_async_operation(operation_name: str, **attributes):
    """Decorator to trace async function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with otel_profiler.trace_operation(operation_name, **attributes):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Context manager for manual tracing
@contextmanager
def trace_span(operation_name: str, **attributes):
    """Context manager for manual span creation"""
    with otel_profiler.trace_operation(operation_name, **attributes) as span:
        yield span
