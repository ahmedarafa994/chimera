"""
Comprehensive performance instrumentation for Chimera FastAPI backend
Provides metrics collection, tracing, and profiling capabilities
"""

import functools
import threading
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Any

import psutil
from opentelemetry import metrics, trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram

# Context variables for request tracking
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='')
user_id_ctx: ContextVar[str] = ContextVar('user_id', default='')

class PerformanceCollector:
    """Central performance data collector"""

    def __init__(self):
        self.setup_prometheus_metrics()
        self.setup_opentelemetry()
        self.setup_custom_metrics()
        self.start_background_monitoring()

    def setup_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # HTTP Request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code', 'provider']
        )

        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint', 'status_code'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        # LLM Provider metrics
        self.llm_requests_total = Counter(
            'chimera_llm_requests_total',
            'Total LLM provider requests',
            ['provider', 'model', 'status', 'technique']
        )

        self.llm_request_duration = Histogram(
            'chimera_llm_request_duration_seconds',
            'LLM request duration',
            ['provider', 'model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )

        self.llm_provider_availability = Gauge(
            'chimera_llm_provider_availability',
            'LLM provider availability (0-1)',
            ['provider']
        )

        self.llm_token_usage = Counter(
            'chimera_llm_tokens_total',
            'Total tokens used',
            ['provider', 'model', 'type']  # type: prompt_tokens, completion_tokens
        )

        # Transformation metrics
        self.transformation_requests_total = Counter(
            'chimera_transformation_requests_total',
            'Total transformation requests',
            ['technique', 'status']
        )

        self.transformation_duration = Histogram(
            'chimera_transformation_duration_seconds',
            'Transformation processing duration',
            ['technique'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )

        self.transformation_timeouts = Counter(
            'chimera_transformation_timeouts_total',
            'Total transformation timeouts',
            ['technique']
        )

        # WebSocket metrics
        self.websocket_connections = Gauge(
            'chimera_websocket_connections',
            'Active WebSocket connections'
        )

        self.websocket_messages = Counter(
            'chimera_websocket_messages_total',
            'Total WebSocket messages',
            ['type', 'direction']  # type: enhance, heartbeat; direction: inbound, outbound
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'chimera_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['provider']
        )

        self.circuit_breaker_failures = Counter(
            'chimera_circuit_breaker_failures_total',
            'Circuit breaker failures',
            ['provider']
        )

        # Cache metrics
        self.cache_operations = Counter(
            'chimera_cache_operations_total',
            'Cache operations',
            ['operation', 'result']  # operation: get, set, delete; result: hit, miss, success, error
        )

        # System metrics
        self.memory_usage = Gauge(
            'chimera_memory_usage_bytes',
            'Memory usage in bytes',
            ['type']  # type: rss, vms, percent
        )

        self.cpu_usage = Gauge(
            'chimera_cpu_usage_percent',
            'CPU usage percentage'
        )

        # AutoDAN metrics
        self.autodan_optimizations = Counter(
            'chimera_autodan_optimizations_total',
            'AutoDAN optimization attempts',
            ['method', 'status']
        )

        self.autodan_duration = Histogram(
            'chimera_autodan_duration_seconds',
            'AutoDAN optimization duration',
            ['method'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
        )

        # GPTFuzz metrics
        self.gptfuzz_mutations = Counter(
            'chimera_gptfuzz_mutations_total',
            'GPTFuzz mutations',
            ['mutator', 'status']
        )

        # Data pipeline metrics
        self.pipeline_jobs = Counter(
            'chimera_pipeline_jobs_total',
            'Data pipeline jobs',
            ['stage', 'status']
        )

        self.pipeline_duration = Histogram(
            'chimera_pipeline_duration_seconds',
            'Data pipeline job duration',
            ['stage']
        )

    def setup_opentelemetry(self):
        """Configure OpenTelemetry tracing and metrics"""
        # Resource configuration
        resource = Resource.create({
            "service.name": "chimera-backend",
            "service.version": "2.0.0",
            "deployment.environment": "development"
        })

        # Tracing setup
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer = trace.get_tracer(__name__)

        # Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )

        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        # Auto-instrumentation
        FastAPIInstrumentor.instrument()
        HTTPXClientInstrumentor.instrument()
        RequestsInstrumentor.instrument()
        SQLAlchemyInstrumentor.instrument()

        # Metrics setup
        reader = PrometheusMetricReader()
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
        self.meter = metrics.get_meter(__name__)

        self.tracer = tracer

    def setup_custom_metrics(self):
        """Setup custom application metrics"""
        self.request_times = deque(maxlen=1000)
        self.error_rates = defaultdict(lambda: deque(maxlen=100))
        self.performance_baseline = {}
        self.hot_paths = defaultdict(int)

    def start_background_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while True:
                try:
                    process = psutil.Process()

                    # Memory metrics
                    memory_info = process.memory_info()
                    self.memory_usage.labels(type='rss').set(memory_info.rss)
                    self.memory_usage.labels(type='vms').set(memory_info.vms)
                    self.memory_usage.labels(type='percent').set(process.memory_percent())

                    # CPU metrics
                    self.cpu_usage.set(process.cpu_percent())

                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    print(f"Background monitoring error: {e}")
                    time.sleep(30)

        thread = threading.Thread(target=monitor_system, daemon=True)
        thread.start()

# Global performance collector instance
performance_collector = PerformanceCollector()

def track_llm_request(provider: str, model: str, technique: str = "none"):
    """Decorator to track LLM requests"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            # Start span
            with performance_collector.tracer.start_as_current_span(
                f"llm_request_{provider}_{model}",
                attributes={
                    "llm.provider": provider,
                    "llm.model": model,
                    "llm.technique": technique,
                }
            ) as span:
                try:
                    result = await func(*args, **kwargs)

                    # Track token usage if available
                    if hasattr(result, 'usage'):
                        usage = result.usage
                        if hasattr(usage, 'prompt_tokens'):
                            performance_collector.llm_token_usage.labels(
                                provider=provider, model=model, type='prompt_tokens'
                            ).inc(usage.prompt_tokens)
                        if hasattr(usage, 'completion_tokens'):
                            performance_collector.llm_token_usage.labels(
                                provider=provider, model=model, type='completion_tokens'
                            ).inc(usage.completion_tokens)

                    return result

                except Exception as e:
                    status = "error"
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    raise
                finally:
                    duration = time.time() - start_time

                    # Update metrics
                    performance_collector.llm_requests_total.labels(
                        provider=provider, model=model, status=status, technique=technique
                    ).inc()

                    performance_collector.llm_request_duration.labels(
                        provider=provider, model=model
                    ).observe(duration)

                    # Update span
                    span.set_attribute("llm.duration", duration)
                    span.set_attribute("llm.status", status)

        return wrapper
    return decorator

def track_transformation(technique: str):
    """Decorator to track transformation requests"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            with performance_collector.tracer.start_as_current_span(
                f"transformation_{technique}",
                attributes={"transformation.technique": technique}
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except TimeoutError:
                    status = "timeout"
                    performance_collector.transformation_timeouts.labels(
                        technique=technique
                    ).inc()
                    raise
                except Exception as e:
                    status = "error"
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    raise
                finally:
                    duration = time.time() - start_time

                    performance_collector.transformation_requests_total.labels(
                        technique=technique, status=status
                    ).inc()

                    performance_collector.transformation_duration.labels(
                        technique=technique
                    ).observe(duration)

                    span.set_attribute("transformation.duration", duration)
                    span.set_attribute("transformation.status", status)

        return wrapper
    return decorator

def track_autodan_optimization(method: str):
    """Decorator to track AutoDAN optimizations"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            with performance_collector.tracer.start_as_current_span(
                f"autodan_{method}",
                attributes={"autodan.method": method}
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    raise
                finally:
                    duration = time.time() - start_time

                    performance_collector.autodan_optimizations.labels(
                        method=method, status=status
                    ).inc()

                    performance_collector.autodan_duration.labels(
                        method=method
                    ).observe(duration)

                    span.set_attribute("autodan.duration", duration)
                    span.set_attribute("autodan.status", status)

        return wrapper
    return decorator

def track_http_request(func):
    """Decorator to track HTTP requests"""
    @functools.wraps(func)
    async def wrapper(request, *args, **kwargs):
        start_time = time.time()
        method = request.method
        path = request.url.path
        status_code = 200

        with performance_collector.tracer.start_as_current_span(
            f"{method} {path}",
            attributes={
                "http.method": method,
                "http.url": str(request.url),
                "http.route": path
            }
        ) as span:
            try:
                response = await func(request, *args, **kwargs)
                if hasattr(response, 'status_code'):
                    status_code = response.status_code
                return response
            except Exception as e:
                status_code = 500
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                duration = time.time() - start_time

                # Track hot paths
                performance_collector.hot_paths[f"{method} {path}"] += 1

                # Update metrics
                performance_collector.http_requests_total.labels(
                    method=method, endpoint=path, status_code=status_code, provider="none"
                ).inc()

                performance_collector.http_request_duration.labels(
                    method=method, endpoint=path, status_code=status_code
                ).observe(duration)

                # Update span
                span.set_attribute("http.status_code", status_code)
                span.set_attribute("http.response_time", duration)

                # Store request timing
                performance_collector.request_times.append({
                    'timestamp': datetime.now(),
                    'duration': duration,
                    'endpoint': path,
                    'method': method,
                    'status_code': status_code
                })

    return wrapper

class WebSocketTracker:
    """Track WebSocket connections and messages"""

    def __init__(self):
        self.connections = 0

    def connection_opened(self):
        self.connections += 1
        performance_collector.websocket_connections.set(self.connections)

    def connection_closed(self):
        self.connections -= 1
        performance_collector.websocket_connections.set(self.connections)

    def message_sent(self, message_type: str):
        performance_collector.websocket_messages.labels(
            type=message_type, direction="outbound"
        ).inc()

    def message_received(self, message_type: str):
        performance_collector.websocket_messages.labels(
            type=message_type, direction="inbound"
        ).inc()

class CircuitBreakerTracker:
    """Track circuit breaker metrics"""

    @staticmethod
    def update_state(provider: str, state: str):
        """Update circuit breaker state (closed=0, open=1, half-open=2)"""
        state_value = {"closed": 0, "open": 1, "half-open": 2}.get(state, 0)
        performance_collector.circuit_breaker_state.labels(provider=provider).set(state_value)

    @staticmethod
    def record_failure(provider: str):
        """Record circuit breaker failure"""
        performance_collector.circuit_breaker_failures.labels(provider=provider).inc()

class CacheTracker:
    """Track cache operations"""

    @staticmethod
    def cache_hit(operation: str):
        performance_collector.cache_operations.labels(
            operation=operation, result="hit"
        ).inc()

    @staticmethod
    def cache_miss(operation: str):
        performance_collector.cache_operations.labels(
            operation=operation, result="miss"
        ).inc()

    @staticmethod
    def cache_error(operation: str):
        performance_collector.cache_operations.labels(
            operation=operation, result="error"
        ).inc()

def get_performance_summary() -> dict[str, Any]:
    """Get current performance summary"""
    now = datetime.now()
    recent_requests = [
        r for r in performance_collector.request_times
        if now - r['timestamp'] < timedelta(minutes=5)
    ]

    if not recent_requests:
        return {"status": "no_recent_data"}

    durations = [r['duration'] for r in recent_requests]

    return {
        "timestamp": now.isoformat(),
        "request_count_5m": len(recent_requests),
        "avg_response_time": sum(durations) / len(durations),
        "p95_response_time": sorted(durations)[int(len(durations) * 0.95)],
        "p99_response_time": sorted(durations)[int(len(durations) * 0.99)],
        "hot_paths": dict(list(performance_collector.hot_paths.items())[:10]),
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_percent": psutil.Process().cpu_percent()
    }

# Global instances
websocket_tracker = WebSocketTracker()
circuit_breaker_tracker = CircuitBreakerTracker()
cache_tracker = CacheTracker()
