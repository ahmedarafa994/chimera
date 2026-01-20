"""AutoDAN Data Flow Mapper and API Connectivity Enhancements.

This module provides comprehensive data flow mapping, observability,
and enhanced API connectivity for AutoDAN modules with improved
error handling, circuit breakers, and monitoring capabilities.
"""

import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from weakref import WeakSet

import aiohttp

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of components in the data flow."""

    API_ENDPOINT = "api_endpoint"
    SERVICE = "service"
    ENGINE = "engine"
    ADAPTER = "adapter"
    CACHE = "cache"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    QUEUE = "queue"
    PIPELINE_STAGE = "pipeline_stage"


class FlowEventType(Enum):
    """Types of flow events."""

    REQUEST_START = "request_start"
    REQUEST_END = "request_end"
    COMPONENT_ENTER = "component_enter"
    COMPONENT_EXIT = "component_exit"
    ERROR_OCCURRED = "error_occurred"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    EXTERNAL_CALL = "external_call"
    QUEUE_ENQUEUE = "queue_enqueue"
    QUEUE_DEQUEUE = "queue_dequeue"


@dataclass
class FlowEvent:
    """Represents a single event in the data flow."""

    event_id: str
    request_id: str
    component_id: str
    component_type: ComponentType
    event_type: FlowEventType
    timestamp: float
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_event_id: str | None = None
    error_info: str | None = None

    @classmethod
    def create(
        cls,
        request_id: str,
        component_id: str,
        component_type: ComponentType,
        event_type: FlowEventType,
        **kwargs,
    ) -> "FlowEvent":
        """Create a new flow event."""
        return cls(
            event_id=str(uuid.uuid4()),
            request_id=request_id,
            component_id=component_id,
            component_type=component_type,
            event_type=event_type,
            timestamp=time.time(),
            **kwargs,
        )


@dataclass
class ComponentDependency:
    """Represents a dependency between components."""

    from_component: str
    to_component: str
    dependency_type: str  # "calls", "uses", "depends_on", "publishes_to"
    weight: float = 1.0  # Strength of dependency
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowTrace:
    """Complete trace of a request flow."""

    request_id: str
    start_time: float
    end_time: float | None = None
    events: list[FlowEvent] = field(default_factory=list)
    components_touched: set[str] = field(default_factory=set)
    total_duration_ms: float | None = None
    error_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    external_call_count: int = 0

    def add_event(self, event: FlowEvent) -> None:
        """Add an event to the trace."""
        self.events.append(event)
        self.components_touched.add(event.component_id)

        # Update counters
        if event.event_type == FlowEventType.ERROR_OCCURRED:
            self.error_count += 1
        elif event.event_type == FlowEventType.CACHE_HIT:
            self.cache_hit_count += 1
        elif event.event_type == FlowEventType.CACHE_MISS:
            self.cache_miss_count += 1
        elif event.event_type == FlowEventType.EXTERNAL_CALL:
            self.external_call_count += 1

    def finalize(self) -> None:
        """Finalize the trace."""
        self.end_time = time.time()
        if self.start_time:
            self.total_duration_ms = (self.end_time - self.start_time) * 1000

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total_cache_operations = self.cache_hit_count + self.cache_miss_count
        if total_cache_operations == 0:
            return 0.0
        return self.cache_hit_count / total_cache_operations

    def get_component_timeline(self) -> dict[str, list[FlowEvent]]:
        """Get events organized by component."""
        timeline = defaultdict(list)
        for event in self.events:
            timeline[event.component_id].append(event)
        return dict(timeline)

    def get_critical_path(self) -> list[FlowEvent]:
        """Get the critical path (longest execution chain)."""
        # Simplified critical path - could be enhanced with proper algorithm
        return sorted(self.events, key=lambda e: e.timestamp)


class DataFlowTracer:
    """Traces data flow across AutoDAN components."""

    def __init__(self, max_traces: int = 1000) -> None:
        self.max_traces = max_traces
        self._active_traces: dict[str, FlowTrace] = {}
        self._completed_traces: deque = deque(maxlen=max_traces)
        self._event_subscribers: WeakSet = WeakSet()
        self.logger = logging.getLogger(f"{__name__}.tracer")

    def start_trace(self, request_id: str) -> FlowTrace:
        """Start a new trace for a request."""
        trace = FlowTrace(request_id=request_id, start_time=time.time())
        self._active_traces[request_id] = trace
        self.logger.debug(f"Started trace for request {request_id}")
        return trace

    def add_event(self, event: FlowEvent) -> None:
        """Add an event to the appropriate trace."""
        trace = self._active_traces.get(event.request_id)
        if trace:
            trace.add_event(event)
            # Notify subscribers
            for subscriber in self._event_subscribers:
                try:
                    if hasattr(subscriber, "on_flow_event"):
                        subscriber.on_flow_event(event)
                except Exception as e:
                    self.logger.warning(f"Error notifying subscriber: {e}")

    def complete_trace(self, request_id: str) -> FlowTrace | None:
        """Complete a trace and move it to completed traces."""
        trace = self._active_traces.pop(request_id, None)
        if trace:
            trace.finalize()
            self._completed_traces.append(trace)
            self.logger.debug(
                f"Completed trace for {request_id} - "
                f"duration: {trace.total_duration_ms:.2f}ms, "
                f"components: {len(trace.components_touched)}, "
                f"events: {len(trace.events)}",
            )
        return trace

    def get_trace(self, request_id: str) -> FlowTrace | None:
        """Get a trace by request ID."""
        # Check active traces first
        if request_id in self._active_traces:
            return self._active_traces[request_id]

        # Check completed traces
        for trace in reversed(self._completed_traces):
            if trace.request_id == request_id:
                return trace

        return None

    def get_recent_traces(self, limit: int = 10) -> list[FlowTrace]:
        """Get recent completed traces."""
        return list(reversed(self._completed_traces))[:limit]

    def subscribe_to_events(self, subscriber) -> None:
        """Subscribe to flow events."""
        self._event_subscribers.add(subscriber)

    def get_statistics(self) -> dict[str, Any]:
        """Get tracing statistics."""
        return {
            "active_traces": len(self._active_traces),
            "completed_traces": len(self._completed_traces),
            "max_traces": self.max_traces,
            "subscribers": len(self._event_subscribers),
        }


class BottleneckAnalyzer:
    """Analyzes performance bottlenecks in data flow."""

    def __init__(self, tracer: DataFlowTracer) -> None:
        self.tracer = tracer
        self.logger = logging.getLogger(f"{__name__}.bottleneck_analyzer")

    def analyze_recent_traces(self, trace_count: int = 50) -> dict[str, Any]:
        """Analyze recent traces for bottlenecks."""
        traces = self.tracer.get_recent_traces(trace_count)
        if not traces:
            return {"error": "No traces available for analysis"}

        component_stats = defaultdict(list)
        total_durations = []

        # Collect component performance data
        for trace in traces:
            if trace.total_duration_ms:
                total_durations.append(trace.total_duration_ms)

            component_timeline = trace.get_component_timeline()
            for component_id, events in component_timeline.items():
                enter_events = [e for e in events if e.event_type == FlowEventType.COMPONENT_ENTER]
                exit_events = [e for e in events if e.event_type == FlowEventType.COMPONENT_EXIT]

                # Calculate component duration
                if enter_events and exit_events:
                    duration = max(e.timestamp for e in exit_events) - min(
                        e.timestamp for e in enter_events
                    )
                    component_stats[component_id].append(duration * 1000)  # Convert to ms

        # Calculate statistics
        analysis = {
            "trace_count": len(traces),
            "avg_total_duration_ms": (
                sum(total_durations) / len(total_durations) if total_durations else 0
            ),
            "component_analysis": {},
            "bottlenecks": [],
            "recommendations": [],
        }

        # Analyze components
        for component_id, durations in component_stats.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)

                analysis["component_analysis"][component_id] = {
                    "avg_duration_ms": avg_duration,
                    "max_duration_ms": max_duration,
                    "min_duration_ms": min_duration,
                    "call_count": len(durations),
                    "total_time_ms": sum(durations),
                }

                # Identify bottlenecks (components taking >20% of average total time)
                if analysis["avg_total_duration_ms"] > 0:
                    time_percentage = (avg_duration / analysis["avg_total_duration_ms"]) * 100
                    if time_percentage > 20:
                        analysis["bottlenecks"].append(
                            {
                                "component": component_id,
                                "avg_duration_ms": avg_duration,
                                "time_percentage": time_percentage,
                                "severity": "high" if time_percentage > 40 else "medium",
                            },
                        )

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        for bottleneck in analysis["bottlenecks"]:
            component = bottleneck["component"]
            severity = bottleneck["severity"]

            if severity == "high":
                recommendations.append(
                    f"Critical: Optimize {component} - taking {bottleneck['time_percentage']:.1f}% "
                    f"of total execution time",
                )
            else:
                recommendations.append(
                    f"Consider optimizing {component} - moderate impact on performance",
                )

        if analysis["avg_total_duration_ms"] > 2000:  # > 2 seconds
            recommendations.append("Overall response time is high - consider parallel processing")

        return recommendations


class DependencyGraph:
    """Manages component dependencies and relationships."""

    def __init__(self) -> None:
        self.dependencies: list[ComponentDependency] = []
        self.components: set[str] = set()
        self.logger = logging.getLogger(f"{__name__}.dependency_graph")

    def add_dependency(self, dependency: ComponentDependency) -> None:
        """Add a component dependency."""
        self.dependencies.append(dependency)
        self.components.add(dependency.from_component)
        self.components.add(dependency.to_component)

    def get_dependencies_for_component(self, component_id: str) -> list[ComponentDependency]:
        """Get all dependencies for a component."""
        return [dep for dep in self.dependencies if dep.from_component == component_id]

    def get_dependents_of_component(self, component_id: str) -> list[ComponentDependency]:
        """Get all components that depend on this component."""
        return [dep for dep in self.dependencies if dep.to_component == component_id]

    def validate_dependencies(self) -> dict[str, Any]:
        """Validate dependency graph for cycles and orphans."""
        validation_result = {
            "valid": True,
            "cycles": [],
            "orphaned_components": [],
            "strongly_connected": True,
            "warnings": [],
        }

        # Check for cycles using DFS
        cycles = self._detect_cycles()
        if cycles:
            validation_result["valid"] = False
            validation_result["cycles"] = cycles

        # Check for orphaned components
        referenced_components = set()
        for dep in self.dependencies:
            referenced_components.add(dep.from_component)
            referenced_components.add(dep.to_component)

        orphaned = self.components - referenced_components
        if orphaned:
            validation_result["orphaned_components"] = list(orphaned)
            validation_result["warnings"].append(f"Found {len(orphaned)} orphaned components")

        return validation_result

    def _detect_cycles(self) -> list[list[str]]:
        """Detect cycles in the dependency graph using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(component, path) -> None:
            if component in rec_stack:
                # Found a cycle
                cycle_start = path.index(component)
                cycles.append([*path[cycle_start:], component])
                return

            if component in visited:
                return

            visited.add(component)
            rec_stack.add(component)

            # Visit dependencies
            for dep in self.get_dependencies_for_component(component):
                dfs(dep.to_component, [*path, component])

            rec_stack.remove(component)

        for component in self.components:
            if component not in visited:
                dfs(component, [])

        return cycles

    def get_dependency_matrix(self) -> dict[str, dict[str, float]]:
        """Get dependency matrix with weights."""
        matrix = defaultdict(lambda: defaultdict(float))

        for dep in self.dependencies:
            matrix[dep.from_component][dep.to_component] = dep.weight

        return {k: dict(v) for k, v in matrix.items()}

    def get_critical_components(self) -> list[tuple[str, int]]:
        """Get components with highest dependency counts."""
        dependency_counts = defaultdict(int)

        for dep in self.dependencies:
            dependency_counts[dep.to_component] += 1

        return sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    timeout_seconds: int = 60
    success_threshold: int = 3
    monitor_window_seconds: int = 300


class CircuitBreaker:
    """Circuit breaker for API calls."""

    def __init__(self, name: str, config: CircuitBreakerConfig = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.failure_times: deque = deque(maxlen=100)
        self.logger = logging.getLogger(f"{__name__}.circuit_breaker.{name}")

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if current_time - self.last_failure_time > self.config.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                return True
            return False
        return self.state == CircuitBreakerState.HALF_OPEN

    def record_success(self) -> None:
        """Record a successful execution."""
        current_time = time.time()
        self.last_success_time = current_time

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on successful execution
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed execution."""
        current_time = time.time()
        self.last_failure_time = current_time
        self.failure_times.append(current_time)
        self.failure_count += 1

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(
                    f"Circuit breaker {self.name} transitioning to OPEN "
                    f"after {self.failure_count} failures",
                )
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} returning to OPEN from HALF_OPEN")

    def get_failure_rate(self, window_seconds: int | None = None) -> float:
        """Get failure rate within time window."""
        if not self.failure_times:
            return 0.0

        window_seconds = window_seconds or self.config.monitor_window_seconds
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        recent_failures = sum(1 for t in self.failure_times if t > cutoff_time)
        return recent_failures / len(self.failure_times)

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "failure_rate": self.get_failure_rate(),
            "recent_failures": len([t for t in self.failure_times if t > time.time() - 300]),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "success_threshold": self.config.success_threshold,
            },
        }


class EnhancedAPIConnector:
    """Enhanced API connector with circuit breakers and monitoring."""

    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.data_flow_tracer = DataFlowTracer()
        self.dependency_graph = DependencyGraph()
        self.performance_monitor = PerformanceMonitor()
        self.logger = logging.getLogger(f"{__name__}.api_connector")

        # Initialize built-in dependencies
        self._initialize_dependencies()

    def _initialize_dependencies(self) -> None:
        """Initialize AutoDAN component dependencies."""
        dependencies = [
            ComponentDependency("autodan_api", "autodan_service", "calls"),
            ComponentDependency("autodan_service", "strategy_registry", "uses"),
            ComponentDependency("autodan_service", "llm_adapter", "calls"),
            ComponentDependency("llm_adapter", "provider_service", "uses"),
            ComponentDependency("strategy_registry", "genetic_optimizer", "calls"),
            ComponentDependency("strategy_registry", "mousetrap_engine", "calls"),
            ComponentDependency("autodan_service", "cache_manager", "uses"),
            ComponentDependency("autodan_service", "pipeline_processor", "uses"),
        ]

        for dep in dependencies:
            self.dependency_graph.add_dependency(dep)

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        """Get or create circuit breaker for endpoint."""
        if endpoint not in self.circuit_breakers:
            self.circuit_breakers[endpoint] = CircuitBreaker(f"api_{endpoint}")
        return self.circuit_breakers[endpoint]

    @asynccontextmanager
    async def traced_request(self, request_id: str, endpoint: str):
        """Context manager for traced API requests."""
        # Start trace
        trace = self.data_flow_tracer.start_trace(request_id)

        # Add request start event
        start_event = FlowEvent.create(
            request_id=request_id,
            component_id="api_connector",
            component_type=ComponentType.ADAPTER,
            event_type=FlowEventType.REQUEST_START,
            metadata={"endpoint": endpoint},
        )
        self.data_flow_tracer.add_event(start_event)

        start_time = time.time()
        try:
            yield trace

            # Add success event
            end_event = FlowEvent.create(
                request_id=request_id,
                component_id="api_connector",
                component_type=ComponentType.ADAPTER,
                event_type=FlowEventType.REQUEST_END,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"endpoint": endpoint, "status": "success"},
            )
            self.data_flow_tracer.add_event(end_event)

        except Exception as e:
            # Add error event
            error_event = FlowEvent.create(
                request_id=request_id,
                component_id="api_connector",
                component_type=ComponentType.ADAPTER,
                event_type=FlowEventType.ERROR_OCCURRED,
                duration_ms=(time.time() - start_time) * 1000,
                error_info=str(e),
                metadata={"endpoint": endpoint, "error": str(e)},
            )
            self.data_flow_tracer.add_event(error_event)
            raise
        finally:
            self.data_flow_tracer.complete_trace(request_id)

    async def make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Make an API request with circuit breaker protection and tracing.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            headers: Request headers
            timeout: Request timeout
            request_id: Request ID for tracing

        Returns:
            API response

        Raises:
            Exception: If request fails or circuit breaker is open

        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        request_id = request_id or str(uuid.uuid4())
        circuit_breaker = self.get_circuit_breaker(endpoint)

        # Check circuit breaker
        if not circuit_breaker.can_execute():
            msg = f"Circuit breaker is OPEN for endpoint {endpoint}"
            raise Exception(msg)

        async with self.traced_request(request_id, endpoint):
            try:
                url = f"{self.base_url}/{endpoint}".rstrip("/")

                # Prepare request
                kwargs = {"timeout": aiohttp.ClientTimeout(total=timeout), "headers": headers or {}}

                if data:
                    kwargs["json"] = data

                # Make request
                async with self.session.request(method, url, **kwargs) as response:
                    response_data = await response.json()

                    if response.status >= 400:
                        circuit_breaker.record_failure()
                        msg = f"API error {response.status}: {response_data}"
                        raise Exception(msg)

                    circuit_breaker.record_success()
                    return response_data

            except Exception as e:
                circuit_breaker.record_failure()
                self.logger.exception(f"Request failed for {endpoint}: {e}")
                raise

    async def execute_autodan_request(
        self,
        request_data: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute AutoDAN request with enhanced monitoring."""
        request_id = request_id or str(uuid.uuid4())

        # Add component entry event
        entry_event = FlowEvent.create(
            request_id=request_id,
            component_id="autodan_service",
            component_type=ComponentType.SERVICE,
            event_type=FlowEventType.COMPONENT_ENTER,
        )
        self.data_flow_tracer.add_event(entry_event)

        try:
            # Execute via pipeline
            from backend_api.app.core.autodan_pipeline import execute_autodan_pipeline

            result = await execute_autodan_pipeline(request_data)

            # Add component exit event
            exit_event = FlowEvent.create(
                request_id=request_id,
                component_id="autodan_service",
                component_type=ComponentType.SERVICE,
                event_type=FlowEventType.COMPONENT_EXIT,
                metadata={"success": result.get("success", False)},
            )
            self.data_flow_tracer.add_event(exit_event)

            return result

        except Exception as e:
            # Add error event
            error_event = FlowEvent.create(
                request_id=request_id,
                component_id="autodan_service",
                component_type=ComponentType.SERVICE,
                event_type=FlowEventType.ERROR_OCCURRED,
                error_info=str(e),
            )
            self.data_flow_tracer.add_event(error_event)
            raise

    def get_flow_trace(self, request_id: str) -> FlowTrace | None:
        """Get flow trace for a request."""
        return self.data_flow_tracer.get_trace(request_id)

    def analyze_bottlenecks(self) -> dict[str, Any]:
        """Analyze performance bottlenecks."""
        analyzer = BottleneckAnalyzer(self.data_flow_tracer)
        return analyzer.analyze_recent_traces()

    def validate_dependencies(self) -> dict[str, Any]:
        """Validate component dependencies."""
        return self.dependency_graph.validate_dependencies()

    def get_connectivity_stats(self) -> dict[str, Any]:
        """Get comprehensive connectivity statistics."""
        return {
            "circuit_breakers": {
                name: breaker.get_stats() for name, breaker in self.circuit_breakers.items()
            },
            "data_flow_tracer": self.data_flow_tracer.get_statistics(),
            "dependency_validation": self.validate_dependencies(),
            "critical_components": self.dependency_graph.get_critical_components()[:5],
            "performance_analysis": self.analyze_bottlenecks(),
        }


class PerformanceMonitor:
    """Monitors performance across AutoDAN components."""

    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger(f"{__name__}.performance_monitor")

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        self.metrics[metric_name].append(value)
        # Keep only last 1000 values
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]

    def increment_counter(self, counter_name: str, increment: int = 1) -> None:
        """Increment a counter."""
        self.counters[counter_name] += increment

    def get_metric_stats(self, metric_name: str) -> dict[str, float] | None:
        """Get statistics for a metric."""
        values = self.metrics.get(metric_name)
        if not values:
            return None

        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p50": sorted(values)[len(values) // 2],
            "p95": sorted(values)[int(len(values) * 0.95)],
            "p99": sorted(values)[int(len(values) * 0.99)],
        }

    def get_all_stats(self) -> dict[str, Any]:
        """Get all performance statistics."""
        return {
            "metrics": {name: self.get_metric_stats(name) for name in self.metrics},
            "counters": dict(self.counters),
        }


# Global instances
_global_tracer: DataFlowTracer | None = None
_global_connector: EnhancedAPIConnector | None = None


def get_data_flow_tracer() -> DataFlowTracer:
    """Get or create global data flow tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DataFlowTracer()
    return _global_tracer


def get_api_connector() -> EnhancedAPIConnector:
    """Get or create global API connector."""
    global _global_connector
    if _global_connector is None:
        _global_connector = EnhancedAPIConnector()
    return _global_connector


async def trace_autodan_request(request_data: dict[str, Any]) -> dict[str, Any]:
    """Convenience function to trace an AutoDAN request."""
    connector = get_api_connector()
    return await connector.execute_autodan_request(request_data)
