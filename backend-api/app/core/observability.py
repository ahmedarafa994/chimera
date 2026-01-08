"""
Observability Module
Structured logging, metrics, and request tracing for production monitoring
"""

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from typing import Any, ClassVar

import sentry_sdk
from fastapi import Request, Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram, generate_latest
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable for request-scoped data
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


# =============================================================================
# Structured JSON Logging
# =============================================================================


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs in JSON format for easy parsing by log aggregation tools
    like ELK, Datadog, Splunk, etc.
    """

    RESERVED_ATTRS: ClassVar[set[str]] = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_record["request_id"] = request_id

        # Add exception info
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            log_record["exception_type"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else None
            )

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)  # Check if serializable
                    log_record[key] = value
                except (TypeError, ValueError):
                    log_record[key] = str(value)

        return json.dumps(log_record, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for development.
    """

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        request_id = request_id_var.get()

        prefix = f"[{request_id[:8]}] " if request_id else ""

        formatted = (
            f"{color}{record.levelname:8}{self.RESET} "
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"{prefix}"
            f"{record.name}: {record.getMessage()}"
        )

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(
    level: str = "INFO", json_format: bool = True, log_file: str | None = None
) -> logging.Logger:
    """
    Set up logging with appropriate formatter and Sentry integration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True for production, False for development)
        log_file: Optional log file path

    Returns:
        Configured root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)

    # Set third-party loggers to WARNING
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "httpx"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Initialize Sentry if DSN is provided
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,  # Capture info and above as breadcrumbs
            event_level=logging.ERROR,  # Send errors as events
        )
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[
                FastApiIntegration(),
                sentry_logging,
            ],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=os.getenv("ENVIRONMENT", "development"),
        )

    return root_logger


def setup_tracing(app):
    """
    Set up OpenTelemetry tracing.
    """
    otlp_endpoint = os.getenv("OTLP_ENDPOINT")
    if otlp_endpoint:
        resource = Resource.create(
            attributes={
                "service.name": "chimera-backend",
                "service.version": "2.0.0",
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)

        FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# =============================================================================
# Request/Response Logging Middleware
# =============================================================================


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging and metrics.

    Features:
    - Unique request ID generation and propagation
    - Request/response timing
    - Structured logging
    - Error tracking

    PERF-022: Optimized to skip logging for excluded paths and reduce duplicate logging.
    This middleware now only handles request ID propagation and response headers.
    Detailed logging is handled by RequestLoggingMiddleware to avoid duplication.
    """

    # Paths to exclude from detailed logging (PERF-023: Extended exclusion list)
    EXCLUDE_PATHS: ClassVar[set[str]] = {
        "/health",
        "/health/ping",
        "/health/ready",
        "/health/live",
        "/metrics",
        "/favicon.ico",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    # PERF-024: Known non-existent paths that generate high 404 traffic
    # These are typically from clients expecting OpenAI/Anthropic API compatibility
    KNOWN_404_PREFIXES: ClassVar[set[str]] = {"/v1/chat/", "/v1/messages", "/v1/models", "/v1/api/event_logging"}

    def __init__(self, app, logger: logging.Logger | None = None):
        super().__init__(app)
        self.logger = logger or get_logger("chimera.http")
        # PERF-025: Track 404 counts to avoid log spam
        self._404_counts: dict[str, int] = {}
        self._404_log_threshold = 10  # Log every Nth 404 for known patterns

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request_id_var.set(request_id)

        # Record start time
        start_time = time.perf_counter()

        # Extract client info
        client_ip = self._get_client_ip(request)

        # PERF-023: Skip detailed logging for excluded paths
        path = request.url.path
        skip_logging = path in self.EXCLUDE_PATHS or path.startswith("/health")

        # PERF-024: Check if this is a known 404 pattern
        is_known_404_pattern = any(path.startswith(prefix) for prefix in self.KNOWN_404_PREFIXES)

        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # PERF-022: Only log if not handled by RequestLoggingMiddleware
            # and not a known 404 pattern (to reduce log spam)
            if not skip_logging:
                should_log = True

                # PERF-025: Rate-limit logging for known 404 patterns
                if response.status_code == 404 and is_known_404_pattern:
                    self._404_counts[path] = self._404_counts.get(path, 0) + 1
                    should_log = self._404_counts[path] % self._404_log_threshold == 1

                if should_log:
                    self._log_request(
                        request=request,
                        response=response,
                        duration_ms=duration_ms,
                        client_ip=client_ip,
                        request_id=request_id,
                    )

            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            self._log_error(
                request=request,
                error=e,
                duration_ms=duration_ms,
                client_ip=client_ip,
                request_id=request_id,
            )

            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, considering proxy headers."""
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"

    def _log_request(
        self,
        request: Request,
        response: Response,
        duration_ms: float,
        client_ip: str,
        request_id: str,
    ):
        """Log a successful request."""
        log_level = logging.WARNING if response.status_code >= 400 else logging.INFO

        self.logger.log(
            log_level,
            f"{request.method} {request.url.path} {response.status_code}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params) if request.query_params else None,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "client_ip": client_ip,
                "user_agent": request.headers.get("User-Agent"),
                "content_length": response.headers.get("Content-Length"),
            },
        )

    def _log_error(
        self,
        request: Request,
        error: Exception,
        duration_ms: float,
        client_ip: str,
        request_id: str,
    ):
        """Log a request error."""
        self.logger.error(
            f"{request.method} {request.url.path} ERROR: {type(error).__name__}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration_ms, 2),
                "client_ip": client_ip,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            exc_info=True,
        )


# =============================================================================
# Metrics Collection (Prometheus)
# =============================================================================


class MetricsCollector:
    """
    Metrics collector using official prometheus_client.
    """

    def __init__(self):
        # Define standard metrics
        self.http_requests_total = Counter(
            "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
        )
        self.http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
        )
        self.jailbreak_attempts_total = Counter(
            "jailbreak_attempts_total", "Total jailbreak attempts", ["technique", "status"]
        )
        self.model_tokens_total = Counter(
            "model_tokens_total",
            "Total tokens consumed",
            ["model", "type"],  # type: prompt or completion
        )

    def increment(self, name: str, value: int = 1, labels: dict[str, str] | None = None):
        """Increment a counter metric."""
        # Map legacy calls to new counters if possible, or ignore
        if name == "http_requests_total":
            self.http_requests_total.labels(**(labels or {})).inc(value)
        elif name == "jailbreak_attempts":
            self.jailbreak_attempts_total.labels(**(labels or {})).inc(value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None):
        """Record a value for histogram."""
        if name == "http_request_duration_seconds":
            self.http_request_duration_seconds.labels(**(labels or {})).observe(value)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest().decode("utf-8")


# Global metrics collector
metrics = MetricsCollector()


# =============================================================================
# Health Check Utilities
# =============================================================================


class HealthCheck:
    """Health check manager for service dependencies."""

    def __init__(self):
        self.checks: dict[str, Callable[[], bool]] = {}

    def register(self, name: str, check: Callable[[], bool]):
        """Register a health check function."""
        self.checks[name] = check

    async def run_all(self) -> dict[str, Any]:
        """Run all health checks."""
        results = {}
        all_healthy = True

        for name, check in self.checks.items():
            try:
                is_healthy = check()
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "healthy": is_healthy,
                }
                if not is_healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {"status": "error", "healthy": False, "error": str(e)}
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": results,
        }


health_check = HealthCheck()


# =============================================================================
# Alerting Utilities
# =============================================================================


class AlertManager:
    """Simple alert manager for critical events."""

    def __init__(self):
        self.logger = get_logger("chimera.alerts")
        self.alert_callbacks: list = []

    def register_callback(self, callback: Callable[[str, str, dict], None]):
        """Register an alert callback (e.g., Slack, PagerDuty)."""
        self.alert_callbacks.append(callback)

    def alert(self, severity: str, message: str, details: dict[str, Any] | None = None):
        """
        Send an alert.

        Args:
            severity: "critical", "warning", or "info"
            message: Alert message
            details: Additional details
        """
        self.logger.log(
            logging.CRITICAL if severity == "critical" else logging.WARNING,
            f"ALERT [{severity.upper()}]: {message}",
            extra={"alert": True, "severity": severity, "details": details or {}},
        )

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(severity, message, details or {})
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    def critical(self, message: str, details: dict[str, Any] | None = None):
        """Send a critical alert."""
        self.alert("critical", message, details)

    def warning(self, message: str, details: dict[str, Any] | None = None):
        """Send a warning alert."""
        self.alert("warning", message, details)


alert_manager = AlertManager()
