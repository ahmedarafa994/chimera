"""
Structured Logging Module for Chimera

MED-001 FIX: Implements JSON-structured logging with correlation IDs
for distributed tracing and log aggregation.

Features:
- JSON-formatted log output for log aggregation systems
- Correlation ID propagation across async contexts
- Request context tracking
- Performance metrics logging
- Configurable log levels and formatters

Usage:
    from app.core.shared.structured_logging import (
        get_logger,
        set_correlation_id,
        log_with_context,
        RequestContextMiddleware
    )

    logger = get_logger(__name__)
    logger.info("Processing request", extra={"user_id": "123"})
"""

import contextvars
import json
import logging
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any, ClassVar, TypeVar

# Context variables for request tracking
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)
request_context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "request_context", default={}
)


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return f"corr_{uuid.uuid4().hex[:16]}"


def get_correlation_id() -> str:
    """Get the current correlation ID from context."""
    return correlation_id_var.get() or generate_correlation_id()


def set_correlation_id(correlation_id: str) -> contextvars.Token:
    """Set the correlation ID in the current context."""
    return correlation_id_var.set(correlation_id)


def get_request_context() -> dict[str, Any]:
    """Get the current request context."""
    return request_context_var.get()


def set_request_context(context: dict[str, Any]) -> contextvars.Token:
    """Set the request context."""
    return request_context_var.set(context)


def update_request_context(**kwargs) -> None:
    """Update the current request context with additional fields."""
    current = request_context_var.get().copy()
    current.update(kwargs)
    request_context_var.set(current)


class StructuredLogFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs in JSON format suitable for log aggregation systems
    like ELK Stack, Datadog, or CloudWatch.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_correlation_id: bool = True,
        include_request_context: bool = True,
        include_exception: bool = True,
        extra_fields: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_correlation_id = include_correlation_id
        self.include_request_context = include_request_context
        self.include_exception = include_exception
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {}

        # Timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Log level
        if self.include_level:
            log_data["level"] = record.levelname
            log_data["level_num"] = record.levelno

        # Logger name
        if self.include_logger:
            log_data["logger"] = record.name

        # Message
        log_data["message"] = record.getMessage()

        # Correlation ID
        if self.include_correlation_id:
            correlation_id = get_correlation_id()
            if correlation_id:
                log_data["correlation_id"] = correlation_id

        # Request context
        if self.include_request_context:
            request_context = get_request_context()
            if request_context:
                log_data["request"] = request_context

        # Source location
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Exception info
        if self.include_exception and record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Extra fields from record
        extra_keys = set(record.__dict__.keys()) - {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "message",
            "taskName",
        }

        if extra_keys:
            log_data["extra"] = {
                key: getattr(record, key) for key in extra_keys if not key.startswith("_")
            }

        # Static extra fields
        if self.extra_fields:
            log_data.update(self.extra_fields)

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development environments.

    Provides colored, readable output while still including
    correlation IDs and context.
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
        """Format log record for human readability."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Get correlation ID
        correlation_id = get_correlation_id()
        corr_str = f"[{correlation_id[:12]}]" if correlation_id else ""

        # Format timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Build message
        parts = [
            f"{timestamp}",
            f"{color}{record.levelname:8}{reset}",
            corr_str,
            f"{record.name}:",
            record.getMessage(),
        ]

        message = " ".join(filter(None, parts))

        # Add exception info if present
        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return message


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    include_correlation_id: bool = True,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True) or human-readable (False)
        include_correlation_id: Include correlation IDs in logs
        extra_fields: Static fields to include in all log entries
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if json_format:
        formatter = StructuredLogFormatter(
            include_correlation_id=include_correlation_id,
            extra_fields=extra_fields,
        )
    else:
        formatter = HumanReadableFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


T = TypeVar("T")


def log_with_context(
    logger: logging.Logger, level: int = logging.INFO, message: str = "", **context
) -> None:
    """
    Log a message with additional context.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context fields
    """
    logger.log(level, message, extra=context)


def log_execution_time(
    logger: logging.Logger,
    operation: str = "operation",
    level: int = logging.DEBUG,
):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance
        operation: Operation name for logging
        level: Log level for timing messages
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.log(
                    level,
                    f"{operation} completed",
                    extra={
                        "operation": operation,
                        "duration_ms": round(elapsed_ms, 2),
                        "status": "success",
                    },
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.log(
                    logging.ERROR,
                    f"{operation} failed: {e}",
                    extra={
                        "operation": operation,
                        "duration_ms": round(elapsed_ms, 2),
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.log(
                    level,
                    f"{operation} completed",
                    extra={
                        "operation": operation,
                        "duration_ms": round(elapsed_ms, 2),
                        "status": "success",
                    },
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.log(
                    logging.ERROR,
                    f"{operation} failed: {e}",
                    extra={
                        "operation": operation,
                        "duration_ms": round(elapsed_ms, 2),
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class RequestContextMiddleware:
    """
    ASGI middleware for request context and correlation ID tracking.

    Usage:
        from fastapi import FastAPI
        from app.core.shared.structured_logging import RequestContextMiddleware

        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate correlation ID
        headers = dict(scope.get("headers", []))
        correlation_id = (
            headers.get(b"x-correlation-id", b"").decode()
            or headers.get(b"x-request-id", b"").decode()
            or generate_correlation_id()
        )

        # Set correlation ID in context
        token = set_correlation_id(correlation_id)

        # Set request context
        request_context = {
            "method": scope.get("method", ""),
            "path": scope.get("path", ""),
            "query_string": scope.get("query_string", b"").decode(),
            "client": scope.get("client", ("", 0))[0] if scope.get("client") else None,
        }
        context_token = set_request_context(request_context)

        # Add correlation ID to response headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Reset context
            correlation_id_var.reset(token)
            request_context_var.reset(context_token)


# Convenience exports
__all__ = [
    "HumanReadableFormatter",
    "RequestContextMiddleware",
    "StructuredLogFormatter",
    "configure_logging",
    "generate_correlation_id",
    "get_correlation_id",
    "get_logger",
    "get_request_context",
    "log_execution_time",
    "log_with_context",
    "set_correlation_id",
    "set_request_context",
    "update_request_context",
]
