"""
Enhanced Structured Logging Module for Chimera Backend

Provides:
- JSON-formatted structured logs for production
- Request tracing with correlation IDs
- Performance metrics logging
- Error tracking with Sentry-compatible format
- Log aggregation support
"""

import json
import logging
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, ClassVar

from app.core.config import settings

# Context variables for request tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


class StructuredLogFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Outputs logs in a format compatible with log aggregation systems.
    """

    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_level:
            log_data["level"] = record.levelname
            log_data["level_num"] = record.levelno

        # Add context variables
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        user_id = user_id_var.get()
        if user_id:
            log_data["user_id"] = user_id

        session_id = session_id_var.get()
        if session_id:
            log_data["session_id"] = session_id

        # Add extra fields from record
        if hasattr(record, "extra_data") and record.extra_data:
            log_data["extra"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
                if record.exc_info[0]
                else None,
            }

        # Add environment info
        log_data["environment"] = settings.ENVIRONMENT
        log_data["service"] = "chimera-backend"
        log_data["version"] = settings.VERSION

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development.
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
        reset = self.RESET

        # Build prefix with context
        prefix_parts = []
        request_id = request_id_var.get()
        if request_id:
            prefix_parts.append(f"[{request_id[:8]}]")

        prefix = " ".join(prefix_parts)
        if prefix:
            prefix = f"{prefix} "

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        return (
            f"{timestamp} {color}{record.levelname:8}{reset} "
            f"{prefix}{record.name}: {record.getMessage()}"
        )


class StructuredLogger(logging.Logger):
    """
    Enhanced logger with structured logging support.
    """

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)

    def _log_with_extra(
        self,
        level: int,
        msg: str,
        args: tuple = (),
        exc_info: Any = None,
        extra: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Log with extra structured data."""
        if extra is None:
            extra = {}

        # Create a new LogRecord with extra_data attribute
        record = self.makeRecord(self.name, level, "(unknown file)", 0, msg, args, exc_info)
        record.extra_data = extra
        self.handle(record)

    def info_with_data(self, msg: str, **data):
        """Log info with structured data."""
        self._log_with_extra(logging.INFO, msg, extra=data)

    def warning_with_data(self, msg: str, **data):
        """Log warning with structured data."""
        self._log_with_extra(logging.WARNING, msg, extra=data)

    def error_with_data(self, msg: str, exc_info: bool = False, **data):
        """Log error with structured data."""
        self._log_with_extra(
            logging.ERROR, msg, exc_info=sys.exc_info() if exc_info else None, extra=data
        )

    def debug_with_data(self, msg: str, **data):
        """Log debug with structured data."""
        self._log_with_extra(logging.DEBUG, msg, extra=data)


def setup_structured_logging() -> StructuredLogger:
    """
    Configure structured logging for the application.

    Returns:
        StructuredLogger: Configured logger instance
    """
    # Set the custom logger class
    logging.setLoggerClass(StructuredLogger)

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("chimera")
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Use JSON formatter in production, human-readable in development
    if settings.ENVIRONMENT == "production":
        formatter = StructuredLogFormatter()
    else:
        formatter = HumanReadableFormatter()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set lower level for third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logger


# Request context management
def set_request_context(
    request_id: str | None = None, user_id: str | None = None, session_id: str | None = None
):
    """Set request context for logging."""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def clear_request_context():
    """Clear request context after request completes."""
    request_id_var.set("")
    user_id_var.set("")
    session_id_var.set("")


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


# Performance logging decorator
def log_performance(logger: logging.Logger | None = None, threshold_ms: float = 1000):
    """
    Decorator to log function performance.

    Args:
        logger: Logger instance to use
        threshold_ms: Log warning if execution exceeds this threshold
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            log = logger or logging.getLogger("chimera.performance")

            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                log_data = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "request_id": request_id_var.get(),
                }

                if elapsed_ms > threshold_ms:
                    log.warning(
                        f"Slow function execution: {func.__name__} took {elapsed_ms:.2f}ms",
                        extra={"extra_data": log_data},
                    )
                else:
                    log.debug(
                        f"Function {func.__name__} completed in {elapsed_ms:.2f}ms",
                        extra={"extra_data": log_data},
                    )

                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                log.error(
                    f"Function {func.__name__} failed after {elapsed_ms:.2f}ms: {e}",
                    exc_info=True,
                    extra={
                        "extra_data": {
                            "function": func.__name__,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "error": str(e),
                        }
                    },
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            log = logger or logging.getLogger("chimera.performance")

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if elapsed_ms > threshold_ms:
                    log.warning(f"Slow function: {func.__name__} took {elapsed_ms:.2f}ms")
                else:
                    log.debug(f"Function {func.__name__} completed in {elapsed_ms:.2f}ms")

                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                log.error(f"Function {func.__name__} failed after {elapsed_ms:.2f}ms: {e}")
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Error tracking (Sentry-compatible format)
class ErrorTracker:
    """
    Error tracking utility that formats errors in Sentry-compatible format.
    Can be extended to send to actual Sentry or other error tracking services.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("chimera.errors")
        self._error_count = 0
        self._errors: list[dict] = []

    def capture_exception(
        self,
        exception: Exception,
        context: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        user: dict[str, str] | None = None,
    ):
        """
        Capture an exception with context.

        Args:
            exception: The exception to capture
            context: Additional context data
            tags: Tags for categorization
            user: User information
        """
        self._error_count += 1

        error_data = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "platform": "python",
            "level": "error",
            "exception": {
                "type": type(exception).__name__,
                "value": str(exception),
                "module": type(exception).__module__,
                "stacktrace": traceback.format_exc(),
            },
            "contexts": {
                "runtime": {
                    "name": "python",
                    "version": sys.version,
                },
                "app": {
                    "app_name": settings.APP_NAME,
                    "app_version": settings.VERSION,
                    "environment": settings.ENVIRONMENT,
                },
            },
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "user_id": user_id_var.get(),
        }

        if context:
            error_data["extra"] = context

        if tags:
            error_data["tags"] = tags

        if user:
            error_data["user"] = user

        # Store error (in production, this would send to Sentry)
        self._errors.append(error_data)
        if len(self._errors) > 100:
            self._errors = self._errors[-100:]  # Keep last 100 errors

        # Log the error
        self.logger.error(
            f"Captured exception: {type(exception).__name__}: {exception}",
            exc_info=True,
            extra={"extra_data": error_data},
        )

        return error_data["event_id"]

    def capture_message(
        self,
        message: str,
        level: str = "info",
        context: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Capture a message event."""
        event_data = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": message,
            "request_id": request_id_var.get(),
        }

        if context:
            event_data["extra"] = context

        if tags:
            event_data["tags"] = tags

        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, message, extra={"extra_data": event_data})

        return event_data["event_id"]

    def get_error_count(self) -> int:
        """Get total error count."""
        return self._error_count

    def get_recent_errors(self, limit: int = 10) -> list[dict]:
        """Get recent errors."""
        return self._errors[-limit:]


# Global instances
structured_logger = setup_structured_logging()
error_tracker = ErrorTracker(structured_logger)

# Export for backward compatibility
logger = structured_logger
