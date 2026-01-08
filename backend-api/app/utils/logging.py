"""
Logging utilities for the backend API.
Provides consistent logging configuration and utilities.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, *args, use_colors: bool = True, **kwargs):
        """Initialize the colored formatter."""
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        log_message = super().format(record)

        if self.use_colors and record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"

        return log_message


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
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
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
            ]:
                log_obj[key] = value

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_obj)


class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in log messages."""

    SENSITIVE_PATTERNS: ClassVar[list[str]] = [
        "password",
        "api_key",
        "token",
        "secret",
        "credential",
        "auth",
        "private",
        "key",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and mask sensitive data in log records."""
        message = record.getMessage().lower()

        # Check for sensitive patterns
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                # Mask the value after the sensitive key
                record.msg = self._mask_sensitive_data(record.msg)
                break

        return True

    def _mask_sensitive_data(self, message: str) -> str:
        """Mask sensitive data in the message."""
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message.lower():
                # Simple masking - can be enhanced with regex for better matching
                parts = message.split("=")
                if len(parts) > 1:
                    masked_parts = []
                    for i, part in enumerate(parts):
                        if i > 0 and any(
                            p in parts[i - 1].lower() for p in self.SENSITIVE_PATTERNS
                        ):
                            # Mask the value part
                            value = part.split()[0] if " " in part else part
                            masked = "***MASKED***" + part[len(value) :]
                            masked_parts.append(masked)
                        else:
                            masked_parts.append(part)
                    message = "=".join(masked_parts)

        return message


class RequestLogger:
    """Logger for HTTP requests with timing and metadata."""

    def __init__(self, logger: logging.Logger):
        """Initialize the request logger."""
        self.logger = logger

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_id: str | None = None,
        user_id: str | None = None,
        error: str | None = None,
        **extra,
    ):
        """Log an HTTP request with metadata."""
        log_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "user_id": user_id,
        }

        if extra:
            log_data.update(extra)

        if error:
            log_data["error"] = error
            self.logger.error(f"Request failed: {method} {path}", extra=log_data)
        elif status_code >= 500:
            self.logger.error(f"Server error: {method} {path}", extra=log_data)
        elif status_code >= 400:
            self.logger.warning(f"Client error: {method} {path}", extra=log_data)
        else:
            self.logger.info(f"Request completed: {method} {path}", extra=log_data)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_format: str = "console",
    use_colors: bool = True,
    enable_sensitive_filter: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        log_format: Format type ('console', 'json')
        use_colors: Whether to use colors in console output
        enable_sensitive_filter: Whether to enable sensitive data filtering
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Create formatters
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = ColoredFormatter(console_format, use_colors=use_colors)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )

        # Always use JSON format for file logging
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Add sensitive data filter
    if enable_sensitive_filter:
        sensitive_filter = SensitiveDataFilter()
        for handler in root_logger.handlers:
            handler.addFilter(sensitive_filter)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_execution_time(logger: logging.Logger | None = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use (optional)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms

                logger.debug(
                    f"Function '{func.__name__}' executed in {execution_time:.2f}ms",
                    extra={
                        "function": func.__name__,
                        "module_name": func.__module__,
                        "execution_time_ms": execution_time,
                    },
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                logger.error(
                    f"Function '{func.__name__}' failed after {execution_time:.2f}ms: {e!s}",
                    extra={
                        "function": func.__name__,
                        "module_name": func.__module__,
                        "execution_time_ms": execution_time,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_api_call(
    logger: logging.Logger | None = None,
    log_request: bool = True,
    log_response: bool = True,
    mask_sensitive: bool = True,
):
    """
    Decorator to log API calls with request and response data.

    Args:
        logger: Logger instance to use
        log_request: Whether to log request data
        log_response: Whether to log response data
        mask_sensitive: Whether to mask sensitive data
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            start_time = time.time()
            request_data = {}

            # Extract request data if available
            if log_request and "request" in kwargs:
                request_obj = kwargs["request"]
                request_data = {
                    "method": getattr(request_obj, "method", "UNKNOWN"),
                    "path": getattr(request_obj, "path", "UNKNOWN"),
                    "headers": dict(getattr(request_obj, "headers", {})),
                }

                if mask_sensitive:
                    # Mask sensitive headers
                    for header in ["Authorization", "X-API-Key"]:
                        if header in request_data["headers"]:
                            request_data["headers"][header] = "***MASKED***"

            logger.info(
                f"API call started: {func.__name__}",
                extra={"function": func.__name__, "request": request_data},
            )

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                response_data = {}
                if log_response and result is not None:
                    if hasattr(result, "status_code"):
                        response_data["status_code"] = result.status_code
                    if hasattr(result, "__dict__"):
                        response_data["type"] = type(result).__name__

                logger.info(
                    f"API call completed: {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "execution_time_ms": execution_time,
                        "response": response_data,
                    },
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                logger.error(
                    f"API call failed: {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "execution_time_ms": execution_time,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


class LogContext:
    """Context manager for adding context to log messages."""

    def __init__(self, logger: logging.Logger, **context):
        """Initialize the log context."""
        self.logger = logger
        self.context = context
        self.old_context = {}

    def __enter__(self):
        """Enter the context and add context data to logger."""
        for key, value in self.context.items():
            if hasattr(self.logger, key):
                self.old_context[key] = getattr(self.logger, key)
            setattr(self.logger, key, value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original logger state."""
        for key in self.context:
            if key in self.old_context:
                setattr(self.logger, key, self.old_context[key])
            else:
                delattr(self.logger, key)


def create_audit_logger(name: str = "audit") -> logging.Logger:
    """
    Create a specialized audit logger.

    Args:
        name: Logger name

    Returns:
        Configured audit logger
    """
    audit_logger = logging.getLogger(name)
    audit_logger.setLevel(logging.INFO)

    # Create audit log file handler
    audit_file = os.getenv("AUDIT_LOG_FILE", "logs/audit.log")
    Path(audit_file).parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        audit_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
    )

    handler.setFormatter(JSONFormatter())
    audit_logger.addHandler(handler)

    return audit_logger
