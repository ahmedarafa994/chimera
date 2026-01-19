"""
Provider Tracing Middleware for Request Lifecycle Tracking.

This middleware traces provider/model usage throughout the request lifecycle,
adding diagnostic headers and logging for observability. It provides:
- Tracing headers for debugging and monitoring
- Selection source tracking
- Validation status propagation
- Performance metrics

Usage:
    from app.middleware.provider_tracing_middleware import (
        ProviderTracingMiddleware
    )
    app.add_middleware(ProviderTracingMiddleware)
"""

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.selection_context import SelectionContext

logger = logging.getLogger(__name__)


class ProviderTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that traces provider/model usage through request lifecycle.

    Adds tracing headers to responses:
    - X-Provider-Selection: current provider
    - X-Model-Selection: current model
    - X-Selection-Source: where selection came from
    - X-Selection-Validated: validation status
    - X-Selection-Trace-ID: unique trace ID for this selection
    - X-Selection-Timestamp: when selection was resolved
    """

    # Default paths to exclude from tracing
    DEFAULT_EXCLUDE_PATHS: set[str] = {
        "/health",
        "/health/ping",
        "/health/ready",
        "/health/live",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
        "/api/v1/health",
        "/api/v1/auth",  # Exclude auth
    }

    def __init__(
        self,
        app,
        exclude_paths: set[str] | None = None,
        include_timing: bool = True,
        include_metadata: bool = False,
        trace_header_prefix: str = "X-Selection-",
        log_selections: bool = True,
    ):
        """
        Initialize the provider tracing middleware.

        Args:
            app: FastAPI application instance
            exclude_paths: Paths to exclude from tracing
            include_timing: Include timing information in headers
            include_metadata: Include additional metadata in headers
            trace_header_prefix: Prefix for tracing headers
            log_selections: Whether to log selection changes
        """
        super().__init__(app)
        self._exclude_paths = exclude_paths or self.DEFAULT_EXCLUDE_PATHS
        self._include_timing = include_timing
        self._include_metadata = include_metadata
        self._header_prefix = trace_header_prefix
        self._log_selections = log_selections

        logger.info(f"ProviderTracingMiddleware initialized " f"(prefix={trace_header_prefix})")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Trace provider/model selection through request.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response with tracing headers added
        """
        # Check if tracing should be skipped
        if self._should_skip(request.url.path):
            return await call_next(request)

        start_time = time.time()

        # Generate trace ID
        trace_id = str(uuid.uuid4())[:8]

        # Store trace ID in request state
        request.state.selection_trace_id = trace_id

        # Collect pre-request selection info
        pre_selection = self._get_selection_info()

        # Process request
        response = await call_next(request)

        # Collect post-request selection info
        post_selection = self._get_selection_info()

        # Calculate timing
        elapsed_ms = (time.time() - start_time) * 1000

        # Add tracing headers
        self._add_tracing_headers(
            response,
            trace_id,
            pre_selection,
            post_selection,
            elapsed_ms,
            request,
        )

        # Log trace info
        self._log_trace(
            trace_id,
            request,
            response,
            pre_selection,
            elapsed_ms,
        )

        return response

    def _should_skip(self, path: str) -> bool:
        """Check if tracing should be skipped for this path."""
        return path in self._exclude_paths

    def _get_selection_info(self) -> dict[str, Any]:
        """Get current selection information from context."""
        info: dict[str, Any] = {
            "provider": None,
            "model": None,
            "source": None,
            "validated": None,
            "scope": None,
        }

        # Try SelectionContext first
        if SelectionContext.is_set():
            selection = SelectionContext.get_selection()
            if selection:
                info["provider"] = selection.provider_id
                info["model"] = selection.model_id
                info["scope"] = selection.scope.value if selection.scope else None
                info["source"] = info["scope"]  # Same for now

        # Try to get validation result if available
        try:
            from app.core.request_context import ContextManager

            if ContextManager.has_context():
                ctx = ContextManager.get_context()
                info["provider"] = info["provider"] or ctx.provider
                info["model"] = info["model"] or ctx.model
        except Exception:
            pass

        return info

    def _add_tracing_headers(
        self,
        response: Response,
        trace_id: str,
        pre_selection: dict[str, Any],
        post_selection: dict[str, Any],
        elapsed_ms: float,
        request: Request,
    ) -> None:
        """Add tracing headers to response."""
        prefix = self._header_prefix

        # Core tracing headers
        response.headers[f"{prefix}Trace-ID"] = trace_id

        # Provider/model headers
        provider = post_selection.get("provider") or pre_selection.get("provider")
        model = post_selection.get("model") or pre_selection.get("model")

        if provider:
            response.headers[f"{prefix}Provider"] = provider
        if model:
            response.headers[f"{prefix}Model"] = model

        # Source header
        source = post_selection.get("source") or pre_selection.get("source")
        if source:
            response.headers[f"{prefix}Source"] = source

        # Validation status
        validated = self._get_validation_status(request)
        response.headers[f"{prefix}Validated"] = str(validated).lower()

        # Timing headers
        if self._include_timing:
            response.headers[f"{prefix}Time-Ms"] = f"{elapsed_ms:.2f}"

        # Check for selection changes
        if pre_selection.get("provider") != post_selection.get("provider") or pre_selection.get(
            "model"
        ) != post_selection.get("model"):
            response.headers[f"{prefix}Changed"] = "true"
        else:
            response.headers[f"{prefix}Changed"] = "false"

        # Additional metadata
        if self._include_metadata:
            scope = post_selection.get("scope")
            if scope:
                response.headers[f"{prefix}Scope"] = scope

    def _get_validation_status(self, request: Request) -> bool:
        """Get validation status from request state."""
        # Check for validation result from SelectionValidationMiddleware
        if hasattr(request.state, "validation_result"):
            return request.state.validation_result.is_valid

        # Default to True if no validation was performed
        return True

    def _log_trace(
        self,
        trace_id: str,
        request: Request,
        response: Response,
        selection: dict[str, Any],
        elapsed_ms: float,
    ) -> None:
        """Log trace information if logging is enabled."""
        if not self._log_selections:
            return

        provider = selection.get("provider", "unknown")
        model = selection.get("model", "unknown")
        source = selection.get("source", "unknown")

        logger.debug(
            f"[Trace:{trace_id}] {request.method} {request.url.path} "
            f"-> {response.status_code} "
            f"| provider={provider} model={model} source={source} "
            f"| {elapsed_ms:.2f}ms"
        )


class SelectionTracer:
    """
    Utility class for manual selection tracing.

    Use this to add tracing points within service code.
    """

    def __init__(self, operation_name: str):
        """
        Initialize a tracer for an operation.

        Args:
            operation_name: Name of the operation being traced
        """
        self._operation = operation_name
        self._trace_id = str(uuid.uuid4())[:8]
        self._start_time = time.time()
        self._events: list[dict[str, Any]] = []

    def record_selection(
        self,
        provider: str,
        model: str,
        source: str = "manual",
    ) -> None:
        """
        Record a selection event.

        Args:
            provider: Provider ID
            model: Model ID
            source: Source of the selection
        """
        self._events.append(
            {
                "type": "selection",
                "provider": provider,
                "model": model,
                "source": source,
                "timestamp_ms": self._elapsed_ms(),
            }
        )

    def record_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a custom event.

        Args:
            event_type: Type of event
            data: Additional event data
        """
        self._events.append(
            {
                "type": event_type,
                "data": data or {},
                "timestamp_ms": self._elapsed_ms(),
            }
        )

    def _elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self._start_time) * 1000

    def get_trace(self) -> dict[str, Any]:
        """Get the complete trace."""
        return {
            "trace_id": self._trace_id,
            "operation": self._operation,
            "duration_ms": self._elapsed_ms(),
            "events": self._events,
        }

    def log_trace(self, level: int = logging.DEBUG) -> None:
        """Log the trace at the specified level."""
        trace = self.get_trace()
        logger.log(
            level,
            f"[Trace:{self._trace_id}] {self._operation} "
            f"completed in {trace['duration_ms']:.2f}ms "
            f"with {len(self._events)} events",
        )


# =============================================================================
# Context Manager for Tracing Blocks
# =============================================================================


class TracingContext:
    """Context manager for tracing code blocks."""

    def __init__(self, operation_name: str):
        self._tracer = SelectionTracer(operation_name)

    def __enter__(self) -> SelectionTracer:
        return self._tracer

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self._tracer.record_event("error", {"type": str(exc_type), "message": str(exc_val)})
        self._tracer.log_trace()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ProviderTracingMiddleware",
    "SelectionTracer",
    "TracingContext",
]
