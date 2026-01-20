"""Stream Metrics Service for Project Chimera.

This module provides comprehensive metrics collection and reporting for
streaming operations. It tracks:
- Stream performance metrics (latency, throughput)
- Error rates by provider/model
- Stream duration and completion rates
- Token generation statistics

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md Section 4.2
- Unified streaming: app/services/unified_streaming_service.py

"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class StreamEventType(str, Enum):
    """Types of stream events for metrics."""

    STREAM_STARTED = "stream_started"
    STREAM_CHUNK = "stream_chunk"
    STREAM_COMPLETED = "stream_completed"
    STREAM_ERROR = "stream_error"
    STREAM_CANCELLED = "stream_cancelled"


class MetricAggregation(str, Enum):
    """Aggregation methods for metrics."""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P99 = "p99"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StreamEvent:
    """A single stream event for metrics tracking."""

    event_type: StreamEventType
    stream_id: str
    provider: str
    model: str
    timestamp: datetime
    session_id: str | None = None
    user_id: str | None = None
    chunk_index: int | None = None
    text_length: int | None = None
    token_count: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamSummary:
    """Summary of a completed stream."""

    stream_id: str
    provider: str
    model: str
    session_id: str | None
    started_at: datetime
    ended_at: datetime
    total_chunks: int
    total_tokens: int
    total_characters: int
    first_chunk_latency_ms: float
    total_duration_ms: float
    tokens_per_second: float
    characters_per_second: float
    finish_reason: str | None
    error: str | None
    was_cancelled: bool


@dataclass
class ProviderMetrics:
    """Aggregated metrics for a provider."""

    provider: str
    total_streams: int = 0
    successful_streams: int = 0
    failed_streams: int = 0
    cancelled_streams: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    avg_first_chunk_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    error_rate: float = 0.0
    models_used: dict[str, int] = field(default_factory=dict)
    errors_by_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "total_streams": self.total_streams,
            "successful_streams": self.successful_streams,
            "failed_streams": self.failed_streams,
            "cancelled_streams": self.cancelled_streams,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "avg_first_chunk_latency_ms": self.avg_first_chunk_latency_ms,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "error_rate": self.error_rate,
            "models_used": self.models_used,
            "errors_by_type": self.errors_by_type,
        }


@dataclass
class TimeWindowMetrics:
    """Metrics for a specific time window."""

    window_start: datetime
    window_end: datetime
    total_streams: int = 0
    active_streams: int = 0
    tokens_generated: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    by_provider: dict[str, ProviderMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "total_streams": self.total_streams,
            "active_streams": self.active_streams,
            "tokens_generated": self.tokens_generated,
            "avg_latency_ms": self.avg_latency_ms,
            "error_count": self.error_count,
            "by_provider": {p: m.to_dict() for p, m in self.by_provider.items()},
        }


# =============================================================================
# Stream Metrics Service
# =============================================================================


class StreamMetricsService:
    """Service for collecting and reporting stream metrics.

    Provides:
    - Real-time metrics collection during streaming
    - Historical metrics storage and aggregation
    - Per-provider and per-model analytics
    - Error rate tracking and alerting
    - Performance monitoring

    Usage:
        >>> service = StreamMetricsService()
        >>> service.record_event(StreamEvent(
        ...     event_type=StreamEventType.STREAM_STARTED,
        ...     stream_id="stream_123",
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     timestamp=datetime.utcnow()
        ... ))
    """

    _instance: Optional["StreamMetricsService"] = None

    def __new__(cls) -> "StreamMetricsService":
        """Singleton pattern for global access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the metrics service."""
        if self._initialized:
            return

        # Event storage
        self._events: list[StreamEvent] = []
        self._max_events = 10000

        # Stream summaries
        self._summaries: list[StreamSummary] = []
        self._max_summaries = 5000

        # Active streams tracking
        self._active_streams: dict[str, StreamEvent] = {}

        # Aggregated metrics by provider
        self._provider_metrics: dict[str, ProviderMetrics] = defaultdict(
            lambda: ProviderMetrics(provider=""),
        )

        # Time-windowed metrics (last 24 hours, hourly buckets)
        self._hourly_metrics: dict[str, TimeWindowMetrics] = {}

        # Alerting thresholds
        self._error_rate_threshold = 0.1  # 10%
        self._latency_threshold_ms = 5000  # 5 seconds

        # Alert callbacks
        self._alert_callbacks: list[callable] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

        self._initialized = True
        logger.info("StreamMetricsService initialized")

    # -------------------------------------------------------------------------
    # Event Recording
    # -------------------------------------------------------------------------

    async def record_event(self, event: StreamEvent) -> None:
        """Record a stream event.

        Args:
            event: The stream event to record

        """
        async with self._lock:
            # Store event
            self._events.append(event)

            # Trim if needed
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

            # Update provider metrics
            self._update_provider_metrics(event)

            # Update hourly metrics
            self._update_hourly_metrics(event)

            # Handle specific event types
            if event.event_type == StreamEventType.STREAM_STARTED:
                self._active_streams[event.stream_id] = event

            elif event.event_type in (
                StreamEventType.STREAM_COMPLETED,
                StreamEventType.STREAM_ERROR,
                StreamEventType.STREAM_CANCELLED,
            ):
                # Remove from active streams
                start_event = self._active_streams.pop(event.stream_id, None)

                # Create summary if we have start event
                if start_event:
                    summary = self._create_summary(start_event, event)
                    self._summaries.append(summary)

                    if len(self._summaries) > self._max_summaries:
                        keep = self._max_summaries
                        self._summaries = self._summaries[-keep:]

            # Check for alerts
            await self._check_alerts(event)

        logger.debug(
            f"Recorded stream event: {event.event_type.value} for stream {event.stream_id}",
        )

    def record_event_sync(self, event: StreamEvent) -> None:
        """Synchronous version of record_event for use in sync contexts."""
        # Store event
        self._events.append(event)

        # Trim if needed
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

        # Update provider metrics
        self._update_provider_metrics(event)

        # Handle specific event types
        if event.event_type == StreamEventType.STREAM_STARTED:
            self._active_streams[event.stream_id] = event

        elif event.event_type in (
            StreamEventType.STREAM_COMPLETED,
            StreamEventType.STREAM_ERROR,
            StreamEventType.STREAM_CANCELLED,
        ):
            start_event = self._active_streams.pop(event.stream_id, None)
            if start_event:
                summary = self._create_summary(start_event, event)
                self._summaries.append(summary)

    def _update_provider_metrics(self, event: StreamEvent) -> None:
        """Update aggregated provider metrics."""
        provider = event.provider
        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = ProviderMetrics(provider=provider)

        metrics = self._provider_metrics[provider]

        if event.event_type == StreamEventType.STREAM_STARTED:
            metrics.total_streams += 1
            model = event.model
            metrics.models_used[model] = metrics.models_used.get(model, 0) + 1

        elif event.event_type == StreamEventType.STREAM_COMPLETED:
            metrics.successful_streams += 1
            if event.token_count:
                metrics.total_tokens += event.token_count
            if event.latency_ms:
                metrics.total_duration_ms += event.latency_ms

        elif event.event_type == StreamEventType.STREAM_ERROR:
            metrics.failed_streams += 1
            if event.error:
                err_parts = event.error.split(":")
                error_type = err_parts[0] if err_parts else "unknown"
                metrics.errors_by_type[error_type] = metrics.errors_by_type.get(error_type, 0) + 1

        elif event.event_type == StreamEventType.STREAM_CANCELLED:
            metrics.cancelled_streams += 1

        # Update derived metrics
        total = metrics.total_streams
        if total > 0:
            metrics.error_rate = metrics.failed_streams / total

    def _update_hourly_metrics(self, event: StreamEvent) -> None:
        """Update hourly time-windowed metrics."""
        # Get hour bucket
        hour = event.timestamp.replace(minute=0, second=0, microsecond=0)
        hour_key = hour.isoformat()

        if hour_key not in self._hourly_metrics:
            self._hourly_metrics[hour_key] = TimeWindowMetrics(
                window_start=hour,
                window_end=hour + timedelta(hours=1),
            )

        metrics = self._hourly_metrics[hour_key]

        if event.event_type == StreamEventType.STREAM_STARTED:
            metrics.total_streams += 1
            metrics.active_streams += 1

        elif event.event_type == StreamEventType.STREAM_COMPLETED:
            metrics.active_streams = max(0, metrics.active_streams - 1)
            if event.token_count:
                metrics.tokens_generated += event.token_count

        elif event.event_type == StreamEventType.STREAM_ERROR:
            metrics.active_streams = max(0, metrics.active_streams - 1)
            metrics.error_count += 1

        elif event.event_type == StreamEventType.STREAM_CANCELLED:
            metrics.active_streams = max(0, metrics.active_streams - 1)

        # Cleanup old hourly metrics (keep last 48 hours)
        cutoff = datetime.utcnow() - timedelta(hours=48)
        self._hourly_metrics = {
            k: v for k, v in self._hourly_metrics.items() if v.window_start > cutoff
        }

    def _create_summary(
        self,
        start_event: StreamEvent,
        end_event: StreamEvent,
    ) -> StreamSummary:
        """Create a stream summary from start and end events."""
        time_delta = end_event.timestamp - start_event.timestamp
        duration_ms = time_delta.total_seconds() * 1000

        tokens = end_event.token_count or 0
        chars = end_event.text_length or 0
        duration_sec = duration_ms / 1000 if duration_ms > 0 else 1

        return StreamSummary(
            stream_id=start_event.stream_id,
            provider=start_event.provider,
            model=start_event.model,
            session_id=start_event.session_id,
            started_at=start_event.timestamp,
            ended_at=end_event.timestamp,
            total_chunks=end_event.chunk_index or 0,
            total_tokens=tokens,
            total_characters=chars,
            first_chunk_latency_ms=end_event.metadata.get("first_chunk_latency_ms", 0),
            total_duration_ms=duration_ms,
            tokens_per_second=tokens / duration_sec if tokens else 0,
            characters_per_second=chars / duration_sec if chars else 0,
            finish_reason=end_event.finish_reason,
            error=end_event.error,
            was_cancelled=(end_event.event_type == StreamEventType.STREAM_CANCELLED),
        )

    # -------------------------------------------------------------------------
    # Alerting
    # -------------------------------------------------------------------------

    async def _check_alerts(self, event: StreamEvent) -> None:
        """Check if event triggers any alerts."""
        alerts = []

        # Check error rate
        provider = event.provider
        if provider in self._provider_metrics:
            metrics = self._provider_metrics[provider]
            if (
                metrics.error_rate > self._error_rate_threshold and metrics.total_streams >= 10
            ):  # Min sample size
                alerts.append(
                    {
                        "type": "high_error_rate",
                        "provider": provider,
                        "error_rate": metrics.error_rate,
                        "threshold": self._error_rate_threshold,
                    },
                )

        # Check latency
        if event.latency_ms and event.latency_ms > self._latency_threshold_ms:
            alerts.append(
                {
                    "type": "high_latency",
                    "provider": provider,
                    "latency_ms": event.latency_ms,
                    "threshold": self._latency_threshold_ms,
                    "stream_id": event.stream_id,
                },
            )

        # Notify callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.exception(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback for metric alerts."""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: callable) -> bool:
        """Remove an alert callback."""
        try:
            self._alert_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # -------------------------------------------------------------------------
    # Metrics Retrieval
    # -------------------------------------------------------------------------

    def get_provider_metrics(
        self,
        provider: str | None = None,
    ) -> dict[str, ProviderMetrics]:
        """Get aggregated metrics by provider.

        Args:
            provider: Optional specific provider to get metrics for

        Returns:
            Dictionary of provider -> ProviderMetrics

        """
        if provider:
            if provider in self._provider_metrics:
                return {provider: self._provider_metrics[provider]}
            return {}
        return dict(self._provider_metrics)

    def get_active_stream_count(self) -> int:
        """Get the number of currently active streams."""
        return len(self._active_streams)

    def get_active_streams_by_provider(self) -> dict[str, int]:
        """Get count of active streams by provider."""
        counts: dict[str, int] = defaultdict(int)
        for event in self._active_streams.values():
            counts[event.provider] += 1
        return dict(counts)

    def get_recent_summaries(
        self,
        limit: int = 100,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[StreamSummary]:
        """Get recent stream summaries.

        Args:
            limit: Maximum number of summaries to return
            provider: Optional filter by provider
            model: Optional filter by model

        Returns:
            List of StreamSummary objects

        """
        summaries = self._summaries

        if provider:
            summaries = [s for s in summaries if s.provider == provider]
        if model:
            summaries = [s for s in summaries if s.model == model]

        return summaries[-limit:]

    def get_hourly_metrics(
        self,
        hours: int = 24,
    ) -> list[TimeWindowMetrics]:
        """Get hourly metrics for the last N hours.

        Args:
            hours: Number of hours to retrieve

        Returns:
            List of TimeWindowMetrics objects

        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        metrics = [m for m in self._hourly_metrics.values() if m.window_start > cutoff]
        return sorted(metrics, key=lambda m: m.window_start)

    def get_overall_summary(self) -> dict[str, Any]:
        """Get overall metrics summary.

        Returns:
            Dictionary with aggregate metrics

        """
        total_streams = sum(m.total_streams for m in self._provider_metrics.values())
        successful = sum(m.successful_streams for m in self._provider_metrics.values())
        failed = sum(m.failed_streams for m in self._provider_metrics.values())
        cancelled = sum(m.cancelled_streams for m in self._provider_metrics.values())
        total_tokens = sum(m.total_tokens for m in self._provider_metrics.values())

        return {
            "total_streams": total_streams,
            "successful_streams": successful,
            "failed_streams": failed,
            "cancelled_streams": cancelled,
            "success_rate": successful / total_streams if total_streams else 0,
            "error_rate": failed / total_streams if total_streams else 0,
            "total_tokens_generated": total_tokens,
            "active_streams": len(self._active_streams),
            "providers_count": len(self._provider_metrics),
            "by_provider": {p: m.to_dict() for p, m in self._provider_metrics.items()},
        }

    def get_recent_errors(
        self,
        limit: int = 50,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent error events.

        Args:
            limit: Maximum errors to return
            provider: Optional filter by provider

        Returns:
            List of error event dictionaries

        """
        errors = [e for e in self._events if e.event_type == StreamEventType.STREAM_ERROR]

        if provider:
            errors = [e for e in errors if e.provider == provider]

        return [
            {
                "stream_id": e.stream_id,
                "provider": e.provider,
                "model": e.model,
                "error": e.error,
                "timestamp": e.timestamp.isoformat(),
                "session_id": e.session_id,
            }
            for e in errors[-limit:]
        ]

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def reset_metrics(self) -> None:
        """Reset all metrics. Use with caution."""
        self._events.clear()
        self._summaries.clear()
        self._active_streams.clear()
        self._provider_metrics.clear()
        self._hourly_metrics.clear()
        logger.info("Stream metrics reset")


# =============================================================================
# Factory Functions
# =============================================================================


_metrics_service: StreamMetricsService | None = None


def get_stream_metrics_service() -> StreamMetricsService:
    """Get the singleton StreamMetricsService instance.

    Returns:
        The global StreamMetricsService instance

    """
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = StreamMetricsService()
    return _metrics_service


async def get_stream_metrics_service_async() -> StreamMetricsService:
    """Async factory for StreamMetricsService.

    Use this as a FastAPI dependency.

    Returns:
        The global StreamMetricsService instance

    """
    return get_stream_metrics_service()


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "MetricAggregation",
    "ProviderMetrics",
    # Data classes
    "StreamEvent",
    # Enums
    "StreamEventType",
    # Main service
    "StreamMetricsService",
    "StreamSummary",
    "TimeWindowMetrics",
    # Factory functions
    "get_stream_metrics_service",
    "get_stream_metrics_service_async",
]
