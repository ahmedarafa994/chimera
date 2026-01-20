"""Unified Streaming Service for Project Chimera.

This service provides unified streaming generation that respects
provider/model selection from GlobalModelSelectionState and
ProviderResolutionService.

Features:
- Uses resolved provider/model from ProviderResolutionService
- Routes to appropriate provider plugin for streaming
- Maintains selection consistency during stream
- Handles provider-specific streaming formats
- Supports both text generation and chat completion streaming
- Comprehensive error handling and metrics collection

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md
- Provider plugins: app/services/provider_plugins/__init__.py
- Global model selection: app/services/global_model_selection_state.py
- Provider resolution: app/services/provider_resolution_service.py

"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.services.global_model_selection_state import GlobalModelSelectionState
from app.services.provider_plugins import GenerationRequest, get_plugin
from app.services.provider_plugins import StreamChunk as PluginStreamChunk
from app.services.provider_resolution_service import get_provider_resolution_service

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Streaming
# =============================================================================


@dataclass
class StreamingRequest:
    """Request for streaming generation.

    Contains all parameters needed for streaming generation including
    provider/model selection options.
    """

    prompt: str
    system_instruction: str | None = None
    messages: list[dict[str, str]] | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    session_id: str | None = None
    user_id: str | None = None
    explicit_provider: str | None = None
    explicit_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingContext:
    """Context for a streaming session.

    Captures the resolved provider/model selection and maintains
    consistency during the entire stream lifetime.
    """

    stream_id: str
    session_id: str | None
    user_id: str | None
    provider: str
    model: str
    resolution_source: str
    started_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "stream_id": self.stream_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "provider": self.provider,
            "model": self.model,
            "resolution_source": self.resolution_source,
            "started_at": self.started_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class UnifiedStreamChunk:
    """Unified stream chunk format for consistent response structure.

    Normalizes provider-specific chunk formats into a unified structure
    with tracing information.
    """

    text: str
    chunk_index: int
    stream_id: str
    provider: str
    model: str
    is_final: bool = False
    finish_reason: str | None = None
    token_count: int | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "stream_id": self.stream_id,
            "provider": self.provider,
            "model": self.model,
            "is_final": self.is_final,
            "finish_reason": self.finish_reason,
            "token_count": self.token_count,
            "timestamp": self.timestamp.isoformat(),
            "usage": self.usage,
            "metadata": self.metadata,
        }


@dataclass
class StreamMetrics:
    """Metrics collected during streaming.

    Used for monitoring and debugging streaming performance.
    """

    stream_id: str
    provider: str
    model: str
    session_id: str | None
    started_at: datetime
    ended_at: datetime | None = None
    total_chunks: int = 0
    total_tokens: int = 0
    total_characters: int = 0
    first_chunk_latency_ms: float | None = None
    total_duration_ms: float | None = None
    finish_reason: str | None = None
    error: str | None = None

    def calculate_throughput(self) -> dict[str, float]:
        """Calculate streaming throughput metrics."""
        if not self.total_duration_ms or self.total_duration_ms == 0:
            return {"tokens_per_second": 0, "characters_per_second": 0}

        duration_seconds = self.total_duration_ms / 1000
        tps = self.total_tokens / duration_seconds if self.total_tokens else 0
        cps = self.total_characters / duration_seconds
        return {
            "tokens_per_second": tps,
            "characters_per_second": cps,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "stream_id": self.stream_id,
            "provider": self.provider,
            "model": self.model,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "total_characters": self.total_characters,
            "first_chunk_latency_ms": self.first_chunk_latency_ms,
            "total_duration_ms": self.total_duration_ms,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }
        result.update(self.calculate_throughput())
        return result


# =============================================================================
# Unified Streaming Service
# =============================================================================


class UnifiedStreamingService:
    """Unified streaming service that respects provider/model selection.

    Features:
    - Uses resolved provider/model from ProviderResolutionService
    - Routes to appropriate provider plugin for streaming
    - Maintains selection consistency during stream
    - Handles provider-specific streaming formats
    - Collects metrics for monitoring

    Usage:
        >>> service = UnifiedStreamingService()
        >>> async for chunk in service.stream_generate(
        ...     prompt="Explain quantum computing",
        ...     session_id="sess_123"
        ... ):
        ...     print(chunk.text, end="")
    """

    _instance: Optional["UnifiedStreamingService"] = None

    def __new__(cls) -> "UnifiedStreamingService":
        """Singleton pattern for global access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the streaming service."""
        if self._initialized:
            return

        self._resolution_service = get_provider_resolution_service()
        self._global_state = GlobalModelSelectionState()

        # Active streams tracking
        self._active_streams: dict[str, StreamingContext] = {}
        self._stream_metrics: list[StreamMetrics] = []
        self._max_metrics_history = 1000

        # Stream listeners for external monitoring
        self._stream_listeners: list[callable] = []

        self._initialized = True
        logger.info("UnifiedStreamingService initialized")

    # -------------------------------------------------------------------------
    # Stream Generation Methods
    # -------------------------------------------------------------------------

    async def stream_generate(
        self,
        prompt: str,
        session_id: str | None = None,
        user_id: str | None = None,
        explicit_provider: str | None = None,
        explicit_model: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[UnifiedStreamChunk]:
        """Stream text generation using current provider/model selection.

        This method resolves the provider/model from the selection hierarchy,
        creates a streaming context, and yields unified stream chunks.

        Args:
            prompt: The text prompt for generation
            session_id: Session identifier for session-scoped selection
            user_id: User identifier for user-scoped selection
            explicit_provider: Optional explicit provider override
            explicit_model: Optional explicit model override
            system_instruction: Optional system instruction
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            UnifiedStreamChunk: Normalized stream chunks

        Example:
            >>> async for chunk in service.stream_generate(
            ...     prompt="Write a poem about coding",
            ...     session_id="sess_123",
            ...     temperature=0.8
            ... ):
            ...     print(chunk.text, end="", flush=True)

        """
        # Create streaming request
        request = StreamingRequest(
            prompt=prompt,
            session_id=session_id,
            user_id=user_id,
            explicit_provider=explicit_provider,
            explicit_model=explicit_model,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=kwargs,
        )

        # Resolve provider/model and create context
        context = await self._create_streaming_context(request)
        metrics = StreamMetrics(
            stream_id=context.stream_id,
            provider=context.provider,
            model=context.model,
            session_id=session_id,
            started_at=context.started_at,
        )

        try:
            # Register active stream
            self._active_streams[context.stream_id] = context

            # Get provider plugin
            plugin = get_plugin(context.provider)
            if not plugin:
                msg = f"Provider plugin not found: {context.provider}"
                raise ValueError(msg)

            if not plugin.supports_streaming():
                msg = f"Provider {context.provider} does not support streaming"
                raise ValueError(msg)

            # Create generation request
            gen_request = GenerationRequest(
                prompt=prompt,
                model=context.model,
                system_instruction=system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get("top_p"),
                top_k=kwargs.get("top_k"),
                stop_sequences=kwargs.get("stop_sequences"),
                stream=True,
                metadata={
                    "stream_id": context.stream_id,
                    "session_id": session_id,
                    **kwargs,
                },
            )

            # Stream from provider
            chunk_index = 0
            start_time = time.time()
            first_chunk_time: float | None = None

            async for plugin_chunk in plugin.generate_stream(gen_request):
                # Record first chunk latency
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = (first_chunk_time - start_time) * 1000
                    metrics.first_chunk_latency_ms = latency

                # Normalize chunk to unified format
                unified_chunk = self._normalize_chunk(
                    plugin_chunk,
                    chunk_index,
                    context,
                )

                # Update metrics
                metrics.total_chunks += 1
                metrics.total_characters += len(unified_chunk.text)
                if unified_chunk.token_count:
                    metrics.total_tokens += unified_chunk.token_count

                if unified_chunk.is_final:
                    metrics.finish_reason = unified_chunk.finish_reason
                    if unified_chunk.usage:
                        metrics.total_tokens = unified_chunk.usage.get(
                            "total_tokens",
                            metrics.total_tokens,
                        )

                # Notify listeners
                await self._notify_listeners("chunk", context, unified_chunk)

                yield unified_chunk
                chunk_index += 1

            # Finalize metrics
            metrics.ended_at = datetime.utcnow()
            metrics.total_duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Stream completed: stream_id={context.stream_id}, "
                f"provider={context.provider}, model={context.model}, "
                f"chunks={metrics.total_chunks}, "
                f"duration_ms={metrics.total_duration_ms:.1f}",
            )

        except asyncio.CancelledError:
            # Handle client disconnect
            metrics.error = "cancelled"
            metrics.ended_at = datetime.utcnow()
            logger.warning(f"Stream cancelled: stream_id={context.stream_id}")
            await self._notify_listeners("cancelled", context, None)
            raise

        except Exception as e:
            # Handle errors
            metrics.error = str(e)
            metrics.ended_at = datetime.utcnow()
            logger.exception(f"Stream error: stream_id={context.stream_id}, error={e}")
            await self._notify_listeners("error", context, e)
            raise

        finally:
            # Cleanup
            self._active_streams.pop(context.stream_id, None)
            self._record_metrics(metrics)
            await self._notify_listeners("completed", context, metrics)

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        session_id: str | None = None,
        user_id: str | None = None,
        explicit_provider: str | None = None,
        explicit_model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[UnifiedStreamChunk]:
        """Stream chat completion using current provider/model selection.

        Similar to stream_generate but uses chat message format instead
        of a single prompt.

        Args:
            messages: List of chat messages with role and content
            session_id: Session identifier for session-scoped selection
            user_id: User identifier for user-scoped selection
            explicit_provider: Optional explicit provider override
            explicit_model: Optional explicit model override
            **kwargs: Additional generation parameters

        Yields:
            UnifiedStreamChunk: Normalized stream chunks

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful"},
            ...     {"role": "user", "content": "What is Python?"}
            ... ]
            >>> async for chunk in service.stream_chat(
            ...     messages=messages,
            ...     session_id="sess_123"
            ... ):
            ...     print(chunk.text, end="", flush=True)

        """
        # Convert messages to prompt (simplified approach)
        # More sophisticated implementations could use provider-specific
        # chat formatting
        prompt_parts = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts)
        if prompt_parts:
            prompt += "\nAssistant:"

        # Use stream_generate with constructed prompt
        async for chunk in self.stream_generate(
            prompt=prompt,
            session_id=session_id,
            user_id=user_id,
            explicit_provider=explicit_provider,
            explicit_model=explicit_model,
            system_instruction=system_instruction,
            **kwargs,
        ):
            yield chunk

    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------

    async def _create_streaming_context(
        self,
        request: StreamingRequest,
    ) -> StreamingContext:
        """Create a streaming context with resolved provider/model.

        Resolves the provider/model from the selection hierarchy and
        creates an immutable context for the stream duration.
        """
        # Resolve provider/model
        resolution = await self._resolution_service.resolve_with_metadata(
            explicit_provider=request.explicit_provider,
            explicit_model=request.explicit_model,
            session_id=request.session_id,
            user_id=request.user_id,
        )

        stream_id = f"stream_{uuid.uuid4().hex[:12]}"

        context = StreamingContext(
            stream_id=stream_id,
            session_id=request.session_id,
            user_id=request.user_id,
            provider=resolution.provider,
            model=resolution.model_id,
            resolution_source=resolution.resolution_source,
            started_at=datetime.utcnow(),
            metadata={
                "resolution_priority": resolution.resolution_priority,
                "request_metadata": request.metadata,
            },
        )

        logger.debug(
            f"Created streaming context: stream_id={stream_id}, "
            f"provider={resolution.provider}, model={resolution.model_id}, "
            f"source={resolution.resolution_source}",
        )

        return context

    # -------------------------------------------------------------------------
    # Chunk Normalization
    # -------------------------------------------------------------------------

    def _normalize_chunk(
        self,
        plugin_chunk: PluginStreamChunk,
        chunk_index: int,
        context: StreamingContext,
    ) -> UnifiedStreamChunk:
        """Normalize a provider-specific chunk to unified format.

        Handles differences between provider chunk formats and adds
        tracing information.
        """
        return UnifiedStreamChunk(
            text=plugin_chunk.text,
            chunk_index=chunk_index,
            stream_id=context.stream_id,
            provider=context.provider,
            model=context.model,
            is_final=plugin_chunk.is_final,
            finish_reason=plugin_chunk.finish_reason,
            token_count=None,  # May be added by provider
            usage=plugin_chunk.usage,
            metadata={
                **plugin_chunk.metadata,
                "session_id": context.session_id,
                "user_id": context.user_id,
            },
        )

    # -------------------------------------------------------------------------
    # Metrics and Monitoring
    # -------------------------------------------------------------------------

    def _record_metrics(self, metrics: StreamMetrics) -> None:
        """Record stream metrics for monitoring."""
        self._stream_metrics.append(metrics)

        # Trim history if needed
        if len(self._stream_metrics) > self._max_metrics_history:
            keep = self._max_metrics_history
            self._stream_metrics = self._stream_metrics[-keep:]

        # Log metrics
        logger.info(f"Stream metrics: {metrics.to_dict()}")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of stream metrics.

        Returns aggregate statistics about recent streams.
        """
        if not self._stream_metrics:
            return {
                "total_streams": 0,
                "by_provider": {},
                "error_rate": 0,
                "avg_duration_ms": 0,
                "avg_first_chunk_latency_ms": 0,
            }

        total = len(self._stream_metrics)
        errors = sum(1 for m in self._stream_metrics if m.error)

        by_provider: dict[str, int] = {}
        total_duration = 0
        total_first_latency = 0
        duration_count = 0
        latency_count = 0

        for m in self._stream_metrics:
            by_provider[m.provider] = by_provider.get(m.provider, 0) + 1
            if m.total_duration_ms:
                total_duration += m.total_duration_ms
                duration_count += 1
            if m.first_chunk_latency_ms:
                total_first_latency += m.first_chunk_latency_ms
                latency_count += 1

        avg_dur = total_duration / duration_count if duration_count > 0 else 0
        avg_lat = total_first_latency / latency_count if latency_count > 0 else 0

        return {
            "total_streams": total,
            "active_streams": len(self._active_streams),
            "by_provider": by_provider,
            "error_rate": errors / total if total > 0 else 0,
            "avg_duration_ms": avg_dur,
            "avg_first_chunk_latency_ms": avg_lat,
        }

    def get_recent_metrics(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent stream metrics."""
        return [m.to_dict() for m in self._stream_metrics[-limit:]]

    # -------------------------------------------------------------------------
    # Stream Management
    # -------------------------------------------------------------------------

    def get_active_streams(self) -> list[dict[str, Any]]:
        """Get information about currently active streams."""
        return [ctx.to_dict() for ctx in self._active_streams.values()]

    async def cancel_stream(self, stream_id: str) -> bool:
        """Request cancellation of an active stream.

        Note: This only removes the stream from tracking. The actual
        cancellation depends on the caller handling CancelledError.
        """
        if stream_id in self._active_streams:
            context = self._active_streams.pop(stream_id)
            logger.info(f"Stream cancelled by request: {stream_id}")
            await self._notify_listeners("cancelled", context, None)
            return True
        return False

    # -------------------------------------------------------------------------
    # Event Listeners
    # -------------------------------------------------------------------------

    def add_stream_listener(self, callback: callable) -> None:
        """Add a listener for stream events.

        Args:
            callback: Async callable(event_type, context, data)

        """
        self._stream_listeners.append(callback)

    def remove_stream_listener(self, callback: callable) -> bool:
        """Remove a stream listener."""
        try:
            self._stream_listeners.remove(callback)
            return True
        except ValueError:
            return False

    async def _notify_listeners(
        self,
        event_type: str,
        context: StreamingContext,
        data: Any,
    ) -> None:
        """Notify all listeners of a stream event."""
        for listener in self._stream_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event_type, context, data)
                else:
                    listener(event_type, context, data)
            except Exception as e:
                logger.exception(f"Error in stream listener: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


_unified_streaming_service: UnifiedStreamingService | None = None


def get_unified_streaming_service() -> UnifiedStreamingService:
    """Get the singleton UnifiedStreamingService instance.

    Returns:
        The global UnifiedStreamingService instance

    """
    global _unified_streaming_service
    if _unified_streaming_service is None:
        _unified_streaming_service = UnifiedStreamingService()
    return _unified_streaming_service


async def get_unified_streaming_service_async() -> UnifiedStreamingService:
    """Async factory for UnifiedStreamingService.

    Use this as a FastAPI dependency:
        @router.post("/stream/generate")
        async def stream_generate(
            streaming: UnifiedStreamingService = Depends(
                get_unified_streaming_service_async
            )
        ):
            ...

    Returns:
        The global UnifiedStreamingService instance

    """
    return get_unified_streaming_service()


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "StreamMetrics",
    "StreamingContext",
    # Data classes
    "StreamingRequest",
    "UnifiedStreamChunk",
    # Main service
    "UnifiedStreamingService",
    # Factory functions
    "get_unified_streaming_service",
    "get_unified_streaming_service_async",
]
