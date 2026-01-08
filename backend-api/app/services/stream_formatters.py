"""
Stream Response Formatters for Project Chimera.

This module provides formatters for streaming responses in different formats:
- Server-Sent Events (SSE)
- WebSocket messages
- Raw JSON Lines (JSONL)

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md Section 4.2
- Unified streaming: app/services/unified_streaming_service.py
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class StreamFormat(str, Enum):
    """Supported streaming response formats."""

    SSE = "sse"
    WEBSOCKET = "websocket"
    JSONL = "jsonl"
    RAW = "raw"


class SSEEventType(str, Enum):
    """Server-Sent Events event types."""

    MESSAGE = "message"
    CHUNK = "chunk"
    ERROR = "error"
    DONE = "done"
    PING = "ping"
    METADATA = "metadata"


# SSE Configuration
SSE_RETRY_MS = 3000
SSE_KEEP_ALIVE_COMMENT = ": keep-alive\n\n"


# =============================================================================
# Stream Chunk Data Class
# =============================================================================


@dataclass
class FormattedChunk:
    """
    A formatted stream chunk ready for delivery.

    Contains the formatted content and metadata about the chunk.
    """

    content: str
    format: StreamFormat
    is_final: bool = False
    byte_size: int = 0

    def __post_init__(self):
        """Calculate byte size after initialization."""
        if self.byte_size == 0:
            self.byte_size = len(self.content.encode("utf-8"))


# =============================================================================
# Stream Response Formatter
# =============================================================================


class StreamResponseFormatter:
    """
    Formats streaming responses for different output formats.

    Supports:
    - Server-Sent Events (SSE) for HTTP streaming
    - WebSocket messages for real-time connections
    - Raw JSON Lines for simple streaming

    Usage:
        >>> formatter = StreamResponseFormatter()
        >>> sse_chunk = formatter.format_sse(stream_chunk)
        >>> # Send sse_chunk as HTTP response
    """

    def __init__(
        self,
        include_metadata: bool = True,
        pretty_json: bool = False,
    ):
        """
        Initialize the formatter.

        Args:
            include_metadata: Include metadata in formatted output
            pretty_json: Use pretty-printed JSON (slower but readable)
        """
        self.include_metadata = include_metadata
        self.pretty_json = pretty_json

    # -------------------------------------------------------------------------
    # SSE Formatting
    # -------------------------------------------------------------------------

    def format_sse(
        self,
        text: str,
        chunk_index: int = 0,
        stream_id: str = "",
        provider: str = "",
        model: str = "",
        is_final: bool = False,
        finish_reason: Optional[str] = None,
        usage: Optional[dict[str, int]] = None,
        event_type: SSEEventType = SSEEventType.CHUNK,
        metadata: Optional[dict[str, Any]] = None,
    ) -> FormattedChunk:
        """
        Format a stream chunk as Server-Sent Events (SSE).

        SSE Format:
            event: <event_type>
            id: <chunk_index>
            data: <json_data>

        Args:
            text: The text content of the chunk
            chunk_index: Index of this chunk in the stream
            stream_id: Unique identifier for the stream
            provider: The provider name
            model: The model name
            is_final: Whether this is the final chunk
            finish_reason: Reason for finishing (if final)
            usage: Token usage information
            event_type: The SSE event type
            metadata: Additional metadata

        Returns:
            FormattedChunk with SSE-formatted content
        """
        # Build data payload
        data = {
            "text": text,
            "chunk_index": chunk_index,
            "is_final": is_final,
        }

        if self.include_metadata:
            data["stream_id"] = stream_id
            data["provider"] = provider
            data["model"] = model
            data["timestamp"] = datetime.utcnow().isoformat()

            if finish_reason:
                data["finish_reason"] = finish_reason
            if usage:
                data["usage"] = usage
            if metadata:
                data["metadata"] = metadata

        # Format as SSE
        json_data = self._to_json(data)

        lines = [
            f"event: {event_type.value}",
            f"id: {chunk_index}",
            f"data: {json_data}",
            "",  # Empty line to end event
            "",  # Additional empty line for clarity
        ]

        content = "\n".join(lines)

        return FormattedChunk(
            content=content,
            format=StreamFormat.SSE,
            is_final=is_final,
        )

    def format_sse_error(
        self,
        error: str,
        error_code: Optional[str] = None,
        stream_id: str = "",
    ) -> FormattedChunk:
        """
        Format an error as SSE.

        Args:
            error: Error message
            error_code: Optional error code
            stream_id: The stream identifier

        Returns:
            FormattedChunk with SSE-formatted error
        """
        data = {
            "error": error,
            "stream_id": stream_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if error_code:
            data["error_code"] = error_code

        json_data = self._to_json(data)

        lines = [
            f"event: {SSEEventType.ERROR.value}",
            f"data: {json_data}",
            "",
            "",
        ]

        return FormattedChunk(
            content="\n".join(lines),
            format=StreamFormat.SSE,
            is_final=True,
        )

    def format_sse_done(
        self,
        stream_id: str = "",
        total_chunks: int = 0,
        total_tokens: Optional[int] = None,
        finish_reason: str = "stop",
    ) -> FormattedChunk:
        """
        Format a completion event as SSE.

        Args:
            stream_id: The stream identifier
            total_chunks: Total number of chunks sent
            total_tokens: Total tokens generated
            finish_reason: Reason for completion

        Returns:
            FormattedChunk with SSE-formatted done event
        """
        data = {
            "stream_id": stream_id,
            "total_chunks": total_chunks,
            "finish_reason": finish_reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if total_tokens is not None:
            data["total_tokens"] = total_tokens

        json_data = self._to_json(data)

        lines = [
            f"event: {SSEEventType.DONE.value}",
            f"data: {json_data}",
            "",
            "",
        ]

        return FormattedChunk(
            content="\n".join(lines),
            format=StreamFormat.SSE,
            is_final=True,
        )

    def format_sse_ping(self) -> FormattedChunk:
        """
        Format a keep-alive ping as SSE.

        Returns:
            FormattedChunk with SSE comment for keep-alive
        """
        return FormattedChunk(
            content=SSE_KEEP_ALIVE_COMMENT,
            format=StreamFormat.SSE,
            is_final=False,
        )

    # -------------------------------------------------------------------------
    # WebSocket Formatting
    # -------------------------------------------------------------------------

    def format_ws(
        self,
        text: str,
        chunk_index: int = 0,
        stream_id: str = "",
        provider: str = "",
        model: str = "",
        is_final: bool = False,
        finish_reason: Optional[str] = None,
        usage: Optional[dict[str, int]] = None,
        message_type: str = "chunk",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Format a stream chunk for WebSocket delivery.

        Returns a dictionary suitable for JSON serialization and
        sending over WebSocket.

        Args:
            text: The text content of the chunk
            chunk_index: Index of this chunk in the stream
            stream_id: Unique identifier for the stream
            provider: The provider name
            model: The model name
            is_final: Whether this is the final chunk
            finish_reason: Reason for finishing (if final)
            usage: Token usage information
            message_type: The message type identifier
            metadata: Additional metadata

        Returns:
            Dictionary for WebSocket message
        """
        message = {
            "type": message_type,
            "payload": {
                "text": text,
                "chunk_index": chunk_index,
                "is_final": is_final,
            },
        }

        if self.include_metadata:
            message["payload"]["stream_id"] = stream_id
            message["payload"]["provider"] = provider
            message["payload"]["model"] = model
            message["payload"]["timestamp"] = datetime.utcnow().isoformat()

            if finish_reason:
                message["payload"]["finish_reason"] = finish_reason
            if usage:
                message["payload"]["usage"] = usage
            if metadata:
                message["payload"]["metadata"] = metadata

        return message

    def format_ws_error(
        self,
        error: str,
        error_code: Optional[str] = None,
        stream_id: str = "",
    ) -> dict[str, Any]:
        """
        Format an error for WebSocket delivery.

        Args:
            error: Error message
            error_code: Optional error code
            stream_id: The stream identifier

        Returns:
            Dictionary for WebSocket error message
        """
        message = {
            "type": "error",
            "payload": {
                "error": error,
                "stream_id": stream_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        if error_code:
            message["payload"]["error_code"] = error_code

        return message

    def format_ws_done(
        self,
        stream_id: str = "",
        total_chunks: int = 0,
        total_tokens: Optional[int] = None,
        finish_reason: str = "stop",
    ) -> dict[str, Any]:
        """
        Format a completion message for WebSocket delivery.

        Args:
            stream_id: The stream identifier
            total_chunks: Total number of chunks sent
            total_tokens: Total tokens generated
            finish_reason: Reason for completion

        Returns:
            Dictionary for WebSocket done message
        """
        message = {
            "type": "done",
            "payload": {
                "stream_id": stream_id,
                "total_chunks": total_chunks,
                "finish_reason": finish_reason,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        if total_tokens is not None:
            message["payload"]["total_tokens"] = total_tokens

        return message

    # -------------------------------------------------------------------------
    # JSON Lines Formatting
    # -------------------------------------------------------------------------

    def format_jsonl(
        self,
        text: str,
        chunk_index: int = 0,
        stream_id: str = "",
        provider: str = "",
        model: str = "",
        is_final: bool = False,
        finish_reason: Optional[str] = None,
        usage: Optional[dict[str, int]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> FormattedChunk:
        """
        Format a stream chunk as JSON Lines.

        Each line is a complete JSON object followed by a newline.

        Args:
            text: The text content of the chunk
            chunk_index: Index of this chunk in the stream
            stream_id: Unique identifier for the stream
            provider: The provider name
            model: The model name
            is_final: Whether this is the final chunk
            finish_reason: Reason for finishing (if final)
            usage: Token usage information
            metadata: Additional metadata

        Returns:
            FormattedChunk with JSON Lines formatted content
        """
        data = {
            "text": text,
            "chunk_index": chunk_index,
            "is_final": is_final,
        }

        if self.include_metadata:
            data["stream_id"] = stream_id
            data["provider"] = provider
            data["model"] = model
            data["timestamp"] = datetime.utcnow().isoformat()

            if finish_reason:
                data["finish_reason"] = finish_reason
            if usage:
                data["usage"] = usage
            if metadata:
                data["metadata"] = metadata

        content = self._to_json(data) + "\n"

        return FormattedChunk(
            content=content,
            format=StreamFormat.JSONL,
            is_final=is_final,
        )

    def format_jsonl_error(
        self,
        error: str,
        error_code: Optional[str] = None,
        stream_id: str = "",
    ) -> FormattedChunk:
        """
        Format an error as JSON Lines.

        Args:
            error: Error message
            error_code: Optional error code
            stream_id: The stream identifier

        Returns:
            FormattedChunk with JSON Lines formatted error
        """
        data = {
            "error": error,
            "stream_id": stream_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if error_code:
            data["error_code"] = error_code

        content = self._to_json(data) + "\n"

        return FormattedChunk(
            content=content,
            format=StreamFormat.JSONL,
            is_final=True,
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _to_json(self, data: dict[str, Any]) -> str:
        """Convert data to JSON string."""
        if self.pretty_json:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, separators=(",", ":"), default=str)

    def format_unified_chunk(
        self,
        chunk: Any,
        format_type: StreamFormat = StreamFormat.SSE,
    ) -> str | dict[str, Any]:
        """
        Format a UnifiedStreamChunk to the specified format.

        Args:
            chunk: A UnifiedStreamChunk instance
            format_type: The desired output format

        Returns:
            Formatted string (SSE/JSONL) or dict (WebSocket)
        """
        # Extract common fields
        text = getattr(chunk, "text", "")
        chunk_index = getattr(chunk, "chunk_index", 0)
        stream_id = getattr(chunk, "stream_id", "")
        provider = getattr(chunk, "provider", "")
        model = getattr(chunk, "model", "")
        is_final = getattr(chunk, "is_final", False)
        finish_reason = getattr(chunk, "finish_reason", None)
        usage = getattr(chunk, "usage", None)
        metadata = getattr(chunk, "metadata", None)

        if format_type == StreamFormat.SSE:
            result = self.format_sse(
                text=text,
                chunk_index=chunk_index,
                stream_id=stream_id,
                provider=provider,
                model=model,
                is_final=is_final,
                finish_reason=finish_reason,
                usage=usage,
                metadata=metadata,
            )
            return result.content

        elif format_type == StreamFormat.WEBSOCKET:
            return self.format_ws(
                text=text,
                chunk_index=chunk_index,
                stream_id=stream_id,
                provider=provider,
                model=model,
                is_final=is_final,
                finish_reason=finish_reason,
                usage=usage,
                metadata=metadata,
            )

        elif format_type == StreamFormat.JSONL:
            result = self.format_jsonl(
                text=text,
                chunk_index=chunk_index,
                stream_id=stream_id,
                provider=provider,
                model=model,
                is_final=is_final,
                finish_reason=finish_reason,
                usage=usage,
                metadata=metadata,
            )
            return result.content

        else:
            # Raw format - just return the text
            return text


# =============================================================================
# Streaming Headers Builder
# =============================================================================


class StreamingHeadersBuilder:
    """
    Builds HTTP headers for streaming responses.

    Provides consistent headers with selection tracing information.
    """

    @staticmethod
    def build_sse_headers(
        provider: str,
        model: str,
        session_id: Optional[str] = None,
        stream_id: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Build headers for SSE streaming response.

        Args:
            provider: The provider being used
            model: The model being used
            session_id: Optional session identifier
            stream_id: Optional stream identifier

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Stream-Provider": provider,
            "X-Stream-Model": model,
            "X-Stream-Started-At": datetime.utcnow().isoformat(),
        }

        if session_id:
            headers["X-Stream-Session-Id"] = session_id
        if stream_id:
            headers["X-Stream-Id"] = stream_id

        return headers

    @staticmethod
    def build_jsonl_headers(
        provider: str,
        model: str,
        session_id: Optional[str] = None,
        stream_id: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Build headers for JSON Lines streaming response.

        Args:
            provider: The provider being used
            model: The model being used
            session_id: Optional session identifier
            stream_id: Optional stream identifier

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Content-Type": "application/x-ndjson",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Stream-Provider": provider,
            "X-Stream-Model": model,
            "X-Stream-Started-At": datetime.utcnow().isoformat(),
        }

        if session_id:
            headers["X-Stream-Session-Id"] = session_id
        if stream_id:
            headers["X-Stream-Id"] = stream_id

        return headers


# =============================================================================
# Factory Functions
# =============================================================================


_default_formatter: Optional[StreamResponseFormatter] = None


def get_stream_formatter(
    include_metadata: bool = True,
    pretty_json: bool = False,
) -> StreamResponseFormatter:
    """
    Get a stream formatter instance.

    Args:
        include_metadata: Include metadata in formatted output
        pretty_json: Use pretty-printed JSON

    Returns:
        StreamResponseFormatter instance
    """
    global _default_formatter

    # Return cached default formatter for common case
    if include_metadata and not pretty_json:
        if _default_formatter is None:
            _default_formatter = StreamResponseFormatter()
        return _default_formatter

    # Create custom formatter
    return StreamResponseFormatter(
        include_metadata=include_metadata,
        pretty_json=pretty_json,
    )


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Main class
    "StreamResponseFormatter",
    # Header builder
    "StreamingHeadersBuilder",
    # Data classes
    "FormattedChunk",
    # Enums
    "StreamFormat",
    "SSEEventType",
    # Constants
    "SSE_RETRY_MS",
    "SSE_KEEP_ALIVE_COMMENT",
    # Factory functions
    "get_stream_formatter",
]
