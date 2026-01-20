"""Integration Tests for Unified Streaming Service.

Tests end-to-end streaming integration including:
- SSE format validation
- Selection tracing headers
- Session-scoped selection
- Provider availability handling
- Context locking during streaming
- Metrics collection

References:
- Streaming endpoints: app/api/v1/endpoints/streaming.py
- Unified streaming service: app/services/unified_streaming_service.py
- Stream context: app/services/stream_context.py

"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Never
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_stream_chunks():
    """Sample stream chunks for testing."""
    return [
        {
            "text": "Hello",
            "chunk_index": 0,
            "stream_id": "stream_test123",
            "provider": "mock",
            "model": "mock-model",
            "is_final": False,
        },
        {
            "text": " world",
            "chunk_index": 1,
            "stream_id": "stream_test123",
            "provider": "mock",
            "model": "mock-model",
            "is_final": False,
        },
        {
            "text": "!",
            "chunk_index": 2,
            "stream_id": "stream_test123",
            "provider": "mock",
            "model": "mock-model",
            "is_final": True,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        },
    ]


@pytest.fixture
def mock_unified_streaming_service(mock_stream_chunks):
    """Create a mock unified streaming service."""
    service = MagicMock()
    service.get_active_streams = MagicMock(return_value=[])
    service.get_metrics_summary = MagicMock(
        return_value={
            "total_streams": 10,
            "active_streams": 0,
            "by_provider": {"mock": 10},
            "error_rate": 0.0,
            "avg_duration_ms": 150.0,
            "avg_first_chunk_latency_ms": 50.0,
        },
    )
    service.get_recent_metrics = MagicMock(return_value=[])
    service.cancel_stream = AsyncMock(return_value=True)

    # Mock stream generation
    async def mock_stream_generate(**kwargs) -> AsyncIterator[Any]:
        from app.services.unified_streaming_service import UnifiedStreamChunk

        for chunk_data in mock_stream_chunks:
            yield UnifiedStreamChunk(
                text=chunk_data["text"],
                chunk_index=chunk_data["chunk_index"],
                stream_id=chunk_data["stream_id"],
                provider=chunk_data["provider"],
                model=chunk_data["model"],
                is_final=chunk_data["is_final"],
                finish_reason=chunk_data.get("finish_reason"),
                usage=chunk_data.get("usage"),
            )

    service.stream_generate = mock_stream_generate
    service.stream_chat = mock_stream_generate

    return service


@pytest.fixture
def mock_resolution_service():
    """Create a mock provider resolution service."""
    service = MagicMock()

    # Mock resolution response
    resolution_result = MagicMock()
    resolution_result.provider = "mock"
    resolution_result.model_id = "mock-model"
    resolution_result.resolution_source = "global_default"
    resolution_result.resolution_priority = 99

    service.resolve_with_metadata = AsyncMock(return_value=resolution_result)
    return service


@pytest.fixture
def mock_provider_plugin():
    """Create a mock provider plugin."""
    plugin = MagicMock()
    plugin.supports_streaming = MagicMock(return_value=True)
    plugin.is_configured = MagicMock(return_value=True)
    return plugin


# =============================================================================
# SSE Format Tests
# =============================================================================


class TestSSEFormatting:
    """Tests for SSE format validation."""

    def test_sse_chunk_format(self) -> None:
        """Test that SSE chunks are properly formatted."""
        from app.services.stream_formatters import StreamResponseFormatter

        formatter = StreamResponseFormatter()
        chunk = formatter.format_sse(
            text="Hello",
            chunk_index=0,
            stream_id="stream_test123",
            provider="openai",
            model="gpt-4o",
            is_final=False,
        )

        # Verify SSE format
        assert chunk.content.startswith("event: chunk")
        assert "id: 0" in chunk.content
        assert "data: " in chunk.content

        # Parse the data line
        lines = chunk.content.split("\n")
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        assert data_line is not None

        data = json.loads(data_line[6:])  # Skip "data: " prefix
        assert data["text"] == "Hello"
        assert data["chunk_index"] == 0
        assert data["stream_id"] == "stream_test123"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o"
        assert data["is_final"] is False

    def test_sse_done_event_format(self) -> None:
        """Test SSE done event format."""
        from app.services.stream_formatters import StreamResponseFormatter

        formatter = StreamResponseFormatter()
        done_chunk = formatter.format_sse_done(
            stream_id="stream_test123",
            total_chunks=5,
            total_tokens=100,
            finish_reason="stop",
        )

        assert done_chunk.content.startswith("event: done")
        assert "data: " in done_chunk.content
        assert done_chunk.is_final is True

        # Parse the data
        lines = done_chunk.content.split("\n")
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        data = json.loads(data_line[6:])

        assert data["stream_id"] == "stream_test123"
        assert data["total_chunks"] == 5
        assert data["finish_reason"] == "stop"

    def test_sse_error_event_format(self) -> None:
        """Test SSE error event format."""
        from app.services.stream_formatters import StreamResponseFormatter

        formatter = StreamResponseFormatter()
        error_chunk = formatter.format_sse_error(
            error="Provider not available",
            error_code="PROVIDER_ERROR",
            stream_id="stream_test123",
        )

        assert error_chunk.content.startswith("event: error")
        assert error_chunk.is_final is True

        lines = error_chunk.content.split("\n")
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        data = json.loads(data_line[6:])

        assert data["error"] == "Provider not available"
        assert data["error_code"] == "PROVIDER_ERROR"

    def test_sse_keepalive_ping(self) -> None:
        """Test SSE keep-alive ping format."""
        from app.services.stream_formatters import SSE_KEEP_ALIVE_COMMENT, StreamResponseFormatter

        formatter = StreamResponseFormatter()
        ping = formatter.format_sse_ping()

        assert ping.content == SSE_KEEP_ALIVE_COMMENT
        assert ping.is_final is False


# =============================================================================
# Selection Headers Tests
# =============================================================================


class TestSelectionHeaders:
    """Tests for selection tracing headers."""

    def test_sse_headers_include_provider_model(self) -> None:
        """Test that SSE headers include provider/model information."""
        from app.services.stream_formatters import StreamingHeadersBuilder

        headers = StreamingHeadersBuilder.build_sse_headers(
            provider="openai",
            model="gpt-4o",
            session_id="sess_123",
            stream_id="stream_abc",
        )

        assert headers["X-Stream-Provider"] == "openai"
        assert headers["X-Stream-Model"] == "gpt-4o"
        assert headers["X-Stream-Session-Id"] == "sess_123"
        assert headers["X-Stream-Id"] == "stream_abc"
        assert "X-Stream-Started-At" in headers
        assert headers["Content-Type"] == "text/event-stream"

    def test_jsonl_headers_include_provider_model(self) -> None:
        """Test that JSONL headers include provider/model information."""
        from app.services.stream_formatters import StreamingHeadersBuilder

        headers = StreamingHeadersBuilder.build_jsonl_headers(
            provider="anthropic",
            model="claude-3-opus",
            session_id="sess_456",
            stream_id="stream_def",
        )

        assert headers["X-Stream-Provider"] == "anthropic"
        assert headers["X-Stream-Model"] == "claude-3-opus"
        assert headers["X-Stream-Session-Id"] == "sess_456"
        assert headers["X-Stream-Id"] == "stream_def"
        assert headers["Content-Type"] == "application/x-ndjson"

    @pytest.mark.asyncio
    async def test_stream_endpoint_returns_headers(
        self,
        mock_unified_streaming_service,
        mock_resolution_service,
        mock_provider_plugin,
    ) -> None:
        """Test that streaming endpoint returns proper headers."""
        with (
            patch(
                "app.api.v1.endpoints.streaming.get_unified_streaming_service",
                return_value=mock_unified_streaming_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming.get_provider_resolution_service",
                return_value=mock_resolution_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming._validate_provider_available",
                return_value=True,
            ),
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/stream/generate",
                    json={"prompt": "Test prompt"},
                    headers={"X-Session-Id": "sess_test"},
                )

                # Should return streaming response
                assert response.status_code == 200
                assert "X-Stream-Provider" in response.headers
                assert "X-Stream-Model" in response.headers


# =============================================================================
# Session Selection Tests
# =============================================================================


class TestSessionSelection:
    """Tests for session-scoped provider/model selection."""

    @pytest.mark.asyncio
    async def test_stream_uses_session_selection(
        self,
        mock_unified_streaming_service,
        mock_resolution_service,
        mock_provider_plugin,
    ) -> None:
        """Test that stream uses session's provider/model selection."""
        # Configure resolution to return session-based selection
        mock_resolution_service.resolve_with_metadata.return_value.resolution_source = "session"
        mock_resolution_service.resolve_with_metadata.return_value.resolution_priority = 20

        with (
            patch(
                "app.api.v1.endpoints.streaming.get_unified_streaming_service",
                return_value=mock_unified_streaming_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming.get_provider_resolution_service",
                return_value=mock_resolution_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming._validate_provider_available",
                return_value=True,
            ),
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                await client.post(
                    "/api/v1/stream/generate",
                    json={"prompt": "Test prompt"},
                    headers={"X-Session-Id": "sess_test123"},
                )

                # Verify resolution was called with session_id
                mock_resolution_service.resolve_with_metadata.assert_called_once()
                call_args = mock_resolution_service.resolve_with_metadata.call_args
                assert call_args.kwargs.get("session_id") == "sess_test123"

    @pytest.mark.asyncio
    async def test_explicit_override_takes_precedence(
        self,
        mock_unified_streaming_service,
        mock_resolution_service,
        mock_provider_plugin,
    ) -> None:
        """Test that explicit provider/model takes precedence over session."""
        mock_resolution_service.resolve_with_metadata.return_value.resolution_source = "explicit"
        mock_resolution_service.resolve_with_metadata.return_value.resolution_priority = 10

        with (
            patch(
                "app.api.v1.endpoints.streaming.get_unified_streaming_service",
                return_value=mock_unified_streaming_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming.get_provider_resolution_service",
                return_value=mock_resolution_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming._validate_provider_available",
                return_value=True,
            ),
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                await client.post(
                    "/api/v1/stream/generate",
                    json={
                        "prompt": "Test prompt",
                        "provider": "anthropic",
                        "model": "claude-3-opus",
                    },
                    headers={"X-Session-Id": "sess_test123"},
                )

                # Verify resolution was called with explicit params
                call_args = mock_resolution_service.resolve_with_metadata.call_args
                assert call_args.kwargs.get("explicit_provider") == "anthropic"
                assert call_args.kwargs.get("explicit_model") == "claude-3-opus"


# =============================================================================
# Provider Availability Tests
# =============================================================================


class TestProviderAvailability:
    """Tests for provider availability handling."""

    @pytest.mark.asyncio
    async def test_provider_not_available_returns_503(
        self,
        mock_unified_streaming_service,
        mock_resolution_service,
    ) -> None:
        """Test proper error when provider not available."""
        with (
            patch(
                "app.api.v1.endpoints.streaming.get_unified_streaming_service",
                return_value=mock_unified_streaming_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming.get_provider_resolution_service",
                return_value=mock_resolution_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming._validate_provider_available",
                return_value=False,
            ),
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/stream/generate",
                    json={"prompt": "Test prompt"},
                )

                assert response.status_code == 503
                data = response.json()
                assert data["detail"]["error"] == "PROVIDER_NOT_AVAILABLE"

    @pytest.mark.asyncio
    async def test_provider_not_available_includes_provider_name(
        self,
        mock_unified_streaming_service,
        mock_resolution_service,
    ) -> None:
        """Test that provider not available error includes provider name."""
        mock_resolution_service.resolve_with_metadata.return_value.provider = "unavailable_provider"

        with (
            patch(
                "app.api.v1.endpoints.streaming.get_unified_streaming_service",
                return_value=mock_unified_streaming_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming.get_provider_resolution_service",
                return_value=mock_resolution_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming._validate_provider_available",
                return_value=False,
            ),
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/stream/generate",
                    json={"prompt": "Test prompt"},
                )

                data = response.json()
                assert data["detail"]["provider"] == "unavailable_provider"


# =============================================================================
# Stream Context Locking Tests
# =============================================================================


class TestStreamContextLocking:
    """Tests for selection lock during streaming."""

    @pytest.mark.asyncio
    async def test_selection_locked_during_stream(self) -> None:
        """Test that selection is locked during streaming."""
        from app.services.stream_context import get_streaming_context

        context = get_streaming_context()

        async with context.locked_stream(
            session_id="sess_test",
            provider="openai",
            model="gpt-4o",
        ) as session:
            # Selection should be locked
            assert context.is_selection_locked("sess_test")
            locked_selection = context.get_locked_selection("sess_test")
            assert locked_selection == ("openai", "gpt-4o")

            # Record some chunks
            session.record_chunk(100)
            session.record_chunk(50)

            assert session.chunks_sent == 2
            assert session.bytes_sent == 150

        # After context exits, selection should be unlocked
        assert not context.is_selection_locked("sess_test")

    @pytest.mark.asyncio
    async def test_concurrent_streams_same_session(self) -> None:
        """Test handling of concurrent streams for same session."""
        from app.services.stream_context import get_streaming_context

        context = get_streaming_context()

        # Start first stream
        async with context.locked_stream(
            session_id="sess_concurrent",
            provider="openai",
            model="gpt-4o",
            stream_id="stream_1",
        ):
            assert context.is_selection_locked("sess_concurrent")

            # Start second stream with same provider/model (should reuse lock)
            async with context.locked_stream(
                session_id="sess_concurrent",
                provider="openai",
                model="gpt-4o",
                stream_id="stream_2",
            ):
                # Both streams should be active
                sessions = context.get_sessions_for_session_id("sess_concurrent")
                assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_stream_cleanup_on_error(self) -> None:
        """Test that stream context cleans up on error."""
        from app.services.stream_context import get_streaming_context

        context = get_streaming_context()

        try:
            async with context.locked_stream(
                session_id="sess_error",
                provider="openai",
                model="gpt-4o",
            ) as session:
                msg = "Simulated error"
                raise RuntimeError(msg)
        except RuntimeError:
            pass

        # Session should be cleaned up
        assert not context.is_selection_locked("sess_error")
        assert session.error == "Simulated error"
        assert session.completed is True

    @pytest.mark.asyncio
    async def test_stream_cleanup_on_cancellation(self) -> None:
        """Test that stream context cleans up on cancellation."""
        from app.services.stream_context import get_streaming_context

        context = get_streaming_context()

        async def simulate_cancelled_stream() -> Never:
            async with context.locked_stream(
                session_id="sess_cancel",
                provider="openai",
                model="gpt-4o",
            ):
                raise asyncio.CancelledError

        with contextlib.suppress(asyncio.CancelledError):
            await simulate_cancelled_stream()

        # Session should be cleaned up and marked cancelled
        assert not context.is_selection_locked("sess_cancel")


# =============================================================================
# Metrics Collection Tests
# =============================================================================


class TestStreamMetrics:
    """Tests for metrics collection during streaming."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_stream_complete(self) -> None:
        """Test that metrics are recorded when stream completes."""
        from app.services.stream_metrics_service import (
            StreamEvent,
            StreamEventType,
            get_stream_metrics_service,
        )

        metrics_service = get_stream_metrics_service()

        # Reset metrics
        metrics_service.reset_metrics()

        # Record stream events
        await metrics_service.record_event(
            StreamEvent(
                event_type=StreamEventType.STREAM_STARTED,
                stream_id="stream_metrics_test",
                provider="openai",
                model="gpt-4o",
                timestamp=datetime.utcnow(),
                session_id="sess_metrics",
            ),
        )

        await metrics_service.record_event(
            StreamEvent(
                event_type=StreamEventType.STREAM_COMPLETED,
                stream_id="stream_metrics_test",
                provider="openai",
                model="gpt-4o",
                timestamp=datetime.utcnow(),
                session_id="sess_metrics",
                token_count=100,
                latency_ms=150.0,
                finish_reason="stop",
            ),
        )

        # Check metrics
        provider_metrics = metrics_service.get_provider_metrics("openai")
        assert "openai" in provider_metrics
        assert provider_metrics["openai"].total_streams >= 1
        assert provider_metrics["openai"].successful_streams >= 1

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_stream_error(self) -> None:
        """Test that metrics are recorded when stream errors."""
        from app.services.stream_metrics_service import (
            StreamEvent,
            StreamEventType,
            get_stream_metrics_service,
        )

        metrics_service = get_stream_metrics_service()

        # Record error event
        await metrics_service.record_event(
            StreamEvent(
                event_type=StreamEventType.STREAM_STARTED,
                stream_id="stream_error_test",
                provider="openai",
                model="gpt-4o",
                timestamp=datetime.utcnow(),
            ),
        )

        await metrics_service.record_event(
            StreamEvent(
                event_type=StreamEventType.STREAM_ERROR,
                stream_id="stream_error_test",
                provider="openai",
                model="gpt-4o",
                timestamp=datetime.utcnow(),
                error="Connection timeout",
            ),
        )

        # Check that error was recorded
        recent_errors = metrics_service.get_recent_errors(limit=10)
        error_found = any(e["stream_id"] == "stream_error_test" for e in recent_errors)
        assert error_found

    def test_get_stream_metrics_endpoint(self, test_client) -> None:
        """Test the streaming metrics endpoint."""
        response = test_client.get("/api/v1/stream/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "total_streams" in data
        assert "active_streams" in data
        assert "by_provider" in data

    def test_get_active_streams_endpoint(self, test_client) -> None:
        """Test the active streams endpoint."""
        response = test_client.get("/api/v1/stream/active")

        assert response.status_code == 200
        data = response.json()
        assert "active_streams" in data
        assert "count" in data


# =============================================================================
# Stream Format Tests
# =============================================================================


class TestStreamFormats:
    """Tests for different stream output formats."""

    def test_jsonl_format(self) -> None:
        """Test JSON Lines format output."""
        from app.services.stream_formatters import StreamResponseFormatter

        formatter = StreamResponseFormatter()
        chunk = formatter.format_jsonl(
            text="Hello",
            chunk_index=0,
            stream_id="stream_jsonl",
            provider="openai",
            model="gpt-4o",
            is_final=False,
        )

        # JSONL should be single line JSON ending with newline
        assert chunk.content.endswith("\n")
        data = json.loads(chunk.content.strip())
        assert data["text"] == "Hello"
        assert data["chunk_index"] == 0

    def test_websocket_format(self) -> None:
        """Test WebSocket format output."""
        from app.services.stream_formatters import StreamResponseFormatter

        formatter = StreamResponseFormatter()
        message = formatter.format_ws(
            text="Hello",
            chunk_index=0,
            stream_id="stream_ws",
            provider="openai",
            model="gpt-4o",
            is_final=False,
        )

        # WebSocket format returns a dict
        assert isinstance(message, dict)
        assert message["type"] == "chunk"
        assert message["payload"]["text"] == "Hello"
        assert message["payload"]["chunk_index"] == 0

    def test_format_unified_chunk_sse(self) -> None:
        """Test formatting unified chunk as SSE."""
        from app.services.stream_formatters import StreamFormat, StreamResponseFormatter
        from app.services.unified_streaming_service import UnifiedStreamChunk

        chunk = UnifiedStreamChunk(
            text="Hello",
            chunk_index=0,
            stream_id="stream_unified",
            provider="openai",
            model="gpt-4o",
            is_final=False,
        )

        formatter = StreamResponseFormatter()
        formatted = formatter.format_unified_chunk(chunk, StreamFormat.SSE)

        assert "event: chunk" in formatted
        assert "data: " in formatted

    def test_format_unified_chunk_jsonl(self) -> None:
        """Test formatting unified chunk as JSONL."""
        from app.services.stream_formatters import StreamFormat, StreamResponseFormatter
        from app.services.unified_streaming_service import UnifiedStreamChunk

        chunk = UnifiedStreamChunk(
            text="Hello",
            chunk_index=0,
            stream_id="stream_unified",
            provider="openai",
            model="gpt-4o",
            is_final=False,
        )

        formatter = StreamResponseFormatter()
        formatted = formatter.format_unified_chunk(chunk, StreamFormat.JSONL)

        data = json.loads(formatted.strip())
        assert data["text"] == "Hello"


# =============================================================================
# Unified Streaming Service Tests
# =============================================================================


class TestUnifiedStreamingService:
    """Tests for the UnifiedStreamingService."""

    @pytest.mark.asyncio
    async def test_stream_generate_creates_context(self) -> None:
        """Test that stream_generate creates proper context."""
        from app.services.unified_streaming_service import get_unified_streaming_service

        # This test would require mocking the provider plugins
        # For now, we test the service instantiation
        service = get_unified_streaming_service()
        assert service is not None
        assert service._initialized is True

    def test_get_metrics_summary(self) -> None:
        """Test getting metrics summary."""
        from app.services.unified_streaming_service import get_unified_streaming_service

        service = get_unified_streaming_service()
        summary = service.get_metrics_summary()

        assert "total_streams" in summary
        assert "active_streams" in summary
        assert "by_provider" in summary
        assert "error_rate" in summary

    def test_get_active_streams(self) -> None:
        """Test getting active streams."""
        from app.services.unified_streaming_service import get_unified_streaming_service

        service = get_unified_streaming_service()
        active = service.get_active_streams()

        assert isinstance(active, list)


# =============================================================================
# Chat Streaming Tests
# =============================================================================


class TestChatStreaming:
    """Tests for chat streaming functionality."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_accepts_messages(
        self,
        mock_unified_streaming_service,
        mock_resolution_service,
    ) -> None:
        """Test that chat endpoint accepts message format."""
        with (
            patch(
                "app.api.v1.endpoints.streaming.get_unified_streaming_service",
                return_value=mock_unified_streaming_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming.get_provider_resolution_service",
                return_value=mock_resolution_service,
            ),
            patch(
                "app.api.v1.endpoints.streaming._validate_provider_available",
                return_value=True,
            ),
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/stream/chat",
                    json={
                        "messages": [
                            {"role": "system", "content": "You are helpful"},
                            {"role": "user", "content": "Hello!"},
                        ],
                        "temperature": 0.7,
                    },
                )

                assert response.status_code == 200


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestStreamingErrorHandling:
    """Tests for streaming error handling."""

    def test_streaming_error_exception(self) -> None:
        """Test StreamingError exception."""
        from app.api.v1.endpoints.streaming import StreamingError

        error = StreamingError(
            message="Test error",
            error_code="TEST_ERROR",
            status_code=500,
            stream_id="stream_test",
        )

        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.status_code == 500
        assert error.stream_id == "stream_test"

    def test_provider_not_available_error(self) -> None:
        """Test ProviderNotAvailableError exception."""
        from app.api.v1.endpoints.streaming import ProviderNotAvailableError

        error = ProviderNotAvailableError(
            provider="unavailable",
            stream_id="stream_test",
        )

        assert error.provider == "unavailable"
        assert error.status_code == 503
        assert error.error_code == "PROVIDER_NOT_AVAILABLE"

    def test_model_not_found_error(self) -> None:
        """Test ModelNotFoundError exception."""
        from app.api.v1.endpoints.streaming import ModelNotFoundError

        error = ModelNotFoundError(
            model="nonexistent-model",
            provider="openai",
            stream_id="stream_test",
        )

        assert error.model == "nonexistent-model"
        assert error.provider == "openai"
        assert error.status_code == 404
        assert error.error_code == "MODEL_NOT_FOUND"

    def test_rate_limit_exceeded_error(self) -> None:
        """Test RateLimitExceededError exception."""
        from app.api.v1.endpoints.streaming import RateLimitExceededError

        error = RateLimitExceededError(
            provider="openai",
            retry_after=60,
            stream_id="stream_test",
        )

        assert error.provider == "openai"
        assert error.retry_after == 60
        assert error.status_code == 429
        assert error.error_code == "RATE_LIMIT_EXCEEDED"


# =============================================================================
# Stream Cancellation Tests
# =============================================================================


class TestStreamCancellation:
    """Tests for stream cancellation."""

    def test_cancel_nonexistent_stream_returns_404(self, test_client) -> None:
        """Test cancelling a nonexistent stream returns 404."""
        response = test_client.delete("/api/v1/stream/nonexistent_stream_id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_active_stream(self, mock_unified_streaming_service) -> None:
        """Test cancelling an active stream."""
        mock_unified_streaming_service.cancel_stream = AsyncMock(return_value=True)

        with patch(
            "app.api.v1.endpoints.streaming.get_unified_streaming_service",
            return_value=mock_unified_streaming_service,
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.delete("/api/v1/stream/stream_active")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "cancelled"


# =============================================================================
# Capabilities Tests
# =============================================================================


class TestStreamingCapabilities:
    """Tests for streaming capabilities endpoint."""

    def test_get_capabilities_endpoint(self, test_client) -> None:
        """Test the streaming capabilities endpoint."""
        response = test_client.get("/api/v1/generate/stream/capabilities")

        assert response.status_code == 200
        data = response.json()
        assert "providers" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
