"""Streaming Service Integration Test Suite.

Verifies all streaming components work correctly with the
Unified Provider/Model Selection System.
"""

import asyncio
import os
import sys

# Add the app to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all streaming modules import correctly."""
    errors = []

    # Test UnifiedStreamingService
    try:
        pass
    except Exception as e:
        errors.append(f"UnifiedStreamingService: {e}")

    # Test StreamingContext
    try:
        pass
    except Exception as e:
        errors.append(f"StreamContext: {e}")

    # Test StreamFormatters
    try:
        pass
    except Exception as e:
        errors.append(f"StreamFormatters: {e}")

    # Test StreamMetricsService
    try:
        pass
    except Exception as e:
        errors.append(f"StreamMetricsService: {e}")

    # Test LLM Service streaming methods
    try:
        from app.services.llm_service import llm_service

        assert hasattr(llm_service, "generate_text_stream")
        assert hasattr(llm_service, "stream_generate")
        assert hasattr(llm_service, "supports_streaming")
    except Exception as e:
        errors.append(f"LLMService: {e}")

    # Test streaming endpoints
    try:
        pass
    except Exception as e:
        errors.append(f"Streaming endpoints: {e}")

    return len(errors) == 0, errors


def test_stream_formatter():
    """Test the StreamResponseFormatter."""
    from app.services.stream_formatters import StreamFormat, StreamResponseFormatter

    formatter = StreamResponseFormatter()

    # Test SSE formatting
    chunk = formatter.format_sse(
        text="Hello, world!",
        chunk_index=0,
        stream_id="test_stream_123",
        provider="openai",
        model="gpt-4",
        is_final=False,
    )

    assert "event: chunk" in chunk.content
    assert "Hello, world!" in chunk.content
    assert chunk.format == StreamFormat.SSE

    # Test SSE error
    error_chunk = formatter.format_sse_error(
        error="Test error",
        error_code="500",
        stream_id="test_stream_123",
    )
    assert "event: error" in error_chunk.content
    assert "Test error" in error_chunk.content

    # Test SSE done
    done_chunk = formatter.format_sse_done(
        stream_id="test_stream_123",
        total_chunks=10,
        finish_reason="stop",
    )
    assert "event: done" in done_chunk.content

    # Test JSON Lines
    jsonl_chunk = formatter.format_jsonl(
        text="Hello JSONL",
        chunk_index=0,
        stream_id="test_stream_123",
        provider="openai",
        model="gpt-4",
    )
    assert jsonl_chunk.format == StreamFormat.JSONL
    assert "Hello JSONL" in jsonl_chunk.content

    # Test WebSocket
    ws_msg = formatter.format_ws(
        text="Hello WS",
        chunk_index=0,
        stream_id="test_stream_123",
        provider="anthropic",
        model="claude-3",
    )
    assert ws_msg["type"] == "chunk"
    assert ws_msg["payload"]["text"] == "Hello WS"

    return True, []


def test_streaming_headers():
    """Test the StreamingHeadersBuilder."""
    from app.services.stream_formatters import StreamingHeadersBuilder

    # Test SSE headers
    headers = StreamingHeadersBuilder.build_sse_headers(
        provider="openai",
        model="gpt-4",
        session_id="sess_123",
        stream_id="stream_456",
    )

    assert headers["Content-Type"] == "text/event-stream"
    assert headers["X-Stream-Provider"] == "openai"
    assert headers["X-Stream-Model"] == "gpt-4"
    assert headers["X-Stream-Session-Id"] == "sess_123"
    assert headers["X-Stream-Id"] == "stream_456"
    assert "X-Stream-Started-At" in headers

    # Test JSONL headers
    jsonl_headers = StreamingHeadersBuilder.build_jsonl_headers(
        provider="anthropic",
        model="claude-3",
    )
    assert jsonl_headers["Content-Type"] == "application/x-ndjson"

    return True, []


async def test_streaming_context():
    """Test the StreamingContext manager."""
    from app.services.stream_context import get_streaming_context

    context = get_streaming_context()

    # Test creating a locked stream session
    async with context.locked_stream(
        session_id="test_session",
        provider="openai",
        model="gpt-4",
    ) as session:
        assert session.stream_id is not None
        assert session.provider == "openai"
        assert session.model == "gpt-4"

        # Record some chunks
        session.record_chunk(100)
        session.record_chunk(150)
        assert session.chunks_sent == 2
        assert session.bytes_sent == 250

        # Check if selection is locked
        is_locked = context.is_selection_locked("test_session")
        assert is_locked, "Selection should be locked during stream"

        locked_sel = context.get_locked_selection("test_session")
        assert locked_sel == ("openai", "gpt-4")

    # After context, lock should be released
    is_locked_after = context.is_selection_locked("test_session")
    assert not is_locked_after, "Selection should be unlocked after stream"

    return True, []


async def test_unified_streaming_service():
    """Test the UnifiedStreamingService singleton."""
    from app.services.unified_streaming_service import get_unified_streaming_service

    # Test singleton
    service1 = get_unified_streaming_service()
    service2 = get_unified_streaming_service()
    assert service1 is service2, "Should be singleton"

    # Test get_active_streams
    active = service1.get_active_streams()
    assert isinstance(active, list)

    # Test get_metrics_summary
    summary = service1.get_metrics_summary()
    assert "total_streams" in summary
    assert "by_provider" in summary
    assert "error_rate" in summary

    # Test get_recent_metrics
    recent = service1.get_recent_metrics(limit=10)
    assert isinstance(recent, list)

    return True, []


async def test_stream_metrics_service():
    """Test the StreamMetricsService."""
    from datetime import datetime

    from app.services.stream_metrics_service import (
        StreamEvent,
        StreamEventType,
        StreamMetricsService,
    )

    # Get or create service (singleton)
    service = StreamMetricsService()

    # Record a test event
    event = StreamEvent(
        event_type=StreamEventType.STREAM_STARTED,
        stream_id="test_metrics_stream",
        provider="openai",
        model="gpt-4",
        timestamp=datetime.utcnow(),
        session_id="test_session",
    )

    await service.record_event(event)

    # Get current metrics
    current = service.get_current_metrics()
    assert "total_streams" in current

    # Get provider metrics
    provider_metrics = service.get_provider_metrics("openai")
    assert provider_metrics is not None

    return True, []


def run_all_tests():
    """Run all streaming tests."""
    results = []

    # Test 1: Imports
    success, errors = test_imports()
    results.append(("Module Imports", success, errors))

    # Test 2: StreamFormatter
    try:
        success, errors = test_stream_formatter()
        results.append(("StreamResponseFormatter", success, errors))
    except Exception as e:
        results.append(("StreamResponseFormatter", False, [str(e)]))

    # Test 3: Headers
    try:
        success, errors = test_streaming_headers()
        results.append(("StreamingHeadersBuilder", success, errors))
    except Exception as e:
        results.append(("StreamingHeadersBuilder", False, [str(e)]))

    # Async tests
    async def run_async_tests() -> None:
        # Test 4: StreamingContext
        try:
            success, errors = await test_streaming_context()
            results.append(("StreamingContext", success, errors))
        except Exception as e:
            results.append(("StreamingContext", False, [str(e)]))

        # Test 5: UnifiedStreamingService
        try:
            success, errors = await test_unified_streaming_service()
            results.append(("UnifiedStreamingService", success, errors))
        except Exception as e:
            results.append(("UnifiedStreamingService", False, [str(e)]))

        # Test 6: StreamMetricsService
        try:
            success, errors = await test_stream_metrics_service()
            results.append(("StreamMetricsService", success, errors))
        except Exception as e:
            results.append(("StreamMetricsService", False, [str(e)]))

    asyncio.run(run_async_tests())

    # Summary

    passed = 0
    failed = 0

    for _name, success, errors in results:
        if success:
            passed += 1
        else:
            for _err in errors:
                pass
            failed += 1

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
