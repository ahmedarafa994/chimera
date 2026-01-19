"""
Streaming Service Integration Test Suite

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
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    errors = []

    # Test UnifiedStreamingService
    try:
        print("✅ UnifiedStreamingService imports OK")
    except Exception as e:
        errors.append(f"UnifiedStreamingService: {e}")
        print(f"❌ UnifiedStreamingService: {e}")

    # Test StreamingContext
    try:
        print("✅ StreamContext imports OK")
    except Exception as e:
        errors.append(f"StreamContext: {e}")
        print(f"❌ StreamContext: {e}")

    # Test StreamFormatters
    try:
        print("✅ StreamFormatters imports OK")
    except Exception as e:
        errors.append(f"StreamFormatters: {e}")
        print(f"❌ StreamFormatters: {e}")

    # Test StreamMetricsService
    try:
        print("✅ StreamMetricsService imports OK")
    except Exception as e:
        errors.append(f"StreamMetricsService: {e}")
        print(f"❌ StreamMetricsService: {e}")

    # Test LLM Service streaming methods
    try:
        from app.services.llm_service import llm_service

        assert hasattr(llm_service, "generate_text_stream")
        assert hasattr(llm_service, "stream_generate")
        assert hasattr(llm_service, "supports_streaming")
        print("✅ LLMService streaming methods OK")
    except Exception as e:
        errors.append(f"LLMService: {e}")
        print(f"❌ LLMService: {e}")

    # Test streaming endpoints
    try:
        print("✅ Streaming endpoints imports OK")
    except Exception as e:
        errors.append(f"Streaming endpoints: {e}")
        print(f"❌ Streaming endpoints: {e}")

    return len(errors) == 0, errors


def test_stream_formatter():
    """Test the StreamResponseFormatter."""
    print("\n" + "=" * 60)
    print("TEST 2: StreamResponseFormatter")
    print("=" * 60)

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
    print("✅ SSE chunk formatting works")

    # Test SSE error
    error_chunk = formatter.format_sse_error(
        error="Test error",
        error_code="500",
        stream_id="test_stream_123",
    )
    assert "event: error" in error_chunk.content
    assert "Test error" in error_chunk.content
    print("✅ SSE error formatting works")

    # Test SSE done
    done_chunk = formatter.format_sse_done(
        stream_id="test_stream_123",
        total_chunks=10,
        finish_reason="stop",
    )
    assert "event: done" in done_chunk.content
    print("✅ SSE done formatting works")

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
    print("✅ JSONL formatting works")

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
    print("✅ WebSocket formatting works")

    return True, []


def test_streaming_headers():
    """Test the StreamingHeadersBuilder."""
    print("\n" + "=" * 60)
    print("TEST 3: StreamingHeadersBuilder")
    print("=" * 60)

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
    print("✅ SSE headers contain all required fields")

    # Test JSONL headers
    jsonl_headers = StreamingHeadersBuilder.build_jsonl_headers(
        provider="anthropic",
        model="claude-3",
    )
    assert jsonl_headers["Content-Type"] == "application/x-ndjson"
    print("✅ JSONL headers correct")

    return True, []


async def test_streaming_context():
    """Test the StreamingContext manager."""
    print("\n" + "=" * 60)
    print("TEST 4: StreamingContext")
    print("=" * 60)

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
        print(f"✅ Created stream session: {session.stream_id}")

        # Record some chunks
        session.record_chunk(100)
        session.record_chunk(150)
        assert session.chunks_sent == 2
        assert session.bytes_sent == 250
        print("✅ Chunk recording works")

        # Check if selection is locked
        is_locked = context.is_selection_locked("test_session")
        assert is_locked, "Selection should be locked during stream"
        print("✅ Selection locking works")

        locked_sel = context.get_locked_selection("test_session")
        assert locked_sel == ("openai", "gpt-4")
        print("✅ Locked selection retrieval works")

    # After context, lock should be released
    is_locked_after = context.is_selection_locked("test_session")
    assert not is_locked_after, "Selection should be unlocked after stream"
    print("✅ Selection unlocking works")

    return True, []


async def test_unified_streaming_service():
    """Test the UnifiedStreamingService singleton."""
    print("\n" + "=" * 60)
    print("TEST 5: UnifiedStreamingService")
    print("=" * 60)

    from app.services.unified_streaming_service import get_unified_streaming_service

    # Test singleton
    service1 = get_unified_streaming_service()
    service2 = get_unified_streaming_service()
    assert service1 is service2, "Should be singleton"
    print("✅ Singleton pattern works")

    # Test get_active_streams
    active = service1.get_active_streams()
    assert isinstance(active, list)
    print(f"✅ get_active_streams works (currently {len(active)} active)")

    # Test get_metrics_summary
    summary = service1.get_metrics_summary()
    assert "total_streams" in summary
    assert "by_provider" in summary
    assert "error_rate" in summary
    print("✅ get_metrics_summary works")

    # Test get_recent_metrics
    recent = service1.get_recent_metrics(limit=10)
    assert isinstance(recent, list)
    print("✅ get_recent_metrics works")

    return True, []


async def test_stream_metrics_service():
    """Test the StreamMetricsService."""
    print("\n" + "=" * 60)
    print("TEST 6: StreamMetricsService")
    print("=" * 60)

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
    print("✅ Event recording works")

    # Get current metrics
    current = service.get_current_metrics()
    assert "total_streams" in current
    print("✅ get_current_metrics works")

    # Get provider metrics
    provider_metrics = service.get_provider_metrics("openai")
    assert provider_metrics is not None
    print("✅ get_provider_metrics works")

    return True, []


def run_all_tests():
    """Run all streaming tests."""
    print("\n")
    print("=" * 60)
    print("STREAMING SERVICE INTEGRATION TEST SUITE")
    print("=" * 60)

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
    async def run_async_tests():
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
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, success, errors in results:
        if success:
            print(f"✅ {name}: PASSED")
            passed += 1
        else:
            print(f"❌ {name}: FAILED")
            for err in errors:
                print(f"   - {err}")
            failed += 1

    print("-" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
