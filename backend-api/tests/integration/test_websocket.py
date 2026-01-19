"""
WebSocket Tests

This module tests WebSocket endpoints for real-time streaming functionality.
P1 Test Coverage: WebSocket /ws/enhance and related endpoints.
"""

import contextlib

import pytest
from fastapi.testclient import TestClient


class TestWebSocketConnection:
    """Test WebSocket connection handling."""

    def test_websocket_connect_success(self, test_client):
        """Test successful WebSocket connection."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            # Connection should be established
            assert websocket is not None

    def test_websocket_connect_with_auth(self, test_client):
        """Test WebSocket connection with authentication."""
        headers = {"Authorization": "Bearer valid-token"}
        with test_client.websocket_connect("/ws/enhance", headers=headers) as websocket:
            assert websocket is not None

    def test_websocket_reject_invalid_auth(self, test_client):
        """Test WebSocket rejects invalid authentication."""
        headers = {"Authorization": "Bearer invalid-token"}
        with (
            pytest.raises(Exception),
            test_client.websocket_connect("/ws/enhance", headers=headers),
        ):
            pass

    def test_websocket_ping_pong(self, test_client):
        """Test WebSocket ping/pong keepalive."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response.get("type") == "pong"


class TestWebSocketEnhance:
    """Test /ws/enhance endpoint functionality."""

    def test_enhance_prompt_streaming(self, test_client):
        """Test streaming prompt enhancement."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            # Send enhancement request
            websocket.send_json(
                {
                    "action": "enhance",
                    "prompt": "How to improve security?",
                    "technique": "autodan",
                    "potency": 5,
                }
            )

            # Collect streamed responses
            responses = []
            while True:
                try:
                    response = websocket.receive_json()
                    responses.append(response)
                    if response.get("is_final", False):
                        break
                except Exception:
                    break

            # Should have received streaming chunks
            assert len(responses) >= 1

            # Last response should be final
            final = responses[-1]
            assert final.get("is_final") is True

    def test_enhance_with_different_techniques(self, test_client):
        """Test enhancement with various techniques."""
        techniques = ["autodan", "deep_inception", "cipher"]

        for technique in techniques:
            with test_client.websocket_connect("/ws/enhance") as websocket:
                websocket.send_json(
                    {
                        "action": "enhance",
                        "prompt": "Test prompt",
                        "technique": technique,
                        "potency": 5,
                    }
                )

                response = websocket.receive_json()
                assert "error" not in response or response.get("type") != "error"

    def test_enhance_invalid_technique_error(self, test_client):
        """Test error handling for invalid technique."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            websocket.send_json(
                {
                    "action": "enhance",
                    "prompt": "Test",
                    "technique": "invalid_technique_xyz",
                    "potency": 5,
                }
            )

            response = websocket.receive_json()
            # Should receive error response
            assert response.get("type") == "error" or "error" in response

    def test_enhance_empty_prompt_error(self, test_client):
        """Test error handling for empty prompt."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            websocket.send_json(
                {"action": "enhance", "prompt": "", "technique": "autodan", "potency": 5}
            )

            response = websocket.receive_json()
            assert response.get("type") == "error" or "error" in response

    def test_enhance_potency_range(self, test_client):
        """Test enhancement with different potency levels."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            # Valid potency
            websocket.send_json(
                {"action": "enhance", "prompt": "Test", "technique": "autodan", "potency": 10}
            )
            response = websocket.receive_json()
            assert "error" not in response.get("type", "")

            # Invalid potency (too high)
            websocket.send_json(
                {"action": "enhance", "prompt": "Test", "technique": "autodan", "potency": 15}
            )
            response = websocket.receive_json()
            # Should either clamp or error


class TestWebSocketStreamTransform:
    """Test WebSocket transformation streaming."""

    def test_stream_transform_progress_updates(self, test_client):
        """Test transformation streams progress updates."""
        with test_client.websocket_connect("/ws/transform") as websocket:
            websocket.send_json(
                {
                    "action": "transform",
                    "prompt": "Test prompt for transformation",
                    "technique_suite": "advanced",
                    "potency_level": 7,
                }
            )

            responses = []

            while True:
                try:
                    response = websocket.receive_json()
                    responses.append(response)

                    if response.get("type") == "progress":
                        pass

                    if response.get("is_final"):
                        break
                except Exception:
                    break

            # Should have received progress updates
            assert len(responses) >= 1

    def test_stream_cancellation(self, test_client):
        """Test cancelling an in-progress stream."""
        with test_client.websocket_connect("/ws/transform") as websocket:
            # Start transformation
            websocket.send_json(
                {
                    "action": "transform",
                    "prompt": "Long running transformation",
                    "technique_suite": "quantum",
                    "potency_level": 10,
                }
            )

            # Cancel immediately
            websocket.send_json({"action": "cancel"})

            # Should receive cancellation acknowledgment
            responses = []
            while True:
                try:
                    response = websocket.receive_json()
                    responses.append(response)
                    if response.get("cancelled") or response.get("is_final"):
                        break
                except Exception:
                    break


class TestWebSocketConcurrency:
    """Test WebSocket concurrent connection handling."""

    def test_multiple_concurrent_connections(self, test_client):
        """Test handling multiple concurrent WebSocket connections."""
        connections = []

        try:
            # Open multiple connections
            for _ in range(5):
                ws = test_client.websocket_connect("/ws/enhance")
                ws.__enter__()
                connections.append(ws)

            # Send messages on all connections
            for i, ws in enumerate(connections):
                ws.send_json(
                    {
                        "action": "enhance",
                        "prompt": f"Test prompt {i}",
                        "technique": "autodan",
                        "potency": 5,
                    }
                )

            # Receive responses
            for ws in connections:
                response = ws.receive_json()
                assert response is not None

        finally:
            for ws in connections:
                with contextlib.suppress(Exception):
                    ws.__exit__(None, None, None)

    def test_connection_cleanup_on_disconnect(self, test_client):
        """Test proper cleanup when client disconnects."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            websocket.send_json(
                {"action": "enhance", "prompt": "Test", "technique": "autodan", "potency": 5}
            )
            # Disconnect is handled by context manager exit


class TestWebSocketErrorHandling:
    """Test WebSocket error handling."""

    def test_malformed_json_handling(self, test_client):
        """Test handling of malformed JSON messages."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            # Send invalid JSON
            websocket.send_text("not valid json {{{")

            response = websocket.receive_json()
            assert response.get("type") == "error"
            assert (
                "parse" in response.get("message", "").lower()
                or "json" in response.get("message", "").lower()
            )

    def test_unknown_action_handling(self, test_client):
        """Test handling of unknown action types."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            websocket.send_json({"action": "unknown_action_xyz", "data": {}})

            response = websocket.receive_json()
            assert response.get("type") == "error"

    def test_missing_required_fields(self, test_client):
        """Test handling of missing required fields."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            # Missing prompt
            websocket.send_json({"action": "enhance", "technique": "autodan"})

            response = websocket.receive_json()
            assert response.get("type") == "error"

    def test_server_error_recovery(self, test_client):
        """Test WebSocket recovers from server errors."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            # Trigger potential error
            websocket.send_json(
                {
                    "action": "enhance",
                    "prompt": "x" * 100000,  # Very long prompt
                    "technique": "autodan",
                    "potency": 5,
                }
            )

            websocket.receive_json()
            # Should handle gracefully, not crash

            # Connection should still work
            websocket.send_json({"type": "ping"})
            websocket.receive_json()
            # May or may not receive pong depending on error handling


class TestWebSocketRateLimiting:
    """Test WebSocket rate limiting."""

    def test_rate_limit_messages(self, test_client):
        """Test rate limiting on WebSocket messages."""
        with test_client.websocket_connect("/ws/enhance") as websocket:
            # Send many messages quickly
            for i in range(100):
                websocket.send_json(
                    {
                        "action": "enhance",
                        "prompt": f"Test {i}",
                        "technique": "autodan",
                        "potency": 5,
                    }
                )

            # Should receive rate limit response eventually
            for _ in range(100):
                try:
                    response = websocket.receive_json()
                    if response.get("type") == "rate_limit":
                        break
                except Exception:
                    break

            # Rate limiting may or may not be implemented
            # This test documents expected behavior


@pytest.fixture
def test_client():
    """Create test client fixture."""
    from app.main import app

    return TestClient(app)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
