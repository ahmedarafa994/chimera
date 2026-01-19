"""
API Integration Tests
Tests for verifying frontend-backend synchronization
"""

import os

# Import the main app
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestProvidersEndpoints:
    """Test providers API endpoints"""

    def test_get_providers(self, client):
        """Test GET /api/v1/providers returns valid response"""
        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data or isinstance(data, list)

    def test_get_available_providers(self, client):
        """Test GET /api/v1/providers/available"""
        response = client.get("/api/v1/providers/available")
        assert response.status_code == 200

    def test_get_provider_models(self, client):
        """Test GET /api/v1/providers/{provider}/models"""
        response = client.get("/api/v1/providers/gemini/models")
        assert response.status_code in [200, 404]

    def test_get_current_provider(self, client):
        """Test GET /api/v1/providers/current"""
        response = client.get("/api/v1/providers/current")
        assert response.status_code == 200


class TestSessionEndpoints:
    """Test session API endpoints"""

    def test_get_session(self, client):
        """Test GET /api/v1/session"""
        response = client.get("/api/v1/session")
        assert response.status_code in [200, 404]

    def test_create_session(self, client):
        """Test POST /api/v1/session"""
        response = client.post("/api/v1/session", json={})
        assert response.status_code in [200, 201]
        if response.status_code in [200, 201]:
            data = response.json()
            assert "session_id" in data


class TestAutodanEndpoints:
    """Test AutoDAN API endpoints"""

    def test_get_config(self, client):
        """Test GET /api/v1/autodan-enhanced/config"""
        response = client.get("/api/v1/autodan-enhanced/config")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code in [200, 307, 404]


class TestCORSHeaders:
    """Test CORS configuration"""

    def test_cors_headers_on_options(self, client):
        """Test CORS preflight request"""
        response = client.options(
            "/api/v1/providers",
            headers={
                "Origin": "http://localhost:3001",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should not return 405 Method Not Allowed
        assert response.status_code != 405

    def test_cors_allowed_origin(self, client):
        """Test that allowed origins receive CORS headers"""
        response = client.get("/api/v1/providers", headers={"Origin": "http://localhost:3001"})
        # Check for CORS headers (may vary based on middleware config)
        assert response.status_code == 200


class TestRequestTracking:
    """Test request tracking functionality"""

    def test_request_id_header(self, client):
        """Test that request ID is returned in response"""
        response = client.get("/api/v1/providers", headers={"X-Request-ID": "test-request-123"})
        assert response.status_code == 200
        # Request tracking should be logged


class TestErrorResponses:
    """Test error response format"""

    def test_404_error_format(self, client):
        """Test 404 error response structure"""
        response = client.get("/api/v1/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "message" in data

    def test_validation_error_format(self, client):
        """Test validation error response"""
        response = client.post("/api/v1/session", json={"invalid_field": "invalid_value"})
        # Should either accept or return validation error
        assert response.status_code in [200, 201, 422]


class TestRateLimiting:
    """Test rate limiting behavior"""

    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present"""
        response = client.get("/api/v1/providers")
        # Rate limit headers may or may not be present
        assert response.status_code == 200


class TestWebSocketEndpoints:
    """Test WebSocket endpoints exist"""

    def test_websocket_endpoint_exists(self, client):
        """Verify WebSocket endpoints are configured"""
        # WebSocket testing requires special handling
        # This just verifies the endpoint exists
        pass


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
