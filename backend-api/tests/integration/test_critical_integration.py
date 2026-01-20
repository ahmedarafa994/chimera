"""Critical Integration Tests.

Tests for critical API endpoints to ensure frontend-backend integration works correctly.
These tests verify the fixes implemented for INT-001 through INT-005.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the API."""
    from app.main import app

    return TestClient(app)


class TestProvidersEndpoint:
    """Tests for /api/v1/providers endpoint (INT-005 FIX)."""

    def test_providers_endpoint_exists(self, client) -> None:
        """Test that /providers endpoint returns 200."""
        response = client.get("/api/v1/providers")
        assert response.status_code == 200

    def test_providers_response_structure(self, client) -> None:
        """Test that providers response has expected structure."""
        response = client.get("/api/v1/providers")
        data = response.json()

        assert "providers" in data
        assert "count" in data
        assert "default" in data
        assert isinstance(data["providers"], list)
        assert isinstance(data["count"], int)


class TestModelsEndpoint:
    """Tests for /api/v1/models endpoint (INT-003 FIX)."""

    def test_models_list_endpoint_exists(self, client) -> None:
        """Test that /models endpoint returns 200."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

    def test_models_response_structure(self, client) -> None:
        """Test that models response has expected structure."""
        response = client.get("/api/v1/models")
        data = response.json()

        assert "providers" in data
        assert "default_provider" in data
        assert "default_model" in data
        assert "total_models" in data
        assert isinstance(data["providers"], list)

    def test_models_validate_endpoint(self, client) -> None:
        """Test that /models/validate endpoint works."""
        response = client.post(
            "/api/v1/models/validate",
            json={"provider": "google", "model": "gemini-2.0-flash-exp"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "message" in data

    def test_models_available_endpoint(self, client) -> None:
        """Test that /models/available endpoint works."""
        response = client.get("/api/v1/models/available")
        assert response.status_code == 200


class TestConnectionEndpoint:
    """Tests for /api/v1/connection endpoints."""

    def test_connection_config_endpoint(self, client) -> None:
        """Test that /connection/config endpoint returns 200."""
        response = client.get("/api/v1/connection/config")
        assert response.status_code == 200

    def test_connection_status_endpoint(self, client) -> None:
        """Test that /connection/status endpoint returns 200."""
        response = client.get("/api/v1/connection/status")
        assert response.status_code == 200

    def test_connection_health_endpoint(self, client) -> None:
        """Test that /connection/health endpoint returns 200."""
        response = client.get("/api/v1/connection/health")
        assert response.status_code == 200


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_ping_endpoint(self, client) -> None:
        """Test lightweight health ping."""
        response = client.get("/health/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_endpoint(self, client) -> None:
        """Test full health check."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_ready_endpoint(self, client) -> None:
        """Test readiness probe."""
        response = client.get("/health/ready")
        # May return 200 or 503 depending on service state
        assert response.status_code in [200, 503]


class TestTechniquesEndpoint:
    """Tests for /api/v1/techniques endpoint."""

    def test_techniques_endpoint_exists(self, client) -> None:
        """Test that /techniques endpoint returns 200."""
        response = client.get("/api/v1/techniques")
        assert response.status_code == 200

    def test_techniques_response_structure(self, client) -> None:
        """Test that techniques response has expected structure."""
        response = client.get("/api/v1/techniques")
        data = response.json()

        assert "techniques" in data
        assert "count" in data
        assert isinstance(data["techniques"], list)


class TestTransformEndpoint:
    """Tests for /api/v1/transform endpoint."""

    def test_transform_endpoint_exists(self, client) -> None:
        """Test that /transform endpoint accepts POST."""
        response = client.post(
            "/api/v1/transform",
            json={"core_request": "Test prompt", "potency_level": 5, "technique_suite": "simple"},
        )
        # Should return 200 or 401 (if auth required)
        assert response.status_code in [200, 401, 422]


class TestOpenAICompatibilityStubs:
    """Tests for OpenAI-compatible endpoint stubs (PERF-034)."""

    def test_chat_completions_returns_501(self, client) -> None:
        """Test that /v1/chat/completions returns 501 Not Implemented."""
        response = client.post("/v1/chat/completions")
        assert response.status_code == 501
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "not_implemented"

    def test_completions_returns_501(self, client) -> None:
        """Test that /v1/completions returns 501 Not Implemented."""
        response = client.post("/v1/completions")
        assert response.status_code == 501

    def test_models_stub_returns_501(self, client) -> None:
        """Test that /v1/models returns 501 Not Implemented."""
        response = client.get("/v1/models")
        assert response.status_code == 501

    def test_messages_returns_501(self, client) -> None:
        """Test that /v1/messages returns 501 Not Implemented."""
        response = client.post("/v1/messages")
        assert response.status_code == 501


class TestSessionEndpoint:
    """Tests for /api/v1/session endpoints."""

    def test_session_create_endpoint(self, client) -> None:
        """Test that session creation works."""
        response = client.post("/api/v1/session", json={})
        # Should return 200 or 401 (if auth required)
        assert response.status_code in [200, 401]


class TestMetricsEndpoint:
    """Tests for /api/v1/metrics endpoint."""

    def test_metrics_endpoint_exists(self, client) -> None:
        """Test that /metrics endpoint returns 200."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "metrics" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client) -> None:
        """Test that root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
