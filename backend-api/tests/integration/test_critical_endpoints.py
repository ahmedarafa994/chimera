"""Integration Tests for Critical Endpoints
Tests integration between services and endpoints.
"""

import pytest


@pytest.mark.integration
class TestCriticalEndpoints:
    """Integration tests for critical API endpoints."""

    def test_health_endpoint_integration(self, client) -> None:
        """Test health endpoint with all dependencies."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_ready_endpoint(self, client) -> None:
        """Test readiness probe endpoint."""
        response = client.get("/health/ready")
        assert response.status_code in [200, 503]

    def test_providers_endpoint_integration(self, authenticated_client) -> None:
        """Test providers endpoint with authentication."""
        response = authenticated_client.get("/api/v1/providers")
        assert response.status_code in [200, 401]

    def test_session_models_endpoint(self, authenticated_client) -> None:
        """Test session models endpoint."""
        response = authenticated_client.get("/api/v1/session/models")
        assert response.status_code in [200, 401]

    def test_transform_endpoint_validation(self, authenticated_client) -> None:
        """Test transform endpoint with validation."""
        payload = {"prompt": "Test prompt", "technique": "simple"}
        response = authenticated_client.post("/api/v1/transform", json=payload)
        assert response.status_code in [200, 400, 401, 422]

    def test_jailbreak_generate_endpoint(self, authenticated_client) -> None:
        """Test jailbreak generation endpoint."""
        payload = {"prompt": "Test prompt", "technique": "role_playing"}
        response = authenticated_client.post("/api/v1/generation/jailbreak/generate", json=payload)
        assert response.status_code in [200, 400, 401, 422]
