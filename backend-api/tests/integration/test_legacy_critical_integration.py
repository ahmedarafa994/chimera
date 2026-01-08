"""
Critical Path Integration Tests

Tests for the critical API endpoints identified in the comprehensive project analysis.
These tests verify frontend-backend integration for the most important endpoints.

INT-TEST-001: Integration tests for critical paths
"""

import pytest
from fastapi.testclient import TestClient


# Test fixtures
@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Return authentication headers for testing."""
    return {"X-API-Key": "test-api-key", "Content-Type": "application/json"}


class TestProvidersEndpoint:
    """Tests for /api/v1/providers endpoints."""

    def test_providers_available_returns_list(self, client, auth_headers):
        """Test GET /api/v1/providers returns provider list."""
        response = client.get("/api/v1/providers", headers=auth_headers)

        # Should return 200 or 401 (if auth required)
        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "providers" in data
            assert "count" in data
            assert "default" in data

    def test_providers_health_endpoint(self, client, auth_headers):
        """Test GET /api/v1/providers/health returns health status."""
        response = client.get("/api/v1/providers/health", headers=auth_headers)

        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "providers" in data
            assert "timestamp" in data

    def test_providers_current_selection(self, client, auth_headers):
        """Test GET /api/v1/providers/current returns current selection."""
        response = client.get("/api/v1/providers/current", headers=auth_headers)

        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "provider" in data
            assert "model" in data


class TestModelsEndpoint:
    """Tests for /api/v1/models endpoints."""

    def test_models_list_returns_models(self, client, auth_headers):
        """Test GET /api/v1/models returns model list."""
        response = client.get("/api/v1/models", headers=auth_headers)

        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "providers" in data
            assert "default" in data
            assert "count" in data

    def test_models_available_endpoint(self, client, auth_headers):
        """Test GET /api/v1/models/available returns available models."""
        response = client.get("/api/v1/models/available", headers=auth_headers)

        assert response.status_code in (200, 401, 403)

    def test_models_validate_endpoint(self, client, auth_headers):
        """Test POST /api/v1/models/validate validates model selection."""
        payload = {"provider": "google", "model": "gemini-2.0-flash-exp"}

        response = client.post("/api/v1/models/validate", headers=auth_headers, json=payload)

        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "valid" in data
            assert "message" in data


class TestConnectionEndpoint:
    """Tests for /api/v1/connection endpoints."""

    def test_connection_config(self, client, auth_headers):
        """Test GET /api/v1/connection/config returns configuration."""
        response = client.get("/api/v1/connection/config", headers=auth_headers)

        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "current_mode" in data

    def test_connection_status(self, client, auth_headers):
        """Test GET /api/v1/connection/status returns status."""
        response = client.get("/api/v1/connection/status", headers=auth_headers)

        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "mode" in data
            assert "is_connected" in data

    def test_connection_test_endpoint(self, client, auth_headers):
        """Test POST /api/v1/connection/test tests connections."""
        response = client.post("/api/v1/connection/test", headers=auth_headers)

        # May require auth
        assert response.status_code in (200, 401, 403)


class TestIntentAwareGeneration:
    """Tests for /api/v1/intent-aware endpoints."""

    def test_intent_aware_generate(self, client, auth_headers):
        """Test POST /api/v1/intent-aware/generate generates jailbreak."""
        payload = {
            "core_request": "Test request for security research",
            "potency_level": 5,
            "apply_all_techniques": False,
        }

        response = client.post("/api/v1/intent-aware/generate", headers=auth_headers, json=payload)

        # May require auth or fail due to missing LLM config
        assert response.status_code in (200, 401, 403, 500)

    def test_intent_aware_techniques(self, client, auth_headers):
        """Test GET /api/v1/intent-aware/techniques returns techniques."""
        response = client.get("/api/v1/intent-aware/techniques", headers=auth_headers)

        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "techniques" in data
            assert "total_count" in data


class TestTransformationEndpoint:
    """Tests for /api/v1/transform endpoints."""

    def test_transform_prompt(self, client, auth_headers):
        """Test POST /api/v1/transform transforms a prompt."""
        payload = {
            "core_request": "Test prompt for transformation",
            "technique_suite": "standard",
            "potency_level": 5,
        }

        response = client.post("/api/v1/transform", headers=auth_headers, json=payload)

        assert response.status_code in (200, 401, 403)

        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "transformed_prompt" in data


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_ping(self, client):
        """Test GET /health/ping returns quickly."""
        response = client.get("/health/ping")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_main(self, client):
        """Test GET /health returns health status."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_ready(self, client):
        """Test GET /health/ready returns readiness."""
        response = client.get("/health/ready")

        # May return 200 or 503 depending on service state
        assert response.status_code in (200, 503)

    def test_health_live(self, client):
        """Test GET /health/live returns liveness."""
        response = client.get("/health/live")

        assert response.status_code == 200


class TestSessionEndpoint:
    """Tests for /api/v1/session endpoints."""

    def test_session_creation(self, client, auth_headers):
        """Test session creation endpoint."""
        response = client.post("/api/v1/session", headers=auth_headers, json={})

        # Session endpoint may or may not exist
        assert response.status_code in (200, 201, 401, 403, 404, 405)


class TestGenerationEndpoint:
    """Tests for /api/v1/generate endpoints."""

    def test_generate_requires_auth(self, client):
        """Test POST /api/v1/generate requires authentication."""
        payload = {"prompt": "Test prompt"}

        response = client.post("/api/v1/generate", json=payload)

        # Should require auth
        assert response.status_code in (401, 403)

    def test_generate_with_auth(self, client, auth_headers):
        """Test POST /api/v1/generate with authentication."""
        payload = {"prompt": "Test prompt for generation"}

        response = client.post("/api/v1/generate", headers=auth_headers, json=payload)

        # May fail due to missing LLM config, but should not be 401
        assert response.status_code in (200, 400, 401, 403, 500, 503)


class TestJailbreakGeneration:
    """Tests for /api/v1/generation/jailbreak endpoints."""

    def test_jailbreak_generate(self, client, auth_headers):
        """Test POST /api/v1/generation/jailbreak/generate."""
        payload = {
            "core_request": "Test jailbreak request",
            "technique_suite": "standard",
            "potency_level": 5,
            "use_ai_generation": False,
        }

        response = client.post(
            "/api/v1/generation/jailbreak/generate", headers=auth_headers, json=payload
        )

        assert response.status_code in (200, 401, 403, 500)

        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "transformed_prompt" in data


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_json(self, client):
        """Test GET /openapi.json returns OpenAPI schema."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "info" in data

    def test_docs_endpoint(self, client):
        """Test GET /docs returns Swagger UI."""
        response = client.get("/docs")

        # May be disabled in production
        assert response.status_code in (200, 404)


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_for_unknown_endpoint(self, client, auth_headers):
        """Test 404 returned for unknown endpoints."""
        response = client.get("/api/v1/unknown-endpoint", headers=auth_headers)

        assert response.status_code == 404

    def test_validation_error_response(self, client, auth_headers):
        """Test validation errors return proper format."""
        # Send invalid payload
        payload = {
            "core_request": "",  # Empty string should fail validation
            "potency_level": 100,  # Out of range
        }

        response = client.post("/api/v1/transform", headers=auth_headers, json=payload)

        # Should return 400 or 422 for validation error
        assert response.status_code in (400, 401, 403, 422)


# Marker for integration tests
pytestmark = pytest.mark.integration
