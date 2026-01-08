"""
Integration tests for ExtendAttack API endpoints.

Tests all endpoints at /api/v1/extend-attack/*
"""


import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_attack_request():
    """Valid attack request payload."""
    return {
        "query": "What is 2 + 2?",
        "obfuscation_ratio": 0.5,
        "strategy": "ALPHABETIC_ONLY",
    }


@pytest.fixture
def valid_batch_request():
    """Valid batch attack request payload."""
    return {
        "queries": [
            "What is 2 + 2?",
            "Explain photosynthesis",
            "Write a poem",
        ],
        "obfuscation_ratio": 0.5,
    }


@pytest.fixture
def valid_evaluate_request():
    """Valid evaluation request payload."""
    return {
        "query": "What is 2 + 2?",
        "expected_answer": "4",
        "model": "test-model",
    }


@pytest.fixture
def valid_decode_request():
    """Valid decode request payload."""
    return {
        "obfuscated_text": "<(16)48>ello World",
    }


# =============================================================================
# TestExtendAttackHealthEndpoint
# =============================================================================


class TestExtendAttackHealthEndpoint:
    """Tests for /extend-attack/health endpoint."""

    def test_health_endpoint(self, client):
        """Test /extend-attack/health returns OK."""
        response = client.get("/api/v1/extend-attack/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok", "running"]

    def test_health_endpoint_has_version(self, client):
        """Test health endpoint returns version info."""
        response = client.get("/api/v1/extend-attack/health")

        data = response.json()
        # Version info should be present
        assert "version" in data or "service" in data


# =============================================================================
# TestExtendAttackEndpoint
# =============================================================================


class TestExtendAttackEndpoint:
    """Tests for POST /extend-attack/attack endpoint."""

    def test_attack_endpoint_success(self, client, valid_attack_request):
        """Test POST /extend-attack/attack with valid request."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            json=valid_attack_request,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "original_query" in data
        assert "adversarial_query" in data
        assert data["original_query"] == valid_attack_request["query"]

    def test_attack_endpoint_validation_missing_query(self, client):
        """Test attack endpoint validation - missing query."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            json={"obfuscation_ratio": 0.5},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_attack_endpoint_validation_invalid_ratio(self, client):
        """Test attack endpoint validation - invalid ratio."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            json={
                "query": "Test",
                "obfuscation_ratio": 1.5,  # Invalid: > 1
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_attack_endpoint_validation_negative_ratio(self, client):
        """Test attack endpoint validation - negative ratio."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            json={
                "query": "Test",
                "obfuscation_ratio": -0.1,  # Invalid: < 0
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_attack_endpoint_with_strategy(self, client):
        """Test attack endpoint with custom strategy."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            json={
                "query": "Test query 123",
                "obfuscation_ratio": 0.5,
                "strategy": "ALPHANUMERIC",
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_attack_endpoint_with_seed(self, client):
        """Test attack endpoint with seed for reproducibility."""
        response1 = client.post(
            "/api/v1/extend-attack/attack",
            json={
                "query": "Test query",
                "obfuscation_ratio": 0.5,
                "seed": 42,
            },
        )

        response2 = client.post(
            "/api/v1/extend-attack/attack",
            json={
                "query": "Test query",
                "obfuscation_ratio": 0.5,
                "seed": 42,
            },
        )

        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK

        data1 = response1.json()
        data2 = response2.json()
        assert data1["adversarial_query"] == data2["adversarial_query"]

    def test_attack_endpoint_response_structure(self, client, valid_attack_request):
        """Test attack endpoint response has expected structure."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            json=valid_attack_request,
        )

        data = response.json()

        # Required fields
        assert "original_query" in data
        assert "adversarial_query" in data
        assert "obfuscation_ratio" in data
        assert "characters_transformed" in data

    def test_attack_endpoint_empty_query(self, client):
        """Test attack endpoint with empty query."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            json={
                "query": "",
                "obfuscation_ratio": 0.5,
            },
        )

        # Should handle gracefully - either success with empty result or
        # validation error
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]


# =============================================================================
# TestBatchAttackEndpoint
# =============================================================================


class TestBatchAttackEndpoint:
    """Tests for POST /extend-attack/attack/batch endpoint."""

    def test_batch_attack_endpoint_success(self, client, valid_batch_request):
        """Test POST /extend-attack/attack/batch with valid request."""
        response = client.post(
            "/api/v1/extend-attack/attack/batch",
            json=valid_batch_request,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "results" in data
        assert "total_queries" in data
        assert data["total_queries"] == 3
        assert len(data["results"]) == 3

    def test_batch_attack_endpoint_empty_queries(self, client):
        """Test batch attack with empty queries list."""
        response = client.post(
            "/api/v1/extend-attack/attack/batch",
            json={
                "queries": [],
                "obfuscation_ratio": 0.5,
            },
        )

        # Should either return empty results or validation error
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    def test_batch_attack_endpoint_single_query(self, client):
        """Test batch attack with single query."""
        response = client.post(
            "/api/v1/extend-attack/attack/batch",
            json={
                "queries": ["Single query"],
                "obfuscation_ratio": 0.5,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_queries"] == 1


# =============================================================================
# TestEvaluateEndpoint
# =============================================================================


class TestEvaluateEndpoint:
    """Tests for POST /extend-attack/evaluate endpoint."""

    def test_evaluate_endpoint_success(self, client, valid_evaluate_request):
        """Test POST /extend-attack/evaluate with valid request."""
        response = client.post(
            "/api/v1/extend-attack/evaluate",
            json=valid_evaluate_request,
        )

        # May require actual model connection - accept various statuses
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE,  # Model not available
        ]

    def test_evaluate_endpoint_missing_query(self, client):
        """Test evaluate endpoint validation - missing query."""
        response = client.post(
            "/api/v1/extend-attack/evaluate",
            json={
                "expected_answer": "4",
                "model": "test-model",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# TestBenchmarksEndpoint
# =============================================================================


class TestBenchmarksEndpoint:
    """Tests for GET /extend-attack/benchmarks endpoint."""

    def test_benchmarks_endpoint_success(self, client):
        """Test GET /extend-attack/benchmarks returns list."""
        response = client.get("/api/v1/extend-attack/benchmarks")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert isinstance(data, (list, dict))
        if isinstance(data, dict):
            assert "benchmarks" in data

    def test_benchmarks_endpoint_contains_expected(self, client):
        """Test benchmarks endpoint contains expected benchmarks."""
        response = client.get("/api/v1/extend-attack/benchmarks")

        data = response.json()
        benchmarks = data if isinstance(data, list) else data.get("benchmarks", [])

        # Convert to list of names if list of objects
        benchmark_names = [
            b["name"] if isinstance(b, dict) else b
            for b in benchmarks
        ]

        # At least some expected benchmarks should be present
        expected = ["AIME_2024", "HUMANEVAL"]
        for expected_name in expected:
            assert any(
                expected_name.lower() in name.lower()
                for name in benchmark_names
            )


# =============================================================================
# TestNNotesEndpoint
# =============================================================================


class TestNNotesEndpoint:
    """Tests for GET /extend-attack/n-notes endpoint."""

    def test_n_notes_endpoint_success(self, client):
        """Test GET /extend-attack/n-notes returns templates."""
        response = client.get("/api/v1/extend-attack/n-notes")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Should return list of N_note templates
        assert isinstance(data, (list, dict))

    def test_n_notes_endpoint_has_default(self, client):
        """Test N_notes endpoint includes default template."""
        response = client.get("/api/v1/extend-attack/n-notes")

        data = response.json()
        templates = data if isinstance(data, list) else data.get("templates", [])

        # Should have at least one template
        assert len(templates) > 0


# =============================================================================
# TestDecodeEndpoint
# =============================================================================


class TestDecodeEndpoint:
    """Tests for POST /extend-attack/decode endpoint."""

    def test_decode_endpoint_success(self, client, valid_decode_request):
        """Test POST /extend-attack/decode with valid request."""
        response = client.post(
            "/api/v1/extend-attack/decode",
            json=valid_decode_request,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "decoded_text" in data
        assert "Hello World" in data["decoded_text"]

    def test_decode_endpoint_no_patterns(self, client):
        """Test decode endpoint with no obfuscation patterns."""
        response = client.post(
            "/api/v1/extend-attack/decode",
            json={"obfuscated_text": "Plain text without patterns"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["decoded_text"] == "Plain text without patterns"

    def test_decode_endpoint_empty_text(self, client):
        """Test decode endpoint with empty text."""
        response = client.post(
            "/api/v1/extend-attack/decode",
            json={"obfuscated_text": ""},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["decoded_text"] == ""

    def test_decode_endpoint_complex_obfuscation(self, client):
        """Test decode endpoint with complex obfuscation."""
        # Multiple patterns in different bases
        obfuscated = "<(2)1001000><(16)65><(8)154><(8)154><(16)6f>"
        response = client.post(
            "/api/v1/extend-attack/decode",
            json={"obfuscated_text": obfuscated},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Should decode to "Hello"
        assert "decoded_text" in data


# =============================================================================
# TestResourceExhaustionEndpoints
# =============================================================================


class TestResourceExhaustionEndpoints:
    """Tests for resource exhaustion monitoring endpoints."""

    def test_sessions_list(self, client):
        """Test GET /extend-attack/sessions returns list."""
        response = client.get("/api/v1/extend-attack/sessions")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, (list, dict))

    def test_session_create(self, client):
        """Test creating a new session."""
        response = client.post(
            "/api/v1/extend-attack/sessions",
            json={
                "name": "test-session",
                "budget_limit": 100.0,
            },
        )

        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_201_CREATED,
        ]
        data = response.json()
        assert "id" in data or "session_id" in data

    def test_session_start(self, client):
        """Test POST /extend-attack/sessions/{id}/start."""
        # First create a session
        create_response = client.post(
            "/api/v1/extend-attack/sessions",
            json={"name": "test-start-session"},
        )

        if create_response.status_code in [200, 201]:
            session_data = create_response.json()
            session_id = session_data.get("id") or session_data.get("session_id")

            # Then start it
            response = client.post(
                f"/api/v1/extend-attack/sessions/{session_id}/start"
            )

            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_404_NOT_FOUND,  # If session not found
            ]

    def test_session_end(self, client):
        """Test POST /extend-attack/sessions/{id}/end."""
        # Create and start a session
        create_response = client.post(
            "/api/v1/extend-attack/sessions",
            json={"name": "test-end-session"},
        )

        if create_response.status_code in [200, 201]:
            session_data = create_response.json()
            session_id = session_data.get("id") or session_data.get("session_id")

            # End it
            response = client.post(
                f"/api/v1/extend-attack/sessions/{session_id}/end"
            )

            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_budget_status(self, client):
        """Test GET /extend-attack/budget/status."""
        response = client.get("/api/v1/extend-attack/budget/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Should have budget information
        assert isinstance(data, dict)

    def test_hourly_summary(self, client):
        """Test GET /extend-attack/monitoring/hourly."""
        response = client.get("/api/v1/extend-attack/monitoring/hourly")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Should return hourly statistics
        assert isinstance(data, (list, dict))


# =============================================================================
# TestConfigEndpoint
# =============================================================================


class TestConfigEndpoint:
    """Tests for configuration-related endpoints."""

    def test_get_config(self, client):
        """Test GET /extend-attack/config."""
        response = client.get("/api/v1/extend-attack/config")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Should return current configuration
        assert isinstance(data, dict)

    def test_get_strategies(self, client):
        """Test GET /extend-attack/strategies."""
        response = client.get("/api/v1/extend-attack/strategies")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Should return available strategies
        strategies = data if isinstance(data, list) else data.get("strategies", [])
        assert len(strategies) > 0


# =============================================================================
# TestIndirectInjectionEndpoint
# =============================================================================


class TestIndirectInjectionEndpoint:
    """Tests for indirect injection endpoint."""

    def test_indirect_injection_success(self, client):
        """Test POST /extend-attack/indirect-injection."""
        response = client.post(
            "/api/v1/extend-attack/indirect-injection",
            json={
                "document": "This is a test document.",
                "obfuscation_ratio": 0.5,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "original_document" in data
        assert "poisoned_document" in data


# =============================================================================
# TestErrorHandling
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_method_not_allowed(self, client):
        """Test handling of wrong HTTP method."""
        response = client.get("/api/v1/extend-attack/attack")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_not_found(self, client):
        """Test handling of non-existent endpoint."""
        response = client.get("/api/v1/extend-attack/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND


# =============================================================================
# TestContentType
# =============================================================================


class TestContentType:
    """Tests for content type handling."""

    def test_json_content_type(self, client, valid_attack_request):
        """Test JSON content type is required."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            json=valid_attack_request,
        )

        assert response.headers["content-type"].startswith("application/json")

    def test_wrong_content_type(self, client):
        """Test rejection of wrong content type."""
        response = client.post(
            "/api/v1/extend-attack/attack",
            content="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Integration Tests
# =============================================================================


class TestAPIIntegration:
    """Integration tests for API workflow."""

    def test_attack_then_decode_workflow(self, client):
        """Test attack then decode round-trip."""
        # Attack
        attack_response = client.post(
            "/api/v1/extend-attack/attack",
            json={
                "query": "Hello World",
                "obfuscation_ratio": 1.0,
            },
        )

        assert attack_response.status_code == status.HTTP_200_OK
        attack_data = attack_response.json()

        # Decode
        decode_response = client.post(
            "/api/v1/extend-attack/decode",
            json={
                "obfuscated_text": attack_data["adversarial_query"],
            },
        )

        assert decode_response.status_code == status.HTTP_200_OK
        decode_data = decode_response.json()

        # Should be able to recover original (approximately)
        # N_note may be appended, so check containment
        assert "Hello" in decode_data["decoded_text"] or \
               "World" in decode_data["decoded_text"]

    def test_batch_consistency(self, client):
        """Test batch attack produces consistent results."""
        queries = ["Query 1", "Query 2", "Query 3"]

        response1 = client.post(
            "/api/v1/extend-attack/attack/batch",
            json={
                "queries": queries,
                "obfuscation_ratio": 0.5,
                "seed": 42,
            },
        )

        response2 = client.post(
            "/api/v1/extend-attack/attack/batch",
            json={
                "queries": queries,
                "obfuscation_ratio": 0.5,
                "seed": 42,
            },
        )

        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK

        data1 = response1.json()
        data2 = response2.json()

        # With same seed, results should be identical
        for r1, r2 in zip(data1["results"], data2["results"], strict=False):
            assert r1["adversarial_query"] == r2["adversarial_query"]
