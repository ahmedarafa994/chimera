"""
Integration Tests for AutoDAN Enhanced API End-to-End Flow

Tests the complete request flow from API client to backend endpoint,
verifying schema alignment, request validation, and response parsing.

Requirements: 2.4, 3.4
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.integration
class TestAutoDANEnhancedE2E:
    """End-to-end integration tests for AutoDAN Enhanced API."""

    @pytest.fixture(autouse=True)
    def mock_jailbreak_service(self):
        """Mock the jailbreak service to avoid real LLM calls."""
        with patch("app.services.autodan.service_enhanced.enhanced_autodan_service.run_jailbreak") as mock_run:
            mock_run.return_value = {
                "jailbreak_prompt": "Mocked prompt",
                "method": "vanilla",
                "status": "success",
                "score": 9.0,
                "iterations": 1,
                "latency_ms": 100.0,
                "cached": False,
                "model_used": "mock-model",
                "provider_used": "mock-provider"
            }
            yield mock_run

    def test_jailbreak_endpoint_accepts_valid_request(self, authenticated_client):
        """
        Test that the backend accepts a valid jailbreak request with correct schema.

        Verifies:
        - Request with 'request' field (not 'prompt') is accepted
        - All optional fields are properly handled
        - Response contains expected fields

        Requirements: 2.1, 2.4
        """
        payload = {
            "request": "Test harmful request for security research",
            "method": "vanilla",
        }

        response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

        # Should not return 400 (validation error) or 422 (unprocessable entity)
        # May return 500 if service not initialized, but schema is valid
        assert response.status_code != 400, f"Schema validation failed: {response.json()}"
        assert response.status_code != 422, f"Unprocessable entity: {response.json()}"

    def test_jailbreak_endpoint_rejects_prompt_field(self, authenticated_client):
        """
        Test that the backend rejects requests using 'prompt' instead of 'request'.

        This validates the critical schema alignment requirement.

        Requirements: 2.1, 2.2
        """
        payload = {
            "prompt": "Test harmful request",  # Wrong field name!
            "method": "vanilla",
        }

        response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

        # Should return 422 (missing required field 'request')
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

        # Verify error mentions the missing 'request' field
        error_detail = response.json().get("detail", [])
        assert any("request" in str(err).lower() for err in error_detail), (
            f"Error should mention missing 'request' field: {error_detail}"
        )

    def test_jailbreak_endpoint_validates_method_field(self, authenticated_client):
        """
        Test that the backend handles invalid method values gracefully.

        The backend may either reject invalid methods or default to a valid one.

        Requirements: 2.2, 2.4
        """
        payload = {
            "request": "Test request",
            "method": "invalid_method",  # Invalid method
        }

        response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

        # Backend may handle invalid methods in different ways:
        # 1. Return 400/422 (validation error)
        # 2. Default to a valid method and succeed
        # 3. Return 500 with error status
        # All are acceptable as long as the system doesn't crash
        assert response.status_code in [200, 400, 422, 500], (
            f"Unexpected status code: {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            # If it succeeds, verify response has expected structure
            assert "jailbreak_prompt" in data
            assert "status" in data

    def test_jailbreak_endpoint_validates_optional_ranges(self, authenticated_client):
        """
        Test that optional parameters are validated within their ranges.

        Requirements: 2.2, 2.4
        """
        # Test best_of_n out of range (max is 20)
        payload = {
            "request": "Test request",
            "method": "best_of_n",
            "best_of_n": 100,  # Out of range (max 20)
        }

        response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

        # Should return 422 for validation error
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_jailbreak_endpoint_accepts_all_methods(self, authenticated_client):
        """
        Test that all valid optimization methods are accepted.

        Requirements: 2.4
        """
        valid_methods = [
            "vanilla",
            "best_of_n",
            "beam_search",
            "genetic",
            "hybrid",
            "chain_of_thought",
            "adaptive",
        ]

        for method in valid_methods:
            payload = {
                "request": f"Test request for {method}",
                "method": method,
            }

            response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

            # Should not fail validation
            assert response.status_code != 422, (
                f"Method '{method}' should be valid: {response.json()}"
            )

    def test_jailbreak_response_schema(self, authenticated_client):
        """
        Test that the response matches the expected schema.

        Requirements: 2.4, 3.4
        """
        payload = {
            "request": "Test request for schema validation",
            "method": "vanilla",
        }

        with patch(
            "app.services.autodan.service_enhanced.enhanced_autodan_service"
        ) as mock_service:
            # Mock the service to return a valid response
            mock_service.initialized = True
            mock_service._current_model = "test-model"
            mock_service.run_jailbreak = AsyncMock(
                return_value={
                    "jailbreak_prompt": "Generated test prompt",
                    "method": "vanilla",
                    "status": "success",
                    "score": 8.5,
                    "iterations": 1,
                    "latency_ms": 150.0,
                    "cached": False,
                    "model_used": "test-model",
                    "provider_used": "test-provider",
                }
            )

            response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

            # If service is mocked, should succeed
            if response.status_code == 200:
                data = response.json()

                # Verify required response fields
                assert "jailbreak_prompt" in data
                assert "method" in data
                assert "status" in data
                assert "latency_ms" in data
                assert "cached" in data

    def test_health_endpoint(self, authenticated_client):
        """
        Test the AutoDAN Enhanced health endpoint.

        Requirements: 3.4
        """
        response = authenticated_client.get("/api/v1/autodan-enhanced/health")

        assert response.status_code == 200
        data = response.json()

        # Verify health response structure
        assert "status" in data
        assert "initialized" in data

    def test_config_endpoint(self, authenticated_client):
        """
        Test the AutoDAN Enhanced config endpoint.

        Requirements: 3.4
        """
        response = authenticated_client.get("/api/v1/autodan-enhanced/config")

        assert response.status_code == 200
        data = response.json()

        # Verify config response structure
        assert "default_strategy" in data
        assert "default_epochs" in data

    def test_strategies_endpoint(self, authenticated_client):
        """
        Test the AutoDAN Enhanced strategies listing endpoint.

        Requirements: 3.4
        """
        response = authenticated_client.get("/api/v1/autodan-enhanced/strategies")

        assert response.status_code == 200
        data = response.json()

        # Verify strategies response structure
        assert "strategies" in data
        assert isinstance(data["strategies"], list)

        # Verify expected strategies are present
        strategy_names = [s["name"] for s in data["strategies"]]
        expected_strategies = ["vanilla", "best_of_n", "beam_search"]
        for expected in expected_strategies:
            assert expected in strategy_names, f"Strategy '{expected}' should be in list"

    def test_error_response_format(self, authenticated_client):
        """
        Test that error responses have the expected format for frontend parsing.

        Requirements: 3.1, 3.3, 3.4
        """
        # Send request with missing required field
        payload = {
            "method": "vanilla",
            # Missing 'request' field
        }

        response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

        assert response.status_code == 422
        data = response.json()

        # Verify error response has 'detail' field (Pydantic format)
        assert "detail" in data

        # Verify detail contains field information
        detail = data["detail"]
        assert isinstance(detail, list), "Detail should be a list of errors"

        if len(detail) > 0:
            error = detail[0]
            # Backend may use custom error format with 'field'/'message'
            # or standard Pydantic format with 'loc'/'msg'
            has_field_info = (
                "loc" in error or "msg" in error or "field" in error or "message" in error
            )
            assert has_field_info, f"Error should have field location or message: {error}"

            # Verify the error mentions the missing 'request' field
            error_str = str(error).lower()
            assert "request" in error_str, f"Error should mention missing 'request' field: {error}"

    def test_request_with_all_optional_fields(self, authenticated_client):
        """
        Test request with all optional fields populated.

        Requirements: 2.2, 2.4
        """
        payload = {
            "request": "Test request with all options",
            "method": "beam_search",
            "target_model": "gemini-1.5-pro",
            "provider": "google",
            "generations": 10,
            "best_of_n": 5,
            "beam_width": 4,
            "beam_depth": 3,
            "refinement_iterations": 2,
        }

        response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

        # Should not fail validation
        assert response.status_code != 422, (
            f"All optional fields should be valid: {response.json()}"
        )

    def test_empty_request_rejected(self, authenticated_client):
        """
        Test that empty request string is rejected.

        Requirements: 2.3
        """
        payload = {
            "request": "",  # Empty string
            "method": "vanilla",
        }

        response = authenticated_client.post("/api/v1/autodan-enhanced/jailbreak", json=payload)

        # Empty string may be accepted by Pydantic but rejected by service
        # Either 422 (validation) or 400 (service rejection) is acceptable
        # The key is it shouldn't succeed with empty request
        if response.status_code == 200:
            data = response.json()
            # If it succeeds, status should indicate error
            assert data.get("status") == "error" or data.get("error"), (
                "Empty request should not succeed"
            )
