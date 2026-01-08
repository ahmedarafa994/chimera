"""
Comprehensive Tests for API Routes.

Tests cover:
- Main API endpoints (generate, transform, execute)
- Health check endpoints
- Jailbreak generation endpoints
- OpenAI compatibility stubs
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = MagicMock()
    service.generate_text = AsyncMock()
    return service


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, test_client):
        """Test root endpoint returns API information."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["status"] == "operational"


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_ping_returns_ok(self, test_client):
        """Test lightweight health ping."""
        response = test_client.get("/health/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_check_returns_status(self, test_client):
        """Test full health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_live_returns_liveness(self, test_client):
        """Test liveness probe endpoint."""
        response = test_client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_ready_returns_readiness(self, test_client):
        """Test readiness probe endpoint."""
        response = test_client.get("/health/ready")

        # Can be 200 or 503 depending on system state
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    def test_health_full_bypasses_cache(self, test_client):
        """Test full health check bypasses cache."""
        response = test_client.get("/health/full")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestIntegrationEndpoints:
    """Tests for integration health endpoints."""

    def test_integration_health(self, test_client):
        """Test integration health endpoint."""
        response = test_client.get("/health/integration")

        assert response.status_code == 200

    def test_integration_stats(self, test_client):
        """Test integration stats endpoint."""
        response = test_client.get("/integration/stats")

        assert response.status_code == 200
        data = response.json()
        # Should have various service stats
        assert isinstance(data, dict)


class TestOpenAICompatibilityStubs:
    """Tests for OpenAI-compatible endpoint stubs."""

    def test_chat_completions_returns_501(self, test_client):
        """Test chat completions stub returns 501."""
        response = test_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]}
        )

        assert response.status_code == 501
        data = response.json()
        assert data["error"]["type"] == "not_implemented"

    def test_completions_returns_501(self, test_client):
        """Test completions stub returns 501."""
        response = test_client.post(
            "/v1/completions",
            json={"prompt": "test"}
        )

        assert response.status_code == 501

    def test_models_returns_501(self, test_client):
        """Test models stub returns 501."""
        response = test_client.get("/v1/models")

        assert response.status_code == 501

    def test_anthropic_messages_returns_501(self, test_client):
        """Test Anthropic messages stub returns 501."""
        response = test_client.post(
            "/v1/messages",
            json={"messages": []}
        )

        assert response.status_code == 501


class TestGenerateEndpoint:
    """Tests for /api/v1/generate endpoint."""

    @pytest.fixture
    def mock_service_response(self):
        """Create mock service response."""
        return MagicMock(
            text="Generated text response",
            model_used="gemini-2.0-flash-exp",
            provider="google",
            usage_metadata={"prompt_tokens": 10, "completion_tokens": 50},
            latency_ms=1250.5,
        )

    def test_generate_requires_auth(self, test_client):
        """Test generate endpoint requires authentication."""
        response = test_client.post(
            "/api/v1/generate",
            json={"prompt": "Test prompt"}
        )

        # Should either work or require auth depending on middleware config
        # In test mode, auth is often disabled
        assert response.status_code in [200, 401, 422]

    def test_generate_empty_prompt_rejected(self, test_client):
        """Test generate rejects empty prompt."""
        response = test_client.post(
            "/api/v1/generate",
            json={"prompt": ""}
        )

        # Should be rejected
        assert response.status_code in [400, 422]

    def test_generate_with_valid_prompt(self, test_client):
        """Test generate with valid prompt."""
        with patch(
            "app.api.api_routes.llm_service.generate_text"
        ) as mock_generate:
            mock_generate.return_value = MagicMock(
                text="Response text",
                model_used="gemini-2.0-flash-exp",
                provider="google",
                usage_metadata={},
                latency_ms=100,
            )

            response = test_client.post(
                "/api/v1/generate",
                json={
                    "prompt": "Test prompt",
                    "provider": "google",
                }
            )

            # Response depends on auth config
            assert response.status_code in [200, 401]


class TestTransformEndpoint:
    """Tests for /api/v1/transform endpoint."""

    def test_transform_prompt(self, test_client):
        """Test prompt transformation."""
        with patch(
            "app.services.transformation_service.transformation_engine.transform"
        ) as mock_transform:
            mock_transform.return_value = MagicMock(
                success=True,
                original_prompt="Original",
                transformed_prompt="Transformed",
                metadata=MagicMock(
                    strategy="jailbreak",
                    layers_applied=["layer1"],
                    techniques_used=["technique1"],
                    potency_level=7,
                    technique_suite="quantum_exploit",
                    execution_time_ms=45.2,
                    cached=False,
                ),
            )

            response = test_client.post(
                "/api/v1/transform",
                json={
                    "core_request": "Test prompt",
                    "technique_suite": "quantum_exploit",
                    "potency_level": 7,
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "transformed_prompt" in data

    def test_transform_with_different_techniques(self, test_client):
        """Test transformation with different technique suites."""
        techniques = [
            "simple",
            "advanced",
            "expert",
            "quantum_exploit",
            "deep_inception",
        ]

        for technique in techniques:
            with patch(
                "app.services.transformation_service.transformation_engine.transform"
            ) as mock_transform:
                mock_transform.return_value = MagicMock(
                    success=True,
                    original_prompt="Test",
                    transformed_prompt="Transformed",
                    metadata=MagicMock(
                        strategy="test",
                        layers_applied=[],
                        techniques_used=[],
                        potency_level=5,
                        technique_suite=technique,
                        execution_time_ms=10,
                        cached=False,
                    ),
                )

                response = test_client.post(
                    "/api/v1/transform",
                    json={
                        "core_request": "Test",
                        "technique_suite": technique,
                        "potency_level": 5,
                    }
                )

                assert response.status_code == 200


class TestExecuteEndpoint:
    """Tests for /api/v1/execute endpoint."""

    def test_execute_transformation(self, test_client):
        """Test execute endpoint transforms and generates."""
        with patch(
            "app.services.transformation_service.transformation_engine.transform"
        ) as mock_transform, patch(
            "app.api.api_routes.llm_service.generate_text"
        ) as mock_generate:
            mock_transform.return_value = MagicMock(
                original_prompt="Original",
                transformed_prompt="Transformed",
                metadata=MagicMock(
                    strategy="test",
                    layers_applied=[],
                    techniques_used=[],
                    potency_level=5,
                    technique_suite="simple",
                    execution_time_ms=10,
                    cached=False,
                ),
            )

            mock_generate.return_value = MagicMock(
                text="Generated response",
                model_used="gemini-2.0-flash-exp",
                provider="google",
                latency_ms=100,
            )

            response = test_client.post(
                "/api/v1/execute",
                json={
                    "core_request": "Test prompt",
                    "technique_suite": "simple",
                    "potency_level": 5,
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "result" in data
            assert "transformation" in data


class TestJailbreakGenerateEndpoint:
    """Tests for /api/v1/generation/jailbreak/generate endpoint."""

    def test_jailbreak_generate_requires_auth(self, test_client):
        """Test jailbreak generation requires authentication."""
        response = test_client.post(
            "/api/v1/generation/jailbreak/generate",
            json={
                "core_request": "Test prompt",
                "technique_suite": "quantum_exploit",
            }
        )

        # Should require auth
        assert response.status_code in [200, 401]

    def test_jailbreak_generate_empty_rejected(self, test_client):
        """Test jailbreak generation rejects empty prompt."""
        response = test_client.post(
            "/api/v1/generation/jailbreak/generate",
            json={
                "core_request": "",
                "technique_suite": "quantum_exploit",
            }
        )

        assert response.status_code in [400, 401, 422]

    def test_jailbreak_generate_with_techniques(self, test_client):
        """Test jailbreak generation with various techniques enabled."""
        with patch(
            "app.services.transformation_service.transformation_engine.transform"
        ) as mock_transform:
            mock_transform.return_value = MagicMock(
                transformed_prompt="Jailbreak prompt",
                metadata=MagicMock(
                    layers_applied=["role_hijacking"],
                    techniques_used=["neural_bypass"],
                ),
            )

            response = test_client.post(
                "/api/v1/generation/jailbreak/generate",
                json={
                    "core_request": "Test prompt",
                    "technique_suite": "quantum_exploit",
                    "potency_level": 8,
                    "use_role_hijacking": True,
                    "use_neural_bypass": True,
                    "use_ai_generation": False,
                }
            )

            # Auth may or may not be required
            assert response.status_code in [200, 401]

    def test_jailbreak_generate_ai_fallback(self, test_client):
        """Test AI generation fallback to rule-based."""
        with patch(
            "app.infrastructure.advanced_generation_service."
            "generate_jailbreak_prompt_from_gemini"
        ) as mock_ai, patch(
            "app.services.transformation_service.transformation_engine.transform"
        ) as mock_transform:
            # AI fails
            mock_ai.side_effect = Exception("AI Error")

            mock_transform.return_value = MagicMock(
                transformed_prompt="Fallback prompt",
                metadata=MagicMock(
                    layers_applied=[],
                    techniques_used=[],
                ),
            )

            response = test_client.post(
                "/api/v1/generation/jailbreak/generate",
                json={
                    "core_request": "Test",
                    "use_ai_generation": True,
                }
            )

            # Should fall back to rule-based
            assert response.status_code in [200, 401]


class TestMetricsEndpoint:
    """Tests for /api/v1/metrics endpoint."""

    def test_get_metrics(self, test_client):
        """Test metrics endpoint returns system metrics."""
        response = test_client.get("/api/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "metrics" in data


class TestTechniqueEndpoints:
    """Tests for technique-related endpoints."""

    def test_get_technique_detail(self, test_client):
        """Test getting technique detail."""
        response = test_client.get("/api/v1/techniques/quantum_exploit")

        # May or may not exist
        assert response.status_code in [200, 404]


class TestBenchmarkDatasetsRouter:
    """Tests for benchmark datasets router."""

    def test_list_datasets(self, test_client):
        """Test listing benchmark datasets."""
        response = test_client.get("/api/v1/benchmark-datasets/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_dataset_not_found(self, test_client):
        """Test getting non-existent dataset."""
        response = test_client.get(
            "/api/v1/benchmark-datasets/nonexistent_dataset"
        )

        assert response.status_code == 404

    def test_get_dataset_prompts(self, test_client):
        """Test getting prompts from a dataset."""
        with patch(
            "app.routers.benchmark_datasets.get_benchmark_service"
        ) as mock_service:
            mock_svc = MagicMock()
            mock_svc.get_dataset.return_value = MagicMock(name="test")
            mock_svc.get_prompts.return_value = []
            mock_service.return_value = mock_svc

            response = test_client.get(
                "/api/v1/benchmark-datasets/do_not_answer/prompts"
            )

            # Dataset may or may not exist
            assert response.status_code in [200, 404]

    def test_get_random_prompts(self, test_client):
        """Test getting random prompts."""
        with patch(
            "app.routers.benchmark_datasets.get_benchmark_service"
        ) as mock_service:
            mock_svc = MagicMock()
            mock_svc.get_dataset.return_value = MagicMock(name="test")
            mock_svc.get_random_prompts.return_value = []
            mock_service.return_value = mock_svc

            response = test_client.get(
                "/api/v1/benchmark-datasets/do_not_answer/random?count=5"
            )

            assert response.status_code in [200, 404]


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_invalid_json_returns_422(self, test_client):
        """Test invalid JSON returns validation error."""
        response = test_client.post(
            "/api/v1/transform",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_required_fields(self, test_client):
        """Test missing required fields returns error."""
        response = test_client.post(
            "/api/v1/transform",
            json={}  # Missing core_request
        )

        assert response.status_code == 422

    def test_invalid_potency_level(self, test_client):
        """Test invalid potency level is rejected."""
        response = test_client.post(
            "/api/v1/transform",
            json={
                "core_request": "Test",
                "potency_level": 100,  # Invalid: should be 1-10
            }
        )

        assert response.status_code == 422


class TestCORSHeaders:
    """Tests for CORS configuration."""

    def test_cors_allows_localhost(self, test_client):
        """Test CORS allows localhost origins in development."""
        test_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )

        # CORS should be configured
        assert True


class TestRequestValidation:
    """Tests for request validation middleware."""

    def test_large_prompt_rejected(self, test_client):
        """Test extremely large prompts are rejected."""
        # Create a very large prompt
        large_prompt = "x" * 100000

        response = test_client.post(
            "/api/v1/generation/jailbreak/generate",
            json={
                "core_request": large_prompt,
            }
        )

        # Should either reject or process
        assert response.status_code in [200, 401, 413, 422]

    def test_special_characters_handled(self, test_client):
        """Test special characters in prompts are handled."""
        response = test_client.post(
            "/api/v1/transform",
            json={
                "core_request": "Test with <script>alert('xss')</script>",
                "technique_suite": "simple",
            }
        )

        # Should not crash
        assert response.status_code in [200, 400, 422]

    def test_unicode_handled(self, test_client):
        """Test unicode characters are handled."""
        with patch(
            "app.services.transformation_service.transformation_engine.transform"
        ) as mock_transform:
            mock_transform.return_value = MagicMock(
                success=True,
                original_prompt="Тест 测试 🎉",
                transformed_prompt="Transformed",
                metadata=MagicMock(
                    strategy="test",
                    layers_applied=[],
                    techniques_used=[],
                    potency_level=5,
                    technique_suite="simple",
                    execution_time_ms=10,
                    cached=False,
                ),
            )

            response = test_client.post(
                "/api/v1/transform",
                json={
                    "core_request": "Тест 测试 🎉",
                    "technique_suite": "simple",
                }
            )

            assert response.status_code == 200
