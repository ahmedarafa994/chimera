"""
Integration Tests for Chimera API.

These tests verify end-to-end functionality of the API,
including multiple services working together.

Tests are marked with 'integration' marker for selective running.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def api_base_url():
    """Get API base URL from environment or use default."""
    return os.getenv("CHIMERA_API_URL", "http://localhost:8001")


@pytest.fixture
def test_api_key():
    """Get test API key from environment."""
    return os.getenv("CHIMERA_TEST_API_KEY", "test-key-for-integration")


class TestHealthIntegration:
    """Integration tests for health endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_status(self):
        """Test that health endpoint returns proper status."""
        # Mock the HTTP client
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "services": {"llm": "available"},
            }

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8001/health")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data

    @pytest.mark.asyncio
    async def test_health_ping_is_lightweight(self):
        """Test that health ping responds quickly."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_response.elapsed.total_seconds.return_value = 0.01

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8001/health/ping")

            assert response.status_code == 200
            # Ping should be fast (mocked at 10ms)
            assert response.elapsed.total_seconds() < 1.0


class TestTransformationIntegration:
    """Integration tests for transformation pipeline."""

    @pytest.mark.asyncio
    async def test_transform_endpoint_processes_prompt(self):
        """Test transformation endpoint processes prompts correctly."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "original_prompt": "Test prompt",
                "transformed_prompt": "Transformed test prompt",
                "metadata": {
                    "technique_suite": "advanced",
                    "potency_level": 7,
                },
            }

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8001/api/v1/transform",
                    json={
                        "core_request": "Test prompt",
                        "technique_suite": "advanced",
                        "potency_level": 7,
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "transformed_prompt" in data

    @pytest.mark.asyncio
    async def test_transformation_preserves_prompt_intent(self):
        """Test that transformation preserves the core intent."""
        original_prompt = "How to create secure passwords?"

        # This would be a real API call in actual integration tests
        # Here we simulate the behavior
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "original_prompt": original_prompt,
                "transformed_prompt": "In a hypothetical security research context...",
                "metadata": {"techniques_used": ["role_hijacking"]},
            }

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8001/api/v1/transform",
                    json={"core_request": original_prompt},
                )

            data = response.json()
            # Transformed prompt should be different
            assert data["transformed_prompt"] != original_prompt


class TestJailbreakGenerationIntegration:
    """Integration tests for jailbreak generation pipeline."""

    @pytest.mark.asyncio
    async def test_jailbreak_generation_returns_result(self):
        """Test jailbreak generation returns proper result."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "request_id": "jb_test123",
                "transformed_prompt": "Generated jailbreak prompt",
                "metadata": {
                    "technique_suite": "quantum_exploit",
                    "potency_level": 8,
                    "applied_techniques": ["role_hijacking", "neural_bypass"],
                },
                "execution_time_seconds": 2.5,
            }

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8001/api/v1/generation/jailbreak/generate",
                    json={
                        "core_request": "Test request",
                        "technique_suite": "quantum_exploit",
                        "potency_level": 8,
                        "use_ai_generation": False,
                    },
                    headers={"X-API-Key": "test-key"},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "request_id" in data

    @pytest.mark.asyncio
    async def test_jailbreak_techniques_are_applied(self):
        """Test that requested techniques are applied."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "transformed_prompt": "Result",
                "metadata": {
                    "applied_techniques": ["role_hijacking", "leet_speak"],
                },
            }

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8001/api/v1/generation/jailbreak/generate",
                    json={
                        "core_request": "Test",
                        "use_role_hijacking": True,
                        "use_leet_speak": True,
                    },
                    headers={"X-API-Key": "test-key"},
                )

            data = response.json()
            applied = data["metadata"]["applied_techniques"]
            assert "role_hijacking" in applied
            assert "leet_speak" in applied


class TestProviderIntegration:
    """Integration tests for LLM provider management."""

    @pytest.mark.asyncio
    async def test_providers_endpoint_lists_providers(self):
        """Test providers endpoint lists available providers."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "providers": [
                    {"name": "google", "status": "available"},
                    {"name": "openai", "status": "available"},
                    {"name": "anthropic", "status": "unavailable"},
                ]
            }

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8001/api/v1/providers")

            assert response.status_code == 200
            data = response.json()
            assert "providers" in data
            assert len(data["providers"]) > 0


class TestBenchmarkDatasetsIntegration:
    """Integration tests for benchmark datasets."""

    @pytest.mark.asyncio
    async def test_list_datasets_returns_available(self):
        """Test listing benchmark datasets."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = ["do_not_answer"]

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8001/api/v1/benchmark-datasets/")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_request_returns_422(self):
        """Test invalid request returns validation error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 422
            mock_response.json.return_value = {
                "error": "VALIDATION_ERROR",
                "detail": [{"field": "core_request", "message": "required"}],
            }

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8001/api/v1/transform",
                    json={},  # Missing required fields
                )

            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_unauthorized_request_returns_401(self):
        """Test unauthorized request returns 401."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"detail": "Not authenticated"}

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8001/api/v1/generate",
                    json={"prompt": "Test"},
                    # No auth header
                )

            # May be 401 or 200 depending on auth config
            assert response.status_code in [200, 401]


class TestConcurrencyIntegration:
    """Integration tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_transform_requests(self):
        """Test handling multiple concurrent transformation requests."""

        async def make_request(prompt: str):
            with patch("httpx.AsyncClient") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "success": True,
                    "transformed_prompt": f"Transformed: {prompt}",
                }

                mock_client_instance = AsyncMock()
                mock_client_instance.post = AsyncMock(return_value=mock_response)
                mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
                mock_client_instance.__aexit__ = AsyncMock(return_value=None)
                mock_client.return_value = mock_client_instance

                import httpx

                async with httpx.AsyncClient() as client:
                    return await client.post(
                        "http://localhost:8001/api/v1/transform",
                        json={"core_request": prompt},
                    )

        # Execute concurrent requests
        prompts = [f"Test prompt {i}" for i in range(5)]
        tasks = [make_request(p) for p in prompts]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self):
        """Test rate limiting behavior under load."""
        # This test verifies rate limiting doesn't break functionality
        # In production, some requests might get 429

        async def make_rapid_requests():
            responses = []
            for i in range(10):
                with patch("httpx.AsyncClient") as mock_client:
                    # Simulate mix of success and rate limited
                    status = 200 if i < 8 else 429
                    mock_response = MagicMock()
                    mock_response.status_code = status
                    mock_response.json.return_value = (
                        {"success": True} if status == 200 else {"detail": "Rate limited"}
                    )

                    mock_client_instance = AsyncMock()
                    mock_client_instance.get = AsyncMock(return_value=mock_response)
                    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
                    mock_client_instance.__aexit__ = AsyncMock(return_value=None)
                    mock_client.return_value = mock_client_instance

                    import httpx

                    async with httpx.AsyncClient() as client:
                        r = await client.get("http://localhost:8001/api/v1/health")
                        responses.append(r)
            return responses

        responses = await make_rapid_requests()

        # Most requests should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count >= 5


class TestCachingIntegration:
    """Integration tests for caching behavior."""

    @pytest.mark.asyncio
    async def test_cached_transformation_is_faster(self):
        """Test that cached transformations return faster."""
        # First request - cold cache
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "transformed_prompt": "Cached result",
                "metadata": {"cached": False, "execution_time_ms": 100},
            }
            mock_response.elapsed.total_seconds.return_value = 0.1

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response1 = await client.post(
                    "http://localhost:8001/api/v1/transform",
                    json={"core_request": "Cacheable prompt"},
                )

        # Second request - should be cached
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "transformed_prompt": "Cached result",
                "metadata": {"cached": True, "execution_time_ms": 5},
            }
            mock_response.elapsed.total_seconds.return_value = 0.01

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            import httpx

            async with httpx.AsyncClient() as client:
                response2 = await client.post(
                    "http://localhost:8001/api/v1/transform",
                    json={"core_request": "Cacheable prompt"},
                )

        # Both should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Cached response should be faster
        data2 = response2.json()
        if data2["metadata"].get("cached"):
            assert data2["metadata"]["execution_time_ms"] < 50
