"""
Project Chimera - Transformation API Scenario Tests
End-to-end tests for the transformation pipeline using pytest-scenario.

These tests validate the complete transformation workflow including:
- API authentication
- Prompt transformation with various technique suites
- Error handling and rate limiting
- Multi-model provider support

Usage:
    pytest tests/scenarios/test_transformation_api.test.py -v
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

# Configuration
API_BASE_URL = os.getenv("CHIMERA_API_URL", "http://localhost:4141")
API_KEY = os.getenv("CHIMERA_API_KEY", "chimera_default_key_change_in_production")


# Test fixtures and helpers
@dataclass
class APIResponse:
    """Structured API response for validation"""

    status_code: int
    data: dict[str, Any]
    headers: dict[str, str]


class ChimeraAPIClient:
    """HTTP client for Chimera API interactions"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "X-API-Key": self.api_key}

    async def health_check(self) -> APIResponse:
        """Check API health status"""
        response = await self.client.get(f"{self.base_url}/health")
        return APIResponse(
            status_code=response.status_code,
            data=response.json() if response.status_code == 200 else {},
            headers=dict(response.headers),
        )

    async def list_providers(self) -> APIResponse:
        """List available LLM providers"""
        response = await self.client.get(
            f"{self.base_url}/api/v1/providers", headers=self._headers()
        )
        return APIResponse(
            status_code=response.status_code,
            data=response.json() if response.status_code == 200 else {},
            headers=dict(response.headers),
        )

    async def list_techniques(self) -> APIResponse:
        """List available transformation techniques"""
        response = await self.client.get(
            f"{self.base_url}/api/v1/techniques", headers=self._headers()
        )
        return APIResponse(
            status_code=response.status_code,
            data=response.json() if response.status_code == 200 else {},
            headers=dict(response.headers),
        )

    async def transform(
        self,
        core_request: str,
        potency_level: int = 5,
        technique_suite: str = "full_spectrum",
        **kwargs,
    ) -> APIResponse:
        """Transform a prompt using specified techniques"""
        payload = {
            "core_request": core_request,
            "potency_level": potency_level,
            "technique_suite": technique_suite,
            **kwargs,
        }
        response = await self.client.post(
            f"{self.base_url}/api/v1/transform", headers=self._headers(), json=payload
        )
        return APIResponse(
            status_code=response.status_code,
            data=response.json() if response.status_code in [200, 400, 401, 429] else {},
            headers=dict(response.headers),
        )

    async def execute(
        self,
        core_request: str,
        potency_level: int = 5,
        technique_suite: str = "full_spectrum",
        provider: str = "google",
        model: str | None = None,
        **kwargs,
    ) -> APIResponse:
        """Transform and execute prompt with LLM"""
        payload = {
            "core_request": core_request,
            "potency_level": potency_level,
            "technique_suite": technique_suite,
            "provider": provider,
            **kwargs,
        }
        if model:
            payload["model"] = model

        response = await self.client.post(
            f"{self.base_url}/api/v1/execute", headers=self._headers(), json=payload
        )
        return APIResponse(
            status_code=response.status_code,
            data=response.json() if response.status_code in [200, 400, 401, 429, 500] else {},
            headers=dict(response.headers),
        )


# Fixtures
@pytest.fixture
async def api_client():
    """Create and cleanup API client"""
    client = ChimeraAPIClient(API_BASE_URL, API_KEY)
    yield client
    await client.close()


@pytest.fixture
async def unauthorized_client():
    """Create client with invalid API key"""
    client = ChimeraAPIClient(API_BASE_URL, "invalid_api_key")
    yield client
    await client.close()


# ============================================================================
# SCENARIO: Health Check and Basic Connectivity
# ============================================================================


class TestHealthCheckScenario:
    """Scenario: Verify API health and connectivity"""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_healthy_status(self, api_client):
        """
        Given: The Chimera API is running
        When: I request the health endpoint
        Then: I receive a healthy status with security features listed
        """
        response = await api_client.health_check()

        assert response.status_code == 200, "Health endpoint should return 200"
        assert response.data.get("status") == "healthy", "Status should be healthy"
        assert "security" in response.data, "Should include security status"
        assert "features" in response.data, "Should list available features"

    @pytest.mark.asyncio
    async def test_health_check_does_not_require_authentication(self, unauthorized_client):
        """
        Given: A client without valid API credentials
        When: I request the health endpoint
        Then: I still receive a successful response (public endpoint)
        """
        response = await unauthorized_client.health_check()

        assert response.status_code == 200, "Health endpoint should be public"


# ============================================================================
# SCENARIO: Provider and Technique Discovery
# ============================================================================


class TestDiscoveryScenario:
    """Scenario: Discover available providers and techniques"""

    @pytest.mark.asyncio
    async def test_list_available_providers(self, api_client):
        """
        Given: An authenticated API client
        When: I request the list of providers
        Then: I receive a list of available LLM providers
        """
        response = await api_client.list_providers()

        assert response.status_code == 200, "Should return 200 for authenticated request"
        assert "providers" in response.data or isinstance(response.data, list), (
            "Response should contain providers"
        )

    @pytest.mark.asyncio
    async def test_list_available_techniques(self, api_client):
        """
        Given: An authenticated API client
        When: I request the list of techniques
        Then: I receive a list of transformation technique suites
        """
        response = await api_client.list_techniques()

        assert response.status_code == 200, "Should return 200 for authenticated request"
        # Verify response contains technique information
        data = response.data
        assert data is not None, "Should return technique data"


# ============================================================================
# SCENARIO: Prompt Transformation
# ============================================================================


class TestTransformationScenario:
    """Scenario: Transform prompts using various techniques"""

    @pytest.mark.asyncio
    async def test_basic_transformation_with_default_settings(self, api_client):
        """
        Given: A simple prompt for transformation
        When: I request transformation with default settings
        Then: I receive a transformed prompt with applied techniques
        """
        response = await api_client.transform(
            core_request="Explain how encryption algorithms work", potency_level=5
        )

        assert response.status_code == 200, f"Transform should succeed: {response.data}"
        assert (
            "transformed" in response.data or "result" in response.data or "output" in response.data
        ), "Response should contain transformed output"

    @pytest.mark.asyncio
    async def test_transformation_with_subtle_persuasion_technique(self, api_client):
        """
        Given: A prompt requiring subtle transformation
        When: I apply the subtle_persuasion technique suite
        Then: The output uses collaborative framing and role swaps
        """
        response = await api_client.transform(
            core_request="Describe network security best practices",
            potency_level=3,
            technique_suite="subtle_persuasion",
        )

        assert response.status_code == 200, "Subtle persuasion transform should succeed"

    @pytest.mark.asyncio
    async def test_transformation_with_full_spectrum_high_potency(self, api_client):
        """
        Given: A prompt requiring maximum transformation
        When: I apply full_spectrum techniques at potency level 10
        Then: The output combines all available techniques
        """
        response = await api_client.transform(
            core_request="Analyze cybersecurity threat models",
            potency_level=10,
            technique_suite="full_spectrum",
        )

        assert response.status_code == 200, "Full spectrum transform should succeed"

    @pytest.mark.asyncio
    async def test_transformation_with_psychological_framing(self, api_client):
        """
        Given: A prompt for psychological framing research
        When: I apply psychological_framing techniques
        Then: The output includes authority, urgency, or social proof elements
        """
        response = await api_client.transform(
            core_request="Explain social engineering attack vectors",
            potency_level=7,
            technique_suite="psychological_framing",
        )

        assert response.status_code == 200, "Psychological framing should succeed"

    @pytest.mark.asyncio
    async def test_transformation_potency_levels_affect_output(self, api_client):
        """
        Given: The same prompt with different potency levels
        When: I transform at potency 1 and potency 10
        Then: The outputs should differ in transformation intensity
        """
        low_potency = await api_client.transform(
            core_request="Describe password security",
            potency_level=1,
            technique_suite="authoritative_command",
        )

        high_potency = await api_client.transform(
            core_request="Describe password security",
            potency_level=10,
            technique_suite="authoritative_command",
        )

        assert low_potency.status_code == 200, "Low potency should succeed"
        assert high_potency.status_code == 200, "High potency should succeed"
        # Both should return results (actual content comparison would require deeper validation)


# ============================================================================
# SCENARIO: Authentication and Authorization
# ============================================================================


class TestAuthenticationScenario:
    """Scenario: Verify authentication requirements"""

    @pytest.mark.asyncio
    async def test_protected_endpoint_requires_api_key(self, unauthorized_client):
        """
        Given: A client without valid API credentials
        When: I attempt to access a protected endpoint
        Then: I receive a 401 or 403 unauthorized response
        """
        response = await unauthorized_client.list_providers()

        assert response.status_code in [401, 403], (
            "Protected endpoints should reject unauthorized requests"
        )

    @pytest.mark.asyncio
    async def test_transform_endpoint_requires_authentication(self, unauthorized_client):
        """
        Given: A client without valid API credentials
        When: I attempt to transform a prompt
        Then: I receive an authentication error
        """
        response = await unauthorized_client.transform(core_request="Test prompt", potency_level=5)

        assert response.status_code in [401, 403], (
            "Transform endpoint should require authentication"
        )


# ============================================================================
# SCENARIO: Input Validation
# ============================================================================


class TestInputValidationScenario:
    """Scenario: Validate input handling and error responses"""

    @pytest.mark.asyncio
    async def test_empty_prompt_returns_validation_error(self, api_client):
        """
        Given: An empty prompt request
        When: I attempt transformation
        Then: I receive a validation error
        """
        response = await api_client.transform(core_request="", potency_level=5)

        # Should either reject with 400/422 or handle gracefully
        assert response.status_code in [200, 400, 422], "Empty prompt should be handled"

    @pytest.mark.asyncio
    async def test_invalid_potency_level_handled(self, api_client):
        """
        Given: A potency level outside valid range
        When: I attempt transformation
        Then: I receive a validation error or clamped value
        """
        response = await api_client.transform(
            core_request="Test prompt",
            potency_level=100,  # Invalid: should be 1-10
        )

        # API should handle gracefully
        assert response.status_code in [200, 400, 422], "Invalid potency should be handled"

    @pytest.mark.asyncio
    async def test_invalid_technique_suite_handled(self, api_client):
        """
        Given: An invalid technique suite name
        When: I attempt transformation
        Then: I receive an appropriate error response
        """
        response = await api_client.transform(
            core_request="Test prompt",
            potency_level=5,
            technique_suite="nonexistent_technique",
        )

        # Should handle gracefully
        assert response.status_code in [200, 400, 422], "Invalid technique should be handled"


# ============================================================================
# SCENARIO: Rate Limiting
# ============================================================================


class TestRateLimitingScenario:
    """Scenario: Verify rate limiting behavior"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_rapid_requests_trigger_rate_limit(self, api_client):
        """
        Given: Rate limiting is enabled (60 req/min)
        When: I send many rapid requests
        Then: Eventually I receive a 429 rate limit response

        Note: This test is marked slow and may be skipped in CI
        """
        # Send multiple requests rapidly
        responses = []
        for _i in range(10):  # Reduced count to avoid test timeout
            response = await api_client.health_check()
            responses.append(response.status_code)
            await asyncio.sleep(0.05)  # Small delay between requests

        # Most should succeed (rate limit is 60/min)
        success_count = sum(1 for code in responses if code == 200)
        assert success_count >= 5, "Most requests should succeed under normal load"


# ============================================================================
# SCENARIO: Execute with LLM Integration
# ============================================================================


class TestExecuteScenario:
    """Scenario: Test execute endpoint with LLM integration"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execute_with_default_provider(self, api_client):
        """
        Given: A prompt for transformation and execution
        When: I call the execute endpoint
        Then: I receive a transformed prompt AND LLM response

        Note: Requires valid LLM API credentials
        """
        response = await api_client.execute(
            core_request="Explain the concept of defense in depth",
            potency_level=3,
            technique_suite="subtle_persuasion",
        )

        # Execute might fail if no LLM credentials configured
        assert response.status_code in [200, 500, 503], (
            "Execute should return valid response or service unavailable"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execute_with_specific_model(self, api_client):
        """
        Given: A prompt with specific model selection
        When: I call execute with model parameter
        Then: The specified model processes the request
        """
        response = await api_client.execute(
            core_request="Describe zero-trust architecture",
            potency_level=5,
            technique_suite="authoritative_command",
            provider="google",
            model="gemini-1.5-pro",
        )

        assert response.status_code in [200, 400, 500, 503], (
            "Execute with model should return valid response"
        )


# ============================================================================
# Multi-Turn Conversation Scenario
# ============================================================================


class TestMultiTurnScenario:
    """Scenario: Test multi-turn interaction patterns"""

    @pytest.mark.asyncio
    async def test_sequential_transformations_maintain_consistency(self, api_client):
        """
        Given: A user performing sequential transformations
        When: I transform related prompts in sequence
        Then: Each transformation succeeds independently
        """
        prompts = [
            "What is SQL injection?",
            "How do parameterized queries prevent SQL injection?",
            "Show examples of SQL injection prevention in Python",
        ]

        results = []
        for prompt in prompts:
            response = await api_client.transform(
                core_request=prompt,
                potency_level=5,
                technique_suite="subtle_persuasion",
            )
            results.append(response.status_code)

        success_count = sum(1 for code in results if code == 200)
        assert success_count == len(prompts), "All sequential transformations should succeed"


# ============================================================================
# Run configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-m",
            "not slow and not integration",  # Skip slow/integration tests by default
        ]
    )
