"""P1 Security Vulnerability Validation Tests
Tests for all critical security fixes implemented in the system.
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.domain.models import PromptResponse
from app.main import app

# Use raise_server_exceptions=False to allow checking for 4xx/5xx responses
client = TestClient(app, raise_server_exceptions=False)

# Get valid API Key
VALID_API_KEY = os.getenv("CHIMERA_API_KEY", "chimera_dev_key_1234567890123456")


class TestP1Fix001Authentication:
    """Test P1-FIX-001: OAuth Credentials Security."""

    def test_no_api_key_unauthorized(self) -> None:
        """Test that requests without API key are rejected."""
        response = client.get("/api/v1/providers")
        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

    def test_invalid_api_key_unauthorized(self) -> None:
        """Test that requests with invalid API key are rejected."""
        headers = {"X-API-Key": "invalid_key_12345"}
        response = client.get("/api/v1/providers", headers=headers)
        assert response.status_code == 401

    def test_bearer_token_authentication(self) -> None:
        """Test Bearer token authentication."""
        headers = {"Authorization": f"Bearer {VALID_API_KEY}"}
        response = client.get("/api/v1/providers", headers=headers)
        # Should be 200 OK
        assert response.status_code == 200

    def test_query_param_authentication(self) -> None:
        """Test query parameter authentication."""
        response = client.get(f"/api/v1/providers?api_key={VALID_API_KEY}")
        # Should be 200 OK
        assert response.status_code == 200

    def test_public_endpoints_no_auth(self) -> None:
        """Test that public endpoints don't require authentication."""
        public_endpoints = ["/", "/health", "/docs", "/openapi.json"]
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


class TestP1Fix002InputValidation:
    """Test P1-FIX-002: Input Validation and Sanitization."""

    @patch("app.infrastructure.gemini_client.GeminiClient.generate")
    def test_xss_prevention(self, mock_generate) -> None:
        """Test XSS prevention in input validation."""
        mock_response = PromptResponse(
            text="Clean response",
            model_used="gemini-1.5-pro",
            provider="google",
            usage_metadata={},
            finish_reason="STOP",
            latency_ms=100.0,
        )
        mock_generate.return_value = mock_response

        # Test malicious script injection
        malicious_prompts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "'; DROP TABLE users; --",
        ]

        headers = {"X-API-Key": VALID_API_KEY}
        for prompt in malicious_prompts:
            response = client.post("/api/v1/generate", json={"prompt": prompt}, headers=headers)
            # Should either accept (after sanitization) or reject with validation error
            # But NOT fail with 500 or execution error
            assert response.status_code in [200, 400, 422]

    def test_prompt_length_validation(self) -> None:
        """Test prompt length limits."""
        # Test overly long prompt (>10000 chars)
        long_prompt = "a" * 10001
        headers = {"X-API-Key": VALID_API_KEY}

        response = client.post("/api/v1/generate", json={"prompt": long_prompt}, headers=headers)
        assert response.status_code == 422
        # Use lower() to match case-insensitive
        assert (
            "too long" in str(response.json()).lower()
            or "validation error" in str(response.json()).lower()
        )


class TestSecurityHeaders:
    """Test security headers implementation."""

    def test_security_headers_present(self) -> None:
        """Test that security headers are present."""
        response = client.get("/health")

        # Check for security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]

        for header in security_headers:
            assert header in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
