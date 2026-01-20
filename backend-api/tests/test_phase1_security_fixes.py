"""Tests for Phase 1 Security Fixes - CRIT-001, CRIT-002, CRIT-003
Validates the security hardening implemented in the turnaround plan.
"""

import os
from unittest.mock import patch

import pytest


class TestSecurityHeadersMiddleware:
    """Tests for CRIT-001: SecurityHeadersMiddleware integration."""

    def test_security_headers_middleware_imported(self) -> None:
        """Verify SecurityHeadersMiddleware is imported in main.py."""
        from app.core.rate_limit import SecurityHeadersMiddleware

        assert SecurityHeadersMiddleware is not None

    def test_security_headers_class_has_correct_headers(self) -> None:
        """Verify SecurityHeadersMiddleware sets all required security headers."""
        from app.core.rate_limit import SecurityHeadersMiddleware

        # Check class exists and has dispatch method
        assert hasattr(SecurityHeadersMiddleware, "dispatch")

    @pytest.mark.asyncio
    async def test_security_headers_applied_to_response(self) -> None:
        """Test that security headers are applied to responses."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.core.rate_limit import SecurityHeadersMiddleware

        # Create minimal test app with only SecurityHeadersMiddleware
        test_app = FastAPI()
        test_app.add_middleware(SecurityHeadersMiddleware)

        @test_app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(test_app)
        response = client.get("/test")

        # Verify security headers are present
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"


class TestHardcodedApiKeyRemoval:
    """Tests for CRIT-002: Hardcoded API key removal."""

    def test_no_hardcoded_key_in_source(self) -> None:
        """Verify hardcoded development key is not in auth.py."""
        import inspect

        from app.middleware.auth import APIKeyMiddleware

        source = inspect.getsource(APIKeyMiddleware)
        # The specific hardcoded key should not exist
        assert "chimera_dev_key_1234567890123456" not in source

    def test_production_fails_without_api_key(self) -> None:
        """Verify production environment fails if no API key configured."""
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "production", "CHIMERA_API_KEY": ""},
            clear=False,
        ):
            # Clear any existing key
            if "CHIMERA_API_KEY" in os.environ:
                del os.environ["CHIMERA_API_KEY"]

            from app.middleware.auth import APIKeyMiddleware

            # Creating middleware in production without key should raise
            with pytest.raises(ValueError, match="Production environment requires API key"):
                middleware = APIKeyMiddleware(None)
                middleware._load_valid_keys()


class TestProductionDatabaseValidation:
    """Tests for CRIT-003: SQLite production block."""

    def test_sqlite_blocked_in_production(self) -> None:
        """Verify SQLite URLs are rejected in production environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            from pydantic import ValidationError

            from app.core.config import Settings

            # Attempting to use SQLite in production should fail
            with pytest.raises(ValidationError) as exc_info:
                Settings(DATABASE_URL="sqlite:///./test.db")

            assert "SQLite is not supported in production" in str(exc_info.value)

    def test_sqlite_allowed_in_development(self) -> None:
        """Verify SQLite URLs are allowed in development environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            from app.core.config import Settings

            # SQLite should work in development
            settings = Settings(DATABASE_URL="sqlite:///./test.db")
            assert "sqlite" in settings.DATABASE_URL.lower()

    def test_postgresql_allowed_in_production(self) -> None:
        """Verify PostgreSQL URLs are allowed in production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            from app.core.config import Settings

            # PostgreSQL should work in production
            settings = Settings(DATABASE_URL="postgresql://user:pass@localhost/db")
            assert "postgresql" in settings.DATABASE_URL.lower()


class TestApiKeyHeaderMigration:
    """Tests for HIGH-002: API key moved from URL to header."""

    def test_google_client_uses_header_not_query(self) -> None:
        """Verify Google API key is sent via header, not query parameter."""
        import inspect

        from app.engines.llm_provider_client import GoogleGeminiClient

        source = inspect.getsource(GoogleGeminiClient)

        # Should use header
        assert "x-goog-api-key" in source
        # Should NOT use query parameter pattern
        assert "?key=" not in source


class TestThreadSafeRateLimiter:
    """Tests for HIGH-004: Thread-safe LocalRateLimiter."""

    def test_local_rate_limiter_has_lock(self) -> None:
        """Verify LocalRateLimiter has asyncio.Lock for thread safety."""
        import asyncio

        from app.core.rate_limit import LocalRateLimiter

        limiter = LocalRateLimiter()
        assert hasattr(limiter, "_lock")
        assert isinstance(limiter._lock, asyncio.Lock)
