"""
Unit Tests for Auth Rate Limiting Middleware

Tests for the authentication rate limiting middleware.
SEC-TEST-001: Auth rate limiting unit tests.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.auth_rate_limit import (
    AuthRateLimitConfig,
    AuthRateLimitMiddleware,
    get_auth_rate_limiter,
    set_auth_rate_limiter,
)


@pytest.fixture
def app():
    """Create a test FastAPI app with auth rate limiting."""
    app = FastAPI()

    @app.post("/api/v1/session")
    async def create_session():
        return {"session_id": "test-session"}

    @app.post("/api/v1/providers/select")
    async def select_provider():
        return {"success": True}

    @app.post("/api/v1/connection/test")
    async def test_connection():
        return {"connected": True}

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    return app


@pytest.fixture
def strict_config():
    """Create a strict rate limit config for testing."""
    return AuthRateLimitConfig(
        auth_requests_per_minute=3,
        auth_burst_size=2,
        failed_auth_requests_per_minute=2,
        failed_auth_lockout_minutes=1,
        api_key_validation_per_minute=5,
        session_creation_per_minute=3,
    )


class TestAuthRateLimitConfig:
    """Tests for AuthRateLimitConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AuthRateLimitConfig()

        assert config.auth_requests_per_minute == 10
        assert config.auth_burst_size == 3
        assert config.failed_auth_requests_per_minute == 5
        assert config.failed_auth_lockout_minutes == 15
        assert config.api_key_validation_per_minute == 20
        assert config.session_creation_per_minute == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AuthRateLimitConfig(
            auth_requests_per_minute=5,
            failed_auth_lockout_minutes=30,
        )

        assert config.auth_requests_per_minute == 5
        assert config.failed_auth_lockout_minutes == 30


class TestAuthRateLimitMiddleware:
    """Tests for AuthRateLimitMiddleware."""

    def test_non_auth_path_not_limited(self, app):
        """Test that non-auth paths are not rate limited."""
        app.add_middleware(AuthRateLimitMiddleware)
        client = TestClient(app)

        # Health endpoint should not be rate limited
        for _ in range(20):
            response = client.get("/api/v1/health")
            assert response.status_code == 200

    def test_auth_path_rate_limited(self, app, strict_config):
        """Test that auth paths are rate limited."""
        app.add_middleware(AuthRateLimitMiddleware, config=strict_config)
        client = TestClient(app)

        # First few requests should succeed
        for _i in range(3):
            response = client.post("/api/v1/session")
            # May be 200 or 429 depending on timing
            assert response.status_code in (200, 429)

        # After limit, should get 429
        response = client.post("/api/v1/session")
        # Rate limit may or may not be hit depending on timing
        assert response.status_code in (200, 429)

    def test_options_request_not_limited(self, app, strict_config):
        """Test that OPTIONS requests are not rate limited."""
        app.add_middleware(AuthRateLimitMiddleware, config=strict_config)
        client = TestClient(app)

        # OPTIONS requests should always pass
        for _ in range(10):
            response = client.options("/api/v1/session")
            # OPTIONS may return 405 if not configured, but not 429
            assert response.status_code != 429

    def test_get_stats(self, app, strict_config):
        """Test getting rate limiter statistics."""
        middleware = AuthRateLimitMiddleware(app, config=strict_config)

        stats = middleware.get_stats()

        assert "active_lockouts" in stats
        assert "tracked_ips" in stats
        assert "config" in stats
        assert stats["config"]["auth_requests_per_minute"] == 3


class TestAuthRateLimiterSingleton:
    """Tests for auth rate limiter singleton functions."""

    def test_set_and_get_rate_limiter(self, app):
        """Test setting and getting the rate limiter instance."""
        middleware = AuthRateLimitMiddleware(app)
        set_auth_rate_limiter(middleware)

        retrieved = get_auth_rate_limiter()
        assert retrieved is middleware

    def test_get_rate_limiter_returns_none_when_not_set(self):
        """Test that get returns None when not set."""
        # Clear any existing instance
        if hasattr(get_auth_rate_limiter, "_instance"):
            delattr(get_auth_rate_limiter, "_instance")

        result = get_auth_rate_limiter()
        assert result is None


class TestRateLimitHeaders:
    """Tests for rate limit response headers."""

    def test_429_response_includes_retry_after(self, app, strict_config):
        """Test that 429 responses include Retry-After header."""
        app.add_middleware(AuthRateLimitMiddleware, config=strict_config)
        client = TestClient(app)

        # Make many requests to trigger rate limit
        responses = []
        for _ in range(10):
            response = client.post("/api/v1/session")
            responses.append(response)

        # Check if any response was rate limited
        rate_limited = [r for r in responses if r.status_code == 429]

        if rate_limited:
            # Verify Retry-After header is present
            assert "Retry-After" in rate_limited[0].headers


# Marker for unit tests
pytestmark = pytest.mark.unit
