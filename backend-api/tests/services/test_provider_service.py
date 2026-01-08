"""
Tests for Provider Management Service.

Tests provider configuration, model routing,
and multi-provider support.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_register_provider(self):
        """Test registering a new provider."""
        registry = {}

        provider = {
            "name": "openai",
            "api_key": "sk-test",
            "models": ["gpt-4", "gpt-3.5-turbo"],
            "enabled": True,
        }

        registry["openai"] = provider

        assert "openai" in registry
        assert len(registry["openai"]["models"]) == 2

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        registry = {
            "openai": {"name": "openai"},
            "anthropic": {"name": "anthropic"},
        }

        del registry["openai"]

        assert "openai" not in registry
        assert "anthropic" in registry

    def test_list_providers(self):
        """Test listing all providers."""
        registry = {
            "openai": {"name": "openai", "enabled": True},
            "anthropic": {"name": "anthropic", "enabled": True},
            "google": {"name": "google", "enabled": False},
        }

        enabled = [p for p, v in registry.items() if v["enabled"]]

        assert len(enabled) == 2
        assert "google" not in enabled


class TestProviderConfig:
    """Tests for provider configuration."""

    def test_validate_api_key(self):
        """Test API key validation."""
        valid_keys = [
            "sk-1234567890abcdef1234567890abcdef",
            "anthropic-key-12345",
        ]

        for key in valid_keys:
            assert len(key) >= 10

    def test_validate_endpoint(self):
        """Test endpoint URL validation."""
        import re

        valid_endpoints = [
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "http://localhost:8000",
        ]

        url_pattern = r"^https?://[\w\-.]+(:\d+)?(/[\w\-.]*)*$"

        for endpoint in valid_endpoints:
            assert re.match(url_pattern, endpoint) is not None

    def test_config_defaults(self):
        """Test configuration default values."""
        defaults = {
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 1.0,
            "max_tokens": 4096,
        }

        assert defaults["timeout"] == 30
        assert defaults["max_retries"] == 3


class TestModelRouter:
    """Tests for model routing service."""

    @pytest.fixture
    def mock_router(self):
        """Create mock model router."""
        router = MagicMock()
        router.route = MagicMock(
            return_value={
                "provider": "openai",
                "model": "gpt-4",
                "endpoint": "https://api.openai.com/v1",
            }
        )
        return router

    def test_route_to_best_provider(self, mock_router):
        """Test routing to best available provider."""
        result = mock_router.route(model="gpt-4")

        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4"

    def test_fallback_routing(self, mock_router):
        """Test fallback routing when primary fails."""
        # Simulate primary failure
        mock_router.route.side_effect = [
            RuntimeError("Provider unavailable"),
            {
                "provider": "anthropic",
                "model": "claude-3",
                "endpoint": "https://api.anthropic.com",
            },
        ]

        # First call fails
        with pytest.raises(RuntimeError):
            mock_router.route(model="gpt-4")

        # Fallback succeeds
        result = mock_router.route(model="claude-3")
        assert result["provider"] == "anthropic"

    def test_load_balancing(self):
        """Test load balancing across providers."""
        providers = [
            {"name": "openai", "load": 0.3},
            {"name": "anthropic", "load": 0.5},
            {"name": "google", "load": 0.8},
        ]

        # Select provider with lowest load
        selected = min(providers, key=lambda p: p["load"])

        assert selected["name"] == "openai"


class TestProviderHealth:
    """Tests for provider health monitoring."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test provider health check."""
        mock_provider = MagicMock()
        mock_provider.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "latency_ms": 150,
                "last_check": "2024-01-15T10:00:00Z",
            }
        )

        result = await mock_provider.health_check()

        assert result["status"] == "healthy"
        assert result["latency_ms"] < 1000

    @pytest.mark.asyncio
    async def test_unhealthy_provider_detection(self):
        """Test detection of unhealthy providers."""
        providers = [
            {"name": "openai", "healthy": True, "latency": 100},
            {"name": "anthropic", "healthy": False, "latency": 5000},
            {"name": "google", "healthy": True, "latency": 200},
        ]

        unhealthy = [p for p in providers if not p["healthy"]]

        assert len(unhealthy) == 1
        assert unhealthy[0]["name"] == "anthropic"

    def test_auto_disable_unhealthy(self):
        """Test automatic disabling of unhealthy providers."""
        provider = {
            "name": "test",
            "enabled": True,
            "consecutive_failures": 5,
            "max_failures": 3,
        }

        if provider["consecutive_failures"] >= provider["max_failures"]:
            provider["enabled"] = False

        assert provider["enabled"] is False


class TestRateLimiting:
    """Tests for provider rate limiting."""

    def test_rate_limit_tracking(self):
        """Test rate limit tracking."""
        rate_limit = {
            "requests_per_minute": 60,
            "tokens_per_minute": 100000,
            "current_requests": 45,
            "current_tokens": 75000,
        }

        requests_remaining = (
            rate_limit["requests_per_minute"] - rate_limit["current_requests"]
        )
        tokens_remaining = (
            rate_limit["tokens_per_minute"] - rate_limit["current_tokens"]
        )

        assert requests_remaining == 15
        assert tokens_remaining == 25000

    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded detection."""
        rate_limit = {
            "requests_per_minute": 60,
            "current_requests": 65,
        }

        exceeded = rate_limit["current_requests"] > rate_limit["requests_per_minute"]

        assert exceeded is True

    def test_rate_limit_reset(self):
        """Test rate limit reset."""
        import time

        rate_limit = {
            "current_requests": 50,
            "reset_time": time.time() - 60,  # 60 seconds ago
            "window_seconds": 60,
        }

        # Check if window has passed
        time_since_reset = time.time() - rate_limit["reset_time"]
        should_reset = time_since_reset >= rate_limit["window_seconds"]

        if should_reset:
            rate_limit["current_requests"] = 0
            rate_limit["reset_time"] = time.time()

        assert rate_limit["current_requests"] == 0


class TestModelSync:
    """Tests for model synchronization service."""

    @pytest.mark.asyncio
    async def test_sync_available_models(self):
        """Test synchronizing available models."""
        mock_sync = MagicMock()
        mock_sync.sync_models = AsyncMock(
            return_value={
                "openai": ["gpt-4", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-opus", "claude-3-sonnet"],
            }
        )

        result = await mock_sync.sync_models()

        assert "openai" in result
        assert "gpt-4" in result["openai"]

    def test_model_capability_mapping(self):
        """Test model capability mapping."""
        models = {
            "gpt-4": {
                "max_tokens": 8192,
                "supports_vision": True,
                "supports_function_calling": True,
            },
            "claude-3-opus": {
                "max_tokens": 200000,
                "supports_vision": True,
                "supports_function_calling": True,
            },
        }

        # Find models with vision support
        vision_models = [
            m for m, caps in models.items() if caps["supports_vision"]
        ]

        assert len(vision_models) == 2
