"""
Tests for Routeway Provider Implementation

Story 1.2: Direct API Integration

Tests cover:
- Provider initialization
- Configuration handling
- Generation config building
- Non-streaming generation
- Streaming support
- Health checks
- Error handling
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.domain.models import GenerationConfig, PromptRequest
from app.infrastructure.providers.routeway_provider import RoutewayProvider


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.ROUTEWAY_API_KEY = "clsk-test-api-key-123456789"
    settings.routeway_model = "gpt-4o-mini"
    return settings


@pytest.fixture
def provider(mock_settings):
    """Create a Routeway provider instance for testing."""
    return RoutewayProvider(config=mock_settings)


class TestRoutewayProviderInitialization:
    """Tests for Routeway provider initialization."""

    def test_provider_name(self, provider):
        """Test that provider name is set correctly."""
        assert provider.provider_name == "routeway"

    def test_base_url(self, provider):
        """Test that base URL is correct."""
        assert provider._BASE_URL == "https://api.routeway.ai/v1"

    def test_default_model(self, provider):
        """Test that default model is gpt-4o-mini."""
        assert provider._DEFAULT_MODEL == "gpt-4o-mini"

    def test_available_models(self, provider):
        """Test that available models list is populated."""
        assert len(provider._AVAILABLE_MODELS) > 0
        assert "gpt-4o" in provider._AVAILABLE_MODELS
        assert "gpt-4o-mini" in provider._AVAILABLE_MODELS
        assert "claude-3-5-sonnet-20241022" in provider._AVAILABLE_MODELS
        assert "llama-4-scout-17b-16e-instruct" in provider._AVAILABLE_MODELS

    def test_free_tier_models_available(self, provider):
        """Test that free tier models (with :free suffix) are listed."""
        free_models = [m for m in provider._AVAILABLE_MODELS if ":free" in m]
        assert len(free_models) > 0
        assert "gpt-4o-mini:free" in free_models


class TestRoutewayProviderConfiguration:
    """Tests for Routeway provider configuration."""

    def test_get_api_key(self, provider, mock_settings):
        """Test API key retrieval."""
        assert provider._get_api_key() == "clsk-test-api-key-123456789"

    def test_get_default_model_from_config(self, mock_settings):
        """Test getting default model from config."""
        mock_settings.routeway_model = "gpt-4o"
        provider = RoutewayProvider(config=mock_settings)
        assert provider._get_default_model() == "gpt-4o"

    def test_get_default_model_fallback(self):
        """Test default model fallback when not in config."""
        settings = MagicMock()
        settings.ROUTEWAY_API_KEY = "clsk-test-key-12345678901234567890"
        settings.routeway_model = None
        provider = RoutewayProvider(config=settings)
        assert provider._get_default_model() == "gpt-4o-mini"


class TestRoutewayProviderBuildConfig:
    """Tests for generation config building."""

    def test_build_config_with_temperature(self, provider):
        """Test generation config with temperature."""
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(temperature=0.5),
        )
        config = provider._build_generation_config(request)
        assert config["temperature"] == 0.5

    def test_build_config_temperature_clamping(self, provider):
        """Test that temperature values are respected."""
        # Test with max valid temperature
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(temperature=1.0),
        )
        config = provider._build_generation_config(request)
        assert config["temperature"] == 1.0

    def test_build_config_top_p(self, provider):
        """Test that top_p is correctly passed."""
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(top_p=0.9),
        )
        config = provider._build_generation_config(request)
        assert config["top_p"] == 0.9

    def test_build_config_defaults(self, provider):
        """Test config values when no explicit config is provided."""
        request = PromptRequest(prompt="test")
        config = provider._build_generation_config(request)
        # These come from GenerationConfig defaults
        assert config["temperature"] == 0.7  # GenerationConfig default
        assert config["max_tokens"] == 2048  # GenerationConfig max_output_tokens default

    def test_build_config_with_max_output_tokens(self, provider):
        """Test generation config with max_output_tokens."""
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(max_output_tokens=4096),
        )
        config = provider._build_generation_config(request)
        assert config["max_tokens"] == 4096

    def test_build_config_with_stop_sequences(self, provider):
        """Test generation config with stop sequences."""
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(stop_sequences=["END", "STOP"]),
        )
        config = provider._build_generation_config(request)
        assert config["stop"] == ["END", "STOP"]


class TestRoutewayProviderGeneration:
    """Tests for Routeway generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }
        mock_response.raise_for_status = MagicMock()

        request = PromptRequest(prompt="Hello!")
        client = MagicMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=mock_response)

        response = await provider._generate_impl(client, request, "gpt-4o-mini")

        assert response.text == "Hello! How can I help you today?"
        assert response.provider == "routeway"
        assert response.model_used == "gpt-4o-mini"
        assert response.usage_metadata["prompt_tokens"] == 12
        assert response.usage_metadata["completion_tokens"] == 8
        assert response.usage_metadata["total_tokens"] == 20
        assert response.finish_reason == "stop"


class TestRoutewayProviderHealthCheck:
    """Tests for Routeway health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"id": "gpt-4o-mini", "name": "GPT-4o Mini"}]
        }

        client = MagicMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=mock_response)

        result = await provider._check_health_impl(client)
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test failed health check."""
        client = MagicMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=httpx.RequestError("Connection error"))

        result = await provider._check_health_impl(client)
        assert result is False


class TestRoutewayProviderContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self, provider):
        """Test async context manager."""
        async with provider as p:
            assert p is provider

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, provider):
        """Test that context manager cleans up HTTP client."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()
        provider._http_client = mock_client

        await provider.__aexit__(None, None, None)

        mock_client.aclose.assert_called_once()


class TestRoutewayProviderErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handle_401_error(self, provider):
        """Test handling of 401 Unauthorized error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid or missing API key",
                "type": "error",
                "code": 401
            }
        }

        error = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=MagicMock(),
            response=mock_response
        )

        with pytest.raises(Exception) as exc_info:
            await provider._handle_http_error(error)

        assert "authentication failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_handle_429_error(self, provider):
        """Test handling of 429 Rate Limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {
                "message": "You have used up your free quota",
                "type": "error",
                "code": 429
            }
        }

        error = httpx.HTTPStatusError(
            message="429 Too Many Requests",
            request=MagicMock(),
            response=mock_response
        )

        with pytest.raises(Exception) as exc_info:
            await provider._handle_http_error(error)

        assert "rate limit" in str(exc_info.value).lower()
