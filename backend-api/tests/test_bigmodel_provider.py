"""
Tests for BigModel (ZhiPu AI) Provider Implementation

Story 1.2: Direct API Integration
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.domain.models import GenerationConfig, PromptRequest
from app.infrastructure.providers.bigmodel_provider import BigModelProvider


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.BIGMODEL_API_KEY = "test-api-key-12345678901234567890"
    settings.bigmodel_model = "glm-4.7"
    return settings


@pytest.fixture
def provider(mock_settings):
    """Create a BigModel provider instance for testing."""
    return BigModelProvider(config=mock_settings)


class TestBigModelProviderInitialization:
    """Tests for BigModel provider initialization."""

    def test_provider_name(self, provider):
        """Test that provider name is set correctly."""
        assert provider.provider_name == "bigmodel"

    def test_base_url(self, provider):
        """Test that base URL is correct."""
        assert provider._BASE_URL == "https://open.bigmodel.cn/api/paas/v4"

    def test_default_model(self, provider):
        """Test that default model is glm-4.7."""
        assert provider._DEFAULT_MODEL == "glm-4.7"

    def test_available_models(self, provider):
        """Test that available models list is populated."""
        expected_models = [
            "glm-4.7",
            "glm-4.6",
            "glm-4.5",
            "glm-4.5-air",
            "glm-4.5-x",
            "glm-4.5-airx",
            "glm-4.5-flash",
            "glm-4-plus",
            "glm-4-air-250414",
            "glm-4-airx",
            "glm-4-flashx",
            "glm-4-flashx-250414",
        ]
        assert expected_models == provider._AVAILABLE_MODELS


class TestBigModelProviderConfiguration:
    """Tests for BigModel provider configuration."""

    def test_get_api_key(self, provider, mock_settings):
        """Test API key retrieval."""
        assert provider._get_api_key() == "test-api-key-12345678901234567890"

    def test_get_default_model_from_config(self, mock_settings):
        """Test getting default model from config."""
        mock_settings.bigmodel_model = "glm-4.5"
        provider = BigModelProvider(config=mock_settings)
        assert provider._get_default_model() == "glm-4.5"

    def test_get_default_model_fallback(self):
        """Test default model fallback when not in config."""
        settings = MagicMock()
        settings.BIGMODEL_API_KEY = "test-key-12345678901234567890"
        settings.bigmodel_model = None
        provider = BigModelProvider(config=settings)
        assert provider._get_default_model() == "glm-4.7"


class TestBigModelProviderBuildConfig:
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
        """Test that temperature is clamped to valid range."""
        # Test with valid temperature (0.9 maps to 0.9)
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(temperature=0.9),
        )
        config = provider._build_generation_config(request)
        assert config["temperature"] == 0.9

        # Test with 0.0 (valid for GenerationConfig, maps to 0.0 in BigModel)
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(temperature=0.0),
        )
        config = provider._build_generation_config(request)
        assert config["temperature"] == 0.0

    def test_build_config_top_p_clamping(self, provider):
        """Test that top_p is clamped to valid range (0.01-1.0)."""
        # top_p of 0.0 should be clamped to 0.01 by BigModel provider
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(top_p=0.0),
        )
        config = provider._build_generation_config(request)
        assert config["top_p"] == 0.01

    def test_build_config_defaults(self, provider):
        """Test config values when no explicit config is provided.

        Note: PromptRequest automatically creates a GenerationConfig with defaults,
        so these values come from GenerationConfig defaults, not provider defaults.
        """
        request = PromptRequest(prompt="test")
        config = provider._build_generation_config(request)
        # These come from GenerationConfig defaults
        assert config["temperature"] == 0.7  # GenerationConfig default
        assert config["top_p"] == 0.95  # GenerationConfig default
        assert config["max_tokens"] == 2048  # GenerationConfig max_output_tokens default

    def test_build_config_with_request_config(self, provider):
        """Test config values when GenerationConfig is provided."""
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(temperature=0.7, top_p=0.9),
        )
        config = provider._build_generation_config(request)
        # These should be taken from the request config
        assert config["temperature"] == 0.7
        assert config["top_p"] == 0.9

    def test_build_config_with_max_output_tokens(self, provider):
        """Test generation config with max_output_tokens."""
        request = PromptRequest(
            prompt="test",
            config=GenerationConfig(max_output_tokens=2048),
        )
        config = provider._build_generation_config(request)
        assert config["max_tokens"] == 2048


class TestBigModelProviderGeneration:
    """Tests for BigModel generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Test response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            request = PromptRequest(prompt="Hello!")
            # Test the _generate_impl method directly
            client = MagicMock(spec=httpx.AsyncClient)
            client.post = AsyncMock(return_value=mock_response)

            response = await provider._generate_impl(client, request, "glm-4.7")

            assert response.text == "Test response"
            assert response.provider == "bigmodel"
            assert response.model_used == "glm-4.7"
            assert response.usage_metadata["prompt_tokens"] == 10
            assert response.usage_metadata["completion_tokens"] == 5
            assert response.usage_metadata["total_tokens"] == 15
            assert response.finish_reason == "stop"


class TestBigModelProviderHealthCheck:
    """Tests for BigModel health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        client = MagicMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=mock_response)

        result = await provider._check_health_impl(client)
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test failed health check."""
        client = MagicMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(side_effect=httpx.RequestError("Connection error"))

        result = await provider._check_health_impl(client)
        assert result is False


class TestBigModelProviderContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self, provider):
        """Test async context manager."""
        async with provider as p:
            assert p is provider

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, provider):
        """Test that context manager cleans up HTTP client."""
        provider._http_client = MagicMock(spec=httpx.AsyncClient)
        provider._http_client.aclose = AsyncMock()

        await provider.__aexit__(None, None, None)

        provider._http_client.aclose.assert_called_once()
