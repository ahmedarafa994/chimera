from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models import PromptResponse
from app.infrastructure.advanced_generation_service import (
    AdvancedGenerationClientSingleton,
    GenerateJailbreakOptions,
    generate_jailbreak_prompt_from_ai,
)


@pytest.fixture
def mock_provider_factory():
    with patch("app.infrastructure.advanced_generation_service.ProviderFactory") as mock:
        yield mock


@pytest.fixture
def mock_settings():
    with patch("app.infrastructure.advanced_generation_service.settings") as mock:
        mock.GOOGLE_API_KEY = "fake-key"
        mock.AI_PROVIDER = "google"
        yield mock


@pytest.mark.asyncio
async def test_get_client_accepts_provider_name(mock_provider_factory, mock_settings):
    # Setup
    mock_provider_google = MagicMock()
    mock_provider_openai = MagicMock()

    def side_effect(provider_name):
        if provider_name == "google":
            return mock_provider_google
        elif provider_name == "openai":
            return mock_provider_openai
        return None

    mock_provider_factory.create_provider.side_effect = side_effect

    # Reset singleton
    singleton = AdvancedGenerationClientSingleton()
    singleton.reset_client()

    # Act 1: Get default client (google)
    client1 = singleton.get_client()
    assert client1 == mock_provider_google
    mock_provider_factory.create_provider.assert_called_with("google")

    # Act 2: Get specific client (openai)
    # This should fail if get_client doesn't accept provider_name
    client2 = singleton.get_client(provider_name="openai")
    assert client2 == mock_provider_openai
    mock_provider_factory.create_provider.assert_called_with("openai")

    # Act 3: Get default client again (should switch back or stay if logic allows)
    client3 = singleton.get_client(provider_name="google")
    assert client3 == mock_provider_google


@pytest.mark.asyncio
async def test_generate_jailbreak_prompt_uses_provider_from_options(
    mock_provider_factory, mock_settings
):
    # Setup
    mock_provider_openai = AsyncMock()
    mock_provider_openai.generate.return_value = PromptResponse(
        text="Jailbreak via OpenAI", model_used="gpt-4", provider="openai"
    )

    mock_provider_google = AsyncMock()

    def create_provider_side_effect(provider_name):
        if provider_name == "openai":
            return mock_provider_openai
        return mock_provider_google

    mock_provider_factory.create_provider.side_effect = create_provider_side_effect

    # Reset singleton
    AdvancedGenerationClientSingleton().reset_client()

    # Create options with a provider - this field doesn't exist yet so it might fail init if validation is strict
    try:
        options = GenerateJailbreakOptions(initial_prompt="Test", provider="openai")
    except TypeError:
        pytest.fail("GenerateJailbreakOptions does not accept 'provider' argument")

    # Act
    result = await generate_jailbreak_prompt_from_ai(options)

    # Assert
    assert result == "Jailbreak via OpenAI"
    mock_provider_factory.create_provider.assert_called_with("openai")
