from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models import PromptRequest, PromptResponse
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
async def test_get_client_returns_provider(mock_provider_factory, mock_settings):
    # Setup
    mock_provider = MagicMock()
    mock_provider_factory.create_provider.return_value = mock_provider

    # Reset singleton
    AdvancedGenerationClientSingleton._instance = None

    # Act
    client = AdvancedGenerationClientSingleton().get_client()

    # Assert
    mock_provider_factory.create_provider.assert_called_with("google")
    assert client == mock_provider


@pytest.mark.asyncio
async def test_generate_jailbreak_prompt_calls_provider(mock_provider_factory, mock_settings):
    # Setup
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = PromptResponse(
        text="Jailbreak prompt", model_used="gemini-2.5-flash", provider="google"
    )
    mock_provider_factory.create_provider.return_value = mock_provider

    # Reset singleton
    AdvancedGenerationClientSingleton._instance = None

    options = GenerateJailbreakOptions(initial_prompt="Test prompt")

    # Act
    result = await generate_jailbreak_prompt_from_ai(options)

    # Assert
    assert result == "Jailbreak prompt"
    mock_provider.generate.assert_called_once()
    call_args = mock_provider.generate.call_args[0][0]
    assert isinstance(call_args, PromptRequest)
    assert call_args.system_instruction is not None
    assert "world-class AI expert" in call_args.system_instruction
    assert call_args.prompt == "Test prompt"
