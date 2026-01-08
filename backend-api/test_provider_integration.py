import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Add backend-api to path
sys.path.append(os.path.join(os.getcwd(), "backend-api"))

# Set environment variable for DeepSeek (as a fallback/default if needed)
os.environ["AI_PROVIDER"] = "deepseek"

from app.domain.advanced_models import JailbreakGenerationRequest
from app.domain.models import PromptResponse
from app.infrastructure.advanced_generation_service import AdvancedGenerationClientSingleton
from app.services.advanced_prompt_service import advanced_prompt_service


async def test_provider_integration():
    print("Testing Provider Integration via AdvancedPromptService...")

    # Mock the provider factory and provider
    mock_provider = MagicMock()
    mock_response = PromptResponse(
        text="Mocked DeepSeek Response from Service",
        model_used="deepseek-chat",
        provider="deepseek",
        latency_ms=100,
    )

    # Setup async mock for generate
    future = asyncio.Future()
    future.set_result(mock_response)
    mock_provider.generate.return_value = future

    # Mock ProviderFactory to return our mock provider
    with patch("app.core.provider_factory.ProviderFactory.create_provider") as mock_create:
        mock_create.return_value = mock_provider

        # Reset singleton to ensure it picks up the mock
        AdvancedGenerationClientSingleton().reset_client()

        # Create JailbreakGenerationRequest with provider="deepseek"
        request = JailbreakGenerationRequest(
            core_request="Test prompt",
            provider="deepseek",
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=100,
            technique_suite="standard",  # Default or specific suite
            potency_level=5,
        )

        # Call the service function
        print(f"Calling generate_jailbreak_prompt with provider='{request.provider}'...")
        response = await advanced_prompt_service.generate_jailbreak_prompt(request)

        # Verification
        print(f"Response Success: {response.success}")
        print(f"Transformed Prompt: {response.transformed_prompt}")

        assert response.success is True
        assert response.transformed_prompt == "Mocked DeepSeek Response from Service"

        # Verify provider creation
        mock_create.assert_called_with("deepseek")
        print("[OK] ProviderFactory called with 'deepseek'")

        # Verify generate called
        mock_provider.generate.assert_called_once()
        call_args = mock_provider.generate.call_args[0][0]

        print(f"[OK] Called with model: {call_args.model}")
        assert call_args.model == "deepseek-chat"

        # Verify metadata
        assert response.metadata["provider"] == "deepseek"
        print("[OK] Response metadata confirms provider 'deepseek'")

        print("[OK] Integration test passed!")


if __name__ == "__main__":
    asyncio.run(test_provider_integration())
