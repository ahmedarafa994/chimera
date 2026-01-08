import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.core.config import get_settings
from app.domain.models import LLMProviderType
from app.engines.llm_provider_client import LLMClientFactory
from app.infrastructure.gemini_client import GeminiClient
from app.services.llm_service import llm_service


async def verify_providers():
    print("--- Verifying Provider Registration ---")
    settings = get_settings()
    print(f"Connection Mode: {settings.get_connection_mode()}")
    print(f"Proxy URL: {settings.get_effective_base_url()}")

    # Simulate Main.py registration
    print("\nRegistering Providers...")
    try:
        gemini = GeminiClient()
        llm_service.register_provider("google", gemini, is_default=True)
        print("✅ Google (GeminiClient) Registered")
    except Exception as e:
        print(f"❌ Google Registration Failed: {e}")

    try:
        openai = LLMClientFactory.from_env(LLMProviderType.OPENAI)
        llm_service.register_provider("openai", openai)
        print("✅ OpenAI Registered")
    except Exception as e:
        print(f"❌ OpenAI Registration Failed: {e}")

    try:
        anthropic = LLMClientFactory.from_env(LLMProviderType.ANTHROPIC)
        llm_service.register_provider("anthropic", anthropic)
        print("✅ Anthropic Registered")
    except Exception as e:
        print(f"❌ Anthropic Registration Failed: {e}")

    # List Providers
    print("\nListing Providers via LLMService:")
    providers = await llm_service.list_providers()
    for p in providers.providers:
        print(f" - {p.provider} (Model: {p.model})")

    # Test Generation (Mock if no keys, or real if keys exist)
    # We won't actually call generate to avoid cost/latency in this quick check,
    # unless we want to verification deep.
    # Just verifying the client objects are created correctly.

    print("\nVerification Complete.")


if __name__ == "__main__":
    asyncio.run(verify_providers())
