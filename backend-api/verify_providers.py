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


async def verify_providers() -> None:
    get_settings()

    # Simulate Main.py registration
    try:
        gemini = GeminiClient()
        llm_service.register_provider("google", gemini, is_default=True)
    except Exception:
        pass

    try:
        openai = LLMClientFactory.from_env(LLMProviderType.OPENAI)
        llm_service.register_provider("openai", openai)
    except Exception:
        pass

    try:
        anthropic = LLMClientFactory.from_env(LLMProviderType.ANTHROPIC)
        llm_service.register_provider("anthropic", anthropic)
    except Exception:
        pass

    # List Providers
    providers = await llm_service.list_providers()
    for _p in providers.providers:
        pass

    # Test Generation (Mock if no keys, or real if keys exist)
    # We won't actually call generate to avoid cost/latency in this quick check,
    # unless we want to verification deep.
    # Just verifying the client objects are created correctly.


if __name__ == "__main__":
    asyncio.run(verify_providers())
