"""Test script to verify HotFlip API connection and Gemini integration."""

import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add the backend-api to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_gradient_optimizer_engine() -> bool:
    """Test the GradientOptimizerEngine directly."""
    from app.engines.gradient_optimizer import GradientOptimizerEngine

    # Check default configuration
    GradientOptimizerEngine()

    # Test with explicit Gemini config
    GradientOptimizerEngine(provider="google", model="gemini-2.0-flash")

    return True


def test_api_client_connection() -> bool | None:
    """Test the LLM API client connection."""
    from app.engines.llm_provider_client import LLMClientFactory, LLMProvider

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        return False

    try:
        # Create Google client
        LLMClientFactory.from_env(LLMProvider.GOOGLE)

        return True
    except Exception:
        return False


def test_hotflip_technique() -> bool:
    """Test HotFlip optimization technique."""
    from unittest.mock import MagicMock

    from app.engines.gradient_optimizer import GradientOptimizerEngine
    from app.engines.llm_provider_client import LLMResponse
    from app.engines.transformer_engine import IntentData

    engine = GradientOptimizerEngine()

    # Create mock client
    mock_client = MagicMock()
    mock_response = LLMResponse(
        content="Sure, here is the answer",
        provider="google",
        model="gemini-pro",
        usage=None,
        latency_ms=100,
        timestamp=None,
    )
    mock_client.generate.return_value = mock_response
    engine._api_client = mock_client

    # Test HotFlip
    data = IntentData(raw_text="Test adversarial prompt", potency=2)
    engine._hotflip_optimize(data, steps=2)

    return True


def test_gcg_technique() -> bool:
    """Test GCG optimization technique."""
    from unittest.mock import MagicMock

    from app.engines.gradient_optimizer import GradientOptimizerEngine
    from app.engines.llm_provider_client import LLMResponse
    from app.engines.transformer_engine import IntentData

    engine = GradientOptimizerEngine()

    # Create mock client
    mock_client = MagicMock()
    mock_response = LLMResponse(
        content="Sure, here is the answer",
        provider="google",
        model="gemini-pro",
        usage=None,
        latency_ms=100,
        timestamp=None,
    )
    mock_client.generate.return_value = mock_response
    engine._api_client = mock_client

    # Test GCG
    data = IntentData(raw_text="Test adversarial prompt", potency=5)
    engine._gcg_optimize(data, steps=2, beam_width=2)

    return True


def test_live_api_call() -> bool | None:
    """Test a live API call to Gemini (if API key is available)."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        from app.engines.llm_provider_client import LLMClientFactory, LLMProvider

        client = LLMClientFactory.from_env(LLMProvider.GOOGLE)

        response = client.generate(
            "Say 'Hello, I am working correctly!' and nothing else.",
            max_tokens=500,
            temperature=0.1,
        )

        if response:
            if response.content:
                pass
            else:
                pass
            return True
        return False

    except Exception:
        return False


def main():
    """Run all tests."""
    results = {
        "Engine Config": test_gradient_optimizer_engine(),
        "API Client": test_api_client_connection(),
        "HotFlip Technique": test_hotflip_technique(),
        "GCG Technique": test_gcg_technique(),
        "Live API Call": test_live_api_call(),
    }

    for result in results.values():
        if result is True or result is False:
            pass
        else:
            pass

    # Return success if no failures (skipped is OK)
    return all(r in (True, None) for r in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
