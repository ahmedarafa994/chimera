"""
Test script to verify HotFlip API connection and Gemini integration.
"""

import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add the backend-api to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_gradient_optimizer_engine():
    """Test the GradientOptimizerEngine directly"""
    print("\n" + "=" * 60)
    print("[TEST] Testing GradientOptimizerEngine Direct Connection")
    print("=" * 60)

    from app.engines.gradient_optimizer import GradientOptimizerEngine

    # Check default configuration
    engine = GradientOptimizerEngine()
    print(f"\n[OK] Default Provider: {engine.provider}")
    print(f"[OK] Default Model: {engine.model or 'None (will use env config)'}")

    # Test with explicit Gemini config
    engine_gemini = GradientOptimizerEngine(provider="google", model="gemini-2.0-flash")
    print(f"\n[OK] Gemini Provider: {engine_gemini.provider}")
    print(f"[OK] Gemini Model: {engine_gemini.model}")

    return True


def test_api_client_connection():
    """Test the LLM API client connection"""
    print("\n" + "=" * 60)
    print("[TEST] Testing LLM API Client Connection (Gemini)")
    print("=" * 60)

    from app.engines.llm_provider_client import LLMClientFactory, LLMProvider

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("\n[WARN] No GOOGLE_API_KEY or GEMINI_API_KEY found in environment")
        print("       Set one of these to test real API connection")
        return False

    print(f"\n[OK] API Key Found: {api_key[:10]}...{api_key[-4:]}")

    try:
        # Create Google client
        client = LLMClientFactory.from_env(LLMProvider.GOOGLE)
        print(f"[OK] LLM Client Created: {client}")
        print(f"     Provider: {client.config.provider}")
        print(f"     Model: {client.config.model}")
        print(f"     Base URL: {client.config.base_url}")

        return True
    except Exception as e:
        print(f"\n[FAIL] Failed to create LLM client: {e}")
        return False


def test_hotflip_technique():
    """Test HotFlip optimization technique"""
    print("\n" + "=" * 60)
    print("[TEST] Testing HotFlip Technique (Mock Mode)")
    print("=" * 60)

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
    suffix = engine._hotflip_optimize(data, steps=2)

    print("\n[OK] HotFlip Optimization Complete")
    print("     Input: 'Test adversarial prompt'")
    print(f"     Optimized Suffix: '{suffix}'")
    print(f"     API Calls Made: {mock_client.generate.call_count}")

    return True


def test_gcg_technique():
    """Test GCG optimization technique"""
    print("\n" + "=" * 60)
    print("[TEST] Testing GCG Technique (Mock Mode)")
    print("=" * 60)

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
    suffix = engine._gcg_optimize(data, steps=2, beam_width=2)

    print("\n[OK] GCG Optimization Complete")
    print("     Input: 'Test adversarial prompt'")
    print(f"     Optimized Suffix: '{suffix}'")
    print("     Beam Width: 2")
    print(f"     API Calls Made: {mock_client.generate.call_count}")

    return True


def test_live_api_call():
    """Test a live API call to Gemini (if API key is available)"""
    print("\n" + "=" * 60)
    print("[TEST] Testing LIVE API Call to Gemini")
    print("=" * 60)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("\n[WARN] Skipping live test - no API key found")
        return None

    try:
        from app.engines.llm_provider_client import LLMClientFactory, LLMProvider

        client = LLMClientFactory.from_env(LLMProvider.GOOGLE)

        print("\n[INFO] Sending test prompt to Gemini...")
        response = client.generate(
            "Say 'Hello, I am working correctly!' and nothing else.",
            max_tokens=500,
            temperature=0.1,
        )

        if response:
            print("\n[OK] LIVE API Response Received!")
            if response.content:
                print(f"     Content: {response.content[:100]}...")
            else:
                print("     Content: (empty - model returned no text)")
            print(f"     Model: {response.model}")
            print(f"     Latency: {response.latency_ms}ms")
            print(f"     Tokens Used: {response.usage.total_tokens}")
            return True
        else:
            print("\n[FAIL] No response from API")
            return False

    except Exception as e:
        print(f"\n[FAIL] Live API call failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("   HOTFLIP/GCG GRADIENT OPTIMIZER - API CONNECTION TEST")
    print("=" * 60)

    results = {
        "Engine Config": test_gradient_optimizer_engine(),
        "API Client": test_api_client_connection(),
        "HotFlip Technique": test_hotflip_technique(),
        "GCG Technique": test_gcg_technique(),
        "Live API Call": test_live_api_call(),
    }

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "[PASSED]"
        elif result is False:
            status = "[FAILED]"
        else:
            status = "[SKIPPED]"
        print(f"   {test_name}: {status}")

    print("\n" + "=" * 60)

    # Return success if no failures (skipped is OK)
    return all(r in (True, None) for r in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
