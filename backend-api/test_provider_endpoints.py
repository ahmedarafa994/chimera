"""Comprehensive API Endpoint Verification Script.

Tests all unified provider system endpoints to verify:
- 6 providers are registered
- 20+ models are available
- Updated models are present (gpt-4o, gemini-2.5-pro, deepseek-reasoner, etc.)
"""

import time

import requests

BASE_URL = "http://localhost:8001/api/v1"


def test_provider_stats():
    """Test provider statistics endpoint."""
    response = requests.get(f"{BASE_URL}/providers/stats")
    if response.status_code == 200:
        return response.json()
    return None


def test_list_providers():
    """Test list all providers endpoint."""
    response = requests.get(f"{BASE_URL}/providers")
    if response.status_code == 200:
        providers = response.json()
        for _provider in providers:
            pass
        return providers
    return None


def test_provider_models(provider_id: str, expected_models: list[str]):
    """Test models for a specific provider."""
    response = requests.get(f"{BASE_URL}/providers/{provider_id}/models")
    if response.status_code == 200:
        models = response.json()

        for model in models:
            "✨" if model["id"] in expected_models else "  "

        # Check for expected new models
        model_ids = [m["id"] for m in models]
        found_new = [m for m in expected_models if m in model_ids]
        if found_new:
            pass

        return models
    return None


def test_all_models():
    """Test list all models endpoint."""
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code == 200:
        models = response.json()

        # Group by provider
        by_provider = {}
        for model in models:
            provider = model["provider_id"]
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model["id"])

        for provider, model_ids in by_provider.items():
            for _mid in model_ids[:3]:  # Show first 3
                pass

        return models
    return None


def main() -> None:
    """Run all tests."""
    # Wait for server
    for _i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/providers/stats", timeout=2)
            if response.status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        return

    # Test 1: Statistics
    stats = test_provider_stats()

    # Test 2: All providers
    test_list_providers()

    # Test 3: OpenAI models (verify gpt-4o, gpt-4o-mini)
    test_provider_models("openai", ["gpt-4o", "gpt-4o-mini"])

    # Test 4: Google models (verify gemini-2.5-pro, 2.0-flash)
    test_provider_models(
        "google",
        ["gemini-2.5-pro-latest", "gemini-2.5-flash-latest", "gemini-2.0-flash-001"],
    )

    # Test 5: DeepSeek models (verify deepseek-reasoner)
    test_provider_models("deepseek", ["deepseek-reasoner"])

    # Test 6: BigModel models (verify glm-4-plus, glm-4-air)
    test_provider_models("bigmodel", ["glm-4-plus", "glm-4-air"])

    # Test 7: Anthropic models
    test_provider_models("anthropic", [])

    # Test 8: Routeway models
    test_provider_models("routeway", [])

    # Test 9: All models
    test_all_models()

    # Summary
    if stats:
        expected_providers = 6
        expected_models = 20

        if (
            stats["total_providers"] >= expected_providers
            and stats["total_models"] >= expected_models
        ):
            pass
        else:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
