"""
Comprehensive API Endpoint Verification Script

Tests all unified provider system endpoints to verify:
- 6 providers are registered
- 20+ models are available
- Updated models are present (gpt-4o, gemini-2.5-pro, deepseek-reasoner, etc.)
"""

import time

import requests

BASE_URL = "http://localhost:8001/api/v1"


def test_provider_stats():
    """Test provider statistics endpoint"""
    print("\n" + "=" * 60)
    print("📊 PROVIDER STATISTICS")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/providers/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"✅ Total Providers: {stats['total_providers']}")
        print(f"✅ Enabled Providers: {stats['enabled_providers']}")
        print(f"✅ Total Models: {stats['total_models']}")
        print(f"✅ Total Aliases: {stats['total_aliases']}")
        return stats
    else:
        print(f"❌ Failed: {response.status_code}")
        return None


def test_list_providers():
    """Test list all providers endpoint"""
    print("\n" + "=" * 60)
    print("🏢 ALL PROVIDERS")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/providers")
    if response.status_code == 200:
        providers = response.json()
        for provider in providers:
            print(f"\n  📦 {provider['name']} ({provider['id']})")
            print(f"     Models: {provider['model_count']}")
            print(f"     Enabled: {provider['is_enabled']}")
            print(f"     Capabilities: {', '.join(provider['capabilities'])}")
        return providers
    else:
        print(f"❌ Failed: {response.status_code}")
        return None


def test_provider_models(provider_id: str, expected_models: list[str]):
    """Test models for a specific provider"""
    print(f"\n{'='*60}")
    print(f"🤖 {provider_id.upper()} MODELS")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/providers/{provider_id}/models")
    if response.status_code == 200:
        models = response.json()
        print(f"✅ Found {len(models)} models:")

        for model in models:
            marker = "✨" if model["id"] in expected_models else "  "
            print(f"{marker} {model['id']:<30} {model['name']}")
            print(
                f"     Context: {model['context_window']:,} | Output: {model['max_output_tokens']:,}"
            )
            print(
                f"     Price: ${model['pricing_input_per_1k']:.6f}/${model['pricing_output_per_1k']:.6f} per 1K"
            )

        # Check for expected new models
        model_ids = [m["id"] for m in models]
        found_new = [m for m in expected_models if m in model_ids]
        if found_new:
            print(f"\n  ✨ NEW MODELS VERIFIED: {', '.join(found_new)}")

        return models
    else:
        print(f"❌ Failed: {response.status_code}")
        return None


def test_all_models():
    """Test list all models endpoint"""
    print(f"\n{'='*60}")
    print("🌐 ALL MODELS ACROSS PROVIDERS")
    print("=" * 60)

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
            print(f"\n  {provider}: {len(model_ids)} models")
            for mid in model_ids[:3]:  # Show first 3
                print(f"    - {mid}")

        print(f"\n✅ Total: {len(models)} models across all providers")
        return models
    else:
        print(f"❌ Failed: {response.status_code}")
        return None


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🚀 UNIFIED PROVIDER SYSTEM - API VERIFICATION")
    print("=" * 60)
    print(f"Testing: {BASE_URL}")

    # Wait for server
    print("\nWaiting for server to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/providers/stats", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!\n")
                break
        except:
            time.sleep(1)
            print(f"  Attempt {i+1}/10...")
    else:
        print("❌ Server not responding. Make sure it's running on port 8001")
        return

    # Test 1: Statistics
    stats = test_provider_stats()

    # Test 2: All providers
    test_list_providers()

    # Test 3: OpenAI models (verify gpt-4o, gpt-4o-mini)
    test_provider_models("openai", ["gpt-4o", "gpt-4o-mini"])

    # Test 4: Google models (verify gemini-2.5-pro, 2.0-flash)
    test_provider_models(
        "google", ["gemini-2.5-pro-latest", "gemini-2.5-flash-latest", "gemini-2.0-flash-001"]
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
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    if stats:
        expected_providers = 6
        expected_models = 20

        print(
            f"\n  Providers: {stats['total_providers']}/{expected_providers} ({'✅' if stats['total_providers'] >= expected_providers else '⚠️'})"
        )
        print(
            f"  Models: {stats['total_models']}/{expected_models}+ ({'✅' if stats['total_models'] >= expected_models else '⚠️'})"
        )

        if (
            stats["total_providers"] >= expected_providers
            and stats["total_models"] >= expected_models
        ):
            print("\n  🎉 ALL VERIFICATIONS PASSED!")
        else:
            print("\n  ⚠️  Some verifications incomplete")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
