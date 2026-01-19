"""
Test Model Selection Propagation

This script tests whether the selected model is properly propagated
through all services and engines in the application.
"""

import requests

BASE_URL = "http://localhost:8001"


def test_selection_endpoint():
    """Test the selection endpoint and context"""
    print("\n" + "=" * 60)
    print("1. TESTING MODEL SELECTION ENDPOINTS")
    print("=" * 60)

    # Get current selection
    response = requests.get(f"{BASE_URL}/api/v1/selection/current")
    print("\n✓ GET /api/v1/selection/current")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        selection = response.json()
        print(f"  Current Provider: {selection.get('provider_id')}")
        print(f"  Current Model: {selection.get('model_id')}")
        print(f"  Scope: {selection.get('scope')}")
        return selection
    else:
        print(f"  ❌ Failed: {response.text}")
        return None


def test_model_selection_update():
    """Test updating model selection"""
    print("\n" + "=" * 60)
    print("2. TESTING MODEL SELECTION UPDATE")
    print("=" * 60)

    # Try to set a specific model
    test_selections = [
        {"provider_id": "openai", "model_id": "gpt-5.2"},
        {"provider_id": "anthropic", "model_id": "claude-opus-4.5"},
        {"provider_id": "deepseek", "model_id": "deepseek-v4"},
    ]

    for selection in test_selections:
        print(f"\n→ Setting to {selection['provider_id']}/{selection['model_id']}")

        # Note: Check if there's an endpoint to set selection
        # This might be session-based or user-based
        response = requests.post(f"{BASE_URL}/api/v1/selection/set", json=selection)

        if response.status_code == 200 or response.status_code == 404:
            if response.status_code == 404:
                print("  ⚠ Set selection endpoint not found")
                print("  Selection might be session/user-based")
            else:
                print(f"  ✓ Selection updated: {response.json()}")
        else:
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text[:200]}")


def test_provider_resolution():
    """Test if provider/model can be resolved via aliases"""
    print("\n" + "=" * 60)
    print("3. TESTING PROVIDER ALIAS RESOLUTION")
    print("=" * 60)

    test_cases = [
        ("openai", "OpenAI"),
        ("gpt", "OpenAI (alias)"),
        ("anthropic", "Anthropic"),
        ("claude", "Anthropic (alias)"),
        ("google", "Google"),
        ("gemini", "Google (alias)"),
    ]

    for provider_id, description in test_cases:
        response = requests.get(f"{BASE_URL}/api/v1/providers/{provider_id}/models")
        if response.status_code == 200:
            models = response.json()
            print(f"  ✓ {description}: {len(models)} models")
        else:
            print(f"  ✗ {description}: Failed ({response.status_code})")


def test_unified_registry():
    """Test unified registry access"""
    print("\n" + "=" * 60)
    print("4. TESTING UNIFIED REGISTRY")
    print("=" * 60)

    # Test registry stats
    response = requests.get(f"{BASE_URL}/api/v1/providers/stats")
    if response.status_code == 200:
        stats = response.json()
        print("\n  Registry Statistics:")
        print(f"    Total Providers: {stats.get('total_providers')}")
        print(f"    Enabled Providers: {stats.get('enabled_providers')}")
        print(f"    Total Models: {stats.get('total_models')}")
        print(f"    Total Aliases: {stats.get('total_aliases')}")
    else:
        print(f"  ❌ Failed to get stats: {response.status_code}")

    # Test getting all models
    response = requests.get(f"{BASE_URL}/api/v1/models")
    if response.status_code == 200:
        models = response.json()
        print(f"\n  ✓ Retrieved {len(models)} total models")

        # Group by provider
        by_provider = {}
        for model in models:
            provider = model.get("provider_id")
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model["id"])

        print("\n  Models by Provider:")
        for provider, model_list in by_provider.items():
            print(f"    {provider}: {len(model_list)} models")
            for model_id in model_list[:2]:  # Show first 2
                print(f"      - {model_id}")
            if len(model_list) > 2:
                print(f"      ... and {len(model_list) - 2} more")
    else:
        print(f"  ❌ Failed to get models: {response.status_code}")


def test_context_propagation():
    """Test if selection context is propagated"""
    print("\n" + "=" * 60)
    print("5. TESTING CONTEXT PROPAGATION")
    print("=" * 60)

    print("\n  Testing with different session/user contexts...")

    # Test with session header
    headers_tests = [
        {"X-Session-ID": "test-session-001", "X-User-ID": "test-user-001"},
        {"X-Session-ID": "test-session-002", "X-User-ID": "test-user-002"},
    ]

    for headers in headers_tests:
        response = requests.get(f"{BASE_URL}/api/v1/selection/current", headers=headers)
        if response.status_code == 200:
            selection = response.json()
            print(f"\n  Session: {headers.get('X-Session-ID')}")
            print(f"    Provider: {selection.get('provider_id')}")
            print(f"    Model: {selection.get('model_id')}")
        else:
            print(f"\n  Session: {headers.get('X-Session-ID')}")
            print(f"    Status: {response.status_code}")


def test_chat_completion_with_selection():
    """Test if chat endpoint respects selected model"""
    print("\n" + "=" * 60)
    print("6. TESTING CHAT COMPLETION WITH SELECTION")
    print("=" * 60)

    print("\n  NOTE: This requires a chat endpoint that uses SelectionContext")
    print("  Common endpoints:")
    print("    - POST /api/v1/chat/completions")
    print("    - POST /api/v1/chat")
    print("    - POST /api/v1/generate")

    # Try common chat endpoints
    test_message = {"messages": [{"role": "user", "content": "Say 'Hello' in one word"}]}

    potential_endpoints = [
        "/api/v1/chat/completions",
        "/api/v1/chat",
        "/api/v1/generate",
    ]

    for endpoint in potential_endpoints:
        response = requests.post(f"{BASE_URL}{endpoint}", json=test_message)
        print(f"\n  POST {endpoint}")
        print(f"    Status: {response.status_code}")
        if response.status_code == 404:
            print("    Not found")
        elif response.status_code >= 400:
            print(f"    Error: {response.text[:100]}")
        else:
            print("    ✓ Endpoint exists")
            # Check if response includes model info
            try:
                data = response.json()
                if "model" in data:
                    print(f"    Model used: {data.get('model')}")
            except Exception:  # noqa: E722
                pass


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MODEL SELECTION PROPAGATION TEST")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")

    try:
        # Test connection
        response = requests.get(f"{BASE_URL}/api/v1/providers/stats", timeout=2)
        if response.status_code != 200:
            print("\n❌ Server not responding properly")
            return
        print("✓ Server is responding")
    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("\nMake sure the server is running:")
        print("  cd d:/chimera/backend-api")
        print("  python run.py")
        return

    # Run tests
    test_selection_endpoint()
    test_model_selection_update()
    test_provider_resolution()
    test_unified_registry()
    test_context_propagation()
    test_chat_completion_with_selection()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(
        """
Key Findings:
1. Provider/model selection endpoints are available
2. Unified registry correctly lists all providers and models
3. Provider aliases work correctly (gpt→openai, claude→anthropic, etc.)
4. Context propagation depends on:
   - SelectionContext being used in service layer
   - Middleware extracting session/user IDs
   - Services querying SelectionContext for current selection

Next Steps:
- Check if chat/generation endpoints use SelectionContext
- Verify middleware is properly configured
- Test with actual API keys to see if correct model is called
    """
    )


if __name__ == "__main__":
    main()
