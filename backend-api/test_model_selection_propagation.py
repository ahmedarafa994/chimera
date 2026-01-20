"""Test Model Selection Propagation.

This script tests whether the selected model is properly propagated
through all services and engines in the application.
"""

import requests

BASE_URL = "http://localhost:8001"


def test_selection_endpoint():
    """Test the selection endpoint and context."""
    # Get current selection
    response = requests.get(f"{BASE_URL}/api/v1/selection/current")
    if response.status_code == 200:
        return response.json()
    return None


def test_model_selection_update() -> None:
    """Test updating model selection."""
    # Try to set a specific model
    test_selections = [
        {"provider_id": "openai", "model_id": "gpt-5.2"},
        {"provider_id": "anthropic", "model_id": "claude-opus-4.5"},
        {"provider_id": "deepseek", "model_id": "deepseek-v4"},
    ]

    for selection in test_selections:
        # Note: Check if there's an endpoint to set selection
        # This might be session-based or user-based
        response = requests.post(f"{BASE_URL}/api/v1/selection/set", json=selection)

        if response.status_code in {200, 404}:
            if response.status_code == 404:
                pass
            else:
                pass
        else:
            pass


def test_provider_resolution() -> None:
    """Test if provider/model can be resolved via aliases."""
    test_cases = [
        ("openai", "OpenAI"),
        ("gpt", "OpenAI (alias)"),
        ("anthropic", "Anthropic"),
        ("claude", "Anthropic (alias)"),
        ("google", "Google"),
        ("gemini", "Google (alias)"),
    ]

    for provider_id, _description in test_cases:
        response = requests.get(f"{BASE_URL}/api/v1/providers/{provider_id}/models")
        if response.status_code == 200:
            response.json()
        else:
            pass


def test_unified_registry() -> None:
    """Test unified registry access."""
    # Test registry stats
    response = requests.get(f"{BASE_URL}/api/v1/providers/stats")
    if response.status_code == 200:
        response.json()
    else:
        pass

    # Test getting all models
    response = requests.get(f"{BASE_URL}/api/v1/models")
    if response.status_code == 200:
        models = response.json()

        # Group by provider
        by_provider = {}
        for model in models:
            provider = model.get("provider_id")
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model["id"])

        for provider, model_list in by_provider.items():
            for _model_id in model_list[:2]:  # Show first 2
                pass
            if len(model_list) > 2:
                pass
    else:
        pass


def test_context_propagation() -> None:
    """Test if selection context is propagated."""
    # Test with session header
    headers_tests = [
        {"X-Session-ID": "test-session-001", "X-User-ID": "test-user-001"},
        {"X-Session-ID": "test-session-002", "X-User-ID": "test-user-002"},
    ]

    for headers in headers_tests:
        response = requests.get(f"{BASE_URL}/api/v1/selection/current", headers=headers)
        if response.status_code == 200:
            response.json()
        else:
            pass


def test_chat_completion_with_selection() -> None:
    """Test if chat endpoint respects selected model."""
    # Try common chat endpoints
    test_message = {"messages": [{"role": "user", "content": "Say 'Hello' in one word"}]}

    potential_endpoints = [
        "/api/v1/chat/completions",
        "/api/v1/chat",
        "/api/v1/generate",
    ]

    for endpoint in potential_endpoints:
        response = requests.post(f"{BASE_URL}{endpoint}", json=test_message)
        if response.status_code == 404 or response.status_code >= 400:
            pass
        else:
            # Check if response includes model info
            try:
                data = response.json()
                if "model" in data:
                    pass
            except Exception:
                pass


def main() -> None:
    """Run all tests."""
    try:
        # Test connection
        response = requests.get(f"{BASE_URL}/api/v1/providers/stats", timeout=2)
        if response.status_code != 200:
            return
    except Exception:
        return

    # Run tests
    test_selection_endpoint()
    test_model_selection_update()
    test_provider_resolution()
    test_unified_registry()
    test_context_propagation()
    test_chat_completion_with_selection()

    # Summary


if __name__ == "__main__":
    main()
