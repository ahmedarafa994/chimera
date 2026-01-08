#!/usr/bin/env python3
"""
Simple test script to verify DeepSeek API connectivity and model availability.
Run from project root: python scripts/test_deepseek_api.py
"""

import os
import sys

# Add backend-api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend-api"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend-api", ".env"))


def test_deepseek_api():
    """Test DeepSeek API with a simple request."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        return False

    # Get configuration
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DIRECT_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    print("Configuration:")
    print(f"  API Key: {api_key[:10]}..." if api_key else "  API Key: NOT SET")
    print(f"  Model: {model}")
    print(f"  Base URL: {base_url}")
    print()

    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set in environment")
        return False

    # Create client
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Testing DeepSeek API with model '{model}'...")
    print()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
            temperature=0.1,
            max_tokens=50,
            stream=False,
        )

        if response and response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            print(f"SUCCESS! Response: {content}")
            print()
            print(f"Model used: {response.model}")
            print(f"Usage: {response.usage}")
            return True
        else:
            print("ERROR: Empty response from API")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check if DEEPSEEK_API_KEY is valid")
        print("  2. Check if DEEPSEEK_MODEL is a valid model name")
        print("     Valid models: deepseek-chat, deepseek-reasoner, deepseek-coder")
        print("  3. Check if your API key has access to the specified model")
        return False


def test_available_models():
    """List available models (if API supports it)."""
    try:
        from openai import OpenAI
    except ImportError:
        return

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DIRECT_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    if not api_key:
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("\nAttempting to list available models...")
    try:
        models = client.models.list()
        print("Available models:")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e:
        print(f"Could not list models: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek API Test Script")
    print("=" * 60)
    print()

    success = test_deepseek_api()
    test_available_models()

    print()
    print("=" * 60)
    sys.exit(0 if success else 1)
