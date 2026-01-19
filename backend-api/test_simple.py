"""
Simple Model Selection Test

Quick test to verify model selection context works
"""

import time

import requests

BASE_URL = "http://localhost:8001"


def wait_for_server(max_attempts=20):
    """Wait for server to be ready"""
    print("Waiting for server...")
    for i in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/api/v1/providers/stats", timeout=1)
            if response.status_code == 200:
                print(f"✓ Server ready after {i+1} attempts")
                return True
        except:
            time.sleep(1)
    print("✗ Server not responding")
    return False


def main():
    if not wait_for_server():
        return

    print("\n" + "=" * 60)
    print("MODEL SELECTION TEST")
    print("=" * 60)

    # Test 1: Get stats
    print("\n1. Provider Stats:")
    r = requests.get(f"{BASE_URL}/api/v1/providers/stats")
    if r.status_code == 200:
        stats = r.json()
        print(f"   Providers: {stats['total_providers']}")
        print(f"   Models: {stats['total_models']}")

    # Test 2: Get current selection
    print("\n2. Current Selection:")
    r = requests.get(f"{BASE_URL}/api/v1/selection/current")
    if r.status_code == 200:
        sel = r.json()
        print(f"   Provider: {sel.get('provider_id')}")
        print(f"   Model: {sel.get('model_id')}")
    else:
        print(f"   Status: {r.status_code}")

    # Test 3: List updated models
    print("\n3. OpenAI Models (should show GPT-5.2):")
    r = requests.get(f"{BASE_URL}/api/v1/providers/openai/models")
    if r.status_code == 200:
        models = r.json()
        for m in models[:3]:
            print(f"   - {m['id']}: {m['name']}")

    print("\n4. Anthropic Models (should show Claude 4.5):")
    r = requests.get(f"{BASE_URL}/api/v1/providers/anthropic/models")
    if r.status_code == 200:
        models = r.json()
        for m in models[:3]:
            print(f"   - {m['id']}: {m['name']}")

    print("\n5. DeepSeek Models (should show V4):")
    r = requests.get(f"{BASE_URL}/api/v1/providers/deepseek/models")
    if r.status_code == 200:
        models = r.json()
        for m in models[:3]:
            print(f"   - {m['id']}: {m['name']}")

    print("\n" + "=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
