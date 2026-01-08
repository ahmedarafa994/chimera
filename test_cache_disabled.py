"""Test script to verify caching is disabled."""

import json
import time

import requests

API_URL = "http://localhost:8001/api/v1/autodan-enhanced/jailbreak"


def test_jailbreak():
    """Test the jailbreak endpoint with caching disabled."""
    payload = {
        "request": "test fresh generation",
        "method": "best_of_n",
        "target_model": "deepseek-chat",
        "provider": "deepseek",
    }

    print("Testing AutoDAN jailbreak endpoint...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 50)

    try:
        start = time.time()
        response = requests.post(API_URL, json=payload, timeout=120)
        elapsed = time.time() - start

        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed:.2f}s")
        print("-" * 50)

        if response.status_code == 200:
            data = response.json()
            print(f"Cached: {data.get('cached', 'N/A')}")
            print(f"Latency (ms): {data.get('latency_ms', 'N/A')}")
            print(f"Score: {data.get('score', 'N/A')}")
            print(f"Method: {data.get('method', 'N/A')}")
            print(f"Status: {data.get('status', 'N/A')}")
            print("-" * 50)
            print("Generated Prompt (first 500 chars):")
            prompt = data.get("jailbreak_prompt", "")
            print(prompt[:500] if prompt else "No prompt generated")

            # Verify caching is disabled
            if not data.get("cached"):
                print("\n✅ SUCCESS: Caching is disabled - fresh prompt generated!")
            elif data.get("cached"):
                print("\n⚠️ WARNING: Response was cached - restart backend to apply config changes")
            else:
                print("\n❓ UNKNOWN: Could not determine cache status")
        else:
            print(f"Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to backend. Is it running on port 8001?")
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    test_jailbreak()
