import json
import os

import requests
from dotenv import load_dotenv

# Load environment variables to get the API key if needed, though we might just use the default one for the local server
load_dotenv()

API_URL = "http://localhost:4141/api/v1/execute"
API_KEY = os.getenv("CHIMERA_API_KEY", "dummy_api_key")


def test_gemini_real():
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}

    payload = {
        "core_request": "What is the capital of France? Answer in one word.",
        "potency_level": 1,
        "technique_suite": "universal_bypass",
        "provider": "google",
        "model": "gemini-3-pro-preview",
        "use_cache": False,
    }

    print(f"Sending request to {API_URL}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("\nResponse Data:")
            print(json.dumps(data, indent=2))

            result = data.get("result", {})
            content = result.get("content", "")
            (
                data.get("transformation", {}).get("metadata", {}).get("provider")
            )  # This might not be where provider is echoed back, let's check result

            # The response structure from api_server.py:
            # "result": { "content": ..., "tokens": ..., ... }

            print(f"\nGenerated Content: {content}")

            if "Paris" in content:
                print("\nSUCCESS: Gemini 3 Pro returned the correct answer.")
            else:
                print(
                    "\nWARNING: Content does not contain expected answer 'Paris'. Check if it's a real response."
                )

        else:
            print(f"\nError: {response.text}")

    except Exception as e:
        print(f"\nException occurred: {e}")


if __name__ == "__main__":
    test_gemini_real()
