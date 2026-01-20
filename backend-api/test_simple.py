"""Simple Model Selection Test.

Quick test to verify model selection context works
"""

import time

import requests

BASE_URL = "http://localhost:8001"


def wait_for_server(max_attempts=20) -> bool:
    """Wait for server to be ready."""
    for _i in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/api/v1/providers/stats", timeout=1)
            if response.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def main() -> None:
    if not wait_for_server():
        return

    # Test 1: Get stats
    r = requests.get(f"{BASE_URL}/api/v1/providers/stats")
    if r.status_code == 200:
        r.json()

    # Test 2: Get current selection
    r = requests.get(f"{BASE_URL}/api/v1/selection/current")
    if r.status_code == 200:
        r.json()
    else:
        pass

    # Test 3: List updated models
    r = requests.get(f"{BASE_URL}/api/v1/providers/openai/models")
    if r.status_code == 200:
        models = r.json()
        for _m in models[:3]:
            pass

    r = requests.get(f"{BASE_URL}/api/v1/providers/anthropic/models")
    if r.status_code == 200:
        models = r.json()
        for _m in models[:3]:
            pass

    r = requests.get(f"{BASE_URL}/api/v1/providers/deepseek/models")
    if r.status_code == 200:
        models = r.json()
        for _m in models[:3]:
            pass


if __name__ == "__main__":
    main()
