#!/usr/bin/env python3
"""
Test multiple model sync endpoints
"""

from fastapi.testclient import TestClient

from app.main import app


def test_multiple_endpoints():
    """Test multiple model sync endpoints"""
    client = TestClient(app)

    # Test health endpoint
    response = client.get("/api/v1/models/health")
    print(f"Health endpoint: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(
            f"Health data: status={data['status']}, models_count={data['models_count']}, version={data['version']}"
        )

    # Test available models endpoint
    response = client.get("/api/v1/models/available")
    print(f"Available models endpoint: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Available models: count={data['count']}, cache_ttl={data['cache_ttl']}")
        print(f"First 3 models: {[model['id'] for model in data['models'][:3]]}")
    else:
        print(f"Available models error: {response.text}")

    # Test validate endpoint (this might fail due to auth, but let's see)
    response = client.post("/api/v1/models/validate/gemini-2.5-pro")
    print(f"Validate endpoint: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Validation result: is_valid={data['is_valid']}, model_id={data['model_id']}")
    else:
        print(f"Validate error: {response.text}")


if __name__ == "__main__":
    test_multiple_endpoints()
