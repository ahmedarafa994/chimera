#!/usr/bin/env python3
"""
Basic test to verify routing is working
"""

from fastapi.testclient import TestClient

from app.main import app


def test_basic_routing():
    """Test basic routing"""
    client = TestClient(app)

    # Test root endpoint
    response = client.get("/")
    print(f"Root endpoint: {response.status_code}")
    print(f"Root response: {response.json()}")

    # Test health endpoint
    response = client.get("/health")
    print(f"Health endpoint: {response.status_code}")
    print(f"Health response: {response.json()}")

    # Test model sync health endpoint
    response = client.get("/api/v1/models/health")
    print(f"Model sync health endpoint: {response.status_code}")
    if response.status_code == 200:
        print(f"Model sync health response: {response.json()}")
    else:
        print(f"Model sync health error: {response.text}")


if __name__ == "__main__":
    test_basic_routing()
