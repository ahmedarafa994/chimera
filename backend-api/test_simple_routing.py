#!/usr/bin/env python3
"""
Simple test to verify routing is working
"""

from fastapi.testclient import TestClient

from app.main import app


def test_simple_routing():
    """Test simple routing"""
    client = TestClient(app)

    # Test model sync health endpoint
    response = client.get("/api/v1/models/health")
    print(f"Model sync health endpoint: {response.status_code}")
    if response.status_code == 200:
        print(f"Model sync health response: {response.json()}")
    else:
        print(f"Model sync health error: {response.text}")


if __name__ == "__main__":
    test_simple_routing()
