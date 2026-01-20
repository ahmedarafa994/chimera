#!/usr/bin/env python3
"""Test multiple model sync endpoints."""

from fastapi.testclient import TestClient

from app.main import app


def test_multiple_endpoints() -> None:
    """Test multiple model sync endpoints."""
    client = TestClient(app)

    # Test health endpoint
    response = client.get("/api/v1/models/health")
    if response.status_code == 200:
        response.json()

    # Test available models endpoint
    response = client.get("/api/v1/models/available")
    if response.status_code == 200:
        response.json()
    else:
        pass

    # Test validate endpoint (this might fail due to auth, but let's see)
    response = client.post("/api/v1/models/validate/gemini-2.5-pro")
    if response.status_code == 200:
        response.json()
    else:
        pass


if __name__ == "__main__":
    test_multiple_endpoints()
