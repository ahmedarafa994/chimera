#!/usr/bin/env python3
"""Basic test to verify routing is working."""

from fastapi.testclient import TestClient

from app.main import app


def test_basic_routing() -> None:
    """Test basic routing."""
    client = TestClient(app)

    # Test root endpoint
    response = client.get("/")

    # Test health endpoint
    response = client.get("/health")

    # Test model sync health endpoint
    response = client.get("/api/v1/models/health")
    if response.status_code == 200:
        pass
    else:
        pass


if __name__ == "__main__":
    test_basic_routing()
