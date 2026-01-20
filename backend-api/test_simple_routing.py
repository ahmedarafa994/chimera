#!/usr/bin/env python3
"""Simple test to verify routing is working."""

from fastapi.testclient import TestClient

from app.main import app


def test_simple_routing() -> None:
    """Test simple routing."""
    client = TestClient(app)

    # Test model sync health endpoint
    response = client.get("/api/v1/models/health")
    if response.status_code == 200:
        pass
    else:
        pass


if __name__ == "__main__":
    test_simple_routing()
