"""Tests for system health endpoint."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_system_health_endpoint(client: AsyncClient):
    """Test system health endpoint returns correct structure."""
    response = await client.get("/api/v1/system-health")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "status" in data
    assert "services" in data
    assert "active_techniques" in data
    assert "api_version" in data

    # Verify services
    assert "database" in data["services"]
    assert "redis" in data["services"]
    assert "llm_providers" in data["services"]

    # Verify techniques list
    assert isinstance(data["active_techniques"], list)
    assert len(data["active_techniques"]) > 0

    # Verify API version
    assert data["api_version"] == "1.0.0"


@pytest.mark.asyncio
async def test_system_health_status_values(client: AsyncClient):
    """Test system health status has valid values."""
    response = await client.get("/api/v1/system-health")

    assert response.status_code == 200
    data = response.json()

    # Status should be either healthy or degraded
    assert data["status"] in ["healthy", "degraded"]

    # Service statuses should be valid
    for _, status in data["services"].items():
        assert status in ["connected", "disconnected", "available", "unavailable"]
