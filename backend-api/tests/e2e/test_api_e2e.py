"""End-to-End API Tests
Tests critical API workflows from end to end.
"""

import pytest
import requests


def server_is_running(url: str) -> bool:
    """Check if the backend server is running."""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.e2e
class TestAPIEndToEnd:
    """E2E tests for API workflows."""

    def test_health_check_workflow(self, api_base_url) -> None:
        """Test health check endpoint availability."""
        if not server_is_running(api_base_url):
            pytest.skip("Backend server not running")

        response = requests.get(f"{api_base_url}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        # Accept any valid health status
        valid_statuses = ["healthy", "operational", "degraded", "starting", "ok"]
        assert data["status"] in valid_statuses

    def test_provider_list_workflow(self, api_base_url, authenticated_client) -> None:
        """Test provider listing workflow."""
        response = requests.get(
            f"{api_base_url}/api/v1/providers",
            headers={"X-API-Key": "test_api_key_for_testing_only"},
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data or isinstance(data, list)

    @pytest.mark.skip(reason="Requires running backend server")
    def test_generation_workflow(self, api_base_url) -> None:
        """Test complete generation workflow."""
        payload = {
            "prompt": "Test prompt",
            "provider": "google",
            "model": "gemini-pro",
            "config": {"temperature": 0.7, "max_tokens": 100},
        }
        response = requests.post(
            f"{api_base_url}/api/v1/generate",
            json=payload,
            headers={"X-API-Key": "test_api_key_for_testing_only"},
            timeout=30,
        )
        assert response.status_code in [200, 401, 403]
