from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import settings
from app.domain.models import PromptResponse
from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    # The middleware excludes specific paths, ensure we hit the right one
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    # Updated expectation to match implementation in routes.py
    assert response.json()["status"] == "healthy"


@patch("app.infrastructure.gemini_client.GeminiClient.generate")
def test_generate_text(mock_generate) -> None:
    # Mock response
    mock_response = PromptResponse(
        text="Test response",
        model_used="gemini-1.5-pro",
        provider="google",
        usage_metadata={},
        finish_reason="STOP",
        latency_ms=100.0,
    )

    # Configure the mock to return the response object
    mock_generate.return_value = mock_response

    request_data = {"prompt": "Hello", "provider": "google", "config": {"temperature": 0.5}}

    # Add API Key header to pass authentication
    # Use valid key
    valid_key = settings.CHIMERA_API_KEY or "chimera_default_key_change_in_production"
    headers = {"X-API-Key": valid_key}

    response = client.post("/api/v1/generate", json=request_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Test response"
    assert data["provider"] == "google"
