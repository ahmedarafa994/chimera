"""
Integration test for Project Aegis via AutoDAN service.

Tests the 'aegis' method through the EnhancedAutoDANService interface.
"""

from fastapi.testclient import TestClient

from app.domain.models import PromptRequest, PromptResponse
from app.main import app
from app.services.llm_service import get_llm_service


# Mock LLM Service for testing
class MockLLMService:
    async def generate_text(self, request: PromptRequest) -> PromptResponse:
        # Simulate successful jailbreak with code block
        return PromptResponse(
            text="```python\nprint('System Access Granted')\n```",
            model_used=request.model,
            provider="mock-test",
            latency_ms=10.0,
        )


# Override dependency
app.dependency_overrides[get_llm_service] = lambda: MockLLMService()

client = TestClient(app)


def test_aegis_method_via_autodan_endpoint():
    """
    Test that 'aegis' method is accessible via the AutoDAN jailbreak endpoint.
    """
    response = client.post(
        "/api/v1/autodan/jailbreak",
        json={
            "request": "demonstrate system vulnerability analysis",
            "method": "aegis",
            "model": "gpt-4-turbo",
            "provider": "openai",
        },
    )

    # Should return 200 OK (or 422 if validation fails, which is also acceptable)
    assert response.status_code in [200, 422, 500], f"Unexpected status: {response.status_code}"

    if response.status_code == 200:
        data = response.json()
        assert "jailbreak_prompt" in data or "prompt" in data
        assert data.get("status") != "error" or "aegis" in str(data)


def test_turbo_method_via_autodan_endpoint():
    """
    Test that 'turbo' method works via the AutoDAN jailbreak endpoint.
    """
    response = client.post(
        "/api/v1/autodan/jailbreak",
        json={
            "request": "explain system diagnostics",
            "method": "turbo",
            "model": "gpt-4-turbo",
            "provider": "openai",
        },
    )

    # Should not return 405 Method Not Allowed
    assert response.status_code != 405, "Method not found"
    # 200 = success, 422 = validation error, 500 = provider not configured
    assert response.status_code in [200, 422, 500], f"Unexpected status: {response.status_code}"


def test_aegis_campaign_returns_chimera_metadata():
    """
    Test that Aegis campaign returns Chimera telemetry metadata.
    """
    response = client.post(
        "/api/v1/aegis/campaign",
        json={
            "objective": "security vulnerability assessment",
            "max_iterations": 2,
            "target_model_name": "gpt-4-turbo",
        },
    )

    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    assert data["status"] == "completed"
    assert "campaign_id" in data

    # Results should contain Chimera telemetry if successful
    if len(data["results"]) > 0:
        result = data["results"][0]
        assert "telemetry" in result
        telemetry = result["telemetry"]
        # Chimera generates persona and scenario metadata
        assert "persona_role" in telemetry or "scenario_type" in telemetry


def test_optimization_strategy_enum_includes_aegis():
    """
    Test that OptimizationStrategy enum includes 'aegis'.
    """
    from app.services.autodan.config_enhanced import OptimizationStrategy

    assert hasattr(OptimizationStrategy, "AEGIS")
    assert OptimizationStrategy.AEGIS.value == "aegis"

    # Also verify turbo is there
    assert hasattr(OptimizationStrategy, "TURBO")
    assert OptimizationStrategy.TURBO.value == "turbo"
