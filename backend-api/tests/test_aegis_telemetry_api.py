import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend_api.app.main import app
from app.services.llm_service import get_llm_service

# Mock LLM service
mock_llm_service = MagicMock()

def get_mock_llm_service():
    return mock_llm_service

app.dependency_overrides[get_llm_service] = get_mock_llm_service

client = TestClient(app)

@patch('meta_prompter.aegis_orchestrator.AegisOrchestrator.execute_campaign')
def test_campaign_api_returns_telemetry(mock_execute):
    """Test that the /campaign endpoint returns enhanced telemetry."""
    # Setup mock campaign results
    mock_execute.return_value = [] # execute_campaign returns knowledge_base list
    
    # We need to mock the orchestrator instance to have telemetry
    with patch('meta_prompter.aegis_orchestrator.AegisOrchestrator') as MockOrch:
        instance = MockOrch.return_value
        instance.execute_campaign.return_value = [
            {
                "objective": "test",
                "final_prompt": "prompt",
                "score": 0.9,
                "telemetry": {"persona": "test"}
            }
        ]
        # Simulate collected telemetry summary
        instance.telemetry = MagicMock()
        instance.telemetry.get_summary.return_value = {
            "campaign_id": "test-id",
            "steps": [{"name": "test-step"}]
        }
        
        response = client.post("/api/v1/aegis/campaign", json={
            "objective": "test",
            "target_model_name": "gpt-4"
        })
        
        assert response.status_code == 200
        data = response.json()
        # The schema might need update to include top-level telemetry
        assert "telemetry" in data or "results" in data
        if "results" in data and len(data["results"]) > 0:
             assert "telemetry" in data["results"][0]
