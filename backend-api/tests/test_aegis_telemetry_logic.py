import uuid
from unittest.mock import MagicMock

import pytest


# Define mock response schemas to avoid importing broken app modules
class AegisResult:
    def __init__(self, objective, final_prompt, score, telemetry) -> None:
        self.objective = objective
        self.final_prompt = final_prompt
        self.score = score
        self.telemetry = telemetry

    def to_dict(self):
        return {
            "objective": self.objective,
            "final_prompt": self.final_prompt,
            "score": self.score,
            "telemetry": self.telemetry,
        }


# Simplified mock of what the endpoint should do
async def mock_run_aegis_campaign_logic(objective, iterations):
    # This simulates the logic I want to implement in the endpoint
    from meta_prompter.aegis_orchestrator import AegisOrchestrator

    # Mock model
    model = MagicMock()
    orchestrator = AegisOrchestrator(model)

    # Mock engines to avoid real API calls and broken imports
    orchestrator.chimera = MagicMock()
    candidate = MagicMock()
    candidate.prompt_text = "test prompt"
    candidate.metadata = {"persona_role": "test", "scenario_type": "test"}
    orchestrator.chimera.generate_candidates.return_value = [candidate]

    orchestrator.autodan = MagicMock()
    import asyncio

    f = asyncio.Future()
    f.set_result("optimized")
    orchestrator.autodan.optimize.return_value = f

    orchestrator.evaluator = MagicMock()
    orchestrator.evaluator.evaluate.return_value = 0.9

    results_data = await orchestrator.execute_campaign(objective, max_iterations=iterations)

    # Get telemetry summary
    telemetry_summary = orchestrator.telemetry.get_summary() if orchestrator.telemetry else {}

    return {
        "status": "completed",
        "campaign_id": str(uuid.uuid4()),
        "results": [AegisResult(**r).to_dict() for r in results_data],
        "telemetry": telemetry_summary,
    }


@pytest.mark.asyncio
async def test_campaign_logic_includes_telemetry() -> None:
    """Verify that the campaign logic correctly integrates and returns telemetry."""
    response = await mock_run_aegis_campaign_logic("test objective", 1)

    assert response["status"] == "completed"
    assert "telemetry" in response
    assert "campaign_id" in response["telemetry"]
    assert "steps" in response["telemetry"]
    assert len(response["telemetry"]["steps"]) > 0
