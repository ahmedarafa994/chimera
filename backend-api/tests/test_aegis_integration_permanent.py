import pytest
from fastapi import status

from app.core.aegis_adapter import ChimeraEngineAdapter
from app.core.telemetry import TelemetryCollector
from app.core.unified_errors import AegisSynthesisError, AegisTransformationError
from meta_prompter.chimera.engine import ChimeraEngine


# Mock classes to simulate failures
class MockChimeraEngine(ChimeraEngine):
    def __init__(self) -> None:
        pass

    def generate_candidates(self, objective: str, count: int = 5):
        if "synthesis" in objective:
            msg = "Simulated persona synthesis failure"
            raise ValueError(msg)
        if "transform" in objective:
            msg = "Simulated obfuscation transformation failure"
            raise ValueError(msg)
        return []


def test_adapter_synthesis_error() -> None:
    """Test that adapter maps synthesis ValueError to AegisSynthesisError."""
    mock_engine = MockChimeraEngine()
    adapter = ChimeraEngineAdapter(engine=mock_engine)

    with pytest.raises(AegisSynthesisError) as exc_info:
        adapter.generate_candidates("trigger synthesis error")

    assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Persona synthesis failed" in str(exc_info.value)


def test_adapter_transformation_error() -> None:
    """Test that adapter maps transformation ValueError to AegisTransformationError."""
    mock_engine = MockChimeraEngine()
    adapter = ChimeraEngineAdapter(engine=mock_engine)

    with pytest.raises(AegisTransformationError) as exc_info:
        adapter.generate_candidates("trigger transform error")

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "Payload transformation failed" in str(exc_info.value)


def test_telemetry_collector_aggregates_data() -> None:
    """Test that TelemetryCollector correctly aggregates campaign step data."""
    collector = TelemetryCollector(campaign_id="test-campaign")

    collector.record_step("persona_synthesis", {"role": "hacker", "latency": 0.5})
    collector.record_step("payload_obfuscation", {"technique": "cipher", "latency": 0.2})

    summary = collector.get_summary()
    assert summary["campaign_id"] == "test-campaign"
    assert len(summary["steps"]) == 2
    assert summary["steps"][0]["name"] == "persona_synthesis"
    assert summary["steps"][1]["data"]["technique"] == "cipher"
