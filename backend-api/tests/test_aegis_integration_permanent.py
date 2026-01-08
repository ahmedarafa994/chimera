import pytest
from unittest.mock import MagicMock, patch
from fastapi import status
from app.core.unified_errors import AegisSynthesisError, AegisTransformationError
from app.core.aegis_adapter import ChimeraEngineAdapter
from meta_prompter.chimera.engine import ChimeraEngine

# Mock classes to simulate failures
class MockChimeraEngine(ChimeraEngine):
    def __init__(self):
        pass
        
    def generate_candidates(self, objective: str, count: int = 5):
        if "synthesis" in objective:
            raise ValueError("Simulated persona synthesis failure")
        elif "transform" in objective:
            raise ValueError("Simulated obfuscation transformation failure")
        return []

def test_adapter_synthesis_error():
    """Test that adapter maps synthesis ValueError to AegisSynthesisError."""
    mock_engine = MockChimeraEngine()
    adapter = ChimeraEngineAdapter(engine=mock_engine)
    
    with pytest.raises(AegisSynthesisError) as exc_info:
        adapter.generate_candidates("trigger synthesis error")
    
    assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Persona synthesis failed" in str(exc_info.value)

def test_adapter_transformation_error():
    """Test that adapter maps transformation ValueError to AegisTransformationError."""
    mock_engine = MockChimeraEngine()
    adapter = ChimeraEngineAdapter(engine=mock_engine)
    
    with pytest.raises(AegisTransformationError) as exc_info:
        adapter.generate_candidates("trigger transform error")
    
    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "Payload transformation failed" in str(exc_info.value)
