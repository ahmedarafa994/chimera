import pytest
from fastapi import status
from app.core.unified_errors import (
    AegisError,
    AegisSynthesisError,
    AegisTransformationError,
    ChimeraError,
    ServiceError
)

def test_aegis_error_inheritance():
    """Test that AegisError inherits from ServiceError."""
    err = AegisError("Test error")
    assert isinstance(err, ServiceError)
    assert isinstance(err, ChimeraError)
    # ServiceError constructs error code as SERVICE_{service_name.upper()}_ERROR
    # If service_name is "aegis", it should be SERVICE_AEGIS_ERROR
    assert err.error_code == "SERVICE_AEGIS_ERROR"
    assert err.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

def test_aegis_synthesis_error():
    """Test AegisSynthesisError properties."""
    err = AegisSynthesisError("Synthesis failed", details={"persona": "dark_triad"})
    assert isinstance(err, AegisError)
    # Synthesis failure is often data related, so 422 or 400 is appropriate. 
    # Let's assume we want 422 by default for synthesis issues based on input
    assert err.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert err.details["persona"] == "dark_triad"

def test_aegis_transformation_error():
    """Test AegisTransformationError properties."""
    err = AegisTransformationError("Transformation failed", status_code=status.HTTP_400_BAD_REQUEST)
    assert isinstance(err, AegisError)
    assert err.status_code == status.HTTP_400_BAD_REQUEST
