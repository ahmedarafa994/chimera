"""Model selection endpoints for managing user's selected AI model and provider."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator

from app.core.logging import logger
from app.services.model_selection_service import model_selection_service

router = APIRouter()


class ModelSelectionRequest(BaseModel):
    """Request to set model selection."""

    provider: str
    model: str

    model_config = ConfigDict(protected_namespaces=())

    @field_validator('provider', 'model')
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        """Validate that provider and model are non-empty strings."""
        if not v or not v.strip():
            raise ValueError(f'{info.field_name} cannot be empty')
        return v.strip()


class ModelSelectionResponse(BaseModel):
    """Response with current model selection."""

    provider: str | None = None
    model: str | None = None

    model_config = ConfigDict(protected_namespaces=())


@router.get("/model-selection", response_model=ModelSelectionResponse)
async def get_model_selection():
    """Get the current selected model and provider."""
    selection = model_selection_service.get_selection()
    if selection:
        return ModelSelectionResponse(provider=selection.provider, model=selection.model)
    return ModelSelectionResponse()


@router.post("/model-selection", response_model=ModelSelectionResponse)
async def set_model_selection(request: ModelSelectionRequest):
    """Set the current selected model and provider."""
    try:
        selection = model_selection_service.set_selection(request.provider, request.model)
        logger.info(f"Model selection updated: {selection.provider}/{selection.model}")
        return ModelSelectionResponse(provider=selection.provider, model=selection.model)
    except Exception as e:
        logger.error(f"Failed to set model selection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set model selection: {e!s}",
        )


@router.delete("/model-selection")
async def clear_model_selection():
    """Clear the current model selection."""
    model_selection_service.clear_selection()
    logger.info("Model selection cleared")
    return {"message": "Model selection cleared"}
