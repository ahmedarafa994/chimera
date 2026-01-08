"""
Model synchronization API endpoints.

INT-003 FIX: Added missing endpoints to align with frontend expectations:
- GET /models - List all models (alias for /models/available)
- POST /models/validate - Validate model selection
"""

import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.domain.model_sync import (
    ModelAvailabilityUpdate,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from app.middleware.auth import get_client_info, get_current_user_id
from app.services.model_sync_service import model_sync_service

# Removed prefix="/models" since it's already included in v1 router
router = APIRouter(tags=["model-sync"])


# Request/Response models for frontend compatibility
class ValidateModelRequest(BaseModel):
    """Request to validate a model selection."""

    provider: str = Field(..., description="Provider name (e.g., 'google', 'openai')")
    model: str = Field(..., description="Model ID to validate")


class ValidateModelResponse(BaseModel):
    """Response from model validation."""

    valid: bool
    message: str
    fallback_model: str | None = None
    fallback_provider: str | None = None


class ModelInfo(BaseModel):
    """Detailed model information."""

    id: str
    name: str
    provider: str
    description: str | None = None
    max_tokens: int | None = None
    supports_streaming: bool = True
    supports_vision: bool = False
    is_default: bool = False
    tier: str = "standard"


class ProviderWithModels(BaseModel):
    """Provider with its available models."""

    provider: str
    status: str
    model: str | None = None
    available_models: list[str]
    models_detail: list[ModelInfo] = []


class ModelsListResponse(BaseModel):
    """Response for listing all models."""

    providers: list[ProviderWithModels]
    default_provider: str
    default_model: str
    total_models: int


# INT-003 FIX: Add root GET endpoint for /models to list all models
@router.get("", response_model=ModelsListResponse)
async def list_models():
    """
    List all available models across all providers.

    This endpoint provides a comprehensive list of all models
    organized by provider, matching the frontend's expected format.
    """
    try:
        available = await model_sync_service.get_available_models()

        # Build provider list by grouping models by provider
        providers_dict: dict[str, list[str]] = {}
        models_by_provider: dict[str, list[ModelInfo]] = {}

        for model in available.models:
            provider_id = model.provider
            if provider_id not in providers_dict:
                providers_dict[provider_id] = []
                models_by_provider[provider_id] = []
            providers_dict[provider_id].append(model.name)
            models_by_provider[provider_id].append(
                ModelInfo(
                    id=model.id,
                    name=model.name,
                    provider=model.provider,
                    description=model.description,
                    max_tokens=model.max_tokens,
                    is_default=(model.name == model_sync_service.default_model),
                )
            )

        # Convert to ProviderWithModels list
        providers_list = []
        total_models = 0

        for provider_id, model_names in providers_dict.items():
            providers_list.append(
                ProviderWithModels(
                    provider=provider_id,
                    status="active",  # Assume active if models exist
                    model=model_names[0] if model_names else None,
                    available_models=model_names,
                    models_detail=models_by_provider.get(provider_id, []),
                )
            )
            total_models += len(model_names)

        # Get default provider and model from settings
        from app.core.config import settings

        default_provider = settings.AI_PROVIDER
        default_model = settings.DEFAULT_MODEL_ID

        return ModelsListResponse(
            providers=providers_list,
            default_provider=default_provider,
            default_model=default_model,
            total_models=total_models,
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        # Return empty response on error with settings defaults
        from app.core.config import settings

        return ModelsListResponse(
            providers=[],
            default_provider=settings.AI_PROVIDER,
            default_model=settings.DEFAULT_MODEL_ID,
            total_models=0,
        )


@router.get("/available", response_model_exclude_none=True)
async def get_available_models():
    """
    Get available models with 5-minute cache TTL.
    Public endpoint for frontend initialization.
    """
    # Service now raises ChimeraError on failure, handled by global handler
    return await model_sync_service.get_available_models()


# INT-003 FIX: Add POST /validate endpoint for model validation
@router.post("/validate", response_model=ValidateModelResponse)
async def validate_model_selection(request: ValidateModelRequest):
    """
    Validate if a provider/model combination is valid.

    Returns validation status and fallback suggestions if invalid.
    """
    from app.core.config import settings

    try:
        # Build the full model ID as stored in the cache (provider:model_name)
        model_id = f"{request.provider}:{request.model}"

        # Check if this provider:model combination exists
        is_valid = await model_sync_service.validate_model_id(model_id)

        if is_valid:
            return ValidateModelResponse(
                valid=True,
                message=f"Model '{request.model}' is valid for provider '{request.provider}'",
            )
        else:
            # Get default fallback from settings
            default_provider = settings.AI_PROVIDER
            default_model = settings.DEFAULT_MODEL_ID
            return ValidateModelResponse(
                valid=False,
                message=f"Model '{request.model}' is not available for provider '{request.provider}'",
                fallback_model=default_model,
                fallback_provider=default_provider,
            )
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        from app.core.config import settings

        return ValidateModelResponse(
            valid=False,
            message=f"Validation error: {e!s}",
            fallback_model=settings.DEFAULT_MODEL_ID,
            fallback_provider=settings.AI_PROVIDER,
        )


@router.post("/session/model", response_model=ModelSelectionResponse)
async def select_model(
    request: ModelSelectionRequest,
    http_request: Request,
    user_id: str = Depends(get_current_user_id),
    client_info: dict = Depends(get_client_info),
):
    """
    Select a model for the user's session.
    Requires authentication via JWT token or API key.
    """
    # Extract session ID from request or generate new one
    session_id = http_request.headers.get("X-Session-ID") or str(uuid.uuid4())

    response = await model_sync_service.select_model(
        request=request,
        user_id=user_id,
        session_id=session_id,
        ip_address=client_info.get("ip_address", "unknown"),
        user_agent=client_info.get("user_agent", "unknown"),
    )

    # Log model selection event for GDPR compliance
    logger.info(
        f"Model selected: user_id={user_id}, "
        f"model_id={request.model_id}, "
        f"session_id={session_id}, "
        f"ip_address={client_info.get('ip_address', 'unknown')}, "
        f"timestamp={request.timestamp}"
    )

    return response


@router.get("/session/current")
async def get_current_model(
    user_id: str = Depends(get_current_user_id), client_info: dict = Depends(get_client_info)
):
    """
    Get the currently selected model for the user's session.
    Requires authentication.
    """
    # Extract session ID from request
    session_id = client_info.get("session_id")
    # Note: Validating session_id presence should ideally happen in middleware
    # or depends, but checking here covers direct calls.

    current_model = None
    if session_id:
        current_model = await model_sync_service.get_user_model(session_id)

    if not current_model:
        # Return default model if no session found
        # We access the default model from the service instance directly
        # to ensure consistency
        default_model = model_sync_service.default_model
        return {
            "model_id": default_model,
            "model_name": "Default Model",
            "provider": "google",
            "is_default": True,
        }

    return {
        "model_id": current_model.id,
        "model_name": current_model.name,
        "provider": current_model.provider,
        "is_default": False,
    }


@router.websocket("/updates")
async def websocket_model_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time model availability updates.
    Clients subscribe to receive model status changes.
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())

    logger.info(f"WebSocket client connected: {client_id}")

    try:
        while True:
            # Send periodic availability updates
            available_models = await model_sync_service.get_available_models()

            # Create update message
            update_message = ModelAvailabilityUpdate(
                type="model_availability",
                model_id="",  # Empty for general availability update
                is_available=True,
                timestamp=datetime.utcnow().isoformat(),
                message=f"Available models: {available_models.count}",
            )

            await websocket.send_json(update_message.dict())

            # Wait before next update (30 seconds)
            await asyncio.sleep(30)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        try:
            error_message = ModelAvailabilityUpdate(
                type="error",
                model_id="",
                is_available=False,
                timestamp=datetime.utcnow().isoformat(),
                message=f"Connection error: {e!s}",
            )
            await websocket.send_json(error_message.dict())
        except Exception:
            pass  # Client already disconnected


@router.post("/validate/{model_id}")
async def validate_model(model_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Validate if a specific model ID is available and valid.
    Public endpoint for frontend validation.
    """
    try:
        is_valid = await model_sync_service.validate_model_id(model_id)

        return {
            "model_id": model_id,
            "is_valid": is_valid,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return {
            "model_id": model_id,
            "is_valid": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


@router.post("/availability/{model_id}")
async def update_model_availability(
    model_id: str,
    is_available: bool,
    error_message: str | None = None,
    user_id: str = Depends(get_current_user_id),
):
    """
    Update availability status for a specific model.
    Admin endpoint for system maintenance.
    """
    # Service methods should handle errors or bubble them up
    await model_sync_service.update_model_availability(
        model_id=model_id, is_available=is_available, error_message=error_message
    )

    return {
        "model_id": model_id,
        "is_available": is_available,
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Availability updated successfully",
    }


@router.get("/health")
async def model_sync_health():
    """
    Health check for model synchronization service.
    """
    try:
        available_models = await model_sync_service.get_available_models()

        return {
            "status": "healthy",
            "models_count": available_models.count,
            "cache_status": "active",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        }

    except Exception as e:
        logger.error(f"Model sync health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}
