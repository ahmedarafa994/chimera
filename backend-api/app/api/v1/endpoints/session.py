"""Session Management API Endpoints.

Provides endpoints for:
- Creating and managing user sessions
- Model synchronization and validation
- Session-based model selection persistence
"""

from typing import Annotated, Any

from fastapi import APIRouter, Cookie, Header, HTTPException, Response
from pydantic import AliasChoices, BaseModel, Field

from app.services.session_service import session_service

router = APIRouter()


# Request/Response Models


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    provider: str | None = Field(None, description="Initial provider selection")
    model: str | None = Field(None, description="Initial model selection")


class CreateSessionResponse(BaseModel):
    """Response after creating a session."""

    success: bool
    session_id: str
    provider: str
    model: str
    message: str


class UpdateModelRequest(BaseModel):
    """Request to update model selection."""

    provider: str = Field(
        ...,
        description="Provider identifier",
        validation_alias=AliasChoices("provider", "provider_id"),
    )
    model: str = Field(
        ...,
        description="Model canonical identifier",
        validation_alias=AliasChoices("model", "model_id"),
    )


class UpdateModelResponse(BaseModel):
    """Response after updating model selection."""

    success: bool
    message: str
    provider: str
    model: str
    reverted_to_default: bool = False


class SessionInfoResponse(BaseModel):
    """Session information response."""

    session_id: str
    provider: str
    model: str
    created_at: str
    last_activity: str
    request_count: int


class ProviderInfo(BaseModel):
    """Provider with models."""

    provider: str
    status: str
    model: str | None
    available_models: list[str]
    models_detail: list[dict[str, Any]]


class ModelsListResponse(BaseModel):
    """Response with all available models."""

    providers: list[ProviderInfo]
    default_provider: str
    default_model: str
    total_models: int


class ValidateModelRequest(BaseModel):
    """Request to validate a model selection."""

    provider: str
    model: str


class ValidateModelResponse(BaseModel):
    """Response from model validation."""

    valid: bool
    message: str
    fallback_model: str | None = None
    fallback_provider: str | None = None


# Endpoints


@router.get("/models", response_model=ModelsListResponse, tags=["Model Sync"])
async def get_available_models():
    """Get the complete list of available models.

    This endpoint returns the master list of all available models
    organized by provider. Use this on frontend initialization to
    populate model selection UI.

    The backend is the single source of truth for model availability.
    """
    providers = session_service.get_providers_with_models()
    default_provider, default_model = session_service.get_default_model()

    total_models = sum(len(p["available_models"]) for p in providers)

    return ModelsListResponse(
        providers=[ProviderInfo(**p) for p in providers],
        default_provider=default_provider,
        default_model=default_model,
        total_models=total_models,
    )


@router.post("/models/validate", response_model=ValidateModelResponse, tags=["Model Sync"])
async def validate_model_selection(request: ValidateModelRequest):
    """Validate a model selection against the master list.

    Use this endpoint to check if a model selection is valid before
    attempting to use it. Returns fallback suggestions if invalid.
    """
    is_valid, message, fallback = session_service.validate_model(request.provider, request.model)

    fallback_provider = None
    if not is_valid and fallback:
        # Determine the fallback provider
        # Note: Accessing protected member is temporary pending refactor
        if request.provider in session_service._master_model_list:
            fallback_provider = request.provider
        else:
            fallback_provider, _ = session_service.get_default_model()

    return ValidateModelResponse(
        valid=is_valid,
        message=message,
        fallback_model=fallback,
        fallback_provider=fallback_provider,
    )


@router.post("", response_model=CreateSessionResponse, tags=["Session"])
async def create_session(request: CreateSessionRequest = None, response: Response = None):
    """Create a new user session.

    Creates a session with optional initial provider/model selection.
    If not specified, defaults will be used. The session ID is returned
    and should be stored client-side (e.g., in localStorage or cookie).
    """
    request = request or CreateSessionRequest()

    # Removed try-except block to allow unified error handling to catch errors
    session = session_service.create_session(provider=request.provider, model=request.model)

    # Set session cookie for convenience
    if response:
        response.set_cookie(
            key="chimera_session_id",
            value=session.session_id,
            httponly=True,
            samesite="lax",
            max_age=3600,  # 1 hour
        )

    return CreateSessionResponse(
        success=True,
        session_id=session.session_id,
        provider=session.selected_provider,
        model=session.selected_model,
        message="Session created successfully",
    )


# Static routes must come BEFORE parameterized routes to avoid conflicts
@router.get("/stats", tags=["Session"])
async def get_session_stats():
    """Get session statistics (admin endpoint).

    Returns statistics about active sessions.
    """
    return session_service.get_session_stats()


@router.get("/current-model", tags=["Session"])
async def get_current_model(
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    chimera_session_id: Annotated[str | None, Cookie()] = None,
):
    """Get the current model selection for routing requests.

    This is used internally by other services to determine which
    model to use for processing requests.
    """
    session_id = x_session_id or chimera_session_id

    if session_id:
        provider, model = session_service.get_session_model(session_id)
    else:
        provider, model = session_service.get_default_model()

    return {"provider": provider, "model": model, "session_active": session_id is not None}


@router.put("/model", response_model=UpdateModelResponse, tags=["Session"])
async def update_session_model(
    request: UpdateModelRequest,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    chimera_session_id: Annotated[str | None, Cookie()] = None,
):
    """Update the model selection for the current session.

    This endpoint validates the model selection against the master list.
    If invalid, it will revert to a default model and return an error
    message with the fallback model information.

    All subsequent requests for this session will use the selected model.
    """
    session_id = x_session_id or chimera_session_id

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID required. Provide via header/cookie.",
        )

    success, message, session = session_service.update_session_model(
        session_id=session_id,
        provider=request.provider,
        model=request.model,
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please create a new session.",
        )

    return UpdateModelResponse(
        success=success,
        message=message,
        provider=session.selected_provider,
        model=session.selected_model,
        reverted_to_default=not success,
    )


@router.get("/{session_id}/model", tags=["Session"])
async def get_session_model(session_id: str):
    """Get current model selection for a specific session."""
    provider, model = session_service.get_session_model(session_id)
    return {
        "session_id": session_id,
        "provider": provider,
        "model": model,
        "provider_id": provider,
        "model_id": model,
    }


@router.put("/{session_id}/model", response_model=UpdateModelResponse, tags=["Session"])
async def update_session_model_by_id(
    session_id: str,
    request: UpdateModelRequest,
):
    """Update model selection for a specific session."""
    success, message, session = session_service.update_session_model(
        session_id=session_id,
        provider=request.provider,
        model=request.model,
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please create a new session.",
        )

    return UpdateModelResponse(
        success=success,
        message=message,
        provider=session.selected_provider,
        model=session.selected_model,
        reverted_to_default=not success,
    )


@router.delete("", tags=["Session"])
async def delete_session(
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    chimera_session_id: Annotated[str | None, Cookie()] = None,
    response: Response = None,
):
    """Delete the current session.

    Removes the session from the server. Client should also clear
    any stored session ID.
    """
    session_id = x_session_id or chimera_session_id

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID required. Provide via header/cookie.",
        )

    deleted = session_service.delete_session(session_id)

    # Clear the cookie
    if response:
        response.delete_cookie("chimera_session_id")

    if deleted:
        return {"success": True, "message": "Session deleted successfully"}
    return {"success": False, "message": "Session not found"}


# Parameterized route for getting session by ID (path parameter)
# This MUST come AFTER all static routes to avoid conflicts
@router.get("/{session_id}", response_model=SessionInfoResponse | None, tags=["Session"])
async def get_session_by_id(session_id: str):
    """Get session information by session ID (path parameter).

    This endpoint allows retrieving session details using the session ID
    as a path parameter, which is useful for RESTful API clients.
    Returns null if no session exists.
    """
    session = session_service.get_session(session_id)

    if not session:
        return None

    return SessionInfoResponse(
        session_id=session.session_id,
        provider=session.selected_provider,
        model=session.selected_model,
        created_at=session.created_at.isoformat(),
        last_activity=session.last_activity.isoformat(),
        request_count=session.request_count,
    )


@router.get("", response_model=SessionInfoResponse | None, tags=["Session"])
async def get_session(
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    chimera_session_id: Annotated[str | None, Cookie()] = None,
):
    """Get current session information.

    Retrieves session details including the currently selected model.
    Session ID can be provided via X-Session-ID header or cookie.
    Returns null if no session exists.
    """
    session_id = x_session_id or chimera_session_id

    if not session_id:
        return None

    session = session_service.get_session(session_id)

    if not session:
        return None

    return SessionInfoResponse(
        session_id=session.session_id,
        provider=session.selected_provider,
        model=session.selected_model,
        created_at=session.created_at.isoformat(),
        last_activity=session.last_activity.isoformat(),
        request_count=session.request_count,
    )
