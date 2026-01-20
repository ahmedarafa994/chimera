"""Provider Synchronization API Router.

REST API and WebSocket endpoints for provider/model synchronization including:
- Full sync endpoint for initial load
- Incremental sync for updates
- WebSocket for real-time updates
- Health and availability endpoints
- Provider and model selection endpoints
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.domain.sync_models import (
    ModelAvailabilityInfo,
    ProviderAvailabilityInfo,
    SyncEvent,
    SyncEventType,
    SyncRequest,
    SyncResponse,
    SyncState,
)
from app.services.provider_management_service import provider_management_service
from app.services.provider_sync_service import provider_sync_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/provider-sync", tags=["provider-sync"])


# =============================================================================
# Request/Response Models for Selection
# =============================================================================


class SelectProviderRequest(BaseModel):
    """Request to select a provider and optionally a model."""

    provider_id: str = Field(..., description="ID of the provider to select")
    model_id: str | None = Field(None, description="ID of the model to select")
    persist: bool = Field(True, description="Whether to persist the selection")

    model_config = ConfigDict(protected_namespaces=())


class SelectProviderResponse(BaseModel):
    """Response from provider selection."""

    success: bool
    provider_id: str
    model_id: str | None = None
    previous_provider_id: str | None = None
    previous_model_id: str | None = None
    fallback_applied: bool = False
    fallback_reason: str | None = None
    message: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class SelectModelRequest(BaseModel):
    """Request to select a model within the current provider."""

    model_id: str = Field(..., description="ID of the model to select")
    persist: bool = Field(True, description="Whether to persist the selection")

    model_config = ConfigDict(protected_namespaces=())


class SelectModelResponse(BaseModel):
    """Response from model selection."""

    success: bool
    model_id: str
    provider_id: str
    previous_model_id: str | None = None
    message: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class ActiveSelectionResponse(BaseModel):
    """Current active provider and model selection."""

    provider_id: str | None = None
    provider_name: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    is_fallback: bool = False

    model_config = ConfigDict(protected_namespaces=())


# =============================================================================
# WebSocket Connection Manager
# =============================================================================


class SyncConnectionManager:
    """Manages WebSocket connections for sync updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"Sync WebSocket connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"Sync WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, event: SyncEvent) -> None:
        """Broadcast sync event to all connected clients."""
        if not self.active_connections:
            return

        message = event.model_dump_json()
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send sync event: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            await self.disconnect(conn)


# Global connection manager
sync_ws_manager = SyncConnectionManager()


# Register broadcast callback with sync service
async def broadcast_sync_event(event: SyncEvent) -> None:
    await sync_ws_manager.broadcast(event)


provider_sync_service.register_event_callback(broadcast_sync_event)


# =============================================================================
# REST API Endpoints
# =============================================================================


@router.get("/state", response_model=SyncState)
async def get_sync_state(
    include_deprecated: Annotated[bool, Query(description="Include deprecated models")] = False,
):
    """Get the current synchronization state.

    Returns the complete state of all providers and models for initial sync.
    """
    try:
        # Get providers from management service
        result = provider_management_service.list_providers(include_health=True)

        # Build sync state
        state = await provider_sync_service.get_full_sync_state(
            providers=result.providers,
            active_provider_id=result.active_provider_id,
            default_provider_id=result.default_provider_id,
        )

        # Filter deprecated if requested
        if not include_deprecated:
            state.all_models = [
                m for m in state.all_models if m.deprecation_status.value == "active"
            ]
            state.model_count = len(state.all_models)

        return state

    except Exception as e:
        logger.exception(f"Failed to get sync state: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)},
        )


@router.post("/sync", response_model=SyncResponse)
async def sync_providers(request: SyncRequest):
    """Synchronize provider and model configurations.

    Supports both full and incremental sync based on client version.
    """
    try:
        # Get current providers
        result = provider_management_service.list_providers(include_health=True)

        # Handle sync request
        return await provider_sync_service.handle_sync_request(
            request=request,
            providers=result.providers,
            active_provider_id=result.active_provider_id,
            default_provider_id=result.default_provider_id,
        )

    except Exception as e:
        logger.exception(f"Sync request failed: {e}")
        return SyncResponse(
            success=False,
            error=str(e),
            retry_after_seconds=5,
        )


@router.get("/version")
async def get_sync_version():
    """Get the current sync version.

    Clients can use this to check if they need to sync.
    """
    return {
        "version": provider_sync_service.version,
        "server_time": datetime.utcnow().isoformat(),
    }


@router.get("/providers/{provider_id}/availability", response_model=ProviderAvailabilityInfo)
async def get_provider_availability(provider_id: str):
    """Get availability information for a specific provider.

    Includes fallback information if the provider is unavailable.
    """
    try:
        provider = provider_management_service.get_provider(provider_id)

        if not provider:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Provider '{provider_id}' not found"},
            )

        # Find fallback provider
        fallback = None
        result = provider_management_service.list_providers()
        for p in result.providers:
            if p.is_fallback and p.id != provider_id:
                fallback = p
                break

        return provider_sync_service.get_provider_availability(
            provider=provider,
            fallback_provider=fallback,
        )

    except Exception as e:
        logger.exception(f"Failed to get provider availability: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)},
        )


@router.get("/models/{model_id}/availability", response_model=ModelAvailabilityInfo)
async def get_model_availability(
    model_id: str,
    provider_id: Annotated[str | None, Query(description="Provider ID for the model")] = None,
):
    """Get availability information for a specific model.

    Includes deprecation warnings and alternative suggestions.
    """
    try:
        # Find the model
        result = provider_management_service.list_providers(include_health=True)

        target_model = None
        target_provider = None
        alternative_models = []

        for provider in result.providers:
            if provider_id and provider.id != provider_id:
                continue

            for model in provider.models:
                if model.id == model_id:
                    target_model = model
                    target_provider = provider
                else:
                    # Collect alternatives from same provider
                    alternative_models.append(model.id)

        if not target_model or not target_provider:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Model '{model_id}' not found"},
            )

        # Convert to ModelSpecification
        from app.domain.sync_models import ModelDeprecationStatus, ModelSpecification

        model_spec = ModelSpecification(
            id=target_model.id,
            name=target_model.name,
            provider_id=target_provider.id,
            description=target_model.description,
            context_window=target_model.max_tokens,
            max_input_tokens=target_model.max_tokens,
            max_output_tokens=target_model.max_output_tokens,
            supports_streaming=target_model.supports_streaming,
            supports_vision=target_model.supports_vision,
            supports_function_calling=target_model.supports_function_calling,
            is_default=target_model.is_default,
            is_available=True,
            deprecation_status=ModelDeprecationStatus.ACTIVE,
        )

        provider_available = (
            target_provider.enabled
            and target_provider.status
            and target_provider.status.value == "available"
        )

        return provider_sync_service.get_model_availability(
            model=model_spec,
            provider_available=provider_available,
            alternative_models=alternative_models[:5],  # Limit to 5 alternatives
        )

    except Exception as e:
        logger.exception(f"Failed to get model availability: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)},
        )


@router.get("/models")
async def list_all_models(
    provider_id: Annotated[str | None, Query(description="Filter by provider")] = None,
    include_deprecated: Annotated[bool, Query(description="Include deprecated models")] = False,
    capability: Annotated[str | None, Query(description="Filter by capability")] = None,
):
    """List all available models across all providers.

    Supports filtering by provider, deprecation status, and capabilities.
    """
    try:
        result = provider_management_service.list_providers(include_health=True)

        models = []
        for provider in result.providers:
            if provider_id and provider.id != provider_id:
                continue

            for model in provider.models:
                model_data = {
                    "id": model.id,
                    "name": model.name,
                    "provider_id": provider.id,
                    "provider_name": provider.display_name,
                    "description": model.description,
                    "context_window": model.max_tokens,
                    "max_output_tokens": model.max_output_tokens,
                    "supports_streaming": model.supports_streaming,
                    "supports_vision": model.supports_vision,
                    "supports_function_calling": model.supports_function_calling,
                    "is_default": model.is_default,
                    "tier": model.tier.value,
                    "is_deprecated": False,  # Would come from model metadata
                }

                # Filter by capability
                if capability:
                    cap_map = {
                        "streaming": model.supports_streaming,
                        "vision": model.supports_vision,
                        "function_calling": model.supports_function_calling,
                    }
                    if not cap_map.get(capability, False):
                        continue

                models.append(model_data)

        return {
            "models": models,
            "total": len(models),
            "filters": {
                "provider_id": provider_id,
                "include_deprecated": include_deprecated,
                "capability": capability,
            },
        }

    except Exception as e:
        logger.exception(f"Failed to list models: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)},
        )


# =============================================================================
# Provider/Model Selection Endpoints
# =============================================================================


@router.get("/active", response_model=ActiveSelectionResponse)
async def get_active_selection():
    """Get the currently active provider and model selection.

    Returns the current selection state for the session.
    """
    try:
        result = provider_management_service.list_providers(include_health=True)

        active_provider = None
        active_model = None
        is_fallback = False

        if result.active_provider_id:
            for provider in result.providers:
                if provider.id == result.active_provider_id:
                    active_provider = provider
                    is_fallback = provider.is_fallback

                    # Find active model
                    if result.active_model_id:
                        for model in provider.models:
                            if model.id == result.active_model_id:
                                active_model = model
                                break
                    elif provider.default_model:
                        for model in provider.models:
                            if model.id == provider.default_model:
                                active_model = model
                                break
                    break

        return ActiveSelectionResponse(
            provider_id=active_provider.id if active_provider else None,
            provider_name=active_provider.display_name if active_provider else None,
            model_id=active_model.id if active_model else None,
            model_name=active_model.name if active_model else None,
            is_fallback=is_fallback,
        )

    except Exception as e:
        logger.exception(f"Failed to get active selection: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)},
        )


@router.post("/select/provider", response_model=SelectProviderResponse)
async def select_provider(request: SelectProviderRequest):
    """Select a provider and optionally a model.

    Validates the selection and broadcasts the change to all connected clients.
    Falls back to a default provider if the selected provider is unavailable.
    """
    try:
        result = provider_management_service.list_providers(include_health=True)

        # Find the requested provider
        target_provider = None
        fallback_provider = None

        for provider in result.providers:
            if provider.id == request.provider_id:
                target_provider = provider
            if provider.is_fallback and provider.id != request.provider_id:
                fallback_provider = provider

        if not target_provider:
            return SelectProviderResponse(
                success=False,
                provider_id=request.provider_id,
                message=f"Provider '{request.provider_id}' not found",
            )

        # Check provider availability
        provider_available = (
            target_provider.enabled
            and target_provider.status
            and target_provider.status.value == "available"
        )

        selected_provider = target_provider
        selected_model_id = request.model_id
        fallback_applied = False
        fallback_reason = None

        # Apply fallback if provider is unavailable
        if not provider_available and fallback_provider:
            fallback_available = (
                fallback_provider.enabled
                and fallback_provider.status
                and fallback_provider.status.value == "available"
            )
            if fallback_available:
                selected_provider = fallback_provider
                fallback_applied = True
                fallback_reason = f"Provider '{target_provider.display_name}' is unavailable"
                # Use fallback provider's default model
                selected_model_id = fallback_provider.default_model

        # Validate model selection
        if selected_model_id:
            model_valid = any(m.id == selected_model_id for m in selected_provider.models)
            if not model_valid:
                # Fall back to provider's default model
                selected_model_id = selected_provider.default_model
        else:
            # Use default model if none specified
            selected_model_id = selected_provider.default_model

        # Update the active selection
        previous_provider_id = result.active_provider_id
        previous_model_id = result.active_model_id

        # Persist selection if requested
        if request.persist:
            try:
                await provider_management_service.set_active_provider(
                    selected_provider.id,
                    selected_model_id,
                )
            except Exception as e:
                logger.warning(f"Failed to persist selection: {e}")

        # Broadcast the change
        await provider_sync_service.notify_active_provider_changed(
            new_provider_id=selected_provider.id,
            new_model_id=selected_model_id,
            old_provider_id=previous_provider_id,
        )

        return SelectProviderResponse(
            success=True,
            provider_id=selected_provider.id,
            model_id=selected_model_id,
            previous_provider_id=previous_provider_id,
            previous_model_id=previous_model_id,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            message=f"Selected provider '{selected_provider.display_name}'",
        )

    except Exception as e:
        logger.exception(f"Failed to select provider: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)},
        )


@router.post("/select/model", response_model=SelectModelResponse)
async def select_model(request: SelectModelRequest):
    """Select a model within the currently active provider.

    Validates that the model belongs to the active provider and broadcasts
    the change to all connected clients.
    """
    try:
        result = provider_management_service.list_providers(include_health=True)

        if not result.active_provider_id:
            return SelectModelResponse(
                success=False,
                model_id=request.model_id,
                provider_id="",
                message="No active provider selected",
            )

        # Find the active provider
        active_provider = None
        for provider in result.providers:
            if provider.id == result.active_provider_id:
                active_provider = provider
                break

        if not active_provider:
            return SelectModelResponse(
                success=False,
                model_id=request.model_id,
                provider_id=result.active_provider_id,
                message=f"Active provider '{result.active_provider_id}' not found",
            )

        # Validate the model belongs to this provider
        target_model = None
        for model in active_provider.models:
            if model.id == request.model_id:
                target_model = model
                break

        if not target_model:
            return SelectModelResponse(
                success=False,
                model_id=request.model_id,
                provider_id=active_provider.id,
                message=f"Model '{request.model_id}' not found in provider '{active_provider.display_name}'",
            )

        previous_model_id = result.active_model_id

        # Persist selection if requested
        if request.persist:
            try:
                await provider_management_service.set_active_provider(
                    active_provider.id,
                    target_model.id,
                )
            except Exception as e:
                logger.warning(f"Failed to persist model selection: {e}")

        # Broadcast the change
        await provider_sync_service.notify_active_provider_changed(
            new_provider_id=active_provider.id,
            new_model_id=target_model.id,
            old_provider_id=active_provider.id,  # Same provider
        )

        return SelectModelResponse(
            success=True,
            model_id=target_model.id,
            provider_id=active_provider.id,
            previous_model_id=previous_model_id,
            message=f"Selected model '{target_model.name}'",
        )

    except Exception as e:
        logger.exception(f"Failed to select model: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)},
        )


# =============================================================================
# WebSocket Endpoint
# =============================================================================


@router.websocket("/ws")
async def websocket_sync(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time sync updates.

    Clients receive events when:
    - Providers are added, updated, or removed
    - Models are added, updated, deprecated, or removed
    - Active provider/model changes
    - Health status updates
    """
    await sync_ws_manager.connect(websocket)

    try:
        # Send initial state
        result = provider_management_service.list_providers(include_health=True)
        state = await provider_sync_service.get_full_sync_state(
            providers=result.providers,
            active_provider_id=result.active_provider_id,
            default_provider_id=result.default_provider_id,
        )

        initial_event = SyncEvent(
            type=SyncEventType.INITIAL_STATE,
            version=provider_sync_service.version,
            data=state.model_dump(),
        )

        await websocket.send_text(initial_event.model_dump_json())

        # Keep connection alive and handle messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                message = json.loads(data)
                msg_type = message.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json(
                        {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    )

                elif msg_type == "sync_request":
                    # Handle sync request over WebSocket
                    request = SyncRequest(**message.get("data", {}))
                    result = provider_management_service.list_providers(include_health=True)
                    response = await provider_sync_service.handle_sync_request(
                        request=request,
                        providers=result.providers,
                        active_provider_id=result.active_provider_id,
                        default_provider_id=result.default_provider_id,
                    )
                    await websocket.send_json(
                        {
                            "type": "sync_response",
                            "data": response.model_dump(),
                        },
                    )

                elif msg_type == "get_version":
                    await websocket.send_json(
                        {
                            "type": "version",
                            "version": provider_sync_service.version,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

            except TimeoutError:
                # Send heartbeat
                await provider_sync_service.send_heartbeat()

    except WebSocketDisconnect:
        logger.info("Sync WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"Sync WebSocket error: {e}")
    finally:
        await sync_ws_manager.disconnect(websocket)
