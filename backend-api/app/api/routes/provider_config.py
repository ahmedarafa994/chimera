"""
Provider Configuration API Router

REST API endpoints for dynamic AI provider management including:
- List available providers
- Get/set active provider
- Provider configuration CRUD
- Health status monitoring
- WebSocket for real-time updates
"""

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, ConfigDict, Field

from app.domain.provider_models import (
    ProviderHealthInfo,
    ProviderHealthResponse,
    ProviderInfo,
    ProviderRegistration,
    ProviderRoutingConfig,
    ProviderSelectionRequest,
    ProviderSelectionResponse,
    ProviderStatus,
    ProviderTestResult,
    ProviderType,
    ProviderUpdate,
)
from app.services.provider_management_service import provider_management_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/provider-config", tags=["provider-config"])


# =============================================================================
# Response Models
# =============================================================================


class ProviderSummary(BaseModel):
    """Summary of a provider for list views"""

    id: str
    name: str
    display_name: str
    provider_type: ProviderType
    status: ProviderStatus
    enabled: bool
    is_default: bool
    is_active: bool = False
    has_api_key: bool = False
    model_count: int = 0

    model_config = ConfigDict(protected_namespaces=())


class ProvidersListResponse(BaseModel):
    """Response for listing all providers"""

    providers: list[ProviderSummary]
    total: int
    default_provider_id: str | None = None
    active_provider_id: str | None = None


class ActiveProviderResponse(BaseModel):
    """Response for active provider query"""

    provider_id: str | None = None
    provider_name: str | None = None
    provider_type: ProviderType | None = None
    status: ProviderStatus | None = None
    model_id: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class HealthStatusResponse(BaseModel):
    """Response for health status of all providers"""

    providers: dict[str, ProviderHealthInfo]
    overall_status: str
    timestamp: datetime


class SetActiveRequest(BaseModel):
    """Request to set active provider"""

    provider_id: str
    model_id: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class ApiKeyUpdateRequest(BaseModel):
    """Request to update API key"""

    api_key: str = Field(..., min_length=10)


# =============================================================================
# WebSocket Connection Manager
# =============================================================================


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)


# Global connection manager
ws_manager = ConnectionManager()


# =============================================================================
# Event Handlers
# =============================================================================


async def on_provider_event(event: dict):
    """Handle provider events and broadcast to WebSocket clients"""
    await ws_manager.broadcast(event)


# Register event callback
provider_management_service.register_event_callback(on_provider_event)


# =============================================================================
# REST API Endpoints
# =============================================================================


@router.get("/providers", response_model=ProvidersListResponse)
async def list_providers(
    enabled_only: bool = Query(default=False, description="Only return enabled providers"),
    include_health: bool = Query(default=True, description="Include health information"),
):
    """
    List all registered AI providers.

    Returns a summary of all providers with their status and configuration.
    """
    try:
        result = provider_management_service.list_providers(
            enabled_only=enabled_only, include_health=include_health
        )

        active_id = result.active_provider_id

        summaries = []
        for p in result.providers:
            summaries.append(
                ProviderSummary(
                    id=p.id,
                    name=p.name,
                    display_name=p.display_name,
                    provider_type=p.provider_type,
                    status=p.health.status if p.health else ProviderStatus.INITIALIZING,
                    enabled=p.enabled,
                    is_default=p.is_default,
                    is_active=(p.id == active_id),
                    has_api_key=bool(provider_management_service.get_api_key(p.id)),
                    model_count=len(p.models),
                )
            )

        return ProvidersListResponse(
            providers=summaries,
            total=result.total,
            default_provider_id=result.default_provider_id,
            active_provider_id=result.active_provider_id,
        )

    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list providers: {e!s}",
        ) from e


@router.get("/providers/active", response_model=ActiveProviderResponse)
async def get_active_provider():
    """
    Get the currently active provider.

    Returns details about the provider currently being used for AI requests.
    """
    try:
        provider = provider_management_service.get_active_provider()

        if not provider:
            return ActiveProviderResponse()

        return ActiveProviderResponse(
            provider_id=provider.id,
            provider_name=provider.name,
            provider_type=provider.provider_type,
            status=provider.health.status if provider.health else ProviderStatus.INITIALIZING,
            model_id=provider.default_model,
        )

    except Exception as e:
        logger.error(f"Failed to get active provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active provider: {e!s}",
        ) from e


@router.post("/providers/active", response_model=ProviderSelectionResponse)
async def set_active_provider(request: SetActiveRequest):
    """
    Set the active provider for AI requests.

    Switches the currently active provider to the specified one.
    """
    try:
        selection_request = ProviderSelectionRequest(
            provider_id=request.provider_id, model_id=request.model_id
        )

        result = await provider_management_service.select_provider(selection_request)

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.message or "Failed to set active provider",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set active provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set active provider: {e!s}",
        ) from e


@router.get("/providers/{provider_id}", response_model=ProviderInfo)
async def get_provider(provider_id: str):
    """
    Get detailed information about a specific provider.
    """
    try:
        provider = provider_management_service.get_provider(provider_id)

        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider '{provider_id}' not found"
            )

        # Update health info
        provider.health = provider_management_service.get_health(provider_id)

        return provider

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get provider {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider: {e!s}",
        ) from e


@router.post("/providers", response_model=ProviderInfo, status_code=status.HTTP_201_CREATED)
async def register_provider(registration: ProviderRegistration):
    """
    Register a new AI provider.

    Adds a new provider to the system with the specified configuration.
    """
    try:
        provider = await provider_management_service.register_provider(registration)
        return provider

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to register provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register provider: {e!s}",
        ) from e


@router.patch("/providers/{provider_id}", response_model=ProviderInfo)
async def update_provider(provider_id: str, update: ProviderUpdate):
    """
    Update an existing provider's configuration.
    """
    try:
        provider = await provider_management_service.update_provider(provider_id, update)
        return provider

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to update provider {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update provider: {e!s}",
        ) from e


@router.delete("/providers/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deregister_provider(provider_id: str):
    """
    Remove a provider from the system.
    """
    try:
        await provider_management_service.deregister_provider(provider_id)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to deregister provider {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deregister provider: {e!s}",
        ) from e


@router.put("/providers/{provider_id}/api-key", status_code=status.HTTP_204_NO_CONTENT)
async def update_api_key(provider_id: str, request: ApiKeyUpdateRequest):
    """
    Update the API key for a provider.
    """
    try:
        update = ProviderUpdate(api_key=request.api_key)
        await provider_management_service.update_provider(provider_id, update)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to update API key for {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update API key: {e!s}",
        ) from e


@router.post("/providers/{provider_id}/test", response_model=ProviderTestResult)
async def test_provider(provider_id: str):
    """
    Test a provider's connectivity and capabilities.
    """
    try:
        result = await provider_management_service.test_provider(provider_id)
        return result

    except Exception as e:
        logger.error(f"Failed to test provider {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test provider: {e!s}",
        ) from e


@router.post("/providers/{provider_id}/default", status_code=status.HTTP_204_NO_CONTENT)
async def set_default_provider(provider_id: str):
    """
    Set a provider as the default.
    """
    try:
        await provider_management_service.set_default_provider(provider_id)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to set default provider {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set default provider: {e!s}",
        ) from e


@router.get("/health", response_model=HealthStatusResponse)
async def get_health_status():
    """
    Get health status for all providers.
    """
    try:
        result = provider_management_service.list_providers(include_health=True)

        health_map = {}
        all_available = True

        for provider in result.providers:
            health = provider_management_service.get_health(provider.id)
            health_map[provider.id] = health

            if provider.enabled and health.status != ProviderStatus.AVAILABLE:
                all_available = False

        overall = "healthy" if all_available else "degraded"
        if not result.providers:
            overall = "no_providers"

        return HealthStatusResponse(
            providers=health_map, overall_status=overall, timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {e!s}",
        ) from e


@router.get("/providers/{provider_id}/health", response_model=ProviderHealthResponse)
async def get_provider_health(provider_id: str):
    """
    Get health status for a specific provider.
    """
    try:
        provider = provider_management_service.get_provider(provider_id)

        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider '{provider_id}' not found"
            )

        health = provider_management_service.get_health(provider_id)

        recommendations = []
        if health.status == ProviderStatus.UNAVAILABLE:
            recommendations.append("Check API key validity")
            recommendations.append("Verify network connectivity")
        elif health.status == ProviderStatus.DEGRADED:
            recommendations.append("Monitor for continued issues")
            recommendations.append("Consider switching to fallback provider")
        elif health.status == ProviderStatus.RATE_LIMITED:
            recommendations.append("Wait for rate limit reset")
            recommendations.append("Consider upgrading API tier")

        return ProviderHealthResponse(
            provider_id=provider_id, health=health, recommendations=recommendations
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health for {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider health: {e!s}",
        ) from e


@router.get("/routing", response_model=ProviderRoutingConfig)
async def get_routing_config():
    """
    Get the current provider routing configuration.
    """
    try:
        return provider_management_service.get_routing_config()

    except Exception as e:
        logger.error(f"Failed to get routing config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get routing config: {e!s}",
        ) from e


@router.put("/routing", response_model=ProviderRoutingConfig)
async def update_routing_config(config: ProviderRoutingConfig):
    """
    Update the provider routing configuration.
    """
    try:
        provider_management_service.set_routing_config(config)
        return provider_management_service.get_routing_config()

    except Exception as e:
        logger.error(f"Failed to update routing config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update routing config: {e!s}",
        ) from e


# =============================================================================
# WebSocket Endpoint
# =============================================================================


@router.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time provider status updates.

    Clients receive events when:
    - Provider status changes
    - Provider is added/removed
    - Active provider changes
    - Health status updates
    """
    await ws_manager.connect(websocket)

    try:
        # Send initial state
        result = provider_management_service.list_providers(include_health=True)
        await websocket.send_json(
            {
                "type": "initial_state",
                "data": {
                    "providers": [p.model_dump() for p in result.providers],
                    "active_provider_id": result.active_provider_id,
                    "default_provider_id": result.default_provider_id,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,  # Ping every 30 seconds
                )

                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")

            except TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_text("ping")
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)
