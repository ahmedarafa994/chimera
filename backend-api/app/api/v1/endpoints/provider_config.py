"""
Provider Configuration API Endpoints

REST API endpoints for dynamic provider management including:
- Provider CRUD operations
- API key management
- Health monitoring
- Model management
- Provider switching
"""

import contextlib
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.services.provider_registry import (
    CircuitBreakerState,
    ModelInfo,
    ProviderConfig,
    ProviderHealthStatus,
    ProviderRegistry,
    ProviderStatus,
    ProviderType,
    get_provider_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/provider-config", tags=["Provider Configuration"])


# =============================================================================
# Request/Response Models
# =============================================================================


class ProviderCreateRequest(BaseModel):
    """Request to create a new provider."""

    provider_id: str = Field(..., description="Unique provider identifier")
    provider_type: str = Field(..., description="Provider type (openai, anthropic, etc.)")
    display_name: str = Field(..., description="Display name for the provider")
    api_key: str | None = Field(None, description="API key (will be encrypted)")
    base_url: str | None = Field(None, description="Custom base URL")
    api_version: str | None = Field(None, description="API version")
    organization_id: str | None = Field(None, description="Organization ID")
    project_id: str | None = Field(None, description="Project ID")
    is_enabled: bool = Field(True, description="Whether provider is enabled")
    is_default: bool = Field(False, description="Whether this is the default provider")
    priority: int = Field(100, description="Priority for fallback (lower = higher priority)")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout_seconds: float = Field(60.0, description="Request timeout in seconds")
    rate_limit_rpm: int | None = Field(None, description="Rate limit (requests per minute)")
    rate_limit_tpm: int | None = Field(None, description="Rate limit (tokens per minute)")
    supported_models: list[str] = Field(default_factory=list, description="Supported model IDs")
    default_model: str | None = Field(None, description="Default model ID")
    capabilities: list[str] = Field(default_factory=list, description="Provider capabilities")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProviderUpdateRequest(BaseModel):
    """Request to update a provider."""

    display_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    organization_id: str | None = None
    project_id: str | None = None
    is_enabled: bool | None = None
    is_default: bool | None = None
    priority: int | None = None
    max_retries: int | None = None
    timeout_seconds: float | None = None
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
    supported_models: list[str] | None = None
    default_model: str | None = None
    capabilities: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ProviderResponse(BaseModel):
    """Provider information response."""

    provider_id: str
    provider_type: str
    display_name: str
    base_url: str | None
    api_version: str | None
    organization_id: str | None
    project_id: str | None
    is_enabled: bool
    is_default: bool
    priority: int
    max_retries: int
    timeout_seconds: float
    rate_limit_rpm: int | None
    rate_limit_tpm: int | None
    supported_models: list[str]
    default_model: str | None
    capabilities: list[str]
    metadata: dict[str, Any]
    has_api_key: bool
    created_at: str
    updated_at: str


class ProviderListResponse(BaseModel):
    """List of providers response."""

    providers: list[ProviderResponse]
    total_count: int
    active_provider_id: str | None
    timestamp: str


class ProviderHealthResponse(BaseModel):
    """Provider health status response."""

    provider_id: str
    status: str
    circuit_breaker_state: str
    last_success: str | None
    last_failure: str | None
    failure_count: int
    success_count: int
    total_requests: int
    success_rate: float
    avg_latency_ms: float | None
    error_rate: float
    last_error_message: str | None
    checked_at: str


class AllProvidersHealthResponse(BaseModel):
    """Health status for all providers."""

    providers: list[ProviderHealthResponse]
    system_health: str
    timestamp: str


class ModelCreateRequest(BaseModel):
    """Request to create a model."""

    model_id: str = Field(..., description="Unique model identifier")
    provider_id: str = Field(..., description="Provider ID")
    display_name: str = Field(..., description="Display name")
    model_type: str = Field("chat", description="Model type (chat, completion, embedding)")
    context_window: int | None = Field(None, description="Context window size")
    max_output_tokens: int | None = Field(None, description="Maximum output tokens")
    supports_streaming: bool = Field(True, description="Supports streaming")
    supports_function_calling: bool = Field(False, description="Supports function calling")
    supports_vision: bool = Field(False, description="Supports vision")
    input_price_per_1k: float | None = Field(None, description="Input price per 1K tokens")
    output_price_per_1k: float | None = Field(None, description="Output price per 1K tokens")
    currency: str = Field("USD", description="Currency for pricing")
    capabilities: list[str] = Field(default_factory=list, description="Model capabilities")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ModelResponse(BaseModel):
    """Model information response."""

    model_id: str
    provider_id: str
    name: str
    display_name: str
    model_type: str
    context_window: int | None
    max_output_tokens: int | None
    supports_streaming: bool
    supports_function_calling: bool
    supports_vision: bool
    pricing: dict[str, Any] | None
    capabilities: list[str]
    metadata: dict[str, Any]


class ModelsListResponse(BaseModel):
    """List of models response."""

    provider_id: str
    models: list[ModelResponse]
    total_count: int
    last_updated: str


class SetActiveProviderRequest(BaseModel):
    """Request to set active provider."""

    provider_id: str = Field(..., description="Provider ID to activate")
    model_id: str | None = Field(None, description="Optional model ID to select")


class ActiveProviderResponse(BaseModel):
    """Active provider response."""

    success: bool
    provider_id: str | None
    model_id: str | None
    message: str


class ApiKeyUpdateRequest(BaseModel):
    """Request to update API key."""

    api_key: str = Field(..., description="New API key")


class ApiKeyResponse(BaseModel):
    """API key update response."""

    success: bool
    provider_id: str
    has_api_key: bool
    key_hash: str | None
    message: str


class FallbackSuggestion(BaseModel):
    """Fallback provider suggestion."""

    provider_id: str
    display_name: str
    priority: int
    status: str
    reason: str


class FallbackResponse(BaseModel):
    """Fallback providers response."""

    suggestions: list[FallbackSuggestion]
    current_provider_id: str | None
    timestamp: str


# =============================================================================
# Helper Functions
# =============================================================================


def provider_config_to_response(config: ProviderConfig) -> ProviderResponse:
    """Convert ProviderConfig to ProviderResponse."""
    data = config.to_dict(include_sensitive=False)
    return ProviderResponse(**data)


def health_status_to_response(health: ProviderHealthStatus) -> ProviderHealthResponse:
    """Convert ProviderHealthStatus to ProviderHealthResponse."""
    return ProviderHealthResponse(
        provider_id=health.provider_id,
        status=health.status.value,
        circuit_breaker_state=health.circuit_breaker_state.value,
        last_success=health.last_success.isoformat() if health.last_success else None,
        last_failure=health.last_failure.isoformat() if health.last_failure else None,
        failure_count=health.failure_count,
        success_count=health.success_count,
        total_requests=health.total_requests,
        success_rate=health.success_rate,
        avg_latency_ms=health.avg_latency_ms,
        error_rate=health.error_rate,
        last_error_message=health.last_error_message,
        checked_at=health.checked_at.isoformat(),
    )


def model_info_to_response(model: ModelInfo) -> ModelResponse:
    """Convert ModelInfo to ModelResponse."""
    data = model.to_dict()
    return ModelResponse(**data)


# =============================================================================
# Provider CRUD Endpoints
# =============================================================================


@router.get("/providers", response_model=ProviderListResponse)
async def list_providers(
    enabled_only: bool = Query(False, description="Only return enabled providers"),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderListResponse:
    """List all registered providers."""
    if enabled_only:
        providers = await registry.get_enabled_providers()
    else:
        providers = await registry.get_all_providers()

    active = await registry.get_active_provider()

    return ProviderListResponse(
        providers=[provider_config_to_response(p) for p in providers],
        total_count=len(providers),
        active_provider_id=active.provider_id if active else None,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.post("/providers", response_model=ProviderResponse, status_code=201)
async def create_provider(
    request: ProviderCreateRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderResponse:
    """Create a new provider."""
    try:
        provider_type = ProviderType(request.provider_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider type: {request.provider_type}. "
            f"Valid types: {[t.value for t in ProviderType]}",
        )

    # Check if provider already exists
    existing = await registry.get_provider(request.provider_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Provider already exists: {request.provider_id}",
        )

    config = ProviderConfig(
        provider_id=request.provider_id,
        provider_type=provider_type,
        display_name=request.display_name,
        api_key=request.api_key,
        base_url=request.base_url,
        api_version=request.api_version,
        organization_id=request.organization_id,
        project_id=request.project_id,
        is_enabled=request.is_enabled,
        is_default=request.is_default,
        priority=request.priority,
        max_retries=request.max_retries,
        timeout_seconds=request.timeout_seconds,
        rate_limit_rpm=request.rate_limit_rpm,
        rate_limit_tpm=request.rate_limit_tpm,
        supported_models=request.supported_models,
        default_model=request.default_model,
        capabilities=request.capabilities,
        metadata=request.metadata,
    )

    success = await registry.register_provider(config)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create provider")

    # Get the updated config (with encrypted key)
    updated = await registry.get_provider(request.provider_id)
    if not updated:
        raise HTTPException(status_code=500, detail="Provider created but not found")

    logger.info(f"Provider created: {request.provider_id}")
    return provider_config_to_response(updated)


@router.get("/providers/{provider_id}", response_model=ProviderResponse)
async def get_provider(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderResponse:
    """Get a specific provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    return provider_config_to_response(config)


@router.patch("/providers/{provider_id}", response_model=ProviderResponse)
async def update_provider(
    provider_id: str,
    request: ProviderUpdateRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderResponse:
    """Update a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    updates = request.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    success = await registry.update_provider(provider_id, updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update provider")

    updated = await registry.get_provider(provider_id)
    if not updated:
        raise HTTPException(status_code=500, detail="Provider updated but not found")

    logger.info(f"Provider updated: {provider_id}")
    return provider_config_to_response(updated)


@router.delete("/providers/{provider_id}")
async def delete_provider(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> dict[str, Any]:
    """Delete a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    success = await registry.deregister_provider(provider_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete provider")

    logger.info(f"Provider deleted: {provider_id}")
    return {"success": True, "message": f"Provider {provider_id} deleted"}


# =============================================================================
# API Key Management
# =============================================================================


@router.put("/providers/{provider_id}/api-key", response_model=ApiKeyResponse)
async def update_api_key(
    provider_id: str,
    request: ApiKeyUpdateRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ApiKeyResponse:
    """Update API key for a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    success = await registry.update_provider(provider_id, {"api_key": request.api_key})
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update API key")

    # Get key hash for verification
    updated = await registry.get_provider(provider_id)
    key_hash = None
    if updated and updated.api_key_encrypted:
        from app.services.provider_registry import EncryptionService

        enc = EncryptionService()
        decrypted = enc.decrypt(updated.api_key_encrypted)
        if decrypted:
            key_hash = enc.hash_key(decrypted)

    logger.info(f"API key updated for provider: {provider_id}")
    return ApiKeyResponse(
        success=True,
        provider_id=provider_id,
        has_api_key=True,
        key_hash=key_hash,
        message="API key updated successfully",
    )


@router.delete("/providers/{provider_id}/api-key", response_model=ApiKeyResponse)
async def delete_api_key(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ApiKeyResponse:
    """Delete API key for a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    # Clear both plaintext and encrypted keys
    success = await registry.update_provider(
        provider_id,
        {"api_key": None, "api_key_encrypted": None},
        encrypt_api_key=False,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete API key")

    logger.info(f"API key deleted for provider: {provider_id}")
    return ApiKeyResponse(
        success=True,
        provider_id=provider_id,
        has_api_key=False,
        key_hash=None,
        message="API key deleted successfully",
    )


@router.get("/providers/{provider_id}/api-key/verify")
async def verify_api_key(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> dict[str, Any]:
    """Verify if a provider has a valid API key configured."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    has_key = bool(config.api_key or config.api_key_encrypted)

    # Try to decrypt and verify
    key_valid = False
    if has_key:
        api_key = await registry.get_provider_api_key(provider_id)
        key_valid = bool(api_key and len(api_key) > 0)

    return {
        "provider_id": provider_id,
        "has_api_key": has_key,
        "key_valid": key_valid,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Active Provider Management
# =============================================================================


@router.get("/active", response_model=ActiveProviderResponse)
async def get_active_provider(
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ActiveProviderResponse:
    """Get the currently active provider."""
    provider = await registry.get_active_provider()

    if provider:
        return ActiveProviderResponse(
            success=True,
            provider_id=provider.provider_id,
            model_id=provider.default_model,
            message="Active provider retrieved",
        )

    return ActiveProviderResponse(
        success=False,
        provider_id=None,
        model_id=None,
        message="No active provider set",
    )


@router.post("/active", response_model=ActiveProviderResponse)
async def set_active_provider(
    request: SetActiveProviderRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ActiveProviderResponse:
    """Set the active provider."""
    config = await registry.get_provider(request.provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {request.provider_id}")

    if not config.is_enabled:
        raise HTTPException(status_code=400, detail=f"Provider is disabled: {request.provider_id}")

    success = await registry.set_active_provider(request.provider_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to set active provider")

    # Update default model if specified
    model_id = request.model_id or config.default_model
    if request.model_id:
        await registry.update_provider(request.provider_id, {"default_model": request.model_id})

    logger.info(f"Active provider set: {request.provider_id}")
    return ActiveProviderResponse(
        success=True,
        provider_id=request.provider_id,
        model_id=model_id,
        message=f"Active provider set to {request.provider_id}",
    )


# =============================================================================
# Health Monitoring
# =============================================================================


@router.get("/health", response_model=AllProvidersHealthResponse)
async def get_all_health(
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> AllProvidersHealthResponse:
    """Get health status for all providers."""
    health_statuses = await registry.get_all_health_status()

    # Determine system health
    statuses = [h.status for h in health_statuses.values()]
    if all(s == ProviderStatus.AVAILABLE for s in statuses):
        system_health = "healthy"
    elif any(s == ProviderStatus.AVAILABLE for s in statuses):
        system_health = "degraded"
    else:
        system_health = "unhealthy"

    return AllProvidersHealthResponse(
        providers=[health_status_to_response(h) for h in health_statuses.values()],
        system_health=system_health,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/providers/{provider_id}/health", response_model=ProviderHealthResponse)
async def get_provider_health(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderHealthResponse:
    """Get health status for a specific provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    health = await registry.get_health_status(provider_id)
    if not health:
        # Return default health status
        health = ProviderHealthStatus(
            provider_id=provider_id,
            status=ProviderStatus.UNKNOWN,
            circuit_breaker_state=CircuitBreakerState.CLOSED,
        )

    return health_status_to_response(health)


@router.post("/providers/{provider_id}/health/check")
async def trigger_health_check(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> dict[str, Any]:
    """Trigger a health check for a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    status = await registry.check_provider_health(provider_id)
    await registry.update_health_status(provider_id, status)

    return {
        "provider_id": provider_id,
        "status": status.value,
        "checked_at": datetime.utcnow().isoformat(),
    }


@router.post("/providers/{provider_id}/circuit-breaker/reset")
async def reset_circuit_breaker(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> dict[str, Any]:
    """Reset the circuit breaker for a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    cb = registry._circuit_breakers.get(provider_id)
    if cb:
        await cb.reset()

    # Update health status
    await registry.update_health_status(provider_id, ProviderStatus.AVAILABLE)

    logger.info(f"Circuit breaker reset for provider: {provider_id}")
    return {
        "success": True,
        "provider_id": provider_id,
        "message": "Circuit breaker reset",
    }


# =============================================================================
# Model Management
# =============================================================================


@router.get("/providers/{provider_id}/models", response_model=ModelsListResponse)
async def list_provider_models(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ModelsListResponse:
    """List all models for a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    models = await registry.get_models(provider_id)

    return ModelsListResponse(
        provider_id=provider_id,
        models=[model_info_to_response(m) for m in models],
        total_count=len(models),
        last_updated=datetime.utcnow().isoformat(),
    )


@router.post("/providers/{provider_id}/models", response_model=ModelResponse, status_code=201)
async def create_model(
    provider_id: str,
    request: ModelCreateRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ModelResponse:
    """Create a new model for a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    if request.provider_id != provider_id:
        raise HTTPException(
            status_code=400,
            detail="Provider ID in request body must match URL parameter",
        )

    model = ModelInfo(
        model_id=request.model_id,
        provider_id=provider_id,
        display_name=request.display_name,
        model_type=request.model_type,
        context_window=request.context_window,
        max_output_tokens=request.max_output_tokens,
        supports_streaming=request.supports_streaming,
        supports_function_calling=request.supports_function_calling,
        supports_vision=request.supports_vision,
        input_price_per_1k=request.input_price_per_1k,
        output_price_per_1k=request.output_price_per_1k,
        currency=request.currency,
        capabilities=request.capabilities,
        metadata=request.metadata,
    )

    success = await registry.register_model(model)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create model")

    logger.info(f"Model created: {request.model_id} for {provider_id}")
    return model_info_to_response(model)


@router.delete("/providers/{provider_id}/models/{model_id}")
async def delete_model(
    provider_id: str,
    model_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> dict[str, Any]:
    """Delete a model from a provider."""
    config = await registry.get_provider(provider_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    success = await registry.deregister_model(provider_id, model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    logger.info(f"Model deleted: {model_id} from {provider_id}")
    return {"success": True, "message": f"Model {model_id} deleted"}


# =============================================================================
# Fallback and Routing
# =============================================================================


@router.get("/fallback", response_model=FallbackResponse)
async def get_fallback_suggestions(
    exclude: str | None = Query(None, description="Comma-separated provider IDs to exclude"),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> FallbackResponse:
    """Get fallback provider suggestions."""
    exclude_set = set(exclude.split(",")) if exclude else set()

    suggestions = []
    providers = await registry.get_enabled_providers()

    # Sort by priority
    sorted_providers = sorted(
        [p for p in providers if p.provider_id not in exclude_set],
        key=lambda p: p.priority,
    )

    for provider in sorted_providers[:5]:  # Top 5 suggestions
        health = await registry.get_health_status(provider.provider_id)
        status = health.status if health else ProviderStatus.UNKNOWN

        can_use = await registry.can_use_provider(provider.provider_id)
        reason = "Available" if can_use else "Circuit breaker open"

        suggestions.append(
            FallbackSuggestion(
                provider_id=provider.provider_id,
                display_name=provider.display_name,
                priority=provider.priority,
                status=status.value,
                reason=reason,
            )
        )

    active = await registry.get_active_provider()

    return FallbackResponse(
        suggestions=suggestions,
        current_provider_id=active.provider_id if active else None,
        timestamp=datetime.utcnow().isoformat(),
    )


# =============================================================================
# Configuration Persistence
# =============================================================================


@router.post("/save")
async def save_configuration(
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> dict[str, Any]:
    """Save provider configurations to file."""
    success = await registry.save_to_file()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    return {
        "success": True,
        "message": "Configuration saved",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/load")
async def load_configuration(
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> dict[str, Any]:
    """Load provider configurations from file."""
    success = await registry.load_from_file()

    return {
        "success": success,
        "message": "Configuration loaded" if success else "No configuration file found",
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# WebSocket for Real-time Updates
# =============================================================================


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]):
        for connection in self.active_connections:
            with contextlib.suppress(Exception):
                await connection.send_json(message)


ws_manager = ConnectionManager()


@router.websocket("/ws/updates")
async def websocket_updates(
    websocket: WebSocket,
    registry: ProviderRegistry = Depends(get_provider_registry),
):
    """WebSocket endpoint for real-time provider updates."""
    await ws_manager.connect(websocket)

    # Register event handlers
    async def on_provider_update(provider_id: str, data: dict[str, Any]):
        await ws_manager.broadcast(
            {
                "type": "provider_updated",
                "provider_id": provider_id,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def on_status_change(provider_id: str, data: dict[str, Any]):
        await ws_manager.broadcast(
            {
                "type": "provider_status_changed",
                "provider_id": provider_id,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def on_active_change(provider_id: str, data: dict[str, Any]):
        await ws_manager.broadcast(
            {
                "type": "active_provider_changed",
                "provider_id": provider_id,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    # Register handlers
    registry.on("provider_updated", on_provider_update)
    registry.on("provider_registered", on_provider_update)
    registry.on("provider_deregistered", on_provider_update)
    registry.on("provider_status_changed", on_status_change)
    registry.on("active_provider_changed", on_active_change)

    try:
        # Send initial state
        providers = await registry.get_all_providers()
        active = await registry.get_active_provider()
        health_statuses = await registry.get_all_health_status()

        await websocket.send_json(
            {
                "type": "initial_state",
                "data": {
                    "providers": [p.to_dict() for p in providers],
                    "active_provider_id": active.provider_id if active else None,
                    "health": {pid: h.to_dict() for pid, h in health_statuses.items()},
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_json()

                # Handle ping/pong
                if data.get("type") == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                # Handle refresh request
                elif data.get("type") == "refresh":
                    providers = await registry.get_all_providers()
                    active = await registry.get_active_provider()
                    health_statuses = await registry.get_all_health_status()

                    await websocket.send_json(
                        {
                            "type": "refresh_response",
                            "data": {
                                "providers": [p.to_dict() for p in providers],
                                "active_provider_id": active.provider_id if active else None,
                                "health": {pid: h.to_dict() for pid, h in health_statuses.items()},
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        # Unregister handlers
        registry.off("provider_updated", on_provider_update)
        registry.off("provider_registered", on_provider_update)
        registry.off("provider_deregistered", on_provider_update)
        registry.off("provider_status_changed", on_status_change)
        registry.off("active_provider_changed", on_active_change)

        ws_manager.disconnect(websocket)


# =============================================================================
# Provider Types Endpoint
# =============================================================================


@router.get("/types")
async def list_provider_types() -> dict[str, Any]:
    """List all supported provider types."""
    return {
        "types": [
            {
                "id": t.value,
                "name": t.name,
                "description": _get_provider_type_description(t),
            }
            for t in ProviderType
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }


def _get_provider_type_description(provider_type: ProviderType) -> str:
    """Get description for a provider type."""
    descriptions = {
        ProviderType.OPENAI: "OpenAI API (GPT-4, GPT-3.5, etc.)",
        ProviderType.ANTHROPIC: "Anthropic API (Claude models)",
        ProviderType.GOOGLE: "Google AI (Gemini models)",
        ProviderType.GEMINI: "Google Gemini API",
        ProviderType.DEEPSEEK: "DeepSeek API",
        ProviderType.QWEN: "Alibaba Qwen API",
        ProviderType.OLLAMA: "Ollama local models",
        ProviderType.LOCAL: "Local/self-hosted models",
        ProviderType.CUSTOM: "Custom provider implementation",
    }
    return descriptions.get(provider_type, "Unknown provider type")
