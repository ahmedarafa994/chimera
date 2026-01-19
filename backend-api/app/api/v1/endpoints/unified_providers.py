"""
Unified Providers API Endpoint for Project Chimera.

Provides REST API endpoints for managing AI providers, discovering models,
and controlling the unified provider system.

Includes:
- Provider discovery and model catalog endpoints
- Session selection persistence and synchronization
- WebSocket support for real-time selection updates
- Optimistic concurrency control for selection changes
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import AliasChoices, BaseModel, Field

from app.services.model_catalog_service import get_model_catalog_service, initialize_model_catalog
from app.services.provider_plugins import get_all_plugins, get_plugin, register_all_plugins
from app.services.provider_registry import get_provider_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/unified-providers", tags=["Unified Providers"])


# =============================================================================
# Response Models
# =============================================================================


class ProviderResponse(BaseModel):
    """Response model for provider information."""

    provider_id: str = Field(..., description="Unique provider identifier")
    display_name: str = Field(..., description="Human-readable name")
    is_available: bool = Field(..., description="Whether provider is ready")
    models_count: int = Field(0, description="Number of available models")
    capabilities: list[str] = Field(default_factory=list, description="Provider capabilities")
    default_model: str | None = Field(None, description="Default model ID")
    status: str = Field("unknown", description="Provider health status")
    has_api_key: bool = Field(False, description="Whether API key is configured")


class ModelResponse(BaseModel):
    """Response model for model information."""

    model_id: str = Field(..., description="Unique model identifier")
    provider_id: str = Field(..., description="Provider this model belongs to")
    display_name: str = Field(..., description="Human-readable name")
    context_window: int | None = Field(None, description="Maximum context window size")
    max_output_tokens: int | None = Field(None, description="Maximum output tokens")
    supports_streaming: bool = Field(True, description="Streaming support")
    supports_vision: bool = Field(False, description="Vision/image support")
    supports_function_calling: bool = Field(False, description="Function calling support")
    input_price_per_1k: float | None = Field(None, description="Input price per 1K tokens")
    output_price_per_1k: float | None = Field(None, description="Output price per 1K tokens")
    capabilities: list[str] = Field(default_factory=list, description="Model capabilities")


class ProvidersListResponse(BaseModel):
    """Response for listing all providers."""

    providers: list[ProviderResponse]
    total: int = Field(..., description="Total number of providers")


class ModelsListResponse(BaseModel):
    """Response for listing models."""

    models: list[ModelResponse]
    total: int = Field(..., description="Total number of models")
    provider_id: str | None = Field(None, description="Provider filter if applied")


class RefreshResponse(BaseModel):
    """Response for refresh operations."""

    success: bool
    message: str
    providers_refreshed: list[str] = Field(default_factory=list)
    total_models: int = 0


class CatalogStatsResponse(BaseModel):
    """Response for catalog statistics."""

    total_providers: int
    total_models: int
    providers_with_models: dict[str, int]
    last_refresh: str | None
    cache_ttl_seconds: float


class ValidationMetricsResponse(BaseModel):
    """Response for validation metrics."""

    total_validations: int = Field(..., description="Total number of validations performed")
    successful_validations: int = Field(..., description="Number of successful validations")
    failed_validations: int = Field(..., description="Number of failed validations")
    success_rate: float = Field(..., description="Validation success rate (0.0-1.0)")
    error_counts: dict[str, int] = Field(
        default_factory=dict, description="Count of each error type"
    )
    source_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of selection sources"
    )
    provider_health_at_validation: dict[str, int] = Field(
        default_factory=dict, description="Provider health count"
    )
    avg_validation_time_ms: float = Field(
        ..., description="Average validation time in milliseconds"
    )
    last_updated: str | None = Field(None, description="When metrics were last updated")


# =============================================================================
# Current Selection Response Model (for GET /selection endpoint)
# =============================================================================


class CurrentSelectionResponse(BaseModel):
    """Response for GET /selection - matches frontend CurrentSelectionResponse type."""

    provider_id: str = Field(..., description="Selected provider identifier")
    model_id: str = Field(..., description="Selected model identifier")
    scope: str = Field("SESSION", description="Selection scope: REQUEST, SESSION, or GLOBAL")
    session_id: str | None = Field(None, description="Session identifier")
    user_id: str | None = Field(None, description="User identifier")
    created_at: str | None = Field(None, description="When selection was created")
    updated_at: str | None = Field(None, description="When selection was last updated")
    version: int | None = Field(None, description="Version for optimistic concurrency")
    metadata: dict | None = Field(None, description="Selection metadata")


# =============================================================================
# Selection Sync Models
# =============================================================================


class SelectionSyncRequest(BaseModel):
    """Request for synchronizing selection between frontend and backend."""

    session_id: str = Field(..., description="Session identifier")
    provider: str = Field(..., description="Provider identifier")
    model: str = Field(..., description="Model identifier")
    version: int | None = Field(None, description="Expected version for optimistic concurrency")
    user_id: str | None = Field(None, description="Optional user identifier")
    source: str = Field(default="frontend", description="Source of the sync request")
    timestamp: str | None = Field(None, description="Client timestamp (ISO 8601)")


class SelectionSyncResponse(BaseModel):
    """Response for selection sync operation."""

    success: bool = Field(..., description="Whether sync succeeded")
    session_id: str = Field(..., description="Session identifier")
    provider: str = Field(..., description="Current provider")
    model: str = Field(..., description="Current model")
    version: int = Field(..., description="Current version after sync")
    conflict: bool = Field(False, description="Whether there was a version conflict")
    serverTimestamp: str = Field(..., description="Server timestamp (ISO 8601)")
    message: str | None = Field(None, description="Optional status message")


class SetSelectionRequest(BaseModel):
    """Request for setting session selection."""

    provider: str = Field(..., description="Provider identifier")
    model: str = Field(..., description="Model identifier")
    expected_version: int | None = Field(
        None, description="Expected version for optimistic locking (409 on mismatch)"
    )
    user_id: str | None = Field(None, description="Optional user identifier")
    metadata: dict | None = Field(None, description="Optional metadata to store")


class SelectionResponse(BaseModel):
    """Response with current selection state."""

    session_id: str = Field(..., description="Session identifier")
    provider: str = Field(..., description="Current provider")
    model: str = Field(..., description="Current model")
    version: int = Field(..., description="Selection version")
    user_id: str | None = Field(None, description="User identifier")
    created_at: str = Field(..., description="When selection was created")
    updated_at: str = Field(..., description="When selection was last updated")
    metadata: dict = Field(default_factory=dict, description="Selection meta")


class SelectionHistoryEntry(BaseModel):
    """Entry in selection history."""

    provider: str
    model: str
    version: int
    timestamp: str
    source: str


class SelectionHistoryResponse(BaseModel):
    """Response for selection history."""

    session_id: str
    entries: list[SelectionHistoryEntry]
    total: int


class ModelValidationRequest(BaseModel):
    """Request to validate a provider/model combination."""

    provider_id: str = Field(
        ...,
        description="Provider identifier",
        validation_alias=AliasChoices("provider_id", "provider"),
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
        validation_alias=AliasChoices("model_id", "model"),
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get(
    "/providers",
    response_model=ProvidersListResponse,
    summary="List all providers",
    description="Get a list of all available AI providers with their status.",
)
async def list_providers() -> ProvidersListResponse:
    """
    List all available providers with their status and capabilities.

    Returns information about all registered and available provider plugins,
    including their health status and configuration state.
    """
    try:
        # Ensure plugins are registered
        register_all_plugins()

        plugins = get_all_plugins()
        catalog = get_model_catalog_service()
        registry = get_provider_registry()

        providers = []
        for plugin in plugins:
            # Get health status from registry
            health = await registry.get_health_status(plugin.provider_type)
            status = health.status.value if health else "unknown"

            # Get model count
            models = await catalog.get_available_models(plugin.provider_type)

            # Check if API key is configured
            has_api_key = bool(plugin._get_api_key(None))

            providers.append(
                ProviderResponse(
                    provider_id=plugin.provider_type,
                    display_name=plugin.display_name,
                    is_available=bool(models) and status != "unavailable",
                    models_count=len(models),
                    capabilities=[cap.value for cap in plugin.capabilities],
                    default_model=plugin.get_default_model(),
                    status=status,
                    has_api_key=has_api_key,
                )
            )

        return ProvidersListResponse(
            providers=providers,
            total=len(providers),
        )
    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list providers: {e!s}")


@router.get(
    "/providers/health",
    summary="Get provider health",
    description="Get health status for all providers.",
)
async def get_providers_health() -> dict[str, Any]:
    """Return health status for all providers."""
    register_all_plugins()
    registry = get_provider_registry()
    health_map: dict[str, Any] = {}

    for plugin in get_all_plugins():
        health = await registry.get_health_status(plugin.provider_type)
        if health:
            health_map[plugin.provider_type] = {
                "status": health.status.value,
                "is_available": health.status.value != "unavailable",
                "health_score": health.success_rate,
                "details": health.to_dict(),
            }
        else:
            health_map[plugin.provider_type] = {
                "status": "unknown",
                "is_available": False,
            }

    return health_map


@router.get(
    "/providers/{provider_id}/health",
    summary="Get provider health",
    description="Get health status for a specific provider.",
)
async def get_provider_health(provider_id: str) -> dict[str, Any]:
    """Return health status for a specific provider."""
    register_all_plugins()
    registry = get_provider_registry()
    health = await registry.get_health_status(provider_id)
    if not health:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    return {
        "status": health.status.value,
        "is_available": health.status.value != "unavailable",
        "health_score": health.success_rate,
        "details": health.to_dict(),
    }


@router.get(
    "/providers/{provider_id}",
    response_model=ProviderResponse,
    summary="Get provider details",
    description="Get detailed information about a specific provider.",
)
async def get_provider(provider_id: str) -> ProviderResponse:
    """
    Get detailed information about a specific provider.

    Args:
        provider_id: The provider identifier (e.g., "openai", "anthropic")

    Returns:
        Provider details including status and capabilities
    """
    try:
        plugin = get_plugin(provider_id)
        if not plugin:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

        catalog = get_model_catalog_service()
        registry = get_provider_registry()

        health = await registry.get_health_status(plugin.provider_type)
        status = health.status.value if health else "unknown"

        models = await catalog.get_available_models(plugin.provider_type)
        has_api_key = bool(plugin._get_api_key(None))

        return ProviderResponse(
            provider_id=plugin.provider_type,
            display_name=plugin.display_name,
            is_available=bool(models) and status != "unavailable",
            models_count=len(models),
            capabilities=[cap.value for cap in plugin.capabilities],
            default_model=plugin.get_default_model(),
            status=status,
            has_api_key=has_api_key,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider {provider_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider: {e!s}")


@router.get(
    "/providers/{provider_id}/models",
    response_model=ModelsListResponse,
    summary="List provider models",
    description="Get all models available for a specific provider.",
)
async def list_provider_models(provider_id: str) -> ModelsListResponse:
    """
    List all models available for a specific provider.

    Args:
        provider_id: The provider identifier

    Returns:
        List of available models with their specifications
    """
    try:
        plugin = get_plugin(provider_id)
        if not plugin:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

        catalog = get_model_catalog_service()
        models = await catalog.get_available_models(provider_id)

        model_responses = [
            ModelResponse(
                model_id=m.model_id,
                provider_id=m.provider_id,
                display_name=m.display_name,
                context_window=m.context_window,
                max_output_tokens=m.max_output_tokens,
                supports_streaming=m.supports_streaming,
                supports_vision=m.supports_vision,
                supports_function_calling=m.supports_function_calling,
                input_price_per_1k=m.input_price_per_1k,
                output_price_per_1k=m.output_price_per_1k,
                capabilities=m.capabilities or [],
            )
            for m in models
        ]

        return ModelsListResponse(
            models=model_responses,
            total=len(model_responses),
            provider_id=provider_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models for {provider_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e!s}")


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List all models",
    description="Get all models across all providers with optional filters.",
)
async def list_all_models(
    provider: str | None = Query(None, description="Filter by provider ID"),
    capability: str | None = Query(
        None, description="Filter by capability (e.g., 'vision', 'code')"
    ),
    supports_streaming: bool | None = Query(None, description="Filter by streaming support"),
    supports_vision: bool | None = Query(None, description="Filter by vision support"),
    min_context: int | None = Query(None, description="Minimum context window size"),
    search: str | None = Query(None, description="Search in model name/ID"),
) -> ModelsListResponse:
    """
    List all models across all providers with optional filters.

    Supports filtering by:
    - Provider
    - Capability
    - Streaming support
    - Vision support
    - Minimum context window
    - Text search
    """
    try:
        catalog = get_model_catalog_service()

        if provider:
            # Get models for specific provider
            models = await catalog.get_available_models(provider)
        else:
            # Get all models with search/filter
            models = await catalog.search_models(
                query=search,
                capability=capability,
                min_context_window=min_context,
                supports_streaming=supports_streaming,
                supports_vision=supports_vision,
            )

        # Apply additional filters if provider-specific
        if provider and (
            capability
            or supports_streaming is not None
            or supports_vision is not None
            or min_context
            or search
        ):
            filtered = []
            for m in models:
                if capability and capability not in (m.capabilities or []):
                    continue
                if supports_streaming is not None and m.supports_streaming != supports_streaming:
                    continue
                if supports_vision is not None and m.supports_vision != supports_vision:
                    continue
                if min_context and (m.context_window or 0) < min_context:
                    continue
                if search:
                    search_text = f"{m.model_id} {m.display_name}".lower()
                    if search.lower() not in search_text:
                        continue
                filtered.append(m)
            models = filtered

        model_responses = [
            ModelResponse(
                model_id=m.model_id,
                provider_id=m.provider_id,
                display_name=m.display_name,
                context_window=m.context_window,
                max_output_tokens=m.max_output_tokens,
                supports_streaming=m.supports_streaming,
                supports_vision=m.supports_vision,
                supports_function_calling=m.supports_function_calling,
                input_price_per_1k=m.input_price_per_1k,
                output_price_per_1k=m.output_price_per_1k,
                capabilities=m.capabilities or [],
            )
            for m in models
        ]

        return ModelsListResponse(
            models=model_responses,
            total=len(model_responses),
            provider_id=provider,
        )
    except Exception as e:
        logger.error(f"Error listing all models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e!s}")


@router.post(
    "/models/validate",
    summary="Validate model selection",
    description="Validate a provider/model combination.",
)
async def validate_model_selection(request: ModelValidationRequest) -> dict[str, Any]:
    """Validate that a model exists for the provider."""
    try:
        catalog = get_model_catalog_service()
        model = await catalog.get_model_by_id(request.model_id, provider=request.provider_id)
        if not model:
            return {
                "is_valid": False,
                "errors": [
                    f"Model '{request.model_id}' not found for provider '{request.provider_id}'"
                ],
                "warnings": [],
            }

        if model.provider_id != request.provider_id:
            return {
                "is_valid": False,
                "errors": [
                    f"Model '{request.model_id}' does not belong to provider '{request.provider_id}'"
                ],
                "warnings": [],
            }

        return {"is_valid": True, "errors": [], "warnings": []}
    except Exception as e:
        logger.error(f"Error validating model selection: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e!s}")


@router.get(
    "/models/{model_id}",
    response_model=ModelResponse,
    summary="Get model details",
    description="Get detailed information about a specific model.",
)
async def get_model(
    model_id: str,
    provider: str | None = Query(None, description="Provider ID hint"),
) -> ModelResponse:
    """
    Get detailed information about a specific model.

    Args:
        model_id: The model identifier
        provider: Optional provider hint for faster lookup

    Returns:
        Model details including capabilities and pricing
    """
    try:
        catalog = get_model_catalog_service()
        model = await catalog.get_model_by_id(model_id, provider=provider)

        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        return ModelResponse(
            model_id=model.model_id,
            provider_id=model.provider_id,
            display_name=model.display_name,
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            supports_streaming=model.supports_streaming,
            supports_vision=model.supports_vision,
            supports_function_calling=model.supports_function_calling,
            input_price_per_1k=model.input_price_per_1k,
            output_price_per_1k=model.output_price_per_1k,
            capabilities=model.capabilities or [],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {e!s}")


@router.post(
    "/providers/refresh",
    response_model=RefreshResponse,
    summary="Refresh provider catalog",
    description="Force refresh of model catalog for providers.",
)
async def refresh_provider_catalog(
    provider: str | None = Query(None, description="Specific provider to refresh"),
) -> RefreshResponse:
    """
    Force refresh of the model catalog.

    This will query each provider's API to update the available models.
    Can be limited to a specific provider or refresh all providers.

    Args:
        provider: Optional specific provider to refresh

    Returns:
        Refresh operation results
    """
    try:
        catalog = get_model_catalog_service()

        result = await catalog.refresh_catalog(provider=provider)

        providers_refreshed = list(result.keys())
        total_models = sum(len(models) for models in result.values())

        return RefreshResponse(
            success=True,
            message=(
                f"Refreshed {len(providers_refreshed)} providers "
                f"with {total_models} total models"
            ),
            providers_refreshed=providers_refreshed,
            total_models=total_models,
        )
    except Exception as e:
        logger.error(f"Error refreshing catalog: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh catalog: {e!s}")


@router.get(
    "/catalog/stats",
    response_model=CatalogStatsResponse,
    summary="Get catalog statistics",
    description="Get statistics about the model catalog.",
)
async def get_catalog_stats() -> CatalogStatsResponse:
    """
    Get statistics about the model catalog.

    Returns information about:
    - Total providers and models
    - Models per provider
    - Cache status
    """
    try:
        catalog = get_model_catalog_service()

        all_models = await catalog.get_all_providers_with_models()

        providers_with_models = {provider: len(models) for provider, models in all_models.items()}

        stats = catalog.get_cache_stats()

        return CatalogStatsResponse(
            total_providers=len(all_models),
            total_models=sum(providers_with_models.values()),
            providers_with_models=providers_with_models,
            last_refresh=stats.get("last_full_refresh"),
            cache_ttl_seconds=stats.get("cache_ttl_seconds", 3600),
        )
    except Exception as e:
        logger.error(f"Error getting catalog stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get catalog stats: {e!s}")


@router.post(
    "/providers/{provider_id}/validate",
    summary="Validate provider API key",
    description="Validate that the API key for a provider is working.",
)
async def validate_provider_api_key(
    provider_id: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Validate that a provider's API key is working.

    Args:
        provider_id: The provider to validate
        api_key: Optional API key to test (uses env var if not provided)

    Returns:
        Validation result with status
    """
    try:
        plugin = get_plugin(provider_id)
        if not plugin:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

        # Get API key to test
        key_to_test = api_key or plugin._get_api_key(None)
        if not key_to_test:
            return {
                "valid": False,
                "provider_id": provider_id,
                "message": "No API key configured or provided",
            }

        # Validate the key
        is_valid = await plugin.validate_api_key(key_to_test)

        return {
            "valid": is_valid,
            "provider_id": provider_id,
            "message": ("API key is valid" if is_valid else "API key validation failed"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating API key for {provider_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate API key: {e!s}")


@router.get(
    "/validation/metrics",
    response_model=ValidationMetricsResponse,
    summary="Get validation metrics",
    description="Get metrics about provider/model selection validation.",
)
async def get_validation_metrics() -> ValidationMetricsResponse:
    """
    Get metrics about selection validation.

    Returns statistics including:
    - Total validations and success rate
    - Error type distribution
    - Selection source distribution
    - Provider health at validation time
    - Average validation time
    """
    try:
        from app.middleware.selection_validation_middleware import (
            get_validation_metrics as get_metrics,
        )

        metrics = get_metrics()
        metrics_dict = metrics.to_dict()

        return ValidationMetricsResponse(
            total_validations=metrics_dict["total_validations"],
            successful_validations=metrics_dict["successful_validations"],
            failed_validations=metrics_dict["failed_validations"],
            success_rate=metrics_dict["success_rate"],
            error_counts=metrics_dict["error_counts"],
            source_distribution=metrics_dict["source_distribution"],
            provider_health_at_validation=metrics_dict["provider_health_at_validation"],
            avg_validation_time_ms=metrics_dict["avg_validation_time_ms"],
            last_updated=metrics_dict.get("last_updated"),
        )
    except ImportError:
        # Return empty metrics if middleware not available
        return ValidationMetricsResponse(
            total_validations=0,
            successful_validations=0,
            failed_validations=0,
            success_rate=0.0,
            error_counts={},
            source_distribution={},
            provider_health_at_validation={},
            avg_validation_time_ms=0.0,
            last_updated=None,
        )
    except Exception as e:
        logger.error(f"Error getting validation metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation metrics: {e!s}")


# =============================================================================
# Selection Synchronization Endpoints
# =============================================================================


@router.post(
    "/selection/sync",
    response_model=SelectionSyncResponse,
    summary="Synchronize selection",
    description="Synchronize provider/model selection frontend/backend",
)
async def sync_selection(
    request: SelectionSyncRequest,
) -> SelectionSyncResponse:
    """
    Synchronize selection between frontend and backend.

    This endpoint handles bidirectional sync with optimistic concurrency.
    If version is provided and doesn't match, returns conflict=True.

    Args:
        request: Selection sync request with session_id and selection

    Returns:
        SelectionSyncResponse with current state and conflict status
    """
    try:
        from app.services.concurrent_selection_handler import ConcurrentSelectionHandler

        handler = ConcurrentSelectionHandler()

        # Get current state with version
        current = await handler.get_selection_with_version(request.session_id)
        current_version = current.version if current else 0

        # Check for version conflict
        if request.version is not None and current and request.version != current_version:
            # Return current state without updating
            return SelectionSyncResponse(
                success=False,
                session_id=request.session_id,
                provider=current.provider,
                model=current.model,
                version=current.version,
                conflict=True,
                server_timestamp=datetime.utcnow().isoformat() + "Z",
                message="Version conflict - selection was modified",
            )

        # Perform atomic update
        result = await handler.update_selection_atomic(
            session_id=request.session_id,
            provider=request.provider,
            model=request.model,
            expected_version=request.version,
        )

        return SelectionSyncResponse(
            success=result.success,
            session_id=request.session_id,
            provider=result.provider,
            model=result.model,
            version=result.new_version,
            conflict=result.conflict,
            server_timestamp=datetime.utcnow().isoformat() + "Z",
            message=result.message,
        )

    except Exception as e:
        logger.error(f"Error syncing selection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync selection: {e!s}")


@router.get(
    "/selection/{session_id}",
    response_model=SelectionResponse,
    summary="Get session selection",
    description="Get current provider/model selection for a session.",
)
async def get_session_selection(
    session_id: str,
    user_id: str | None = Query(None, description="User ID for filtering"),
) -> SelectionResponse:
    """
    Get current selection for a session.

    Args:
        session_id: The session identifier
        user_id: Optional user identifier for filtering

    Returns:
        Current selection with version information
    """
    try:
        from app.services.selection_recovery_service import SelectionRecoveryService
        from app.services.session_selection_persistence import SessionSelectionPersistenceService

        persistence = SessionSelectionPersistenceService()
        recovery = SelectionRecoveryService()

        # Try to load selection
        load_result = await persistence.load_selection(
            session_id=session_id,
            user_id=user_id,
        )

        if load_result.found and load_result.record:
            record = load_result.record
            return SelectionResponse(
                session_id=record.session_id,
                provider=record.provider,
                model=record.model,
                version=record.version,
                user_id=record.user_id,
                created_at=record.created_at.isoformat() + "Z",
                updated_at=record.updated_at.isoformat() + "Z",
                metadata=record.metadata,
            )

        # Try to recover selection
        recovered = await recovery.recover_selection(
            session_id=session_id,
            user_id=user_id,
            use_defaults=True,
        )

        if recovered.success:
            now = datetime.utcnow().isoformat() + "Z"
            return SelectionResponse(
                session_id=session_id,
                provider=recovered.provider,
                model=recovered.model,
                version=0,
                user_id=user_id,
                created_at=now,
                updated_at=now,
                metadata={"source": recovered.source},
            )

        raise HTTPException(status_code=404, detail=f"No selection found for session: {session_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting selection for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get selection: {e!s}")


@router.post(
    "/selection/{session_id}",
    response_model=SelectionResponse,
    summary="Set session selection",
    description="Set provider/model selection with optimistic concurrency.",
)
async def set_session_selection(
    session_id: str,
    request: SetSelectionRequest,
) -> SelectionResponse:
    """
    Set selection for a session with optimistic concurrency.

    If expected_version is provided and doesn't match current version,
    returns 409 Conflict with the current state.

    Args:
        session_id: The session identifier
        request: Selection request with provider, model, optional version

    Returns:
        Updated selection with new version
    """
    try:
        from app.services.concurrent_selection_handler import ConcurrentSelectionHandler
        from app.services.session_selection_persistence import SessionSelectionPersistenceService

        handler = ConcurrentSelectionHandler()
        persistence = SessionSelectionPersistenceService()

        # Perform atomic update with optional version check
        result = await handler.update_selection_atomic(
            session_id=session_id,
            provider=request.provider,
            model=request.model,
            expected_version=request.expected_version,
        )

        if result.conflict:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Version conflict",
                    "current_version": result.old_version,
                    "expected_version": request.expected_version,
                    "current_provider": result.provider,
                    "current_model": result.model,
                },
            )

        # Also persist to database
        save_result = await persistence.save_selection(
            session_id=session_id,
            provider=request.provider,
            model=request.model,
            user_id=request.user_id,
            metadata=request.metadata,
        )

        now = datetime.utcnow()
        return SelectionResponse(
            session_id=session_id,
            provider=result.provider,
            model=result.model,
            version=result.new_version,
            user_id=request.user_id,
            created_at=(
                save_result.record.created_at.isoformat() + "Z"
                if save_result.record
                else now.isoformat() + "Z"
            ),
            updated_at=now.isoformat() + "Z",
            metadata=request.metadata or {},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting selection for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set selection: {e!s}")


@router.delete(
    "/selection/{session_id}",
    summary="Delete session selection",
    description="Delete selection for a session.",
)
async def delete_session_selection(
    session_id: str,
) -> dict[str, Any]:
    """
    Delete selection for a session.

    Args:
        session_id: The session identifier

    Returns:
        Deletion confirmation
    """
    try:
        from app.infrastructure.cache.selection_cache import SelectionCache
        from app.services.session_selection_persistence import SessionSelectionPersistenceService

        persistence = SessionSelectionPersistenceService()
        cache = SelectionCache()

        # Delete from persistence
        await persistence.delete_selection(session_id)

        # Delete from cache
        await cache.delete(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "message": "Selection deleted",
        }

    except Exception as e:
        logger.error(f"Error deleting selection for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete selection: {e!s}")


@router.get(
    "/selection/user/{user_id}",
    summary="Get user selections",
    description="Get all selections for a user.",
)
async def get_user_selections(
    user_id: str,
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """
    Get all selections for a user.

    Args:
        user_id: The user identifier
        limit: Maximum number of selections to return

    Returns:
        List of user's selections
    """
    try:
        from app.services.session_selection_persistence import SessionSelectionPersistenceService

        persistence = SessionSelectionPersistenceService()
        selections = await persistence.get_user_selections(user_id)

        return {
            "user_id": user_id,
            "selections": [
                {
                    "session_id": s.session_id,
                    "provider": s.provider,
                    "model": s.model,
                    "version": s.version,
                    "created_at": s.created_at.isoformat() + "Z",
                    "updated_at": s.updated_at.isoformat() + "Z",
                }
                for s in selections[:limit]
            ],
            "total": len(selections),
        }

    except Exception as e:
        logger.error(f"Error getting selections for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user selections: {e!s}")


# =============================================================================
# WebSocket Endpoint for Real-Time Selection Updates
# =============================================================================


class WebSocketConnectionManager:
    """Manager for WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept connection and register for session."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        logger.info(f"WebSocket connected for session: {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove connection from session."""
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session: {session_id}")

    async def broadcast_to_session(
        self,
        session_id: str,
        message: dict,
        exclude: WebSocket | None = None,
    ):
        """Broadcast message to all connections for a session."""
        if session_id not in self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections[session_id]:
            if connection == exclude:
                continue
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn, session_id)


# Global connection manager instance
ws_manager = WebSocketConnectionManager()


@router.websocket("/selection/subscribe/{session_id}")
async def subscribe_selection_changes(
    websocket: WebSocket,
    session_id: str,
):
    """
    WebSocket endpoint for real-time selection updates.

    Clients connect to receive updates when the selection changes.
    Also supports sending selection updates through the WebSocket.

    Protocol:
    - Server: {"type": "selection_changed", "provider": ...}
    - Client: {"type": "update", "provider": ..., "model": ...}
    - Server: {"type": "update_result", "success": ..., "version": ...}
    """
    await ws_manager.connect(websocket, session_id)

    try:
        # Send current selection on connect
        from app.services.concurrent_selection_handler import ConcurrentSelectionHandler

        handler = ConcurrentSelectionHandler()
        current = await handler.get_selection_with_version(session_id)

        if current:
            await websocket.send_json(
                {
                    "type": "current_selection",
                    "session_id": session_id,
                    "provider": current.provider,
                    "model": current.model,
                    "version": current.version,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )

        # Listen for client messages
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "update":
                # Client wants to update selection
                provider = data.get("provider")
                model = data.get("model")
                expected_version = data.get("version")

                if provider and model:
                    result = await handler.update_selection_atomic(
                        session_id=session_id,
                        provider=provider,
                        model=model,
                        expected_version=expected_version,
                    )

                    # Send result to client
                    await websocket.send_json(
                        {
                            "type": "update_result",
                            "success": result.success,
                            "provider": result.provider,
                            "model": result.model,
                            "version": result.new_version,
                            "conflict": result.conflict,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }
                    )

                    # Broadcast to other clients if successful
                    if result.success and not result.conflict:
                        ts = datetime.utcnow().isoformat() + "Z"
                        await ws_manager.broadcast_to_session(
                            session_id,
                            {
                                "type": "selection_changed",
                                "session_id": session_id,
                                "provider": result.provider,
                                "model": result.model,
                                "version": result.new_version,
                                "timestamp": ts,
                            },
                            exclude=websocket,
                        )
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Missing provider or model",
                        }
                    )

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        ws_manager.disconnect(websocket, session_id)


# =============================================================================
# Selection Recovery and Cleanup Endpoints
# =============================================================================


@router.post(
    "/selection/recover/{session_id}",
    response_model=SelectionResponse,
    summary="Recover session selection",
    description="Attempt to recover selection from database/cache/defaults.",
)
async def recover_session_selection(
    session_id: str,
    user_id: str | None = Query(None, description="User ID for recovery"),
) -> SelectionResponse:
    """
    Recover selection for a session.

    Tries multiple sources: cache, database, user defaults, system defaults.

    Args:
        session_id: The session identifier
        user_id: Optional user identifier

    Returns:
        Recovered selection
    """
    try:
        from app.services.selection_recovery_service import SelectionRecoveryService

        recovery = SelectionRecoveryService()
        result = await recovery.recover_selection(
            session_id=session_id,
            user_id=user_id,
            use_defaults=True,
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=f"Recovery failed: {result.error}")

        now = datetime.utcnow().isoformat() + "Z"
        return SelectionResponse(
            session_id=session_id,
            provider=result.provider,
            model=result.model,
            version=0,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            metadata={
                "source": result.source,
                "was_migrated": result.was_migrated,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recovering selection for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to recover selection: {e!s}")


@router.post(
    "/selection/cleanup",
    summary="Cleanup expired selections",
    description="Remove expired session selections.",
)
async def cleanup_expired_selections(
    max_age_hours: int = Query(24, ge=1, le=168),
) -> dict[str, Any]:
    """
    Cleanup expired session selections.

    Args:
        max_age_hours: Maximum age in hours before cleanup

    Returns:
        Cleanup results
    """
    try:
        from app.services.session_selection_persistence import SessionSelectionPersistenceService

        persistence = SessionSelectionPersistenceService()
        deleted_count = await persistence.cleanup_expired_sessions(max_age_hours=max_age_hours)

        return {
            "success": True,
            "deleted_count": deleted_count,
            "max_age_hours": max_age_hours,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        logger.error(f"Error cleaning up selections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup selections: {e!s}")


@router.get(
    "/selection/recovery/stats",
    summary="Get recovery statistics",
    description="Get statistics about selection recovery operations.",
)
async def get_recovery_stats() -> dict[str, Any]:
    """
    Get statistics about selection recovery.

    Returns:
        Recovery statistics including sources and counts
    """
    try:
        from app.services.selection_recovery_service import SelectionRecoveryService

        recovery = SelectionRecoveryService()
        stats = recovery.get_recovery_stats()

        return {
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        logger.error(f"Error getting recovery stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery stats: {e!s}")


# =============================================================================
# Base Selection Endpoint
# =============================================================================


@router.get(
    "/selection/current",
    response_model=CurrentSelectionResponse,
    summary="Get current selection",
    description="Get current provider/model selection for a session.",
)
async def get_current_selection(
    session_id: str | None = Query(None, description="Session ID for lookup"),
    user_id: str | None = Query(None, description="User ID for filtering"),
) -> CurrentSelectionResponse:
    """
    Get current selection for a session.

    This endpoint returns the current provider/model selection with the
    following priority:
    1. Session-specific selection (if session_id provided and found)
    2. User default selection (if user_id provided)
    3. Global default selection

    Args:
        session_id: Optional session identifier
        user_id: Optional user identifier for filtering

    Returns:
        CurrentSelectionResponse with provider_id, model_id, scope, etc.
    """
    try:
        from app.services.selection_recovery_service import SelectionRecoveryService
        from app.services.session_selection_persistence import SessionSelectionPersistenceService

        persistence = SessionSelectionPersistenceService()
        recovery = SelectionRecoveryService()

        # If session_id provided, try to load session-specific selection
        if session_id:
            load_result = await persistence.load_selection(
                session_id=session_id,
                user_id=user_id,
            )

            if load_result.found and load_result.record:
                record = load_result.record
                return CurrentSelectionResponse(
                    provider_id=record.provider,
                    model_id=record.model,
                    scope="SESSION",
                    session_id=record.session_id,
                    user_id=record.user_id,
                    created_at=record.created_at.isoformat() + "Z",
                    updated_at=record.updated_at.isoformat() + "Z",
                    version=record.version,
                    metadata=record.metadata,
                )

        # Try to recover selection (uses cache, database, or defaults)
        recovered = await recovery.recover_selection(
            session_id=session_id,
            user_id=user_id,
            use_defaults=True,
        )

        if recovered.success:
            now = datetime.utcnow().isoformat() + "Z"
            scope = "SESSION" if session_id else "GLOBAL"
            if recovered.source == "default":
                scope = "GLOBAL"

            return CurrentSelectionResponse(
                provider_id=recovered.provider,
                model_id=recovered.model,
                scope=scope,
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                updated_at=now,
                version=0,
                metadata={"source": recovered.source},
            )

        # Return default selection as fallback
        # Get default from first available provider
        plugins = get_all_plugins()
        if plugins:
            first_plugin = plugins[0]
            default_model = first_plugin.get_default_model()
            now = datetime.utcnow().isoformat() + "Z"

            return CurrentSelectionResponse(
                provider_id=first_plugin.provider_type,
                model_id=default_model or "unknown",
                scope="GLOBAL",
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                updated_at=now,
                version=0,
                metadata={"source": "default_fallback"},
            )

        # Absolute fallback
        now = datetime.utcnow().isoformat() + "Z"
        return CurrentSelectionResponse(
            provider_id="openai",
            model_id="gpt-4",
            scope="GLOBAL",
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            version=0,
            metadata={"source": "hardcoded_fallback"},
        )

    except Exception as e:
        logger.error(f"Error getting current selection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current selection: {e!s}")


@router.post(
    "/selection",
    response_model=CurrentSelectionResponse,
    summary="Save provider/model selection",
    description="Save provider/model selection with optional session scope.",
)
async def save_selection(
    request: SelectionSyncRequest,
) -> CurrentSelectionResponse:
    """
    Save provider/model selection.

    This endpoint provides compatibility with the frontend's expected API.
    It delegates to the session-specific selection logic.

    Args:
        request: Selection request with provider, model, and optional session_id

    Returns:
        CurrentSelectionResponse with the saved selection
    """
    try:
        from app.services.concurrent_selection_handler import ConcurrentSelectionHandler
        from app.services.session_selection_persistence import SessionSelectionPersistenceService

        # Generate session ID if not provided
        session_id = request.session_id or f"frontend-{datetime.utcnow().timestamp()}"

        handler = ConcurrentSelectionHandler()
        persistence = SessionSelectionPersistenceService()

        # Perform atomic update
        result = await handler.update_selection_atomic(
            session_id=session_id,
            provider=request.provider,
            model=request.model,
            expected_version=request.version,
        )

        if not result.success:
            raise HTTPException(
                status_code=409 if result.conflict else 500,
                detail=result.message or "Failed to save selection",
            )

        # Also persist to database
        save_result = await persistence.save_selection(
            session_id=session_id,
            provider=request.provider,
            model=request.model,
            user_id=request.user_id,
            metadata={"source": request.source},
        )

        now = datetime.utcnow()
        return CurrentSelectionResponse(
            provider_id=result.provider,
            model_id=result.model,
            scope="SESSION",
            session_id=session_id,
            user_id=request.user_id,
            created_at=(
                save_result.record.created_at.isoformat() + "Z"
                if save_result.record
                else now.isoformat() + "Z"
            ),
            updated_at=now.isoformat() + "Z",
            version=result.new_version,
            metadata={"source": request.source},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving selection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save selection: {e!s}")


@router.on_event("startup")
async def startup_initialize_catalog():
    """Initialize the model catalog on application startup."""
    try:
        logger.info("Initializing unified provider system...")
        register_all_plugins()
        await initialize_model_catalog()
        logger.info("Unified provider system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize provider system: {e}")
