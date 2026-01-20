"""Connection Management API Endpoints
Direct-only API connection management.
Includes provider endpoint management and validation.
"""

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.auth import TokenPayload, get_current_user
from app.core.config import get_settings
from app.services.connection_manager import get_connection_manager

# Router without global auth - auth is added per-endpoint where needed
router = APIRouter()


# Request/Response Models
class ConnectionConfigResponse(BaseModel):
    """Current connection configuration."""

    current_mode: str
    direct: dict
    providers: dict | None = None


class ConnectionStatusResponse(BaseModel):
    """Connection status response."""

    mode: str
    is_connected: bool
    base_url: str
    error_message: str | None = None
    latency_ms: float | None = None
    available_models: list | None = None


class ProviderEndpointInfo(BaseModel):
    """Information about a provider endpoint."""

    direct: str


class EndpointsListResponse(BaseModel):
    """Response listing all provider endpoints."""

    providers: dict[str, ProviderEndpointInfo]


class ValidateEndpointRequest(BaseModel):
    """Request to validate a provider endpoint."""

    provider: str = Field(..., description="Provider name (e.g., 'google', 'openai')")


class ValidateEndpointResponse(BaseModel):
    """Response from endpoint validation."""

    provider: str
    url: str
    is_valid: bool
    error_message: str | None = None


class ProviderHealthResponse(BaseModel):
    """Response from provider health check."""

    provider: str
    is_healthy: bool
    latency_ms: float | None = None
    error_message: str | None = None


# Endpoints
@router.get("/config", response_model=ConnectionConfigResponse)
async def get_connection_config():
    """Get current connection configuration.

    Returns the current direct connection configuration.
    """
    manager = get_connection_manager()
    config = manager.get_config_summary()
    return ConnectionConfigResponse(**config)


@router.get("/status", response_model=ConnectionStatusResponse)
async def get_connection_status():
    """Get current connection status.

    Checks the active connection and returns status information including
    latency and available models.
    """
    manager = get_connection_manager()
    status_result = await manager.check_connection()

    return ConnectionStatusResponse(
        mode=status_result.mode,
        is_connected=status_result.is_connected,
        base_url=status_result.base_url,
        error_message=status_result.error_message,
        latency_ms=status_result.latency_ms,
        available_models=status_result.available_models,
    )


@router.get("/health")
async def connection_health():
    """Quick health check for current connection.

    Returns a simple status without detailed model information.
    """
    manager = get_connection_manager()
    status_result = await manager.check_connection()

    return {
        "healthy": status_result.is_connected,
        "mode": status_result.mode,
        "latency_ms": status_result.latency_ms,
        "error": status_result.error_message,
    }


@router.get("/endpoints", response_model=EndpointsListResponse)
async def list_provider_endpoints():
    """List all configured direct endpoints for all providers."""
    cfg = get_settings()
    all_endpoints = cfg.get_all_provider_endpoints()

    providers = {}
    for provider, url in all_endpoints.items():
        providers[provider] = ProviderEndpointInfo(direct=url)

    return EndpointsListResponse(providers=providers)


@router.post("/endpoints/validate", response_model=ValidateEndpointResponse)
async def validate_provider_endpoint(
    request: ValidateEndpointRequest,
    _user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Validate a provider endpoint URL format and configuration.

    Checks if the direct endpoint URL for the specified provider
    is properly formatted and configured.
    """
    cfg = get_settings()
    endpoint = cfg.get_provider_endpoint(request.provider)

    if not endpoint:
        return ValidateEndpointResponse(
            provider=request.provider,
            url="",
            is_valid=False,
            error_message=f"No endpoint configured for {request.provider}",
        )

    is_valid = cfg.validate_endpoint_url(endpoint)
    error_msg = None if is_valid else f"Invalid URL format: {endpoint}"

    return ValidateEndpointResponse(
        provider=request.provider,
        url=endpoint,
        is_valid=is_valid,
        error_message=error_msg,
    )


@router.post("/endpoints/test/{provider}", response_model=ProviderHealthResponse)
async def test_provider_connection(
    provider: str, _user: Annotated[TokenPayload, Depends(get_current_user)]
):
    """Test connectivity for a specific provider endpoint.

    Performs a health check against the provider's direct endpoint
    to verify it's reachable and responding.
    """
    import time as timer

    cfg = get_settings()
    endpoint = cfg.get_provider_endpoint(provider)

    if not endpoint:
        return ProviderHealthResponse(
            provider=provider,
            is_healthy=False,
            error_message=f"No endpoint for {provider}",
        )

    start_time = timer.time()
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.get(endpoint)
            latency = (timer.time() - start_time) * 1000

            # Treat 2xx, 401, 403 as "healthy" (server responded)
            is_healthy = response.status_code < 500

            return ProviderHealthResponse(
                provider=provider,
                is_healthy=is_healthy,
                latency_ms=latency,
                error_message=(None if is_healthy else f"Server error: {response.status_code}"),
            )
    except httpx.ConnectError as e:
        latency = (timer.time() - start_time) * 1000
        return ProviderHealthResponse(
            provider=provider,
            is_healthy=False,
            latency_ms=latency,
            error_message=f"Connection failed: {e!s}",
        )
    except httpx.TimeoutException:
        latency = (timer.time() - start_time) * 1000
        return ProviderHealthResponse(
            provider=provider,
            is_healthy=False,
            latency_ms=latency,
            error_message="Connection timed out",
        )
    except Exception as e:
        return ProviderHealthResponse(
            provider=provider,
            is_healthy=False,
            error_message=f"Unexpected error: {e!s}",
        )
