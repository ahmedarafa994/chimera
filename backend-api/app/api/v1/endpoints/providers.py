"""
Provider Management API Endpoints

Provides endpoints for:
- Listing available providers (Gemini, DeepSeek)
- Getting provider-specific models
- Selecting provider/model for session
- Provider health status
- Real-time model selection sync via WebSocket
- Per-model rate limiting
- Graceful degradation with fallback suggestions
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    Header,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.services.model_rate_limiter import (
    ModelRateLimiter,
    get_model_rate_limiter,
)
from app.services.model_router_service import (
    ModelRouterService,
    ModelSelectionEvent,
    get_model_router_service,
)
from app.services.session_service import session_service

router = APIRouter(prefix="/providers", tags=["providers"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ProviderModel(BaseModel):
    """Model information within a provider"""

    id: str
    name: str
    description: str | None = None
    max_tokens: int = 4096
    is_default: bool = False
    tier: str = "standard"


class ProviderInfo(BaseModel):
    """Provider information with models"""

    provider: str
    display_name: str
    status: str
    is_healthy: bool
    models: list[str]
    default_model: str | None
    latency_ms: float | None = None


class ProvidersListResponse(BaseModel):
    """Response for listing all providers"""

    providers: list[ProviderInfo]
    count: int
    default_provider: str
    default_model: str


class ProviderModelsResponse(BaseModel):
    """Response for provider-specific models"""

    provider: str
    display_name: str
    models: list[ProviderModel]
    default_model: str | None
    count: int


class SelectProviderRequest(BaseModel):
    """Request to select a provider and model"""

    provider: str = Field(..., description="Provider identifier (gemini or deepseek)")
    model: str = Field(..., description="Model identifier")


class SelectProviderResponse(BaseModel):
    """Response after selecting a provider"""

    success: bool
    message: str
    provider: str
    model: str
    session_id: str


class CurrentSelectionResponse(BaseModel):
    """Response for current provider/model selection"""

    provider: str
    model: str
    display_name: str
    session_id: str | None
    is_default: bool


class ProviderHealthResponse(BaseModel):
    """Response for provider health status"""

    providers: list[dict[str, Any]]
    timestamp: str


class RateLimitInfoResponse(BaseModel):
    """Response for rate limit information"""

    allowed: bool
    remaining_requests: int
    remaining_tokens: int
    reset_at: str
    retry_after_seconds: int | None = None
    limit_type: str | None = None
    tier: str = "free"
    fallback_provider: str | None = None


class WebSocketMessage(BaseModel):
    """WebSocket message format"""

    type: str
    data: dict[str, Any]
    timestamp: str


# ============================================================================
# Helper Functions
# ============================================================================


def get_session_id(
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
) -> str | None:
    """Extract session ID from header or cookie"""
    return x_session_id or chimera_session_id


def get_or_create_session_id(
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
) -> str:
    """Get existing session ID or create a new one"""
    session_id = x_session_id or chimera_session_id
    if not session_id:
        # Create a new session
        session = session_service.create_session()
        session_id = session.session_id
    return session_id


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/", response_model=ProvidersListResponse)
async def get_available_providers(
    router_service: ModelRouterService = Depends(get_model_router_service),
):
    """
    Get all available providers with their health status.

    Returns a list of supported providers (Gemini and DeepSeek) with:
    - Available models for each provider
    - Health status and latency information
    - Default model for each provider

    Use this endpoint on frontend initialization to populate the provider
    selection UI.
    """
    await router_service.initialize()

    providers_data = router_service.get_supported_providers()
    health_data = router_service.get_provider_health()

    # Merge health data into provider info
    health_map = {h["provider"]: h for h in health_data}

    providers = []
    for p in providers_data:
        health = health_map.get(p["provider"], {})
        providers.append(
            ProviderInfo(
                provider=p["provider"],
                display_name=p["display_name"],
                status=p["status"],
                is_healthy=p["is_healthy"],
                models=p["models"],
                default_model=p["default_model"],
                latency_ms=health.get("latency_ms"),
            )
        )

    default_provider, default_model = session_service.get_default_model()

    return ProvidersListResponse(
        providers=providers,
        count=len(providers),
        default_provider=default_provider,
        default_model=default_model,
    )


@router.get("/{provider}/models", response_model=ProviderModelsResponse)
async def get_provider_models(
    provider: str, router_service: ModelRouterService = Depends(get_model_router_service)
):
    """
    Get available models for a specific provider.

    Args:
        provider: Provider identifier (gemini or deepseek)

    Returns detailed model information including:
    - Model ID and display name
    - Maximum token limits
    - Model tier (standard/premium/experimental)
    """
    await router_service.initialize()

    # Normalize provider name
    normalized = router_service._normalize_provider(provider)

    if normalized not in router_service.SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' not supported. Use 'gemini' or 'deepseek'.",
        )

    # Get provider data
    providers_data = router_service.get_supported_providers()
    provider_data = next((p for p in providers_data if p["provider"] == normalized), None)

    if not provider_data:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")

    # Get detailed model info from session service
    master_list = session_service.get_master_model_list()
    lookup_key = "google" if normalized == "gemini" else normalized
    model_details = master_list.get(lookup_key, [])

    models = []
    for model_id in provider_data["models"]:
        detail = next((m for m in model_details if m["id"] == model_id), None)
        models.append(
            ProviderModel(
                id=model_id,
                name=detail["name"] if detail else model_id,
                description=detail.get("description") if detail else None,
                max_tokens=detail.get("max_tokens", 4096) if detail else 4096,
                is_default=detail.get("is_default", False) if detail else False,
                tier=detail.get("tier", "standard") if detail else "standard",
            )
        )

    display_name = "Google Gemini" if normalized == "gemini" else "DeepSeek"

    return ProviderModelsResponse(
        provider=normalized,
        display_name=display_name,
        models=models,
        default_model=provider_data["default_model"],
        count=len(models),
    )


@router.post("/select", response_model=SelectProviderResponse)
async def select_provider(
    request: SelectProviderRequest,
    response: Response,
    session_id: str = Depends(get_or_create_session_id),
    router_service: ModelRouterService = Depends(get_model_router_service),
    rate_limiter: ModelRateLimiter = Depends(get_model_rate_limiter),
):
    """
    Select a provider and model for the current session.

    This endpoint:
    1. Checks rate limits for the selected provider
    2. Validates the provider/model combination
    3. Updates the session with the new selection
    4. Broadcasts the change to connected WebSocket clients

    All subsequent requests for this session will use the selected provider/model.

    Rate limit headers are included in the response.
    """
    await router_service.initialize()

    # Check rate limits before selection
    user_id = session_id  # Use session_id as user identifier
    tier = "free"  # Default tier, could be fetched from user profile

    rate_result = await rate_limiter.check_rate_limit(
        user_id=user_id,
        provider=request.provider,
        model=request.model,
        tier=tier,
    )

    # Add rate limit headers to response
    rate_headers = rate_limiter.get_rate_limit_headers(rate_result, request.provider)
    for header, value in rate_headers.items():
        response.headers[header] = value

    if not rate_result.allowed:
        # Suggest fallback provider
        fallback = rate_limiter.suggest_fallback_provider(
            current_provider=request.provider,
            user_id=user_id,
            tier=tier,
        )

        error_detail = {
            "message": f"Rate limit exceeded for provider '{request.provider}'",
            "retry_after": rate_result.retry_after_seconds,
            "limit_type": rate_result.limit_type,
            "fallback_provider": fallback,
        }

        raise HTTPException(
            status_code=429,
            detail=error_detail,
            headers={"Retry-After": str(rate_result.retry_after_seconds or 60)},
        )

    success, message, session_info = await router_service.select_model(
        session_id=session_id,
        provider=request.provider,
        model=request.model,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    # Record the request for rate limiting
    await rate_limiter.record_request(
        user_id=user_id,
        provider=request.provider,
        model=request.model,
    )

    return SelectProviderResponse(
        success=True,
        message=message,
        provider=session_info.get("provider", request.provider),
        model=session_info.get("model", request.model),
        session_id=session_id,
    )


@router.get("/rate-limit", response_model=RateLimitInfoResponse)
async def check_rate_limit(
    provider: str,
    model: str,
    session_id: str | None = Depends(get_session_id),
    rate_limiter: ModelRateLimiter = Depends(get_model_rate_limiter),
):
    """
    Check rate limit status for a provider/model combination.

    Returns current rate limit status including:
    - Whether requests are allowed
    - Remaining requests and tokens
    - When limits reset
    - Suggested fallback provider if rate limited
    """
    user_id = session_id or "anonymous"
    tier = "free"

    rate_result = await rate_limiter.check_rate_limit(
        user_id=user_id,
        provider=provider,
        model=model,
        tier=tier,
    )

    fallback = None
    if not rate_result.allowed:
        fallback = rate_limiter.suggest_fallback_provider(
            current_provider=provider,
            user_id=user_id,
            tier=tier,
        )

    return RateLimitInfoResponse(
        allowed=rate_result.allowed,
        remaining_requests=rate_result.remaining_requests,
        remaining_tokens=rate_result.remaining_tokens,
        reset_at=rate_result.reset_at.isoformat(),
        retry_after_seconds=rate_result.retry_after_seconds,
        limit_type=rate_result.limit_type,
        tier=tier,
        fallback_provider=fallback,
    )


@router.get("/current", response_model=CurrentSelectionResponse)
async def get_current_selection(
    session_id: str | None = Depends(get_session_id),
    router_service: ModelRouterService = Depends(get_model_router_service),
):
    """
    Get the current provider/model selection for the session.

    Returns the currently selected provider and model, or defaults if
    no session exists.
    """
    await router_service.initialize()

    is_default = False

    if session_id:
        provider, model = session_service.get_session_model(session_id)
    else:
        provider, model = session_service.get_default_model()
        is_default = True

    # Normalize and get display name
    normalized = router_service._normalize_provider(provider)
    display_name = "Google Gemini" if normalized == "gemini" else "DeepSeek"

    return CurrentSelectionResponse(
        provider=normalized,
        model=model,
        display_name=display_name,
        session_id=session_id,
        is_default=is_default,
    )


@router.get("/health", response_model=ProviderHealthResponse)
async def get_provider_health(
    router_service: ModelRouterService = Depends(get_model_router_service),
):
    """
    Get health status for all providers.

    Returns health information including:
    - Current health status (healthy/unhealthy)
    - Last health check timestamp
    - Response latency
    - Consecutive failure count
    """
    await router_service.initialize()

    health_data = router_service.get_provider_health()

    return ProviderHealthResponse(
        providers=health_data,
        timestamp=datetime.utcnow().isoformat(),
    )


# ============================================================================
# WebSocket for Real-time Model Selection Sync
# ============================================================================

# Store active WebSocket connections
_active_connections: dict[str, WebSocket] = {}


@router.websocket("/ws/selection")
async def websocket_selection_sync(websocket: WebSocket):
    """
    WebSocket endpoint for real-time model selection synchronization.

    Clients connect to receive:
    - Model selection change notifications
    - Provider health updates
    - Periodic heartbeat messages

    Message format:
    {
        "type": "selection_change" | "health_update" | "heartbeat",
        "data": { ... },
        "timestamp": "ISO timestamp"
    }
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())
    _active_connections[client_id] = websocket

    logger.info(f"WebSocket client connected for selection sync: {client_id}")

    # Get router service
    router_service = get_model_router_service()
    await router_service.initialize()

    # Subscribe to selection changes
    async def on_selection_change(event: ModelSelectionEvent):
        message = WebSocketMessage(
            type="selection_change",
            data={
                "session_id": event.session_id,
                "provider": event.provider,
                "model": event.model,
                "previous_provider": event.previous_provider,
                "previous_model": event.previous_model,
            },
            timestamp=event.timestamp.isoformat(),
        )
        try:
            await websocket.send_json(message.model_dump())
        except Exception as e:
            logger.error(f"Failed to send selection change to {client_id}: {e}")

    unsubscribe = router_service.subscribe_to_selection_changes(on_selection_change)

    try:
        # Send initial connection confirmation
        await websocket.send_json(
            {
                "type": "connected",
                "data": {"client_id": client_id},
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Main loop: handle incoming messages and send heartbeats
        heartbeat_interval = 30  # seconds
        last_heartbeat = datetime.utcnow()

        while True:
            try:
                # Wait for message with timeout for heartbeat
                message = await asyncio.wait_for(
                    websocket.receive_json(), timeout=heartbeat_interval
                )

                # Handle client messages
                msg_type = message.get("type")

                if msg_type == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "data": {},
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                elif msg_type == "get_health":
                    health_data = router_service.get_provider_health()
                    await websocket.send_json(
                        {
                            "type": "health_update",
                            "data": {"providers": health_data},
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                elif msg_type == "get_current":
                    session_id = message.get("data", {}).get("session_id")
                    if session_id:
                        provider, model = session_service.get_session_model(session_id)
                    else:
                        provider, model = session_service.get_default_model()

                    await websocket.send_json(
                        {
                            "type": "current_selection",
                            "data": {
                                "provider": provider,
                                "model": model,
                                "session_id": session_id,
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

            except TimeoutError:
                # Send heartbeat
                now = datetime.utcnow()
                if (now - last_heartbeat).total_seconds() >= heartbeat_interval:
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "data": {},
                            "timestamp": now.isoformat(),
                        }
                    )
                    last_heartbeat = now

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        # Cleanup
        unsubscribe()
        _active_connections.pop(client_id, None)


async def broadcast_to_all_clients(message: dict):
    """Broadcast a message to all connected WebSocket clients"""
    disconnected = []

    for client_id, websocket in _active_connections.items():
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to broadcast to {client_id}: {e}")
            disconnected.append(client_id)

    # Remove disconnected clients
    for client_id in disconnected:
        _active_connections.pop(client_id, None)
