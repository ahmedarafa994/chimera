# =============================================================================
# Chimera - API Key Failover Status Endpoints
# =============================================================================
# REST API endpoints for exposing failover status and history.
# Allows monitoring of automatic API key failover events and manual reset.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# Subtask 3.3: Create Failover Status Endpoints
# =============================================================================

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import TokenPayload, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/failover", tags=["failover", "api-keys"])


# =============================================================================
# Response Models
# =============================================================================


class ProviderFailoverStatusResponse(BaseModel):
    """Failover status for a single provider."""

    provider_id: str = Field(..., description="Provider identifier")
    configured: bool = Field(..., description="Whether failover is configured for this provider")
    current_key_id: str | None = Field(default=None, description="Currently active key ID")
    primary_key_id: str | None = Field(default=None, description="Primary key ID")
    is_using_backup: bool = Field(default=False, description="Whether currently using a backup key")
    last_failover_at: str | None = Field(default=None, description="When last failover occurred (ISO format)")
    failover_count: int = Field(default=0, ge=0, description="Total failover count for this provider")
    strategy: str = Field(default="priority", description="Failover strategy (priority, round_robin, least_used, random)")
    config: dict[str, Any] = Field(default_factory=dict, description="Failover configuration")
    key_cooldowns: list[dict[str, Any]] = Field(default_factory=list, description="Key cooldown states")


class AllFailoverStatusResponse(BaseModel):
    """Failover status for all providers."""

    providers: dict[str, ProviderFailoverStatusResponse] = Field(
        default_factory=dict, description="Failover status per provider"
    )
    summary: dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    updated_at: str = Field(..., description="Last update time (ISO format)")


class FailoverEventResponse(BaseModel):
    """A single failover event."""

    event_id: str = Field(..., description="Unique event identifier")
    provider_id: str = Field(..., description="Provider identifier")
    from_key_id: str = Field(..., description="Key that was failed over from")
    to_key_id: str | None = Field(default=None, description="Key that was failed over to")
    reason: str = Field(..., description="Reason for failover")
    error_message: str | None = Field(default=None, description="Error message that triggered failover")
    triggered_at: str = Field(..., description="When failover occurred (ISO format)")
    cooldown_until: str | None = Field(default=None, description="When the from_key will be available again")
    success: bool = Field(..., description="Whether failover succeeded")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")


class FailoverHistoryResponse(BaseModel):
    """Failover event history response."""

    events: list[FailoverEventResponse] = Field(default_factory=list, description="List of failover events")
    total_count: int = Field(default=0, ge=0, description="Total number of events returned")
    provider_id: str | None = Field(default=None, description="Provider filter (if specified)")
    reason: str | None = Field(default=None, description="Reason filter (if specified)")


class ProviderResetResponse(BaseModel):
    """Response for provider reset operation."""

    success: bool = Field(..., description="Whether the reset operation succeeded")
    provider_id: str = Field(..., description="Provider identifier")
    message: str = Field(..., description="Result message")
    reset_at: str = Field(..., description="Reset time (ISO format)")


class CooldownClearResponse(BaseModel):
    """Response for cooldown clear operation."""

    success: bool = Field(..., description="Whether the cooldown was cleared")
    key_id: str = Field(..., description="Key identifier")
    message: str = Field(..., description="Result message")


class FailoverConfigUpdateRequest(BaseModel):
    """Request to update failover configuration for a provider."""

    enabled: bool = Field(default=True, description="Whether failover is enabled")
    strategy: str = Field(
        default="priority",
        description="Failover strategy: priority, round_robin, least_used, random",
    )
    rate_limit_cooldown_seconds: int = Field(
        default=60, ge=5, le=3600, description="Cooldown for rate limit errors"
    )
    error_cooldown_seconds: int = Field(
        default=30, ge=5, le=3600, description="Cooldown for transient errors"
    )
    max_cooldown_seconds: int = Field(
        default=600, ge=60, le=7200, description="Maximum cooldown duration"
    )
    use_exponential_backoff: bool = Field(
        default=True, description="Use exponential backoff for cooldown"
    )
    max_consecutive_failures: int = Field(
        default=5, ge=1, le=20, description="Max consecutive failures before permanent block"
    )
    auto_recover: bool = Field(
        default=True, description="Automatically try to recover to primary key"
    )


class FailoverConfigResponse(BaseModel):
    """Response for failover configuration update."""

    success: bool = Field(..., description="Whether the update succeeded")
    provider_id: str = Field(..., description="Provider identifier")
    config: dict[str, Any] = Field(..., description="Updated configuration")


# =============================================================================
# Service Dependency
# =============================================================================


def get_failover_service():
    """Get the API key failover service instance."""
    from app.services.api_key_failover_service import get_api_key_failover_service
    return get_api_key_failover_service()


# =============================================================================
# Status Endpoints
# =============================================================================


@router.get(
    "/status",
    response_model=AllFailoverStatusResponse,
    summary="Current failover state for all providers",
    description="""
Get the current failover status for all configured providers.

Returns per-provider information including:
- Current active key ID
- Primary key ID
- Whether currently using a backup key
- Last failover timestamp
- Total failover count
- Failover configuration
- Key cooldown states

**Use Cases**:
- Monitoring failover status across all providers
- Dashboard display of failover states
- Alerting on providers using backup keys
""",
    responses={
        200: {"description": "Failover status retrieved successfully"},
        401: {"description": "Authentication required"},
    },
)
async def get_all_failover_status(
    user: TokenPayload = Depends(get_current_user),
) -> AllFailoverStatusResponse:
    """Get current failover state for all providers."""
    logger.debug(f"Getting all failover status for user {user.sub}")

    try:
        failover_service = get_failover_service()
        all_status = await failover_service.get_all_failover_status()

        # Convert provider statuses to response models
        providers_response = {}
        for provider_id, provider_status in all_status.get("providers", {}).items():
            providers_response[provider_id] = ProviderFailoverStatusResponse(
                provider_id=provider_id,
                configured=provider_status.get("configured", False),
                current_key_id=provider_status.get("current_key_id"),
                primary_key_id=provider_status.get("primary_key_id"),
                is_using_backup=provider_status.get("is_using_backup", False),
                last_failover_at=provider_status.get("last_failover_at"),
                failover_count=provider_status.get("failover_count", 0),
                strategy=provider_status.get("strategy", "priority"),
                config=provider_status.get("config", {}),
                key_cooldowns=provider_status.get("key_cooldowns", []),
            )

        return AllFailoverStatusResponse(
            providers=providers_response,
            summary={
                "total_providers": len(providers_response),
                "total_failovers": all_status.get("total_failovers", 0),
                "using_backup_count": all_status.get("using_backup_count", 0),
            },
            updated_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get failover status: {e}", exc_info=True)
        return AllFailoverStatusResponse(
            providers={},
            summary={"error": str(e)},
            updated_at=datetime.utcnow().isoformat(),
        )


@router.get(
    "/status/{provider_id}",
    response_model=ProviderFailoverStatusResponse,
    summary="Failover status for a specific provider",
    description="""
Get the current failover status for a specific provider.

Returns detailed information about the provider's failover state including
key cooldowns and configuration.

**Use Cases**:
- Provider detail views
- Troubleshooting specific providers
- Checking failover readiness
""",
    responses={
        200: {"description": "Provider failover status retrieved"},
        401: {"description": "Authentication required"},
        404: {"description": "Provider not configured"},
    },
)
async def get_provider_failover_status(
    provider_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> ProviderFailoverStatusResponse:
    """Get failover status for a specific provider."""
    logger.debug(f"Getting failover status for provider {provider_id} by user {user.sub}")

    try:
        failover_service = get_failover_service()
        provider_status = await failover_service.get_failover_status(provider_id)

        return ProviderFailoverStatusResponse(
            provider_id=provider_id,
            configured=provider_status.get("configured", False),
            current_key_id=provider_status.get("current_key_id"),
            primary_key_id=provider_status.get("primary_key_id"),
            is_using_backup=provider_status.get("is_using_backup", False),
            last_failover_at=provider_status.get("last_failover_at"),
            failover_count=provider_status.get("failover_count", 0),
            strategy=provider_status.get("strategy", "priority"),
            config=provider_status.get("config", {}),
            key_cooldowns=provider_status.get("key_cooldowns", []),
        )

    except Exception as e:
        logger.error(f"Failed to get provider failover status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provider failover status",
        )


# =============================================================================
# History Endpoints
# =============================================================================


@router.get(
    "/history",
    response_model=FailoverHistoryResponse,
    summary="Recent failover events",
    description="""
Get recent failover event history.

Query Parameters:
- `provider_id`: Filter by specific provider (optional)
- `reason`: Filter by failover reason (rate_limited, error, timeout, etc.)
- `limit`: Maximum events to return (default: 50, max: 200)

**Use Cases**:
- Failover event log dashboard
- Incident investigation
- Trend analysis
""",
    responses={
        200: {"description": "Failover history retrieved successfully"},
        401: {"description": "Authentication required"},
    },
)
async def get_failover_history(
    provider_id: str | None = Query(None, description="Filter by provider ID"),
    reason: str | None = Query(
        None,
        description="Filter by reason (rate_limited, error, timeout, circuit_breaker_open, key_inactive, key_expired, manual)",
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
    user: TokenPayload = Depends(get_current_user),
) -> FailoverHistoryResponse:
    """Get recent failover events."""
    logger.debug(f"Getting failover history for user {user.sub}, provider={provider_id}, reason={reason}")

    try:
        failover_service = get_failover_service()

        # Convert reason string to enum if provided
        reason_enum = None
        if reason:
            try:
                from app.services.api_key_failover_service import FailoverReason
                reason_enum = FailoverReason(reason)
            except ValueError:
                # Invalid reason, just don't filter
                logger.warning(f"Invalid failover reason filter: {reason}")

        history = await failover_service.get_failover_history(
            provider_id=provider_id,
            reason=reason_enum,
            limit=limit,
        )

        # Convert to response models
        events = [
            FailoverEventResponse(
                event_id=event.get("event_id", ""),
                provider_id=event.get("provider_id", ""),
                from_key_id=event.get("from_key_id", ""),
                to_key_id=event.get("to_key_id"),
                reason=event.get("reason", "unknown"),
                error_message=event.get("error_message"),
                triggered_at=event.get("triggered_at", datetime.utcnow().isoformat()),
                cooldown_until=event.get("cooldown_until"),
                success=event.get("success", False),
                metadata=event.get("metadata", {}),
            )
            for event in history
        ]

        return FailoverHistoryResponse(
            events=events,
            total_count=len(events),
            provider_id=provider_id,
            reason=reason,
        )

    except Exception as e:
        logger.error(f"Failed to get failover history: {e}", exc_info=True)
        return FailoverHistoryResponse(
            events=[],
            total_count=0,
            provider_id=provider_id,
            reason=reason,
        )


# =============================================================================
# Reset Endpoints
# =============================================================================


@router.post(
    "/reset/{provider_id}",
    response_model=ProviderResetResponse,
    summary="Reset provider to primary key",
    description="""
Reset a provider to its primary key.

This operation:
- Clears all key cooldowns for the provider
- Resets the failover state to use the primary key
- Clears consecutive failure counters
- Clears rate limit flags in the API key service

**Use Cases**:
- Manual recovery after rate limit issues are resolved
- Testing failover configuration
- Resetting after false positive rate limit detection
""",
    responses={
        200: {"description": "Provider reset successfully"},
        401: {"description": "Authentication required"},
        404: {"description": "Provider not configured for failover"},
    },
)
async def reset_provider_failover(
    provider_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> ProviderResetResponse:
    """Reset a provider to its primary key."""
    logger.info(f"Resetting failover for provider {provider_id} by user {user.sub}")

    try:
        failover_service = get_failover_service()
        success = await failover_service.reset_provider(provider_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider not configured for failover: {provider_id}",
            )

        return ProviderResetResponse(
            success=True,
            provider_id=provider_id,
            message=f"Successfully reset failover state for provider {provider_id}",
            reset_at=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset provider failover: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset provider failover",
        )


@router.post(
    "/clear-cooldown/{key_id}",
    response_model=CooldownClearResponse,
    summary="Clear cooldown for a specific key",
    description="""
Clear the cooldown state for a specific API key.

This operation:
- Removes the cooldown timer for the key
- Resets the consecutive failure counter
- Makes the key available for selection again

**Use Cases**:
- Manual recovery of a specific key
- Clearing false positive cooldowns
- Testing key availability
""",
    responses={
        200: {"description": "Cooldown cleared successfully"},
        401: {"description": "Authentication required"},
        404: {"description": "Key not found in cooldown tracking"},
    },
)
async def clear_key_cooldown(
    key_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> CooldownClearResponse:
    """Clear cooldown for a specific key."""
    logger.info(f"Clearing cooldown for key {key_id} by user {user.sub}")

    try:
        failover_service = get_failover_service()
        success = await failover_service.clear_cooldown(key_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Key not found in cooldown tracking: {key_id}",
            )

        return CooldownClearResponse(
            success=True,
            key_id=key_id,
            message=f"Successfully cleared cooldown for key {key_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear key cooldown: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear key cooldown",
        )


# =============================================================================
# Configuration Endpoints
# =============================================================================


@router.put(
    "/config/{provider_id}",
    response_model=FailoverConfigResponse,
    summary="Update failover configuration",
    description="""
Update the failover configuration for a provider.

Configurable options include:
- **enabled**: Enable/disable failover for this provider
- **strategy**: Key selection strategy (priority, round_robin, least_used, random)
- **rate_limit_cooldown_seconds**: Cooldown duration after rate limit errors
- **error_cooldown_seconds**: Cooldown duration after other errors
- **max_cooldown_seconds**: Maximum cooldown duration with exponential backoff
- **use_exponential_backoff**: Enable exponential backoff for cooldowns
- **max_consecutive_failures**: Max failures before permanent key block
- **auto_recover**: Automatically recover to primary key when cooldown expires

**Use Cases**:
- Tuning failover behavior per provider
- Adjusting cooldown times for specific providers
- Changing failover strategies
""",
    responses={
        200: {"description": "Configuration updated successfully"},
        401: {"description": "Authentication required"},
        400: {"description": "Invalid configuration"},
    },
)
async def update_failover_config(
    provider_id: str,
    config: FailoverConfigUpdateRequest,
    user: TokenPayload = Depends(get_current_user),
) -> FailoverConfigResponse:
    """Update failover configuration for a provider."""
    logger.info(f"Updating failover config for provider {provider_id} by user {user.sub}")

    try:
        # Validate strategy
        valid_strategies = ["priority", "round_robin", "least_used", "random"]
        if config.strategy not in valid_strategies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy: {config.strategy}. Must be one of: {valid_strategies}",
            )

        failover_service = get_failover_service()

        # Convert strategy string to enum
        from app.services.api_key_failover_service import FailoverStrategy
        strategy_enum = FailoverStrategy(config.strategy)

        # Configure the provider
        failover_service.configure_provider(
            provider_id=provider_id,
            enabled=config.enabled,
            strategy=strategy_enum,
            rate_limit_cooldown_seconds=config.rate_limit_cooldown_seconds,
            error_cooldown_seconds=config.error_cooldown_seconds,
            max_cooldown_seconds=config.max_cooldown_seconds,
            use_exponential_backoff=config.use_exponential_backoff,
            max_consecutive_failures=config.max_consecutive_failures,
            auto_recover=config.auto_recover,
        )

        # Get the updated configuration
        status_data = await failover_service.get_failover_status(provider_id)

        return FailoverConfigResponse(
            success=True,
            provider_id=provider_id,
            config=status_data.get("config", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update failover config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update failover configuration",
        )


@router.get(
    "/cooldown/{key_id}",
    response_model=dict[str, Any],
    summary="Get cooldown info for a key",
    description="""
Get detailed cooldown information for a specific API key.

Returns:
- Current cooldown state (available, cooling_down, permanently_blocked)
- Cooldown end time
- Consecutive and total failure counts
- Remaining cooldown seconds

**Use Cases**:
- Debugging key availability issues
- Monitoring key cooldown status
- Understanding why a key is not being selected
""",
    responses={
        200: {"description": "Cooldown info retrieved"},
        401: {"description": "Authentication required"},
        404: {"description": "Key not found in cooldown tracking"},
    },
)
async def get_key_cooldown_info(
    key_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Get cooldown info for a specific key."""
    logger.debug(f"Getting cooldown info for key {key_id} by user {user.sub}")

    try:
        failover_service = get_failover_service()
        cooldown_info = await failover_service.get_cooldown_info(key_id)

        if cooldown_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Key not found in cooldown tracking: {key_id}",
            )

        return cooldown_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get key cooldown info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve key cooldown info",
        )
