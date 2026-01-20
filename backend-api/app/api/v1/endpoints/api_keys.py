# =============================================================================
# Chimera - API Key Management Endpoints
# =============================================================================
# REST API endpoints for secure API key management with proper authentication.
# Supports CRUD operations for all LLM providers with encrypted storage.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# Subtask 1.3: Create API Key Management Endpoints
# =============================================================================

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import TokenPayload, get_current_user
from app.domain.api_key_models import (
    ApiKeyCreate,
    ApiKeyListResponse,
    ApiKeyResponse,
    ApiKeyRole,
    ApiKeyStatus,
    ApiKeyTestResult,
    ApiKeyUpdate,
    ProviderKeySummary,
)
from app.services.api_key_service import (
    ApiKeyDuplicateError,
    ApiKeyNotFoundError,
    ApiKeyServiceError,
    ApiKeyStorageService,
    ApiKeyValidationError,
    get_api_key_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ApiKeyBatchDeleteRequest(BaseModel):
    """Request model for batch deletion of API keys."""

    key_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of key IDs to delete",
    )


class ApiKeyBatchDeleteResponse(BaseModel):
    """Response model for batch deletion of API keys."""

    success: bool = Field(..., description="Whether the batch operation succeeded")
    deleted_count: int = Field(..., ge=0, description="Number of keys deleted")
    failed_ids: list[str] = Field(
        default_factory=list,
        description="List of key IDs that failed to delete",
    )
    errors: dict[str, str] = Field(default_factory=dict, description="Errors for failed deletions")


class ProvidersSummaryResponse(BaseModel):
    """Response model for provider summaries."""

    providers: list[ProviderKeySummary] = Field(..., description="List of provider key summaries")
    total_keys: int = Field(..., ge=0, description="Total number of keys across all providers")
    configured_providers: int = Field(
        ...,
        ge=0,
        description="Number of providers with at least one active key",
    )


class MessageResponse(BaseModel):
    """Simple message response."""

    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Response message")


# ============================================================================
# Dependency Injection
# ============================================================================


def get_service() -> ApiKeyStorageService:
    """Get the API key storage service instance."""
    return get_api_key_service()


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Add a new API key",
    description="""
Add a new API key for a provider with encrypted storage.

The API key will be validated for the specified provider and encrypted using AES-256
before storage. The response returns a masked version of the key for security.

**Supported Providers**: google, openai, anthropic, deepseek, qwen, bigmodel, routeway, cursor

**Key Roles**:
- `primary`: Main key for the provider (one active primary per provider)
- `backup`: Fallback key for failover scenarios
- `fallback`: Tertiary key for additional redundancy
""",
    responses={
        201: {"description": "API key added successfully"},
        400: {"description": "Invalid request or validation error"},
        401: {"description": "Authentication required"},
        409: {"description": "Duplicate primary key exists"},
    },
)
async def create_api_key(
    request: ApiKeyCreate,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyResponse:
    """Add a new API key for a provider."""
    try:
        logger.info(f"Creating API key for provider {request.provider_id} by user {user.sub}")
        key_response = await service.create_key(request)
        logger.info(f"Created API key {key_response.id} for provider {request.provider_id}")
        return key_response

    except ApiKeyDuplicateError as e:
        logger.warning(f"Duplicate key error: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except ApiKeyValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except ApiKeyServiceError as e:
        logger.exception(f"Service error creating key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        ) from e


@router.get(
    "",
    summary="List all API keys",
    description="""
List all API keys with optional filtering.

Returns keys with masked values (never exposes raw keys). Filters can be applied
for provider, role, and status. By default, inactive/revoked keys are excluded.
""",
    responses={
        200: {"description": "List of API keys"},
        401: {"description": "Authentication required"},
    },
)
async def list_api_keys(
    provider_id: Annotated[str | None, Query(description="Filter by provider ID")] = None,
    role: Annotated[ApiKeyRole | None, Query(description="Filter by key role")] = None,
    status_filter: Annotated[
        ApiKeyStatus | None, Query(alias="status", description="Filter by key status")
    ] = None,
    include_inactive: Annotated[bool, Query(description="Include inactive/revoked keys")] = False,
    user: TokenPayload = Depends(get_current_user),
    service: ApiKeyStorageService = Depends(get_service),
) -> ApiKeyListResponse:
    """List all API keys with optional filtering."""
    logger.debug(f"Listing API keys for user {user.sub}")
    return await service.list_keys(
        provider_id=provider_id,
        role=role,
        status=status_filter,
        include_inactive=include_inactive,
    )


@router.get(
    "/providers",
    summary="Get provider key summaries",
    description="""
Get a summary of API key configuration status for all supported providers.

Returns the number of keys, active keys, primary key ID, and backup key IDs
for each provider. Useful for dashboard displays.
""",
    responses={
        200: {"description": "Provider key summaries"},
        401: {"description": "Authentication required"},
    },
)
async def get_provider_summaries(
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ProvidersSummaryResponse:
    """Get summaries of API keys for all providers."""
    logger.debug(f"Getting provider summaries for user {user.sub}")

    summaries = await service.get_all_provider_summaries()

    total_keys = sum(s.total_keys for s in summaries)
    configured_providers = sum(1 for s in summaries if s.has_valid_key)

    return ProvidersSummaryResponse(
        providers=summaries,
        total_keys=total_keys,
        configured_providers=configured_providers,
    )


@router.get(
    "/{key_id}",
    summary="Get API key by ID",
    description="""
Get details of a specific API key by its ID.

Returns the key with masked value. Never exposes the raw API key.
""",
    responses={
        200: {"description": "API key details"},
        401: {"description": "Authentication required"},
        404: {"description": "API key not found"},
    },
)
async def get_api_key(
    key_id: str,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyResponse:
    """Get a specific API key by ID."""
    logger.debug(f"Getting API key {key_id} for user {user.sub}")

    key_response = await service.get_key(key_id)
    if not key_response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        )

    return key_response


@router.put(
    "/{key_id}",
    summary="Update API key",
    description="""
Update an existing API key.

Only provided fields will be updated. If a new api_key value is provided,
it will be encrypted before storage.
""",
    responses={
        200: {"description": "API key updated successfully"},
        400: {"description": "Validation error"},
        401: {"description": "Authentication required"},
        404: {"description": "API key not found"},
    },
)
async def update_api_key(
    key_id: str,
    update: ApiKeyUpdate,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyResponse:
    """Update an existing API key."""
    try:
        logger.info(f"Updating API key {key_id} by user {user.sub}")
        key_response = await service.update_key(key_id, update)
        logger.info(f"Updated API key {key_id}")
        return key_response

    except ApiKeyNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        ) from e
    except ApiKeyValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except ApiKeyServiceError as e:
        logger.exception(f"Service error updating key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update API key",
        ) from e


@router.delete(
    "/{key_id}",
    summary="Delete API key",
    description="""
Permanently delete an API key.

This action cannot be undone. Consider using revoke instead for audit purposes.
""",
    responses={
        200: {"description": "API key deleted successfully"},
        401: {"description": "Authentication required"},
        404: {"description": "API key not found"},
    },
)
async def delete_api_key(
    key_id: str,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> MessageResponse:
    """Delete an API key."""
    try:
        logger.info(f"Deleting API key {key_id} by user {user.sub}")
        await service.delete_key(key_id)
        logger.info(f"Deleted API key {key_id}")
        return MessageResponse(
            success=True,
            message=f"API key {key_id} deleted successfully",
        )

    except ApiKeyNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        ) from e


@router.post(
    "/{key_id}/revoke",
    summary="Revoke API key",
    description="""
Revoke an API key (soft delete).

The key will be marked as revoked and will no longer be used for requests.
Unlike delete, the key record is preserved for audit purposes.
""",
    responses={
        200: {"description": "API key revoked successfully"},
        401: {"description": "Authentication required"},
        404: {"description": "API key not found"},
    },
)
async def revoke_api_key(
    key_id: str,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyResponse:
    """Revoke an API key (soft delete)."""
    try:
        logger.info(f"Revoking API key {key_id} by user {user.sub}")
        key_response = await service.revoke_key(key_id)
        logger.info(f"Revoked API key {key_id}")
        return key_response

    except ApiKeyNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        ) from e


@router.post(
    "/{key_id}/test",
    summary="Test API key connectivity",
    description="""
Test an API key's connectivity by making a lightweight API call.

Returns latency, available models, and any rate limit information.
""",
    responses={
        200: {"description": "Test results"},
        401: {"description": "Authentication required"},
        404: {"description": "API key not found"},
    },
)
async def test_api_key(
    key_id: str,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyTestResult:
    """Test an API key's connectivity."""
    logger.info(f"Testing API key {key_id} by user {user.sub}")

    # Get the key record first to get provider_id
    record = await service.get_key_by_id(key_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        )

    # Create test request
    from app.domain.api_key_models import ApiKeyTestRequest

    test_request = ApiKeyTestRequest(
        provider_id=record.provider_id,
        key_id=key_id,
    )

    result = await service.test_key(test_request)
    logger.info(f"Test result for key {key_id}: success={result.success}")
    return result


@router.post(
    "/test",
    summary="Test API key (inline)",
    description="""
Test an API key's connectivity without storing it.

Useful for validating a key before adding it to the system.
""",
    responses={
        200: {"description": "Test results"},
        400: {"description": "Invalid request"},
        401: {"description": "Authentication required"},
    },
)
async def test_api_key_inline(
    provider_id: Annotated[str, Query(description="Provider ID (e.g., 'openai', 'google')")] = ...,
    api_key: Annotated[str, Query(min_length=10, description="API key to test")] = ...,
    user: TokenPayload = Depends(get_current_user),
    service: ApiKeyStorageService = Depends(get_service),
) -> ApiKeyTestResult:
    """Test an API key without storing it."""
    logger.info(f"Testing inline API key for provider {provider_id} by user {user.sub}")

    from app.domain.api_key_models import ApiKeyTestRequest

    test_request = ApiKeyTestRequest(
        provider_id=provider_id,
        api_key=api_key,
    )

    result = await service.test_key(test_request)
    logger.info(f"Inline test result for provider {provider_id}: success={result.success}")
    return result


@router.delete(
    "",
    summary="Batch delete API keys",
    description="""
Delete multiple API keys in a single operation.

Returns a summary of successful and failed deletions.
""",
    responses={
        200: {"description": "Batch deletion results"},
        401: {"description": "Authentication required"},
    },
)
async def batch_delete_api_keys(
    request: ApiKeyBatchDeleteRequest,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyBatchDeleteResponse:
    """Delete multiple API keys."""
    logger.info(f"Batch deleting {len(request.key_ids)} API keys by user {user.sub}")

    deleted_count = 0
    failed_ids = []
    errors = {}

    for key_id in request.key_ids:
        try:
            await service.delete_key(key_id)
            deleted_count += 1
        except ApiKeyNotFoundError:
            failed_ids.append(key_id)
            errors[key_id] = "Not found"
        except Exception as e:
            failed_ids.append(key_id)
            errors[key_id] = str(e)

    success = len(failed_ids) == 0
    logger.info(f"Batch delete: {deleted_count} deleted, {len(failed_ids)} failed")

    return ApiKeyBatchDeleteResponse(
        success=success,
        deleted_count=deleted_count,
        failed_ids=failed_ids,
        errors=errors,
    )


@router.get(
    "/provider/{provider_id}",
    summary="List keys for a provider",
    description="""
List all API keys for a specific provider.

Shortcut for filtering by provider_id.
""",
    responses={
        200: {"description": "List of API keys for the provider"},
        401: {"description": "Authentication required"},
    },
)
async def list_keys_for_provider(
    provider_id: str,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyListResponse:
    """List all API keys for a specific provider."""
    logger.debug(f"Listing API keys for provider {provider_id} by user {user.sub}")
    return await service.list_keys(provider_id=provider_id)


@router.get(
    "/provider/{provider_id}/summary",
    summary="Get provider key summary",
    description="""
Get a summary of API key configuration for a specific provider.

Returns the number of keys, active keys, and primary/backup key IDs.
""",
    responses={
        200: {"description": "Provider key summary"},
        401: {"description": "Authentication required"},
        404: {"description": "Unknown provider"},
    },
)
async def get_provider_summary(
    provider_id: str,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ProviderKeySummary:
    """Get a summary of API keys for a specific provider."""
    logger.debug(f"Getting summary for provider {provider_id} by user {user.sub}")

    summary = await service.get_provider_summary(provider_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider_id}",
        )

    return summary


@router.post(
    "/{key_id}/activate",
    summary="Activate API key",
    description="""
Activate a previously inactive or rate-limited API key.

Sets the key status to ACTIVE.
""",
    responses={
        200: {"description": "API key activated successfully"},
        401: {"description": "Authentication required"},
        404: {"description": "API key not found"},
    },
)
async def activate_api_key(
    key_id: str,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyResponse:
    """Activate an API key."""
    try:
        logger.info(f"Activating API key {key_id} by user {user.sub}")
        key_response = await service.update_key(
            key_id,
            ApiKeyUpdate(status=ApiKeyStatus.ACTIVE),
        )
        logger.info(f"Activated API key {key_id}")
        return key_response

    except ApiKeyNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        ) from e


@router.post(
    "/{key_id}/deactivate",
    summary="Deactivate API key",
    description="""
Deactivate an API key.

Sets the key status to INACTIVE. The key can be reactivated later.
""",
    responses={
        200: {"description": "API key deactivated successfully"},
        401: {"description": "Authentication required"},
        404: {"description": "API key not found"},
    },
)
async def deactivate_api_key(
    key_id: str,
    user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ApiKeyStorageService, Depends(get_service)],
) -> ApiKeyResponse:
    """Deactivate an API key."""
    try:
        logger.info(f"Deactivating API key {key_id} by user {user.sub}")
        key_response = await service.update_key(
            key_id,
            ApiKeyUpdate(status=ApiKeyStatus.INACTIVE),
        )
        logger.info(f"Deactivated API key {key_id}")
        return key_response

    except ApiKeyNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        ) from e
