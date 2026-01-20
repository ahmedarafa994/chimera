"""User Profile Management API Endpoints.

Provides user profile endpoints:
- GET /me - Get current user profile
- PUT /me - Update current user profile
- POST /me/change-password - Change user password
- GET /me/api-keys - List user's API keys
- POST /me/api-keys - Create a new API key
- DELETE /me/api-keys/{id} - Revoke/delete an API key

All endpoints require authentication and return sanitized user data (no password hash).
"""

import hashlib
import logging
import re
import secrets
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import AuditAction, audit_log
from app.core.auth import TokenPayload, get_current_user
from app.core.database import get_async_session_factory
from app.core.exceptions import ValidationError
from app.core.rate_limit import RateLimitConfig, get_rate_limit_key, get_rate_limiter
from app.db.models import User, UserAPIKey, UserRole
from app.repositories.user_repository import get_user_repository
from app.services.password_validator import validate_password
from app.services.user_service import get_user_service

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Database Session Dependency
# =============================================================================


async def get_db_session() -> AsyncSession:
    """Get an async database session for the request.

    Yields a session and ensures proper cleanup.
    """
    session = get_async_session_factory()()
    try:
        yield session
    finally:
        await session.close()


# =============================================================================
# Response Models
# =============================================================================


class UserProfileResponse(BaseModel):
    """Response model for user profile - sanitized user data."""

    id: int
    email: str
    username: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime | None = None
    last_login: datetime | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "email": "user@example.com",
                "username": "john_doe",
                "role": "researcher",
                "is_active": True,
                "is_verified": True,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-20T14:45:00Z",
                "last_login": "2024-01-25T09:00:00Z",
            },
        }


class UserProfileUpdateRequest(BaseModel):
    """Request model for updating user profile."""

    username: str | None = Field(
        None,
        min_length=3,
        max_length=50,
        description="New username (3-50 characters)",
        examples=["new_username"],
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str | None) -> str | None:
        """Validate username format if provided."""
        if v is None:
            return v

        # Allow alphanumeric, underscores, and hyphens
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            msg = "Username can only contain letters, numbers, underscores, and hyphens"
            raise ValueError(msg)
        # Cannot start with a number
        if v[0].isdigit():
            msg = "Username cannot start with a number"
            raise ValueError(msg)
        return v.lower()  # Normalize to lowercase


class UserProfileUpdateResponse(BaseModel):
    """Response model for successful profile update."""

    success: bool = True
    message: str = "Profile updated successfully."
    profile: UserProfileResponse


class ProfileUpdateErrorResponse(BaseModel):
    """Response model for profile update errors."""

    success: bool = False
    error: str
    field: str | None = None


class ChangePasswordRequest(BaseModel):
    """Request model for changing user password."""

    current_password: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Current password for verification",
    )
    new_password: str = Field(
        ...,
        min_length=12,
        max_length=128,
        description="New password (minimum 12 characters with mixed case, numbers, and special characters)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "current_password": "MyCurrentP@ssw0rd!",
                "new_password": "MyNewSecureP@ssw0rd123!",
            },
        }


class ChangePasswordResponse(BaseModel):
    """Response model for successful password change."""

    success: bool = True
    message: str = "Password changed successfully."


class ChangePasswordErrorResponse(BaseModel):
    """Response model for password change errors."""

    success: bool = False
    error: str
    incorrect_password: bool = False
    password_errors: list[str] | None = None


# =============================================================================
# Rate Limiting Configuration
# =============================================================================

# Change password: 5 requests per minute per IP to prevent brute force
CHANGE_PASSWORD_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=5,
    burst_size=2,
    cost_weight=1,
)


async def _check_change_password_rate_limit(request: Request) -> None:
    """Check rate limit for change password requests.

    Raises:
        HTTPException: If rate limit is exceeded

    """
    rate_limiter = get_rate_limiter()
    key = get_rate_limit_key(request, prefix="change_password")
    allowed, headers = await rate_limiter.check_rate_limit(key, CHANGE_PASSWORD_RATE_LIMIT)
    if not allowed:
        logger.warning(f"Change password rate limit exceeded for IP: {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many password change attempts. Please try again later.",
            headers=headers,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _user_to_profile_response(user: User) -> UserProfileResponse:
    """Convert a User model to a UserProfileResponse.

    Sanitizes user data by excluding sensitive fields like password hash.

    Args:
        user: User model from database

    Returns:
        UserProfileResponse with safe user data

    """
    return UserProfileResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        role=user.role.value if isinstance(user.role, UserRole) else user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
    )


async def _get_authenticated_user(
    current_user: TokenPayload,
    session: AsyncSession,
) -> User:
    """Get the authenticated database user from the JWT token.

    Args:
        current_user: Token payload from JWT authentication
        session: Database session

    Returns:
        User model from database

    Raises:
        HTTPException: If user not found or is API client

    """
    # Check if this is an API client (not a database user)
    if current_user.type == "api_key" or current_user.sub == "api_client":
        logger.warning("Profile access attempted with API key authentication")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Profile endpoints require user authentication, not API key",
        )

    # Get user ID from token
    try:
        user_id = int(current_user.sub)
    except (ValueError, TypeError):
        logger.warning(f"Invalid user ID in token: {current_user.sub}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    # Get user from database
    user_service = get_user_service(session)
    user = await user_service.get_user_by_id(user_id)

    if not user:
        logger.warning(f"User not found for profile request: id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not user.is_active:
        logger.warning(f"Inactive user attempted profile access: id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user


# =============================================================================
# API Endpoints
# =============================================================================


@router.get(
    "/me",
    status_code=status.HTTP_200_OK,
    summary="Get current user profile",
    description="""
Get the profile information for the currently authenticated user.

**Returns**:
- User ID
- Email address
- Username
- Role (admin, researcher, viewer)
- Account status (active, verified)
- Timestamps (created, updated, last login)

**Note**: Sensitive information like password hash is never returned.

**Authentication**: Requires JWT token (API key authentication not supported for profile endpoints)

**Example Response**:
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "john_doe",
  "role": "researcher",
  "is_active": true,
  "is_verified": true,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-20T14:45:00Z",
  "last_login": "2024-01-25T09:00:00Z"
}
```
    """,
    responses={
        200: {
            "description": "User profile retrieved successfully",
            "model": UserProfileResponse,
        },
        401: {"description": "Not authenticated"},
        403: {"description": "API key authentication not supported for profile endpoints"},
        404: {"description": "User not found"},
    },
    tags=["users", "profile"],
)
async def get_current_user_profile(
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserProfileResponse:
    """Get the current authenticated user's profile.

    Returns sanitized user information (no password hash).
    """
    user = await _get_authenticated_user(current_user, session)

    logger.info(f"Profile retrieved for user: {user.email} (id={user.id})")

    return _user_to_profile_response(user)


@router.put(
    "/me",
    status_code=status.HTTP_200_OK,
    summary="Update current user profile",
    description="""
Update the profile information for the currently authenticated user.

**Updatable Fields**:
- `username` - New username (3-50 characters, alphanumeric with underscores/hyphens)

**Not Updatable via this endpoint**:
- `email` - Email changes require re-verification (use separate endpoint)
- `password` - Use the change-password endpoint
- `role` - Requires admin privileges

**Username Requirements**:
- 3-50 characters
- Only letters, numbers, underscores, and hyphens
- Cannot start with a number
- Must be unique across all users

**Authentication**: Requires JWT token

**Example Request**:
```json
{
  "username": "new_username"
}
```

**Example Response**:
```json
{
  "success": true,
  "message": "Profile updated successfully.",
  "profile": {
    "id": 1,
    "email": "user@example.com",
    "username": "new_username",
    "role": "researcher",
    "is_active": true,
    "is_verified": true,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-25T10:00:00Z",
    "last_login": "2024-01-25T09:00:00Z"
  }
}
```
    """,
    responses={
        200: {
            "description": "Profile updated successfully",
            "model": UserProfileUpdateResponse,
        },
        400: {
            "description": "Invalid request data",
            "model": ProfileUpdateErrorResponse,
        },
        401: {"description": "Not authenticated"},
        403: {"description": "API key authentication not supported for profile endpoints"},
        404: {"description": "User not found"},
        409: {
            "description": "Username already taken",
            "model": ProfileUpdateErrorResponse,
        },
    },
    tags=["users", "profile"],
)
async def update_current_user_profile(
    request_data: UserProfileUpdateRequest,
    request: Request,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserProfileUpdateResponse:
    """Update the current authenticated user's profile.

    Only allows updating safe fields (username).
    Email changes require re-verification and are handled separately.
    """
    user = await _get_authenticated_user(current_user, session)
    client_ip = request.client.host if request.client else None

    # Check if there's anything to update
    update_fields = {}
    if request_data.username is not None:
        update_fields["username"] = request_data.username

    if not update_fields:
        logger.debug(f"No fields to update for user: {user.email}")
        # Return current profile with success message
        return UserProfileUpdateResponse(
            success=True,
            message="No changes requested.",
            profile=_user_to_profile_response(user),
        )

    # Track old values for audit log
    old_username = user.username

    # Perform update
    user_service = get_user_service(session)

    try:
        updated_user = await user_service.update_profile(user.id, **update_fields)

        if not updated_user:
            logger.error(f"Failed to update profile for user: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update profile",
            )

        # Log successful update to audit trail
        audit_log(
            action=AuditAction.USER_PROFILE_UPDATE,
            user_id=str(user.id),
            resource="/api/v1/users/me",
            details={
                "action": "profile_update",
                "changes": {
                    "username": (
                        {
                            "old": old_username,
                            "new": request_data.username,
                        }
                        if request_data.username
                        else None
                    ),
                },
            },
            ip_address=client_ip,
        )

        logger.info(f"Profile updated for user: {updated_user.email} (id={updated_user.id})")

        return UserProfileUpdateResponse(
            success=True,
            message="Profile updated successfully.",
            profile=_user_to_profile_response(updated_user),
        )

    except ValidationError as e:
        logger.info(f"Profile update validation error for user {user.id}: {e.message}")

        # Determine if it's a username conflict
        if "username" in str(e.message).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "success": False,
                    "error": e.message,
                    "field": "username",
                },
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": e.message,
                "field": e.details.get("field") if e.details else None,
            },
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error updating profile: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating profile",
        )


@router.post(
    "/me/change-password",
    status_code=status.HTTP_200_OK,
    summary="Change user password",
    description="""
Change the password for the currently authenticated user.

**Requires**:
- Current password verification
- New password meeting strength requirements

**Password Requirements**:
- Minimum 12 characters
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one digit (0-9)
- At least one special character (!@#$%^&* etc.)
- Cannot be a common password
- Cannot contain email or username

**Rate Limiting**: 5 requests per minute per IP

**Security Notes**:
- Current password is verified before any changes are made
- Failed attempts are logged for security monitoring
- Password change is logged to audit trail

**Example Request**:
```json
{
  "current_password": "MyCurrentP@ssw0rd!",
  "new_password": "MyNewSecureP@ssw0rd123!"
}
```

**Example Response**:
```json
{
  "success": true,
  "message": "Password changed successfully."
}
```
    """,
    responses={
        200: {
            "description": "Password changed successfully",
            "model": ChangePasswordResponse,
        },
        400: {
            "description": "Invalid request or password validation failed",
            "model": ChangePasswordErrorResponse,
        },
        401: {"description": "Not authenticated or incorrect current password"},
        403: {"description": "API key authentication not supported for password change"},
        429: {"description": "Too many password change attempts"},
    },
    tags=["users", "profile", "security"],
)
async def change_password(
    request_data: ChangePasswordRequest,
    request: Request,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChangePasswordResponse:
    """Change the current authenticated user's password.

    Requires current password verification and validates new password strength.
    """
    # Check rate limit
    await _check_change_password_rate_limit(request)

    # Get authenticated user
    user = await _get_authenticated_user(current_user, session)
    client_ip = request.client.host if request.client else None

    # First, validate new password strength before attempting change
    # This provides early feedback on password requirements
    validation_result = validate_password(
        request_data.new_password,
        email=user.email,
        username=user.username,
    )

    if not validation_result.is_valid:
        password_errors = [e.message for e in validation_result.errors]
        logger.info(f"Password change rejected: weak password for user {user.id}")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "New password does not meet requirements",
                "incorrect_password": False,
                "password_errors": password_errors,
            },
        )

    # Attempt password change via user service
    user_service = get_user_service(session)

    try:
        result = await user_service.change_password(
            user_id=user.id,
            current_password=request_data.current_password,
            new_password=request_data.new_password,
        )

        if not result.success:
            # Determine the error type
            if result.error == "Current password is incorrect":
                # Log failed password verification attempt
                audit_log(
                    action=AuditAction.AUTH_FAILED,
                    user_id=str(user.id),
                    resource="/api/v1/users/me/change-password",
                    details={
                        "action": "password_change_failed",
                        "reason": "incorrect_current_password",
                    },
                    ip_address=client_ip,
                )

                logger.info(
                    f"Password change failed: incorrect current password for user {user.id}",
                )

                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "success": False,
                        "error": "Current password is incorrect",
                        "incorrect_password": True,
                        "password_errors": None,
                    },
                )

            # Password validation failed (shouldn't happen as we validate above, but handle it)
            if result.validation_result and not result.validation_result.is_valid:
                password_errors = [e.message for e in result.validation_result.errors]
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "success": False,
                        "error": result.error or "New password does not meet requirements",
                        "incorrect_password": False,
                        "password_errors": password_errors,
                    },
                )

            # Other error
            logger.error(f"Password change failed for user {user.id}: {result.error}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": result.error or "Password change failed",
                    "incorrect_password": False,
                    "password_errors": None,
                },
            )

        # Password changed successfully - log to audit trail
        audit_log(
            action=AuditAction.USER_PASSWORD_CHANGE,
            user_id=str(user.id),
            resource="/api/v1/users/me/change-password",
            details={
                "action": "password_changed",
                "user_email": user.email,
            },
            ip_address=client_ip,
        )

        logger.info(f"Password changed successfully for user: {user.email} (id={user.id})")

        return ChangePasswordResponse(
            success=True,
            message="Password changed successfully.",
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error changing password: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while changing password",
        )


# =============================================================================
# API Key Management Response Models
# =============================================================================


class APIKeyResponse(BaseModel):
    """Response model for a single API key (masked)."""

    id: int
    name: str | None = None
    key_prefix: str
    is_active: bool
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    usage_count: int = 0
    created_at: datetime
    revoked_at: datetime | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "Production API Key",
                "key_prefix": "chim_abc1",
                "is_active": True,
                "expires_at": "2025-01-15T00:00:00Z",
                "last_used_at": "2024-01-25T10:30:00Z",
                "usage_count": 42,
                "created_at": "2024-01-01T00:00:00Z",
                "revoked_at": None,
            },
        }


class APIKeyListResponse(BaseModel):
    """Response model for listing API keys."""

    success: bool = True
    api_keys: list[APIKeyResponse]
    total_count: int
    active_count: int


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating a new API key."""

    name: str | None = Field(
        None,
        min_length=1,
        max_length=100,
        description="Optional friendly name for the API key",
        examples=["Production Key", "CI/CD Pipeline"],
    )
    expires_in_days: int | None = Field(
        None,
        ge=1,
        le=365,
        description="Optional expiration in days (1-365). If not set, key never expires.",
        examples=[30, 90, 365],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Validate and sanitize key name."""
        if v is None:
            return v
        # Strip whitespace and validate
        v = v.strip()
        if len(v) == 0:
            return None
        return v


class CreateAPIKeyResponse(BaseModel):
    """Response model for creating an API key.

    IMPORTANT: The full API key is only returned ONCE at creation time.
    Store it securely - it cannot be retrieved again.
    """

    success: bool = True
    message: str = "API key created successfully."
    api_key: str = Field(
        ...,
        description="The full API key. Store this securely - it cannot be retrieved again.",
    )
    key_info: APIKeyResponse

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "API key created successfully.",
                "api_key": "chim_abc123def456...",
                "key_info": {
                    "id": 1,
                    "name": "Production API Key",
                    "key_prefix": "chim_abc1",
                    "is_active": True,
                    "expires_at": "2025-01-15T00:00:00Z",
                    "last_used_at": None,
                    "usage_count": 0,
                    "created_at": "2024-01-01T00:00:00Z",
                    "revoked_at": None,
                },
            },
        }


class DeleteAPIKeyResponse(BaseModel):
    """Response model for deleting/revoking an API key."""

    success: bool = True
    message: str = "API key revoked successfully."


class APIKeyErrorResponse(BaseModel):
    """Response model for API key operation errors."""

    success: bool = False
    error: str


# =============================================================================
# API Key Configuration
# =============================================================================

# Maximum API keys per user
MAX_API_KEYS_PER_USER = 10

# API key prefix for identification
API_KEY_PREFIX = "chim_"


# =============================================================================
# API Key Helper Functions
# =============================================================================


def _generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key with prefix.

    Returns:
        Tuple of (full_key, hashed_key, key_prefix)

    """
    # Generate 32 bytes of random data (256 bits of entropy)
    random_part = secrets.token_urlsafe(32)
    full_key = f"{API_KEY_PREFIX}{random_part}"

    # Hash the key for storage using SHA-256
    hashed_key = hashlib.sha256(full_key.encode()).hexdigest()

    # Extract prefix for identification (first 8 chars after prefix)
    key_prefix = full_key[: len(API_KEY_PREFIX) + 4]

    return full_key, hashed_key, key_prefix


def _api_key_to_response(api_key: UserAPIKey) -> APIKeyResponse:
    """Convert a UserAPIKey model to an APIKeyResponse.

    Args:
        api_key: UserAPIKey model from database

    Returns:
        APIKeyResponse with safe key data

    """
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        is_active=api_key.is_active,
        expires_at=api_key.expires_at,
        last_used_at=api_key.last_used_at,
        usage_count=api_key.usage_count,
        created_at=api_key.created_at,
        revoked_at=api_key.revoked_at,
    )


# =============================================================================
# API Key Endpoints
# =============================================================================


@router.get(
    "/me/api-keys",
    status_code=status.HTTP_200_OK,
    summary="List user's API keys",
    description="""
List all API keys for the currently authenticated user.

**Returns**:
- List of API keys with masked key values (only prefix shown)
- Total count of all keys
- Count of active (non-revoked, non-expired) keys

**Note**: Full API key values are never returned after creation.
Only the key prefix is shown for identification.

**Authentication**: Requires JWT token (API key authentication not supported for key management)

**Example Response**:
```json
{
  "success": true,
  "api_keys": [
    {
      "id": 1,
      "name": "Production Key",
      "key_prefix": "chim_abc1",
      "is_active": true,
      "expires_at": "2025-01-15T00:00:00Z",
      "last_used_at": "2024-01-25T10:30:00Z",
      "usage_count": 42,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total_count": 1,
  "active_count": 1
}
```
    """,
    responses={
        200: {
            "description": "API keys retrieved successfully",
            "model": APIKeyListResponse,
        },
        401: {"description": "Not authenticated"},
        403: {"description": "API key authentication not supported for key management"},
    },
    tags=["users", "api-keys"],
)
async def list_api_keys(
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> APIKeyListResponse:
    """List all API keys for the current user.

    Returns masked API key data (only prefix shown).
    """
    user = await _get_authenticated_user(current_user, session)

    # Get user repository for API key operations
    user_repo = get_user_repository(session)

    # List all keys
    api_keys = await user_repo.list_api_keys(user.id)

    # Count active keys
    active_count = await user_repo.count_active_api_keys(user.id)

    # Convert to response models
    key_responses = [_api_key_to_response(key) for key in api_keys]

    logger.info(f"Listed {len(api_keys)} API keys for user: {user.email} (id={user.id})")

    return APIKeyListResponse(
        success=True,
        api_keys=key_responses,
        total_count=len(api_keys),
        active_count=active_count,
    )


@router.post(
    "/me/api-keys",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API key",
    description="""
Create a new API key for the currently authenticated user.

**IMPORTANT**: The full API key is only returned ONCE at creation time.
Copy and store it securely - it cannot be retrieved again.

**Limits**:
- Maximum 10 API keys per user
- Keys can optionally expire after 1-365 days

**Request Parameters**:
- `name` (optional): Friendly name for identification
- `expires_in_days` (optional): Days until expiration (1-365)

**Authentication**: Requires JWT token (API key authentication not supported for key management)

**Example Request**:
```json
{
  "name": "Production Key",
  "expires_in_days": 90
}
```

**Example Response**:
```json
{
  "success": true,
  "message": "API key created successfully.",
  "api_key": "chim_abc123def456ghi789jkl012mno345pqr678stu",
  "key_info": {
    "id": 1,
    "name": "Production Key",
    "key_prefix": "chim_abc1",
    "is_active": true,
    "expires_at": "2024-04-15T00:00:00Z",
    "created_at": "2024-01-15T00:00:00Z"
  }
}
```
    """,
    responses={
        201: {
            "description": "API key created successfully",
            "model": CreateAPIKeyResponse,
        },
        400: {
            "description": "Maximum API keys limit reached",
            "model": APIKeyErrorResponse,
        },
        401: {"description": "Not authenticated"},
        403: {"description": "API key authentication not supported for key management"},
    },
    tags=["users", "api-keys"],
)
async def create_api_key(
    request_data: CreateAPIKeyRequest,
    request: Request,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CreateAPIKeyResponse:
    """Create a new API key for the current user.

    The full API key is only returned once - store it securely.
    """
    user = await _get_authenticated_user(current_user, session)
    client_ip = request.client.host if request.client else None

    # Get user repository
    user_repo = get_user_repository(session)

    # Check limit
    active_key_count = await user_repo.count_active_api_keys(user.id)
    if active_key_count >= MAX_API_KEYS_PER_USER:
        logger.warning(
            f"User {user.id} exceeded API key limit ({active_key_count}/{MAX_API_KEYS_PER_USER})",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": f"Maximum API keys limit reached ({MAX_API_KEYS_PER_USER}). Please revoke unused keys.",
            },
        )

    # Generate API key
    full_key, hashed_key, key_prefix = _generate_api_key()

    # Calculate expiration if specified
    expires_at = None
    if request_data.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=request_data.expires_in_days)

    # Create the API key
    try:
        api_key = await user_repo.create_api_key(
            user_id=user.id,
            hashed_key=hashed_key,
            key_prefix=key_prefix,
            name=request_data.name,
            expires_at=expires_at,
        )

        # Commit the transaction
        await session.commit()

        # Log to audit trail
        audit_log(
            action=AuditAction.API_KEY_CREATED,
            user_id=str(user.id),
            resource="/api/v1/users/me/api-keys",
            details={
                "key_id": api_key.id,
                "key_prefix": key_prefix,
                "key_name": request_data.name,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
            ip_address=client_ip,
        )

        logger.info(f"Created API key for user: {user.email} (id={user.id}, key_id={api_key.id})")

        return CreateAPIKeyResponse(
            success=True,
            message="API key created successfully. Store this key securely - it cannot be retrieved again.",
            api_key=full_key,
            key_info=_api_key_to_response(api_key),
        )

    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to create API key for user {user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )


@router.delete(
    "/me/api-keys/{key_id}",
    status_code=status.HTTP_200_OK,
    summary="Revoke/delete an API key",
    description="""
Revoke (soft-delete) an API key owned by the current user.

The API key will be deactivated immediately and can no longer be used
for authentication. The key record is kept for audit purposes.

**Parameters**:
- `key_id`: The ID of the API key to revoke

**Authentication**: Requires JWT token (API key authentication not supported for key management)

**Security Notes**:
- Revocation is logged to the audit trail
- Revoked keys cannot be reactivated
- Use this when a key is compromised or no longer needed

**Example Response**:
```json
{
  "success": true,
  "message": "API key revoked successfully."
}
```
    """,
    responses={
        200: {
            "description": "API key revoked successfully",
            "model": DeleteAPIKeyResponse,
        },
        401: {"description": "Not authenticated"},
        403: {"description": "API key authentication not supported for key management"},
        404: {
            "description": "API key not found",
            "model": APIKeyErrorResponse,
        },
    },
    tags=["users", "api-keys"],
)
async def delete_api_key(
    key_id: int,
    request: Request,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DeleteAPIKeyResponse:
    """Revoke (soft-delete) an API key.

    The key is deactivated but the record is kept for audit purposes.
    """
    user = await _get_authenticated_user(current_user, session)
    client_ip = request.client.host if request.client else None

    # Get user repository
    user_repo = get_user_repository(session)

    # Get the key first for audit logging
    api_key = await user_repo.get_api_key_by_id(key_id, user.id)
    if not api_key:
        logger.warning(f"API key {key_id} not found for user {user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "error": "API key not found",
            },
        )

    # Check if already revoked
    if not api_key.is_active:
        logger.info(f"API key {key_id} already revoked for user {user.id}")
        return DeleteAPIKeyResponse(
            success=True,
            message="API key was already revoked.",
        )

    # Revoke the key
    try:
        success = await user_repo.revoke_api_key(key_id, user.id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to revoke API key",
            )

        # Commit the transaction
        await session.commit()

        # Log to audit trail
        audit_log(
            action=AuditAction.API_KEY_REVOKED,
            user_id=str(user.id),
            resource=f"/api/v1/users/me/api-keys/{key_id}",
            details={
                "key_id": key_id,
                "key_prefix": api_key.key_prefix,
                "key_name": api_key.name,
            },
            ip_address=client_ip,
        )

        logger.info(f"Revoked API key {key_id} for user: {user.email} (id={user.id})")

        return DeleteAPIKeyResponse(
            success=True,
            message="API key revoked successfully.",
        )

    except HTTPException:
        raise

    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to revoke API key {key_id} for user {user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )
