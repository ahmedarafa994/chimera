"""
User Profile Management API Endpoints

Provides user profile endpoints:
- GET /me - Get current user profile
- PUT /me - Update current user profile
- POST /me/change-password - Change user password

All endpoints require authentication and return sanitized user data (no password hash).
"""

import logging
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import AuditAction, audit_log
from app.core.auth import TokenPayload, get_current_user
from app.core.database import get_async_session_factory
from app.core.exceptions import ValidationError
from app.core.rate_limit import RateLimitConfig, get_rate_limiter, get_rate_limit_key
from app.db.models import User, UserRole
from app.services.password_validator import validate_password
from app.services.user_service import UserService, get_user_service

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Database Session Dependency
# =============================================================================


async def get_db_session() -> AsyncSession:
    """
    Get an async database session for the request.

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
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

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
            }
        }


class UserProfileUpdateRequest(BaseModel):
    """Request model for updating user profile."""

    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
        description="New username (3-50 characters)",
        examples=["new_username"],
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: Optional[str]) -> Optional[str]:
        """Validate username format if provided."""
        if v is None:
            return v

        # Allow alphanumeric, underscores, and hyphens
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        # Cannot start with a number
        if v[0].isdigit():
            raise ValueError("Username cannot start with a number")
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
    field: Optional[str] = None


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
            }
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
    password_errors: Optional[list[str]] = None


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
    """
    Check rate limit for change password requests.

    Raises:
        HTTPException: If rate limit is exceeded
    """
    rate_limiter = get_rate_limiter()
    key = get_rate_limit_key(request, prefix="change_password")
    allowed, headers = await rate_limiter.check_rate_limit(
        key, CHANGE_PASSWORD_RATE_LIMIT
    )
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
    """
    Convert a User model to a UserProfileResponse.

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
    """
    Get the authenticated database user from the JWT token.

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
    response_model=UserProfileResponse,
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
    current_user: TokenPayload = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> UserProfileResponse:
    """
    Get the current authenticated user's profile.

    Returns sanitized user information (no password hash).
    """
    user = await _get_authenticated_user(current_user, session)

    logger.info(f"Profile retrieved for user: {user.email} (id={user.id})")

    return _user_to_profile_response(user)


@router.put(
    "/me",
    status_code=status.HTTP_200_OK,
    response_model=UserProfileUpdateResponse,
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
    current_user: TokenPayload = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> UserProfileUpdateResponse:
    """
    Update the current authenticated user's profile.

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
            action=AuditAction.USER_MODIFY,
            user_id=str(user.id),
            resource="/api/v1/users/me",
            details={
                "action": "profile_update",
                "changes": {
                    "username": {
                        "old": old_username,
                        "new": request_data.username,
                    }
                    if request_data.username
                    else None
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
    response_model=ChangePasswordResponse,
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
    current_user: TokenPayload = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ChangePasswordResponse:
    """
    Change the current authenticated user's password.

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

                logger.info(f"Password change failed: incorrect current password for user {user.id}")

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
            action=AuditAction.USER_MODIFY,
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
