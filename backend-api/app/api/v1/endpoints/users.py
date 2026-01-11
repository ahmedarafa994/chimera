"""
User Profile Management API Endpoints

Provides user profile endpoints:
- GET /me - Get current user profile
- PUT /me - Update current user profile

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
from app.db.models import User, UserRole
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
