"""Campaign Sharing API Endpoints.

Provides endpoints for managing campaign sharing:
- POST /campaigns/{id}/share - Share campaign with a user
- GET /campaigns/{id}/shares - List all shares for a campaign
- DELETE /campaigns/{id}/share/{user_id} - Remove a share

All endpoints require authentication and check campaign ownership/admin permissions.
"""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import AuditAction, audit_log
from app.core.auth import TokenPayload, get_current_user
from app.core.database import get_async_session_factory
from app.db.models import Campaign, CampaignShare, CampaignSharePermission, User
from app.services.campaign_auth_service import CampaignAuthService

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
# Request/Response Models
# =============================================================================


class ShareCampaignRequest(BaseModel):
    """Request model for sharing a campaign with a user."""

    email: EmailStr = Field(
        ...,
        description="Email address of the user to share with",
        examples=["user@example.com"],
    )
    permission: str = Field(
        default="view",
        description="Permission level: 'view' or 'edit'",
        examples=["view", "edit"],
    )

    @field_validator("permission")
    @classmethod
    def validate_permission(cls, v: str) -> str:
        """Validate permission is valid."""
        v = v.lower().strip()
        if v not in ("view", "edit"):
            msg = "Permission must be 'view' or 'edit'"
            raise ValueError(msg)
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "email": "colleague@example.com",
                "permission": "edit",
            },
        }


class CampaignShareResponse(BaseModel):
    """Response model for a single campaign share."""

    id: int
    user_id: int
    user_email: str
    username: str
    permission: str
    shared_by_id: int | None = None
    shared_by_email: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "user_id": 42,
                "user_email": "colleague@example.com",
                "username": "john_doe",
                "permission": "edit",
                "shared_by_id": 1,
                "shared_by_email": "owner@example.com",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": None,
            },
        }


class ShareCampaignSuccessResponse(BaseModel):
    """Response model for successful campaign share."""

    success: bool = True
    message: str
    share: CampaignShareResponse


class CampaignShareListResponse(BaseModel):
    """Response model for listing campaign shares."""

    success: bool = True
    campaign_id: int
    campaign_name: str
    shares: list[CampaignShareResponse]
    total_count: int


class ShareErrorResponse(BaseModel):
    """Response model for share operation errors."""

    success: bool = False
    error: str


class RemoveShareResponse(BaseModel):
    """Response model for removing a share."""

    success: bool = True
    message: str


# =============================================================================
# Helper Functions
# =============================================================================


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
        logger.warning("Campaign share access attempted with API key authentication")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Campaign sharing endpoints require user authentication, not API key",
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
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        logger.warning(f"User not found for share request: id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not user.is_active:
        logger.warning(f"Inactive user attempted share access: id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user


async def _get_campaign(session: AsyncSession, campaign_id: int) -> Campaign:
    """Get a campaign by ID.

    Args:
        session: Database session
        campaign_id: Campaign ID

    Returns:
        Campaign model

    Raises:
        HTTPException: If campaign not found

    """
    stmt = select(Campaign).where(Campaign.id == campaign_id)
    result = await session.execute(stmt)
    campaign = result.scalar_one_or_none()

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found",
        )

    return campaign


async def _check_share_permission(
    session: AsyncSession,
    user: User,
    campaign: Campaign,
) -> None:
    """Check if user can manage shares for a campaign.

    Only owner and admins can manage sharing.

    Args:
        session: Database session
        user: Current user
        campaign: Campaign to check

    Raises:
        HTTPException: If user cannot manage shares

    """
    auth_service = CampaignAuthService(session)
    access_result = await auth_service.check_access(
        user_id=user.id,
        campaign_id=campaign.id,
        user_role=user.role,
    )

    if not access_result.can_manage_shares:
        logger.warning(
            f"User {user.id} attempted to manage shares for campaign {campaign.id} "
            f"without permission",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only campaign owner or admin can manage sharing",
        )


def _share_to_response(
    share: CampaignShare,
    user: User,
    shared_by: User | None = None,
) -> CampaignShareResponse:
    """Convert a CampaignShare model to a CampaignShareResponse.

    Args:
        share: CampaignShare model
        user: User who has the share
        shared_by: User who created the share (optional)

    Returns:
        CampaignShareResponse

    """
    return CampaignShareResponse(
        id=share.id,
        user_id=share.user_id,
        user_email=user.email,
        username=user.username,
        permission=share.permission.value,
        shared_by_id=share.shared_by_id,
        shared_by_email=shared_by.email if shared_by else None,
        created_at=share.created_at,
        updated_at=share.updated_at,
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/{campaign_id}/share",
    status_code=status.HTTP_201_CREATED,
    summary="Share campaign with a user",
    description="""
Share a campaign with another user by their email address.

**Requirements**:
- Only the campaign owner or admin can share a campaign
- The target user must exist in the system
- Cannot share a campaign with yourself
- If the user already has access, their permission level will be updated

**Permission Levels**:
- `view` - User can view the campaign but not modify it
- `edit` - User can view and modify the campaign

**Authentication**: Requires JWT token

**Example Request**:
```json
{
  "email": "colleague@example.com",
  "permission": "edit"
}
```

**Example Response**:
```json
{
  "success": true,
  "message": "Campaign shared successfully with colleague@example.com",
  "share": {
    "id": 1,
    "user_id": 42,
    "user_email": "colleague@example.com",
    "username": "john_doe",
    "permission": "edit",
    "shared_by_id": 1,
    "shared_by_email": "owner@example.com",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```
    """,
    responses={
        201: {
            "description": "Campaign shared successfully",
            "model": ShareCampaignSuccessResponse,
        },
        400: {
            "description": "Invalid request or cannot share with self",
            "model": ShareErrorResponse,
        },
        403: {"description": "Not authorized to share this campaign"},
        404: {"description": "Campaign or target user not found"},
    },
    tags=["campaigns", "sharing"],
)
async def share_campaign(
    campaign_id: int,
    request_data: ShareCampaignRequest,
    request: Request,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ShareCampaignSuccessResponse:
    """Share a campaign with another user.

    Creates a new share or updates an existing one.
    """
    user = await _get_authenticated_user(current_user, session)
    client_ip = request.client.host if request.client else None

    # Get the campaign
    campaign = await _get_campaign(session, campaign_id)

    # Check permission to manage shares
    await _check_share_permission(session, user, campaign)

    # Find the target user by email
    stmt = select(User).where(User.email == request_data.email.lower())
    result = await session.execute(stmt)
    target_user = result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email {request_data.email} not found",
        )

    # Cannot share with yourself
    if target_user.id == user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot share a campaign with yourself",
        )

    # Cannot share with the owner
    if target_user.id == campaign.owner_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot share with the campaign owner - they already have full access",
        )

    # Map permission string to enum
    permission = (
        CampaignSharePermission.EDIT
        if request_data.permission == "edit"
        else CampaignSharePermission.VIEW
    )

    # Check if share already exists
    stmt = select(CampaignShare).where(
        and_(
            CampaignShare.campaign_id == campaign_id,
            CampaignShare.user_id == target_user.id,
        ),
    )
    result = await session.execute(stmt)
    existing_share = result.scalar_one_or_none()

    try:
        if existing_share:
            # Update existing share
            old_permission = existing_share.permission
            existing_share.permission = permission
            existing_share.shared_by_id = user.id
            share = existing_share
            is_update = True
        else:
            # Create new share
            share = CampaignShare(
                campaign_id=campaign_id,
                user_id=target_user.id,
                permission=permission,
                shared_by_id=user.id,
            )
            session.add(share)
            is_update = False

        await session.commit()
        await session.refresh(share)

        # Log to audit trail
        audit_log(
            action=AuditAction.CAMPAIGN_SHARE,
            user_id=str(user.id),
            resource=f"/api/v1/campaigns/{campaign_id}/share",
            details={
                "action": "campaign_share_updated" if is_update else "campaign_shared",
                "campaign_id": campaign_id,
                "campaign_name": campaign.name,
                "target_user_id": target_user.id,
                "target_user_email": target_user.email,
                "permission": permission.value,
                "previous_permission": old_permission.value if is_update else None,
            },
            ip_address=client_ip,
        )

        message = (
            f"Campaign sharing updated for {target_user.email}"
            if is_update
            else f"Campaign shared successfully with {target_user.email}"
        )

        logger.info(
            f"Campaign {campaign_id} shared with user {target_user.id} "
            f"by user {user.id} (permission={permission.value})",
        )

        return ShareCampaignSuccessResponse(
            success=True,
            message=message,
            share=_share_to_response(share, target_user, user),
        )

    except Exception as e:
        await session.rollback()
        logger.error(
            f"Failed to share campaign {campaign_id} with user {target_user.id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to share campaign",
        )


@router.get(
    "/{campaign_id}/shares",
    status_code=status.HTTP_200_OK,
    summary="List all shares for a campaign",
    description="""
List all users who have been granted access to a campaign.

**Requirements**:
- Only the campaign owner or admin can view shares

**Returns**:
- List of shares with user details and permission levels
- Total count of shares

**Authentication**: Requires JWT token

**Example Response**:
```json
{
  "success": true,
  "campaign_id": 1,
  "campaign_name": "My Research Campaign",
  "shares": [
    {
      "id": 1,
      "user_id": 42,
      "user_email": "colleague@example.com",
      "username": "john_doe",
      "permission": "edit",
      "shared_by_id": 1,
      "shared_by_email": "owner@example.com",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total_count": 1
}
```
    """,
    responses={
        200: {
            "description": "Shares retrieved successfully",
            "model": CampaignShareListResponse,
        },
        403: {"description": "Not authorized to view shares for this campaign"},
        404: {"description": "Campaign not found"},
    },
    tags=["campaigns", "sharing"],
)
async def list_campaign_shares(
    campaign_id: int,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CampaignShareListResponse:
    """List all shares for a campaign.

    Returns all users who have been granted access to the campaign.
    """
    user = await _get_authenticated_user(current_user, session)

    # Get the campaign
    campaign = await _get_campaign(session, campaign_id)

    # Check permission to view shares
    await _check_share_permission(session, user, campaign)

    # Get all shares for the campaign with user info
    stmt = (
        select(CampaignShare, User)
        .join(User, CampaignShare.user_id == User.id)
        .where(CampaignShare.campaign_id == campaign_id)
        .order_by(CampaignShare.created_at.desc())
    )
    result = await session.execute(stmt)
    shares_with_users = result.all()

    # Build response with shared_by info
    share_responses = []
    for share, share_user in shares_with_users:
        # Get shared_by user if available
        shared_by = None
        if share.shared_by_id:
            stmt = select(User).where(User.id == share.shared_by_id)
            result = await session.execute(stmt)
            shared_by = result.scalar_one_or_none()

        share_responses.append(_share_to_response(share, share_user, shared_by))

    logger.info(
        f"Listed {len(share_responses)} shares for campaign {campaign_id} by user {user.id}",
    )

    return CampaignShareListResponse(
        success=True,
        campaign_id=campaign_id,
        campaign_name=campaign.name,
        shares=share_responses,
        total_count=len(share_responses),
    )


@router.delete(
    "/{campaign_id}/share/{target_user_id}",
    status_code=status.HTTP_200_OK,
    summary="Remove a campaign share",
    description="""
Remove a user's access to a campaign.

**Requirements**:
- Only the campaign owner or admin can remove shares
- The target user must have an existing share

**Authentication**: Requires JWT token

**Example Response**:
```json
{
  "success": true,
  "message": "Share removed successfully"
}
```
    """,
    responses={
        200: {
            "description": "Share removed successfully",
            "model": RemoveShareResponse,
        },
        403: {"description": "Not authorized to remove shares for this campaign"},
        404: {"description": "Campaign or share not found"},
    },
    tags=["campaigns", "sharing"],
)
async def remove_campaign_share(
    campaign_id: int,
    target_user_id: int,
    request: Request,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RemoveShareResponse:
    """Remove a user's access to a campaign.

    Deletes the share record, revoking the user's access.
    """
    user = await _get_authenticated_user(current_user, session)
    client_ip = request.client.host if request.client else None

    # Get the campaign
    campaign = await _get_campaign(session, campaign_id)

    # Check permission to manage shares
    await _check_share_permission(session, user, campaign)

    # Find the share to remove
    stmt = select(CampaignShare).where(
        and_(
            CampaignShare.campaign_id == campaign_id,
            CampaignShare.user_id == target_user_id,
        ),
    )
    result = await session.execute(stmt)
    share = result.scalar_one_or_none()

    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share not found - user may not have access to this campaign",
        )

    # Get target user info for audit log
    stmt = select(User).where(User.id == target_user_id)
    result = await session.execute(stmt)
    target_user = result.scalar_one_or_none()

    try:
        # Store permission for audit log before deletion
        old_permission = share.permission.value

        # Delete the share
        await session.delete(share)
        await session.commit()

        # Log to audit trail
        audit_log(
            action=AuditAction.CAMPAIGN_UNSHARE,
            user_id=str(user.id),
            resource=f"/api/v1/campaigns/{campaign_id}/share/{target_user_id}",
            details={
                "action": "campaign_share_removed",
                "campaign_id": campaign_id,
                "campaign_name": campaign.name,
                "target_user_id": target_user_id,
                "target_user_email": target_user.email if target_user else None,
                "removed_permission": old_permission,
            },
            ip_address=client_ip,
        )

        logger.info(
            f"Removed share for user {target_user_id} from campaign {campaign_id} "
            f"by user {user.id}",
        )

        return RemoveShareResponse(
            success=True,
            message="Share removed successfully",
        )

    except Exception as e:
        await session.rollback()
        logger.error(
            f"Failed to remove share for user {target_user_id} from campaign {campaign_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove share",
        )
