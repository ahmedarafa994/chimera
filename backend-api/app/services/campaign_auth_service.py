"""
Campaign Authorization Service - Business logic for campaign access control.

This module provides authorization checks for campaign access based on:
- Ownership (owner has full access)
- Visibility levels (private/team/public)
- Explicit sharing with permission levels (VIEW/EDIT)
- User role (admin can access all campaigns)

Uses the Campaign and CampaignShare models from the database layer.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Campaign,
    CampaignShare,
    CampaignSharePermission,
    CampaignVisibility,
    UserRole,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Authorization Results
# =============================================================================


class CampaignPermission(str, Enum):
    """Permission levels for campaign access."""

    NONE = "none"  # No access
    VIEW = "view"  # Read-only access
    EDIT = "edit"  # Can modify campaign content
    OWNER = "owner"  # Full access (owner)
    ADMIN = "admin"  # Admin override access


@dataclass
class CampaignAccessResult:
    """Result of campaign access check."""

    has_access: bool
    permission: CampaignPermission
    campaign: Optional[Campaign] = None
    reason: Optional[str] = None

    @property
    def can_view(self) -> bool:
        """Check if user can view the campaign."""
        return self.permission in (
            CampaignPermission.VIEW,
            CampaignPermission.EDIT,
            CampaignPermission.OWNER,
            CampaignPermission.ADMIN,
        )

    @property
    def can_edit(self) -> bool:
        """Check if user can edit the campaign."""
        return self.permission in (
            CampaignPermission.EDIT,
            CampaignPermission.OWNER,
            CampaignPermission.ADMIN,
        )

    @property
    def is_owner(self) -> bool:
        """Check if user is the owner of the campaign."""
        return self.permission == CampaignPermission.OWNER

    @property
    def can_manage_shares(self) -> bool:
        """Check if user can manage sharing for the campaign (owner only)."""
        return self.permission in (
            CampaignPermission.OWNER,
            CampaignPermission.ADMIN,
        )

    @property
    def can_delete(self) -> bool:
        """Check if user can delete the campaign (owner and admin only)."""
        return self.permission in (
            CampaignPermission.OWNER,
            CampaignPermission.ADMIN,
        )


# =============================================================================
# Campaign Authorization Service
# =============================================================================


class CampaignAuthService:
    """
    Service layer for campaign authorization.

    Provides methods to check user access to campaigns based on:
    - Ownership: Campaign owner has full access
    - Visibility: Controls default access for non-owners
      - PRIVATE: Only owner and explicitly shared users can access
      - TEAM: All authenticated users can view (future: team-based)
      - PUBLIC: All authenticated users can view
    - Explicit sharing: CampaignShare grants specific permissions to users
    - User role: Admin users can access all campaigns

    Usage:
        auth_service = CampaignAuthService(session)
        result = await auth_service.check_access(user_id, campaign_id, user_role)
        if result.can_edit:
            # Allow edit operation
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize campaign authorization service.

        Args:
            session: SQLAlchemy async session for database operations
        """
        self._session = session

    # =========================================================================
    # Core Authorization Methods
    # =========================================================================

    async def check_access(
        self,
        user_id: int,
        campaign_id: int,
        user_role: UserRole = UserRole.VIEWER,
    ) -> CampaignAccessResult:
        """
        Check if a user can access a campaign and determine permission level.

        This is the main authorization method that considers all access paths:
        1. Admin role grants full access to all campaigns
        2. Owner has full access to their campaigns
        3. Explicit sharing grants VIEW or EDIT access
        4. Visibility (TEAM/PUBLIC) grants VIEW access to all users

        Args:
            user_id: The user's ID
            campaign_id: The campaign's ID
            user_role: The user's role (for admin check)

        Returns:
            CampaignAccessResult with permission level and campaign details
        """
        # First, get the campaign
        campaign = await self._get_campaign(campaign_id)
        if not campaign:
            logger.debug(f"Campaign not found: id={campaign_id}")
            return CampaignAccessResult(
                has_access=False,
                permission=CampaignPermission.NONE,
                reason="Campaign not found",
            )

        # Check if campaign is active
        if not campaign.is_active:
            logger.debug(f"Campaign is inactive: id={campaign_id}")
            # Admins and owners can still access inactive campaigns
            if user_role != UserRole.ADMIN and campaign.owner_id != user_id:
                return CampaignAccessResult(
                    has_access=False,
                    permission=CampaignPermission.NONE,
                    campaign=campaign,
                    reason="Campaign is inactive",
                )

        # 1. Admin override - admins can access all campaigns
        if user_role == UserRole.ADMIN:
            logger.debug(f"Admin access granted: user={user_id}, campaign={campaign_id}")
            return CampaignAccessResult(
                has_access=True,
                permission=CampaignPermission.ADMIN,
                campaign=campaign,
                reason="Admin access",
            )

        # 2. Owner check - owner has full access
        if campaign.owner_id == user_id:
            logger.debug(f"Owner access granted: user={user_id}, campaign={campaign_id}")
            return CampaignAccessResult(
                has_access=True,
                permission=CampaignPermission.OWNER,
                campaign=campaign,
                reason="Owner access",
            )

        # 3. Check explicit sharing
        share = await self._get_share(campaign_id, user_id)
        if share:
            permission = (
                CampaignPermission.EDIT
                if share.permission == CampaignSharePermission.EDIT
                else CampaignPermission.VIEW
            )
            logger.debug(
                f"Shared access granted: user={user_id}, campaign={campaign_id}, "
                f"permission={permission}"
            )
            return CampaignAccessResult(
                has_access=True,
                permission=permission,
                campaign=campaign,
                reason=f"Shared with {permission.value} access",
            )

        # 4. Check visibility-based access
        if campaign.visibility == CampaignVisibility.PUBLIC:
            logger.debug(f"Public access granted: user={user_id}, campaign={campaign_id}")
            return CampaignAccessResult(
                has_access=True,
                permission=CampaignPermission.VIEW,
                campaign=campaign,
                reason="Public campaign",
            )

        if campaign.visibility == CampaignVisibility.TEAM:
            # TEAM visibility: all authenticated users can view
            # In future, this could be restricted to same team/organization
            logger.debug(f"Team access granted: user={user_id}, campaign={campaign_id}")
            return CampaignAccessResult(
                has_access=True,
                permission=CampaignPermission.VIEW,
                campaign=campaign,
                reason="Team campaign",
            )

        # 5. PRIVATE campaign with no share - deny access
        logger.debug(
            f"Access denied (private, no share): user={user_id}, campaign={campaign_id}"
        )
        return CampaignAccessResult(
            has_access=False,
            permission=CampaignPermission.NONE,
            campaign=campaign,
            reason="Private campaign - no access granted",
        )

    async def can_view(
        self,
        user_id: int,
        campaign_id: int,
        user_role: UserRole = UserRole.VIEWER,
    ) -> bool:
        """
        Check if a user can view a campaign.

        Convenience method that returns a simple boolean.

        Args:
            user_id: The user's ID
            campaign_id: The campaign's ID
            user_role: The user's role

        Returns:
            True if user can view the campaign
        """
        result = await self.check_access(user_id, campaign_id, user_role)
        return result.can_view

    async def can_edit(
        self,
        user_id: int,
        campaign_id: int,
        user_role: UserRole = UserRole.VIEWER,
    ) -> bool:
        """
        Check if a user can edit a campaign.

        Convenience method that returns a simple boolean.

        Args:
            user_id: The user's ID
            campaign_id: The campaign's ID
            user_role: The user's role

        Returns:
            True if user can edit the campaign
        """
        result = await self.check_access(user_id, campaign_id, user_role)
        return result.can_edit

    async def can_delete(
        self,
        user_id: int,
        campaign_id: int,
        user_role: UserRole = UserRole.VIEWER,
    ) -> bool:
        """
        Check if a user can delete a campaign.

        Only campaign owner or admin can delete a campaign.

        Args:
            user_id: The user's ID
            campaign_id: The campaign's ID
            user_role: The user's role

        Returns:
            True if user can delete the campaign
        """
        result = await self.check_access(user_id, campaign_id, user_role)
        return result.can_delete

    async def can_manage_shares(
        self,
        user_id: int,
        campaign_id: int,
        user_role: UserRole = UserRole.VIEWER,
    ) -> bool:
        """
        Check if a user can manage sharing for a campaign.

        Only campaign owner or admin can manage shares.

        Args:
            user_id: The user's ID
            campaign_id: The campaign's ID
            user_role: The user's role

        Returns:
            True if user can manage campaign shares
        """
        result = await self.check_access(user_id, campaign_id, user_role)
        return result.can_manage_shares

    # =========================================================================
    # Bulk Access Checks
    # =========================================================================

    async def filter_accessible_campaigns(
        self,
        user_id: int,
        campaign_ids: list[int],
        user_role: UserRole = UserRole.VIEWER,
        require_edit: bool = False,
    ) -> list[int]:
        """
        Filter a list of campaign IDs to only those the user can access.

        Useful for batch operations or listing campaigns.

        Args:
            user_id: The user's ID
            campaign_ids: List of campaign IDs to check
            user_role: The user's role
            require_edit: If True, only return campaigns user can edit

        Returns:
            List of campaign IDs the user can access
        """
        accessible = []
        for campaign_id in campaign_ids:
            result = await self.check_access(user_id, campaign_id, user_role)
            if require_edit and result.can_edit:
                accessible.append(campaign_id)
            elif not require_edit and result.can_view:
                accessible.append(campaign_id)
        return accessible

    async def get_accessible_campaigns_query(
        self,
        user_id: int,
        user_role: UserRole = UserRole.VIEWER,
        include_archived: bool = False,
    ):
        """
        Build a SQLAlchemy query for campaigns accessible to a user.

        This method builds an optimized query that can be used for
        paginated listing of accessible campaigns.

        Args:
            user_id: The user's ID
            user_role: The user's role
            include_archived: Whether to include archived campaigns

        Returns:
            SQLAlchemy select statement for accessible campaigns
        """
        # Base conditions
        conditions = [Campaign.is_active == True]  # noqa: E712
        if not include_archived:
            conditions.append(Campaign.is_archived == False)  # noqa: E712

        # Admin can see all campaigns
        if user_role == UserRole.ADMIN:
            stmt = select(Campaign).where(and_(*conditions))
        else:
            # Non-admin: owner OR shared OR public/team visibility
            # Subquery for shared campaigns
            shared_campaign_ids = (
                select(CampaignShare.campaign_id)
                .where(CampaignShare.user_id == user_id)
                .scalar_subquery()
            )

            access_conditions = or_(
                Campaign.owner_id == user_id,  # Owner
                Campaign.id.in_(shared_campaign_ids),  # Explicitly shared
                Campaign.visibility == CampaignVisibility.PUBLIC,  # Public
                Campaign.visibility == CampaignVisibility.TEAM,  # Team
            )

            conditions.append(access_conditions)
            stmt = select(Campaign).where(and_(*conditions))

        return stmt.order_by(Campaign.updated_at.desc(), Campaign.created_at.desc())

    # =========================================================================
    # Permission Details
    # =========================================================================

    async def get_user_permission_for_campaign(
        self,
        user_id: int,
        campaign_id: int,
        user_role: UserRole = UserRole.VIEWER,
    ) -> CampaignPermission:
        """
        Get the specific permission level a user has for a campaign.

        Args:
            user_id: The user's ID
            campaign_id: The campaign's ID
            user_role: The user's role

        Returns:
            CampaignPermission enum value
        """
        result = await self.check_access(user_id, campaign_id, user_role)
        return result.permission

    async def get_share_for_user(
        self,
        campaign_id: int,
        user_id: int,
    ) -> Optional[CampaignShare]:
        """
        Get the CampaignShare record for a specific user and campaign.

        Args:
            campaign_id: The campaign's ID
            user_id: The user's ID

        Returns:
            CampaignShare if exists, None otherwise
        """
        return await self._get_share(campaign_id, user_id)

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    async def _get_campaign(self, campaign_id: int) -> Optional[Campaign]:
        """
        Get a campaign by ID.

        Args:
            campaign_id: The campaign's ID

        Returns:
            Campaign if found, None otherwise
        """
        try:
            stmt = select(Campaign).where(Campaign.id == campaign_id)
            result = await self._session.execute(stmt)
            campaign = result.scalar_one_or_none()

            if campaign:
                logger.debug(f"Found campaign: id={campaign_id}, owner={campaign.owner_id}")
            else:
                logger.debug(f"Campaign not found: id={campaign_id}")

            return campaign

        except Exception as e:
            logger.error(f"Failed to get campaign id={campaign_id}: {e}", exc_info=True)
            raise

    async def _get_share(
        self,
        campaign_id: int,
        user_id: int,
    ) -> Optional[CampaignShare]:
        """
        Get a campaign share record.

        Args:
            campaign_id: The campaign's ID
            user_id: The user's ID

        Returns:
            CampaignShare if exists, None otherwise
        """
        try:
            stmt = select(CampaignShare).where(
                and_(
                    CampaignShare.campaign_id == campaign_id,
                    CampaignShare.user_id == user_id,
                )
            )
            result = await self._session.execute(stmt)
            share = result.scalar_one_or_none()

            if share:
                logger.debug(
                    f"Found share: campaign={campaign_id}, user={user_id}, "
                    f"permission={share.permission}"
                )
            else:
                logger.debug(f"No share found: campaign={campaign_id}, user={user_id}")

            return share

        except Exception as e:
            logger.error(
                f"Failed to get share for campaign={campaign_id}, user={user_id}: {e}",
                exc_info=True,
            )
            raise


# =============================================================================
# Factory Functions
# =============================================================================


def get_campaign_auth_service(session: AsyncSession) -> CampaignAuthService:
    """
    Get campaign authorization service instance.

    Args:
        session: SQLAlchemy async session

    Returns:
        CampaignAuthService instance
    """
    return CampaignAuthService(session)


# =============================================================================
# Dependency Injection for FastAPI
# =============================================================================


async def get_campaign_auth_service_dependency(
    session: AsyncSession,
) -> CampaignAuthService:
    """
    FastAPI dependency for getting campaign authorization service.

    Usage:
        @router.get("/campaigns/{campaign_id}")
        async def get_campaign(
            campaign_id: int,
            auth_service: CampaignAuthService = Depends(get_campaign_auth_service_dependency)
        ):
            ...

    Note: This requires a session dependency to be injected.
    """
    return get_campaign_auth_service(session)
