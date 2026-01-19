"""
Team Workspaces & Collaboration Endpoints

Phase 3 enterprise feature for scalability:
- Multi-user team environments
- Role-based access control (Admin, Researcher, Viewer)
- Shared assessment history and API key pools
- Activity logging and audit trails
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.database import get_db
from app.core.email import send_invitation_email
from app.core.observability import get_logger
from app.db.models import User

logger = get_logger("chimera.api.workspaces")
router = APIRouter()


# Team Models
class TeamRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class InvitationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"


class ActivityType(str, Enum):
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ROLE_CHANGED = "role_changed"
    ASSESSMENT_CREATED = "assessment_created"
    ASSESSMENT_SHARED = "assessment_shared"
    REPORT_GENERATED = "report_generated"
    API_KEY_ADDED = "api_key_added"
    API_KEY_REMOVED = "api_key_removed"
    WORKSPACE_CREATED = "workspace_created"
    WORKSPACE_UPDATED = "workspace_updated"


class TeamMember(BaseModel):
    """Team member information"""

    user_id: str
    username: str
    email: str
    role: TeamRole
    joined_at: datetime
    last_active: datetime | None = None
    is_active: bool = True


class TeamWorkspace(BaseModel):
    """Team workspace configuration"""

    workspace_id: str
    name: str
    description: str | None = None

    # Ownership
    owner_id: str
    created_at: datetime
    updated_at: datetime

    # Settings
    settings: dict[str, Any] = Field(default_factory=dict)

    # Statistics
    member_count: int = 0
    assessment_count: int = 0

    # Features
    features_enabled: list[str] = Field(default_factory=list)


class TeamInvitation(BaseModel):
    """Team invitation details"""

    invitation_id: str
    workspace_id: str
    workspace_name: str
    invited_email: str
    invited_by: str
    invited_by_name: str
    role: TeamRole

    status: InvitationStatus = InvitationStatus.PENDING
    created_at: datetime
    expires_at: datetime
    accepted_at: datetime | None = None

    message: str | None = None


class ActivityLogEntry(BaseModel):
    """Activity log entry"""

    activity_id: str
    workspace_id: str
    user_id: str
    username: str
    activity_type: ActivityType

    # Activity details
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Timing
    timestamp: datetime

    # Context
    ip_address: str | None = None
    user_agent: str | None = None


class WorkspaceCreate(BaseModel):
    """Request to create a new workspace"""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    settings: dict[str, Any] = Field(default_factory=dict)


class WorkspaceUpdate(BaseModel):
    """Request to update workspace"""

    name: str | None = None
    description: str | None = None
    settings: dict[str, Any] | None = None


class InviteUserRequest(BaseModel):
    """Request to invite user to workspace"""

    email: EmailStr
    role: TeamRole
    message: str | None = None


class UpdateMemberRole(BaseModel):
    """Request to update member role"""

    role: TeamRole


class WorkspaceListResponse(BaseModel):
    """Response for workspace listing"""

    workspaces: list[TeamWorkspace]
    total: int


class MembersListResponse(BaseModel):
    """Response for members listing"""

    members: list[TeamMember]
    invitations: list[TeamInvitation]
    total_members: int
    total_invitations: int


class ActivityLogResponse(BaseModel):
    """Response for activity log"""

    activities: list[ActivityLogEntry]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# In-memory storage for demo (in production, would use database)
workspaces_storage: dict[str, TeamWorkspace] = {}
members_storage: dict[str, list[TeamMember]] = {}
invitations_storage: dict[str, list[TeamInvitation]] = {}
activity_log_storage: dict[str, list[ActivityLogEntry]] = {}


def check_workspace_permission(workspace_id: str, user_id: str, required_role: TeamRole) -> bool:
    """Check if user has required permission in workspace"""
    members = members_storage.get(workspace_id, [])
    user_member = next((m for m in members if m.user_id == user_id), None)

    if not user_member or not user_member.is_active:
        return False

    # Role hierarchy: owner > admin > researcher > viewer
    role_levels = {TeamRole.VIEWER: 1, TeamRole.RESEARCHER: 2, TeamRole.ADMIN: 3, TeamRole.OWNER: 4}

    return role_levels.get(user_member.role, 0) >= role_levels.get(required_role, 0)


def log_activity(
    workspace_id: str,
    user_id: str,
    username: str,
    activity_type: ActivityType,
    description: str,
    metadata: dict[str, Any] | None = None,
):
    """Log workspace activity"""
    if workspace_id not in activity_log_storage:
        activity_log_storage[workspace_id] = []

    activity = ActivityLogEntry(
        activity_id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        user_id=user_id,
        username=username,
        activity_type=activity_type,
        description=description,
        metadata=metadata or {},
        timestamp=datetime.utcnow(),
    )

    activity_log_storage[workspace_id].insert(0, activity)

    # Keep only last 1000 activities per workspace
    activity_log_storage[workspace_id] = activity_log_storage[workspace_id][:1000]


@router.post("/", response_model=TeamWorkspace, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    workspace_data: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new team workspace"""
    try:
        # Generate workspace ID
        workspace_id = f"ws_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Create workspace
        workspace = TeamWorkspace(
            workspace_id=workspace_id,
            name=workspace_data.name,
            description=workspace_data.description,
            owner_id=current_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            settings=workspace_data.settings,
            member_count=1,
            features_enabled=["assessments", "reports", "sessions", "api_keys"],
        )

        # Store workspace
        workspaces_storage[workspace_id] = workspace

        # Add owner as first member
        owner_member = TeamMember(
            user_id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            role=TeamRole.OWNER,
            joined_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            is_active=True,
        )

        members_storage[workspace_id] = [owner_member]

        # Log activity
        log_activity(
            workspace_id=workspace_id,
            user_id=current_user.id,
            username=current_user.username,
            activity_type=ActivityType.WORKSPACE_CREATED,
            description=f"Created workspace '{workspace.name}'",
            metadata={"workspace_name": workspace.name},
        )

        logger.info(f"Created workspace {workspace_id} for user {current_user.id}")

        return workspace

    except Exception as e:
        logger.error(f"Failed to create workspace: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create workspace"
        )


@router.get("/", response_model=WorkspaceListResponse)
async def list_workspaces(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """List user's accessible workspaces"""
    try:
        accessible_workspaces = []

        # Find workspaces where user is a member
        for workspace_id, workspace in workspaces_storage.items():
            members = members_storage.get(workspace_id, [])
            user_member = next(
                (m for m in members if m.user_id == current_user.id and m.is_active), None
            )

            if user_member:
                # Update member count
                workspace.member_count = len([m for m in members if m.is_active])
                accessible_workspaces.append(workspace)

        # Sort by updated_at descending
        accessible_workspaces.sort(key=lambda x: x.updated_at, reverse=True)

        logger.info(f"Listed {len(accessible_workspaces)} workspaces for user {current_user.id}")

        return WorkspaceListResponse(
            workspaces=accessible_workspaces, total=len(accessible_workspaces)
        )

    except Exception as e:
        logger.error(f"Failed to list workspaces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workspaces",
        )


@router.get("/{workspace_id}", response_model=TeamWorkspace)
async def get_workspace(
    workspace_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get workspace details"""
    try:
        workspace = workspaces_storage.get(workspace_id)

        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        # Check access
        if not check_workspace_permission(workspace_id, current_user.id, TeamRole.VIEWER):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this workspace"
            )

        # Update member count
        members = members_storage.get(workspace_id, [])
        workspace.member_count = len([m for m in members if m.is_active])

        return workspace

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workspace {workspace_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve workspace"
        )


@router.patch("/{workspace_id}", response_model=TeamWorkspace)
async def update_workspace(
    workspace_id: str,
    update_data: WorkspaceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update workspace settings (admin+ only)"""
    try:
        workspace = workspaces_storage.get(workspace_id)

        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        # Check permissions
        if not check_workspace_permission(workspace_id, current_user.id, TeamRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to update workspace",
            )

        # Update fields
        if update_data.name is not None:
            workspace.name = update_data.name
        if update_data.description is not None:
            workspace.description = update_data.description
        if update_data.settings is not None:
            workspace.settings.update(update_data.settings)

        workspace.updated_at = datetime.utcnow()

        # Log activity
        log_activity(
            workspace_id=workspace_id,
            user_id=current_user.id,
            username=current_user.username,
            activity_type=ActivityType.WORKSPACE_UPDATED,
            description="Updated workspace settings",
            metadata={"changes": update_data.dict(exclude_unset=True)},
        )

        logger.info(f"Updated workspace {workspace_id} by user {current_user.id}")

        return workspace

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update workspace {workspace_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update workspace"
        )


@router.get("/{workspace_id}/members", response_model=MembersListResponse)
async def list_workspace_members(
    workspace_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """List workspace members and pending invitations"""
    try:
        workspace = workspaces_storage.get(workspace_id)

        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        # Check access
        if not check_workspace_permission(workspace_id, current_user.id, TeamRole.VIEWER):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this workspace"
            )

        # Get members and invitations
        members = members_storage.get(workspace_id, [])
        active_members = [m for m in members if m.is_active]

        invitations = invitations_storage.get(workspace_id, [])
        pending_invitations = [inv for inv in invitations if inv.status == InvitationStatus.PENDING]

        logger.info(
            f"Listed {len(active_members)} members and {len(pending_invitations)} invitations for workspace {workspace_id}"
        )

        return MembersListResponse(
            members=active_members,
            invitations=pending_invitations,
            total_members=len(active_members),
            total_invitations=len(pending_invitations),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list workspace members: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workspace members",
        )


@router.post("/{workspace_id}/invite")
async def invite_user_to_workspace(
    workspace_id: str,
    invite_request: InviteUserRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Invite user to workspace (admin+ only)"""
    try:
        workspace = workspaces_storage.get(workspace_id)

        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        # Check permissions
        if not check_workspace_permission(workspace_id, current_user.id, TeamRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to invite users",
            )

        # Check if user is already a member
        members = members_storage.get(workspace_id, [])
        existing_member = next(
            (m for m in members if m.email == invite_request.email and m.is_active), None
        )

        if existing_member:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already a member of this workspace",
            )

        # Check for existing pending invitation
        invitations = invitations_storage.get(workspace_id, [])
        existing_invitation = next(
            (
                inv
                for inv in invitations
                if inv.invited_email == invite_request.email
                and inv.status == InvitationStatus.PENDING
            ),
            None,
        )

        if existing_invitation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already has a pending invitation",
            )

        # Create invitation
        invitation = TeamInvitation(
            invitation_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            workspace_name=workspace.name,
            invited_email=invite_request.email,
            invited_by=current_user.id,
            invited_by_name=current_user.username,
            role=invite_request.role,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=7),
            message=invite_request.message,
        )

        # Store invitation
        if workspace_id not in invitations_storage:
            invitations_storage[workspace_id] = []
        invitations_storage[workspace_id].append(invitation)

        # Send invitation email in background
        background_tasks.add_task(
            send_invitation_email,
            invitation.invited_email,
            workspace.name,
            current_user.username,
            invitation.invitation_id,
            invite_request.role.value,
            invite_request.message,
        )

        # Log activity
        log_activity(
            workspace_id=workspace_id,
            user_id=current_user.id,
            username=current_user.username,
            activity_type=ActivityType.USER_JOINED,
            description=f"Invited {invite_request.email} as {invite_request.role.value}",
            metadata={"invited_email": invite_request.email, "role": invite_request.role.value},
        )

        logger.info(f"Sent invitation to {invite_request.email} for workspace {workspace_id}")

        return {"message": f"Invitation sent to {invite_request.email}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to invite user to workspace: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to send invitation"
        )


@router.patch("/{workspace_id}/members/{user_id}/role")
async def update_member_role(
    workspace_id: str,
    user_id: str,
    role_update: UpdateMemberRole,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update member role (admin+ only)"""
    try:
        workspace = workspaces_storage.get(workspace_id)

        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        # Check permissions
        if not check_workspace_permission(workspace_id, current_user.id, TeamRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to update member roles",
            )

        # Cannot change owner role
        if workspace.owner_id == user_id and role_update.role != TeamRole.OWNER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot change workspace owner role"
            )

        # Find and update member
        members = members_storage.get(workspace_id, [])
        member = next((m for m in members if m.user_id == user_id and m.is_active), None)

        if not member:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

        old_role = member.role
        member.role = role_update.role

        # Log activity
        log_activity(
            workspace_id=workspace_id,
            user_id=current_user.id,
            username=current_user.username,
            activity_type=ActivityType.ROLE_CHANGED,
            description=f"Changed {member.username} role from {old_role.value} to {role_update.role.value}",
            metadata={
                "target_user_id": user_id,
                "target_username": member.username,
                "old_role": old_role.value,
                "new_role": role_update.role.value,
            },
        )

        logger.info(f"Updated role for user {user_id} in workspace {workspace_id}")

        return {"message": f"Updated {member.username} role to {role_update.role.value}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update member role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update member role"
        )


@router.delete("/{workspace_id}/members/{user_id}")
async def remove_member_from_workspace(
    workspace_id: str,
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove member from workspace (admin+ only)"""
    try:
        workspace = workspaces_storage.get(workspace_id)

        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        # Check permissions (admin+ or removing self)
        if user_id != current_user.id and not check_workspace_permission(
            workspace_id, current_user.id, TeamRole.ADMIN
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to remove other members",
            )

        # Cannot remove workspace owner
        if workspace.owner_id == user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot remove workspace owner"
            )

        # Find and deactivate member
        members = members_storage.get(workspace_id, [])
        member = next((m for m in members if m.user_id == user_id and m.is_active), None)

        if not member:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

        member.is_active = False

        # Log activity
        action = "left" if user_id == current_user.id else "was removed from"
        log_activity(
            workspace_id=workspace_id,
            user_id=current_user.id,
            username=current_user.username,
            activity_type=ActivityType.USER_LEFT,
            description=f"{member.username} {action} the workspace",
            metadata={
                "target_user_id": user_id,
                "target_username": member.username,
                "self_removal": user_id == current_user.id,
            },
        )

        logger.info(f"Removed user {user_id} from workspace {workspace_id}")

        return {"message": f"Removed {member.username} from workspace"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove member: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to remove member"
        )


@router.get("/{workspace_id}/activity", response_model=ActivityLogResponse)
async def get_workspace_activity_log(
    workspace_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    activity_type: ActivityType | None = Query(None, description="Filter by activity type"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get workspace activity log (admin+ only)"""
    try:
        workspace = workspaces_storage.get(workspace_id)

        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        # Check permissions
        if not check_workspace_permission(workspace_id, current_user.id, TeamRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to view activity log",
            )

        # Get activity log
        activities = activity_log_storage.get(workspace_id, [])

        # Apply filters
        if activity_type:
            activities = [a for a in activities if a.activity_type == activity_type]

        # Apply pagination
        total = len(activities)
        offset = (page - 1) * page_size
        activities_page = activities[offset : offset + page_size]

        logger.info(
            f"Retrieved {len(activities_page)} activity entries for workspace {workspace_id}"
        )

        return ActivityLogResponse(
            activities=activities_page,
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + page_size < total,
            has_prev=page > 1,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get activity log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve activity log",
        )


@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Delete workspace (owner only)"""
    try:
        workspace = workspaces_storage.get(workspace_id)

        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        # Only owner can delete workspace
        if workspace.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only workspace owner can delete workspace",
            )

        # Delete workspace and related data
        del workspaces_storage[workspace_id]
        members_storage.pop(workspace_id, None)
        invitations_storage.pop(workspace_id, None)
        activity_log_storage.pop(workspace_id, None)

        logger.info(f"Deleted workspace {workspace_id} by owner {current_user.id}")

        return {"message": f"Workspace '{workspace.name}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workspace: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete workspace"
        )
