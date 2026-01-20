"""Attack Session Replay & Sharing Endpoints.

Phase 2 feature for competitive differentiation:
- Complete session saving with full context
- Reproducible test execution
- Secure sharing with team members
- Import/export capabilities
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.database import get_db
from app.core.observability import get_logger
from app.db.models import User

logger = get_logger("chimera.api.sessions")
router = APIRouter()


# Session Models
class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    SHARED = "shared"


class SharePermission(str, Enum):
    VIEW = "view"
    REPLAY = "replay"
    EDIT = "edit"
    ADMIN = "admin"


class AttackStep(BaseModel):
    """Individual step in an attack session."""

    step_id: str
    timestamp: datetime
    step_type: str  # "prompt", "transform", "execute", "analyze"

    # Step data
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Results
    success: bool
    execution_time: float  # seconds
    error_message: str | None = None

    # Context
    technique_used: str | None = None
    target_model: str | None = None
    target_provider: str | None = None


class AttackSession(BaseModel):
    """Complete attack session with all steps and context."""

    session_id: str
    name: str
    description: str | None = None

    # Ownership and sharing
    owner_id: str
    created_by: str
    created_at: datetime
    updated_at: datetime

    # Session state
    status: SessionStatus = SessionStatus.ACTIVE

    # Attack configuration
    target_config: dict[str, Any] = Field(default_factory=dict)
    techniques_config: dict[str, Any] = Field(default_factory=dict)
    original_prompt: str

    # Session data
    steps: list[AttackStep] = Field(default_factory=list)
    results_summary: dict[str, Any] = Field(default_factory=dict)

    # Metrics
    total_steps: int = 0
    successful_steps: int = 0
    total_execution_time: float = 0.0

    # Sharing
    is_public: bool = False
    shared_with: list[dict[str, Any]] = Field(default_factory=list)

    # Tags and categorization
    tags: list[str] = Field(default_factory=list)
    category: str | None = None


class SessionCreate(BaseModel):
    """Request to create a new attack session."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)

    # Initial configuration
    target_provider: str
    target_model: str
    target_config: dict[str, Any] = Field(default_factory=dict)
    original_prompt: str = Field(..., min_length=1)

    # Optional configuration
    techniques_config: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    category: str | None = None


class SessionUpdate(BaseModel):
    """Request to update session metadata."""

    name: str | None = None
    description: str | None = None
    status: SessionStatus | None = None
    tags: list[str] | None = None
    category: str | None = None


class SessionShare(BaseModel):
    """Request to share session with users."""

    user_ids: list[str] = Field(..., min_length=1)
    permission: SharePermission = SharePermission.VIEW
    message: str | None = None
    expires_at: datetime | None = None


class SessionReplay(BaseModel):
    """Request to replay a session."""

    session_id: str
    replay_name: str | None = None

    # Replay options
    start_from_step: int = 0
    stop_at_step: int | None = None
    modify_target: dict[str, str] | None = None  # Change provider/model
    skip_steps: list[int] = Field(default_factory=list)


class SessionExport(BaseModel):
    """Session export data."""

    session: AttackSession
    export_format: str = "json"
    export_metadata: dict[str, Any] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    """Response for session listing."""

    sessions: list[AttackSession]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# In-memory storage for demo (in production, would use database)
sessions_storage: dict[str, AttackSession] = {}
session_shares: dict[str, list[dict[str, Any]]] = {}


@router.post("/", response_model=AttackSession, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: SessionCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Create a new attack session for recording and replay.

    This endpoint initializes a new session that can record all attack steps,
    enabling full reproducibility and sharing capabilities.
    """
    try:
        # Generate session ID
        session_id = f"sess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Create session object
        session = AttackSession(
            session_id=session_id,
            name=session_data.name,
            description=session_data.description,
            owner_id=current_user.id,
            created_by=current_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            target_config={
                "provider": session_data.target_provider,
                "model": session_data.target_model,
                **session_data.target_config,
            },
            techniques_config=session_data.techniques_config,
            original_prompt=session_data.original_prompt,
            tags=session_data.tags,
            category=session_data.category,
        )

        # Store session (in production, save to database)
        sessions_storage[session_id] = session

        logger.info(f"Created attack session {session_id} for user {current_user.id}")

        return session

    except Exception as e:
        logger.exception(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create attack session",
        )


@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    status_filter: Annotated[
        SessionStatus | None, Query(description="Filter by status", alias="status")
    ] = None,
    category: Annotated[str | None, Query(description="Filter by category")] = None,
    search: Annotated[str | None, Query(description="Search in name and description")] = None,
    owner_only: Annotated[bool, Query(description="Show only owned sessions")] = False,
    shared_only: Annotated[bool, Query(description="Show only shared sessions")] = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List attack sessions with filtering and pagination."""
    try:
        # Get all sessions accessible to user
        accessible_sessions = []

        for session in sessions_storage.values():
            # Check access permissions
            has_access = False

            # Owner access
            if session.owner_id == current_user.id and not shared_only:
                has_access = True

            # Shared access
            if session.session_id in session_shares:
                user_shares = session_shares[session.session_id]
                if any(share["user_id"] == current_user.id for share in user_shares):
                    if not owner_only:
                        has_access = True

            # Public access
            if session.is_public and not owner_only:
                has_access = True

            if has_access:
                accessible_sessions.append(session)

        # Apply filters
        filtered_sessions = accessible_sessions

        if status_filter:
            filtered_sessions = [s for s in filtered_sessions if s.status == status_filter]

        if category:
            filtered_sessions = [s for s in filtered_sessions if s.category == category]

        if search:
            search_term = search.lower()
            filtered_sessions = [
                s
                for s in filtered_sessions
                if search_term in s.name.lower()
                or (s.description and search_term in s.description.lower())
            ]

        # Sort by updated_at descending
        filtered_sessions.sort(key=lambda x: x.updated_at, reverse=True)

        # Apply pagination
        total = len(filtered_sessions)
        offset = (page - 1) * page_size
        sessions_page = filtered_sessions[offset : offset + page_size]

        logger.info(f"Listed {len(sessions_page)} sessions for user {current_user.id}")

        return SessionListResponse(
            sessions=sessions_page,
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + page_size < total,
            has_prev=page > 1,
        )

    except Exception as e:
        logger.exception(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions",
        )


@router.get("/{session_id}", response_model=AttackSession)
async def get_session(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Get detailed session information including all steps."""
    try:
        session = sessions_storage.get(session_id)

        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        # Check access permissions
        has_access = False

        if session.owner_id == current_user.id or session.is_public:
            has_access = True
        elif session_id in session_shares:
            user_shares = session_shares[session_id]
            has_access = any(share["user_id"] == current_user.id for share in user_shares)

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this session",
            )

        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session",
        )


@router.patch("/{session_id}", response_model=AttackSession)
async def update_session(
    session_id: str,
    update_data: SessionUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Update session metadata (owner only)."""
    try:
        session = sessions_storage.get(session_id)

        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        if session.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only session owner can update metadata",
            )

        # Update fields
        if update_data.name is not None:
            session.name = update_data.name
        if update_data.description is not None:
            session.description = update_data.description
        if update_data.status is not None:
            session.status = update_data.status
        if update_data.tags is not None:
            session.tags = update_data.tags
        if update_data.category is not None:
            session.category = update_data.category

        session.updated_at = datetime.utcnow()

        logger.info(f"Updated session {session_id} for user {current_user.id}")

        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update session",
        )


@router.post("/{session_id}/steps", response_model=AttackStep)
async def add_session_step(
    session_id: str,
    step_data: dict[str, Any],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Add a step to an active session."""
    try:
        session = sessions_storage.get(session_id)

        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        if session.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only session owner can add steps",
            )

        if session.status not in [SessionStatus.ACTIVE, SessionStatus.PAUSED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot add steps to completed or failed sessions",
            )

        # Create step
        step = AttackStep(
            step_id=f"step_{len(session.steps) + 1}_{str(uuid.uuid4())[:8]}",
            timestamp=datetime.utcnow(),
            step_type=step_data.get("step_type", "execute"),
            input_data=step_data.get("input_data", {}),
            output_data=step_data.get("output_data", {}),
            metadata=step_data.get("metadata", {}),
            success=step_data.get("success", False),
            execution_time=step_data.get("execution_time", 0.0),
            error_message=step_data.get("error_message"),
            technique_used=step_data.get("technique_used"),
            target_model=step_data.get("target_model"),
            target_provider=step_data.get("target_provider"),
        )

        # Add step to session
        session.steps.append(step)
        session.total_steps = len(session.steps)
        session.successful_steps = len([s for s in session.steps if s.success])
        session.total_execution_time += step.execution_time
        session.updated_at = datetime.utcnow()

        logger.info(f"Added step to session {session_id}")

        return step

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to add step to session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add session step",
        )


@router.post("/{session_id}/share")
async def share_session(
    session_id: str,
    share_data: SessionShare,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Share session with other users."""
    try:
        session = sessions_storage.get(session_id)

        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        if session.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only session owner can share sessions",
            )

        # Validate user IDs (in production, check if users exist)
        for user_id in share_data.user_ids:
            # TODO: Validate user exists in database
            pass

        # Initialize shares for session if not exists
        if session_id not in session_shares:
            session_shares[session_id] = []

        # Add sharing entries
        for user_id in share_data.user_ids:
            # Check if already shared with this user
            existing_share = next(
                (s for s in session_shares[session_id] if s["user_id"] == user_id),
                None,
            )

            if existing_share:
                # Update existing share
                existing_share["permission"] = share_data.permission.value
                existing_share["expires_at"] = share_data.expires_at
                existing_share["updated_at"] = datetime.utcnow()
            else:
                # Add new share
                session_shares[session_id].append(
                    {
                        "user_id": user_id,
                        "permission": share_data.permission.value,
                        "shared_by": current_user.id,
                        "shared_at": datetime.utcnow(),
                        "expires_at": share_data.expires_at,
                        "message": share_data.message,
                    },
                )

        # Update session sharing info
        session.shared_with = session_shares[session_id]
        session.status = (
            SessionStatus.SHARED if session.status == SessionStatus.COMPLETED else session.status
        )
        session.updated_at = datetime.utcnow()

        logger.info(f"Shared session {session_id} with {len(share_data.user_ids)} users")

        return {"message": f"Session shared with {len(share_data.user_ids)} users"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to share session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to share session",
        )


@router.post("/{session_id}/replay", response_model=AttackSession)
async def replay_session(
    session_id: str,
    replay_data: SessionReplay,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Create a replay of an existing session."""
    try:
        original_session = sessions_storage.get(session_id)

        if not original_session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        # Check replay permissions
        has_replay_access = False

        if original_session.owner_id == current_user.id or original_session.is_public:
            has_replay_access = True
        elif session_id in session_shares:
            user_shares = session_shares[session_id]
            user_share = next((s for s in user_shares if s["user_id"] == current_user.id), None)
            if user_share and user_share["permission"] in ["replay", "edit", "admin"]:
                has_replay_access = True

        if not has_replay_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to replay this session",
            )

        # Create replay session
        replay_session_id = (
            f"replay_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        )

        # Copy original session configuration
        replay_session = AttackSession(
            session_id=replay_session_id,
            name=replay_data.replay_name or f"Replay of {original_session.name}",
            description=f"Replay of session {session_id}",
            owner_id=current_user.id,
            created_by=current_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            target_config=original_session.target_config.copy(),
            techniques_config=original_session.techniques_config.copy(),
            original_prompt=original_session.original_prompt,
            tags=[*original_session.tags, "replay"],
            category=original_session.category,
        )

        # Apply modifications if specified
        if replay_data.modify_target:
            if "provider" in replay_data.modify_target:
                replay_session.target_config["provider"] = replay_data.modify_target["provider"]
            if "model" in replay_data.modify_target:
                replay_session.target_config["model"] = replay_data.modify_target["model"]

        # TODO: In production, this would trigger actual replay execution
        # For now, we'll copy the steps with modified timestamps

        steps_to_replay = original_session.steps[replay_data.start_from_step :]
        if replay_data.stop_at_step:
            steps_to_replay = steps_to_replay[
                : replay_data.stop_at_step - replay_data.start_from_step
            ]

        # Filter out skipped steps
        if replay_data.skip_steps:
            steps_to_replay = [
                step for i, step in enumerate(steps_to_replay) if i not in replay_data.skip_steps
            ]

        # Copy steps with new timestamps and IDs
        for i, original_step in enumerate(steps_to_replay):
            new_step = AttackStep(
                step_id=f"replay_step_{i + 1}_{str(uuid.uuid4())[:8]}",
                timestamp=datetime.utcnow(),
                step_type=original_step.step_type,
                input_data=original_step.input_data.copy(),
                output_data=original_step.output_data.copy(),
                metadata={**original_step.metadata, "replayed_from": original_step.step_id},
                success=original_step.success,
                execution_time=original_step.execution_time,
                error_message=original_step.error_message,
                technique_used=original_step.technique_used,
                target_model=replay_session.target_config.get("model"),
                target_provider=replay_session.target_config.get("provider"),
            )
            replay_session.steps.append(new_step)

        # Update session metrics
        replay_session.total_steps = len(replay_session.steps)
        replay_session.successful_steps = len([s for s in replay_session.steps if s.success])
        replay_session.total_execution_time = sum(s.execution_time for s in replay_session.steps)
        replay_session.status = SessionStatus.COMPLETED

        # Store replay session
        sessions_storage[replay_session_id] = replay_session

        logger.info(f"Created replay session {replay_session_id} from {session_id}")

        return replay_session

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to replay session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to replay session",
        )


@router.get("/{session_id}/export", response_model=SessionExport)
async def export_session(
    session_id: str,
    format: Annotated[str, Query(description="Export format")] = "json",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export session data for sharing or backup."""
    try:
        session = sessions_storage.get(session_id)

        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        # Check export permissions (same as view permissions)
        has_access = False

        if session.owner_id == current_user.id or session.is_public:
            has_access = True
        elif session_id in session_shares:
            user_shares = session_shares[session_id]
            has_access = any(share["user_id"] == current_user.id for share in user_shares)

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this session",
            )

        # Generate export metadata
        export_metadata = {
            "exported_by": current_user.id,
            "exported_at": datetime.utcnow(),
            "export_format": format,
            "original_session_id": session_id,
            "chimera_version": "2.0.0",  # TODO: Get actual version
        }

        return SessionExport(session=session, export_format=format, export_metadata=export_metadata)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to export session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export session",
        )


@router.post("/import", response_model=AttackSession)
async def import_session(
    import_data: SessionExport,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Import a previously exported session."""
    try:
        # Generate new session ID for imported session
        new_session_id = (
            f"import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        )

        # Create imported session
        imported_session = import_data.session.model_copy()
        imported_session.session_id = new_session_id
        imported_session.owner_id = current_user.id
        imported_session.created_by = current_user.id
        imported_session.created_at = datetime.utcnow()
        imported_session.updated_at = datetime.utcnow()

        # Clear sharing data for imported session
        imported_session.shared_with = []
        imported_session.is_public = False

        # Add import metadata to tags
        if "imported" not in imported_session.tags:
            imported_session.tags.append("imported")

        # Update step IDs to avoid conflicts
        for i, step in enumerate(imported_session.steps):
            step.step_id = f"import_step_{i + 1}_{str(uuid.uuid4())[:8]}"

        # Store imported session
        sessions_storage[new_session_id] = imported_session

        logger.info(f"Imported session {new_session_id} for user {current_user.id}")

        return imported_session

    except Exception as e:
        logger.exception(f"Failed to import session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import session",
        )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete a session (owner only)."""
    try:
        session = sessions_storage.get(session_id)

        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        if session.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only session owner can delete sessions",
            )

        # Delete session and its shares
        del sessions_storage[session_id]
        session_shares.pop(session_id, None)

        logger.info(f"Deleted session {session_id} for user {current_user.id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session",
        )
