"""
Audit Log Endpoints

Provides access to system audit logs for compliance and security monitoring.

**Endpoints**:
- GET /audit/logs - Query audit logs (admin only)
- GET /audit/stats - Get audit statistics (admin only)
- POST /audit/verify - Verify audit chain integrity (admin only)
- GET /audit/my-activity - Get current user's activity history
- GET /audit/users/{user_id}/activity - Get specific user's activity (admin only)
"""

import logging
from contextlib import suppress
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import AuditAction, AuditEntry, get_audit_logger
from app.core.auth import TokenPayload, get_current_user
from app.core.config import settings
from app.core.database import get_async_session_factory
from app.db.models import User, UserRole
from app.services.user_service import get_user_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])
security = HTTPBearer()


class AuditLogResponse(BaseModel):
    logs: list[AuditEntry]
    total: int
    verified: bool


class AuditStatsResponse(BaseModel):
    total_events: int
    events_by_severity: dict[str, int]
    events_by_action: dict[str, int]
    last_verification: str


# =============================================================================
# User Activity Response Models
# =============================================================================


class AuditActivityEntry(BaseModel):
    """Individual user activity entry with sanitized details."""

    timestamp: str
    action: str
    action_display: str = Field(..., description="Human-readable action name")
    severity: str
    resource: str
    details: dict = Field(default_factory=dict)
    ip_address: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-25T10:30:00Z",
                "action": "auth.login",
                "action_display": "Login",
                "severity": "info",
                "resource": "/api/v1/auth/login",
                "details": {"method": "password"},
                "ip_address": "192.168.1.1",
            }
        }


class UserActivityResponse(BaseModel):
    """Response model for user activity history."""

    success: bool = True
    user_id: str
    activities: list[AuditActivityEntry]
    total: int
    page: int = 1
    page_size: int = 50
    has_more: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "user_id": "123",
                "activities": [],
                "total": 0,
                "page": 1,
                "page_size": 50,
                "has_more": False,
            }
        }


class UserActivityErrorResponse(BaseModel):
    """Error response for user activity endpoints."""

    success: bool = False
    error: str


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
# Helper Functions
# =============================================================================


def _get_action_display_name(action: str) -> str:
    """Convert audit action code to human-readable name."""
    action_names = {
        # Authentication events
        "auth.login": "Login",
        "auth.logout": "Logout",
        "auth.failed": "Authentication Failed",
        "auth.token_refresh": "Token Refresh",
        "auth.mfa_challenge": "MFA Challenge",
        "auth.mfa_verified": "MFA Verified",
        # API key events
        "apikey.created": "API Key Created",
        "apikey.rotated": "API Key Rotated",
        "apikey.revoked": "API Key Revoked",
        "apikey.used": "API Key Used",
        # Prompt events
        "prompt.transform": "Prompt Transformed",
        "prompt.enhance": "Prompt Enhanced",
        "prompt.jailbreak": "Jailbreak Attempt",
        "prompt.batch_process": "Batch Processing",
        # Config events
        "config.change": "Configuration Changed",
        "config.view": "Configuration Viewed",
        # User management events
        "user.create": "User Created",
        "user.modify": "User Modified",
        "user.delete": "User Deleted",
        "user.role_change": "Role Changed",
        "user.register": "User Registered",
        "user.verify": "Email Verified",
        "user.password_change": "Password Changed",
        "user.profile_update": "Profile Updated",
        # Campaign events
        "campaign.create": "Campaign Created",
        "campaign.update": "Campaign Updated",
        "campaign.delete": "Campaign Deleted",
        "campaign.share": "Campaign Shared",
        "campaign.unshare": "Campaign Share Removed",
        # Security events
        "security.rate_limit": "Rate Limited",
        "security.blocked_request": "Request Blocked",
        "security.injection_attempt": "Injection Attempt Detected",
        "security.unauthorized": "Unauthorized Access",
    }
    return action_names.get(action, action.replace(".", " ").title())


def _entry_to_activity(entry: AuditEntry) -> AuditActivityEntry:
    """Convert AuditEntry to AuditActivityEntry."""
    return AuditActivityEntry(
        timestamp=entry.timestamp,
        action=entry.action,
        action_display=_get_action_display_name(entry.action),
        severity=entry.severity,
        resource=entry.resource,
        details=entry.details,
        ip_address=entry.ip_address,
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
        logger.warning("Activity access attempted with API key authentication")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Activity endpoints require user authentication, not API key",
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
        logger.warning(f"User not found for activity request: id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not user.is_active:
        logger.warning(f"Inactive user attempted activity access: id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user


# =============================================================================
# Legacy Admin Access Verification (API Key Based)
# =============================================================================


async def verify_admin_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify admin access.
    In a real app, this would check roles/claims.
    For now, we check the API key against the admin key.
    """
    if not credentials or credentials.credentials != settings.CHIMERA_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )
    return credentials.credentials


@router.get(
    "/logs",
    response_model=AuditLogResponse,
    summary="Query audit logs",
    description="Retrieve audit logs with filtering capabilities.",
)
async def get_audit_logs(
    action: str | None = Query(None, description="Filter by action type"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    start_date: datetime | None = Query(None, description="Start date filter"),
    end_date: datetime | None = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=1000),
    severity: str | None = Query(None, description="Filter by severity"),
    _admin: str = Depends(verify_admin_access),
):
    """
    Retrieve audit logs based on filters.
    """
    audit_logger = get_audit_logger()

    # Convert string action to Enum if provided
    audit_action = None
    if action:
        with suppress(ValueError):
            audit_action = AuditAction(action)

    logs = audit_logger.query(
        action=audit_action, user_id=user_id, start_time=start_date, end_time=end_date, limit=limit
    )

    # Manual severity filter since query() might not support it directly in all implementations
    if severity:
        logs = [log for log in logs if log.severity == severity]

    # Verify chain integrity for the returned logs (simplified check)
    is_valid, _, _ = audit_logger.verify_chain()

    return AuditLogResponse(logs=logs, total=len(logs), verified=is_valid)


@router.get(
    "/stats",
    response_model=AuditStatsResponse,
    summary="Get audit statistics",
    description="Retrieve summary statistics of audit events.",
)
async def get_audit_stats(_admin: str = Depends(verify_admin_access)):
    """
    Get statistical summary of audit logs.
    """
    audit_logger = get_audit_logger()
    all_logs = (
        audit_logger.storage.get_all()
    )  # Get all logs for stats (might be heavy in prod, use DB agg)

    severity_counts = {}
    action_counts = {}

    for log in all_logs:
        severity_counts[log.severity] = severity_counts.get(log.severity, 0) + 1
        action_counts[log.action] = action_counts.get(log.action, 0) + 1

    return AuditStatsResponse(
        total_events=len(all_logs),
        events_by_severity=severity_counts,
        events_by_action=action_counts,
        last_verification=datetime.utcnow().isoformat(),
    )


@router.post(
    "/verify",
    summary="Verify audit chain",
    description="Trigger a cryptographic verification of the audit log chain.",
)
async def verify_audit_chain(_admin: str = Depends(verify_admin_access)):
    """
    Cryptographically verify the tamper-evident hash chain.
    """
    audit_logger = get_audit_logger()
    is_valid, failed_hash, failed_index = audit_logger.verify_chain()

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail={
                "message": "Audit chain verification FAILED",
                "failed_hash": failed_hash,
                "failed_index": failed_index,
            },
        )

    return {
        "status": "success",
        "message": "Audit chain verified successfully",
        "timestamp": datetime.utcnow().isoformat(),
        "total_verified": len(audit_logger.storage.get_all()),
    }


# =============================================================================
# User Activity Endpoints
# =============================================================================


@router.get(
    "/my-activity",
    status_code=status.HTTP_200_OK,
    response_model=UserActivityResponse,
    summary="Get current user's activity history",
    description="""
Get the activity history for the currently authenticated user.

**Features**:
- View your own login attempts, password changes, API key usage, etc.
- Paginated results with configurable page size
- Filter by action type (e.g., auth.login, user.password_change)
- Filter by date range

**Supported Actions**:
- `auth.login` - Login attempts
- `auth.logout` - Logout events
- `auth.failed` - Failed authentication
- `user.password_change` - Password changes
- `user.profile_update` - Profile updates
- `apikey.created` - API key creation
- `apikey.revoked` - API key revocation
- And more...

**Authentication**: Requires JWT token (API key authentication not supported)

**Example Response**:
```json
{
  "success": true,
  "user_id": "123",
  "activities": [
    {
      "timestamp": "2024-01-25T10:30:00Z",
      "action": "auth.login",
      "action_display": "Login",
      "severity": "info",
      "resource": "/api/v1/auth/login",
      "details": {},
      "ip_address": "192.168.1.1"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 50,
  "has_more": false
}
```
    """,
    responses={
        200: {
            "description": "Activity history retrieved successfully",
            "model": UserActivityResponse,
        },
        401: {"description": "Not authenticated"},
        403: {"description": "API key authentication not supported for activity endpoints"},
    },
    tags=["audit", "users"],
)
async def get_my_activity(
    action: str | None = Query(
        None,
        description="Filter by action type (e.g., auth.login, user.password_change)",
        examples=["auth.login", "user.password_change", "apikey.created"],
    ),
    start_date: datetime | None = Query(
        None,
        description="Start date for filtering (ISO 8601 format)",
    ),
    end_date: datetime | None = Query(
        None,
        description="End date for filtering (ISO 8601 format)",
    ),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=100, description="Number of results per page"),
    current_user: TokenPayload = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> UserActivityResponse:
    """
    Get the current user's activity history.

    Returns a paginated list of audit entries for the authenticated user.
    """
    user = await _get_authenticated_user(current_user, session)
    user_id_str = str(user.id)

    audit_logger = get_audit_logger()

    # Convert string action to Enum if provided
    audit_action = None
    if action:
        with suppress(ValueError):
            audit_action = AuditAction(action)

    # Calculate offset for pagination
    # We fetch extra to determine if there are more results
    fetch_limit = page_size + 1
    offset = (page - 1) * page_size

    # Query audit logs for this user
    all_user_logs = audit_logger.query(
        action=audit_action,
        user_id=user_id_str,
        start_time=start_date,
        end_time=end_date,
        limit=offset + fetch_limit,
    )

    # Apply pagination
    total = len(all_user_logs)
    paginated_logs = all_user_logs[offset : offset + fetch_limit]

    # Check if there are more results
    has_more = len(paginated_logs) > page_size
    if has_more:
        paginated_logs = paginated_logs[:page_size]

    # Convert to activity entries
    activities = [_entry_to_activity(entry) for entry in paginated_logs]

    logger.info(f"Retrieved {len(activities)} activity entries for user {user.id} (page {page})")

    return UserActivityResponse(
        success=True,
        user_id=user_id_str,
        activities=activities,
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more,
    )


@router.get(
    "/users/{user_id}/activity",
    status_code=status.HTTP_200_OK,
    response_model=UserActivityResponse,
    summary="Get specific user's activity history (Admin only)",
    description="""
Get the activity history for a specific user.

**ADMIN ONLY**: This endpoint requires admin role.

**Features**:
- View any user's login attempts, password changes, API key usage, etc.
- Paginated results with configurable page size
- Filter by action type and date range

**Authentication**: Requires JWT token with admin role

**Path Parameters**:
- `user_id` - The ID of the user whose activity to retrieve

**Example Response**:
```json
{
  "success": true,
  "user_id": "456",
  "activities": [
    {
      "timestamp": "2024-01-25T10:30:00Z",
      "action": "auth.login",
      "action_display": "Login",
      "severity": "info",
      "resource": "/api/v1/auth/login",
      "details": {},
      "ip_address": "192.168.1.1"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 50,
  "has_more": false
}
```
    """,
    responses={
        200: {
            "description": "Activity history retrieved successfully",
            "model": UserActivityResponse,
        },
        401: {"description": "Not authenticated"},
        403: {"description": "Admin role required or API key authentication not supported"},
        404: {"description": "User not found"},
    },
    tags=["audit", "admin"],
)
async def get_user_activity(
    user_id: int = Path(
        ...,
        description="The ID of the user whose activity to retrieve",
        ge=1,
    ),
    action: str | None = Query(
        None,
        description="Filter by action type (e.g., auth.login, user.password_change)",
        examples=["auth.login", "user.password_change", "apikey.created"],
    ),
    start_date: datetime | None = Query(
        None,
        description="Start date for filtering (ISO 8601 format)",
    ),
    end_date: datetime | None = Query(
        None,
        description="End date for filtering (ISO 8601 format)",
    ),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=100, description="Number of results per page"),
    current_user: TokenPayload = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> UserActivityResponse:
    """
    Get activity history for a specific user.

    Admin only endpoint. Returns a paginated list of audit entries.
    """
    # Authenticate current user
    admin_user = await _get_authenticated_user(current_user, session)

    # Check admin role
    if admin_user.role != UserRole.ADMIN:
        logger.warning(
            f"Non-admin user {admin_user.id} attempted to access user {user_id} activity"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to view other users' activity",
        )

    # Verify target user exists
    user_service = get_user_service(session)
    target_user = await user_service.get_user_by_id(user_id)

    if not target_user:
        logger.warning(f"Admin {admin_user.id} requested activity for non-existent user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )

    user_id_str = str(user_id)
    audit_logger = get_audit_logger()

    # Convert string action to Enum if provided
    audit_action = None
    if action:
        with suppress(ValueError):
            audit_action = AuditAction(action)

    # Calculate offset for pagination
    fetch_limit = page_size + 1
    offset = (page - 1) * page_size

    # Query audit logs for the target user
    all_user_logs = audit_logger.query(
        action=audit_action,
        user_id=user_id_str,
        start_time=start_date,
        end_time=end_date,
        limit=offset + fetch_limit,
    )

    # Apply pagination
    total = len(all_user_logs)
    paginated_logs = all_user_logs[offset : offset + fetch_limit]

    # Check if there are more results
    has_more = len(paginated_logs) > page_size
    if has_more:
        paginated_logs = paginated_logs[:page_size]

    # Convert to activity entries
    activities = [_entry_to_activity(entry) for entry in paginated_logs]

    logger.info(
        f"Admin {admin_user.id} retrieved {len(activities)} activity entries "
        f"for user {user_id} (page {page})"
    )

    return UserActivityResponse(
        success=True,
        user_id=user_id_str,
        activities=activities,
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more,
    )
