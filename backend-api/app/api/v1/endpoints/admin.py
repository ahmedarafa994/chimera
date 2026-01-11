"""
Admin Endpoints for Feature Flag and System Management

Provides administrative controls for:
- Feature flag management
- Technique configuration
- System statistics
- User management (admin-only)

Part of Phase 3: Transformation implementation.
Part of Phase 7: Admin User Management.
"""

import logging
import secrets
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import AuditAction, audit_log
from app.core.auth import TokenPayload, get_current_user
from app.core.config import settings
from app.core.database import get_async_session_factory
from app.core.logging import logger
from app.db.models import User, UserRole
from app.repositories.user_repository import get_user_repository
from app.services.feature_flag_service import FeatureFlagService, get_feature_flags
from app.services.tenant_service import TenantService, TenantTier, get_tenant_service
from app.services.usage_tracker import UsageTracker, get_usage_tracker
from app.services.user_service import get_user_service

admin_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBearer()


# =============================================================================
# Request/Response Models
# =============================================================================


class TechniqueToggleRequest(BaseModel):
    technique_name: str
    enabled: bool


class TechniqueToggleResponse(BaseModel):
    success: bool
    technique_name: str
    enabled: bool
    message: str


class TechniqueListResponse(BaseModel):
    techniques: list[dict[str, Any]]
    total_count: int


class FeatureFlagStatsResponse(BaseModel):
    total_techniques: int
    enabled_count: int
    disabled_count: int
    risk_distribution: dict[str, int]
    approval_required_count: int
    plugin_enabled: bool


# =============================================================================
# Authentication
# =============================================================================


async def verify_admin_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Verify API key for admin endpoints using timing-safe comparison."""

    if not credentials or not settings.CHIMERA_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Timing-safe comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, settings.CHIMERA_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/feature-flags", response_model=TechniqueListResponse)
async def list_feature_flags(
    _api_key: str = Depends(verify_admin_api_key),
    flags: FeatureFlagService = Depends(get_feature_flags),
):
    """
    List all technique configurations with their feature flags.

    Returns all techniques with their enabled status, risk levels,
    and approval requirements.
    """
    techniques = []
    for name in flags.list_enabled_techniques():
        config = flags.get_technique_config(name)
        if config:
            techniques.append(
                {
                    "name": name,
                    "enabled": config.get("enabled", False),
                    "risk_level": config.get("risk_level", "unknown"),
                    "requires_approval": config.get("requires_approval", False),
                    "description": config.get("description", ""),
                }
            )

    # Also include disabled techniques
    for name, config in flags._config.items():
        if (
            isinstance(config, dict)
            and name != "plugin_settings"
            and not config.get("enabled", False)
        ):
            techniques.append(
                {
                    "name": name,
                    "enabled": False,
                    "risk_level": config.get("risk_level", "unknown"),
                    "requires_approval": config.get("requires_approval", False),
                    "description": config.get("description", ""),
                }
            )

    return TechniqueListResponse(techniques=techniques, total_count=len(techniques))


@router.get("/feature-flags/stats", response_model=FeatureFlagStatsResponse)
async def get_feature_flag_stats(
    _api_key: str = Depends(verify_admin_api_key),
    flags: FeatureFlagService = Depends(get_feature_flags),
):
    """
    Get statistics about feature flag configuration.

    Returns counts by risk level, enabled/disabled status,
    and approval requirements.
    """
    stats = flags.get_statistics()
    return FeatureFlagStatsResponse(**stats)


@router.post("/feature-flags/toggle", response_model=TechniqueToggleResponse)
async def toggle_feature_flag(
    request: TechniqueToggleRequest,
    _api_key: str = Depends(verify_admin_api_key),
    flags: FeatureFlagService = Depends(get_feature_flags),
):
    """
    Toggle a technique's enabled status.

    Note: This is a runtime change and does NOT persist to disk.
    For permanent changes, modify techniques.yaml directly.
    """
    success = flags.set_technique_enabled(request.technique_name, request.enabled)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Technique '{request.technique_name}' not found",
        )

    logger.info(f"Admin toggled technique '{request.technique_name}' to {request.enabled}")

    return TechniqueToggleResponse(
        success=True,
        technique_name=request.technique_name,
        enabled=request.enabled,
        message=f"Technique {'enabled' if request.enabled else 'disabled'} (runtime only)",
    )


@router.post("/feature-flags/reload")
async def reload_feature_flags(
    _api_key: str = Depends(verify_admin_api_key),
    flags: FeatureFlagService = Depends(get_feature_flags),
):
    """
    Reload feature flag configuration from disk.

    Use this after modifying techniques.yaml to apply changes
    without restarting the server.
    """
    success = flags.reload_config()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload configuration",
        )

    logger.info("Admin reloaded feature flag configuration")

    return {
        "success": True,
        "message": "Configuration reloaded successfully",
        "enabled_techniques": len(flags.list_enabled_techniques()),
    }


@router.get("/feature-flags/{technique_name}")
async def get_technique_details(
    technique_name: str,
    _api_key: str = Depends(verify_admin_api_key),
    flags: FeatureFlagService = Depends(get_feature_flags),
):
    """
    Get detailed configuration for a specific technique.
    """
    config = flags.get_technique_config(technique_name)

    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Technique '{technique_name}' not found"
        )

    return {"name": technique_name, **config}


# =============================================================================
# Enterprise: Tenant Management Endpoints
# =============================================================================


class CreateTenantRequest(BaseModel):
    tenant_id: str
    name: str
    tier: str = "free"
    api_key: str | None = None


class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    tier: str
    rate_limit_per_minute: int
    monthly_quota: int
    is_active: bool


@router.get("/tenants")
async def list_tenants(
    _api_key: str = Depends(verify_admin_api_key),
    tenants: TenantService = Depends(get_tenant_service),
):
    """List all tenants."""
    tenant_list = tenants.list_tenants()
    return {
        "tenants": [
            {
                "tenant_id": t.tenant_id,
                "name": t.name,
                "tier": t.tier.value,
                "rate_limit_per_minute": t.rate_limit_per_minute,
                "monthly_quota": t.monthly_quota,
                "is_active": t.is_active,
            }
            for t in tenant_list
        ],
        "total_count": len(tenant_list),
    }


@router.post("/tenants", response_model=TenantResponse)
async def create_tenant(
    request: CreateTenantRequest,
    _api_key: str = Depends(verify_admin_api_key),
    tenants: TenantService = Depends(get_tenant_service),
):
    """Create a new tenant."""
    try:
        tier = TenantTier(request.tier.lower())
    except ValueError:
        tier = TenantTier.FREE

    try:
        tenant = tenants.create_tenant(
            tenant_id=request.tenant_id, name=request.name, tier=tier, api_key=request.api_key
        )
        return TenantResponse(
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            tier=tenant.tier.value,
            rate_limit_per_minute=tenant.rate_limit_per_minute,
            monthly_quota=tenant.monthly_quota,
            is_active=tenant.is_active,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/tenants/{tenant_id}")
async def get_tenant(
    tenant_id: str,
    _api_key: str = Depends(verify_admin_api_key),
    tenants: TenantService = Depends(get_tenant_service),
):
    """Get tenant details."""
    tenant = tenants.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")

    return {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "tier": tenant.tier.value,
        "rate_limit_per_minute": tenant.rate_limit_per_minute,
        "monthly_quota": tenant.monthly_quota,
        "is_active": tenant.is_active,
        "allowed_techniques": tenant.allowed_techniques,
        "blocked_techniques": tenant.blocked_techniques,
        "custom_settings": tenant.custom_settings,
        "created_at": tenant.created_at.isoformat(),
    }


@router.delete("/tenants/{tenant_id}")
async def delete_tenant(
    tenant_id: str,
    _api_key: str = Depends(verify_admin_api_key),
    tenants: TenantService = Depends(get_tenant_service),
):
    """Delete a tenant."""
    success = tenants.delete_tenant(tenant_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Tenant not found or cannot delete: {tenant_id}"
        )
    return {"success": True, "message": f"Tenant {tenant_id} deleted"}


@router.get("/tenants/stats/summary")
async def get_tenant_statistics(
    _api_key: str = Depends(verify_admin_api_key),
    tenants: TenantService = Depends(get_tenant_service),
):
    """Get tenant statistics."""
    return tenants.get_statistics()


# =============================================================================
# Enterprise: Usage Analytics Endpoints
# =============================================================================


@router.get("/usage/global")
async def get_global_usage(
    _api_key: str = Depends(verify_admin_api_key),
    tracker: UsageTracker = Depends(get_usage_tracker),
):
    """Get global usage statistics."""
    return tracker.get_global_statistics()


@router.get("/usage/tenant/{tenant_id}")
async def get_tenant_usage(
    tenant_id: str,
    hours: int = 24,
    _api_key: str = Depends(verify_admin_api_key),
    tracker: UsageTracker = Depends(get_usage_tracker),
):
    """Get usage summary for a specific tenant."""
    summary = tracker.get_tenant_summary(tenant_id, period_hours=hours)
    return {
        "tenant_id": summary.tenant_id,
        "period_start": summary.period_start.isoformat(),
        "period_end": summary.period_end.isoformat(),
        "total_requests": summary.total_requests,
        "requests_by_endpoint": summary.requests_by_endpoint,
        "requests_by_technique": summary.requests_by_technique,
        "total_tokens": summary.total_tokens,
        "total_errors": summary.total_errors,
        "cache_hit_rate": f"{summary.cache_hit_rate:.1f}%",
        "avg_duration_ms": f"{summary.avg_duration_ms:.1f}",
    }


@router.get("/usage/techniques/top")
async def get_top_techniques(
    limit: int = 10,
    tenant_id: str | None = None,
    _api_key: str = Depends(verify_admin_api_key),
    tracker: UsageTracker = Depends(get_usage_tracker),
):
    """Get most used techniques."""
    top = tracker.get_top_techniques(tenant_id=tenant_id, limit=limit)
    return {
        "techniques": [{"name": name, "count": count} for name, count in top],
        "tenant_filter": tenant_id,
    }


@router.get("/usage/quota/{tenant_id}")
async def check_tenant_quota(
    tenant_id: str,
    _api_key: str = Depends(verify_admin_api_key),
    tenants: TenantService = Depends(get_tenant_service),
    tracker: UsageTracker = Depends(get_usage_tracker),
):
    """Check quota status for a tenant."""
    tenant = tenants.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")

    within_quota, remaining = tracker.check_quota(tenant_id, tenant.monthly_quota)
    current_usage = tracker.get_monthly_usage(tenant_id)

    return {
        "tenant_id": tenant_id,
        "monthly_quota": tenant.monthly_quota,
        "current_usage": current_usage,
        "remaining": remaining if remaining >= 0 else "unlimited",
        "within_quota": within_quota,
        "quota_percentage": (current_usage / tenant.monthly_quota * 100)
        if tenant.monthly_quota > 0
        else 0,
    }


# =============================================================================
# Admin User Management Models
# =============================================================================


class AdminUserResponse(BaseModel):
    """Response model for a single user in admin context."""

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


class AdminUserListResponse(BaseModel):
    """Response model for paginated user list."""

    success: bool = True
    users: list[AdminUserResponse]
    total: int
    page: int
    page_size: int
    has_more: bool
    filters: dict = Field(default_factory=dict, description="Applied filters")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "users": [
                    {
                        "id": 1,
                        "email": "admin@example.com",
                        "username": "admin",
                        "role": "admin",
                        "is_active": True,
                        "is_verified": True,
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": None,
                        "last_login": "2024-01-25T10:00:00Z",
                    }
                ],
                "total": 25,
                "page": 1,
                "page_size": 20,
                "has_more": True,
                "filters": {"role": "admin", "is_active": True},
            }
        }


class AdminUserListErrorResponse(BaseModel):
    """Error response for admin user list endpoint."""

    success: bool = False
    error: str


# =============================================================================
# Database Session Dependency for Admin Endpoints
# =============================================================================


async def get_db_session() -> AsyncSession:
    """
    Get an async database session for admin endpoints.

    Yields a session and ensures proper cleanup.
    """
    session = get_async_session_factory()()
    try:
        yield session
    finally:
        await session.close()


# =============================================================================
# Admin Role Verification Helpers
# =============================================================================


async def _get_admin_user(
    current_user: TokenPayload,
    session: AsyncSession,
) -> User:
    """
    Get the authenticated admin user from the JWT token.

    Args:
        current_user: Token payload from JWT authentication
        session: Database session

    Returns:
        User model from database (must be admin)

    Raises:
        HTTPException: If user not found, is API client, or not admin
    """
    # Check if this is an API client (not a database user)
    if current_user.type == "api_key" or current_user.sub == "api_client":
        admin_logger.warning("Admin endpoint accessed with API key authentication")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin endpoints require user authentication with JWT token",
        )

    # Get user ID from token
    try:
        user_id = int(current_user.sub)
    except (ValueError, TypeError):
        admin_logger.warning(f"Invalid user ID in token: {current_user.sub}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    # Get user from database
    user_service = get_user_service(session)
    user = await user_service.get_user_by_id(user_id)

    if not user:
        admin_logger.warning(f"User not found for admin request: id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not user.is_active:
        admin_logger.warning(f"Inactive user attempted admin access: id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Check admin role
    if user.role != UserRole.ADMIN:
        admin_logger.warning(
            f"Non-admin user {user_id} attempted to access admin endpoint (role: {user.role})"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation",
        )

    return user


def _user_to_admin_response(user: User) -> AdminUserResponse:
    """
    Convert a User model to an AdminUserResponse.

    Sanitizes user data by excluding sensitive fields like password hash.

    Args:
        user: User model from database

    Returns:
        AdminUserResponse with safe user data
    """
    return AdminUserResponse(
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


# =============================================================================
# Admin User Management Endpoints
# =============================================================================


@router.get(
    "/users",
    status_code=status.HTTP_200_OK,
    response_model=AdminUserListResponse,
    summary="List all users (Admin only)",
    description="""
List all users in the system with pagination, filtering, and search capabilities.

**ADMIN ONLY**: This endpoint requires admin role.

**Features**:
- Paginated results with configurable page size (1-100)
- Filter by role (admin, researcher, viewer)
- Filter by active status (active/inactive)
- Filter by verification status (verified/unverified)
- Search by email or username (partial match)

**Query Parameters**:
- `page` - Page number (1-indexed, default: 1)
- `page_size` - Number of results per page (1-100, default: 20)
- `role` - Filter by role: admin, researcher, viewer
- `is_active` - Filter by active status: true/false
- `is_verified` - Filter by verification status: true/false
- `search` - Search term for email or username

**Authentication**: Requires JWT token with admin role

**Example Response**:
```json
{
  "success": true,
  "users": [
    {
      "id": 1,
      "email": "admin@example.com",
      "username": "admin",
      "role": "admin",
      "is_active": true,
      "is_verified": true,
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": null,
      "last_login": "2024-01-25T10:00:00Z"
    }
  ],
  "total": 25,
  "page": 1,
  "page_size": 20,
  "has_more": true,
  "filters": {"role": "admin", "is_active": true}
}
```
    """,
    responses={
        200: {
            "description": "User list retrieved successfully",
            "model": AdminUserListResponse,
        },
        401: {"description": "Not authenticated"},
        403: {
            "description": "Admin role required or API key authentication not supported",
            "model": AdminUserListErrorResponse,
        },
    },
    tags=["admin", "users"],
)
async def list_users(
    request: Request,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of results per page"),
    role: Optional[str] = Query(
        None,
        description="Filter by role: admin, researcher, viewer",
        examples=["admin", "researcher", "viewer"],
    ),
    is_active: Optional[bool] = Query(
        None,
        description="Filter by active status",
    ),
    is_verified: Optional[bool] = Query(
        None,
        description="Filter by verification status",
    ),
    search: Optional[str] = Query(
        None,
        min_length=1,
        max_length=100,
        description="Search by email or username (partial match)",
    ),
    current_user: TokenPayload = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AdminUserListResponse:
    """
    List all users with pagination, filtering, and search.

    Admin-only endpoint. Returns a paginated list of users.
    """
    # Authenticate and verify admin role
    admin_user = await _get_admin_user(current_user, session)
    client_ip = request.client.host if request.client else None

    # Parse role filter
    user_role: Optional[UserRole] = None
    if role:
        role_lower = role.lower()
        try:
            user_role = UserRole(role_lower)
        except ValueError:
            admin_logger.warning(f"Invalid role filter: {role}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {role}. Valid roles are: admin, researcher, viewer",
            )

    # Get user repository
    user_repo = get_user_repository(session)

    # Calculate offset for pagination
    offset = (page - 1) * page_size

    # Fetch users with filters
    try:
        users, total = await user_repo.list_users(
            limit=page_size,
            offset=offset,
            role=user_role,
            is_active=is_active,
            is_verified=is_verified,
            search=search,
        )
    except Exception as e:
        admin_logger.error(f"Failed to list users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users",
        )

    # Check if there are more results
    has_more = (offset + len(users)) < total

    # Convert to response models
    user_responses = [_user_to_admin_response(user) for user in users]

    # Build applied filters dict
    applied_filters: dict[str, Any] = {}
    if role:
        applied_filters["role"] = role.lower()
    if is_active is not None:
        applied_filters["is_active"] = is_active
    if is_verified is not None:
        applied_filters["is_verified"] = is_verified
    if search:
        applied_filters["search"] = search

    # Log the admin action
    audit_log(
        action=AuditAction.CONFIG_VIEW,
        user_id=str(admin_user.id),
        resource="/api/v1/admin/users",
        details={
            "action": "list_users",
            "page": page,
            "page_size": page_size,
            "filters": applied_filters,
            "total_results": total,
            "returned_results": len(users),
        },
        ip_address=client_ip,
    )

    admin_logger.info(
        f"Admin {admin_user.id} listed users: page={page}, total={total}, "
        f"filters={applied_filters}"
    )

    return AdminUserListResponse(
        success=True,
        users=user_responses,
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more,
        filters=applied_filters,
    )
