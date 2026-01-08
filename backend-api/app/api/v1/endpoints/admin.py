"""
Admin Endpoints for Feature Flag and System Management

Provides administrative controls for:
- Feature flag management
- Technique configuration
- System statistics

Part of Phase 3: Transformation implementation.
"""

import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger
from app.services.feature_flag_service import FeatureFlagService, get_feature_flags
from app.services.tenant_service import TenantService, TenantTier, get_tenant_service
from app.services.usage_tracker import UsageTracker, get_usage_tracker

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
