"""
Tenant Management Service

Provides multi-tenant capabilities for enterprise deployments:
- Tenant isolation for techniques
- Tenant-specific configuration
- Rate limiting per tenant
- Usage quotas

Part of Phase 3: Enterprise Readiness implementation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Optional

logger = logging.getLogger(__name__)


class TenantTier(str, Enum):
    """Subscription tiers with different capabilities."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantConfig:
    """
    Configuration for a tenant.

    Attributes:
        tenant_id: Unique identifier
        name: Display name
        tier: Subscription tier
        api_key: Tenant's API key
        allowed_techniques: Techniques this tenant can access
        blocked_techniques: Techniques blocked for this tenant
        rate_limit_per_minute: API rate limit
        monthly_quota: Monthly request quota
        custom_settings: Tenant-specific settings
        created_at: Creation timestamp
        is_active: Whether tenant is active
    """

    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.FREE
    api_key: str | None = None
    allowed_techniques: list[str] = field(default_factory=list)
    blocked_techniques: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60
    monthly_quota: int = 1000
    custom_settings: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


# Default tier configurations
TIER_DEFAULTS = {
    TenantTier.FREE: {
        "rate_limit_per_minute": 10,
        "monthly_quota": 100,
        "allowed_risk_levels": ["low"],
        "max_potency": 5,
    },
    TenantTier.STARTER: {
        "rate_limit_per_minute": 30,
        "monthly_quota": 1000,
        "allowed_risk_levels": ["low", "medium"],
        "max_potency": 7,
    },
    TenantTier.PROFESSIONAL: {
        "rate_limit_per_minute": 100,
        "monthly_quota": 10000,
        "allowed_risk_levels": ["low", "medium", "high"],
        "max_potency": 10,
    },
    TenantTier.ENTERPRISE: {
        "rate_limit_per_minute": 1000,
        "monthly_quota": -1,  # Unlimited
        "allowed_risk_levels": ["low", "medium", "high", "critical"],
        "max_potency": 10,
    },
}


class TenantService:
    """
    Manages tenant configurations and access control.

    Provides methods for tenant CRUD operations and
    technique access validation.
    """

    _instance: Optional["TenantService"] = None
    _tenants: ClassVar[dict[str, TenantConfig]] = {}
    _api_key_index: ClassVar[dict[str, str]] = {}  # api_key -> tenant_id

    def __new__(cls) -> "TenantService":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tenants = {}
            cls._api_key_index = {}
            cls._instance._create_default_tenant()
        return cls._instance

    def _create_default_tenant(self) -> None:
        """Create a default tenant for non-multi-tenant mode."""
        default = TenantConfig(
            tenant_id="default", name="Default Tenant", tier=TenantTier.ENTERPRISE, is_active=True
        )
        self._tenants["default"] = default
        logger.info("Created default tenant")

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        api_key: str | None = None,
        **kwargs,
    ) -> TenantConfig:
        """
        Create a new tenant.

        Args:
            tenant_id: Unique identifier
            name: Display name
            tier: Subscription tier
            api_key: Optional API key
            **kwargs: Additional configuration

        Returns:
            Created TenantConfig
        """
        if tenant_id in self._tenants:
            raise ValueError(f"Tenant already exists: {tenant_id}")

        # Apply tier defaults
        tier_config = TIER_DEFAULTS.get(tier, TIER_DEFAULTS[TenantTier.FREE])

        tenant = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            api_key=api_key,
            rate_limit_per_minute=kwargs.get(
                "rate_limit_per_minute", tier_config["rate_limit_per_minute"]
            ),
            monthly_quota=kwargs.get("monthly_quota", tier_config["monthly_quota"]),
            custom_settings={
                "allowed_risk_levels": tier_config["allowed_risk_levels"],
                "max_potency": tier_config["max_potency"],
                **kwargs.get("custom_settings", {}),
            },
        )

        self._tenants[tenant_id] = tenant
        if api_key:
            self._api_key_index[api_key] = tenant_id

        logger.info(f"Created tenant: {tenant_id} (tier: {tier})")
        return tenant

    def get_tenant(self, tenant_id: str) -> TenantConfig | None:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    def get_tenant_by_api_key(self, api_key: str) -> TenantConfig | None:
        """Get tenant by API key."""
        tenant_id = self._api_key_index.get(api_key)
        return self._tenants.get(tenant_id) if tenant_id else None

    def update_tenant(self, tenant_id: str, **updates) -> TenantConfig | None:
        """Update tenant configuration."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)

        logger.info(f"Updated tenant: {tenant_id}")
        return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant."""
        if tenant_id == "default":
            logger.warning("Cannot delete default tenant")
            return False

        tenant = self._tenants.pop(tenant_id, None)
        if tenant and tenant.api_key:
            self._api_key_index.pop(tenant.api_key, None)

        logger.info(f"Deleted tenant: {tenant_id}")
        return tenant is not None

    def list_tenants(self) -> list[TenantConfig]:
        """List all tenants."""
        return list(self._tenants.values())

    def can_access_technique(
        self, tenant_id: str, technique_name: str, technique_risk_level: str = "medium"
    ) -> tuple[bool, str]:
        """
        Check if a tenant can access a specific technique.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        tenant = self._tenants.get(tenant_id)

        if not tenant:
            return False, f"Tenant not found: {tenant_id}"

        if not tenant.is_active:
            return False, f"Tenant is inactive: {tenant_id}"

        # Check explicit blocks
        if technique_name in tenant.blocked_techniques:
            return False, f"Technique blocked for tenant: {technique_name}"

        # Check explicit allows (if specified, only these are allowed)
        if tenant.allowed_techniques and technique_name not in tenant.allowed_techniques:
            return False, f"Technique not in allowed list: {technique_name}"

        # Check risk level against tier
        allowed_risk_levels = tenant.custom_settings.get("allowed_risk_levels", ["low"])
        if technique_risk_level not in allowed_risk_levels:
            return False, f"Risk level '{technique_risk_level}' not allowed for tier {tenant.tier}"

        return True, "Access granted"

    def get_max_potency(self, tenant_id: str) -> int:
        """Get maximum potency level for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return 5  # Default low
        return tenant.custom_settings.get("max_potency", 5)

    def get_rate_limit(self, tenant_id: str) -> int:
        """Get rate limit per minute for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return 10  # Default low
        return tenant.rate_limit_per_minute

    def get_statistics(self) -> dict[str, Any]:
        """Get tenant statistics."""
        tenants_by_tier = {}
        for tenant in self._tenants.values():
            tier = tenant.tier.value
            tenants_by_tier[tier] = tenants_by_tier.get(tier, 0) + 1

        return {
            "total_tenants": len(self._tenants),
            "active_tenants": sum(1 for t in self._tenants.values() if t.is_active),
            "tenants_by_tier": tenants_by_tier,
        }


# Global instance
_tenant_service = TenantService()


def get_tenant_service() -> TenantService:
    """Get the global tenant service."""
    return _tenant_service
