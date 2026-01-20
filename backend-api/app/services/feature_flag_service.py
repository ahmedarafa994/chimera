"""Feature Flag Service for Technique Management.

Provides runtime control over technique availability based on:
- Enabled/disabled status
- Risk levels
- Approval requirements
- Tenant restrictions (future)

Part of Phase 3: Transformation implementation.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml

logger = logging.getLogger(__name__)


class FeatureFlagService:
    """Manages feature flags for jailbreak techniques.

    Loads configuration from techniques.yaml and provides runtime
    checks for technique availability and risk validation.
    """

    _instance: ClassVar[Optional["FeatureFlagService"]] = None
    _config: ClassVar[dict[str, Any]] = {}
    _config_path: ClassVar[Path] = Path(__file__).parent.parent / "config" / "techniques.yaml"

    def __new__(cls) -> "FeatureFlagService":
        """Singleton pattern for consistent flag state."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load technique configuration from YAML file."""
        try:
            if self._config_path.exists():
                with open(self._config_path) as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Loaded {len(self._config)} technique configurations")
            else:
                logger.warning(f"Config file not found: {self._config_path}")
                self._config = {}
        except Exception as e:
            logger.exception(f"Failed to load technique config: {e}")
            self._config = {}

    def reload_config(self) -> bool:
        """Reload configuration from disk. Returns True if successful."""
        try:
            self._load_config()
            return True
        except Exception as e:
            logger.exception(f"Failed to reload config: {e}")
            return False

    def is_technique_enabled(self, technique_name: str) -> bool:
        """Check if a technique is enabled."""
        technique = self._config.get(technique_name, {})
        return technique.get("enabled", False)

    def get_risk_level(self, technique_name: str) -> str:
        """Get the risk level of a technique."""
        technique = self._config.get(technique_name, {})
        return technique.get("risk_level", "unknown")

    def requires_approval(self, technique_name: str) -> bool:
        """Check if a technique requires admin approval."""
        technique = self._config.get(technique_name, {})
        return technique.get("requires_approval", False)

    def get_technique_config(self, technique_name: str) -> dict[str, Any] | None:
        """Get full configuration for a technique."""
        return self._config.get(technique_name)

    def list_enabled_techniques(self) -> list[str]:
        """Get list of all enabled technique names."""
        return [
            name
            for name, config in self._config.items()
            if isinstance(config, dict)
            and config.get("enabled", False)
            and name != "plugin_settings"
        ]

    def list_techniques_by_risk_level(self, risk_level: str) -> list[str]:
        """Get techniques filtered by risk level."""
        return [
            name
            for name, config in self._config.items()
            if isinstance(config, dict)
            and config.get("risk_level") == risk_level
            and config.get("enabled", False)
        ]

    def validate_technique_access(
        self,
        technique_name: str,
        user_has_approval: bool = False,
        tenant_id: str | None = None,
    ) -> tuple[bool, str]:
        """Validate if a user can access a technique.

        Returns:
            Tuple of (allowed: bool, reason: str)

        """
        technique = self._config.get(technique_name)

        if technique is None:
            return False, f"Technique '{technique_name}' not found"

        if not technique.get("enabled", False):
            return False, f"Technique '{technique_name}' is disabled"

        if technique.get("requires_approval", False) and not user_has_approval:
            return False, f"Technique '{technique_name}' requires admin approval"

        # Future: tenant restrictions
        restrictions = technique.get("tenant_restrictions", [])
        if restrictions and tenant_id and tenant_id not in restrictions:
            return False, f"Tenant not authorized for technique '{technique_name}'"

        return True, "Access granted"

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about technique configuration."""
        techniques = [
            (name, config)
            for name, config in self._config.items()
            if isinstance(config, dict) and name != "plugin_settings"
        ]

        enabled_count = sum(1 for _, c in techniques if c.get("enabled", False))

        risk_distribution = {}
        for _, config in techniques:
            risk = config.get("risk_level", "unknown")
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1

        approval_required = sum(1 for _, c in techniques if c.get("requires_approval", False))

        return {
            "total_techniques": len(techniques),
            "enabled_count": enabled_count,
            "disabled_count": len(techniques) - enabled_count,
            "risk_distribution": risk_distribution,
            "approval_required_count": approval_required,
            "plugin_enabled": self._config.get("plugin_settings", {}).get("enabled", False),
        }

    def set_technique_enabled(self, technique_name: str, enabled: bool) -> bool:
        """Runtime toggle for technique enabled status.
        Note: This does NOT persist to disk - use for temporary overrides.
        """
        if technique_name in self._config:
            self._config[technique_name]["enabled"] = enabled
            logger.info(f"Technique '{technique_name}' {'enabled' if enabled else 'disabled'}")
            return True
        return False


# Global instance
feature_flags = FeatureFlagService()


def get_feature_flags() -> FeatureFlagService:
    """Dependency injection helper for FastAPI."""
    return feature_flags
