"""
Provider Registry with AIConfigManager Integration

Provides centralized provider registration and lookup with:
- Config-driven provider list from AIConfigManager
- Hot-reload support via config change subscription
- Health status tracking with config-driven circuit breaker settings
- Provider capability lookup from configuration
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _get_config_manager():
    """
    Get AIConfigManager instance with graceful fallback.

    Returns None if config manager is not available.
    """
    try:
        from app.core.service_registry import get_ai_config_manager
        config_manager = get_ai_config_manager()
        if config_manager.is_loaded():
            return config_manager
    except Exception as e:
        logger.debug(f"Config manager not available: {e}")
    return None


@dataclass
class ProviderInfo:
    """Provider information with extended metadata."""
    name: str
    direct_url: str
    models: list[str]
    enabled: bool = True
    priority: int = 50
    capabilities: dict[str, bool] = field(default_factory=dict)
    circuit_breaker_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealthStatus:
    """Health status for a provider."""
    provider_name: str
    is_healthy: bool = True
    last_check: datetime | None = None
    failure_count: int = 0
    circuit_state: str = "closed"  # closed, open, half_open
    last_error: str | None = None


class ProviderRegistry:
    """
    Provider registry with AIConfigManager integration.

    Features:
    - Config-driven provider list
    - Hot-reload via config change subscription
    - Health status tracking
    - Provider capability lookup
    """
    _instance: ClassVar[Optional["ProviderRegistry"]] = None
    _providers: ClassVar[dict[str, ProviderInfo]] = {}
    _health_status: ClassVar[dict[str, ProviderHealthStatus]] = {}
    _config_subscribed: ClassVar[bool] = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the registry with providers from config or settings."""
        self.reload_configuration()
        self._subscribe_to_config_changes()

    def _subscribe_to_config_changes(self):
        """Subscribe to AIConfigManager for hot-reload support."""
        if ProviderRegistry._config_subscribed:
            return

        config_manager = _get_config_manager()
        if config_manager:
            try:
                config_manager.on_change(self._on_config_change)
                ProviderRegistry._config_subscribed = True
                logger.info(
                    "ProviderRegistry subscribed to config changes"
                )
            except Exception as e:
                logger.warning(f"Failed to subscribe to config changes: {e}")

    def _on_config_change(self, event):
        """Handle config change events for hot-reload."""
        logger.info(
            f"Config change detected: {event.event_type}, reloading providers"
        )
        self.reload_configuration()

    def register_provider(
        self,
        name: str,
        direct_url: str,
        models: list[str],
        enabled: bool = True,
        priority: int = 50,
        capabilities: dict[str, bool] | None = None,
        circuit_breaker_config: dict[str, Any] | None = None,
    ):
        """Register or update a provider configuration."""
        self._providers[name.lower()] = ProviderInfo(
            name=name,
            direct_url=direct_url,
            models=models,
            enabled=enabled,
            priority=priority,
            capabilities=capabilities or {},
            circuit_breaker_config=circuit_breaker_config or {},
        )

        # Initialize health status if not exists
        if name.lower() not in self._health_status:
            self._health_status[name.lower()] = ProviderHealthStatus(
                provider_name=name
            )

    def get_endpoint(self, provider: str) -> str:
        """Get the direct endpoint URL for a specific provider."""
        provider_info = self._providers.get(provider.lower())
        if provider_info:
            return provider_info.direct_url

        # Try config manager
        config_manager = _get_config_manager()
        if config_manager:
            try:
                provider_config = config_manager.get_provider(provider)
                if provider_config:
                    return provider_config.api.base_url
            except Exception:
                pass

        # Fallback to settings
        settings = get_settings()
        return settings.get_provider_endpoint(provider)

    def list_providers(self) -> dict[str, ProviderInfo]:
        """List all registered providers."""
        return self._providers.copy()

    def get_available_providers(self) -> list[str]:
        """
        Get list of available (enabled) provider names.

        Uses config if available for accurate enabled status.
        """
        config_manager = _get_config_manager()
        if config_manager:
            try:
                enabled = config_manager.get_enabled_providers()
                return [p.provider_id for p in enabled]
            except Exception as e:
                logger.debug(f"Failed to get providers from config: {e}")

        # Fallback to registered providers that are enabled
        return [
            name for name, info in self._providers.items()
            if info.enabled
        ]

    def is_provider_enabled(self, provider: str) -> bool:
        """
        Check if a provider is enabled.

        Uses AIConfigManager if available.
        """
        config_manager = _get_config_manager()
        if config_manager:
            try:
                provider_config = config_manager.get_provider(provider)
                if provider_config:
                    return provider_config.enabled
            except Exception:
                pass

        # Fallback to local registry
        provider_info = self._providers.get(provider.lower())
        return provider_info.enabled if provider_info else False

    def get_provider_capabilities(
        self, provider: str
    ) -> dict[str, bool] | None:
        """
        Get capabilities for a provider from config.

        Returns dict with capability flags or None if not found.
        """
        config_manager = _get_config_manager()
        if config_manager:
            try:
                provider_config = config_manager.get_provider(provider)
                if provider_config:
                    caps = provider_config.capabilities
                    return {
                        "supports_streaming": caps.supports_streaming,
                        "supports_vision": caps.supports_vision,
                        "supports_function_calling": (
                            caps.supports_function_calling
                        ),
                        "supports_json_mode": caps.supports_json_mode,
                        "supports_system_prompt": caps.supports_system_prompt,
                        "supports_token_counting": (
                            caps.supports_token_counting
                        ),
                        "supports_embeddings": caps.supports_embeddings,
                    }
            except Exception as e:
                logger.debug(f"Failed to get capabilities: {e}")

        # Fallback to local registry
        provider_info = self._providers.get(provider.lower())
        return provider_info.capabilities if provider_info else None

    def get_circuit_breaker_config(
        self, provider: str
    ) -> dict[str, Any]:
        """
        Get circuit breaker configuration for a provider.

        Returns config dict with enabled, failure_threshold, recovery_timeout.
        """
        config_manager = _get_config_manager()
        if config_manager:
            try:
                provider_config = config_manager.get_provider(provider)
                if provider_config:
                    cb = provider_config.circuit_breaker
                    return {
                        "enabled": cb.enabled,
                        "failure_threshold": cb.failure_threshold,
                        "recovery_timeout_seconds": (
                            cb.recovery_timeout_seconds
                        ),
                        "half_open_max_requests": cb.half_open_max_requests,
                    }
            except Exception:
                pass

        # Fallback to local registry or defaults
        provider_info = self._providers.get(provider.lower())
        if provider_info and provider_info.circuit_breaker_config:
            return provider_info.circuit_breaker_config

        # Default circuit breaker config
        return {
            "enabled": True,
            "failure_threshold": 5,
            "recovery_timeout_seconds": 60,
            "half_open_max_requests": 3,
        }

    def update_health_status(
        self,
        provider: str,
        is_healthy: bool,
        error: str | None = None,
    ):
        """Update health status for a provider."""
        name = provider.lower()
        if name not in self._health_status:
            self._health_status[name] = ProviderHealthStatus(
                provider_name=provider
            )

        status = self._health_status[name]
        status.last_check = datetime.utcnow()
        status.is_healthy = is_healthy

        if is_healthy:
            status.failure_count = 0
            status.circuit_state = "closed"
            status.last_error = None
        else:
            status.failure_count += 1
            status.last_error = error

            # Check circuit breaker threshold
            cb_config = self.get_circuit_breaker_config(provider)
            if status.failure_count >= cb_config.get("failure_threshold", 5):
                status.circuit_state = "open"
                logger.warning(
                    f"Circuit breaker opened for provider: {provider}"
                )

    def get_health_status(
        self, provider: str
    ) -> ProviderHealthStatus | None:
        """Get health status for a provider."""
        return self._health_status.get(provider.lower())

    def is_circuit_open(self, provider: str) -> bool:
        """Check if circuit breaker is open for a provider."""
        status = self._health_status.get(provider.lower())
        return status.circuit_state == "open" if status else False

    def validate_provider_endpoint(self, provider: str) -> bool:
        """Validate if a provider endpoint is correctly configured."""
        endpoint = self.get_endpoint(provider)
        settings = get_settings()
        return settings.validate_endpoint_url(endpoint)

    def get_provider_by_model(self, model: str) -> str | None:
        """Find which provider hosts a specific model."""
        # Try config manager first
        config_manager = _get_config_manager()
        if config_manager:
            try:
                config = config_manager.get_config()
                for provider_id, provider in config.providers.items():
                    if model in provider.models:
                        return provider_id
            except Exception:
                pass

        # Fallback to local registry
        for name, info in self._providers.items():
            if model in info.models:
                return name
        return None

    def reload_configuration(self):
        """Reload configuration from AIConfigManager or settings."""
        config_manager = _get_config_manager()

        if config_manager:
            try:
                config = config_manager.get_config()
                logger.info(
                    f"Reloading providers from config: "
                    f"{len(config.providers)} providers"
                )

                for provider_id, provider_config in config.providers.items():
                    self.register_provider(
                        name=provider_id,
                        direct_url=provider_config.api.base_url,
                        models=list(provider_config.models.keys()),
                        enabled=provider_config.enabled,
                        priority=provider_config.priority,
                        capabilities={
                            "supports_streaming": (
                                provider_config.capabilities.supports_streaming
                            ),
                            "supports_vision": (
                                provider_config.capabilities.supports_vision
                            ),
                            "supports_function_calling": (
                                provider_config
                                .capabilities.supports_function_calling
                            ),
                        },
                        circuit_breaker_config={
                            "enabled": (
                                provider_config.circuit_breaker.enabled
                            ),
                            "failure_threshold": (
                                provider_config
                                .circuit_breaker.failure_threshold
                            ),
                            "recovery_timeout_seconds": (
                                provider_config
                                .circuit_breaker.recovery_timeout_seconds
                            ),
                        },
                    )
                return
            except Exception as e:
                logger.warning(
                    f"Failed to load from config manager, "
                    f"falling back to settings: {e}"
                )

        # Fallback to settings
        settings = get_settings()
        endpoints = settings.get_all_provider_endpoints()
        models_map = settings.get_provider_models()

        for provider, url in endpoints.items():
            self.register_provider(
                name=provider,
                direct_url=url,
                models=models_map.get(provider, []),
            )

        logger.info(
            f"Loaded {len(self._providers)} providers from settings"
        )


# Global instance
provider_registry = ProviderRegistry()
