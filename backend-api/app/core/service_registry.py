"""
Centralized Service Registry for Dependency Management

This module provides a unified service registry pattern for managing all backend services.
It ensures consistent initialization, lifecycle management, and dependency injection across
the entire application.

Features:
- Eager registration: Register pre-instantiated services
- Lazy registration: Register factory functions that create services on demand
- Lifecycle management: Initialize and shutdown all services
- Health monitoring: Check service status and health
- AI Configuration Manager integration for provider configuration
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Optional

if TYPE_CHECKING:
    from app.core.ai_config_manager import AIConfigManager
    from app.core.config_validator import ConfigValidator
    from app.core.fallback_manager import FallbackManager

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Centralized service registry for dependency management.

    Supports both eager and lazy service registration:
    - Eager: Register pre-instantiated service objects
    - Lazy: Register factory functions that create services on first access

    Example:
        # Eager registration
        registry.register("llm_service", LLMService())

        # Lazy registration (service created on first get())
        registry.register_lazy("heavy_service", lambda: HeavyService())

        # Get service (lazy services are created here)
        service = registry.get("heavy_service")
    """

    _instance: ClassVar[Optional["ServiceRegistry"]] = None
    _services: ClassVar[dict[str, Any]] = {}
    _lazy_factories: ClassVar[dict[str, Callable[[], Any]]] = {}
    _initialized: ClassVar[bool] = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._lazy_factories = {}
        return cls._instance

    def register(self, name: str, service: Any) -> None:
        """Register a pre-instantiated service instance."""
        if name in self._services:
            logger.warning(f"Service {name} already registered, overwriting")
        if name in self._lazy_factories:
            del self._lazy_factories[name]
            logger.debug(f"Removed lazy factory for {name} (replaced with instance)")
        self._services[name] = service
        logger.info(f"Registered service: {name}")

    def register_lazy(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a lazy service factory.

        The factory function will be called on first access to create the service.
        This is useful for heavy services that should only be loaded when needed.

        Args:
            name: Service name
            factory: Callable that returns the service instance
            overwrite: If True, overwrite existing registration
        """
        if name in self._services and not overwrite:
            logger.warning(f"Service {name} already registered, skipping lazy factory")
            return
        if name in self._lazy_factories and not overwrite:
            logger.warning(f"Lazy factory {name} already registered, skipping")
            return
        self._lazy_factories[name] = factory
        logger.info(f"Registered lazy service factory: {name}")

    def get(self, name: str) -> Any:
        """
        Get a service instance.

        For lazy services, the factory is called on first access and
        the result is cached for subsequent calls.
        """
        # Check if already instantiated
        if name in self._services:
            return self._services[name]

        # Check for lazy factory
        if name in self._lazy_factories:
            logger.info(f"Lazy loading service: {name}")
            factory = self._lazy_factories[name]
            try:
                service = factory()
                self._services[name] = service
                del self._lazy_factories[name]
                logger.info(f"Lazy loaded service: {name}")
                return service
            except Exception as e:
                logger.error(f"Failed to lazy load service {name}: {e}")
                raise

        raise KeyError(f"Service {name} not registered")

    def has(self, name: str) -> bool:
        """Check if service is registered (eager or lazy)."""
        return name in self._services or name in self._lazy_factories

    def is_loaded(self, name: str) -> bool:
        """Check if a service is actually loaded (not just registered)."""
        return name in self._services

    async def initialize_all(self, *, eager_only: bool = True) -> None:
        """
        Initialize all registered services.

        Args:
            eager_only: If True (default), only initialize eagerly registered services.
                       If False, also load and initialize all lazy services.
        """
        if self._initialized:
            logger.warning("Services already initialized")
            return

        # Optionally load all lazy services first
        if not eager_only:
            lazy_names = list(self._lazy_factories.keys())
            for name in lazy_names:
                logger.info(f"Pre-loading lazy service: {name}")
                self.get(name)  # This loads and caches the service

        # Initialize all loaded services
        for name, service in self._services.items():
            if hasattr(service, "initialize"):
                logger.info(f"Initializing service: {name}")
                try:
                    await service.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize service {name}: {e}")
                    raise

        self._initialized = True
        logger.info("All services initialized successfully")

    async def shutdown_all(self) -> None:
        """Shutdown all registered services."""
        for name, service in self._services.items():
            if hasattr(service, "shutdown"):
                logger.info(f"Shutting down service: {name}")
                try:
                    await service.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down service {name}: {e}")

        self._initialized = False
        logger.info("All services shut down")

    def list_services(self) -> list[str]:
        """List all registered services (both eager and lazy)."""
        all_services = set(self._services.keys()) | set(self._lazy_factories.keys())
        return list(all_services)

    def list_loaded_services(self) -> list[str]:
        """List only loaded (instantiated) services."""
        return list(self._services.keys())

    def list_lazy_services(self) -> list[str]:
        """List services registered with lazy factories (not yet loaded)."""
        return list(self._lazy_factories.keys())

    def get_service_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered services."""
        status = {}

        # Status for loaded services
        for name, service in self._services.items():
            service_info = {
                "registered": True,
                "loaded": True,
                "type": type(service).__name__,
                "has_initialize": hasattr(service, "initialize"),
                "has_shutdown": hasattr(service, "shutdown"),
            }

            # Check if service has health check method
            if hasattr(service, "health_check"):
                try:
                    service_info["health"] = "healthy"
                except Exception:
                    service_info["health"] = "unhealthy"

            status[name] = service_info

        # Status for lazy (not yet loaded) services
        for name in self._lazy_factories:
            status[name] = {
                "registered": True,
                "loaded": False,
                "type": "lazy_factory",
                "has_initialize": "unknown",
                "has_shutdown": "unknown",
            }

        return status

    def preload(self, *service_names: str) -> None:
        """
        Preload specific lazy services.

        Useful for warming up services that will definitely be needed.

        Args:
            service_names: Names of lazy services to preload
        """
        for name in service_names:
            if name in self._lazy_factories:
                logger.info(f"Preloading service: {name}")
                self.get(name)
            elif name not in self._services:
                logger.warning(f"Service {name} not registered, cannot preload")

    def reset(self) -> None:
        """
        Reset the registry (for testing purposes).

        Clears all registered services and factories.
        """
        self._services.clear()
        self._lazy_factories.clear()
        self._initialized = False
        logger.info("Service registry reset")


# Global registry instance
service_registry = ServiceRegistry()


# =============================================================================
# AI Configuration Manager Integration
# =============================================================================


def get_ai_config_manager() -> "AIConfigManager":
    """
    Get the AI configuration manager from the service registry.

    This is a convenience function that provides dependency injection
    for the AIConfigManager. It ensures the manager is registered
    if not already present.

    Returns:
        AIConfigManager instance

    Example:
        from app.core.service_registry import get_ai_config_manager

        config_manager = get_ai_config_manager()
        provider = config_manager.get_provider("openai")
    """
    if not service_registry.has("ai_config_manager"):
        # Register lazily if not already registered
        from app.core.ai_config_manager import ai_config_manager

        service_registry.register("ai_config_manager", ai_config_manager)

    return service_registry.get("ai_config_manager")


def register_core_services() -> None:
    """
    Register core services including the AI configuration manager.

    This function should be called during application startup to ensure
    all core services are available. It registers the AIConfigManager
    as a core service with eager initialization.

    Example:
        # In application startup (e.g., main.py)
        from app.core.service_registry import register_core_services

        register_core_services()
    """
    from app.core.ai_config_manager import ai_config_manager

    # Register config manager (eager, as it's needed early)
    if not service_registry.has("ai_config_manager"):
        service_registry.register("ai_config_manager", ai_config_manager)
        logger.info("Registered AIConfigManager as core service")


async def initialize_ai_config(config_path: str | None = None) -> None:
    """
    Initialize the AI configuration manager with the providers.yaml file.

    This function loads the provider configuration from YAML and makes it
    available through the service registry. Should be called during
    application startup after register_core_services().

    Args:
        config_path: Optional path to providers.yaml file.
                    If not provided, uses default location.

    Example:
        # In application startup
        from app.core.service_registry import (
            register_core_services,
            initialize_ai_config
        )

        register_core_services()
        await initialize_ai_config()
    """
    from pathlib import Path

    config_manager = get_ai_config_manager()

    if config_path:
        path = Path(config_path)
    else:
        path = None  # Let manager use default path resolution

    try:
        await config_manager.load_config(path)
        logger.info("AI provider configuration loaded successfully")
    except FileNotFoundError:
        logger.warning(
            "AI provider configuration file not found. "
            "Using defaults. Create providers.yaml to configure providers."
        )
    except Exception as e:
        logger.error(f"Failed to load AI provider configuration: {e}")
        raise


# =============================================================================
# Config Validator Integration
# =============================================================================


def get_config_validator() -> "ConfigValidator":
    """
    Get the config validator from the service registry.

    Returns:
        ConfigValidator instance
    """
    if not service_registry.has("config_validator"):
        from app.core.config_validator import config_validator

        service_registry.register("config_validator", config_validator)

    return service_registry.get("config_validator")


# =============================================================================
# Fallback Manager Integration
# =============================================================================


def get_fallback_manager() -> "FallbackManager":
    """
    Get the fallback manager from the service registry.

    Returns:
        FallbackManager instance
    """
    if not service_registry.has("fallback_manager"):
        from app.core.fallback_manager import fallback_manager

        service_registry.register("fallback_manager", fallback_manager)

    return service_registry.get("fallback_manager")


# =============================================================================
# Extended Core Services Registration
# =============================================================================


def register_validation_services() -> None:
    """
    Register validation and fallback services.

    This function should be called during application startup after
    register_core_services() to enable validation checkpoints and
    fallback mechanisms.

    Example:
        from app.core.service_registry import (
            register_core_services,
            register_validation_services,
        )

        register_core_services()
        register_validation_services()
    """
    from app.core.config_validator import config_validator
    from app.core.fallback_manager import fallback_manager

    # Register config validator
    if not service_registry.has("config_validator"):
        service_registry.register("config_validator", config_validator)
        logger.info("Registered ConfigValidator as validation service")

    # Register fallback manager
    if not service_registry.has("fallback_manager"):
        service_registry.register("fallback_manager", fallback_manager)
        logger.info("Registered FallbackManager as fallback service")


async def run_startup_validation(
    test_connectivity: bool = False,
    fail_on_warnings: bool = False,
) -> bool:
    """
    Run startup validation checks.

    This function should be called during application startup after
    AI configuration is loaded. It validates the configuration and
    optionally tests provider connectivity.

    Args:
        test_connectivity: Whether to test provider connectivity
        fail_on_warnings: Whether to fail on validation warnings

    Returns:
        True if validation passed, False otherwise

    Example:
        from app.core.service_registry import (
            register_core_services,
            register_validation_services,
            initialize_ai_config,
            run_startup_validation,
        )

        register_core_services()
        register_validation_services()
        await initialize_ai_config()
        is_ready = await run_startup_validation(test_connectivity=True)
    """
    from app.core.startup_validator import StartupValidator

    try:
        is_ready = await StartupValidator.validate_on_startup(
            test_connectivity=test_connectivity,
            fail_on_warnings=fail_on_warnings,
        )

        if is_ready:
            logger.info("Startup validation passed - system ready")
        else:
            logger.warning("Startup validation found issues")

        return is_ready

    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        return False


async def initialize_all_services(
    config_path: str | None = None,
    test_connectivity: bool = False,
    fail_on_warnings: bool = False,
) -> bool:
    """
    Initialize all core services including validation.

    This is a convenience function that performs the complete
    initialization sequence:
    1. Register core services
    2. Register validation services
    3. Initialize AI configuration
    4. Run startup validation

    Args:
        config_path: Optional path to providers.yaml
        test_connectivity: Whether to test provider connectivity
        fail_on_warnings: Whether to fail on validation warnings

    Returns:
        True if all initialization succeeded, False otherwise

    Example:
        from app.core.service_registry import initialize_all_services

        is_ready = await initialize_all_services(
            test_connectivity=True,
            fail_on_warnings=False,
        )

        if not is_ready:
            logger.warning("System starting with degraded state")
    """
    try:
        # Step 1: Register core services
        register_core_services()

        # Step 2: Register validation services
        register_validation_services()

        # Step 3: Initialize AI configuration
        await initialize_ai_config(config_path)

        # Step 4: Run startup validation
        is_ready = await run_startup_validation(
            test_connectivity=test_connectivity,
            fail_on_warnings=fail_on_warnings,
        )

        # Step 5: Initialize all registered services
        await service_registry.initialize_all(eager_only=True)

        logger.info("All services initialized successfully")
        return is_ready

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False
