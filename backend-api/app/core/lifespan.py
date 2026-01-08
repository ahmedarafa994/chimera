"""
Application Lifespan Management
Handles startup and shutdown events for the FastAPI application.

PERF-001 FIX: Added worker pool management for CPU-bound operations.
Story 1.1: Added hot-reload integration for provider configuration.
Story 1.3: Added proxy mode integration with health monitoring.
Story 1.4: Added provider health monitoring service.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from app.core.service_registry import service_registry

if TYPE_CHECKING:
    from app.core.config_manager import EnhancedConfigManager
    from app.infrastructure.proxy.proxy_health import ProxyHealthMonitor
    from app.services.integration_health_service import IntegrationHealthService

logger = logging.getLogger("chimera.lifespan")

# PERF-001 FIX: Global worker pools for CPU-bound operations
_worker_pools: dict[str, ThreadPoolExecutor] = {}

# Story 1.1: Config manager instance for hot-reload
_config_manager: EnhancedConfigManager | None = None

# Story 1.3: Proxy health monitor instance
_proxy_health_monitor: ProxyHealthMonitor | None = None

# Story 1.4: Integration health service instance
_integration_health_service: IntegrationHealthService | None = None


def get_worker_pool(name: str, max_workers: int = 2) -> ThreadPoolExecutor:
    """Get or create a named worker pool for CPU-bound operations."""
    global _worker_pools
    if name not in _worker_pools:
        _worker_pools[name] = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{name}_worker_",
        )
        logger.info(f"Created worker pool '{name}' with {max_workers} workers")
    return _worker_pools[name]


async def shutdown_worker_pools():
    """Shutdown all worker pools gracefully."""
    global _worker_pools
    for name, pool in _worker_pools.items():
        logger.info(f"Shutting down worker pool '{name}'...")
        pool.shutdown(wait=True)
    _worker_pools.clear()
    logger.info("All worker pools shut down")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting up Chimera Backend...")

    # Task 3.1: Validate database configuration for production
    from app.core.config import settings

    if settings.ENVIRONMENT == "production":
        db_url = getattr(settings, "DATABASE_URL", "") or ""
        if "sqlite" in db_url.lower():
            logger.error(
                "CRITICAL: SQLite detected in production! "
                "Use PostgreSQL for production deployments. "
                "Set DATABASE_URL to a PostgreSQL connection string."
            )

    # 1. Register Services
    from app.services.llm_service import llm_service
    from app.services.metamorph_service import metamorph_service
    from app.services.model_router_service import model_router_service
    from app.services.transformation_service import transformation_engine

    logger.info("Registering services in Service Registry...")
    service_registry.register("llm_service", llm_service)
    service_registry.register("transformation_engine", transformation_engine)
    service_registry.register("metamorph_service", metamorph_service)
    service_registry.register("model_router_service", model_router_service)

    # 2. Configure LLM Providers based on available API keys
    try:
        from app.core.provider_factory import ProviderFactory

        logger.info("Configuring AI providers based on available API keys...")

        # Get the configured default provider from settings (AI_PROVIDER env var)
        configured_default = getattr(settings, "AI_PROVIDER", "deepseek").lower()
        # Normalize provider names (google/gemini are aliases)
        if configured_default == "google":
            configured_default = "gemini"

        logger.info(f"Configured default AI provider: {configured_default}")

        # Define providers to register (order determines fallback priority)
        # The configured_default from AI_PROVIDER setting determines which is default
        # Story 1.1: Added Qwen and Cursor support
        # Story 1.2: Added BigModel and Routeway
        provider_configs = [
            ("deepseek", "DEEPSEEK_API_KEY"),
            ("gemini", "GOOGLE_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("qwen", "QWEN_API_KEY"),
            ("cursor", "CURSOR_API_KEY"),
            ("bigmodel", "BIGMODEL_API_KEY"),
            ("routeway", "ROUTEWAY_API_KEY"),
        ]

        registered_count = 0
        default_set = False

        for provider_name, api_key_env in provider_configs:
            api_key = getattr(settings, api_key_env, None)

            if api_key:
                try:
                    provider = ProviderFactory.create_provider(provider_name)
                    # Set as default if it matches the configured default and default not yet set
                    is_default = (provider_name == configured_default) and not default_set
                    llm_service.register_provider(provider_name, provider, is_default=is_default)
                    if is_default:
                        default_set = True
                        logger.info(
                            f"[OK] {provider_name} provider registered (DEFAULT - from AI_PROVIDER setting)"
                        )
                    else:
                        logger.info(f"[OK] {provider_name} provider registered")
                    registered_count += 1
                except Exception as e:
                    logger.warning(f"[FAIL] Failed to initialize {provider_name}: {e}")
            else:
                logger.debug(f"[SKIP] {provider_name} skipped (no API key)")

        # If configured default wasn't available, set first registered provider as default
        if not default_set and registered_count > 0:
            # Get first registered provider and set as default
            first_provider = llm_service.get_default_provider_name()
            if first_provider:
                logger.warning(
                    f"Configured default '{configured_default}' not available. "
                    f"Using '{first_provider}' as fallback default."
                )

        if registered_count == 0:
            logger.critical("No LLM providers could be registered! Check API keys.")
            raise RuntimeError("Cannot start: No LLM providers available")

        logger.info(f"Successfully registered {registered_count} LLM provider(s)")

        # Story 1.1: Set up hot-reload callback for provider re-registration
        if settings.ENABLE_CONFIG_HOT_RELOAD:
            _setup_config_hot_reload(llm_service, ProviderFactory)

    except Exception as e:
        logger.error(f"Failed to register LLM providers: {e}")
        raise

    # Story 1.3: Initialize proxy mode if enabled
    if settings.PROXY_MODE_ENABLED:
        await _setup_proxy_mode(llm_service, ProviderFactory)

    # Story 1.4: Start integration health monitoring service
    await _setup_integration_health_monitoring()

    # 3. Initialize All Services
    logger.info("Initializing all registered services...")
    await service_registry.initialize_all()

    logger.info("Application startup complete")
    yield  # Application runs here

    # 4. Shutdown All Services
    logger.info("Shutting down...")
    await service_registry.shutdown_all()

    # Story 1.4: Shutdown integration health monitoring service
    await _shutdown_integration_health_monitoring()

    # Story 1.3: Shutdown proxy health monitor
    await _shutdown_proxy_mode()

    # PERF-001 FIX: Shutdown worker pools
    await shutdown_worker_pools()

    # PERF-001 FIX: Close Redis L2 cache connection if enabled
    try:
        if settings.CACHE_ENABLE_L2:
            from app.core.redis_cache import get_multi_level_cache

            cache = get_multi_level_cache()
            await cache.close()
            logger.info("Redis L2 cache connection closed")
    except Exception as e:
        logger.warning(f"Failed to close Redis L2 cache: {e}")


def _setup_config_hot_reload(llm_service, ProviderFactory):
    """
    Story 1.1: Set up configuration hot-reload with provider re-registration.

    When configuration is reloaded via the EnhancedConfigManager, this callback
    will re-register providers with updated API keys.
    """
    global _config_manager

    try:
        from app.core.config import settings
        from app.core.config_manager import EnhancedConfigManager

        _config_manager = EnhancedConfigManager(settings)

        def on_config_reload(changes: dict):
            """Callback invoked when configuration is reloaded."""
            logger.info(f"Configuration reload detected: {changes}")

            # Re-register providers if any were added or modified
            added = changes.get("added", [])
            modified = changes.get("modified", [])

            if added or modified:
                logger.info("Re-registering affected providers...")
                _reregister_providers(
                    llm_service,
                    ProviderFactory,
                    added + modified
                )

        _config_manager.register_reload_callback(on_config_reload)
        logger.info("Configuration hot-reload callback registered")

    except ImportError as e:
        logger.warning(f"Could not set up config hot-reload: {e}")
    except Exception as e:
        logger.error(f"Error setting up config hot-reload: {e}")


def _reregister_providers(llm_service, ProviderFactory, provider_names: list):
    """
    Re-register specific providers after configuration reload.

    Story 1.1 Requirement: Provider configuration should be hot-reloadable
    without application restart.
    """
    from app.core.config import settings

    provider_key_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "qwen": "QWEN_API_KEY",
        "cursor": "CURSOR_API_KEY",
        "bigmodel": "BIGMODEL_API_KEY",
        "routeway": "ROUTEWAY_API_KEY",
    }

    for provider_name in provider_names:
        normalized_name = provider_name.lower()
        if normalized_name == "google":
            normalized_name = "gemini"

        api_key_env = provider_key_map.get(normalized_name)
        if not api_key_env:
            logger.warning(f"Unknown provider for re-registration: {provider_name}")
            continue

        api_key = getattr(settings, api_key_env, None)
        if api_key:
            try:
                provider = ProviderFactory.create_provider(normalized_name)
                # Re-register (will update existing registration)
                llm_service.register_provider(normalized_name, provider, is_default=False)
                logger.info(f"[HOT-RELOAD] Re-registered provider: {normalized_name}")
            except Exception as e:
                logger.error(f"[HOT-RELOAD] Failed to re-register {normalized_name}: {e}")
        else:
            logger.info(f"[HOT-RELOAD] Provider {normalized_name} has no API key, skipping")


async def _setup_proxy_mode(llm_service, ProviderFactory):
    """
    Story 1.3: Set up proxy mode with health monitoring.

    When proxy mode is enabled (PROXY_MODE_ENABLED=true), this function:
    1. Initializes the proxy client
    2. Starts the proxy health monitor
    3. Wraps registered providers with proxy adapters (if fallback enabled)
    """
    global _proxy_health_monitor

    try:
        from app.core.config import settings
        from app.infrastructure.proxy.proxy_client import get_proxy_client
        from app.infrastructure.proxy.proxy_health import start_health_monitoring

        logger.info("Initializing proxy mode...")
        logger.info(f"Proxy endpoint: {settings.PROXY_MODE_ENDPOINT}")
        logger.info(f"Proxy timeout: {settings.PROXY_MODE_TIMEOUT}s")
        logger.info(f"Fallback to direct: {settings.PROXY_MODE_FALLBACK_TO_DIRECT}")

        # Initialize proxy client
        proxy_client = get_proxy_client()

        # Perform initial health check
        health_status = await proxy_client.check_health()
        if health_status.is_healthy:
            logger.info("Proxy server is healthy and reachable")
        else:
            logger.warning(
                f"Proxy server initial health check failed: {health_status.error}"
            )
            if not settings.PROXY_MODE_FALLBACK_TO_DIRECT:
                raise RuntimeError(
                    "Proxy mode enabled but proxy server is unavailable "
                    "and fallback is disabled"
                )

        # Start health monitoring
        await start_health_monitoring()
        logger.info("Proxy health monitoring started")

        # Get the health monitor instance
        from app.infrastructure.proxy.proxy_health import get_health_monitor
        _proxy_health_monitor = get_health_monitor()

        # Register proxy service
        service_registry.register("proxy_client", proxy_client)

        logger.info("Proxy mode initialized successfully")

    except ImportError as e:
        logger.error(f"Failed to import proxy modules: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize proxy mode: {e}")
        if not settings.PROXY_MODE_FALLBACK_TO_DIRECT:
            raise


async def _shutdown_proxy_mode():
    """
    Story 1.3: Shutdown proxy mode resources.
    """
    global _proxy_health_monitor

    from app.core.config import settings

    if not settings.PROXY_MODE_ENABLED:
        return

    try:
        from app.infrastructure.proxy.proxy_client import close_proxy_client
        from app.infrastructure.proxy.proxy_health import stop_health_monitoring

        # Stop health monitoring
        await stop_health_monitoring()
        _proxy_health_monitor = None
        logger.info("Proxy health monitoring stopped")

        # Close proxy client
        await close_proxy_client()
        logger.info("Proxy client closed")

    except Exception as e:
        logger.warning(f"Error during proxy shutdown: {e}")


async def trigger_config_reload() -> dict:
    """
    Trigger a configuration reload programmatically.

    Story 1.1: This can be called from an API endpoint to reload
    configuration without restarting the application.

    Returns:
        Dictionary with reload status and changes
    """
    global _config_manager

    if _config_manager is None:
        return {
            "status": "error",
            "message": "Config manager not initialized or hot-reload disabled"
        }

    try:
        result = _config_manager.reload_config()
        return result
    except Exception as e:
        logger.error(f"Error during config reload: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


async def _setup_integration_health_monitoring():
    """
    Story 1.4: Set up integration health monitoring service.

    This function:
    1. Gets the global IntegrationHealthService instance
    2. Starts the health monitoring background task
    3. Registers health change callbacks for circuit breaker integration
    """
    global _integration_health_service

    try:
        from app.core.config import settings
        from app.services.integration_health_service import get_health_service

        # Check if health monitoring is enabled (default: true)
        if not getattr(settings, "HEALTH_MONITORING_ENABLED", True):
            logger.info("Provider health monitoring is disabled")
            return

        logger.info("Initializing integration health monitoring service...")

        # Get or create the health service
        _integration_health_service = get_health_service()

        # Register health change callback for circuit breaker integration
        def on_health_change(
            provider_name: str, old_status, new_status
        ):
            """Callback when provider health status changes."""
            from app.services.integration_health_service import HealthStatus

            # If provider becomes unhealthy, consider triggering circuit breaker
            if new_status == HealthStatus.UNHEALTHY:
                logger.warning(
                    f"Provider {provider_name} is now UNHEALTHY - "
                    f"circuit breaker may be activated"
                )

        _integration_health_service.register_health_change_callback(on_health_change)

        # Start health monitoring
        await _integration_health_service.start()

        logger.info("Integration health monitoring service started")

    except Exception as e:
        logger.error(f"Failed to initialize integration health monitoring: {e}")
        # Non-fatal: continue without health monitoring


async def _shutdown_integration_health_monitoring():
    """
    Story 1.4: Shutdown integration health monitoring service.
    """
    global _integration_health_service

    try:
        if _integration_health_service is None:
            return

        # Stop health monitoring
        await _integration_health_service.stop()
        _integration_health_service = None
        logger.info("Integration health monitoring service stopped")

    except Exception as e:
        logger.warning(f"Error during integration health monitoring shutdown: {e}")
