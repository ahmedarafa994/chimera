"""
Unified Provider Registry - Plugin-based registry for the Dynamic Provider Selection System.

This module implements the provider registry component of the unified provider selection
system, managing provider plugins and providing a central access point for provider operations.

This is distinct from the legacy AIConfigManager-based provider_registry.py and implements
the new architecture with Protocol-based plugins, three-tier selection hierarchy, and
capability-based filtering.
"""

import logging
from collections.abc import Callable
from threading import Lock
from typing import Any

from app.domain.interfaces import BaseLLMClient, ProviderPlugin
from app.domain.models import Capability, Model, Provider

logger = logging.getLogger(__name__)


class UnifiedProviderRegistry:
    """
    Singleton registry for managing AI provider plugins (Unified Selection System).

    This registry implements the plugin-based architecture for the Dynamic Provider
    and Model Selection System. It provides:

    - Thread-safe singleton pattern
    - Protocol-based plugin registration (ProviderPlugin)
    - Provider alias support (e.g., 'google' and 'gemini')
    - Capability-based filtering
    - Model validation against provider capabilities
    - Health checking across providers
    - Statistics and monitoring

    Usage:
        >>> registry = UnifiedProviderRegistry()
        >>> registry.register_provider(openai_plugin, aliases=["gpt"])
        >>> client = registry.create_client("openai", "gpt-4")
        >>> models = registry.get_models("openai", with_capabilities={Capability.STREAMING})
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Ensure singleton pattern - only one registry instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry (only happens once due to singleton)."""
        if self._initialized:
            return

        self._providers: dict[str, ProviderPlugin] = {}
        self._aliases: dict[str, str] = {}  # alias -> canonical_provider_id
        self._models_cache: dict[str, list[Model]] = {}
        self._initialization_hooks: list[Callable[[ProviderPlugin], None]] = []
        self._initialized = True

        logger.info("Unified provider registry initialized")

    def register_provider(
        self,
        plugin: ProviderPlugin,
        aliases: list[str] | None = None,
        run_hooks: bool = True,
    ) -> None:
        """
        Register a provider plugin with the registry.

        Args:
            plugin: The provider plugin instance to register
            aliases: Optional list of alternative names for this provider
            run_hooks: Whether to run initialization hooks (default: True)

        Raises:
            ValueError: If provider_id is already registered or invalid
            TypeError: If plugin doesn't implement ProviderPlugin protocol

        Example:
            >>> registry.register_provider(
            ...     openai_plugin,
            ...     aliases=["gpt"],
            ...     run_hooks=True
            ... )
        """
        if not isinstance(plugin, ProviderPlugin):
            raise TypeError(f"Plugin must implement ProviderPlugin protocol, got {type(plugin)}")

        provider_id = plugin.provider_id
        if not provider_id or not isinstance(provider_id, str):
            raise ValueError(f"Invalid provider_id: {provider_id}")

        with self._lock:
            if provider_id in self._providers:
                raise ValueError(f"Provider '{provider_id}' is already registered")

            # Validate configuration
            if not plugin.validate_config():
                logger.warning(
                    f"Provider '{provider_id}' registered with invalid configuration. "
                    "Some features may not work correctly."
                )

            # Register provider
            self._providers[provider_id] = plugin
            logger.info(f"Registered unified provider: {provider_id}")

            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases:
                        raise ValueError(
                            f"Alias '{alias}' already registered for provider "
                            f"'{self._aliases[alias]}'"
                        )
                    self._aliases[alias] = provider_id
                    logger.debug(f"Registered alias: {alias} -> {provider_id}")

            # Cache models
            try:
                models = plugin.get_available_models()
                self._models_cache[provider_id] = models
                logger.debug(f"Cached {len(models)} models for provider '{provider_id}'")
            except Exception as e:
                logger.error(f"Failed to cache models for '{provider_id}': {e}")
                self._models_cache[provider_id] = []

            # Run initialization hooks
            if run_hooks:
                for hook in self._initialization_hooks:
                    try:
                        hook(plugin)
                    except Exception as e:
                        logger.error(f"Initialization hook failed for '{provider_id}': {e}")

    def unregister_provider(self, provider_id: str) -> None:
        """
        Unregister a provider plugin.

        Args:
            provider_id: The provider ID or alias to unregister

        Raises:
            KeyError: If provider_id is not registered
        """
        canonical_id = self._resolve_provider_id(provider_id)

        with self._lock:
            # Remove provider
            del self._providers[canonical_id]

            # Remove aliases
            self._aliases = {k: v for k, v in self._aliases.items() if v != canonical_id}

            # Clear cache
            self._models_cache.pop(canonical_id, None)

            logger.info(f"Unregistered provider: {canonical_id}")

    def get_provider(self, provider_id: str) -> ProviderPlugin:
        """
        Get a provider plugin by ID or alias.

        Args:
            provider_id: The provider ID or alias

        Returns:
            The provider plugin instance

        Raises:
            KeyError: If provider is not registered
        """
        canonical_id = self._resolve_provider_id(provider_id)
        return self._providers[canonical_id]

    def has_provider(self, provider_id: str) -> bool:
        """
        Check if a provider is registered.

        Args:
            provider_id: The provider ID or alias to check

        Returns:
            True if provider is registered, False otherwise
        """
        try:
            self._resolve_provider_id(provider_id)
            return True
        except KeyError:
            return False

    def list_providers(
        self, enabled_only: bool = True, with_capabilities: set[Capability] | None = None
    ) -> list[Provider]:
        """
        List all registered providers.

        Args:
            enabled_only: Only return enabled providers (default: True)
            with_capabilities: Filter by required capabilities (optional)

        Returns:
            List of Provider metadata objects
        """
        providers = []

        for plugin in self._providers.values():
            try:
                provider = plugin.provider_metadata

                # Filter by enabled status
                if enabled_only and not provider.is_enabled:
                    continue

                # Filter by capabilities
                if with_capabilities:
                    if not with_capabilities.issubset(provider.capabilities):
                        continue

                providers.append(provider)
            except Exception as e:
                logger.error(f"Failed to get metadata for provider '{plugin.provider_id}': {e}")

        return providers

    def get_models(
        self,
        provider_id: str | None = None,
        enabled_only: bool = True,
        with_capabilities: set[Capability] | None = None,
    ) -> list[Model]:
        """
        Get available models, optionally filtered by provider and capabilities.

        Args:
            provider_id: Filter by specific provider (optional)
            enabled_only: Only return enabled models (default: True)
            with_capabilities: Filter by required capabilities (optional)

        Returns:
            List of Model metadata objects

        Raises:
            KeyError: If provider_id is specified but not found
        """
        if provider_id:
            # Get models for specific provider
            canonical_id = self._resolve_provider_id(provider_id)
            models = self._models_cache.get(canonical_id, [])
            provider_ids = [canonical_id]
        else:
            # Get models for all providers
            models = []
            provider_ids = list(self._providers.keys())
            for pid in provider_ids:
                models.extend(self._models_cache.get(pid, []))

        # Apply filters
        filtered_models = []
        for model in models:
            # Filter by enabled status
            if enabled_only and not model.is_enabled:
                continue

            # Filter by capabilities
            if with_capabilities:
                if not with_capabilities.issubset(model.capabilities):
                    continue

            filtered_models.append(model)

        return filtered_models

    def create_client(
        self, provider_id: str, model_id: str, **kwargs: Any
    ) -> BaseLLMClient:
        """
        Create a client instance for a specific provider and model.

        Args:
            provider_id: The provider ID or alias
            model_id: The model identifier
            **kwargs: Additional configuration parameters

        Returns:
            BaseLLMClient instance configured for the model

        Raises:
            KeyError: If provider is not registered
            ValueError: If model is not available or invalid configuration
        """
        plugin = self.get_provider(provider_id)

        try:
            client = plugin.create_client(model_id, **kwargs)
            logger.debug(f"Created unified client for {provider_id}/{model_id}")
            return client
        except Exception as e:
            logger.error(f"Failed to create client for {provider_id}/{model_id}: {e}")
            raise

    def validate_selection(self, provider_id: str, model_id: str) -> bool:
        """
        Validate that a provider/model selection is valid.

        Args:
            provider_id: The provider ID or alias
            model_id: The model identifier

        Returns:
            True if selection is valid, False otherwise
        """
        try:
            # Check provider exists
            canonical_id = self._resolve_provider_id(provider_id)

            # Check model exists in provider's model list
            models = self._models_cache.get(canonical_id, [])
            model_ids = {m.id for m in models}

            return model_id in model_ids
        except KeyError:
            return False

    async def health_check(self, provider_id: str | None = None) -> dict[str, bool]:
        """
        Perform health check on providers.

        Args:
            provider_id: Check specific provider (optional, checks all if None)

        Returns:
            Dictionary mapping provider IDs to health status (True = healthy)
        """
        results = {}

        if provider_id:
            # Check specific provider
            try:
                plugin = self.get_provider(provider_id)
                canonical_id = self._resolve_provider_id(provider_id)
                results[canonical_id] = await plugin.health_check()
            except Exception as e:
                logger.error(f"Health check failed for '{provider_id}': {e}")
                results[provider_id] = False
        else:
            # Check all providers
            for pid, plugin in self._providers.items():
                try:
                    results[pid] = await plugin.health_check()
                except Exception as e:
                    logger.error(f"Health check failed for '{pid}': {e}")
                    results[pid] = False

        return results

    def add_initialization_hook(self, hook: Callable[[ProviderPlugin], None]) -> None:
        """
        Add a hook that runs when new providers are registered.

        Args:
            hook: Callable that takes a ProviderPlugin and returns None
        """
        self._initialization_hooks.append(hook)
        logger.debug("Added initialization hook to unified registry")

    def refresh_models_cache(self, provider_id: str | None = None) -> None:
        """
        Refresh the cached model list for provider(s).

        Args:
            provider_id: Refresh specific provider (optional, refreshes all if None)
        """
        if provider_id:
            canonical_id = self._resolve_provider_id(provider_id)
            plugin = self._providers[canonical_id]
            try:
                models = plugin.get_available_models()
                self._models_cache[canonical_id] = models
                logger.debug(f"Refreshed model cache for '{canonical_id}': {len(models)} models")
            except Exception as e:
                logger.error(f"Failed to refresh models for '{canonical_id}': {e}")
        else:
            for pid, plugin in self._providers.items():
                try:
                    models = plugin.get_available_models()
                    self._models_cache[pid] = models
                    logger.debug(f"Refreshed model cache for '{pid}': {len(models)} models")
                except Exception as e:
                    logger.error(f"Failed to refresh models for '{pid}': {e}")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        total_models = sum(len(models) for models in self._models_cache.values())
        enabled_providers = sum(
            1 for p in self._providers.values() if p.provider_metadata.is_enabled
        )

        return {
            "total_providers": len(self._providers),
            "enabled_providers": enabled_providers,
            "total_aliases": len(self._aliases),
            "total_models": total_models,
            "providers": list(self._providers.keys()),
            "aliases": dict(self._aliases),
        }

    def _resolve_provider_id(self, provider_id: str) -> str:
        """
        Resolve a provider ID or alias to the canonical provider ID.

        Args:
            provider_id: The provider ID or alias

        Returns:
            The canonical provider ID

        Raises:
            KeyError: If provider_id is not found
        """
        # Check if it's an alias
        if provider_id in self._aliases:
            return self._aliases[provider_id]

        # Check if it's a canonical ID
        if provider_id in self._providers:
            return provider_id

        # Not found
        raise KeyError(f"Provider '{provider_id}' not registered in unified registry")

    def __repr__(self) -> str:
        """String representation of registry."""
        return (
            f"<UnifiedProviderRegistry providers={len(self._providers)} "
            f"aliases={len(self._aliases)} "
            f"models={sum(len(m) for m in self._models_cache.values())}>"
        )


# Global registry instance
unified_registry = UnifiedProviderRegistry()
