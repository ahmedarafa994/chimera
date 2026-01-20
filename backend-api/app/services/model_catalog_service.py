"""Model Catalog Service for Project Chimera.

Provides dynamic model discovery and catalog management across all
registered AI providers. Supports caching, refresh, and aggregation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from app.services.provider_plugins import ModelInfo, ProviderInfo, get_all_plugins, get_plugin

logger = logging.getLogger(__name__)


class ModelCatalogCache:
    """Cache for model catalog data with TTL support."""

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        """Initialize the cache.

        Args:
            default_ttl_seconds: Default time-to-live in seconds (1 hour)

        """
        self._cache: dict[str, list[ModelInfo]] = {}
        self._timestamps: dict[str, datetime] = {}
        self._ttl = timedelta(seconds=default_ttl_seconds)
        self._lock = asyncio.Lock()

    async def get(self, provider: str) -> list[ModelInfo] | None:
        """Get cached models for a provider if not expired."""
        async with self._lock:
            if provider not in self._cache:
                return None

            timestamp = self._timestamps.get(provider)
            if timestamp and datetime.utcnow() - timestamp > self._ttl:
                # Cache expired
                del self._cache[provider]
                del self._timestamps[provider]
                return None

            return self._cache[provider]

    async def set(self, provider: str, models: list[ModelInfo]) -> None:
        """Set cached models for a provider."""
        async with self._lock:
            self._cache[provider] = models
            self._timestamps[provider] = datetime.utcnow()

    async def invalidate(self, provider: str | None = None) -> None:
        """Invalidate cache for a provider or all providers."""
        async with self._lock:
            if provider:
                self._cache.pop(provider, None)
                self._timestamps.pop(provider, None)
            else:
                self._cache.clear()
                self._timestamps.clear()

    async def get_all(self) -> dict[str, list[ModelInfo]]:
        """Get all cached models (non-expired only)."""
        async with self._lock:
            result = {}
            now = datetime.utcnow()
            expired = []

            for provider, models in self._cache.items():
                timestamp = self._timestamps.get(provider)
                if timestamp and now - timestamp <= self._ttl:
                    result[provider] = models
                else:
                    expired.append(provider)

            # Clean up expired entries
            for provider in expired:
                del self._cache[provider]
                del self._timestamps[provider]

            return result


class ModelCatalogService:
    """Service for dynamic model discovery and catalog management.

    Features:
    - Query each provider's API for available models
    - Cache model lists with configurable TTL
    - Support forced refresh via admin API
    - Return model metadata: name, context_length, capabilities, pricing
    """

    def __init__(self, cache_ttl_seconds: int = 3600) -> None:
        """Initialize the Model Catalog Service.

        Args:
            cache_ttl_seconds: Cache TTL in seconds (default 1 hour)

        """
        self._cache = ModelCatalogCache(cache_ttl_seconds)
        self._discovery_lock = asyncio.Lock()
        self._last_full_refresh: datetime | None = None

    async def discover_models(self, provider: str, force_refresh: bool = False) -> list[ModelInfo]:
        """Discover models for a specific provider.

        Args:
            provider: Provider type (e.g., "openai", "anthropic")
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            List of ModelInfo objects for the provider

        """
        # Check cache first unless force refresh
        if not force_refresh:
            cached = await self._cache.get(provider)
            if cached is not None:
                logger.debug(f"Returning cached models for {provider}")
                return cached

        # Get plugin for provider
        plugin = get_plugin(provider)
        if not plugin:
            logger.warning(f"No plugin found for provider: {provider}")
            return []

        try:
            logger.info(f"Discovering models for provider: {provider}")
            models = await plugin.list_models()

            # Cache the results
            await self._cache.set(provider, models)

            logger.info(f"Discovered {len(models)} models for {provider}")
            return models
        except Exception as e:
            logger.exception(f"Error discovering models for {provider}: {e}")
            # Return cached data if available, even if expired
            cached = await self._cache.get(provider)
            return cached if cached else []

    async def refresh_catalog(self, provider: str | None = None) -> dict[str, list[ModelInfo]]:
        """Force refresh of model catalog.

        Args:
            provider: Optional provider to refresh.
                      If None, refreshes all providers.

        Returns:
            Dict mapping provider names to their model lists

        """
        async with self._discovery_lock:
            if provider:
                # Refresh single provider
                await self._cache.invalidate(provider)
                models = await self.discover_models(provider, force_refresh=True)
                return {provider: models}
            # Refresh all providers
            await self._cache.invalidate()
            return await self._discover_all_providers(force_refresh=True)

    async def get_available_models(self, provider: str) -> list[ModelInfo]:
        """Get available models for a provider.

        Uses cache if available, otherwise discovers models.

        Args:
            provider: Provider type

        Returns:
            List of available models

        """
        return await self.discover_models(provider, force_refresh=False)

    async def get_all_providers_with_models(self) -> dict[str, list[ModelInfo]]:
        """Get all providers with their available models.

        Returns:
            Dict mapping provider names to their model lists

        """
        return await self._discover_all_providers(force_refresh=False)

    async def _discover_all_providers(
        self,
        force_refresh: bool = False,
    ) -> dict[str, list[ModelInfo]]:
        """Discover models from all registered providers.

        Args:
            force_refresh: If True, bypass cache

        Returns:
            Dict mapping provider names to model lists

        """
        plugins = get_all_plugins()

        if not plugins:
            logger.warning("No provider plugins registered")
            return {}

        # Discover models concurrently with timeout
        async def discover_with_timeout(provider_type: str) -> tuple[str, list[ModelInfo]]:
            try:
                models = await asyncio.wait_for(
                    self.discover_models(provider_type, force_refresh),
                    timeout=30.0,
                )
                return (provider_type, models)
            except TimeoutError:
                logger.warning(f"Timeout discovering models for {provider_type}")
                return (provider_type, [])
            except Exception as e:
                logger.exception(f"Error discovering {provider_type}: {e}")
                return (provider_type, [])

        # Create tasks for all providers
        tasks = [discover_with_timeout(plugin.provider_type) for plugin in plugins]

        # Run all discoveries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        catalog: dict[str, list[ModelInfo]] = {}
        for result in results:
            if isinstance(result, tuple):
                provider_type, models = result
                catalog[provider_type] = models

        self._last_full_refresh = datetime.utcnow()
        return catalog

    async def get_provider_info(self, provider: str) -> ProviderInfo | None:
        """Get detailed information about a provider.

        Args:
            provider: Provider type

        Returns:
            ProviderInfo object or None if not found

        """
        plugin = get_plugin(provider)
        if not plugin:
            return None

        models = await self.get_available_models(provider)

        return ProviderInfo(
            provider_id=plugin.provider_type,
            display_name=plugin.display_name,
            is_available=bool(models),
            models_count=len(models),
            capabilities=[cap.value for cap in plugin.capabilities],
            default_model=plugin.get_default_model(),
        )

    async def get_all_provider_info(self) -> list[ProviderInfo]:
        """Get information about all registered providers.

        Returns:
            List of ProviderInfo objects

        """
        plugins = get_all_plugins()

        async def get_info(provider_type: str) -> ProviderInfo | None:
            return await self.get_provider_info(provider_type)

        tasks = [get_info(p.provider_type) for p in plugins]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if isinstance(r, ProviderInfo)]

    async def search_models(
        self,
        query: str | None = None,
        capability: str | None = None,
        min_context_window: int | None = None,
        supports_streaming: bool | None = None,
        supports_vision: bool | None = None,
    ) -> list[ModelInfo]:
        """Search for models across all providers with filters.

        Args:
            query: Text search in model name/display name
            capability: Required capability (e.g., "vision", "code")
            min_context_window: Minimum context window size
            supports_streaming: Filter by streaming support
            supports_vision: Filter by vision support

        Returns:
            List of matching ModelInfo objects

        """
        all_models = await self.get_all_providers_with_models()

        results = []
        for models in all_models.values():
            for model in models:
                # Apply filters
                if query:
                    search_text = f"{model.model_id} {model.display_name}".lower()
                    if query.lower() not in search_text:
                        continue

                if capability and model.capabilities:
                    if capability not in model.capabilities:
                        continue

                if min_context_window:
                    if (model.context_window or 0) < min_context_window:
                        continue

                if supports_streaming is not None:
                    if model.supports_streaming != supports_streaming:
                        continue

                if supports_vision is not None:
                    if model.supports_vision != supports_vision:
                        continue

                results.append(model)

        return results

    async def get_model_by_id(
        self,
        model_id: str,
        provider: str | None = None,
    ) -> ModelInfo | None:
        """Get a specific model by ID.

        Args:
            model_id: The model identifier
            provider: Optional provider to search in

        Returns:
            ModelInfo or None if not found

        """
        if provider:
            models = await self.get_available_models(provider)
            for model in models:
                if model.model_id == model_id:
                    return model
            return None

        # Search all providers
        all_models = await self.get_all_providers_with_models()
        for models in all_models.values():
            for model in models:
                if model.model_id == model_id:
                    return model

        return None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the model cache."""
        return {
            "last_full_refresh": (
                self._last_full_refresh.isoformat() if self._last_full_refresh else None
            ),
            "cache_ttl_seconds": self._cache._ttl.total_seconds(),
        }


# Singleton instance
_model_catalog_service: ModelCatalogService | None = None


def get_model_catalog_service() -> ModelCatalogService:
    """Get the singleton ModelCatalogService instance."""
    global _model_catalog_service
    if _model_catalog_service is None:
        _model_catalog_service = ModelCatalogService()
    return _model_catalog_service


async def initialize_model_catalog() -> None:
    """Initialize the model catalog at application startup.

    This pre-populates the cache with model information
    from all available providers.
    """
    logger.info("Initializing model catalog...")
    service = get_model_catalog_service()

    try:
        catalog = await service.get_all_providers_with_models()
        total_models = sum(len(models) for models in catalog.values())
        logger.info(
            f"Model catalog initialized: {len(catalog)} providers, {total_models} total models",
        )
    except Exception as e:
        logger.exception(f"Error initializing model catalog: {e}")
