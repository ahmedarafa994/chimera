"""
Configuration Service with Caching

Story 1.3: Configuration Persistence System
Provides business logic for configuration management with in-memory caching.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.infrastructure.database.connection import DatabaseConnection, get_database
from app.infrastructure.database.schema import initialize_schema
from app.infrastructure.database.unit_of_work import UnitOfWork
from app.infrastructure.repositories.api_key_repository import ApiKeyEntity
from app.infrastructure.repositories.config_repository import ProviderConfigEntity

logger = logging.getLogger(__name__)


class ConfigServiceError(Exception):
    """Base exception for configuration service."""


class ConfigNotFoundError(ConfigServiceError):
    """Raised when configuration is not found."""


@dataclass
class CacheEntry:
    """Cache entry with expiration tracking."""

    value: Any
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) > self.expires_at


class ConfigurationService:
    """
    Configuration service with in-memory caching.

    Provides high-level operations for managing provider configurations
    and API keys with automatic cache invalidation.
    """

    DEFAULT_CACHE_TTL = timedelta(minutes=5)

    def __init__(
        self,
        db: DatabaseConnection,
        cache_ttl: timedelta | None = None
    ):
        """
        Initialize configuration service.

        Args:
            db: Database connection instance.
            cache_ttl: Cache time-to-live. Defaults to 5 minutes.
        """
        self._db = db
        self._cache_ttl = cache_ttl or self.DEFAULT_CACHE_TTL
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    def _cache_key(self, namespace: str, key: str) -> str:
        """Generate cache key."""
        return f"{namespace}:{key}"

    def _get_cached(self, namespace: str, key: str) -> Any | None:
        """Get value from cache if not expired."""
        cache_key = self._cache_key(namespace, key)
        entry = self._cache.get(cache_key)

        if entry is None:
            return None

        if entry.is_expired:
            del self._cache[cache_key]
            return None

        return entry.value

    def _set_cached(self, namespace: str, key: str, value: Any) -> None:
        """Set value in cache with expiration."""
        cache_key = self._cache_key(namespace, key)
        self._cache[cache_key] = CacheEntry(
            value=value,
            expires_at=datetime.now(UTC) + self._cache_ttl
        )

    def _invalidate_cache(self, namespace: str, key: str | None = None) -> None:
        """
        Invalidate cache entries.

        Args:
            namespace: Cache namespace to invalidate.
            key: Specific key to invalidate. If None, invalidates all in namespace.
        """
        if key:
            cache_key = self._cache_key(namespace, key)
            self._cache.pop(cache_key, None)
        else:
            prefix = f"{namespace}:"
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._cache[k]

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Configuration cache cleared")

    # =========================================================================
    # Provider Configuration Operations
    # =========================================================================

    async def get_config(self, config_id: str) -> ProviderConfigEntity | None:
        """
        Get provider configuration by ID.

        Args:
            config_id: Configuration ID.

        Returns:
            Configuration entity or None.
        """
        cached = self._get_cached("config", config_id)
        if cached is not None:
            return cached

        uow = UnitOfWork(self._db)
        config = await uow.configs.get_by_id(config_id)

        if config:
            self._set_cached("config", config_id, config)

        return config

    async def get_all_configs(self) -> list[ProviderConfigEntity]:
        """Get all provider configurations."""
        cached = self._get_cached("config", "__all__")
        if cached is not None:
            return cached

        uow = UnitOfWork(self._db)
        configs = await uow.configs.get_all()

        self._set_cached("config", "__all__", configs)
        return configs

    async def get_configs_by_provider(
        self, provider_type: str
    ) -> list[ProviderConfigEntity]:
        """
        Get configurations for a specific provider type.

        Args:
            provider_type: Provider type to filter.

        Returns:
            List of configurations.
        """
        cache_key = f"provider:{provider_type}"
        cached = self._get_cached("config", cache_key)
        if cached is not None:
            return cached

        uow = UnitOfWork(self._db)
        configs = await uow.configs.get_by_provider_type(provider_type)

        self._set_cached("config", cache_key, configs)
        return configs

    async def get_default_config(
        self, provider_type: str | None = None
    ) -> ProviderConfigEntity | None:
        """
        Get default configuration.

        Args:
            provider_type: Optional provider type filter.

        Returns:
            Default configuration or None.
        """
        cache_key = f"default:{provider_type or 'any'}"
        cached = self._get_cached("config", cache_key)
        if cached is not None:
            return cached

        uow = UnitOfWork(self._db)
        config = await uow.configs.get_default(provider_type)

        if config:
            self._set_cached("config", cache_key, config)

        return config

    async def create_config(
        self,
        provider_type: str,
        name: str,
        settings: dict[str, Any] | None = None,
        is_default: bool = False
    ) -> ProviderConfigEntity:
        """
        Create new provider configuration.

        Args:
            provider_type: Provider type (e.g., 'openai', 'anthropic').
            name: Configuration name.
            settings: Provider-specific settings.
            is_default: Whether this is the default configuration.

        Returns:
            Created configuration entity.
        """
        entity = ProviderConfigEntity(
            provider_type=provider_type,
            name=name,
            settings=settings or {},
            is_default=is_default
        )

        async with self._lock:
            uow = UnitOfWork(self._db)
            created = await uow.configs.create(entity)
            self._invalidate_cache("config")

        logger.info(f"Created config: {name} for {provider_type}")
        return created

    async def update_config(
        self,
        config_id: str,
        name: str | None = None,
        settings: dict[str, Any] | None = None,
        is_default: bool | None = None
    ) -> ProviderConfigEntity:
        """
        Update existing configuration.

        Args:
            config_id: Configuration ID to update.
            name: New name (optional).
            settings: New settings (optional).
            is_default: New default status (optional).

        Returns:
            Updated configuration entity.

        Raises:
            ConfigNotFoundError: If configuration not found.
        """
        async with self._lock:
            uow = UnitOfWork(self._db)
            config = await uow.configs.get_by_id(config_id)

            if not config:
                raise ConfigNotFoundError(f"Config {config_id} not found")

            if name is not None:
                config.name = name
            if settings is not None:
                config.settings = settings
            if is_default is not None:
                config.is_default = is_default

            updated = await uow.configs.update(config)
            self._invalidate_cache("config")

        logger.info(f"Updated config: {config_id}")
        return updated

    async def delete_config(self, config_id: str) -> bool:
        """
        Delete configuration.

        Args:
            config_id: Configuration ID to delete.

        Returns:
            True if deleted successfully.
        """
        async with self._lock:
            uow = UnitOfWork(self._db)
            result = await uow.configs.delete(config_id)
            if result:
                self._invalidate_cache("config")

        return result

    async def set_default_config(self, config_id: str) -> ProviderConfigEntity:
        """
        Set configuration as default.

        Args:
            config_id: Configuration ID to set as default.

        Returns:
            Updated configuration.
        """
        async with self._lock:
            uow = UnitOfWork(self._db)
            config = await uow.configs.set_default(config_id)
            self._invalidate_cache("config")

        return config

    # =========================================================================
    # API Key Operations
    # =========================================================================

    async def get_api_key(self, key_id: str) -> ApiKeyEntity | None:
        """
        Get API key by ID.

        Args:
            key_id: API key ID.

        Returns:
            API key entity or None.
        """
        cached = self._get_cached("api_key", key_id)
        if cached is not None:
            return cached

        uow = UnitOfWork(self._db)
        key = await uow.api_keys.get_by_id(key_id)

        if key:
            self._set_cached("api_key", key_id, key)

        return key

    async def get_active_api_key(
        self, provider_type: str
    ) -> ApiKeyEntity | None:
        """
        Get active API key for a provider.

        Args:
            provider_type: Provider type.

        Returns:
            Active API key or None.
        """
        cache_key = f"active:{provider_type}"
        cached = self._get_cached("api_key", cache_key)
        if cached is not None:
            return cached

        uow = UnitOfWork(self._db)
        key = await uow.api_keys.get_active_key(provider_type)

        if key:
            self._set_cached("api_key", cache_key, key)

        return key

    async def store_api_key(
        self,
        provider_type: str,
        key_name: str,
        api_key: str
    ) -> ApiKeyEntity:
        """
        Store new API key (encrypted at rest).

        Args:
            provider_type: Provider type.
            key_name: Key identifier name.
            api_key: The API key to store.

        Returns:
            Created API key entity.
        """
        entity = ApiKeyEntity(
            provider_type=provider_type,
            key_name=key_name,
            api_key=api_key
        )

        async with self._lock:
            uow = UnitOfWork(self._db)
            created = await uow.api_keys.create(entity)
            self._invalidate_cache("api_key")

        logger.info(f"Stored API key: {key_name} for {provider_type}")
        return created

    async def rotate_api_key(
        self,
        key_id: str,
        new_api_key: str
    ) -> ApiKeyEntity:
        """
        Rotate an existing API key.

        Args:
            key_id: Key ID to rotate.
            new_api_key: New API key value.

        Returns:
            Updated API key entity.

        Raises:
            ConfigNotFoundError: If key not found.
        """
        async with self._lock:
            uow = UnitOfWork(self._db)
            key = await uow.api_keys.get_by_id(key_id)

            if not key:
                raise ConfigNotFoundError(f"API key {key_id} not found")

            key.api_key = new_api_key
            updated = await uow.api_keys.update(key)
            self._invalidate_cache("api_key")

        logger.info(f"Rotated API key: {key_id}")
        return updated

    async def deactivate_api_key(self, key_id: str) -> bool:
        """
        Deactivate an API key.

        Args:
            key_id: Key ID to deactivate.

        Returns:
            True if deactivated.
        """
        async with self._lock:
            uow = UnitOfWork(self._db)
            result = await uow.api_keys.deactivate(key_id)
            if result:
                self._invalidate_cache("api_key")

        return result

    async def delete_api_key(self, key_id: str) -> bool:
        """
        Delete an API key.

        Args:
            key_id: Key ID to delete.

        Returns:
            True if deleted.
        """
        async with self._lock:
            uow = UnitOfWork(self._db)
            result = await uow.api_keys.delete(key_id)
            if result:
                self._invalidate_cache("api_key")

        return result

    async def record_key_usage(self, key_id: str) -> bool:
        """
        Record API key usage timestamp.

        Args:
            key_id: Key ID to update.

        Returns:
            True if updated.
        """
        uow = UnitOfWork(self._db)
        return await uow.api_keys.record_usage(key_id)


# Module-level service singleton
_config_service: ConfigurationService | None = None


async def get_config_service() -> ConfigurationService:
    """
    Get or create configuration service singleton.

    Initializes database and schema if needed.

    Returns:
        ConfigurationService instance.
    """
    global _config_service

    if _config_service is None:
        db = await get_database()
        await initialize_schema(db)
        _config_service = ConfigurationService(db)
        logger.info("Configuration service initialized")

    return _config_service


async def reset_config_service() -> None:
    """Reset configuration service (for testing)."""
    global _config_service
    if _config_service:
        _config_service.clear_cache()
    _config_service = None
