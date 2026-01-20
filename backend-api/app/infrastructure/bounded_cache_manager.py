"""Bounded cache manager for jailbreak service.

This module provides a production-ready cache manager with bounded memory usage,
LRU eviction, and Redis backend support to prevent memory exhaustion attacks.
"""

import json
import logging
from typing import Any

from app.core.cache import CacheConfig, HybridCache

logger = logging.getLogger(__name__)


class BoundedCacheManager:
    """Jailbreak service cache manager with bounded memory and Redis backend.

    Wraps HybridCache to provide jailbreak-specific caching semantics
    with protection against memory exhaustion attacks.

    Features:
    - Bounded size with LRU eviction
    - Per-value size limits
    - Redis backend with memory fallback
    - Jailbreak-specific convenience methods
    """

    # Jailbreak-specific configuration
    DEFAULT_TTL = 3600  # 1 hour
    EXECUTION_RESULT_TTL = 1800  # 30 minutes
    TECHNIQUE_CACHE_TTL = 7200  # 2 hours
    SAFETY_RESULT_TTL = 300  # 5 minutes
    MAX_VALUE_SIZE_BYTES = 1_000_000  # 1MB per entry
    KEY_PREFIX = "jailbreak:"

    def __init__(
        self,
        config: CacheConfig | None = None,
        max_items: int = 5000,
        max_value_size: int | None = None,
    ) -> None:
        cache_config = config or CacheConfig(
            max_memory_items=max_items,
            key_prefix="chimera:jailbreak:",
        )
        self._cache = HybridCache(cache_config)
        self._initialized = False
        self._max_value_size = max_value_size or self.MAX_VALUE_SIZE_BYTES

    async def initialize(self) -> None:
        """Initialize the cache backend."""
        if not self._initialized:
            await self._cache.initialize()
            self._initialized = True
            logger.info("BoundedCacheManager initialized")

    async def shutdown(self) -> None:
        """Shutdown cache backend."""
        if self._initialized:
            await self._cache.shutdown()
            self._initialized = False
            logger.info("BoundedCacheManager shutdown")

    def _validate_value_size(self, value: Any) -> bool:
        """Reject values that exceed maximum size."""
        try:
            serialized = json.dumps(value, default=str)
            return len(serialized.encode()) <= self._max_value_size
        except Exception:
            return False

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        full_key = f"{self.KEY_PREFIX}{key}"
        return await self._cache.get(full_key)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Set value in cache with size validation.

        Returns False if value exceeds size limit.
        """
        if not self._validate_value_size(value):
            logger.warning(f"Cache value for key {key} exceeds size limit, skipping")
            return False

        full_key = f"{self.KEY_PREFIX}{key}"
        ttl = ttl_seconds or self.DEFAULT_TTL
        await self._cache.set(full_key, value, ttl)
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        full_key = f"{self.KEY_PREFIX}{key}"
        return await self._cache.delete(full_key)

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        full_key = f"{self.KEY_PREFIX}{key}"
        value = await self._cache.get(full_key)
        return value is not None

    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        full_pattern = f"{self.KEY_PREFIX}{pattern}"
        return await self._cache.clear_pattern(full_pattern)

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = await self._cache.get_stats()
        stats["max_value_size_bytes"] = self._max_value_size
        stats["key_prefix"] = self.KEY_PREFIX
        return stats

    # Jailbreak-specific convenience methods

    async def cache_execution_result(self, execution_id: str, result: dict[str, Any]) -> bool:
        """Cache an execution result."""
        key = f"result:{execution_id}"
        return await self.set(key, result, self.EXECUTION_RESULT_TTL)

    async def get_execution_result(self, execution_id: str) -> dict[str, Any] | None:
        """Get cached execution result."""
        key = f"result:{execution_id}"
        return await self.get(key)

    async def cache_technique(self, technique_id: str, technique_data: dict[str, Any]) -> bool:
        """Cache technique data."""
        key = f"technique:{technique_id}"
        return await self.set(key, technique_data, self.TECHNIQUE_CACHE_TTL)

    async def get_technique(self, technique_id: str) -> dict[str, Any] | None:
        """Get cached technique data."""
        key = f"technique:{technique_id}"
        return await self.get(key)

    async def cache_safety_result(self, prompt_hash: str, result: dict[str, Any]) -> bool:
        """Cache safety validation result."""
        key = f"safety:{prompt_hash}"
        return await self.set(key, result, self.SAFETY_RESULT_TTL)

    async def get_safety_result(self, prompt_hash: str) -> dict[str, Any] | None:
        """Get cached safety validation result."""
        key = f"safety:{prompt_hash}"
        return await self.get(key)

    async def invalidate_technique(self, technique_id: str) -> bool:
        """Invalidate cached technique data."""
        key = f"technique:{technique_id}"
        return await self.delete(key)

    async def invalidate_all_techniques(self) -> int:
        """Invalidate all cached techniques."""
        return await self.clear_pattern("technique:*")


# Global instance (initialized on startup)
cache_manager: BoundedCacheManager | None = None


async def get_cache_manager() -> BoundedCacheManager:
    """Get or create the global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        from app.core.config import settings

        max_items = getattr(settings, "CACHE_MAX_MEMORY_ITEMS", 5000)
        max_value_size = getattr(settings, "CACHE_MAX_VALUE_SIZE_BYTES", 1_000_000)
        cache_manager = BoundedCacheManager(max_items=max_items, max_value_size=max_value_size)
        await cache_manager.initialize()
    return cache_manager


async def shutdown_cache_manager() -> None:
    """Shutdown the global cache manager."""
    global cache_manager
    if cache_manager:
        await cache_manager.shutdown()
        cache_manager = None
