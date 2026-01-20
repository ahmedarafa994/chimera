"""Redis L2 Cache Implementation.

PERF-001 FIX: Implements distributed L2 caching with Redis for multi-level
caching strategy. Provides cache consistency, TTL management, and graceful
degradation when Redis is unavailable.

Features:
- Async Redis operations with aioredis
- Cache key versioning for invalidation
- Automatic serialization/deserialization
- Connection pooling and health monitoring
- Graceful fallback to L1 cache on Redis failure
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    url: str = "redis://localhost:6379/0"
    db: int = 0
    password: str | None = None
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    decode_responses: bool = True


@dataclass
class RedisStats:
    """Redis cache statistics."""

    hits: int = 0
    misses: int = 0
    failures: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0
    last_error: str | None = None
    last_error_time: float | None = None
    is_healthy: bool = True

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "failures": self.failures,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "is_healthy": self.is_healthy,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
        }


class RedisL2Cache:
    """Redis L2 cache with async operations and graceful degradation.

    PERF-001 FIX: Provides distributed caching layer with automatic
    failover to L1 cache when Redis is unavailable.
    """

    def __init__(
        self,
        config: RedisConfig | None = None,
        key_prefix: str = "chimera",
        default_ttl: int = 3600,
    ) -> None:
        """Initialize Redis L2 cache.

        Args:
            config: Redis connection configuration
            key_prefix: Prefix for all cache keys
            default_ttl: Default TTL in seconds

        """
        self.config = config or RedisConfig(
            url=getattr(settings, "REDIS_URL", "redis://localhost:6379/0"),
            password=getattr(settings, "REDIS_PASSWORD", None),
        )
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.stats = RedisStats()
        self._client: Any = None
        self._lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None
        self._initialized = False

    def _get_async_lock(self) -> asyncio.Lock:
        """Get or create async lock."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is not None:
            return self._client

        async with self._get_async_lock():
            # Double-check after acquiring lock
            if self._client is not None:
                return self._client

            try:
                # Try to import aioredis/redis
                try:
                    import redis.asyncio as aioredis
                except ImportError:
                    try:
                        import aioredis
                    except ImportError:
                        logger.warning(
                            "Redis library not found. Install with: pip install 'hiredis>=2.0.0,!=2.0.3,!=2.0.4' redis",
                        )
                        self.stats.is_healthy = False
                        return None

                # Create Redis client
                self._client = await aioredis.from_url(
                    self.config.url,
                    db=self.config.db,
                    password=self.config.password,
                    max_connections=self.config.max_connections,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    decode_responses=self.config.decode_responses,
                )

                # Test connection
                await self._client.ping()
                self._initialized = True
                self.stats.is_healthy = True
                logger.info(f"Redis L2 cache initialized: {self.config.url}")

            except Exception as e:
                logger.exception(f"Failed to initialize Redis L2 cache: {e}")
                self.stats.is_healthy = False
                self.stats.last_error = str(e)
                self.stats.last_error_time = time.time()
                self._client = None

        return self._client

    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.key_prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        """Get value from Redis cache.

        Args:
            key: Cache key (without prefix)

        Returns:
            Cached value or None if not found/error

        """
        start_time = time.time()
        self.stats.total_requests += 1

        try:
            client = await self._get_client()
            if client is None:
                self.stats.misses += 1
                return None

            redis_key = self._make_key(key)
            value = await client.get(redis_key)

            latency_ms = (time.time() - start_time) * 1000
            self.stats.total_latency_ms += latency_ms

            if value is not None:
                self.stats.hits += 1
                # Deserialize JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Return as string if not JSON
                    return value
            else:
                self.stats.misses += 1
                return None

        except Exception as e:
            self.stats.failures += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = time.time()
            logger.warning(f"Redis L2 cache get failed: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set value in Redis cache.

        Args:
            key: Cache key (without prefix)
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if successful, False otherwise

        """
        start_time = time.time()

        try:
            client = await self._get_client()
            if client is None:
                return False

            redis_key = self._make_key(key)
            ttl = ttl if ttl is not None else self.default_ttl

            # Serialize value
            if isinstance(value, str | int | float | bool):
                serialized = str(value)
            else:
                serialized = json.dumps(value)

            await client.setex(redis_key, ttl, serialized)

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Redis L2 cache set: {key} in {latency_ms:.2f}ms")

            return True

        except Exception as e:
            self.stats.failures += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = time.time()
            logger.warning(f"Redis L2 cache set failed: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache.

        Args:
            key: Cache key (without prefix)

        Returns:
            True if deleted, False otherwise

        """
        try:
            client = await self._get_client()
            if client is None:
                return False

            redis_key = self._make_key(key)
            await client.delete(redis_key)
            return True

        except Exception as e:
            logger.warning(f"Redis L2 cache delete failed: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all keys with the configured prefix.

        Returns:
            True if successful, False otherwise

        """
        try:
            client = await self._get_client()
            if client is None:
                return False

            pattern = f"{self.key_prefix}:*"
            keys = []

            # Scan for keys (more efficient than keys() for large datasets)
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await client.delete(*keys)

            logger.info(f"Cleared {len(keys)} keys from Redis L2 cache")
            return True

        except Exception as e:
            logger.exception(f"Redis L2 cache clear failed: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key (without prefix)

        Returns:
            True if key exists, False otherwise

        """
        try:
            client = await self._get_client()
            if client is None:
                return False

            redis_key = self._make_key(key)
            return await client.exists(redis_key) > 0

        except Exception as e:
            logger.warning(f"Redis L2 cache exists check failed: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary of cache statistics

        """
        # Also get Redis INFO if available
        redis_info = {}
        try:
            client = await self._get_client()
            if client is not None:
                info = await client.info()
                redis_info = {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "total_commands": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
        except Exception as e:
            logger.warning(f"Failed to get Redis INFO: {e}")

        return {
            "stats": self.stats.to_dict(),
            "redis_info": redis_info,
            "config": {
                "url": self.config.url,
                "key_prefix": self.key_prefix,
                "default_ttl": self.default_ttl,
            },
        }

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._initialized = False
            logger.info("Redis L2 cache connection closed")

    def is_available(self) -> bool:
        """Check if Redis cache is available."""
        return self._client is not None and self.stats.is_healthy


class MultiLevelCache:
    """Multi-level cache combining L1 (in-memory) and L2 (Redis) caching.

    PERF-001 FIX: Implements tiered caching strategy:
    - L1: Fast in-memory cache with LRU eviction
    - L2: Distributed Redis cache for horizontal scaling
    - Automatic promotion/demotion between levels
    - Graceful degradation when L2 is unavailable
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_ttl: int = 300,  # 5 minutes
        l2_ttl: int = 3600,  # 1 hour
        l2_config: RedisConfig | None = None,
        enable_l2: bool = True,
    ) -> None:
        """Initialize multi-level cache.

        Args:
            l1_max_size: Maximum size of L1 cache
            l1_ttl: Default TTL for L1 cache entries
            l2_ttl: Default TTL for L2 cache entries
            l2_config: Redis configuration
            enable_l2: Whether to enable L2 cache

        """
        # Import L1 cache here to avoid circular imports
        from app.services.transformation_service import TransformationCache

        self.l1 = TransformationCache(
            ttl_seconds=l1_ttl,
            max_size=l1_max_size,
        )
        self.l2 = RedisL2Cache(config=l2_config, default_ttl=l2_ttl) if enable_l2 else None
        self.enable_l2 = enable_l2
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l1_misses": 0,
            "l2_misses": 0,
            "writes": 0,
        }

    async def get(self, key: str) -> Any | None:
        """Get value from cache (checks L1, then L2).

        Args:
            key: Cache key

        Returns:
            Cached value or None

        """
        # Try L1 first (fastest)
        l1_result = self.l1.get_raw(key)
        if l1_result is not None:
            self.stats["l1_hits"] += 1
            return l1_result
        self.stats["l1_misses"] += 1

        # Try L2 if enabled
        if self.enable_l2 and self.l2:
            l2_result = await self.l2.get(key)
            if l2_result is not None:
                self.stats["l2_hits"] += 1
                # Promote to L1
                self.l1.set_raw(key, l2_result)
                return l2_result
            self.stats["l2_misses"] += 1

        return None

    async def set(
        self,
        key: str,
        value: Any,
        l1_ttl: int | None = None,
        l2_ttl: int | None = None,
    ) -> None:
        """Set value in both L1 and L2 cache.

        Args:
            key: Cache key
            value: Value to cache
            l1_ttl: L1 TTL (uses default if None)
            l2_ttl: L2 TTL (uses default if None)

        """
        self.stats["writes"] += 1

        # Set in L1
        self.l1.set_raw(key, value, ttl=l1_ttl)

        # Set in L2 if enabled
        if self.enable_l2 and self.l2:
            await self.l2.set(key, value, ttl=l2_ttl)

    async def delete(self, key: str) -> None:
        """Delete value from both L1 and L2 cache.

        Args:
            key: Cache key

        """
        # Delete from L1
        self.l1.delete_raw(key)

        # Delete from L2 if enabled
        if self.enable_l2 and self.l2:
            await self.l2.delete(key)

    async def clear(self) -> None:
        """Clear both L1 and L2 cache."""
        self.l1.clear()
        if self.enable_l2 and self.l2:
            await self.l2.clear()
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l1_misses": 0,
            "l2_misses": 0,
            "writes": 0,
        }

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics for both cache levels.

        Returns:
            Dictionary of combined statistics

        """
        l2_stats = {}
        if self.enable_l2 and self.l2:
            l2_stats = await self.l2.get_stats()

        l1_stats = self.l1.get_metrics()

        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"]
        total_misses = self.stats["l1_misses"] + self.stats["l2_misses"]
        total_requests = total_hits + total_misses

        return {
            "multi_level": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
                "l1_hit_rate": (
                    self.stats["l1_hits"] / (self.stats["l1_hits"] + self.stats["l1_misses"])
                    if (self.stats["l1_hits"] + self.stats["l1_misses"]) > 0
                    else 0.0
                ),
                "l2_hit_rate": (
                    self.stats["l2_hits"] / (self.stats["l2_hits"] + self.stats["l2_misses"])
                    if (self.stats["l2_hits"] + self.stats["l2_misses"]) > 0
                    else 0.0
                ),
                "writes": self.stats["writes"],
                "l2_enabled": self.enable_l2,
            },
            "l1": l1_stats,
            "l2": l2_stats,
        }

    async def close(self) -> None:
        """Close L2 connection."""
        if self.enable_l2 and self.l2:
            await self.l2.close()


# Global multi-level cache instance
_multi_level_cache: MultiLevelCache | None = None


def get_multi_level_cache(
    l1_max_size: int = 1000,
    l1_ttl: int = 300,
    l2_ttl: int = 3600,
    enable_l2: bool = True,
) -> MultiLevelCache:
    """Get or create global multi-level cache instance."""
    global _multi_level_cache
    if _multi_level_cache is None:
        _multi_level_cache = MultiLevelCache(
            l1_max_size=l1_max_size,
            l1_ttl=l1_ttl,
            l2_ttl=l2_ttl,
            enable_l2=enable_l2,
        )
    return _multi_level_cache


__all__ = [
    "MultiLevelCache",
    "RedisConfig",
    "RedisL2Cache",
    "RedisStats",
    "get_multi_level_cache",
]
