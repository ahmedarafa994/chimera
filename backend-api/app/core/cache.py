"""
High-performance caching system with Redis and in-memory fallback.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600  # 1 hour
    max_memory_items: int = 10000
    enable_compression: bool = True
    key_prefix: str = "chimera:"

    # TTL settings for different data types
    transformation_ttl: int = 1800  # 30 minutes
    api_response_ttl: int = 600  # 10 minutes
    session_ttl: int = 7200  # 2 hours
    metrics_ttl: int = 300  # 5 minutes


class MemoryCache:
    """High-performance in-memory cache with LRU eviction."""

    def __init__(self, max_items: int = 10000):
        self.max_items = max_items
        self._cache: dict[str, dict[str, Any]] = {}
        self._access_times: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get item from cache."""
        async with self._lock:
            if key not in self._cache:
                return None

            item = self._cache[key]
            if time.time() > item["expires_at"]:
                del self._cache[key]
                del self._access_times[key]
                return None

            # Update access time for LRU
            self._access_times[key] = time.time()
            return item["value"]

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set item in cache with TTL."""
        ttl = ttl or CacheConfig.default_ttl
        expires_at = time.time() + ttl

        async with self._lock:
            # Evict items if cache is full
            if len(self._cache) >= self.max_items and key not in self._cache:
                await self._evict_lru()

            self._cache[key] = {"value": value, "expires_at": expires_at, "created_at": time.time()}
            self._access_times[key] = time.time()

    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all items from cache."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()

    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return

        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._cache[lru_key]
        del self._access_times[lru_key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_items": len(self._cache),
            "max_items": self.max_items,
            "memory_usage_kb": sum(len(str(item)) for item in self._cache.values()) // 1024,
        }


class RedisCache:
    """Redis-based distributed cache."""

    def __init__(self, redis_url: str, key_prefix: str = "chimera:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis: redis.Redis | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to memory cache.")
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Get item from Redis cache."""
        if not self._connected or not self._redis:
            return None

        try:
            value = await self._redis.get(self._make_key(key))
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set item in Redis cache."""
        if not self._connected or not self._redis:
            return False

        try:
            ttl = ttl or CacheConfig.default_ttl
            serialized_value = json.dumps(value, default=str)
            await self._redis.setex(self._make_key(key), ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        if not self._connected or not self._redis:
            return False

        try:
            result = await self._redis.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all items with prefix."""
        if not self._connected or not self._redis:
            return False

        try:
            pattern = f"{self.key_prefix}*"
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get Redis cache statistics."""
        if not self._connected or not self._redis:
            return {"connected": False}

        try:
            info = await self._redis.info()
            return {
                "connected": True,
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"connected": False, "error": str(e)}


class HybridCache:
    """Hybrid cache that uses Redis with memory fallback."""

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.memory_cache = MemoryCache(self.config.max_memory_items)
        self.redis_cache = RedisCache(self.config.redis_url, self.config.key_prefix)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize cache system."""
        if self._initialized:
            return

        await self.redis_cache.connect()
        self._initialized = True
        logger.info("Hybrid cache system initialized")

    async def shutdown(self) -> None:
        """Shutdown cache system."""
        await self.redis_cache.disconnect()
        await self.memory_cache.clear()
        logger.info("Hybrid cache system shutdown")

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    async def get(self, key: str, check_memory: bool = True) -> Any | None:
        """Get item from cache (memory first, then Redis)."""
        if check_memory:
            # Try memory cache first (fastest)
            memory_value = await self.memory_cache.get(key)
            if memory_value is not None:
                return memory_value

        # Try Redis cache
        redis_value = await self.redis_cache.get(key)
        if redis_value is not None:
            # Cache in memory for faster access next time
            await self.memory_cache.set(key, redis_value, ttl=300)  # 5 minutes
            return redis_value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        store_memory: bool = True,
        store_redis: bool = True,
    ) -> None:
        """Set item in cache."""
        tasks = []

        if store_memory:
            tasks.append(self.memory_cache.set(key, value, ttl))

        if store_redis:
            tasks.append(self.redis_cache.set(key, value, ttl))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        memory_deleted = await self.memory_cache.delete(key)
        redis_deleted = await self.redis_cache.delete(key)
        return memory_deleted or redis_deleted

    async def clear(self) -> None:
        """Clear all caches."""
        await asyncio.gather(
            self.memory_cache.clear(), self.redis_cache.clear(), return_exceptions=True
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = await self.redis_cache.get_stats()

        return {
            "memory_cache": memory_stats,
            "redis_cache": redis_stats,
            "hybrid_enabled": self.redis_cache._connected,
        }


# Global cache instance
cache = HybridCache()


def cached(prefix: str, ttl: int | None = None, key_args: list[str] | None = None):
    """Decorator for caching function results."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_args:
                # Use only specified arguments for key generation
                key_kwargs = {k: v for k, v in kwargs.items() if k in key_args}
                cache_key = cache._generate_key(prefix, *args, **key_kwargs)
            else:
                cache_key = cache._generate_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key[:50]}...")
                return cached_result

            # Execute function and cache result
            try:
                result = await func(*args, **kwargs)
                await cache.set(cache_key, result, ttl=ttl)
                logger.debug(f"Cached result for key: {cache_key[:50]}...")
                return result
            except Exception as e:
                logger.error(f"Function execution error for {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


# Specialized cache decorators
@cached(prefix="transformation", ttl=CacheConfig.transformation_ttl)
async def cache_transformation(prompt: str, potency: int, technique: str, **kwargs):
    """Cache transformation results."""
    pass  # Decorator handles caching


@cached(prefix="api_response", ttl=CacheConfig.api_response_ttl)
async def cache_api_response(endpoint: str, params: dict | None = None):
    """Cache API responses."""
    pass  # Decorator handles caching


@cached(prefix="metrics", ttl=CacheConfig.metrics_ttl)
async def cache_metrics(metric_type: str, filters: dict | None = None):
    """Cache metrics data."""
    pass  # Decorator handles caching


class CacheWarmer:
    """Preloads frequently accessed data into cache."""

    def __init__(self, cache_instance: HybridCache):
        self.cache = cache_instance

    async def warm_technique_suites(self) -> None:
        """Warm cache with technique suites data."""
        try:
            from app.core.config import settings

            suites = settings.transformation.technique_suites

            for suite_name, suite_data in suites.items():
                key = f"technique_suite:{suite_name}"
                await self.cache.set(key, suite_data, ttl=3600)

            logger.info(f"Warmed cache with {len(suites)} technique suites")
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")

    async def warm_provider_configs(self) -> None:
        """Warm cache with provider configurations."""
        try:
            from app.core.config import settings

            provider_configs = {
                "google": {"model": settings.GOOGLE_MODEL, "available": True},
                "openai": {
                    "model": settings.OPENAI_MODEL,
                    "available": bool(settings.OPENAI_API_KEY),
                },
                "anthropic": {
                    "model": settings.ANTHROPIC_MODEL,
                    "available": bool(settings.ANTHROPIC_API_KEY),
                },
            }

            for provider, config in provider_configs.items():
                key = f"provider_config:{provider}"
                await self.cache.set(key, config, ttl=1800)

            logger.info("Warmed cache with provider configurations")
        except Exception as e:
            logger.error(f"Provider cache warming failed: {e}")


# Global cache warmer
cache_warmer = CacheWarmer(cache)


async def initialize_cache():
    """Initialize the cache system and warm essential data."""
    await cache.initialize()

    # MED-003 FIX: Wrap background tasks with error handling
    async def safe_warm_task(task_coro, task_name: str):
        """Wrapper to safely run cache warming tasks with error logging."""
        try:
            await task_coro
            logger.info(f"Cache warming task '{task_name}' completed successfully")
        except Exception as e:
            logger.error(f"Cache warming task '{task_name}' failed: {e}")

    # Warm essential data in background with error handling
    asyncio.create_task(safe_warm_task(cache_warmer.warm_technique_suites(), "technique_suites"))
    asyncio.create_task(safe_warm_task(cache_warmer.warm_provider_configs(), "provider_configs"))

    logger.info("Cache system initialized with data warming")


# Cache cleanup task
async def cleanup_expired_cache():
    """Periodic cleanup of expired cache entries."""
    try:
        # Memory cache already handles TTL, but we can force cleanup
        stats = await cache.get_stats()
        logger.info(f"Cache cleanup - Memory items: {stats['memory_cache']['total_items']}")

        # Clear memory cache if it's getting full
        if stats["memory_cache"]["total_items"] > stats["memory_cache"]["max_items"] * 0.8:
            await cache.memory_cache.clear()
            logger.info("Cleared memory cache to prevent overflow")

    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
