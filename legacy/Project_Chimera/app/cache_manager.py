"""
Advanced caching manager with Redis/memcached support and in-memory fallback.
Provides intelligent caching for transformation results and API responses.
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    data: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    size_bytes: int = 0


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if time.time() - entry.timestamp > entry.ttl:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.access_count += 1
            self.hits += 1
            return entry.data

    def put(self, key: str, data: Any, ttl: float = 3600):
        """Put item in cache."""
        with self.lock:
            # Calculate size
            size_bytes = len(pickle.dumps(data))

            # Remove old entries if needed
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            entry = CacheEntry(data=data, timestamp=time.time(), ttl=ttl, size_bytes=size_bytes)

            self.cache[key] = entry
            self.cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            total_size = sum(entry.size_bytes for entry in self.cache.values())

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
            }


class CacheManager:
    """
    Advanced cache manager with multiple backends and intelligent caching strategies.
    """

    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.memory_cache = LRUCache(max_size=1000)
        self.redis_client = None
        self.compression_enabled = os.getenv("CACHE_COMPRESSION", "false").lower() == "true"

        # Initialize Redis if available
        self._init_redis()

        # Cache statistics
        self.stats = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "memory_hits": 0,
            "memory_misses": 0,
        }

    def _init_redis(self):
        """Initialize Redis client if available."""
        try:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                import redis

                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=False,  # Handle binary data
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
        except ImportError:
            logger.warning("Redis not available, using memory-only cache")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}, using memory-only cache")

    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key from arguments."""
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"

    def get(self, key: str) -> Any | None:
        """Get value from cache (tries memory first, then Redis)."""
        self.stats["requests"] += 1

        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            self.stats["hits"] += 1
            self.stats["memory_hits"] += 1
            return value

        self.stats["memory_misses"] += 1

        # Try Redis if available
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    # Decompress if needed
                    if self.compression_enabled:
                        import gzip

                        cached_data = gzip.decompress(cached_data)

                    value = pickle.loads(cached_data)

                    # Store in memory cache for faster access
                    self.memory_cache.put(key, value, self.default_ttl)

                    self.stats["hits"] += 1
                    self.stats["redis_hits"] += 1
                    return value

                self.stats["redis_misses"] += 1
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        self.stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache (both memory and Redis)."""
        ttl = ttl or self.default_ttl

        try:
            # Store in memory cache
            self.memory_cache.put(key, value, ttl)

            # Store in Redis if available
            if self.redis_client:
                serialized_data = pickle.dumps(value)

                # Compress if enabled and data is large enough
                if self.compression_enabled and len(serialized_data) > 1024:
                    import gzip

                    serialized_data = gzip.compress(serialized_data)

                self.redis_client.setex(key, ttl, serialized_data)

            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from all cache backends."""
        try:
            # Delete from memory
            self.memory_cache.delete(key)

            # Delete from Redis
            if self.redis_client:
                self.redis_client.delete(key)

            return True

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear(self):
        """Clear all cache backends."""
        try:
            self.memory_cache.clear()

            if self.redis_client:
                self.redis_client.flushdb()

            logger.info("All caches cleared successfully")

        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.stats["requests"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        memory_stats = self.memory_cache.get_stats()

        stats = {
            "overall": {
                "total_requests": total_requests,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
            },
            "memory": memory_stats,
            "redis": {
                "hits": self.stats["redis_hits"],
                "misses": self.stats["redis_misses"],
                "hit_rate": self.stats["redis_hits"]
                / (self.stats["redis_hits"] + self.stats["redis_misses"])
                if (self.stats["redis_hits"] + self.stats["redis_misses"]) > 0
                else 0,
            }
            if self.redis_client
            else {"status": "disabled"},
            "compression_enabled": self.compression_enabled,
        }

        return stats

    def cache_function_result(self, prefix: str, ttl: int | None = None):
        """Decorator to cache function results."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(prefix, *args, **kwargs)

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def cache_async_function_result(self, prefix: str, ttl: int | None = None):
        """Decorator to cache async function results."""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(prefix, *args, **kwargs)

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = await func(*args, **kwargs)
                self.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator


# Global cache manager instance
cache_manager = CacheManager()


# Convenient decorators
def cached(prefix: str, ttl: int = 3600):
    """Convenient decorator for caching function results."""
    return cache_manager.cache_function_result(prefix, ttl)


def async_cached(prefix: str, ttl: int = 3600):
    """Convenient decorator for caching async function results."""
    return cache_manager.cache_async_function_result(prefix, ttl)


# Cache warming utilities
class CacheWarmer:
    """Utilities for warming up the cache with frequently used data."""

    @staticmethod
    def warm_technique_suites():
        """Warm cache with popular technique suites."""
        from app.technique_manager import technique_manager

        popular_suites = ["universal_bypass", "full_spectrum", "mega_chimera", "dan_persona"]

        for suite_name in popular_suites:
            try:
                cache_key = f"technique_suite:{suite_name}"
                if cache_manager.get(cache_key) is None:
                    suite = technique_manager.get_suite(suite_name)
                    cache_manager.set(cache_key, suite, ttl=7200)  # 2 hours
                    logger.info(f"Warmed cache for technique suite: {suite_name}")
            except Exception as e:
                logger.warning(f"Failed to warm cache for {suite_name}: {e}")

    @staticmethod
    def warm_response_templates():
        """Warm cache with common response templates."""
        common_responses = {
            "health_check": {"status": "healthy"},
            "providers_list": {"providers": [], "count": 0, "default": "google"},
            "techniques_list": {"techniques": [], "count": 0},
        }

        for name, response in common_responses.items():
            try:
                cache_key = f"response_template:{name}"
                cache_manager.set(cache_key, response, ttl=1800)  # 30 minutes
                logger.info(f"Warmed cache for response template: {name}")
            except Exception as e:
                logger.warning(f"Failed to warm cache for {name}: {e}")


def initialize_cache():
    """Initialize cache and warm common data."""
    logger.info("Initializing cache system...")

    # Warm common data
    warmer = CacheWarmer()
    warmer.warm_technique_suites()
    warmer.warm_response_templates()

    # Log cache statistics
    stats = cache_manager.get_stats()
    logger.info(f"Cache initialized: {json.dumps(stats, indent=2)}")
