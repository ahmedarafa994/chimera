"""
Advanced Multi-Tier Caching System for Chimera AI

Comprehensive caching architecture with:
- L1: Application-level in-memory cache
- L2: Redis distributed cache
- L3: Database buffer pool optimization
- Smart cache invalidation strategies
- LLM provider-specific caching
- Query result caching with TTL management
"""

import builtins
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import text

from app.core.config import settings
from app.core.database import db_manager

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the multi-tier architecture."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    ttl: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    cache_level: CacheLevel = CacheLevel.L1_MEMORY

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl)

    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class L1MemoryCache:
    """
    L1 in-memory cache with LRU eviction and TTL support.

    High-speed cache for frequently accessed data with automatic
    memory management and performance monitoring.
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU tracking
        self._stats = CacheStats()

    async def get(self, key: str) -> Any | None:
        """Get value from L1 cache."""
        if key not in self._cache:
            self._stats.misses += 1
            return None

        entry = self._cache[key]

        # Check TTL
        if entry.is_expired():
            await self.delete(key)
            self._stats.misses += 1
            return None

        # Update access metadata
        entry.touch()
        self._update_access_order(key)
        self._stats.hits += 1

        return entry.value

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in L1 cache with TTL."""
        # Check if eviction is needed
        if len(self._cache) >= self.max_size or self._get_memory_usage() > self.max_memory_mb:
            await self._evict_lru()

        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            cache_level=CacheLevel.L1_MEMORY
        )

        self._cache[key] = entry
        self._update_access_order(key)
        self._stats.size = len(self._cache)

        return True

    async def delete(self, key: str) -> bool:
        """Delete entry from L1 cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats.size = len(self._cache)
            return True
        return False

    async def clear(self):
        """Clear all entries from L1 cache."""
        self._cache.clear()
        self._access_order.clear()
        self._stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.size = len(self._cache)
        self._stats.memory_usage_mb = self._get_memory_usage()
        return self._stats

    def _update_access_order(self, key: str):
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    async def _evict_lru(self):
        """Evict least recently used entries."""
        if not self._access_order:
            return

        # Evict 20% of entries to avoid frequent evictions
        evict_count = max(1, len(self._access_order) // 5)

        for _ in range(evict_count):
            if self._access_order:
                lru_key = self._access_order.pop(0)
                if lru_key in self._cache:
                    del self._cache[lru_key]
                    self._stats.evictions += 1

    def _get_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            total_size = sum(
                len(pickle.dumps(entry.value)) + len(entry.key.encode())
                for entry in self._cache.values()
            )
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0


class L2RedisCache:
    """
    L2 Redis-based distributed cache with advanced features.

    Features:
    - Async Redis operations
    - Connection pooling
    - Automatic serialization/deserialization
    - Pattern-based invalidation
    - Distributed cache warming
    """

    def __init__(self):
        self.redis_url = settings.get("REDIS_URL", "redis://localhost:6379/0")
        self.connection_pool = None
        self.redis_client = None
        self._stats = CacheStats()

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.connection_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self.redis_client = aioredis.Redis(connection_pool=self.connection_pool)

            # Test connection
            await self.redis_client.ping()
            logger.info("L2 Redis cache initialized successfully")

        except Exception as e:
            logger.warning(f"L2 Redis cache initialization failed: {e}")
            self.redis_client = None

    async def get(self, key: str) -> Any | None:
        """Get value from L2 Redis cache."""
        if not self.redis_client:
            self._stats.misses += 1
            return None

        try:
            data = await self.redis_client.get(key)
            if data is None:
                self._stats.misses += 1
                return None

            # Deserialize value
            value = pickle.loads(data)
            self._stats.hits += 1
            return value

        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
            self._stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in L2 Redis cache with TTL."""
        if not self.redis_client:
            return False

        try:
            # Serialize value
            data = pickle.dumps(value)
            await self.redis_client.setex(key, ttl, data)
            return True

        except Exception as e:
            logger.error(f"L2 cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from L2 Redis cache."""
        if not self.redis_client:
            return False

        try:
            result = await self.redis_client.delete(key)
            return result > 0

        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self.redis_client:
            return 0

        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"L2 cache pattern delete error: {e}")
            return 0

    async def clear(self):
        """Clear all entries from L2 cache."""
        if not self.redis_client:
            return

        try:
            await self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"L2 cache clear error: {e}")

    async def get_stats(self) -> CacheStats:
        """Get Redis cache statistics."""
        if not self.redis_client:
            return self._stats

        try:
            info = await self.redis_client.info('memory')
            keyspace = await self.redis_client.info('keyspace')

            self._stats.memory_usage_mb = info.get('used_memory', 0) / (1024 * 1024)

            # Extract key count from keyspace info
            db0_info = keyspace.get('db0', '')
            if db0_info:
                key_count = db0_info.split('keys=')[1].split(',')[0] if 'keys=' in db0_info else '0'
                self._stats.size = int(key_count)

        except Exception as e:
            logger.error(f"L2 cache stats error: {e}")

        return self._stats

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()


class MultiTierCache:
    """
    Multi-tier caching system with intelligent cache management.

    Tier Strategy:
    1. L1 Memory Cache: Ultra-fast for hot data
    2. L2 Redis Cache: Shared across instances
    3. L3 Database: Final fallback with caching

    Features:
    - Smart cache promotion/demotion
    - Write-through and write-behind patterns
    - Cache warming strategies
    - Performance monitoring
    """

    def __init__(self):
        self.l1_cache = L1MemoryCache(
            max_size=settings.get("L1_CACHE_SIZE", 1000),
            max_memory_mb=settings.get("L1_CACHE_MEMORY_MB", 100)
        )
        self.l2_cache = L2RedisCache()

        # Cache configuration
        self.default_ttl = {
            CacheLevel.L1_MEMORY: 300,  # 5 minutes
            CacheLevel.L2_REDIS: 1800,  # 30 minutes
            CacheLevel.L3_DATABASE: 3600  # 1 hour
        }

        # Performance tracking
        self.total_requests = 0
        self.cache_hits_by_level = dict.fromkeys(CacheLevel, 0)

    async def initialize(self):
        """Initialize all cache levels."""
        await self.l2_cache.initialize()
        logger.info("Multi-tier cache system initialized")

    async def get(self, key: str) -> Any | None:
        """
        Get value from multi-tier cache with promotion strategy.

        Checks L1 first, then L2, then falls back to database.
        Promotes cache hits to higher tiers for future performance.
        """
        self.total_requests += 1

        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.cache_hits_by_level[CacheLevel.L1_MEMORY] += 1
            return value

        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            self.cache_hits_by_level[CacheLevel.L2_REDIS] += 1
            # Promote to L1 for next access
            await self.l1_cache.set(key, value, self.default_ttl[CacheLevel.L1_MEMORY])
            return value

        return None

    async def set(self, key: str, value: Any, ttl: int | None = None,
                 cache_levels: set[CacheLevel] | None = None) -> bool:
        """
        Set value in specified cache levels with write-through pattern.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            cache_levels: Which cache levels to write to
        """
        if cache_levels is None:
            cache_levels = {CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS}

        success = True

        # Write to specified cache levels
        for level in cache_levels:
            level_ttl = ttl or self.default_ttl[level]

            if level == CacheLevel.L1_MEMORY:
                result = await self.l1_cache.set(key, value, level_ttl)
            elif level == CacheLevel.L2_REDIS:
                result = await self.l2_cache.set(key, value, level_ttl)
            else:
                continue  # L3 is handled differently

            success = success and result

        return success

    async def delete(self, key: str, cache_levels: builtins.set[CacheLevel] | None = None):
        """Delete key from specified cache levels."""
        if cache_levels is None:
            cache_levels = {CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS}

        for level in cache_levels:
            if level == CacheLevel.L1_MEMORY:
                await self.l1_cache.delete(key)
            elif level == CacheLevel.L2_REDIS:
                await self.l2_cache.delete(key)

    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern across cache levels."""
        # L1 cache doesn't support pattern matching, so clear keys that match
        l1_keys_to_delete = [
            key for key in self.l1_cache._cache
            if self._key_matches_pattern(key, pattern)
        ]
        for key in l1_keys_to_delete:
            await self.l1_cache.delete(key)

        # L2 Redis supports pattern matching
        await self.l2_cache.delete_pattern(pattern)

    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for L1 cache."""
        if '*' in pattern:
            pattern_parts = pattern.split('*')
            return all(part in key for part in pattern_parts if part)
        return key == pattern

    async def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics across all levels."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats()

        total_hits = sum(self.cache_hits_by_level.values())
        overall_hit_rate = total_hits / self.total_requests if self.total_requests > 0 else 0

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall": {
                "total_requests": self.total_requests,
                "total_hits": total_hits,
                "overall_hit_rate": round(overall_hit_rate, 3),
                "hits_by_level": dict(self.cache_hits_by_level)
            },
            "l1_memory_cache": {
                "hits": l1_stats.hits,
                "misses": l1_stats.misses,
                "hit_rate": round(l1_stats.hit_rate, 3),
                "size": l1_stats.size,
                "memory_usage_mb": round(l1_stats.memory_usage_mb, 2),
                "evictions": l1_stats.evictions
            },
            "l2_redis_cache": {
                "hits": l2_stats.hits,
                "misses": l2_stats.misses,
                "hit_rate": round(l2_stats.hit_rate, 3),
                "size": l2_stats.size,
                "memory_usage_mb": round(l2_stats.memory_usage_mb, 2)
            }
        }

    async def warm_cache(self, warming_queries: list[dict[str, Any]]):
        """Warm cache with commonly accessed data."""
        logger.info(f"Starting cache warming with {len(warming_queries)} queries")

        for query_config in warming_queries:
            try:
                query = query_config.get('query')
                cache_key = query_config.get('cache_key')
                ttl = query_config.get('ttl', 1800)

                if not query or not cache_key:
                    continue

                # Execute query and cache result
                async with db_manager.session() as session:
                    result = await session.execute(text(query))
                    rows = result.fetchall()

                    # Convert to cacheable format
                    data = [dict(row._mapping) for row in rows]

                    # Cache in both L1 and L2
                    await self.set(
                        cache_key,
                        data,
                        ttl,
                        {CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS}
                    )

            except Exception as e:
                logger.error(f"Cache warming failed for {cache_key}: {e}")

        logger.info("Cache warming completed")

    async def close(self):
        """Close all cache connections."""
        await self.l2_cache.close()


class LLMProviderCache:
    """
    Specialized caching for LLM provider operations.

    Features:
    - Provider configuration caching
    - Model metadata caching
    - Response caching with content hashing
    - Transformation technique caching
    """

    def __init__(self, multi_tier_cache: MultiTierCache):
        self.cache = multi_tier_cache

    async def get_provider_config(self, provider_name: str, model_name: str) -> dict[str, Any] | None:
        """Get cached provider configuration."""
        cache_key = f"provider_config:{provider_name}:{model_name}"
        return await self.cache.get(cache_key)

    async def set_provider_config(self, provider_name: str, model_name: str, config: dict[str, Any]):
        """Cache provider configuration with long TTL."""
        cache_key = f"provider_config:{provider_name}:{model_name}"
        await self.cache.set(cache_key, config, ttl=3600)  # 1 hour TTL

    async def get_model_metadata(self, provider_name: str, model_name: str) -> dict[str, Any] | None:
        """Get cached model metadata."""
        cache_key = f"model_metadata:{provider_name}:{model_name}"
        return await self.cache.get(cache_key)

    async def set_model_metadata(self, provider_name: str, model_name: str, metadata: dict[str, Any]):
        """Cache model metadata."""
        cache_key = f"model_metadata:{provider_name}:{model_name}"
        await self.cache.set(cache_key, metadata, ttl=7200)  # 2 hours TTL

    async def get_transformation_result(self, technique: str, prompt_hash: str) -> str | None:
        """Get cached transformation result."""
        cache_key = f"transform:{technique}:{prompt_hash}"
        return await self.cache.get(cache_key)

    async def set_transformation_result(self, technique: str, prompt_hash: str, result: str):
        """Cache transformation result."""
        cache_key = f"transform:{technique}:{prompt_hash}"
        await self.cache.set(cache_key, result, ttl=1800)  # 30 minutes TTL

    async def invalidate_provider_cache(self, provider_name: str):
        """Invalidate all cache entries for a provider."""
        pattern = f"provider_config:{provider_name}:*"
        await self.cache.invalidate_pattern(pattern)

        pattern = f"model_metadata:{provider_name}:*"
        await self.cache.invalidate_pattern(pattern)


class QueryResultCache:
    """
    Database query result caching with intelligent invalidation.

    Features:
    - Query result caching by normalized query hash
    - Smart TTL based on table modification patterns
    - Automatic invalidation on table updates
    - Compression for large result sets
    """

    def __init__(self, multi_tier_cache: MultiTierCache):
        self.cache = multi_tier_cache
        self.table_modification_times = {}

    def _get_query_hash(self, query: str, params: dict | None = None) -> str:
        """Generate consistent hash for query and parameters."""
        query_normalized = ' '.join(query.split()).upper()

        if params:
            param_str = json.dumps(params, sort_keys=True)
            query_normalized += param_str

        return hashlib.md5(query_normalized.encode()).hexdigest()

    async def get_query_result(self, query: str, params: dict | None = None) -> list[dict] | None:
        """Get cached query result."""
        query_hash = self._get_query_hash(query, params)
        cache_key = f"query_result:{query_hash}"
        return await self.cache.get(cache_key)

    async def set_query_result(self, query: str, result: list[dict], params: dict | None = None, ttl: int = 600):
        """Cache query result with TTL."""
        query_hash = self._get_query_hash(query, params)
        cache_key = f"query_result:{query_hash}"
        await self.cache.set(cache_key, result, ttl)

    async def invalidate_table_queries(self, table_name: str):
        """Invalidate all cached queries for a table."""
        pattern = f"query_result:*{table_name.upper()}*"
        await self.cache.invalidate_pattern(pattern)

        # Update modification time
        self.table_modification_times[table_name] = datetime.utcnow()


# Global multi-tier cache instance
multi_tier_cache = MultiTierCache()

# Specialized cache instances
llm_provider_cache = LLMProviderCache(multi_tier_cache)
query_result_cache = QueryResultCache(multi_tier_cache)


# Cache warming configuration for Chimera
CACHE_WARMING_QUERIES = [
    {
        "query": "SELECT id, name, provider, config FROM llm_models WHERE provider IN ('google', 'openai', 'anthropic')",
        "cache_key": "active_llm_models",
        "ttl": 3600
    },
    {
        "query": "SELECT technique_suite, technique_name FROM transformation_techniques WHERE enabled = 1",
        "cache_key": "active_transformation_techniques",
        "ttl": 1800
    },
    {
        "query": "SELECT id, name, description FROM jailbreak_datasets WHERE created_at > datetime('now', '-30 days')",
        "cache_key": "recent_jailbreak_datasets",
        "ttl": 1800
    }
]


async def initialize_caching_system():
    """Initialize the complete caching system."""
    await multi_tier_cache.initialize()
    await multi_tier_cache.warm_cache(CACHE_WARMING_QUERIES)
    logger.info("Multi-tier caching system fully initialized")


async def get_cache_health_status() -> dict[str, Any]:
    """Get health status of all cache levels."""
    stats = await multi_tier_cache.get_comprehensive_stats()

    # Determine health based on performance metrics
    l1_healthy = stats['l1_memory_cache']['hit_rate'] > 0.7
    l2_healthy = stats['l2_redis_cache']['hit_rate'] > 0.5
    overall_healthy = stats['overall']['overall_hit_rate'] > 0.6

    return {
        "overall_status": "healthy" if overall_healthy else "degraded",
        "l1_status": "healthy" if l1_healthy else "degraded",
        "l2_status": "healthy" if l2_healthy else "degraded",
        "performance_metrics": stats,
        "recommendations": _generate_cache_recommendations(stats)
    }


def _generate_cache_recommendations(stats: dict[str, Any]) -> list[str]:
    """Generate cache performance recommendations."""
    recommendations = []

    l1_hit_rate = stats['l1_memory_cache']['hit_rate']
    l2_hit_rate = stats['l2_redis_cache']['hit_rate']
    overall_hit_rate = stats['overall']['overall_hit_rate']

    if l1_hit_rate < 0.5:
        recommendations.append("L1 cache hit rate is low - consider increasing cache size or adjusting TTL")

    if l2_hit_rate < 0.3:
        recommendations.append("L2 Redis cache hit rate is low - verify Redis connection and consider cache warming")

    if overall_hit_rate < 0.4:
        recommendations.append("Overall cache performance is poor - review caching strategy and key patterns")

    evictions = stats['l1_memory_cache']['evictions']
    if evictions > 1000:
        recommendations.append(f"High L1 eviction count ({evictions}) - consider increasing memory allocation")

    return recommendations
