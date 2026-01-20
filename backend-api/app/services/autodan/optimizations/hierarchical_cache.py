"""Advanced Caching System for AutoDAN with Similarity Matching.

This module provides a sophisticated caching system with:
- Multi-tier caching (L1: Memory, L2: Redis, L3: Disk)
- FAISS-based similarity search for semantic cache hits
- Bloom filters for efficient negative lookups
- Intelligent cache eviction policies
- Compression and memory mapping for large data
- Comprehensive cache analytics and monitoring
"""

import asyncio
import hashlib
import logging
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, similarity search will be disabled")

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, L2 cache will be disabled")

try:
    from pybloom_live import BloomFilter

    BLOOM_AVAILABLE = True
except ImportError:
    BLOOM_AVAILABLE = False
    logger.warning("Bloom filter not available, will use set-based negative cache")


class CacheLevel(Enum):
    """Cache tier levels."""

    L1 = "memory"  # In-memory LRU cache
    L2 = "redis"  # Redis cache
    L3 = "disk"  # Persistent disk cache


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    RANDOM = "random"  # Random eviction


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: float | None = None
    size_bytes: int = 0
    embedding: np.ndarray | None = None

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    similarity_hits: int = 0
    bloom_filter_saves: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def similarity_hit_rate(self) -> float:
        """Calculate similarity-based hit rate."""
        return self.similarity_hits / self.cache_hits if self.cache_hits > 0 else 0.0


class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Get entry by key."""

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set entry."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete entry."""

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all entries."""

    @abstractmethod
    async def size(self) -> int:
        """Get number of entries."""


class MemoryCache(CacheBackend):
    """In-memory LRU cache backend."""

    def __init__(
        self,
        max_size: int = 1000,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ) -> None:
        self.max_size = max_size
        self.max_size_bytes = max_size_bytes
        self.eviction_policy = eviction_policy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CacheEntry | None:
        async with self._lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]

            if entry.is_expired:
                await self._remove_entry(key)
                return None

            entry.touch()

            # Move to end for LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self.cache.move_to_end(key)

            return entry

    async def set(self, key: str, entry: CacheEntry) -> bool:
        async with self._lock:
            # Remove existing entry if it exists
            if key in self.cache:
                await self._remove_entry(key)

            # Evict entries if necessary
            while (
                len(self.cache) >= self.max_size
                or self.current_size_bytes + entry.size_bytes > self.max_size_bytes
            ):
                if not await self._evict_one():
                    return False

            self.cache[key] = entry
            self.current_size_bytes += entry.size_bytes
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return await self._remove_entry(key)

    async def clear(self) -> bool:
        async with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
            return True

    async def size(self) -> int:
        return len(self.cache)

    async def _remove_entry(self, key: str) -> bool:
        """Remove entry and update size."""
        if key not in self.cache:
            return False

        entry = self.cache.pop(key)
        self.current_size_bytes -= entry.size_bytes
        return True

    async def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self.cache:
            return False

        if self.eviction_policy == EvictionPolicy.LRU:
            key = next(iter(self.cache))
        elif self.eviction_policy == EvictionPolicy.LFU:
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Find expired entries first
            expired = [k for k, v in self.cache.items() if v.is_expired]
            if expired:
                key = expired[0]
            else:
                key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        else:  # RANDOM
            key = next(iter(self.cache))

        return await self._remove_entry(key)


class RedisCache(CacheBackend):
    """Redis-based cache backend."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "autodan_cache:",
        compression: bool = True,
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.compression = compression
        self.redis: aioredis.Redis | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            msg = "Redis not available"
            raise RuntimeError(msg)

        self.redis = aioredis.from_url(self.redis_url)

    async def get(self, key: str) -> CacheEntry | None:
        if not self.redis:
            return None

        try:
            data = await self.redis.get(f"{self.key_prefix}{key}")
            if data is None:
                return None

            # Decompress and deserialize
            if self.compression:
                data = zlib.decompress(data)

            entry = pickle.loads(data)

            # Check TTL
            if entry.is_expired:
                await self.delete(key)
                return None

            entry.touch()
            return entry

        except Exception as e:
            logger.exception(f"Redis get error: {e}")
            return None

    async def set(self, key: str, entry: CacheEntry) -> bool:
        if not self.redis:
            return False

        try:
            # Serialize and compress
            data = pickle.dumps(entry)
            if self.compression:
                data = zlib.compress(data)

            # Set with TTL if specified
            ttl = entry.ttl_seconds if entry.ttl_seconds else None
            await self.redis.set(f"{self.key_prefix}{key}", data, ex=ttl)
            return True

        except Exception as e:
            logger.exception(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        if not self.redis:
            return False

        try:
            result = await self.redis.delete(f"{self.key_prefix}{key}")
            return result > 0
        except Exception as e:
            logger.exception(f"Redis delete error: {e}")
            return False

    async def clear(self) -> bool:
        if not self.redis:
            return False

        try:
            keys = await self.redis.keys(f"{self.key_prefix}*")
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception as e:
            logger.exception(f"Redis clear error: {e}")
            return False

    async def size(self) -> int:
        if not self.redis:
            return 0

        try:
            keys = await self.redis.keys(f"{self.key_prefix}*")
            return len(keys)
        except Exception as e:
            logger.exception(f"Redis size error: {e}")
            return 0

    async def cleanup(self) -> None:
        """Cleanup Redis connection."""
        if self.redis:
            await self.redis.close()


class DiskCache(CacheBackend):
    """Disk-based persistent cache."""

    def __init__(
        self,
        cache_dir: str | Path = "cache/autodan",
        max_size_mb: int = 1000,
        compression: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression = compression
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _get_file_path(self, key: str) -> Path:
        """Get file path for key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"

    async def get(self, key: str) -> CacheEntry | None:
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return None

        try:
            async with self._lock:
                with open(file_path, "rb") as f:
                    data = f.read()

                if self.compression:
                    data = zlib.decompress(data)

                entry = pickle.loads(data)

                if entry.is_expired:
                    file_path.unlink(missing_ok=True)
                    return None

                entry.touch()
                return entry

        except Exception as e:
            logger.exception(f"Disk cache get error: {e}")
            return None

    async def set(self, key: str, entry: CacheEntry) -> bool:
        file_path = self._get_file_path(key)

        try:
            async with self._lock:
                # Check disk space
                if await self._get_cache_size() > self.max_size_bytes:
                    await self._cleanup_old_files()

                data = pickle.dumps(entry)
                if self.compression:
                    data = zlib.compress(data)

                with open(file_path, "wb") as f:
                    f.write(data)

                return True

        except Exception as e:
            logger.exception(f"Disk cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        file_path = self._get_file_path(key)

        try:
            file_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.exception(f"Disk cache delete error: {e}")
            return False

    async def clear(self) -> bool:
        try:
            async with self._lock:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink(missing_ok=True)
                return True
        except Exception as e:
            logger.exception(f"Disk cache clear error: {e}")
            return False

    async def size(self) -> int:
        try:
            return len(list(self.cache_dir.glob("*.cache")))
        except Exception as e:
            logger.exception(f"Disk cache size error: {e}")
            return 0

    async def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for file_path in self.cache_dir.glob("*.cache"):
            try:
                total_size += file_path.stat().st_size
            except Exception:
                continue
        return total_size

    async def _cleanup_old_files(self) -> None:
        """Remove oldest files to free space."""
        files = list(self.cache_dir.glob("*.cache"))
        files.sort(key=lambda f: f.stat().st_mtime)

        # Remove oldest 20% of files
        num_to_remove = max(1, len(files) // 5)
        for file_path in files[:num_to_remove]:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                continue


class SimilarityIndex:
    """FAISS-based similarity index for semantic cache hits."""

    def __init__(self, dimension: int = 768, index_type: str = "IVF") -> None:
        self.dimension = dimension
        self.index_type = index_type
        self.index: faiss.Index | None = None
        self.key_to_id: dict[str, int] = {}
        self.id_to_key: dict[int, str] = {}
        self.embeddings: list[np.ndarray] = []
        self.next_id = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, similarity search disabled")
            return

        async with self._lock:
            if self.index_type == "IVF":
                # IVF index for large datasets
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                # Train with dummy data
                dummy_data = np.random.random((1000, self.dimension)).astype("float32")
                self.index.train(dummy_data)
            else:
                # Flat index for smaller datasets
                self.index = faiss.IndexFlatIP(self.dimension)

    async def add_embedding(self, key: str, embedding: np.ndarray) -> None:
        """Add embedding to index."""
        if not self.index or not FAISS_AVAILABLE:
            return

        async with self._lock:
            # Normalize embedding
            normalized_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            self.key_to_id[key] = self.next_id
            self.id_to_key[self.next_id] = key
            self.embeddings.append(normalized_embedding)

            # Add to FAISS index
            self.index.add(normalized_embedding.reshape(1, -1).astype("float32"))
            self.next_id += 1

    async def search_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.9,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings."""
        if not self.index or not FAISS_AVAILABLE or self.next_id == 0:
            return []

        async with self._lock:
            # Normalize query embedding
            normalized_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            # Search
            similarities, indices = self.index.search(
                normalized_embedding.reshape(1, -1).astype("float32"),
                min(k, self.next_id),
            )

            # Filter by threshold and return results
            results = []
            for sim, idx in zip(similarities[0], indices[0], strict=False):
                if idx != -1 and sim >= threshold:
                    key = self.id_to_key.get(idx)
                    if key:
                        results.append((key, float(sim)))

            return results

    async def remove_embedding(self, key: str) -> None:
        """Remove embedding (note: FAISS doesn't support removal, so we mark as deleted)."""
        if key in self.key_to_id:
            async with self._lock:
                # In a production system, you'd need to rebuild the index periodically
                # or use a different approach for deletions
                del self.key_to_id[key]
                if key in self.id_to_key.values():
                    id_to_remove = next(k for k, v in self.id_to_key.items() if v == key)
                    del self.id_to_key[id_to_remove]


class HierarchicalCache:
    """Multi-tier hierarchical cache with similarity matching.

    Features:
    - L1: Fast in-memory cache
    - L2: Redis for distributed caching
    - L3: Persistent disk cache
    - FAISS-based similarity search
    - Bloom filters for negative lookups
    - Comprehensive metrics and monitoring
    """

    def __init__(
        self,
        memory_config: dict[str, Any] | None = None,
        redis_config: dict[str, Any] | None = None,
        disk_config: dict[str, Any] | None = None,
        similarity_threshold: float = 0.95,
        enable_similarity: bool = True,
        enable_bloom_filter: bool = True,
        embedding_dimension: int = 768,
    ) -> None:
        # Initialize backends
        self.l1_cache = MemoryCache(**(memory_config or {}))
        self.l2_cache = RedisCache(**(redis_config or {})) if REDIS_AVAILABLE else None
        self.l3_cache = DiskCache(**(disk_config or {}))

        # Similarity search
        self.similarity_threshold = similarity_threshold
        self.enable_similarity = enable_similarity and FAISS_AVAILABLE
        self.similarity_index = (
            SimilarityIndex(embedding_dimension) if self.enable_similarity else None
        )

        # Bloom filter for negative lookups
        self.enable_bloom_filter = enable_bloom_filter and BLOOM_AVAILABLE
        self.bloom_filter = (
            BloomFilter(capacity=100000, error_rate=0.01) if self.enable_bloom_filter else set()
        )

        # Statistics
        self.stats = CacheStats()
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize all cache components."""
        if self.l2_cache:
            await self.l2_cache.initialize()

        if self.similarity_index:
            await self.similarity_index.initialize()

        logger.info("Hierarchical cache initialized")

    async def get(self, key: str, embedding: np.ndarray | None = None) -> Any | None:
        """Get value from cache with similarity matching."""
        start_time = time.time()
        self.stats.total_requests += 1

        try:
            # Check bloom filter first for negative lookups
            if self.enable_bloom_filter:
                if isinstance(self.bloom_filter, BloomFilter):
                    if key not in self.bloom_filter:
                        self.stats.bloom_filter_saves += 1
                        self.stats.cache_misses += 1
                        return None
                elif key not in self.bloom_filter:
                    self.stats.bloom_filter_saves += 1
                    self.stats.cache_misses += 1
                    return None

            # Try direct lookup in all tiers
            entry = await self._get_from_tier(CacheLevel.L1, key)
            if not entry and self.l2_cache:
                entry = await self._get_from_tier(CacheLevel.L2, key)
            if not entry:
                entry = await self._get_from_tier(CacheLevel.L3, key)

            # If found, promote to higher tiers
            if entry:
                await self._promote_entry(key, entry)
                self.stats.cache_hits += 1
                return entry.value

            # Try similarity search if embedding provided
            if self.enable_similarity and embedding is not None and self.similarity_index:
                similar_keys = await self.similarity_index.search_similar(
                    embedding,
                    k=5,
                    threshold=self.similarity_threshold,
                )

                for similar_key, similarity in similar_keys:
                    similar_entry = await self._get_from_any_tier(similar_key)
                    if similar_entry:
                        # Cache the result under the original key
                        await self.set(key, similar_entry.value, embedding=embedding)
                        self.stats.cache_hits += 1
                        self.stats.similarity_hits += 1
                        logger.debug(f"Similarity hit: {similarity:.3f}")
                        return similar_entry.value

            self.stats.cache_misses += 1
            return None

        except Exception as e:
            self.stats.errors += 1
            logger.exception(f"Cache get error: {e}")
            return None
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"Cache get took {elapsed:.3f}s")

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
        embedding: np.ndarray | None = None,
    ) -> bool:
        """Set value in cache."""
        try:
            # Calculate size
            size_bytes = len(pickle.dumps(value))

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                embedding=embedding,
            )

            # Store in all tiers
            success = True
            success &= await self.l1_cache.set(key, entry)

            if self.l2_cache:
                success &= await self.l2_cache.set(key, entry)

            success &= await self.l3_cache.set(key, entry)

            # Add to similarity index
            if self.enable_similarity and embedding is not None and self.similarity_index:
                await self.similarity_index.add_embedding(key, embedding)

            # Add to bloom filter
            if self.enable_bloom_filter:
                if isinstance(self.bloom_filter, BloomFilter):
                    self.bloom_filter.add(key)
                else:
                    self.bloom_filter.add(key)

            # Update stats
            self.stats.total_size_bytes += size_bytes

            return success

        except Exception as e:
            self.stats.errors += 1
            logger.exception(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete from all cache tiers."""
        try:
            success = True
            success &= await self.l1_cache.delete(key)

            if self.l2_cache:
                success &= await self.l2_cache.delete(key)

            success &= await self.l3_cache.delete(key)

            # Remove from similarity index
            if self.enable_similarity and self.similarity_index:
                await self.similarity_index.remove_embedding(key)

            return success

        except Exception as e:
            self.stats.errors += 1
            logger.exception(f"Cache delete error: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache tiers."""
        try:
            success = True
            success &= await self.l1_cache.clear()

            if self.l2_cache:
                success &= await self.l2_cache.clear()

            success &= await self.l3_cache.clear()

            # Reset bloom filter
            if self.enable_bloom_filter:
                if isinstance(self.bloom_filter, BloomFilter):
                    self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.01)
                else:
                    self.bloom_filter.clear()

            # Reset stats
            self.stats = CacheStats()

            return success

        except Exception as e:
            self.stats.errors += 1
            logger.exception(f"Cache clear error: {e}")
            return False

    async def _get_from_tier(self, tier: CacheLevel, key: str) -> CacheEntry | None:
        """Get entry from specific tier."""
        if tier == CacheLevel.L1:
            return await self.l1_cache.get(key)
        if tier == CacheLevel.L2 and self.l2_cache:
            return await self.l2_cache.get(key)
        if tier == CacheLevel.L3:
            return await self.l3_cache.get(key)
        return None

    async def _get_from_any_tier(self, key: str) -> CacheEntry | None:
        """Get entry from any available tier."""
        entry = await self._get_from_tier(CacheLevel.L1, key)
        if not entry and self.l2_cache:
            entry = await self._get_from_tier(CacheLevel.L2, key)
        if not entry:
            entry = await self._get_from_tier(CacheLevel.L3, key)
        return entry

    async def _promote_entry(self, key: str, entry: CacheEntry) -> None:
        """Promote entry to higher cache tiers."""
        # Always ensure L1 has the entry
        await self.l1_cache.set(key, entry)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "hit_rate": self.stats.hit_rate,
            "similarity_hit_rate": self.stats.similarity_hit_rate,
            "total_requests": self.stats.total_requests,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "similarity_hits": self.stats.similarity_hits,
            "bloom_filter_saves": self.stats.bloom_filter_saves,
            "errors": self.stats.errors,
            "total_size_mb": self.stats.total_size_bytes / (1024 * 1024),
            "l1_size": len(self.l1_cache.cache) if hasattr(self.l1_cache, "cache") else 0,
            "l2_available": self.l2_cache is not None,
            "l3_size": (
                len(list(self.l3_cache.cache_dir.glob("*.cache")))
                if hasattr(self.l3_cache, "cache_dir")
                else 0
            ),
            "similarity_enabled": self.enable_similarity,
            "bloom_filter_enabled": self.enable_bloom_filter,
        }

    async def cleanup(self) -> None:
        """Cleanup all cache resources."""
        if self.l2_cache:
            await self.l2_cache.cleanup()

        logger.info("Hierarchical cache cleaned up")


# Factory function for easy cache creation
def create_hierarchical_cache(**kwargs) -> HierarchicalCache:
    """Create a hierarchical cache with sensible defaults."""
    return HierarchicalCache(**kwargs)


# Context manager for managed cache lifecycle
@asynccontextmanager
async def managed_hierarchical_cache(**kwargs):
    """Context manager for automatic cache lifecycle management."""
    cache = HierarchicalCache(**kwargs)
    try:
        await cache.initialize()
        yield cache
    finally:
        await cache.cleanup()
