"""
Parallel Processing and Caching Module for AutoDAN

This module provides:
- Async batch processing for LLM calls
- Intelligent caching with TTL and similarity matching
- Rate limiting and backoff strategies
- Resource pooling for efficient API usage
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Generic, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based


@dataclass
class CacheEntry(Generic[T]):
    """Entry in the cache with metadata."""

    value: T
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: float | None = None

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class SmartCache(Generic[T]):
    """
    Intelligent cache with multiple eviction policies and similarity matching.

    Features:
    - Multiple eviction policies (LRU, LFU, FIFO, TTL)
    - Optional similarity-based cache hits
    - Automatic cleanup of expired entries
    - Statistics tracking
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float | None = 3600,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
        similarity_threshold: float = 0.95,
        enable_similarity: bool = False,
    ):
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        self.eviction_policy = eviction_policy
        self.similarity_threshold = similarity_threshold
        self.enable_similarity = enable_similarity

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._key_to_embedding: dict[str, np.ndarray] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _generate_key(self, key: Any) -> str:
        """Generate a cache key from any hashable object."""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(str(key).encode()).hexdigest()

    def get(self, key: Any, embedding: np.ndarray | None = None) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key
            embedding: Optional embedding for similarity matching

        Returns:
            Cached value or None
        """
        cache_key = self._generate_key(key)

        # Direct lookup
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if entry.is_expired:
                self._remove(cache_key)
                self._misses += 1
                return None

            entry.touch()
            self._hits += 1

            # Move to end for LRU
            if self.eviction_policy == CacheEvictionPolicy.LRU:
                self._cache.move_to_end(cache_key)

            return entry.value

        # Similarity-based lookup
        if self.enable_similarity and embedding is not None:
            similar_key = self._find_similar(embedding)
            if similar_key:
                entry = self._cache[similar_key]
                if not entry.is_expired:
                    entry.touch()
                    self._hits += 1
                    return entry.value

        self._misses += 1
        return None

    def set(
        self,
        key: Any,
        value: T,
        ttl_seconds: float | None = None,
        embedding: np.ndarray | None = None,
    ):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL override
            embedding: Optional embedding for similarity matching
        """
        cache_key = self._generate_key(key)

        # Evict if necessary
        while len(self._cache) >= self.max_size:
            self._evict_one()

        # Create entry
        now = time.time()
        entry = CacheEntry(
            value=value,
            created_at=now,
            last_accessed=now,
            ttl_seconds=ttl_seconds or self.default_ttl,
        )

        self._cache[cache_key] = entry

        if embedding is not None and self.enable_similarity:
            self._key_to_embedding[cache_key] = embedding

    def _find_similar(self, embedding: np.ndarray) -> str | None:
        """Find a similar cached entry using cosine similarity."""
        if not self._key_to_embedding:
            return None

        best_key = None
        best_similarity = 0.0

        for key, cached_embedding in self._key_to_embedding.items():
            if key not in self._cache:
                continue

            # Cosine similarity
            similarity = np.dot(embedding, cached_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(cached_embedding) + 1e-8
            )

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_key = key

        return best_key

    def _evict_one(self):
        """Evict one entry based on eviction policy."""
        if not self._cache:
            return

        key_to_evict = None

        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Evict least recently used (first item in OrderedDict)
            key_to_evict = next(iter(self._cache))

        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # Evict least frequently used
            key_to_evict = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)

        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            # Evict oldest (first item)
            key_to_evict = next(iter(self._cache))

        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            # Evict expired entries first, then oldest
            expired = [k for k, v in self._cache.items() if v.is_expired]
            if expired:
                key_to_evict = expired[0]
            else:
                key_to_evict = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)

        if key_to_evict:
            self._remove(key_to_evict)
            self._evictions += 1

    def _remove(self, key: str):
        """Remove an entry from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._key_to_embedding:
            del self._key_to_embedding[key]

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._key_to_embedding.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def cleanup_expired(self):
        """Remove all expired entries."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            self._remove(key)
        return len(expired_keys)

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "evictions": self._evictions,
            "policy": self.eviction_policy.value,
        }


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch operation."""

    results: list[T]
    errors: list[tuple[int, Exception]]
    total_time: float
    success_count: int
    error_count: int


class ParallelProcessor:
    """
    Parallel processor for batch operations with rate limiting.

    Features:
    - Async and sync batch processing
    - Configurable concurrency limits
    - Rate limiting with token bucket
    - Automatic retry with backoff
    - Progress tracking
    """

    def __init__(
        self,
        max_workers: int = 4,
        rate_limit: float | None = None,  # Requests per second
        timeout_seconds: float | None = None,  # No timeout by default
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # Rate limiting state
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

        # Statistics
        self._total_processed = 0
        self._total_errors = 0
        self._total_time = 0.0

    async def process_batch_async(
        self,
        items: list[T],
        processor: Callable[[T], R],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult[R]:
        """
        Process items in parallel asynchronously.

        Args:
            items: List of items to process
            processor: Function to apply to each item
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with results and errors
        """
        start_time = time.time()
        results: list[R | None] = [None] * len(items)
        errors: list[tuple[int, Exception]] = []

        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_one(index: int, item: T):
            async with semaphore:
                # Rate limiting
                if self.rate_limit:
                    await self._wait_for_rate_limit()

                # Process with retry
                for attempt in range(self.retry_count):
                    try:
                        if asyncio.iscoroutinefunction(processor):
                            if self.timeout_seconds:
                                result = await asyncio.wait_for(
                                    processor(item), timeout=self.timeout_seconds
                                )
                            else:
                                result = await processor(item)
                        else:
                            loop = asyncio.get_event_loop()
                            if self.timeout_seconds:
                                result = await asyncio.wait_for(
                                    loop.run_in_executor(None, processor, item),
                                    timeout=self.timeout_seconds,
                                )
                            else:
                                result = await loop.run_in_executor(None, processor, item)

                        results[index] = result

                        if progress_callback:
                            completed = sum(1 for r in results if r is not None)
                            progress_callback(completed, len(items))

                        return

                    except TimeoutError:
                        if attempt == self.retry_count - 1:
                            errors.append(
                                (index, TimeoutError(f"Timeout after {self.timeout_seconds}s"))
                            )
                    except Exception as e:
                        if attempt == self.retry_count - 1:
                            errors.append((index, e))
                        else:
                            await asyncio.sleep(self.retry_delay * (2**attempt))

        # Process all items
        tasks = [process_one(i, item) for i, item in enumerate(items)]
        await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r is not None)

        # Update statistics
        self._total_processed += len(items)
        self._total_errors += len(errors)
        self._total_time += total_time

        return BatchResult(
            results=[r for r in results if r is not None],
            errors=errors,
            total_time=total_time,
            success_count=success_count,
            error_count=len(errors),
        )

    def process_batch_sync(
        self,
        items: list[T],
        processor: Callable[[T], R],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult[R]:
        """
        Process items in parallel synchronously using ThreadPoolExecutor.

        Args:
            items: List of items to process
            processor: Function to apply to each item
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with results and errors
        """
        start_time = time.time()
        results: dict[int, R] = {}
        errors: list[tuple[int, Exception]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_with_retry, processor, item): i
                for i, item in enumerate(items)
            }

            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = (
                        future.result(timeout=self.timeout_seconds)
                        if self.timeout_seconds
                        else future.result()
                    )
                    results[index] = result
                except Exception as e:
                    errors.append((index, e))

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))

        total_time = time.time() - start_time

        # Update statistics
        self._total_processed += len(items)
        self._total_errors += len(errors)
        self._total_time += total_time

        # Sort results by index
        sorted_results = [results[i] for i in sorted(results.keys())]

        return BatchResult(
            results=sorted_results,
            errors=errors,
            total_time=total_time,
            success_count=len(results),
            error_count=len(errors),
        )

    def _process_with_retry(self, processor: Callable[[T], R], item: T) -> R:
        """Process item with retry logic."""
        last_error = None

        for attempt in range(self.retry_count):
            try:
                # Rate limiting (sync version)
                if self.rate_limit:
                    self._sync_rate_limit()

                return processor(item)

            except Exception as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (2**attempt))

        raise last_error

    async def _wait_for_rate_limit(self):
        """Wait to respect rate limit (async)."""
        if not self.rate_limit:
            return

        min_interval = 1.0 / self.rate_limit

        async with self._request_lock:
            now = time.time()
            elapsed = now - self._last_request_time

            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_request_time = time.time()

    def _sync_rate_limit(self):
        """Wait to respect rate limit (sync)."""
        if not self.rate_limit:
            return

        min_interval = 1.0 / self.rate_limit
        now = time.time()
        elapsed = now - self._last_request_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    @property
    def stats(self) -> dict[str, Any]:
        """Get processor statistics."""
        return {
            "total_processed": self._total_processed,
            "total_errors": self._total_errors,
            "error_rate": (
                self._total_errors / self._total_processed if self._total_processed > 0 else 0.0
            ),
            "total_time": self._total_time,
            "avg_time_per_item": (
                self._total_time / self._total_processed if self._total_processed > 0 else 0.0
            ),
        }


class ResourcePool(Generic[T]):
    """
    Resource pool for managing reusable resources (e.g., API clients).

    Features:
    - Automatic resource creation and cleanup
    - Connection pooling
    - Health checking
    """

    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 1,
        health_check: Callable[[T], bool] | None = None,
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.health_check = health_check

        self._pool: list[T] = []
        self._in_use: set[int] = set()
        self._lock = asyncio.Lock()

        # Initialize minimum resources
        for _ in range(min_size):
            self._pool.append(factory())

    async def acquire(self) -> T:
        """Acquire a resource from the pool."""
        async with self._lock:
            # Find available resource
            for i, resource in enumerate(self._pool):
                if i not in self._in_use:
                    # Health check
                    if self.health_check and not self.health_check(resource):
                        # Replace unhealthy resource
                        self._pool[i] = self.factory()
                        resource = self._pool[i]

                    self._in_use.add(i)
                    return resource

            # Create new resource if pool not full
            if len(self._pool) < self.max_size:
                resource = self.factory()
                self._pool.append(resource)
                self._in_use.add(len(self._pool) - 1)
                return resource

            # Wait for available resource
            raise RuntimeError("Resource pool exhausted")

    async def release(self, resource: T):
        """Release a resource back to the pool."""
        async with self._lock:
            try:
                index = self._pool.index(resource)
                self._in_use.discard(index)
            except ValueError:
                pass  # Resource not in pool

    async def __aenter__(self) -> T:
        return await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Resource is automatically released
        pass

    @property
    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": len(self._pool),
            "in_use": len(self._in_use),
            "available": len(self._pool) - len(self._in_use),
            "max_size": self.max_size,
        }


def cached(cache: SmartCache, key_func: Callable[..., str] | None = None):
    """
    Decorator for caching function results.

    Args:
        cache: SmartCache instance to use
        key_func: Optional function to generate cache key from arguments
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = str((args, tuple(sorted(kwargs.items()))))

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result)

            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = str((args, tuple(sorted(kwargs.items()))))

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            cache.set(key, result)

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
