# Technical Remediation Plan: Backend-API Service

**Document Version:** 1.0
**Date:** December 4, 2025
**Author:** Backend Architecture Team
**Classification:** Internal Engineering

---

## Executive Summary

This document provides a comprehensive remediation strategy for four critical-to-moderate defects identified in the `backend-api` service. The findings span memory management, distributed state handling, and error handling patterns. The recommended solutions prioritize **Redis-based distributed state management** to enable horizontal scalability while resolving immediate memory leak vulnerabilities.

---

## Table of Contents

1. [Defect #1: Memory Leak in ExecutionTracker](#defect-1-critical-memory-leak-in-executiontracker)
2. [Defect #2: Unbounded In-Memory Cache](#defect-2-high-unbounded-in-memory-cache)
3. [Defect #3: Non-Distributed State](#defect-3-moderate-non-distributed-state)
4. [Defect #4: Brittle Error Handling](#defect-4-moderate-brittle-error-handling)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Configuration Changes](#configuration-changes)

---

## Defect #1: Critical – Memory Leak in ExecutionTracker

### Location
`backend-api/app/services/jailbreak/jailbreak_service.py` – Class `ExecutionTracker`, Line 235

### Defect Description
The `_completed_executions` dictionary indefinitely stores the history of every finished job without any cleanup mechanism or size limit:

```python
# Current problematic code (Line 269-270)
self._completed_executions[execution_id] = execution
# No cleanup, no size limit, no TTL
```

### Root Cause Analysis

| Factor | Analysis |
|--------|----------|
| **Design Flaw** | The class was designed without considering long-running production workloads. The dictionary acts as an append-only log with no retention policy. |
| **Missing Lifecycle Management** | No background task or hook exists to prune old entries. |
| **Unbounded Growth Pattern** | Each execution adds ~500 bytes; at 1000 req/sec, memory grows by ~1.7GB/hour. |

### Impact Assessment
- **Severity:** Critical (OOM crash imminent under production load)
- **MTTR Impact:** Service restart required, all in-flight executions lost
- **Business Impact:** Complete service unavailability

### Architectural Solution

Implement a **Redis-backed execution tracker** with automatic TTL-based expiration and bounded local cache for performance.

```python
# backend-api/app/infrastructure/redis_execution_tracker.py

import redis.asyncio as redis
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import OrderedDict
import asyncio

logger = logging.getLogger(__name__)

class RedisExecutionTracker:
    """
    Distributed execution tracker with Redis backend.

    Features:
    - Automatic TTL-based cleanup (no memory leaks)
    - Distributed state for horizontal scaling
    - Local LRU cache for read performance
    - Graceful degradation to memory-only mode
    """

    # Configuration
    ACTIVE_EXECUTION_TTL = 3600  # 1 hour (max execution time)
    COMPLETED_EXECUTION_TTL = 86400  # 24 hours retention
    LOCAL_CACHE_MAX_SIZE = 1000
    REDIS_KEY_PREFIX = "chimera:exec:"

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._connected = False

        # Bounded local cache for active executions (LRU)
        self._local_active_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._local_completed_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Establish Redis connection."""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            await self._redis.ping()
            self._connected = True
            logger.info("ExecutionTracker connected to Redis")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed, using memory fallback: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Clean disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _active_key(self, execution_id: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}active:{execution_id}"

    def _completed_key(self, execution_id: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}completed:{execution_id}"

    def _technique_active_set(self, technique_id: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}technique_active:{technique_id}"

    async def _evict_local_cache(self, cache: OrderedDict) -> None:
        """Evict oldest entries when cache exceeds max size."""
        while len(cache) > self.LOCAL_CACHE_MAX_SIZE:
            cache.popitem(last=False)

    async def start_execution(
        self,
        execution_id: str,
        technique_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Start tracking an execution with automatic TTL."""
        execution_data = {
            "execution_id": execution_id,
            "technique_id": technique_id,
            "user_id": user_id,
            "start_time": datetime.utcnow().isoformat(),
            "status": "running"
        }

        async with self._lock:
            # Update local cache
            self._local_active_cache[execution_id] = execution_data
            self._local_active_cache.move_to_end(execution_id)
            await self._evict_local_cache(self._local_active_cache)

        if self._connected and self._redis:
            try:
                pipe = self._redis.pipeline()
                # Store execution data with TTL
                pipe.setex(
                    self._active_key(execution_id),
                    self.ACTIVE_EXECUTION_TTL,
                    json.dumps(execution_data)
                )
                # Add to technique's active set for concurrent counting
                pipe.sadd(self._technique_active_set(technique_id), execution_id)
                pipe.expire(self._technique_active_set(technique_id), self.ACTIVE_EXECUTION_TTL)
                await pipe.execute()
            except Exception as e:
                logger.error(f"Redis start_execution error: {e}")

        return True

    async def complete_execution(
        self,
        execution_id: str,
        success: bool,
        execution_time_ms: float
    ) -> None:
        """Mark execution as completed with automatic TTL cleanup."""
        async with self._lock:
            execution = self._local_active_cache.pop(execution_id, None)

        if not execution:
            # Try fetching from Redis
            if self._connected and self._redis:
                try:
                    data = await self._redis.get(self._active_key(execution_id))
                    if data:
                        execution = json.loads(data)
                except Exception as e:
                    logger.error(f"Redis fetch error: {e}")
                    return

        if not execution:
            logger.warning(f"No active execution found for {execution_id}")
            return

        # Update execution data
        execution.update({
            "status": "completed" if success else "failed",
            "success": success,
            "execution_time_ms": execution_time_ms,
            "end_time": datetime.utcnow().isoformat()
        })

        async with self._lock:
            # Store in completed cache (bounded)
            self._local_completed_cache[execution_id] = execution
            self._local_completed_cache.move_to_end(execution_id)
            await self._evict_local_cache(self._local_completed_cache)

        if self._connected and self._redis:
            try:
                pipe = self._redis.pipeline()
                # Remove from active
                pipe.delete(self._active_key(execution_id))
                # Remove from technique's active set
                pipe.srem(
                    self._technique_active_set(execution["technique_id"]),
                    execution_id
                )
                # Store in completed with TTL (auto-cleanup!)
                pipe.setex(
                    self._completed_key(execution_id),
                    self.COMPLETED_EXECUTION_TTL,
                    json.dumps(execution)
                )
                await pipe.execute()
            except Exception as e:
                logger.error(f"Redis complete_execution error: {e}")

    async def fail_execution(self, execution_id: str, error_message: str) -> None:
        """Mark execution as failed."""
        async with self._lock:
            execution = self._local_active_cache.pop(execution_id, None)

        if not execution and self._connected and self._redis:
            try:
                data = await self._redis.get(self._active_key(execution_id))
                if data:
                    execution = json.loads(data)
            except Exception as e:
                logger.error(f"Redis fetch error: {e}")

        if not execution:
            return

        execution.update({
            "status": "failed",
            "success": False,
            "error_message": error_message,
            "end_time": datetime.utcnow().isoformat()
        })

        async with self._lock:
            self._local_completed_cache[execution_id] = execution
            await self._evict_local_cache(self._local_completed_cache)

        if self._connected and self._redis:
            try:
                pipe = self._redis.pipeline()
                pipe.delete(self._active_key(execution_id))
                pipe.srem(
                    self._technique_active_set(execution["technique_id"]),
                    execution_id
                )
                pipe.setex(
                    self._completed_key(execution_id),
                    self.COMPLETED_EXECUTION_TTL,
                    json.dumps(execution)
                )
                await pipe.execute()
            except Exception as e:
                logger.error(f"Redis fail_execution error: {e}")

    async def get_active_count(self, technique_id: Optional[str] = None) -> int:
        """Get count of active executions (distributed-aware)."""
        if self._connected and self._redis and technique_id:
            try:
                return await self._redis.scard(self._technique_active_set(technique_id))
            except Exception as e:
                logger.error(f"Redis get_active_count error: {e}")

        # Fallback to local cache
        if technique_id:
            return len([
                e for e in self._local_active_cache.values()
                if e.get("technique_id") == technique_id
            ])
        return len(self._local_active_cache)

    async def is_concurrent_limit_exceeded(
        self,
        technique_id: str,
        max_concurrent: int
    ) -> bool:
        """Check concurrent execution limit (cluster-wide)."""
        count = await self.get_active_count(technique_id)
        return count >= max_concurrent

    async def enforce_cooldown(
        self,
        technique_id: str,
        user_id: Optional[str],
        cooldown_seconds: int
    ) -> bool:
        """Check cooldown period using Redis-based recent execution tracking."""
        if cooldown_seconds <= 0:
            return False

        cooldown_key = f"{self.REDIS_KEY_PREFIX}cooldown:{technique_id}:{user_id or 'anonymous'}"

        if self._connected and self._redis:
            try:
                exists = await self._redis.exists(cooldown_key)
                return exists > 0
            except Exception as e:
                logger.error(f"Redis cooldown check error: {e}")

        # Fallback: check local completed cache
        now = datetime.utcnow()
        for exec_data in self._local_completed_cache.values():
            if (exec_data.get("technique_id") == technique_id and
                (user_id is None or exec_data.get("user_id") == user_id)):
                end_time_str = exec_data.get("end_time")
                if end_time_str:
                    end_time = datetime.fromisoformat(end_time_str)
                    if (now - end_time).total_seconds() < cooldown_seconds:
                        return True
        return False

    async def set_cooldown(
        self,
        technique_id: str,
        user_id: Optional[str],
        cooldown_seconds: int
    ) -> None:
        """Set cooldown after execution completion."""
        if cooldown_seconds <= 0:
            return

        cooldown_key = f"{self.REDIS_KEY_PREFIX}cooldown:{technique_id}:{user_id or 'anonymous'}"

        if self._connected and self._redis:
            try:
                await self._redis.setex(cooldown_key, cooldown_seconds, "1")
            except Exception as e:
                logger.error(f"Redis set_cooldown error: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        stats = {
            "redis_connected": self._connected,
            "local_active_count": len(self._local_active_cache),
            "local_completed_count": len(self._local_completed_cache),
            "local_cache_max_size": self.LOCAL_CACHE_MAX_SIZE
        }

        if self._connected and self._redis:
            try:
                active_keys = await self._redis.keys(f"{self.REDIS_KEY_PREFIX}active:*")
                completed_keys = await self._redis.keys(f"{self.REDIS_KEY_PREFIX}completed:*")
                stats.update({
                    "redis_active_count": len(active_keys),
                    "redis_completed_count": len(completed_keys)
                })
            except Exception as e:
                stats["redis_error"] = str(e)

        return stats
```

### Migration Strategy

1. **Phase 1 - Parallel Operation (Week 1)**
   - Deploy `RedisExecutionTracker` alongside existing tracker
   - Write to both, read from new
   - Monitor for discrepancies

2. **Phase 2 - Cutover (Week 2)**
   - Switch reads to new tracker
   - Deprecate old tracker
   - Enable Redis clustering

3. **Phase 3 - Cleanup (Week 3)**
   - Remove legacy `ExecutionTracker` class
   - Update all dependent services

---

## Defect #2: High – Unbounded In-Memory Cache

### Location
`backend-api/app/services/jailbreak/jailbreak_service.py` – Class `CacheManager`, Line 84

### Defect Description
The `_cache` dictionary stores arbitrary data with TTL but lacks maximum size limits or eviction policies:

```python
# Current problematic code (Lines 79-81)
def __init__(self):
    self._cache: Dict[str, Any] = {}  # No size limit
    self._timestamps: Dict[str, float] = {}
    self._ttls: Dict[str, int] = {}
```

### Root Cause Analysis

| Factor | Analysis |
|--------|----------|
| **No Eviction Policy** | Cache grows until process memory is exhausted |
| **Passive Expiration Only** | TTL only checked on read; expired entries linger in memory |
| **Attack Surface** | Fuzzing attacks can fill cache with unique keys faster than TTL can expire them |

### Impact Assessment
- **Severity:** High (DoS vulnerability)
- **Attack Vector:** Malicious actor sends unique requests to exhaust memory
- **Time to Exploit:** Minutes under sustained attack

### Architectural Solution

The existing `HybridCache` in `app/core/cache.py` already implements proper LRU eviction and Redis backend. **Refactor `CacheManager` to use `HybridCache` as its backend.**

```python
# backend-api/app/services/jailbreak/bounded_cache_manager.py

import logging
from typing import Any, Optional, Dict
from app.core.cache import HybridCache, CacheConfig

logger = logging.getLogger(__name__)

class BoundedCacheManager:
    """
    Jailbreak service cache manager with bounded memory and Redis backend.

    Wraps HybridCache to provide jailbreak-specific caching semantics
    with protection against memory exhaustion attacks.
    """

    # Jailbreak-specific configuration
    DEFAULT_TTL = 3600  # 1 hour
    EXECUTION_RESULT_TTL = 1800  # 30 minutes
    TECHNIQUE_CACHE_TTL = 7200  # 2 hours
    MAX_VALUE_SIZE_BYTES = 1_000_000  # 1MB per entry
    KEY_PREFIX = "jailbreak:"

    def __init__(self, config: Optional[CacheConfig] = None):
        cache_config = config or CacheConfig(
            max_memory_items=5000,  # Bounded!
            key_prefix="chimera:jailbreak:"
        )
        self._cache = HybridCache(cache_config)
        self._initialized = False

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

    def _validate_value_size(self, value: Any) -> bool:
        """Reject values that exceed maximum size."""
        try:
            import json
            serialized = json.dumps(value, default=str)
            return len(serialized.encode()) <= self.MAX_VALUE_SIZE_BYTES
        except Exception:
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        full_key = f"{self.KEY_PREFIX}{key}"
        return await self._cache.get(full_key)

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = None
    ) -> bool:
        """
        Set value in cache with size validation.

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

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self._cache.get_stats()

    # Jailbreak-specific convenience methods

    async def cache_execution_result(
        self,
        execution_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """Cache an execution result."""
        key = f"result:{execution_id}"
        return await self.set(key, result, self.EXECUTION_RESULT_TTL)

    async def get_execution_result(
        self,
        execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached execution result."""
        key = f"result:{execution_id}"
        return await self.get(key)

    async def cache_technique(
        self,
        technique_id: str,
        technique_data: Dict[str, Any]
    ) -> bool:
        """Cache technique data."""
        key = f"technique:{technique_id}"
        return await self.set(key, technique_data, self.TECHNIQUE_CACHE_TTL)

    async def get_technique(
        self,
        technique_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached technique data."""
        key = f"technique:{technique_id}"
        return await self.get(key)
```

### Key Improvements

| Improvement | Benefit |
|-------------|---------|
| **Bounded Size** | `max_memory_items=5000` prevents unbounded growth |
| **LRU Eviction** | Automatic eviction of least-recently-used entries |
| **Value Size Limit** | Rejects entries > 1MB to prevent memory bombs |
| **Redis Backend** | Distributed caching with automatic TTL |
| **Hybrid Fallback** | Memory-only mode if Redis unavailable |

---

## Defect #3: Moderate – Non-Distributed State (Scalability Bottleneck)

### Location
Multiple components in `JailbreakService`:
- `RateLimiter._requests`
- `CacheManager._cache`
- `ExecutionTracker._active_executions`

### Defect Description
All critical state is stored in Python process memory, making horizontal scaling impossible.

### Root Cause Analysis

| Factor | Analysis |
|--------|----------|
| **Single-Instance Design** | Original architecture assumed single deployment |
| **No External State Store** | No Redis/database integration for shared state |
| **Data Loss on Restart** | All audit logs, rate limits, and cache data are ephemeral |

### Impact Assessment
- **Severity:** Moderate (blocks scaling, causes data loss)
- **Scalability Impact:** Cannot add replicas behind load balancer
- **Reliability Impact:** All state lost on container restart

### Architectural Solution

Implement a **Redis-backed Rate Limiter** using the sliding window algorithm for distributed rate limiting.

```python
# backend-api/app/infrastructure/redis_rate_limiter.py

import redis.asyncio as redis
import logging
import time
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class RedisRateLimiter:
    """
    Distributed rate limiter using Redis sliding window algorithm.

    Features:
    - Cluster-wide rate limiting across all service instances
    - Sliding window for accurate rate calculations
    - Graceful degradation to local limiting if Redis unavailable
    - Configurable limits per key pattern
    """

    REDIS_KEY_PREFIX = "chimera:ratelimit:"

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._connected = False

        # Fallback local rate limiter
        self._local_requests: dict = {}

    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5.0
            )
            await self._redis.ping()
            self._connected = True
            logger.info("RateLimiter connected to Redis")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed for rate limiter: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._connected = False

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60
    ) -> bool:
        """
        Check if request is allowed under rate limit using sliding window.

        Uses Redis sorted set with timestamps as scores for O(log n) operations.
        """
        now = time.time()
        redis_key = f"{self.REDIS_KEY_PREFIX}{key}"
        window_start = now - window_seconds

        if self._connected and self._redis:
            try:
                pipe = self._redis.pipeline()

                # Remove entries outside the window
                pipe.zremrangebyscore(redis_key, 0, window_start)

                # Count current requests in window
                pipe.zcard(redis_key)

                results = await pipe.execute()
                current_count = results[1]

                if current_count < limit:
                    # Add new request
                    pipe = self._redis.pipeline()
                    pipe.zadd(redis_key, {f"{now}:{id(now)}": now})
                    pipe.expire(redis_key, window_seconds + 1)
                    await pipe.execute()
                    return True

                return False

            except Exception as e:
                logger.error(f"Redis rate limit error: {e}")
                # Fall through to local limiter

        # Local fallback
        return await self._local_is_allowed(key, limit, window_seconds)

    async def _local_is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> bool:
        """Local fallback rate limiter."""
        now = time.time()

        if key not in self._local_requests:
            self._local_requests[key] = []

        # Clean old requests
        self._local_requests[key] = [
            t for t in self._local_requests[key]
            if now - t < window_seconds
        ]

        if len(self._local_requests[key]) < limit:
            self._local_requests[key].append(now)
            return True

        return False

    async def get_remaining_requests(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60
    ) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        redis_key = f"{self.REDIS_KEY_PREFIX}{key}"
        window_start = now - window_seconds

        if self._connected and self._redis:
            try:
                # Clean and count
                await self._redis.zremrangebyscore(redis_key, 0, window_start)
                current_count = await self._redis.zcard(redis_key)
                return max(0, limit - current_count)
            except Exception as e:
                logger.error(f"Redis get_remaining error: {e}")

        # Local fallback
        if key not in self._local_requests:
            return limit

        recent = [t for t in self._local_requests[key] if now - t < window_seconds]
        return max(0, limit - len(recent))

    async def get_reset_time(
        self,
        key: str,
        window_seconds: int = 60
    ) -> int:
        """Get Unix timestamp when rate limit resets."""
        redis_key = f"{self.REDIS_KEY_PREFIX}{key}"

        if self._connected and self._redis:
            try:
                # Get oldest request in window
                oldest = await self._redis.zrange(redis_key, 0, 0, withscores=True)
                if oldest:
                    return int(oldest[0][1] + window_seconds)
            except Exception as e:
                logger.error(f"Redis get_reset_time error: {e}")

        # Local fallback
        if key in self._local_requests and self._local_requests[key]:
            return int(min(self._local_requests[key]) + window_seconds)

        return int(time.time())

    async def reset_limit(self, key: str) -> None:
        """Reset rate limit for a key."""
        redis_key = f"{self.REDIS_KEY_PREFIX}{key}"

        if self._connected and self._redis:
            try:
                await self._redis.delete(redis_key)
            except Exception as e:
                logger.error(f"Redis reset_limit error: {e}")

        self._local_requests.pop(key, None)

    async def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        stats = {
            "redis_connected": self._connected,
            "local_keys_tracked": len(self._local_requests)
        }

        if self._connected and self._redis:
            try:
                keys = await self._redis.keys(f"{self.REDIS_KEY_PREFIX}*")
                stats["redis_keys_tracked"] = len(keys)
            except Exception as e:
                stats["redis_error"] = str(e)

        return stats
```

### State Migration Summary

| Component | Current State | Target State |
|-----------|---------------|--------------|
| `RateLimiter` | Local dict | Redis sorted sets |
| `CacheManager` | Local dict | `HybridCache` (Redis + LRU) |
| `ExecutionTracker` | Local dicts | Redis hashes + sets |
| `AuditLogger` | Local list | Database + Redis queue |

---

## Defect #4: Moderate – Brittle Error Handling in API Routes

### Location
`backend-api/app/api/api_routes.py` – Multiple endpoints

### Defect Description
Generic exception handling masks specific error conditions:

```python
# Current problematic pattern (Lines 55-60)
except Exception as e:
    logger.error(f"Generation failed: {str(e)}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(e)  # Exposes internal details!
    )
```

### Root Cause Analysis

| Factor | Analysis |
|--------|----------|
| **Catch-All Pattern** | `except Exception` catches everything, including programming errors |
| **Information Leakage** | `detail=str(e)` may expose stack traces, file paths, or credentials |
| **Loss of Specificity** | Client cannot distinguish between input errors and server failures |
| **Debugging Difficulty** | Generic 500 errors provide no actionable information |

### Impact Assessment
- **Severity:** Moderate (security risk + poor UX)
- **Security Risk:** Stack traces may reveal implementation details
- **UX Impact:** Clients cannot programmatically handle specific errors

### Architectural Solution

Implement structured exception handling using the existing `APIException` hierarchy in `app/core/exceptions.py`.

```python
# backend-api/app/api/error_handlers.py

import logging
from typing import Callable, TypeVar, Awaitable
from functools import wraps
from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    APIException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    LLMProviderError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMQuotaExceededError,
    TransformationError,
    ServiceUnavailableError,
    ProviderNotAvailableError,
    handle_exception
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

def api_error_handler(
    operation_name: str,
    default_error_message: str = "An unexpected error occurred"
):
    """
    Decorator for standardized API error handling.

    Features:
    - Maps known exceptions to appropriate HTTP status codes
    - Sanitizes error messages for external clients
    - Logs full error details internally
    - Preserves exception chains for debugging

    Usage:
        @api_error_handler("generate_content")
        async def generate_content(request: PromptRequest):
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)

            except HTTPException:
                # Re-raise FastAPI HTTP exceptions as-is
                raise

            except APIException as e:
                # Our custom exceptions - map to appropriate HTTP response
                logger.warning(
                    f"{operation_name} failed with {e.error_code}: {e.message}",
                    extra={"details": e.details}
                )
                raise HTTPException(
                    status_code=e.status_code,
                    detail={
                        "error": e.error_code,
                        "message": e.message,
                        "details": e.details
                    }
                )

            except ValueError as e:
                # Input validation errors
                logger.warning(f"{operation_name} validation error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "VALIDATION_ERROR",
                        "message": str(e)
                    }
                )

            except TimeoutError as e:
                logger.error(f"{operation_name} timeout: {e}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "error": "TIMEOUT_ERROR",
                        "message": "The request timed out. Please try again."
                    }
                )

            except ConnectionError as e:
                logger.error(f"{operation_name} connection error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "SERVICE_UNAVAILABLE",
                        "message": "Unable to connect to required service. Please try again later."
                    }
                )

            except Exception as e:
                # Unexpected errors - log full details, return sanitized message
                logger.exception(
                    f"{operation_name} unexpected error: {type(e).__name__}: {e}"
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "INTERNAL_ERROR",
                        "message": default_error_message,
                        # Include request ID for support correlation
                        "support_reference": kwargs.get("request_id", "N/A")
                    }
                )

        return wrapper
    return decorator


class ErrorResponseBuilder:
    """
    Builder for consistent error responses.

    Usage:
        raise ErrorResponseBuilder.bad_request("Invalid input", field="prompt")
    """

    @staticmethod
    def bad_request(message: str, **details) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "BAD_REQUEST",
                "message": message,
                "details": details
            }
        )

    @staticmethod
    def not_found(resource: str, identifier: str) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "NOT_FOUND",
                "message": f"{resource} not found",
                "details": {"identifier": identifier}
            }
        )

    @staticmethod
    def rate_limited(
        retry_after: int,
        limit: int,
        window: str = "minute"
    ) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": f"Rate limit of {limit} requests per {window} exceeded",
                "details": {"retry_after_seconds": retry_after}
            },
            headers={"Retry-After": str(retry_after)}
        )

    @staticmethod
    def provider_unavailable(provider: str, reason: str = None) -> HTTPException:
        message = f"Provider '{provider}' is currently unavailable"
        if reason:
            message = f"{message}: {reason}"
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "PROVIDER_UNAVAILABLE",
                "message": message,
                "details": {"provider": provider}
            }
        )

    @staticmethod
    def internal_error(support_ref: str = None) -> HTTPException:
        detail = {
            "error": "INTERNAL_ERROR",
            "message": "An internal error occurred. Please try again later."
        }
        if support_ref:
            detail["support_reference"] = support_ref
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


# Exception handler registration for FastAPI app
async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Global handler for APIException and subclasses."""
    logger.warning(
        f"API Exception: {exc.error_code} - {exc.message}",
        extra={"path": request.url.path, "details": exc.details}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


def register_exception_handlers(app):
    """Register all custom exception handlers with FastAPI app."""
    from app.core.exceptions import (
        APIException, ValidationError, AuthenticationError,
        LLMProviderError, TransformationError
    )

    app.add_exception_handler(APIException, api_exception_handler)

    # You can add more specific handlers here if needed
    # app.add_exception_handler(ValidationError, validation_error_handler)
```

### Refactored Endpoint Example

```python
# backend-api/app/api/api_routes.py (refactored)

from app.api.error_handlers import api_error_handler, ErrorResponseBuilder
from app.core.exceptions import ValidationError, LLMProviderError

@router.post("/generate", response_model=PromptResponse)
@api_error_handler("generate_content", "Failed to generate content")
async def generate_content(
    request: PromptRequest,
    service: LLMService = Depends(get_llm_service)
):
    """Generate text content using the configured LLM provider."""
    if not request.prompt or not request.prompt.strip():
        raise ValidationError(
            message="Prompt cannot be empty",
            details={"field": "prompt"}
        )

    return await service.generate_text(request)


@router.post("/transform", response_model=TransformationResponse)
@api_error_handler("transform_prompt", "Failed to transform prompt")
async def transform_prompt(request: TransformationRequest):
    """Transform a prompt without executing it."""
    from app.services.transformation_service import transformation_engine

    result = await transformation_engine.transform(
        prompt=request.core_request,
        potency_level=request.potency_level,
        technique_suite=request.technique_suite,
    )

    return TransformationResponse(
        success=result.success,
        original_prompt=result.original_prompt,
        transformed_prompt=result.transformed_prompt,
        metadata={
            "strategy": result.metadata.strategy,
            "layers_applied": result.metadata.layers_applied,
            "techniques_used": result.metadata.techniques_used,
            "potency_level": result.metadata.potency_level,
            "technique_suite": result.metadata.technique_suite,
            "execution_time_ms": result.metadata.execution_time_ms,
            "cached": result.metadata.cached,
        },
    )
```

### Error Response Format

All errors now follow a consistent JSON structure:

```json
{
    "error": "VALIDATION_ERROR",
    "message": "Invalid potency level. Must be between 1 and 10",
    "details": {
        "field": "potency_level",
        "provided": 15,
        "allowed_range": [1, 10]
    }
}
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1-2)

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Deploy Redis cluster | Critical | 2 days | DevOps |
| Implement `RedisExecutionTracker` | Critical | 3 days | Backend |
| Add config for Redis connection | Critical | 1 day | Backend |
| Unit tests for new tracker | Critical | 2 days | QA |

### Phase 2: High Priority (Week 3-4)

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Implement `BoundedCacheManager` | High | 2 days | Backend |
| Implement `RedisRateLimiter` | High | 2 days | Backend |
| Integration testing | High | 3 days | QA |
| Performance benchmarking | High | 2 days | Backend |

### Phase 3: Moderate Priority (Week 5-6)

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Implement error handler decorator | Moderate | 2 days | Backend |
| Refactor all API endpoints | Moderate | 3 days | Backend |
| Update API documentation | Moderate | 1 day | Backend |
| End-to-end testing | Moderate | 2 days | QA |

### Phase 4: Hardening (Week 7-8)

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Load testing with Redis | Normal | 3 days | QA |
| Chaos engineering tests | Normal | 2 days | DevOps |
| Documentation update | Normal | 2 days | Backend |
| Production deployment | Normal | 1 day | DevOps |

---

## Configuration Changes

### Environment Variables

Add the following to `.env` and `app/core/config.py`:

```python
# backend-api/app/core/config.py (additions)

class Settings(BaseSettings):
    # ... existing settings ...

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SSL: bool = False
    REDIS_CONNECTION_TIMEOUT: int = 5
    REDIS_SOCKET_TIMEOUT: int = 5

    # Execution Tracker
    EXECUTION_ACTIVE_TTL: int = 3600  # 1 hour
    EXECUTION_COMPLETED_TTL: int = 86400  # 24 hours
    EXECUTION_LOCAL_CACHE_SIZE: int = 1000

    # Cache Manager
    CACHE_MAX_MEMORY_ITEMS: int = 5000
    CACHE_MAX_VALUE_SIZE_BYTES: int = 1_000_000
    CACHE_DEFAULT_TTL: int = 3600

    # Rate Limiter
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT_LIMIT: int = 60
    RATE_LIMIT_DEFAULT_WINDOW: int = 60
```

### Docker Compose Updates

```yaml
# docker-compose.yml (additions)

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend-api:
    depends_on:
      redis:
        condition: service_healthy
    environment:
      - REDIS_URL=redis://redis:6379/0

volumes:
  redis_data:
```

---

## Summary

| Defect | Severity | Solution | Risk Reduction |
|--------|----------|----------|----------------|
| Memory Leak (ExecutionTracker) | Critical | Redis-backed tracker with TTL | OOM crash eliminated |
| Unbounded Cache | High | HybridCache with LRU + size limits | DoS attack mitigated |
| Non-Distributed State | Moderate | Redis for all shared state | Horizontal scaling enabled |
| Brittle Error Handling | Moderate | Structured exception hierarchy | Security + UX improved |

**Total Estimated Effort:** 8 weeks
**Risk Level After Remediation:** Low
**Recommended Start Date:** Immediate
