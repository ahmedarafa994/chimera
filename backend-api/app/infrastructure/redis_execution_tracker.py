"""Distributed execution tracker with Redis backend.

This module provides a production-ready execution tracker that stores state
in Redis for horizontal scalability and automatic TTL-based cleanup to
prevent memory leaks.
"""

import asyncio
import json
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisExecutionTracker:
    """Distributed execution tracker with Redis backend.

    Features:
    - Automatic TTL-based cleanup (no memory leaks)
    - Distributed state for horizontal scaling
    - Local LRU cache for read performance
    - Graceful degradation to memory-only mode
    """

    # Configuration defaults
    ACTIVE_EXECUTION_TTL = 3600  # 1 hour (max execution time)
    COMPLETED_EXECUTION_TTL = 86400  # 24 hours retention
    LOCAL_CACHE_MAX_SIZE = 1000
    REDIS_KEY_PREFIX = "chimera:exec:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        active_ttl: int | None = None,
        completed_ttl: int | None = None,
        local_cache_size: int | None = None,
    ) -> None:
        self.redis_url = redis_url
        self._redis: redis.Redis | None = None
        self._connected = False

        # Configurable TTLs
        self.active_ttl = active_ttl or self.ACTIVE_EXECUTION_TTL
        self.completed_ttl = completed_ttl or self.COMPLETED_EXECUTION_TTL
        self.local_cache_max_size = local_cache_size or self.LOCAL_CACHE_MAX_SIZE

        # Bounded local cache for active executions (LRU)
        self._local_active_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._local_completed_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Establish Redis connection."""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
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
        while len(cache) > self.local_cache_max_size:
            cache.popitem(last=False)

    async def start_execution(
        self,
        execution_id: str,
        technique_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Start tracking an execution with automatic TTL."""
        execution_data = {
            "execution_id": execution_id,
            "technique_id": technique_id,
            "user_id": user_id,
            "start_time": datetime.utcnow().isoformat(),
            "status": "running",
        }

        async with self._lock:
            # Check if already exists
            if execution_id in self._local_active_cache:
                return False

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
                    self.active_ttl,
                    json.dumps(execution_data),
                )
                # Add to technique's active set for concurrent counting
                pipe.sadd(self._technique_active_set(technique_id), execution_id)
                pipe.expire(self._technique_active_set(technique_id), self.active_ttl)
                await pipe.execute()
            except Exception as e:
                logger.exception(f"Redis start_execution error: {e}")

        return True

    async def complete_execution(
        self,
        execution_id: str,
        success: bool,
        execution_time_ms: float,
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
                    logger.exception(f"Redis fetch error: {e}")
                    return

        if not execution:
            logger.warning(f"No active execution found for {execution_id}")
            return

        # Update execution data
        execution.update(
            {
                "status": "completed" if success else "failed",
                "success": success,
                "execution_time_ms": execution_time_ms,
                "end_time": datetime.utcnow().isoformat(),
            },
        )

        async with self._lock:
            # Store in completed cache (bounded)
            self._local_completed_cache[execution_id] = execution
            self._local_completed_cache.move_to_end(execution_id)
            await self._evict_local_cache(self._local_completed_cache)

        if self._connected and self._redis:
            try:
                technique_id = execution.get("technique_id", "")
                pipe = self._redis.pipeline()
                # Remove from active
                pipe.delete(self._active_key(execution_id))
                # Remove from technique's active set
                if technique_id:
                    pipe.srem(self._technique_active_set(technique_id), execution_id)
                # Store in completed with TTL (auto-cleanup!)
                pipe.setex(
                    self._completed_key(execution_id),
                    self.completed_ttl,
                    json.dumps(execution),
                )
                await pipe.execute()
            except Exception as e:
                logger.exception(f"Redis complete_execution error: {e}")

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
                logger.exception(f"Redis fetch error: {e}")

        if not execution:
            return

        execution.update(
            {
                "status": "failed",
                "success": False,
                "error_message": error_message,
                "end_time": datetime.utcnow().isoformat(),
            },
        )

        async with self._lock:
            self._local_completed_cache[execution_id] = execution
            await self._evict_local_cache(self._local_completed_cache)

        if self._connected and self._redis:
            try:
                technique_id = execution.get("technique_id", "")
                pipe = self._redis.pipeline()
                pipe.delete(self._active_key(execution_id))
                if technique_id:
                    pipe.srem(self._technique_active_set(technique_id), execution_id)
                pipe.setex(
                    self._completed_key(execution_id),
                    self.completed_ttl,
                    json.dumps(execution),
                )
                await pipe.execute()
            except Exception as e:
                logger.exception(f"Redis fail_execution error: {e}")

    async def get_active_count(self, technique_id: str | None = None) -> int:
        """Get count of active executions (distributed-aware)."""
        if self._connected and self._redis and technique_id:
            try:
                return await self._redis.scard(self._technique_active_set(technique_id))
            except Exception as e:
                logger.exception(f"Redis get_active_count error: {e}")

        # Fallback to local cache
        if technique_id:
            return len(
                [
                    e
                    for e in self._local_active_cache.values()
                    if e.get("technique_id") == technique_id
                ],
            )
        return len(self._local_active_cache)

    async def is_concurrent_limit_exceeded(self, technique_id: str, max_concurrent: int) -> bool:
        """Check concurrent execution limit (cluster-wide)."""
        count = await self.get_active_count(technique_id)
        return count >= max_concurrent

    async def enforce_cooldown(
        self,
        technique_id: str,
        user_id: str | None,
        cooldown_seconds: int,
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
                logger.exception(f"Redis cooldown check error: {e}")

        # Fallback: check local completed cache
        now = datetime.utcnow()
        for exec_data in self._local_completed_cache.values():
            if exec_data.get("technique_id") == technique_id and (
                user_id is None or exec_data.get("user_id") == user_id
            ):
                end_time_str = exec_data.get("end_time")
                if end_time_str:
                    end_time = datetime.fromisoformat(end_time_str)
                    if (now - end_time).total_seconds() < cooldown_seconds:
                        return True
        return False

    async def set_cooldown(
        self,
        technique_id: str,
        user_id: str | None,
        cooldown_seconds: int,
    ) -> None:
        """Set cooldown after execution completion."""
        if cooldown_seconds <= 0:
            return

        cooldown_key = f"{self.REDIS_KEY_PREFIX}cooldown:{technique_id}:{user_id or 'anonymous'}"

        if self._connected and self._redis:
            try:
                await self._redis.setex(cooldown_key, cooldown_seconds, "1")
            except Exception as e:
                logger.exception(f"Redis set_cooldown error: {e}")

    async def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        """Get execution data by ID."""
        # Check active cache first
        if execution_id in self._local_active_cache:
            return self._local_active_cache[execution_id]

        # Check completed cache
        if execution_id in self._local_completed_cache:
            return self._local_completed_cache[execution_id]

        # Try Redis
        if self._connected and self._redis:
            try:
                # Try active first
                data = await self._redis.get(self._active_key(execution_id))
                if data:
                    return json.loads(data)
                # Try completed
                data = await self._redis.get(self._completed_key(execution_id))
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.exception(f"Redis get_execution error: {e}")

        return None

    async def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        stats = {
            "redis_connected": self._connected,
            "local_active_count": len(self._local_active_cache),
            "local_completed_count": len(self._local_completed_cache),
            "local_cache_max_size": self.local_cache_max_size,
            "active_ttl_seconds": self.active_ttl,
            "completed_ttl_seconds": self.completed_ttl,
        }

        if self._connected and self._redis:
            try:
                active_keys = await self._redis.keys(f"{self.REDIS_KEY_PREFIX}active:*")
                completed_keys = await self._redis.keys(f"{self.REDIS_KEY_PREFIX}completed:*")
                stats.update(
                    {
                        "redis_active_count": len(active_keys),
                        "redis_completed_count": len(completed_keys),
                    },
                )
            except Exception as e:
                stats["redis_error"] = str(e)

        return stats


# Global instance (initialized on startup)
execution_tracker: RedisExecutionTracker | None = None


async def get_execution_tracker() -> RedisExecutionTracker:
    """Get or create the global execution tracker instance."""
    global execution_tracker
    if execution_tracker is None:
        from app.core.config import settings

        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        execution_tracker = RedisExecutionTracker(redis_url=redis_url)
        await execution_tracker.connect()
    return execution_tracker


async def shutdown_execution_tracker() -> None:
    """Shutdown the global execution tracker."""
    global execution_tracker
    if execution_tracker:
        await execution_tracker.disconnect()
        execution_tracker = None
