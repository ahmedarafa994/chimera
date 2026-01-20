"""Distributed rate limiter with Redis backend.

This module provides a production-ready rate limiter using Redis
sorted sets for sliding window rate limiting across multiple instances.
"""

import asyncio
import logging
import time
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """Distributed rate limiter using Redis sliding window algorithm.

    Features:
    - Cluster-wide rate limiting across all service instances
    - Sliding window for accurate rate calculations
    - Graceful degradation to local limiting if Redis unavailable
    - Configurable limits per key pattern
    """

    REDIS_KEY_PREFIX = "chimera:ratelimit:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_limit: int = 60,
        default_window: int = 60,
    ) -> None:
        self.redis_url = redis_url
        self._redis: redis.Redis | None = None
        self._connected = False
        self._default_limit = default_limit
        self._default_window = default_window

        # Fallback local rate limiter
        self._local_requests: dict[str, list] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
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
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> bool:
        """Check if request is allowed under rate limit using sliding window.

        Uses Redis sorted set with timestamps as scores for O(log n) operations.

        Args:
            key: Rate limit key (e.g., user ID, IP address)
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            True if request is allowed, False if rate limited

        """
        limit = limit or self._default_limit
        window_seconds = window_seconds or self._default_window

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
                    # Add new request with unique member
                    pipe = self._redis.pipeline()
                    member = f"{now}:{id(now)}:{time.time_ns()}"
                    pipe.zadd(redis_key, {member: now})
                    pipe.expire(redis_key, window_seconds + 1)
                    await pipe.execute()
                    return True

                return False

            except Exception as e:
                logger.exception(f"Redis rate limit error: {e}")
                # Fall through to local limiter

        # Local fallback
        return await self._local_is_allowed(key, limit, window_seconds)

    async def _local_is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        """Local fallback rate limiter."""
        now = time.time()

        async with self._lock:
            if key not in self._local_requests:
                self._local_requests[key] = []

            # Clean old requests
            self._local_requests[key] = [
                t for t in self._local_requests[key] if now - t < window_seconds
            ]

            if len(self._local_requests[key]) < limit:
                self._local_requests[key].append(now)
                return True

            return False

    async def get_remaining_requests(
        self,
        key: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> int:
        """Get remaining requests in current window."""
        limit = limit or self._default_limit
        window_seconds = window_seconds or self._default_window

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
                logger.exception(f"Redis get_remaining error: {e}")

        # Local fallback
        async with self._lock:
            if key not in self._local_requests:
                return limit

            recent = [t for t in self._local_requests[key] if now - t < window_seconds]
            return max(0, limit - len(recent))

    async def get_reset_time(self, key: str, window_seconds: int | None = None) -> int:
        """Get Unix timestamp when rate limit resets."""
        window_seconds = window_seconds or self._default_window
        redis_key = f"{self.REDIS_KEY_PREFIX}{key}"

        if self._connected and self._redis:
            try:
                # Get oldest request in window
                oldest = await self._redis.zrange(redis_key, 0, 0, withscores=True)
                if oldest:
                    return int(oldest[0][1] + window_seconds)
            except Exception as e:
                logger.exception(f"Redis get_reset_time error: {e}")

        # Local fallback
        async with self._lock:
            if self._local_requests.get(key):
                return int(min(self._local_requests[key]) + window_seconds)

        return int(time.time())

    async def reset_limit(self, key: str) -> None:
        """Reset rate limit for a key."""
        redis_key = f"{self.REDIS_KEY_PREFIX}{key}"

        if self._connected and self._redis:
            try:
                await self._redis.delete(redis_key)
            except Exception as e:
                logger.exception(f"Redis reset_limit error: {e}")

        async with self._lock:
            self._local_requests.pop(key, None)

    async def get_usage(
        self,
        key: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Get detailed rate limit usage for a key."""
        limit = limit or self._default_limit
        window_seconds = window_seconds or self._default_window

        remaining = await self.get_remaining_requests(key, limit, window_seconds)
        reset_time = await self.get_reset_time(key, window_seconds)

        return {
            "limit": limit,
            "remaining": remaining,
            "used": limit - remaining,
            "reset_at": reset_time,
            "window_seconds": window_seconds,
        }

    async def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        stats = {
            "redis_connected": self._connected,
            "default_limit": self._default_limit,
            "default_window": self._default_window,
        }

        async with self._lock:
            stats["local_keys_tracked"] = len(self._local_requests)

        if self._connected and self._redis:
            try:
                keys = await self._redis.keys(f"{self.REDIS_KEY_PREFIX}*")
                stats["redis_keys_tracked"] = len(keys)
            except Exception as e:
                stats["redis_error"] = str(e)

        return stats

    async def cleanup_expired(self) -> int:
        """Clean up expired local rate limit entries."""
        now = time.time()
        cleaned = 0

        async with self._lock:
            keys_to_remove = []
            for key, timestamps in self._local_requests.items():
                # Filter to only recent requests
                self._local_requests[key] = [
                    t for t in timestamps if now - t < self._default_window
                ]
                # Mark empty keys for removal
                if not self._local_requests[key]:
                    keys_to_remove.append(key)
                    cleaned += 1

            for key in keys_to_remove:
                del self._local_requests[key]

        return cleaned


# Global instance (initialized on startup)
rate_limiter: RedisRateLimiter | None = None


async def get_rate_limiter() -> RedisRateLimiter:
    """Get or create the global rate limiter instance."""
    global rate_limiter
    if rate_limiter is None:
        from app.core.config import settings

        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        default_limit = getattr(settings, "RATE_LIMIT_DEFAULT_LIMIT", 60)
        default_window = getattr(settings, "RATE_LIMIT_DEFAULT_WINDOW", 60)
        rate_limiter = RedisRateLimiter(
            redis_url=redis_url,
            default_limit=default_limit,
            default_window=default_window,
        )
        await rate_limiter.connect()
    return rate_limiter


async def shutdown_rate_limiter() -> None:
    """Shutdown the global rate limiter."""
    global rate_limiter
    if rate_limiter:
        await rate_limiter.disconnect()
        rate_limiter = None
