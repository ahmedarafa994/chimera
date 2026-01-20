"""Redis-backed Session Storage for Project Chimera.

Task 1.8: Implements persistent session storage using Redis,
replacing the in-memory dictionary that loses data on restart.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime

from app.core.config import settings
from app.core.logging import logger


class SessionBackend(ABC):
    """Abstract base for session storage backends."""

    @abstractmethod
    async def get(self, session_id: str) -> dict | None:
        """Get session data by ID."""

    @abstractmethod
    async def set(self, session_id: str, data: dict, ttl_seconds: int = 3600) -> bool:
        """Store session data with TTL."""

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete session by ID."""

    @abstractmethod
    async def count(self) -> int:
        """Count active sessions."""


class InMemorySessionBackend(SessionBackend):
    """In-memory session storage (for development/testing)."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[dict, datetime]] = {}

    async def get(self, session_id: str) -> dict | None:
        if session_id in self._store:
            data, _created = self._store[session_id]
            return data
        return None

    async def set(self, session_id: str, data: dict, ttl_seconds: int = 3600) -> bool:
        self._store[session_id] = (data, datetime.utcnow())
        return True

    async def delete(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    async def count(self) -> int:
        return len(self._store)


class RedisSessionBackend(SessionBackend):
    """Redis-backed session storage for production."""

    SESSION_PREFIX = "chimera:session:"

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or settings.REDIS_URL
        self._client = None

    async def _get_client(self):
        """Lazy-initialize Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = await aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                logger.info("Redis session backend connected")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                raise
        return self._client

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.SESSION_PREFIX}{session_id}"

    async def get(self, session_id: str) -> dict | None:
        try:
            client = await self._get_client()
            data = await client.get(self._key(session_id))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None

    async def set(self, session_id: str, data: dict, ttl_seconds: int = 3600) -> bool:
        try:
            client = await self._get_client()
            await client.setex(self._key(session_id), ttl_seconds, json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False

    async def delete(self, session_id: str) -> bool:
        try:
            client = await self._get_client()
            result = await client.delete(self._key(session_id))
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False

    async def count(self) -> int:
        try:
            client = await self._get_client()
            keys = await client.keys(f"{self.SESSION_PREFIX}*")
            return len(keys)
        except Exception as e:
            logger.error(f"Redis COUNT error: {e}")
            return 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


def get_session_backend() -> SessionBackend:
    """Get the appropriate session backend based on configuration.
    Uses Redis if REDIS_URL is configured, otherwise falls back to in-memory.
    """
    if settings.REDIS_URL and settings.REDIS_URL != "redis://localhost:6379":
        logger.info("Using Redis session backend")
        return RedisSessionBackend()
    logger.info("Using in-memory session backend (development mode)")
    return InMemorySessionBackend()
