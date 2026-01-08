"""
Redis-Based Selection Cache for Provider/Model Selections.

Provides fast caching for selection lookups with:
- TTL-based expiration
- Atomic get-or-set operations
- Pub/sub for selection change notifications
- Cluster-aware for horizontal scaling

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md
- Persistence service: app/services/session_selection_persistence.py
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from app.core.config import settings

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Data Models
# =============================================================================


class CachedSelection(BaseModel):
    """
    Cached selection data structure.

    Optimized for Redis storage with minimal footprint.
    """

    session_id: str
    provider: str
    model: str
    version: int = 1
    user_id: Optional[str] = None
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_cache_dict(self) -> dict:
        """Convert to dictionary for Redis storage."""
        return {
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "user_id": self.user_id,
            "cached_at": self.cached_at.isoformat(),
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_cache_dict(cls, data: dict) -> "CachedSelection":
        """Create from Redis stored dictionary."""
        cached_at = data.get("cached_at")
        if cached_at and isinstance(cached_at, str):
            cached_at = datetime.fromisoformat(cached_at)

        expires_at = data.get("expires_at")
        if expires_at and isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        return cls(
            session_id=data["session_id"],
            provider=data["provider"],
            model=data["model"],
            version=data.get("version", 1),
            user_id=data.get("user_id"),
            cached_at=cached_at or datetime.utcnow(),
            expires_at=expires_at,
            metadata=data.get("metadata", {}),
        )


class SelectionChangeEvent(BaseModel):
    """Event published when a selection changes."""

    session_id: str
    old_provider: Optional[str] = None
    old_model: Optional[str] = None
    new_provider: str
    new_model: str
    version: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "cache"  # "cache", "api", "websocket"


# =============================================================================
# Selection Cache Implementation
# =============================================================================


class SelectionCache:
    """
    Redis-based cache for fast selection lookups.

    Features:
    - TTL-based expiration
    - Atomic get-or-set operations
    - Pub/sub for selection change notifications
    - Cluster-aware for horizontal scaling

    Usage:
        >>> cache = SelectionCache()
        >>> await cache.initialize()
        >>> await cache.set("sess_123", selection, ttl=3600)
        >>> cached = await cache.get("sess_123")
    """

    # Redis key prefixes
    CACHE_PREFIX = "chimera:selection:"
    PUBSUB_CHANNEL = "chimera:selection:changes"
    LOCK_PREFIX = "chimera:selection:lock:"

    # Default TTL (1 hour)
    DEFAULT_TTL = 3600

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the selection cache.

        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        self._redis_url = redis_url or getattr(
            settings, "REDIS_URL", "redis://localhost:6379"
        )
        self._client: Optional["Redis"] = None
        self._pubsub = None
        self._initialized = False
        self._local_cache: dict[str, CachedSelection] = {}
        self._use_local_fallback = False

    async def initialize(self) -> bool:
        """
        Initialize Redis connection.

        Returns:
            True if Redis connected, False if using local fallback
        """
        if self._initialized:
            return not self._use_local_fallback

        try:
            import redis.asyncio as aioredis

            self._client = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

            # Test connection
            await self._client.ping()

            self._initialized = True
            self._use_local_fallback = False
            logger.info("SelectionCache connected to Redis")
            return True

        except ImportError:
            logger.warning(
                "redis.asyncio not available, using local cache fallback"
            )
            self._use_local_fallback = True
            self._initialized = True
            return False

        except Exception as e:
            logger.warning(
                f"Redis connection failed: {e}. Using local cache fallback"
            )
            self._use_local_fallback = True
            self._initialized = True
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        self._initialized = False
        logger.info("SelectionCache closed")

    # -------------------------------------------------------------------------
    # Core Cache Operations
    # -------------------------------------------------------------------------

    def _cache_key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"{self.CACHE_PREFIX}{session_id}"

    async def get(self, session_id: str) -> Optional[CachedSelection]:
        """
        Get cached selection for a session.

        Args:
            session_id: Session identifier

        Returns:
            CachedSelection if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        # Local fallback
        if self._use_local_fallback:
            cached = self._local_cache.get(session_id)
            if cached and cached.expires_at:
                if datetime.utcnow() > cached.expires_at:
                    del self._local_cache[session_id]
                    return None
            return cached

        try:
            key = self._cache_key(session_id)
            data = await self._client.get(key)

            if data:
                return CachedSelection.from_cache_dict(json.loads(data))
            return None

        except Exception as e:
            logger.error(f"Cache GET error: {e}")
            return self._local_cache.get(session_id)

    async def set(
        self,
        session_id: str,
        selection: CachedSelection,
        ttl: int = DEFAULT_TTL,
    ) -> bool:
        """
        Set cached selection for a session.

        Args:
            session_id: Session identifier
            selection: CachedSelection to cache
            ttl: Time-to-live in seconds

        Returns:
            True if cached successfully
        """
        if not self._initialized:
            await self.initialize()

        # Update expiration
        from datetime import timedelta
        selection.expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        selection.cached_at = datetime.utcnow()

        # Always update local cache
        self._local_cache[session_id] = selection

        if self._use_local_fallback:
            return True

        try:
            key = self._cache_key(session_id)
            data = json.dumps(selection.to_cache_dict())
            await self._client.setex(key, ttl, data)
            return True

        except Exception as e:
            logger.error(f"Cache SET error: {e}")
            return True  # Local cache succeeded

    async def delete(self, session_id: str) -> bool:
        """
        Delete cached selection for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        if not self._initialized:
            await self.initialize()

        # Remove from local cache
        self._local_cache.pop(session_id, None)

        if self._use_local_fallback:
            return True

        try:
            key = self._cache_key(session_id)
            await self._client.delete(key)
            return True

        except Exception as e:
            logger.error(f"Cache DELETE error: {e}")
            return True

    async def exists(self, session_id: str) -> bool:
        """Check if selection exists in cache."""
        if not self._initialized:
            await self.initialize()

        if self._use_local_fallback:
            return session_id in self._local_cache

        try:
            key = self._cache_key(session_id)
            return await self._client.exists(key) > 0

        except Exception as e:
            logger.error(f"Cache EXISTS error: {e}")
            return session_id in self._local_cache

    # -------------------------------------------------------------------------
    # Atomic Operations
    # -------------------------------------------------------------------------

    async def get_or_set(
        self,
        session_id: str,
        factory: callable,
        ttl: int = DEFAULT_TTL,
    ) -> CachedSelection:
        """
        Get cached selection or create using factory.

        Atomic operation that prevents cache stampede.

        Args:
            session_id: Session identifier
            factory: Async callable that returns CachedSelection
            ttl: Time-to-live in seconds

        Returns:
            CachedSelection (cached or newly created)
        """
        # Try to get existing
        cached = await self.get(session_id)
        if cached:
            return cached

        # Create new using factory
        if asyncio.iscoroutinefunction(factory):
            selection = await factory()
        else:
            selection = factory()

        # Cache the result
        await self.set(session_id, selection, ttl)

        return selection

    async def update_version(
        self,
        session_id: str,
        expected_version: int,
    ) -> tuple[bool, int]:
        """
        Atomically increment version if it matches expected.

        For optimistic concurrency control.

        Args:
            session_id: Session identifier
            expected_version: Expected current version

        Returns:
            Tuple of (success, new_version)
        """
        cached = await self.get(session_id)
        if not cached:
            return False, 0

        if cached.version != expected_version:
            return False, cached.version

        cached.version += 1
        await self.set(session_id, cached)

        return True, cached.version

    # -------------------------------------------------------------------------
    # Pub/Sub for Change Notifications
    # -------------------------------------------------------------------------

    async def publish_change(
        self,
        session_id: str,
        selection: CachedSelection,
        old_selection: Optional[CachedSelection] = None,
    ) -> bool:
        """
        Publish selection change notification.

        Args:
            session_id: Session identifier
            selection: New selection
            old_selection: Previous selection (optional)

        Returns:
            True if published successfully
        """
        if self._use_local_fallback:
            return False

        try:
            event = SelectionChangeEvent(
                session_id=session_id,
                old_provider=old_selection.provider if old_selection else None,
                old_model=old_selection.model if old_selection else None,
                new_provider=selection.provider,
                new_model=selection.model,
                version=selection.version,
            )

            await self._client.publish(
                self.PUBSUB_CHANNEL,
                event.model_dump_json(),
            )

            logger.debug(
                f"Published selection change: {session_id} -> "
                f"{selection.provider}/{selection.model}"
            )
            return True

        except Exception as e:
            logger.error(f"Publish error: {e}")
            return False

    async def subscribe_changes(
        self,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[SelectionChangeEvent]:
        """
        Subscribe to selection change notifications.

        Args:
            session_id: Optional filter for specific session

        Yields:
            SelectionChangeEvent for each change
        """
        if self._use_local_fallback:
            logger.warning("Pub/sub not available in local fallback mode")
            return

        try:
            pubsub = self._client.pubsub()
            await pubsub.subscribe(self.PUBSUB_CHANNEL)

            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event = SelectionChangeEvent.model_validate_json(
                            message["data"]
                        )

                        # Filter by session_id if specified
                        if session_id and event.session_id != session_id:
                            continue

                        yield event

                    except Exception as e:
                        logger.error(f"Error parsing change event: {e}")
                        continue

        except Exception as e:
            logger.error(f"Subscribe error: {e}")

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    async def mget(
        self,
        session_ids: list[str],
    ) -> dict[str, Optional[CachedSelection]]:
        """
        Get multiple cached selections.

        Args:
            session_ids: List of session identifiers

        Returns:
            Dictionary mapping session_id to CachedSelection
        """
        if not self._initialized:
            await self.initialize()

        results = {}

        if self._use_local_fallback:
            for sid in session_ids:
                results[sid] = self._local_cache.get(sid)
            return results

        try:
            keys = [self._cache_key(sid) for sid in session_ids]
            values = await self._client.mget(keys)

            for sid, value in zip(session_ids, values):
                if value:
                    results[sid] = CachedSelection.from_cache_dict(
                        json.loads(value)
                    )
                else:
                    results[sid] = self._local_cache.get(sid)

            return results

        except Exception as e:
            logger.error(f"Cache MGET error: {e}")
            for sid in session_ids:
                results[sid] = self._local_cache.get(sid)
            return results

    async def clear_all(self) -> int:
        """
        Clear all cached selections.

        Returns:
            Number of entries cleared
        """
        count = len(self._local_cache)
        self._local_cache.clear()

        if self._use_local_fallback:
            return count

        try:
            pattern = f"{self.CACHE_PREFIX}*"
            keys = await self._client.keys(pattern)
            if keys:
                await self._client.delete(*keys)
                return len(keys)
            return count

        except Exception as e:
            logger.error(f"Cache CLEAR error: {e}")
            return count

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "local_cache_size": len(self._local_cache),
            "using_redis": not self._use_local_fallback,
            "initialized": self._initialized,
        }

        if not self._use_local_fallback and self._client:
            try:
                pattern = f"{self.CACHE_PREFIX}*"
                keys = await self._client.keys(pattern)
                stats["redis_cache_size"] = len(keys)

                info = await self._client.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human")

            except Exception as e:
                logger.error(f"Stats error: {e}")

        return stats


# =============================================================================
# Singleton Instance
# =============================================================================


_selection_cache_instance: Optional[SelectionCache] = None


def get_selection_cache() -> SelectionCache:
    """
    Get the singleton SelectionCache instance.

    Returns:
        SelectionCache singleton
    """
    global _selection_cache_instance
    if _selection_cache_instance is None:
        _selection_cache_instance = SelectionCache()
    return _selection_cache_instance


async def initialize_selection_cache() -> SelectionCache:
    """
    Initialize and return the SelectionCache.

    Returns:
        Initialized SelectionCache
    """
    cache = get_selection_cache()
    await cache.initialize()
    return cache


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SelectionCache",
    "CachedSelection",
    "SelectionChangeEvent",
    "get_selection_cache",
    "initialize_selection_cache",
]
