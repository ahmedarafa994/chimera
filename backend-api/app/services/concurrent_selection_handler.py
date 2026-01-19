"""
Concurrent Selection Handler for Provider/Model Selection System.

Handles concurrent requests with consistent provider/model selection.

Ensures:
- Same session gets same selection during concurrent requests
- Selection changes are atomic
- No race conditions when updating selections
- Proper locking for critical sections
- Optimistic concurrency control with version tracking

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md
- Persistence service: app/services/session_selection_persistence.py
- Selection cache: app/infrastructure/cache/selection_cache.py
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from app.core.config import settings

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class SelectionLock(BaseModel):
    """
    Represents a lock on a session's selection.

    Used to prevent concurrent modifications to the same selection.
    """

    lock_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique lock identifier"
    )
    session_id: str = Field(..., description="Session being locked")
    acquired_at: datetime = Field(
        default_factory=datetime.utcnow, description="When lock was acquired"
    )
    expires_at: datetime = Field(..., description="When lock expires")
    owner: str | None = Field(None, description="Owner identifier (request ID)")

    def is_expired(self) -> bool:
        """Check if lock has expired."""
        return datetime.utcnow() > self.expires_at


class SelectionUpdateResult(BaseModel):
    """Result of an atomic selection update."""

    success: bool = Field(..., description="Whether update succeeded")
    provider: str = Field(..., description="Provider after update")
    model: str = Field(..., description="Model after update")
    version: int = Field(..., description="Version after update")
    previous_version: int | None = Field(None, description="Version before update")
    conflict: bool = Field(default=False, description="Whether version conflict occurred")
    error: str | None = Field(None, description="Error message if failed")


class SelectionWithVersion(BaseModel):
    """Selection data with version for optimistic concurrency."""

    session_id: str
    provider: str
    model: str
    version: int
    user_id: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Lock Manager
# =============================================================================


class SelectionLockManager:
    """
    Manages distributed locks for selection operations.

    Uses Redis for distributed locking with fallback to local locks.
    Implements lock acquisition with timeout and automatic expiration.
    """

    LOCK_PREFIX = "chimera:selection:lock:"
    DEFAULT_LOCK_TIMEOUT = 5.0  # seconds
    DEFAULT_LOCK_TTL = 10  # seconds

    def __init__(self, redis_url: str | None = None):
        """
        Initialize the lock manager.

        Args:
            redis_url: Redis connection URL
        """
        self._redis_url = redis_url or getattr(settings, "REDIS_URL", "redis://localhost:6379")
        self._client: Redis | None = None
        self._local_locks: dict[str, SelectionLock] = {}
        self._local_lock = asyncio.Lock()
        self._use_local = False
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize Redis connection."""
        if self._initialized:
            return not self._use_local

        try:
            import redis.asyncio as aioredis

            self._client = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._client.ping()

            self._initialized = True
            self._use_local = False
            logger.info("SelectionLockManager connected to Redis")
            return True

        except Exception as e:
            logger.warning(f"Redis unavailable for locks: {e}. Using local locks.")
            self._use_local = True
            self._initialized = True
            return False

    async def acquire(
        self,
        session_id: str,
        timeout: float = DEFAULT_LOCK_TIMEOUT,
        ttl: int = DEFAULT_LOCK_TTL,
        owner: str | None = None,
    ) -> SelectionLock | None:
        """
        Acquire a lock on a session's selection.

        Args:
            session_id: Session to lock
            timeout: Maximum time to wait for lock
            ttl: Lock time-to-live in seconds
            owner: Optional owner identifier

        Returns:
            SelectionLock if acquired, None if timeout
        """
        if not self._initialized:
            await self.initialize()

        lock_id = str(uuid.uuid4())
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            acquired = await self._try_acquire(session_id, lock_id, ttl, owner)

            if acquired:
                from datetime import timedelta

                lock = SelectionLock(
                    lock_id=lock_id,
                    session_id=session_id,
                    acquired_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(seconds=ttl),
                    owner=owner,
                )
                logger.debug(f"Lock acquired: {session_id} ({lock_id})")
                return lock

            # Wait before retry
            await asyncio.sleep(0.1)

        logger.warning(f"Lock acquisition timeout: {session_id}")
        return None

    async def _try_acquire(
        self,
        session_id: str,
        lock_id: str,
        ttl: int,
        owner: str | None,
    ) -> bool:
        """Try to acquire lock (single attempt)."""
        if self._use_local:
            return await self._try_acquire_local(session_id, lock_id, ttl, owner)

        try:
            key = f"{self.LOCK_PREFIX}{session_id}"

            # Use SET NX EX for atomic acquire
            result = await self._client.set(
                key,
                lock_id,
                nx=True,
                ex=ttl,
            )

            return result is not None

        except Exception as e:
            logger.error(f"Redis lock error: {e}")
            return await self._try_acquire_local(session_id, lock_id, ttl, owner)

    async def _try_acquire_local(
        self,
        session_id: str,
        lock_id: str,
        ttl: int,
        owner: str | None,
    ) -> bool:
        """Try to acquire local lock."""
        async with self._local_lock:
            existing = self._local_locks.get(session_id)

            # Check if existing lock is expired
            if existing and not existing.is_expired():
                return False

            # Create new lock
            from datetime import timedelta

            self._local_locks[session_id] = SelectionLock(
                lock_id=lock_id,
                session_id=session_id,
                acquired_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=ttl),
                owner=owner,
            )
            return True

    async def release(self, lock: SelectionLock) -> bool:
        """
        Release a lock.

        Args:
            lock: Lock to release

        Returns:
            True if released, False if lock was invalid/expired
        """
        if self._use_local:
            return await self._release_local(lock)

        try:
            key = f"{self.LOCK_PREFIX}{lock.session_id}"

            # Only release if we own the lock
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """

            result = await self._client.eval(script, 1, key, lock.lock_id)

            if result:
                logger.debug(f"Lock released: {lock.session_id}")
                return True

            logger.warning(f"Lock release failed (not owner): {lock.session_id}")
            return False

        except Exception as e:
            logger.error(f"Redis unlock error: {e}")
            return await self._release_local(lock)

    async def _release_local(self, lock: SelectionLock) -> bool:
        """Release local lock."""
        async with self._local_lock:
            existing = self._local_locks.get(lock.session_id)

            if existing and existing.lock_id == lock.lock_id:
                del self._local_locks[lock.session_id]
                logger.debug(f"Local lock released: {lock.session_id}")
                return True

            return False

    async def extend(
        self,
        lock: SelectionLock,
        additional_ttl: int = DEFAULT_LOCK_TTL,
    ) -> bool:
        """
        Extend a lock's TTL.

        Args:
            lock: Lock to extend
            additional_ttl: Additional time in seconds

        Returns:
            True if extended
        """
        if self._use_local:
            async with self._local_lock:
                existing = self._local_locks.get(lock.session_id)
                if existing and existing.lock_id == lock.lock_id:
                    from datetime import timedelta

                    existing.expires_at = datetime.utcnow() + timedelta(seconds=additional_ttl)
                    return True
            return False

        try:
            key = f"{self.LOCK_PREFIX}{lock.session_id}"

            # Only extend if we own the lock
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("expire", KEYS[1], ARGV[2])
            else
                return 0
            end
            """

            result = await self._client.eval(script, 1, key, lock.lock_id, additional_ttl)
            return bool(result)

        except Exception as e:
            logger.error(f"Lock extend error: {e}")
            return False


# =============================================================================
# Concurrent Selection Handler
# =============================================================================


class ConcurrentSelectionHandler:
    """
    Handles concurrent requests with consistent provider/model selection.

    Ensures:
    - Same session gets same selection during concurrent requests
    - Selection changes are atomic
    - No race conditions when updating selections
    - Proper locking for critical sections

    Usage:
        >>> handler = ConcurrentSelectionHandler()
        >>> async with handler.selection_lock("sess_123") as lock:
        ...     result = await handler.update_selection_atomic(
        ...         "sess_123", "openai", "gpt-4"
        ...     )
    """

    _instance: Optional["ConcurrentSelectionHandler"] = None

    def __new__(cls) -> "ConcurrentSelectionHandler":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the handler."""
        if getattr(self, "_initialized", False):
            return

        self._lock_manager = SelectionLockManager()
        self._cache = None
        self._persistence = None
        self._initialized = True
        logger.info("ConcurrentSelectionHandler initialized")

    async def initialize(
        self,
        cache=None,
        persistence=None,
    ) -> None:
        """
        Initialize with cache and persistence services.

        Args:
            cache: SelectionCache instance
            persistence: SessionSelectionPersistenceService instance
        """
        await self._lock_manager.initialize()
        self._cache = cache
        self._persistence = persistence
        logger.info("ConcurrentSelectionHandler fully initialized")

    # -------------------------------------------------------------------------
    # Lock Context Manager
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def selection_lock(
        self,
        session_id: str,
        timeout: float = 5.0,
        owner: str | None = None,
    ):
        """
        Context manager for acquiring a selection lock.

        Usage:
            async with handler.selection_lock("sess_123") as lock:
                # Critical section
                ...

        Args:
            session_id: Session to lock
            timeout: Lock acquisition timeout
            owner: Optional owner identifier

        Yields:
            SelectionLock if acquired

        Raises:
            TimeoutError: If lock cannot be acquired
        """
        lock = await self._lock_manager.acquire(
            session_id,
            timeout=timeout,
            owner=owner,
        )

        if not lock:
            raise TimeoutError(f"Could not acquire selection lock for {session_id}")

        try:
            yield lock
        finally:
            await self._lock_manager.release(lock)

    async def acquire_selection_lock(
        self,
        session_id: str,
        timeout: float = 5.0,
        owner: str | None = None,
    ) -> SelectionLock:
        """
        Acquire a lock on a session's selection.

        Args:
            session_id: Session to lock
            timeout: Maximum time to wait
            owner: Optional owner identifier

        Returns:
            SelectionLock

        Raises:
            TimeoutError: If lock cannot be acquired
        """
        lock = await self._lock_manager.acquire(
            session_id,
            timeout=timeout,
            owner=owner,
        )

        if not lock:
            raise TimeoutError(f"Could not acquire selection lock for {session_id}")

        return lock

    async def release_selection_lock(self, lock: SelectionLock) -> bool:
        """
        Release a selection lock.

        Args:
            lock: Lock to release

        Returns:
            True if released
        """
        return await self._lock_manager.release(lock)

    # -------------------------------------------------------------------------
    # Atomic Operations
    # -------------------------------------------------------------------------

    async def update_selection_atomic(
        self,
        session_id: str,
        provider: str,
        model: str,
        expected_version: int | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        db_session=None,
    ) -> SelectionUpdateResult:
        """
        Atomically update a selection with optimistic concurrency.

        If expected_version is provided, the update only succeeds if
        the current version matches. Otherwise, the update always succeeds.

        Args:
            session_id: Session identifier
            provider: New provider
            model: New model
            expected_version: Expected current version (optional)
            user_id: User identifier
            metadata: Additional metadata
            db_session: Database session

        Returns:
            SelectionUpdateResult with success status
        """
        try:
            async with self.selection_lock(session_id):
                # Get current version if needed
                current = await self.get_selection_with_version(session_id)
                current_version = current[1] if current else 0

                # Check version if expected
                if expected_version is not None:
                    if current_version != expected_version:
                        return SelectionUpdateResult(
                            success=False,
                            provider=current[0].provider if current else provider,
                            model=current[0].model if current else model,
                            version=current_version,
                            previous_version=expected_version,
                            conflict=True,
                            error=(
                                f"Version conflict: expected {expected_version}, "
                                f"found {current_version}"
                            ),
                        )

                # Calculate new version
                new_version = current_version + 1

                # Update cache
                if self._cache:
                    from app.infrastructure.cache.selection_cache import CachedSelection

                    selection = CachedSelection(
                        session_id=session_id,
                        provider=provider,
                        model=model,
                        version=new_version,
                        user_id=user_id,
                        metadata=metadata or {},
                    )

                    await self._cache.set(session_id, selection)

                    # Publish change
                    old_selection = (
                        CachedSelection(
                            session_id=session_id,
                            provider=current[0].provider,
                            model=current[0].model,
                            version=current_version,
                        )
                        if current
                        else None
                    )
                    await self._cache.publish_change(session_id, selection, old_selection)

                # Update persistence
                if self._persistence and db_session:
                    await self._persistence.save_selection(
                        session_id=session_id,
                        provider=provider,
                        model=model,
                        user_id=user_id,
                        metadata=metadata,
                        db_session=db_session,
                    )

                logger.info(
                    f"Selection updated atomically: {session_id} -> "
                    f"{provider}/{model} (v{new_version})"
                )

                return SelectionUpdateResult(
                    success=True,
                    provider=provider,
                    model=model,
                    version=new_version,
                    previous_version=current_version,
                    conflict=False,
                )

        except TimeoutError as e:
            logger.warning(f"Update timeout: {session_id}")
            return SelectionUpdateResult(
                success=False,
                provider=provider,
                model=model,
                version=0,
                error=str(e),
            )

        except Exception as e:
            logger.error(f"Atomic update error: {e}")
            return SelectionUpdateResult(
                success=False,
                provider=provider,
                model=model,
                version=0,
                error=str(e),
            )

    async def get_selection_with_version(
        self,
        session_id: str,
    ) -> tuple[SelectionWithVersion, int] | None:
        """
        Get selection with its current version.

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (SelectionWithVersion, version) or None
        """
        # Try cache first
        if self._cache:
            cached = await self._cache.get(session_id)
            if cached:
                return (
                    SelectionWithVersion(
                        session_id=cached.session_id,
                        provider=cached.provider,
                        model=cached.model,
                        version=cached.version,
                        user_id=cached.user_id,
                    ),
                    cached.version,
                )

        # Try persistence
        if self._persistence:
            result = await self._persistence.load_selection(session_id)
            if result.found and result.record:
                return (
                    SelectionWithVersion(
                        session_id=result.record.session_id,
                        provider=result.record.provider,
                        model=result.record.model,
                        version=result.record.version,
                        user_id=result.record.user_id,
                    ),
                    result.record.version,
                )

        return None

    # -------------------------------------------------------------------------
    # Concurrent Read with Consistency
    # -------------------------------------------------------------------------

    async def get_consistent_selection(
        self,
        session_id: str,
        require_lock: bool = False,
    ) -> SelectionWithVersion | None:
        """
        Get selection with read consistency.

        If require_lock is True, acquires a short-lived lock to ensure
        the read is not affected by concurrent writes.

        Args:
            session_id: Session identifier
            require_lock: Whether to acquire lock for read

        Returns:
            SelectionWithVersion or None
        """
        if require_lock:
            try:
                async with self.selection_lock(session_id, timeout=2.0):
                    result = await self.get_selection_with_version(session_id)
                    return result[0] if result else None
            except TimeoutError:
                # Fall back to non-locked read
                pass

        result = await self.get_selection_with_version(session_id)
        return result[0] if result else None

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    async def batch_get_selections(
        self,
        session_ids: list[str],
    ) -> dict[str, SelectionWithVersion]:
        """
        Get selections for multiple sessions.

        Args:
            session_ids: List of session identifiers

        Returns:
            Dictionary mapping session_id to SelectionWithVersion
        """
        results: dict[str, SelectionWithVersion] = {}

        if self._cache:
            cached = await self._cache.mget(session_ids)
            for session_id, selection in cached.items():
                if selection:
                    results[session_id] = SelectionWithVersion(
                        session_id=selection.session_id,
                        provider=selection.provider,
                        model=selection.model,
                        version=selection.version,
                        user_id=selection.user_id,
                    )

        return results


# =============================================================================
# Dependency Injection
# =============================================================================


def get_concurrent_selection_handler() -> ConcurrentSelectionHandler:
    """
    FastAPI dependency for ConcurrentSelectionHandler.

    Returns:
        The singleton handler instance
    """
    return ConcurrentSelectionHandler()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ConcurrentSelectionHandler",
    "SelectionLock",
    "SelectionLockManager",
    "SelectionUpdateResult",
    "SelectionWithVersion",
    "get_concurrent_selection_handler",
]
