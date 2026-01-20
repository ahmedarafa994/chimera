"""Stream Context Manager for Project Chimera.

This module provides context management for streaming operations,
ensuring selection consistency during streaming and proper cleanup.

Features:
- Maintains provider/model selection during entire stream
- Provides locking mechanism to prevent mid-stream changes
- Handles proper cleanup on errors or cancellation
- Collects metrics for stream duration and performance

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md Section 4.2
- Unified streaming: app/services/unified_streaming_service.py

"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StreamSession:
    """Represents an active streaming session.

    Contains all context needed for a streaming operation including
    the locked provider/model selection and timing information.
    """

    stream_id: str
    session_id: str | None
    user_id: str | None
    provider: str
    model: str
    started_at: datetime
    lock: asyncio.Lock
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tracking fields
    chunks_sent: int = 0
    bytes_sent: int = 0
    last_activity: datetime | None = None
    error: str | None = None
    cancelled: bool = False
    completed: bool = False

    def record_chunk(self, size: int) -> None:
        """Record a chunk being sent."""
        self.chunks_sent += 1
        self.bytes_sent += size
        self.last_activity = datetime.utcnow()

    def mark_error(self, error: str) -> None:
        """Mark the session as having an error."""
        self.error = error
        self.completed = True

    def mark_cancelled(self) -> None:
        """Mark the session as cancelled."""
        self.cancelled = True
        self.completed = True

    def mark_complete(self) -> None:
        """Mark the session as successfully completed."""
        self.completed = True

    @property
    def duration_ms(self) -> float:
        """Get the session duration in milliseconds."""
        end = self.last_activity or datetime.utcnow()
        delta = end - self.started_at
        return delta.total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "stream_id": self.stream_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "provider": self.provider,
            "model": self.model,
            "started_at": self.started_at.isoformat(),
            "chunks_sent": self.chunks_sent,
            "bytes_sent": self.bytes_sent,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "cancelled": self.cancelled,
            "completed": self.completed,
            "metadata": self.metadata,
        }


@dataclass
class StreamLock:
    """Lock entry for preventing selection changes during streaming.

    When a stream is active for a session, this lock prevents
    provider/model selection changes until the stream completes.
    """

    stream_id: str
    session_id: str
    provider: str
    model: str
    locked_at: datetime
    lock: asyncio.Lock

    def is_expired(self, max_duration_seconds: int = 300) -> bool:
        """Check if the lock has expired (default 5 minutes)."""
        elapsed = (datetime.utcnow() - self.locked_at).total_seconds()
        return elapsed > max_duration_seconds


# =============================================================================
# Streaming Context Manager
# =============================================================================


class StreamingContext:
    """Context manager that maintains provider/model selection during streaming.

    Ensures:
    - Selection doesn't change mid-stream
    - Proper cleanup on errors
    - Metrics collection for stream duration
    - Prevention of concurrent conflicting streams

    Usage:
        >>> context = StreamingContext()
        >>> async with context.locked_stream(
        ...     session_id="sess_123",
        ...     provider="openai",
        ...     model="gpt-4"
        ... ) as session:
        ...     # Stream within locked context
        ...     async for chunk in generate_stream():
        ...         session.record_chunk(len(chunk))
        ...         yield chunk
    """

    _instance: Optional["StreamingContext"] = None

    def __new__(cls) -> "StreamingContext":
        """Singleton pattern for global access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the streaming context manager."""
        if self._initialized:
            return

        # Active stream sessions by stream_id
        self._active_sessions: dict[str, StreamSession] = {}

        # Selection locks by session_id
        self._selection_locks: dict[str, StreamLock] = {}

        # Global lock for managing locks
        self._global_lock = asyncio.Lock()

        # Callbacks for stream events
        self._on_stream_start: list[Callable] = []
        self._on_stream_end: list[Callable] = []

        # Configuration
        self._max_concurrent_per_session = 3
        self._lock_timeout_seconds = 300  # 5 minutes

        self._initialized = True
        logger.info("StreamingContext initialized")

    @asynccontextmanager
    async def locked_stream(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        provider: str = "",
        model: str = "",
        stream_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamSession]:
        """Create a locked streaming session context.

        The selection (provider/model) is locked for the duration of the
        stream to ensure consistency. Other attempts to change selection
        for this session will be blocked or deferred.

        Args:
            session_id: Session identifier (None for anonymous)
            user_id: User identifier
            provider: The provider to lock for this stream
            model: The model to lock for this stream
            stream_id: Optional custom stream ID
            metadata: Optional metadata for the session

        Yields:
            StreamSession: The active stream session context

        Raises:
            StreamLockError: If unable to acquire lock
            asyncio.CancelledError: If stream is cancelled

        """
        stream_id = stream_id or f"stream_{uuid.uuid4().hex[:12]}"
        session_lock = asyncio.Lock()

        session = StreamSession(
            stream_id=stream_id,
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            model=model,
            started_at=datetime.utcnow(),
            lock=session_lock,
            metadata=metadata or {},
        )

        # Acquire selection lock if session provided
        selection_lock: StreamLock | None = None
        if session_id:
            selection_lock = await self._acquire_selection_lock(
                stream_id=stream_id,
                session_id=session_id,
                provider=provider,
                model=model,
            )

        try:
            # Register active session
            async with self._global_lock:
                self._active_sessions[stream_id] = session

            # Notify listeners
            await self._notify_start(session)

            logger.debug(
                f"Stream session started: stream_id={stream_id}, "
                f"provider={provider}, model={model}",
            )

            yield session

            # Mark complete if no error
            if not session.error and not session.cancelled:
                session.mark_complete()

        except asyncio.CancelledError:
            session.mark_cancelled()
            logger.warning(f"Stream session cancelled: {stream_id}")
            raise

        except Exception as e:
            session.mark_error(str(e))
            logger.exception(f"Stream session error: {stream_id}, error={e}")
            raise

        finally:
            # Cleanup
            async with self._global_lock:
                self._active_sessions.pop(stream_id, None)

            # Release selection lock
            if selection_lock and session_id:
                await self._release_selection_lock(session_id, stream_id)

            # Notify listeners
            await self._notify_end(session)

            logger.debug(
                f"Stream session ended: stream_id={stream_id}, "
                f"chunks={session.chunks_sent}, "
                f"duration_ms={session.duration_ms:.1f}",
            )

    async def _acquire_selection_lock(
        self,
        stream_id: str,
        session_id: str,
        provider: str,
        model: str,
    ) -> StreamLock:
        """Acquire a selection lock for a session.

        This prevents selection changes during streaming.
        """
        async with self._global_lock:
            # Check for existing lock
            existing = self._selection_locks.get(session_id)
            if existing:
                # Check if it's expired
                if existing.is_expired(self._lock_timeout_seconds):
                    logger.warning(
                        f"Releasing expired lock: session={session_id}, "
                        f"stream={existing.stream_id}",
                    )
                    del self._selection_locks[session_id]
                # Allow if same provider/model, otherwise wait
                elif existing.provider == provider and existing.model == model:
                    logger.debug(f"Reusing existing lock: session={session_id}")
                    return existing
                else:
                    # Different selection - this shouldn't happen often
                    logger.warning(
                        f"Selection conflict: session={session_id}, "
                        f"existing={existing.provider}/{existing.model}, "
                        f"requested={provider}/{model}",
                    )

            # Create new lock
            lock = StreamLock(
                stream_id=stream_id,
                session_id=session_id,
                provider=provider,
                model=model,
                locked_at=datetime.utcnow(),
                lock=asyncio.Lock(),
            )
            self._selection_locks[session_id] = lock

            logger.debug(
                f"Selection lock acquired: session={session_id}, "
                f"provider={provider}, model={model}",
            )

            return lock

    async def _release_selection_lock(
        self,
        session_id: str,
        stream_id: str,
    ) -> bool:
        """Release a selection lock for a session."""
        async with self._global_lock:
            lock = self._selection_locks.get(session_id)
            if lock and lock.stream_id == stream_id:
                del self._selection_locks[session_id]
                logger.debug(f"Selection lock released: session={session_id}")
                return True
            return False

    def is_selection_locked(self, session_id: str) -> bool:
        """Check if selection is locked for a session."""
        lock = self._selection_locks.get(session_id)
        if lock:
            return not lock.is_expired(self._lock_timeout_seconds)
        return False

    def get_locked_selection(self, session_id: str) -> tuple[str, str] | None:
        """Get the locked selection for a session.

        Returns:
            Tuple of (provider, model) if locked, None otherwise

        """
        lock = self._selection_locks.get(session_id)
        if lock and not lock.is_expired(self._lock_timeout_seconds):
            return (lock.provider, lock.model)
        return None

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def get_active_sessions(self) -> list[dict[str, Any]]:
        """Get information about all active streaming sessions."""
        return [s.to_dict() for s in self._active_sessions.values()]

    def get_session(self, stream_id: str) -> StreamSession | None:
        """Get a specific streaming session."""
        return self._active_sessions.get(stream_id)

    def get_sessions_for_session_id(self, session_id: str) -> list[StreamSession]:
        """Get all active streams for a session."""
        return [s for s in self._active_sessions.values() if s.session_id == session_id]

    def count_active_streams(self, session_id: str | None = None) -> int:
        """Count active streams, optionally filtered by session."""
        if session_id is None:
            return len(self._active_sessions)
        return len(self.get_sessions_for_session_id(session_id))

    # -------------------------------------------------------------------------
    # Event Callbacks
    # -------------------------------------------------------------------------

    def on_stream_start(self, callback: Callable) -> None:
        """Register a callback for stream start events."""
        self._on_stream_start.append(callback)

    def on_stream_end(self, callback: Callable) -> None:
        """Register a callback for stream end events."""
        self._on_stream_end.append(callback)

    async def _notify_start(self, session: StreamSession) -> None:
        """Notify listeners of stream start."""
        for callback in self._on_stream_start:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(session)
                else:
                    callback(session)
            except Exception as e:
                logger.exception(f"Error in stream start callback: {e}")

    async def _notify_end(self, session: StreamSession) -> None:
        """Notify listeners of stream end."""
        for callback in self._on_stream_end:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(session)
                else:
                    callback(session)
            except Exception as e:
                logger.exception(f"Error in stream end callback: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def cleanup_expired_locks(self) -> int:
        """Clean up expired selection locks.

        Returns:
            Number of locks cleaned up

        """
        cleaned = 0
        async with self._global_lock:
            expired = [
                sid
                for sid, lock in self._selection_locks.items()
                if lock.is_expired(self._lock_timeout_seconds)
            ]
            for sid in expired:
                del self._selection_locks[sid]
                cleaned += 1

        if cleaned:
            logger.info(f"Cleaned up {cleaned} expired selection locks")

        return cleaned


# =============================================================================
# Factory Functions
# =============================================================================


_streaming_context: StreamingContext | None = None


def get_streaming_context() -> StreamingContext:
    """Get the singleton StreamingContext instance.

    Returns:
        The global StreamingContext instance

    """
    global _streaming_context
    if _streaming_context is None:
        _streaming_context = StreamingContext()
    return _streaming_context


async def get_streaming_context_async() -> StreamingContext:
    """Async factory for StreamingContext.

    Use this as a FastAPI dependency.

    Returns:
        The global StreamingContext instance

    """
    return get_streaming_context()


# =============================================================================
# Exceptions
# =============================================================================


class StreamLockError(Exception):
    """Raised when unable to acquire a stream lock."""

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        existing_provider: str | None = None,
        existing_model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.existing_provider = existing_provider
        self.existing_model = existing_model


class StreamCancelledError(Exception):
    """Raised when a stream is cancelled."""

    def __init__(
        self,
        message: str,
        stream_id: str,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stream_id = stream_id
        self.reason = reason


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "StreamCancelledError",
    "StreamLock",
    # Exceptions
    "StreamLockError",
    # Data classes
    "StreamSession",
    # Main class
    "StreamingContext",
    # Factory functions
    "get_streaming_context",
    "get_streaming_context_async",
]
