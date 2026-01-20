"""Session Selection Persistence Service.

Service for persisting and restoring provider/model selections across sessions.

Features:
- Database persistence with SQLAlchemy
- Redis caching for fast access
- Session expiration handling
- Migration from old selection format
- Optimistic concurrency control with version tracking

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md
- Global state: app/services/global_model_selection_state.py

"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.infrastructure.cache.selection_cache import SelectionCache

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class SelectionRecord(BaseModel):
    """Record representing a provider/model selection with versioning.

    Supports optimistic concurrency control via version field.
    """

    session_id: str = Field(..., description="Session identifier")
    provider: str = Field(..., description="Provider identifier")
    model: str = Field(..., description="Model identifier")
    version: int = Field(default=1, description="Version for optimistic concurrency")
    user_id: str | None = Field(None, description="User identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    expires_at: datetime | None = Field(None, description="Expiration timestamp")

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class SelectionSaveResult(BaseModel):
    """Result of a selection save operation."""

    success: bool
    record: SelectionRecord | None = None
    error: str | None = None
    version: int = 0


class SelectionLoadResult(BaseModel):
    """Result of a selection load operation."""

    found: bool
    record: SelectionRecord | None = None
    source: str = "none"  # "cache", "database", "default"


# =============================================================================
# Session Selection Persistence Service
# =============================================================================


class SessionSelectionPersistenceService:
    """Service for persisting and restoring provider/model selections across sessions.

    Features:
    - Database persistence with SQLAlchemy
    - Redis caching for fast access
    - Session expiration handling
    - Migration from old selection format
    - Optimistic concurrency control

    Usage:
        >>> service = SessionSelectionPersistenceService()
        >>> await service.initialize()
        >>> record = await service.save_selection("sess_123", "openai", "gpt-4")
        >>> loaded = await service.load_selection("sess_123")
    """

    _instance: Optional["SessionSelectionPersistenceService"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    # Configuration
    DEFAULT_TTL_HOURS = 24
    CACHE_PREFIX = "selection:"

    def __new__(cls) -> "SessionSelectionPersistenceService":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the service."""
        if getattr(self, "_initialized", False):
            return

        self._cache: SelectionCache | None = None
        self._db_available = False
        self._cache_available = False
        self._default_provider = getattr(settings, "DEFAULT_PROVIDER", "openai")
        self._default_model = getattr(settings, "DEFAULT_MODEL", "gpt-4")

        self._initialized = True
        logger.info("SessionSelectionPersistenceService initialized")

    async def initialize(
        self,
        cache: SelectionCache | None = None,
    ) -> None:
        """Initialize the service with optional cache.

        Args:
            cache: Optional SelectionCache instance for Redis caching

        """
        if cache:
            self._cache = cache
            self._cache_available = True
            logger.info("Selection persistence service connected to cache")

        self._db_available = True
        logger.info("Selection persistence service initialized")

    # -------------------------------------------------------------------------
    # Core CRUD Operations
    # -------------------------------------------------------------------------

    async def save_selection(
        self,
        session_id: str,
        provider: str,
        model: str,
        user_id: str | None = None,
        metadata: dict | None = None,
        db_session: AsyncSession | None = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> SelectionSaveResult:
        """Save a provider/model selection for a session.

        Features:
        - Atomic database update with version increment
        - Cache invalidation and update
        - Automatic expiration setting

        Args:
            session_id: Session identifier
            provider: Provider identifier
            model: Model identifier
            user_id: Optional user identifier
            metadata: Optional additional metadata
            db_session: Database session for persistence
            ttl_hours: Time-to-live in hours

        Returns:
            SelectionSaveResult with success status and record

        """
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(hours=ttl_hours)

            # Create or update record
            record = SelectionRecord(
                session_id=session_id,
                provider=provider,
                model=model,
                user_id=user_id,
                selection_metadata=metadata or {},
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                version=1,
            )

            # Save to database if session provided
            if db_session:
                record = await self._save_to_database(record, db_session)

            # Save to cache
            if self._cache_available and self._cache:
                await self._cache.set(
                    session_id,
                    record,
                    ttl=ttl_hours * 3600,
                )

            logger.debug(
                f"Selection saved: {session_id} -> {provider}/{model} (version={record.version})",
            )

            return SelectionSaveResult(
                success=True,
                record=record,
                version=record.version,
            )

        except Exception as e:
            logger.exception(f"Error saving selection: {e}")
            return SelectionSaveResult(
                success=False,
                error=str(e),
            )

    async def load_selection(
        self,
        session_id: str,
        user_id: str | None = None,
        db_session: AsyncSession | None = None,
    ) -> SelectionLoadResult:
        """Load a provider/model selection for a session.

        Resolution order:
        1. Check Redis cache (fast path)
        2. Load from database if not cached
        3. Return None if not found

        Args:
            session_id: Session identifier
            user_id: Optional user identifier for user-specific selections
            db_session: Database session for loading from DB

        Returns:
            SelectionLoadResult with record if found

        """
        # Try cache first
        if self._cache_available and self._cache:
            cached = await self._cache.get(session_id)
            if cached:
                logger.debug(f"Selection cache hit: {session_id}")
                return SelectionLoadResult(
                    found=True,
                    record=cached,
                    source="cache",
                )

        # Try database
        if db_session:
            record = await self._load_from_database(session_id, user_id, db_session)
            if record:
                # Populate cache
                if self._cache_available and self._cache:
                    await self._cache.set(
                        session_id,
                        record,
                        ttl=self.DEFAULT_TTL_HOURS * 3600,
                    )

                logger.debug(f"Selection loaded from database: {session_id}")
                return SelectionLoadResult(
                    found=True,
                    record=record,
                    source="database",
                )

        logger.debug(f"Selection not found: {session_id}")
        return SelectionLoadResult(found=False, source="none")

    async def delete_selection(
        self,
        session_id: str,
        db_session: AsyncSession | None = None,
    ) -> bool:
        """Delete a selection for a session.

        Args:
            session_id: Session identifier
            db_session: Database session for deletion

        Returns:
            True if deleted, False otherwise

        """
        try:
            # Delete from cache
            if self._cache_available and self._cache:
                await self._cache.delete(session_id)

            # Delete from database
            if db_session:
                await self._delete_from_database(session_id, db_session)

            logger.info(f"Selection deleted: {session_id}")
            return True

        except Exception as e:
            logger.exception(f"Error deleting selection: {e}")
            return False

    async def get_user_selections(
        self,
        user_id: str,
        db_session: AsyncSession,
        limit: int = 100,
    ) -> list[SelectionRecord]:
        """Get all selections for a user.

        Args:
            user_id: User identifier
            db_session: Database session
            limit: Maximum number of records to return

        Returns:
            List of SelectionRecord objects

        """
        try:
            from app.infrastructure.database.models import SelectionModel

            query = (
                select(SelectionModel)
                .where(SelectionModel.user_id == user_id)
                .order_by(SelectionModel.updated_at.desc())
                .limit(limit)
            )

            result = await db_session.execute(query)
            records = result.scalars().all()

            return [
                SelectionRecord(
                    session_id=r.session_id,
                    provider=r.provider_id,
                    model=r.model_id,
                    user_id=r.user_id,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                    version=getattr(r, "version", 1),
                    metadata=getattr(r, "selection_metadata", {}),
                )
                for r in records
            ]

        except Exception as e:
            logger.exception(f"Error getting user selections: {e}")
            return []

    async def cleanup_expired_sessions(
        self,
        db_session: AsyncSession,
        max_age_hours: int = 24,
    ) -> int:
        """Clean up expired session selections.

        Args:
            db_session: Database session
            max_age_hours: Maximum age in hours

        Returns:
            Number of records deleted

        """
        try:
            from app.infrastructure.database.models import SelectionModel

            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

            # Delete expired records
            query = delete(SelectionModel).where(SelectionModel.updated_at < cutoff_time)

            result = await db_session.execute(query)
            await db_session.commit()

            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} expired selections")

            return deleted_count

        except Exception as e:
            logger.exception(f"Error cleaning up expired sessions: {e}")
            await db_session.rollback()
            return 0

    # -------------------------------------------------------------------------
    # Database Operations
    # -------------------------------------------------------------------------

    async def _save_to_database(
        self,
        record: SelectionRecord,
        db_session: AsyncSession,
    ) -> SelectionRecord:
        """Save selection to database with upsert logic."""
        try:
            from app.infrastructure.database.models import SelectionModel

            # Check for existing record
            query = select(SelectionModel).where(SelectionModel.session_id == record.session_id)
            if record.user_id:
                query = query.where(SelectionModel.user_id == record.user_id)

            result = await db_session.execute(query)
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing record with version increment
                existing.provider_id = record.provider
                existing.model_id = record.model
                existing.updated_at = datetime.utcnow()

                # Increment version
                current_version = getattr(existing, "version", 0)
                if hasattr(existing, "version"):
                    existing.version = current_version + 1

                record.version = current_version + 1
                record.created_at = existing.created_at
                record.updated_at = existing.updated_at
            else:
                # Create new record
                db_record = SelectionModel(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    provider_id=record.provider,
                    model_id=record.model,
                )
                db_session.add(db_record)

            await db_session.commit()
            return record

        except Exception as e:
            logger.exception(f"Database save error: {e}")
            await db_session.rollback()
            raise

    async def _load_from_database(
        self,
        session_id: str,
        user_id: str | None,
        db_session: AsyncSession,
    ) -> SelectionRecord | None:
        """Load selection from database."""
        try:
            from app.infrastructure.database.models import SelectionModel

            query = select(SelectionModel).where(SelectionModel.session_id == session_id)
            if user_id:
                query = query.where(SelectionModel.user_id == user_id)

            result = await db_session.execute(query)
            db_record = result.scalar_one_or_none()

            if db_record:
                return SelectionRecord(
                    session_id=db_record.session_id,
                    provider=db_record.provider_id,
                    model=db_record.model_id,
                    user_id=db_record.user_id,
                    created_at=db_record.created_at,
                    updated_at=db_record.updated_at,
                    version=getattr(db_record, "version", 1),
                    metadata=getattr(db_record, "metadata", {}),
                )

            return None

        except Exception as e:
            logger.exception(f"Database load error: {e}")
            return None

    async def _delete_from_database(
        self,
        session_id: str,
        db_session: AsyncSession,
    ) -> None:
        """Delete selection from database."""
        try:
            from app.infrastructure.database.models import SelectionModel

            query = delete(SelectionModel).where(SelectionModel.session_id == session_id)
            await db_session.execute(query)
            await db_session.commit()

        except Exception as e:
            logger.exception(f"Database delete error: {e}")
            await db_session.rollback()

    # -------------------------------------------------------------------------
    # Migration Support
    # -------------------------------------------------------------------------

    async def migrate_legacy_selection(
        self,
        old_format: dict,
        db_session: AsyncSession | None = None,
    ) -> SelectionRecord | None:
        """Migrate from old selection format to new format.

        Handles various legacy formats:
        - {"provider_id": ..., "model_id": ...}
        - {"selected_provider": ..., "selected_model": ...}
        - {"provider": ..., "model": ...}

        Args:
            old_format: Dictionary with old selection format
            db_session: Optional database session

        Returns:
            SelectionRecord if migration successful

        """
        try:
            # Extract provider
            provider = (
                old_format.get("provider_id")
                or old_format.get("selected_provider")
                or old_format.get("provider")
                or self._default_provider
            )

            # Extract model
            model = (
                old_format.get("model_id")
                or old_format.get("selected_model")
                or old_format.get("model")
                or self._default_model
            )

            # Extract session_id
            session_id = (
                old_format.get("session_id") or old_format.get("session") or old_format.get("id")
            )

            if not session_id:
                logger.warning("Cannot migrate selection without session_id")
                return None

            # Extract user_id
            user_id = old_format.get("user_id") or old_format.get("user")

            # Extract metadata
            metadata = {}
            if "preferences" in old_format:
                metadata["legacy_preferences"] = old_format["preferences"]
            if "model_settings" in old_format:
                metadata["legacy_model_settings"] = old_format["model_settings"]

            # Save migrated record
            result = await self.save_selection(
                session_id=session_id,
                provider=provider,
                model=model,
                user_id=user_id,
                metadata=metadata,
                db_session=db_session,
            )

            if result.success:
                logger.info(f"Migrated legacy selection: {session_id} -> {provider}/{model}")
                return result.record

            return None

        except Exception as e:
            logger.exception(f"Migration error: {e}")
            return None

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    async def batch_load_selections(
        self,
        session_ids: list[str],
        db_session: AsyncSession,
    ) -> dict[str, SelectionRecord]:
        """Batch load selections for multiple sessions.

        Args:
            session_ids: List of session identifiers
            db_session: Database session

        Returns:
            Dictionary mapping session_id to SelectionRecord

        """
        try:
            from app.infrastructure.database.models import SelectionModel

            # Check cache first
            results: dict[str, SelectionRecord] = {}
            cache_misses: list[str] = []

            if self._cache_available and self._cache:
                for sid in session_ids:
                    cached = await self._cache.get(sid)
                    if cached:
                        results[sid] = cached
                    else:
                        cache_misses.append(sid)
            else:
                cache_misses = session_ids

            # Load remaining from database
            if cache_misses:
                query = select(SelectionModel).where(SelectionModel.session_id.in_(cache_misses))
                result = await db_session.execute(query)
                records = result.scalars().all()

                for r in records:
                    record = SelectionRecord(
                        session_id=r.session_id,
                        provider=r.provider_id,
                        model=r.model_id,
                        user_id=r.user_id,
                        created_at=r.created_at,
                        updated_at=r.updated_at,
                        version=getattr(r, "version", 1),
                        metadata=getattr(r, "selection_metadata", {}),
                    )
                    results[r.session_id] = record

                    # Populate cache
                    if self._cache_available and self._cache:
                        await self._cache.set(
                            r.session_id,
                            record,
                            ttl=self.DEFAULT_TTL_HOURS * 3600,
                        )

            return results

        except Exception as e:
            logger.exception(f"Batch load error: {e}")
            return {}


# =============================================================================
# Dependency Injection
# =============================================================================


def get_selection_persistence_service() -> SessionSelectionPersistenceService:
    """FastAPI dependency for SessionSelectionPersistenceService.

    Returns:
        The singleton service instance

    """
    return SessionSelectionPersistenceService()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SelectionLoadResult",
    "SelectionRecord",
    "SelectionSaveResult",
    "SessionSelectionPersistenceService",
    "get_selection_persistence_service",
]
