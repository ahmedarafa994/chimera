"""
Selection Recovery Service for Provider/Model Selection System.

Service for recovering selection state after failures.

Features:
- Automatic recovery from database
- Fallback to defaults on missing selection
- Audit logging for recovery events
- Migration support for legacy selections

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md
- Persistence service: app/services/session_selection_persistence.py
"""

import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Recovery Data Models
# =============================================================================


class RecoveryResult(BaseModel):
    """Result of a selection recovery operation."""

    success: bool = Field(..., description="Whether recovery succeeded")
    session_id: str = Field(..., description="Session that was recovered")
    provider: str = Field(..., description="Recovered provider")
    model: str = Field(..., description="Recovered model")
    source: str = Field(
        default="default", description="Source of recovery (database, cache, default, migration)"
    )
    was_migrated: bool = Field(
        default=False, description="Whether data was migrated from legacy format"
    )
    error: str | None = Field(None, description="Error if recovery failed")


class RecoveryAuditEntry(BaseModel):
    """Audit log entry for recovery operations."""

    session_id: str
    user_id: str | None = None
    recovery_type: str  # "automatic", "manual", "migration"
    source: str  # "database", "cache", "default", "migration"
    provider: str
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: dict[str, Any] = Field(default_factory=dict)


class LegacySelectionFormat(BaseModel):
    """Represents various legacy selection formats."""

    # Format 1: old user_model_preferences format
    selected_provider: str | None = None
    selected_model: str | None = None

    # Format 2: simple key-value format
    provider: str | None = None
    model: str | None = None

    # Format 3: ID-based format
    provider_id: str | None = None
    model_id: str | None = None

    # Session/user identifiers
    session_id: str | None = None
    session: str | None = None
    id: str | None = None
    user_id: str | None = None
    user: str | None = None

    # Metadata
    preferences: dict | None = None
    model_settings: dict | None = None

    @property
    def resolved_provider(self) -> str | None:
        """Get provider from any format."""
        return self.provider_id or self.selected_provider or self.provider

    @property
    def resolved_model(self) -> str | None:
        """Get model from any format."""
        return self.model_id or self.selected_model or self.model

    @property
    def resolved_session_id(self) -> str | None:
        """Get session ID from any format."""
        return self.session_id or self.session or self.id

    @property
    def resolved_user_id(self) -> str | None:
        """Get user ID from any format."""
        return self.user_id or self.user


# =============================================================================
# Selection Recovery Service
# =============================================================================


class SelectionRecoveryService:
    """
    Service for recovering selection state after failures.

    Features:
    - Automatic recovery from database
    - Fallback to defaults on missing selection
    - Audit logging for recovery events
    - Migration support for legacy selections

    Usage:
        >>> recovery_service = SelectionRecoveryService()
        >>> result = await recovery_service.recover_selection("sess_123")
        >>> print(f"Recovered: {result.provider}/{result.model}")
    """

    _instance: Optional["SelectionRecoveryService"] = None

    def __new__(cls) -> "SelectionRecoveryService":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the recovery service."""
        if getattr(self, "_initialized", False):
            return

        self._default_provider = getattr(settings, "DEFAULT_PROVIDER", "openai")
        self._default_model = getattr(settings, "DEFAULT_MODEL", "gpt-4")
        self._persistence = None
        self._cache = None
        self._audit_log: list[RecoveryAuditEntry] = []
        self._max_audit_entries = 1000

        self._initialized = True
        logger.info("SelectionRecoveryService initialized")

    async def initialize(
        self,
        persistence=None,
        cache=None,
    ) -> None:
        """
        Initialize with persistence and cache services.

        Args:
            persistence: SessionSelectionPersistenceService instance
            cache: SelectionCache instance
        """
        self._persistence = persistence
        self._cache = cache
        logger.info("SelectionRecoveryService dependencies initialized")

    # -------------------------------------------------------------------------
    # Recovery Operations
    # -------------------------------------------------------------------------

    async def recover_selection(
        self,
        session_id: str,
        user_id: str | None = None,
        db_session: AsyncSession | None = None,
        use_defaults: bool = True,
    ) -> RecoveryResult:
        """
        Recover selection state for a session.

        Recovery order:
        1. Try to load from cache
        2. Try to load from database
        3. Try to find user's default selection
        4. Fall back to system defaults

        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            db_session: Database session for persistence operations
            use_defaults: Whether to fall back to defaults

        Returns:
            RecoveryResult with recovered selection
        """
        logger.info(f"Starting recovery for session: {session_id}")

        # Try cache first
        if self._cache:
            try:
                cached = await self._cache.get(session_id)
                if cached:
                    result = RecoveryResult(
                        success=True,
                        session_id=session_id,
                        provider=cached.provider,
                        model=cached.model,
                        source="cache",
                    )
                    self._log_recovery(result, user_id, "automatic")
                    return result
            except Exception as e:
                logger.warning(f"Cache recovery failed: {e}")

        # Try database
        if self._persistence and db_session:
            try:
                load_result = await self._persistence.load_selection(
                    session_id, user_id, db_session
                )
                if load_result.found and load_result.record:
                    result = RecoveryResult(
                        success=True,
                        session_id=session_id,
                        provider=load_result.record.provider,
                        model=load_result.record.model,
                        source="database",
                    )

                    # Update cache
                    if self._cache:
                        from app.infrastructure.cache.selection_cache import CachedSelection

                        await self._cache.set(
                            session_id,
                            CachedSelection(
                                session_id=session_id,
                                provider=load_result.record.provider,
                                model=load_result.record.model,
                                version=load_result.record.version,
                                user_id=user_id,
                            ),
                        )

                    self._log_recovery(result, user_id, "automatic")
                    return result

            except Exception as e:
                logger.warning(f"Database recovery failed: {e}")

        # Try to find user's default selection
        if user_id and db_session:
            try:
                user_default = await self._get_user_default_selection(user_id, db_session)
                if user_default:
                    result = RecoveryResult(
                        success=True,
                        session_id=session_id,
                        provider=user_default[0],
                        model=user_default[1],
                        source="user_default",
                    )
                    self._log_recovery(result, user_id, "automatic")
                    return result
            except Exception as e:
                logger.warning(f"User default recovery failed: {e}")

        # Fall back to system defaults
        if use_defaults:
            result = RecoveryResult(
                success=True,
                session_id=session_id,
                provider=self._default_provider,
                model=self._default_model,
                source="default",
            )
            self._log_recovery(result, user_id, "automatic")
            return result

        # Recovery failed
        return RecoveryResult(
            success=False,
            session_id=session_id,
            provider=self._default_provider,
            model=self._default_model,
            source="none",
            error="No selection found and defaults not allowed",
        )

    async def recover_from_legacy(
        self,
        old_format: dict,
        db_session: AsyncSession | None = None,
    ) -> RecoveryResult:
        """
        Recover selection from legacy format.

        Handles migration from various old formats to new format.

        Args:
            old_format: Dictionary with old selection format
            db_session: Database session for persistence

        Returns:
            RecoveryResult with migrated selection
        """
        try:
            # Parse legacy format
            legacy = LegacySelectionFormat(**old_format)

            # Get session ID
            session_id = legacy.resolved_session_id
            if not session_id:
                return RecoveryResult(
                    success=False,
                    session_id="unknown",
                    provider=self._default_provider,
                    model=self._default_model,
                    source="none",
                    error="No session ID in legacy format",
                )

            # Get provider and model
            provider = legacy.resolved_provider or self._default_provider
            model = legacy.resolved_model or self._default_model
            user_id = legacy.resolved_user_id

            # Build metadata from legacy fields
            metadata = {}
            if legacy.preferences:
                metadata["legacy_preferences"] = legacy.preferences
            if legacy.model_settings:
                metadata["legacy_model_settings"] = legacy.model_settings

            # Save migrated selection
            if self._persistence and db_session:
                await self._persistence.save_selection(
                    session_id=session_id,
                    provider=provider,
                    model=model,
                    user_id=user_id,
                    metadata=metadata,
                    db_session=db_session,
                )

            result = RecoveryResult(
                success=True,
                session_id=session_id,
                provider=provider,
                model=model,
                source="migration",
                was_migrated=True,
            )

            self._log_recovery(
                result,
                user_id,
                "migration",
                {
                    "original_format": old_format,
                },
            )

            logger.info(f"Migrated legacy selection: {session_id} -> " f"{provider}/{model}")

            return result

        except Exception as e:
            logger.error(f"Legacy migration failed: {e}")
            return RecoveryResult(
                success=False,
                session_id=old_format.get("session_id", "unknown"),
                provider=self._default_provider,
                model=self._default_model,
                source="none",
                error=str(e),
            )

    async def migrate_legacy_selection(
        self,
        old_format: dict,
    ):
        """
        Migrate from old selection format to new format.

        Alias for recover_from_legacy for API compatibility.

        Args:
            old_format: Dictionary with old selection format

        Returns:
            SelectionRecord if migration successful, None otherwise
        """
        from app.services.session_selection_persistence import SelectionRecord

        result = await self.recover_from_legacy(old_format)

        if result.success:
            return SelectionRecord(
                session_id=result.session_id,
                provider=result.provider,
                model=result.model,
            )

        return None

    # -------------------------------------------------------------------------
    # Batch Recovery
    # -------------------------------------------------------------------------

    async def recover_batch(
        self,
        session_ids: list[str],
        user_id: str | None = None,
        db_session: AsyncSession | None = None,
    ) -> dict[str, RecoveryResult]:
        """
        Recover selections for multiple sessions.

        Args:
            session_ids: List of session identifiers
            user_id: Optional user identifier
            db_session: Database session

        Returns:
            Dictionary mapping session_id to RecoveryResult
        """
        results: dict[str, RecoveryResult] = {}

        for session_id in session_ids:
            results[session_id] = await self.recover_selection(session_id, user_id, db_session)

        return results

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _get_user_default_selection(
        self,
        user_id: str,
        db_session: AsyncSession,
    ) -> tuple[str, str] | None:
        """Get user's default provider/model selection."""
        try:
            from sqlalchemy import select

            from app.infrastructure.database.models import UserModelPreference

            query = select(UserModelPreference).where(UserModelPreference.user_id == user_id)
            result = await db_session.execute(query)
            preference = result.scalar_one_or_none()

            if preference:
                return (preference.selected_provider, preference.selected_model)

            return None

        except Exception as e:
            logger.warning(f"Error getting user default: {e}")
            return None

    def _log_recovery(
        self,
        result: RecoveryResult,
        user_id: str | None,
        recovery_type: str,
        details: dict | None = None,
    ) -> None:
        """Log recovery operation for audit."""
        entry = RecoveryAuditEntry(
            session_id=result.session_id,
            user_id=user_id,
            recovery_type=recovery_type,
            source=result.source,
            provider=result.provider,
            model=result.model,
            details=details or {},
        )

        self._audit_log.append(entry)

        # Trim audit log if too large
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries :]

        logger.debug(f"Recovery logged: {result.session_id} " f"({result.source}, {recovery_type})")

    # -------------------------------------------------------------------------
    # Audit Access
    # -------------------------------------------------------------------------

    def get_recovery_audit(
        self,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[RecoveryAuditEntry]:
        """
        Get recovery audit log entries.

        Args:
            session_id: Optional filter by session
            limit: Maximum entries to return

        Returns:
            List of RecoveryAuditEntry
        """
        entries = self._audit_log

        if session_id:
            entries = [e for e in entries if e.session_id == session_id]

        return entries[-limit:]

    def get_recovery_stats(self) -> dict[str, Any]:
        """
        Get recovery statistics.

        Returns:
            Dictionary with recovery statistics
        """
        if not self._audit_log:
            return {
                "total_recoveries": 0,
                "by_source": {},
                "by_type": {},
                "migrations": 0,
            }

        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}
        migrations = 0

        for entry in self._audit_log:
            by_source[entry.source] = by_source.get(entry.source, 0) + 1
            by_type[entry.recovery_type] = by_type.get(entry.recovery_type, 0) + 1
            if entry.recovery_type == "migration":
                migrations += 1

        return {
            "total_recoveries": len(self._audit_log),
            "by_source": by_source,
            "by_type": by_type,
            "migrations": migrations,
        }

    def clear_audit_log(self) -> int:
        """
        Clear the audit log.

        Returns:
            Number of entries cleared
        """
        count = len(self._audit_log)
        self._audit_log.clear()
        logger.info(f"Cleared {count} audit log entries")
        return count


# =============================================================================
# Dependency Injection
# =============================================================================


def get_selection_recovery_service() -> SelectionRecoveryService:
    """
    FastAPI dependency for SelectionRecoveryService.

    Returns:
        The singleton service instance
    """
    return SelectionRecoveryService()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LegacySelectionFormat",
    "RecoveryAuditEntry",
    "RecoveryResult",
    "SelectionRecoveryService",
    "get_selection_recovery_service",
]
