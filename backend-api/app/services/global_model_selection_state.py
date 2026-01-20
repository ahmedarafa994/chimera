"""Global Model Selection State Service.

Centralized state management for provider/model selection.

This module implements the backend centralized state management for the
Dynamic AI Provider and Model Selection System. It provides persistent
storage of user model selections with thread-safe operations, database
persistence, and integration with FastAPI.

Key Features:
- Singleton pattern for process-wide state consistency
- Thread-safe selection updates using asyncio locks
- Database persistence using existing SelectionModel
- In-memory caching for fast access
- Default fallback when no selection exists
- Integration with existing SelectionContext for request-scoped propagation
- Integration with model_selection_service for global model selection

Architecture:
    GlobalModelSelectionState (Persistent Storage)
            ↓
    SelectionContext (Request-Scoped Context)
            ↓
    Service Layer (Uses selection from context)

Resolution Order (Four-Tier Hierarchy):
    1. Request Override (from headers/query params)
    2. Session Preference (from database/cache)
    3. Global Model Selection (from model_selection_service - user's UI selection)
    4. Static Default (from environment/config)

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md
- Selection context: app/core/selection_context.py
- Provider registry pattern: app/services/provider_registry.py
- Database models: app/infrastructure/database/models.py
- Model selection service: app/services/model_selection_service.py

"""

import asyncio
import logging
from contextvars import ContextVar, Token
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.models import Selection, SelectionScope

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for API/Service Layer
# =============================================================================


class ModelSelection(BaseModel):
    """Pydantic model representing a model selection.

    This is the primary data transfer object for selection operations,
    containing all necessary information about a provider/model selection.
    """

    provider: str = Field(..., description="Provider identifier (e.g., 'openai', 'anthropic')")
    model_id: str = Field(..., description="Model identifier within the provider")
    user_id: str | None = Field(None, description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Selection timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    scope: SelectionScope = Field(
        default=SelectionScope.SESSION,
        description="Selection scope level",
    )

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class SelectionContextData(BaseModel):
    """Pydantic model for request-scoped selection context.

    This model captures the complete context for a request, including
    the resolved selection and propagation information for tracing.
    """

    request_id: str = Field(..., description="Unique request identifier for tracing")
    selection: ModelSelection | None = Field(None, description="Resolved model selection")
    propagation_path: list[str] = Field(
        default_factory=list,
        description="Path of context propagation for debugging",
    )
    resolved_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the selection was resolved",
    )
    resolution_source: str = Field(
        default="default",
        description="Source of resolution (request, session, global)",
    )

    class Config:
        from_attributes = True


# =============================================================================
# Context Variables for Request-Scoped State
# =============================================================================


# Request context variable for selection state
_request_selection_context: ContextVar[SelectionContextData | None] = ContextVar(
    "request_selection_context",
    default=None,
)


# =============================================================================
# GlobalModelSelectionState Singleton Service
# =============================================================================


class GlobalModelSelectionState:
    """Singleton service for global model selection state management.

    This service provides centralized management of provider/model selections
    with the following features:

    - Thread-safe operations using asyncio locks
    - Database persistence for session-based selections
    - In-memory caching for fast access
    - Default fallback when no selection exists
    - Request-scoped context propagation

    Usage:
        >>> state = GlobalModelSelectionState()
        >>> selection = await state.get_current_selection("session_123")
        >>> await state.set_selection("session_123", "openai", "gpt-4")

    Thread Safety:
        All mutation operations are protected by asyncio.Lock to ensure
        thread-safety in concurrent environments.

    Caching Strategy:
        - Selections are cached in memory for fast access
        - Cache is invalidated on selection changes
        - Database is the source of truth for persistence
    """

    _instance: Optional["GlobalModelSelectionState"] = None
    _lock: asyncio.Lock = asyncio.Lock()
    _initialized: bool = False

    def __new__(cls) -> "GlobalModelSelectionState":
        """Implement singleton pattern.

        Ensures only one instance of GlobalModelSelectionState exists
        across the entire application.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the singleton instance.

        Only performs initialization on first instantiation.
        """
        if self._initialized:
            return

        # In-memory cache for selections: session_id -> ModelSelection
        self._cache: dict[str, ModelSelection] = {}

        # Lock for cache operations
        self._cache_lock = asyncio.Lock()

        # Default selection fallback
        self._default_provider = getattr(settings, "DEFAULT_PROVIDER", "deepseek")
        self._default_model = getattr(settings, "DEFAULT_MODEL", "deepseek-chat")

        # Registered event listeners
        self._event_listeners: list[callable] = []

        # Subscription system for WebSocket updates
        self._subscriptions: dict[
            str, tuple[str, callable]
        ] = {}  # subscription_id -> (user_id, callback)
        self._subscription_counter = 0

        self._initialized = True
        logger.info(
            f"GlobalModelSelectionState initialized with defaults: "
            f"{self._default_provider}/{self._default_model}",
        )

    # -------------------------------------------------------------------------
    # Core Selection Methods
    # -------------------------------------------------------------------------

    async def get_current_selection(
        self,
        session_id: str,
        user_id: str | None = None,
        db_session: AsyncSession | None = None,
    ) -> ModelSelection:
        """Get the current model selection for a session.

        Resolution order (Four-Tier Hierarchy):
        1. Check in-memory cache (session-level)
        2. Load from database if not cached (session-level)
        3. Check model_selection_service (global user selection from UI)
        4. Return static default selection if none exists

        Args:
            session_id: Session identifier
            user_id: Optional user identifier for user-specific selections
            db_session: Optional database session for persistence operations

        Returns:
            ModelSelection with the current or default selection

        Example:
            >>> selection = await state.get_current_selection("sess_123")
            >>> print(f"Using {selection.provider}/{selection.model_id}")

        """
        cache_key = self._get_cache_key(session_id, user_id)

        # Tier 1: Check cache first (session-level selections)
        async with self._cache_lock:
            if cache_key in self._cache:
                logger.debug(f"Cache hit for selection: {cache_key}")
                return self._cache[cache_key]

        # Tier 2: Try to load from database (session-level)
        if db_session:
            selection = await self._load_from_database(session_id, user_id, db_session)
            if selection:
                async with self._cache_lock:
                    self._cache[cache_key] = selection
                logger.debug(f"Loaded selection from database: {cache_key}")
                return selection

        # Tier 3: Check model_selection_service (global user selection from UI)
        global_selection = self._get_global_model_selection(session_id, user_id)
        if global_selection:
            logger.debug(
                f"Using global model selection: "
                f"{global_selection.provider}/{global_selection.model_id}",
            )
            return global_selection

        # Tier 4: Return static default selection
        default_selection = self._get_default_selection(session_id, user_id)
        logger.debug(f"Using static default selection for: {cache_key}")
        return default_selection

    def _get_global_model_selection(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> ModelSelection | None:
        """Get the global model selection from model_selection_service.

        This reads from the model_selection_service which persists the user's
        selected provider/model from the UI to a JSON file.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier

        Returns:
            ModelSelection if set, None otherwise

        """
        try:
            from app.services.model_selection_service import model_selection_service

            selection = model_selection_service.get_selection()
            if selection:
                provider_id = selection.provider
                model_id = selection.model

                logger.debug(f"Found global model selection: {provider_id}/{model_id}")
                return ModelSelection(
                    provider=provider_id,
                    model_id=model_id,
                    user_id=user_id,
                    session_id=session_id,
                    scope=SelectionScope.GLOBAL,
                )
            return None
        except ImportError as e:
            logger.warning(f"model_selection_service not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get global model selection: {e}")
            return None

    async def set_selection(
        self,
        session_id: str,
        provider: str,
        model_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        db_session: AsyncSession | None = None,
        persist: bool = True,
    ) -> ModelSelection:
        """Set the model selection for a session.

        This method updates both the in-memory cache and optionally
        persists to the database.

        Args:
            session_id: Session identifier
            provider: Provider identifier (e.g., 'openai', 'anthropic')
            model_id: Model identifier within the provider
            user_id: Optional user identifier
            metadata: Optional additional metadata
            db_session: Optional database session for persistence
            persist: Whether to persist to database (default: True)

        Returns:
            The created ModelSelection object

        Raises:
            ValueError: If provider or model_id is invalid

        Example:
            >>> selection = await state.set_selection(
            ...     "sess_123", "anthropic", "claude-3-opus"
            ... )

        """
        # Validate selection
        if not await self.validate_selection(provider, model_id):
            logger.warning(
                f"Selection validation warning for {provider}/{model_id}. "
                "Proceeding anyway as provider may be dynamically registered.",
            )

        selection = ModelSelection(
            provider=provider,
            model_id=model_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            scope=SelectionScope.SESSION,
        )

        cache_key = self._get_cache_key(session_id, user_id)

        # Update cache
        async with self._cache_lock:
            old_selection = self._cache.get(cache_key)
            self._cache[cache_key] = selection

        # Persist to database if requested
        if persist and db_session:
            await self._save_to_database(selection, db_session)

        # Notify listeners
        await self._notify_selection_change(old_selection, selection)

        logger.info(f"Selection updated: {cache_key} -> {provider}/{model_id}")

        return selection

    async def clear_selection(
        self,
        session_id: str,
        user_id: str | None = None,
        db_session: AsyncSession | None = None,
    ) -> bool:
        """Clear the selection for a session.

        Removes the selection from both cache and database, reverting
        to default selection behavior.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            db_session: Optional database session for persistence

        Returns:
            True if a selection was cleared, False if none existed

        """
        cache_key = self._get_cache_key(session_id, user_id)

        # Clear from cache
        async with self._cache_lock:
            old_selection = self._cache.pop(cache_key, None)

        # Clear from database
        if db_session:
            await self._delete_from_database(session_id, user_id, db_session)

        if old_selection:
            await self._notify_selection_change(old_selection, None)
            logger.info(f"Selection cleared: {cache_key}")
            return True

        return False

    # -------------------------------------------------------------------------
    # Request Context Methods
    # -------------------------------------------------------------------------

    def get_request_context(self) -> SelectionContextData | None:
        """Get the current request's selection context.

        This retrieves the selection context from the current request's
        contextvars, allowing access to the resolved selection anywhere
        in the request call stack.

        Returns:
            SelectionContextData if set, None otherwise

        """
        return _request_selection_context.get()

    def set_request_context(self, context: SelectionContextData) -> Token:
        """Set the selection context for the current request.

        This should be called at request entry (typically in middleware)
        to establish the selection context for the request.

        Args:
            context: The SelectionContextData to set

        Returns:
            Token for resetting the context

        """
        token = _request_selection_context.set(context)
        provider = context.selection.provider if context.selection else None
        model = context.selection.model_id if context.selection else None
        logger.debug(
            f"Request context set: request_id={context.request_id}, selection={provider}/{model}",
        )
        return token

    def clear_request_context(self, token: Token | None = None) -> None:
        """Clear the current request's selection context.

        Should be called at request completion (typically in middleware)
        to clean up the context.

        Args:
            token: Optional token from set_request_context for proper reset

        """
        if token:
            _request_selection_context.reset(token)
        else:
            _request_selection_context.set(None)
        logger.debug("Request context cleared")

    async def resolve_request_selection(
        self,
        request_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        request_override: tuple[str, str] | None = None,
        db_session: AsyncSession | None = None,
    ) -> SelectionContextData:
        """Resolve the selection for a request using four-tier hierarchy.

        Resolution order:
        1. Request override (from query params or headers)
        2. Session preference (from database/cache)
        3. Global model selection (from model_selection_service - user's UI selection)
        4. Static default (from settings/config)

        Args:
            request_id: Unique request identifier
            session_id: Session identifier
            user_id: User identifier
            request_override: Optional (provider, model_id) override
            db_session: Database session for loading session preferences

        Returns:
            SelectionContextData with resolved selection

        """
        resolution_source = "default"
        selection: ModelSelection | None = None

        # Tier 1: Request override
        if request_override:
            provider, model_id = request_override
            selection = ModelSelection(
                provider=provider,
                model_id=model_id,
                user_id=user_id,
                session_id=session_id or "request_override",
                scope=SelectionScope.REQUEST,
            )
            resolution_source = "request"
            logger.debug(f"Request override: {provider}/{model_id}")

        # Tier 2: Session preference (from DB/cache)
        if selection is None and session_id:
            cached = await self.get_current_selection(session_id, user_id, db_session)
            # Only use if it's a session-level selection (not global/default)
            if cached.scope == SelectionScope.SESSION:
                selection = cached
                resolution_source = "session"
                logger.debug(f"Session preference: {cached.provider}/{cached.model_id}")

        # Tier 3: Global model selection (from model_selection_service)
        if selection is None:
            global_sel = self._get_global_model_selection(session_id or "default", user_id)
            if global_sel:
                selection = global_sel
                resolution_source = "global_model_selection"
                logger.debug(f"Global model selection: {selection.provider}/{selection.model_id}")

        # Tier 4: Static default (from settings)
        if selection is None:
            selection = self._get_default_selection(session_id or "default", user_id)
            selection.scope = SelectionScope.GLOBAL
            resolution_source = "static_default"
            logger.debug(f"Static default: {selection.provider}/{selection.model_id}")

        context = SelectionContextData(
            request_id=request_id,
            selection=selection,
            propagation_path=[f"resolve:{resolution_source}"],
            resolution_source=resolution_source,
        )

        logger.debug(
            f"Selection resolved for request {request_id}: "
            f"{selection.provider}/{selection.model_id} "
            f"(source={resolution_source})",
        )

        return context

    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------

    async def validate_selection(self, provider: str, model_id: str) -> bool:
        """Validate that a provider/model combination is valid.

        This method checks against the registered providers and models.
        Note: Validation is advisory; unknown providers may be dynamically
        registered later.

        Args:
            provider: Provider identifier
            model_id: Model identifier

        Returns:
            True if valid, False otherwise

        """
        # Import here to avoid circular imports
        try:
            from app.services.provider_registry import ProviderRegistry

            registry = ProviderRegistry()
            provider_info = await registry.get_provider(provider)

            if not provider_info:
                logger.debug(f"Provider not found in registry: {provider}")
                return False

            # Check if model exists for provider
            if hasattr(provider_info, "models"):
                valid_models = [m.id if hasattr(m, "id") else m for m in provider_info.models]
                if model_id not in valid_models:
                    logger.debug(
                        f"Model {model_id} not found for provider "
                        f"{provider}. Valid: {valid_models}",
                    )
                    return False

            return True

        except ImportError:
            logger.warning("ProviderRegistry not available for validation")
            return True  # Allow if registry not available

        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return True  # Allow on error to avoid blocking

    # -------------------------------------------------------------------------
    # Subscription System for WebSocket Updates
    # -------------------------------------------------------------------------

    def subscribe_to_changes(
        self,
        user_id: str,
        callback: callable,
    ) -> str:
        """Subscribe to selection changes for a specific user.

        This is used by the WebSocket system to receive real-time updates
        when a user's selection changes.

        Args:
            user_id: User identifier to subscribe to
            callback: Callback function(selection: Selection) called on changes

        Returns:
            Subscription ID for later unsubscription

        """
        self._subscription_counter += 1
        subscription_id = f"sub_{user_id}_{self._subscription_counter}"
        self._subscriptions[subscription_id] = (user_id, callback)
        logger.debug(f"Added subscription {subscription_id} for user {user_id}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from selection changes.

        Args:
            subscription_id: The subscription ID returned by subscribe_to_changes

        Returns:
            True if unsubscribed successfully, False if not found

        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.debug(f"Removed subscription {subscription_id}")
            return True
        return False

    async def get_selection(self, user_id: str) -> Selection:
        """Get the current selection for a user.

        This is a convenience method for WebSocket handlers that provides
        a Selection domain object.

        Args:
            user_id: User identifier

        Returns:
            Selection domain object with current selection

        """
        # Try to get from cache or default
        model_selection = await self.get_current_selection(
            session_id=user_id,
            user_id=user_id,
        )

        # Convert to domain Selection (which uses provider_id, model_id)
        # IMPORTANT: Selection validates that GLOBAL scope cannot have user_id/session_id
        # so we must only include them for SESSION scope
        is_session_scope = model_selection.scope == SelectionScope.SESSION

        return Selection(
            provider_id=model_selection.provider,
            model_id=model_selection.model_id,
            scope=model_selection.scope,
            user_id=user_id if is_session_scope else None,
            session_id=model_selection.session_id if is_session_scope else None,
            created_at=model_selection.timestamp,
            updated_at=model_selection.timestamp,
        )

    async def _notify_subscribers(
        self,
        user_id: str,
        selection: "Selection",
    ) -> None:
        """Notify all subscribers for a user about a selection change.

        Args:
            user_id: User whose selection changed
            selection: New selection

        """
        for sub_id, (sub_user_id, callback) in list(self._subscriptions.items()):
            if sub_user_id == user_id:
                try:
                    callback(selection)
                except Exception as e:
                    logger.exception(f"Error in subscription callback {sub_id}: {e}")

    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------

    def add_selection_listener(self, callback: callable) -> None:
        """Add a listener for selection change events.

        Args:
            callback: Async callable(old_selection, new_selection)

        """
        self._event_listeners.append(callback)
        logger.debug(f"Added selection listener: {callback.__name__}")

    def remove_selection_listener(self, callback: callable) -> bool:
        """Remove a selection change listener.

        Args:
            callback: The callback to remove

        Returns:
            True if removed, False if not found

        """
        try:
            self._event_listeners.remove(callback)
            return True
        except ValueError:
            return False

    async def _notify_selection_change(
        self,
        old_selection: ModelSelection | None,
        new_selection: ModelSelection | None,
    ) -> None:
        """Notify all listeners of a selection change."""
        for listener in self._event_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(old_selection, new_selection)
                else:
                    listener(old_selection, new_selection)
            except Exception as e:
                logger.exception(f"Error in selection listener {listener.__name__}: {e}")

    # -------------------------------------------------------------------------
    # Database Operations
    # -------------------------------------------------------------------------

    async def _load_from_database(
        self,
        session_id: str,
        user_id: str | None,
        db_session: AsyncSession,
    ) -> ModelSelection | None:
        """Load selection from database using existing SelectionModel."""
        try:
            # Import here to avoid circular imports
            from app.infrastructure.database.models import SelectionModel

            query = select(SelectionModel).where(SelectionModel.session_id == session_id)
            if user_id:
                query = query.where(SelectionModel.user_id == user_id)

            result = await db_session.execute(query)
            db_record = result.scalar_one_or_none()

            if db_record:
                return ModelSelection(
                    provider=db_record.provider_id,
                    model_id=db_record.model_id,
                    user_id=db_record.user_id,
                    session_id=db_record.session_id,
                    timestamp=db_record.updated_at,
                    metadata={},
                    scope=SelectionScope.SESSION,
                )

            return None

        except Exception as e:
            logger.exception(f"Database load error: {e}")
            return None

    async def _save_to_database(
        self,
        selection: ModelSelection,
        db_session: AsyncSession,
    ) -> None:
        """Save selection to database using existing SelectionModel."""
        try:
            # Import here to avoid circular imports
            from app.infrastructure.database.models import SelectionModel

            # Check for existing record
            query = select(SelectionModel).where(SelectionModel.session_id == selection.session_id)
            if selection.user_id:
                query = query.where(SelectionModel.user_id == selection.user_id)

            result = await db_session.execute(query)
            db_record = result.scalar_one_or_none()

            if db_record:
                # Update existing
                db_record.provider_id = selection.provider
                db_record.model_id = selection.model_id
                db_record.updated_at = datetime.utcnow()
            else:
                # Create new
                db_record = SelectionModel(
                    user_id=selection.user_id,
                    session_id=selection.session_id,
                    provider_id=selection.provider,
                    model_id=selection.model_id,
                )
                db_session.add(db_record)

            await db_session.commit()
            logger.debug(f"Selection saved to database: {selection.session_id}")

        except Exception as e:
            logger.exception(f"Database save error: {e}")
            await db_session.rollback()
            raise

    async def _delete_from_database(
        self,
        session_id: str,
        user_id: str | None,
        db_session: AsyncSession,
    ) -> None:
        """Delete selection from database."""
        try:
            from sqlalchemy import delete

            from app.infrastructure.database.models import SelectionModel

            query = delete(SelectionModel).where(SelectionModel.session_id == session_id)
            if user_id:
                query = query.where(SelectionModel.user_id == user_id)

            await db_session.execute(query)
            await db_session.commit()
            logger.debug(f"Selection deleted from database: {session_id}")

        except Exception as e:
            logger.exception(f"Database delete error: {e}")
            await db_session.rollback()

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_cache_key(self, session_id: str, user_id: str | None = None) -> str:
        """Generate cache key from session and user IDs."""
        if user_id:
            return f"{user_id}:{session_id}"
        return session_id

    def _get_default_selection(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> ModelSelection:
        """Get the default selection."""
        return ModelSelection(
            provider=self._default_provider,
            model_id=self._default_model,
            user_id=user_id,
            session_id=session_id,
            scope=SelectionScope.GLOBAL,
        )

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    async def clear_cache(self) -> int:
        """Clear the in-memory cache.

        Returns:
            Number of entries cleared

        """
        async with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} entries")
            return count

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics

        """
        async with self._cache_lock:
            return {
                "size": len(self._cache),
                "keys": list(self._cache.keys()),
            }

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    async def load_all_sessions(
        self,
        db_session: AsyncSession,
        limit: int = 1000,
    ) -> int:
        """Preload selections from database into cache.

        Useful for warming the cache on startup.

        Args:
            db_session: Database session
            limit: Maximum number of records to load

        Returns:
            Number of selections loaded

        """
        try:
            from app.infrastructure.database.models import SelectionModel

            query = select(SelectionModel).limit(limit)
            result = await db_session.execute(query)
            records = result.scalars().all()

            count = 0
            async with self._cache_lock:
                for record in records:
                    cache_key = self._get_cache_key(record.session_id, record.user_id)

                    self._cache[cache_key] = ModelSelection(
                        provider=record.provider_id,
                        model_id=record.model_id,
                        user_id=record.user_id,
                        session_id=record.session_id,
                        timestamp=record.updated_at,
                        metadata={},
                        scope=SelectionScope.SESSION,
                    )
                    count += 1

            logger.info(f"Preloaded {count} selections into cache")
            return count

        except Exception as e:
            logger.exception(f"Error preloading selections: {e}")
            return 0


# =============================================================================
# FastAPI Dependency
# =============================================================================


def get_global_model_selection_state() -> GlobalModelSelectionState:
    """FastAPI dependency to get the GlobalModelSelectionState singleton.

    Usage in route:
        @app.get("/api/v1/models")
        async def get_models(
            state: GlobalModelSelectionState = Depends(
                get_global_model_selection_state
            )
        ):
            selection = await state.get_current_selection(session_id)
            ...

    Returns:
        The singleton GlobalModelSelectionState instance

    """
    return GlobalModelSelectionState()


async def get_model_selection_state() -> GlobalModelSelectionState:
    """Async FastAPI dependency for GlobalModelSelectionState.

    Use this when you need async initialization or when the dependency
    chain requires an async function.

    Returns:
        The singleton GlobalModelSelectionState instance

    """
    return GlobalModelSelectionState()


# =============================================================================
# Request Context Middleware Support
# =============================================================================


class ModelSelectionMiddlewareHelper:
    """Helper class for middleware integration.

    Provides utilities for setting up and tearing down request context
    in FastAPI middleware.
    """

    def __init__(self) -> None:
        self.state = GlobalModelSelectionState()

    async def setup_request_context(
        self,
        request_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        provider_override: str | None = None,
        model_override: str | None = None,
        db_session: AsyncSession | None = None,
    ) -> tuple[SelectionContextData, Token]:
        """Set up the selection context for a request.

        Args:
            request_id: Unique request identifier
            session_id: Session identifier from cookie/header
            user_id: User identifier from auth
            provider_override: Provider override from query/header
            model_override: Model override from query/header
            db_session: Database session for loading preferences

        Returns:
            Tuple of (context, token) for cleanup

        """
        # Determine override
        override = None
        if provider_override and model_override:
            override = (provider_override, model_override)

        # Resolve selection
        context = await self.state.resolve_request_selection(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            request_override=override,
            db_session=db_session,
        )

        # Set in contextvars
        token = self.state.set_request_context(context)

        return context, token

    def cleanup_request_context(self, token: Token) -> None:
        """Clean up the selection context after request completion.

        Args:
            token: Token from setup_request_context

        """
        self.state.clear_request_context(token)


# =============================================================================
# Conversion Utilities
# =============================================================================


def model_selection_to_domain_selection(model_selection: ModelSelection) -> Selection:
    """Convert ModelSelection to domain Selection model.

    Useful for integration with existing code that uses the Selection model
    from app.domain.models.

    Args:
        model_selection: ModelSelection instance

    Returns:
        Selection domain model instance

    """
    return Selection(
        provider_id=model_selection.provider,
        model_id=model_selection.model_id,
        scope=model_selection.scope,
        user_id=model_selection.user_id,
        session_id=model_selection.session_id,
        created_at=model_selection.timestamp,
        updated_at=model_selection.timestamp,
    )


def domain_selection_to_model_selection(
    selection: Selection,
    metadata: dict[str, Any] | None = None,
) -> ModelSelection:
    """Convert domain Selection to ModelSelection.

    Args:
        selection: Selection domain model instance
        metadata: Optional additional metadata

    Returns:
        ModelSelection instance

    """
    return ModelSelection(
        provider=selection.provider_id,
        model_id=selection.model_id,
        user_id=selection.user_id,
        session_id=selection.session_id or "unknown",
        timestamp=selection.updated_at,
        metadata=metadata or {},
        scope=selection.scope,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main service
    "GlobalModelSelectionState",
    # Models
    "ModelSelection",
    # Middleware support
    "ModelSelectionMiddlewareHelper",
    "Selection",  # Re-export domain Selection for WebSocket use
    "SelectionContextData",
    "domain_selection_to_model_selection",
    # Dependencies
    "get_global_model_selection_state",
    "get_model_selection_state",
    # Utilities
    "model_selection_to_domain_selection",
]
