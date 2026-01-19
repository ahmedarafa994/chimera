"""
Provider Resolution Service for Project Chimera.

This service provides centralized provider/model resolution using a
priority-based hierarchy. It ensures that all services use the correct
provider and model based on:

1. Explicit parameters (if provided in API call)
2. Request context (from GlobalModelSelectionState via contextvars)
3. Session selection (from database/cache)
4. User default (from user preferences)
5. Global default (from configuration)

This is a critical component for the Unified Provider/Model Selection System,
ensuring consistent model selection across all API endpoints and services.

References:
- Design doc: docs/UNIFIED_PROVIDER_SYSTEM_DESIGN.md
- GlobalModelSelectionState: app/services/global_model_selection_state.py
- Provider plugins: app/services/provider_plugins/__init__.py
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.global_model_selection_state import GlobalModelSelectionState
from app.services.session_service import SessionService, get_session_service

logger = logging.getLogger(__name__)


# =============================================================================
# Resolution Result Models
# =============================================================================


@dataclass
class ResolutionResult:
    """
    Result of provider/model resolution.

    Contains the resolved provider and model along with metadata about
    how the resolution was performed.
    """

    provider: str
    model_id: str
    resolution_source: str = "default"
    resolution_priority: int = 5
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved_at: datetime = field(default_factory=datetime.utcnow)

    # Resolution source priority levels:
    # 1 = explicit parameters (highest)
    # 2 = request context
    # 3 = session selection
    # 4 = user default
    # 5 = global default (lowest)

    def to_tuple(self) -> tuple[str, str]:
        """Return provider and model_id as a tuple."""
        return (self.provider, self.model_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "resolution_source": self.resolution_source,
            "resolution_priority": self.resolution_priority,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "resolved_at": self.resolved_at.isoformat(),
        }


# =============================================================================
# Resolution Strategies
# =============================================================================


class ResolutionStrategy:
    """Base class for resolution strategies."""

    priority: int = 5
    name: str = "base"

    async def can_resolve(self, context: dict[str, Any]) -> bool:
        """Check if this strategy can provide a resolution."""
        raise NotImplementedError

    async def resolve(self, context: dict[str, Any]) -> tuple[str, str] | None:
        """
        Attempt to resolve provider/model.

        Returns (provider, model_id) or None.
        """
        raise NotImplementedError


class ExplicitParameterStrategy(ResolutionStrategy):
    """Resolution from explicit parameters (highest priority)."""

    priority = 1
    name = "explicit"

    async def can_resolve(self, context: dict[str, Any]) -> bool:
        return bool(context.get("explicit_provider") and context.get("explicit_model"))

    async def resolve(self, context: dict[str, Any]) -> tuple[str, str] | None:
        provider = context.get("explicit_provider")
        model = context.get("explicit_model")
        if provider and model:
            return (provider, model)
        return None


class RequestContextStrategy(ResolutionStrategy):
    """Resolution from request context (via contextvars)."""

    priority = 2
    name = "request_context"

    def __init__(self, state: GlobalModelSelectionState):
        self.state = state

    async def can_resolve(self, context: dict[str, Any]) -> bool:
        # First check SelectionContext (set by SelectionMiddleware)
        try:
            from app.core.selection_context import SelectionContext

            selection = SelectionContext.get_selection()
            if selection:
                return True
        except Exception:
            pass

        # Fall back to GlobalModelSelectionState
        request_context = self.state.get_request_context()
        return bool(request_context and request_context.selection)

    async def resolve(self, context: dict[str, Any]) -> tuple[str, str] | None:
        # First check SelectionContext (set by SelectionMiddleware)
        try:
            from app.core.selection_context import SelectionContext

            selection = SelectionContext.get_selection()
            if selection:
                logger.debug(
                    f"Using SelectionContext: {selection.provider_id}/{selection.model_id}"
                )
                return (selection.provider_id, selection.model_id)
        except Exception:
            pass

        # Fall back to GlobalModelSelectionState
        request_context = self.state.get_request_context()
        if request_context and request_context.selection:
            return (request_context.selection.provider, request_context.selection.model_id)
        return None


class SessionSelectionStrategy(ResolutionStrategy):
    """Resolution from session selection (database/cache)."""

    priority = 3
    name = "session"

    def __init__(
        self,
        state: GlobalModelSelectionState,
        session_service: SessionService | None = None,
    ):
        self.state = state
        self.session_service = session_service

    async def can_resolve(self, context: dict[str, Any]) -> bool:
        return bool(context.get("session_id"))

    async def resolve(self, context: dict[str, Any]) -> tuple[str, str] | None:
        session_id = context.get("session_id")
        user_id = context.get("user_id")
        db_session = context.get("db_session")

        if not session_id:
            return None

        # Try GlobalModelSelectionState first (faster, cached)
        try:
            selection = await self.state.get_current_selection(
                session_id=session_id,
                user_id=user_id,
                db_session=db_session,
            )
            # Only use if it's a session-level selection, not a default
            if selection.scope.value != "global":
                return (selection.provider, selection.model_id)
        except Exception as e:
            logger.warning(f"GlobalModelSelectionState lookup failed: {e}")

        # Fallback to SessionService
        if self.session_service:
            try:
                provider, model = self.session_service.get_session_model(session_id)
                if provider and model:
                    return (provider, model)
            except Exception as e:
                logger.warning(f"SessionService lookup failed: {e}")

        return None


class GlobalModelSelectionStrategy(ResolutionStrategy):
    """
    Resolution from model_selection_service (user's UI selection).

    This reads from the model_selection_service which persists the user's
    selected provider/model from the UI to a JSON file. This is tier 3.5
    in the hierarchy - after session but before user default.
    """

    priority = 35  # Between session (3) and user_default (4) - use 35 for 3.5
    name = "global_model_selection"

    async def can_resolve(self, context: dict[str, Any]) -> bool:
        try:
            from app.services.model_selection_service import model_selection_service

            selection = model_selection_service.get_selection()
            can_resolve = selection is not None
            logger.info(
                f"GlobalModelSelectionStrategy.can_resolve: {can_resolve}, "
                f"selection={selection}"
            )
            return can_resolve
        except Exception as e:
            logger.warning(f"GlobalModelSelectionStrategy.can_resolve failed: {e}")
            return False

    async def resolve(self, context: dict[str, Any]) -> tuple[str, str] | None:
        try:
            from app.services.model_selection_service import model_selection_service

            selection = model_selection_service.get_selection()
            if selection:
                logger.info(
                    f"GlobalModelSelectionStrategy.resolve: "
                    f"returning {selection.provider}/{selection.model}"
                )
                return (selection.provider, selection.model)
            logger.info("GlobalModelSelectionStrategy.resolve: no selection found")
            return None
        except Exception as e:
            logger.warning(f"GlobalModelSelectionStrategy.resolve failed: {e}")
            return None


class UserDefaultStrategy(ResolutionStrategy):
    """Resolution from user default preferences."""

    priority = 40  # After global_model_selection (35)
    name = "user_default"

    async def can_resolve(self, context: dict[str, Any]) -> bool:
        # TODO: Implement user preferences lookup
        return False

    async def resolve(self, context: dict[str, Any]) -> tuple[str, str] | None:
        user_id = context.get("user_id")
        if not user_id:
            return None

        # TODO: Implement user preferences lookup from database
        # This would query a user_preferences table for default provider/model
        return None


class GlobalDefaultStrategy(ResolutionStrategy):
    """Resolution from global configuration (fallback)."""

    priority = 50  # Lowest priority - static default
    name = "global_default"

    async def can_resolve(self, context: dict[str, Any]) -> bool:
        return True  # Always available

    async def resolve(self, context: dict[str, Any]) -> tuple[str, str] | None:
        provider = getattr(settings, "DEFAULT_PROVIDER", None)
        model = getattr(settings, "DEFAULT_MODEL", None)

        if not provider:
            provider = getattr(settings, "AI_PROVIDER", "openai")
        if not model:
            model = getattr(settings, "DEFAULT_MODEL_ID", "gpt-4")

        return (provider, model)


# =============================================================================
# Provider Resolution Service
# =============================================================================


class ProviderResolutionService:
    """
    Resolves the active provider/model for a given context.

    This service implements the priority-based resolution hierarchy:
    1. Explicit parameters (if provided)
    2. Request context (from GlobalModelSelectionState)
    3. Session selection (from database)
    4. User default (from user preferences)
    5. Global default (from config)

    Usage:
        >>> service = ProviderResolutionService()
        >>> provider, model = await service.resolve(
        ...     session_id="sess_123",
        ...     explicit_provider="openai",  # Optional override
        ... )
        >>> # Use provider and model for API call

    Thread Safety:
        This service is thread-safe and can be used across concurrent requests.
        All mutable state is contained in the underlying services.
    """

    _instance: Optional["ProviderResolutionService"] = None

    def __new__(cls) -> "ProviderResolutionService":
        """Implement singleton pattern for global access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the resolution service."""
        if self._initialized:
            return

        self._state = GlobalModelSelectionState()
        self._session_service: SessionService | None = None
        self._strategies: list[ResolutionStrategy] = []
        self._resolution_cache: dict[str, ResolutionResult] = {}
        self._cache_ttl_seconds = 60  # Cache resolutions for 60 seconds

        self._setup_strategies()
        self._initialized = True

        logger.info("ProviderResolutionService initialized")

    def _setup_strategies(self) -> None:
        """Set up resolution strategies in priority order."""
        try:
            self._session_service = get_session_service()
        except Exception as e:
            logger.warning(f"Could not get SessionService: {e}")

        self._strategies = [
            ExplicitParameterStrategy(),
            RequestContextStrategy(self._state),
            SessionSelectionStrategy(self._state, self._session_service),
            GlobalModelSelectionStrategy(),  # NEW: reads from model_selection_service
            UserDefaultStrategy(),
            GlobalDefaultStrategy(),
        ]

        # Sort by priority (lowest number = highest priority)
        self._strategies.sort(key=lambda s: s.priority)

    async def resolve(
        self,
        explicit_provider: str | None = None,
        explicit_model: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        db_session: AsyncSession | None = None,
        use_cache: bool = True,
    ) -> tuple[str, str]:
        """
        Resolve provider/model using priority hierarchy.

        Args:
            explicit_provider: Explicit provider override (highest priority)
            explicit_model: Explicit model override (highest priority)
            session_id: Session identifier for session-level resolution
            user_id: User identifier for user-level resolution
            db_session: Database session for persistence operations
            use_cache: Whether to use cached resolutions

        Returns:
            Tuple of (provider, model_id)

        Example:
            >>> provider, model = await service.resolve(
            ...     session_id="sess_123",
            ...     explicit_provider="anthropic",
            ...     explicit_model="claude-3-opus",
            ... )
        """
        result = await self.resolve_with_metadata(
            explicit_provider=explicit_provider,
            explicit_model=explicit_model,
            session_id=session_id,
            user_id=user_id,
            db_session=db_session,
            use_cache=use_cache,
        )

        return result.to_tuple()

    async def resolve_with_metadata(
        self,
        explicit_provider: str | None = None,
        explicit_model: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        db_session: AsyncSession | None = None,
        use_cache: bool = True,
    ) -> ResolutionResult:
        """
        Resolve provider/model with full resolution metadata.

        Returns a ResolutionResult containing the resolved provider/model
        along with information about how the resolution was performed.

        Args:
            explicit_provider: Explicit provider override
            explicit_model: Explicit model override
            session_id: Session identifier
            user_id: User identifier
            db_session: Database session
            use_cache: Whether to use cached resolutions

        Returns:
            ResolutionResult with full resolution details
        """
        # Build context for strategies
        context: dict[str, Any] = {
            "explicit_provider": explicit_provider,
            "explicit_model": explicit_model,
            "session_id": session_id,
            "user_id": user_id,
            "db_session": db_session,
        }

        # Check cache
        cache_key = self._get_cache_key(context)
        logger.info(f"Resolution cache_key: {cache_key}, use_cache={use_cache}")
        if use_cache and cache_key in self._resolution_cache:
            cached = self._resolution_cache[cache_key]
            age = (datetime.utcnow() - cached.resolved_at).total_seconds()
            if age < self._cache_ttl_seconds:
                logger.info(
                    f"Using cached resolution: {cached.to_tuple()}, "
                    f"age={age:.1f}s, source={cached.resolution_source}"
                )
                return cached
            else:
                logger.info(f"Cache expired (age={age:.1f}s > {self._cache_ttl_seconds}s)")
        else:
            if use_cache:
                logger.info("Cache miss - no cached resolution found")

        # Try each strategy in priority order
        for strategy in self._strategies:
            try:
                if await strategy.can_resolve(context):
                    result = await strategy.resolve(context)
                    if result:
                        provider, model_id = result

                        resolution = ResolutionResult(
                            provider=provider,
                            model_id=model_id,
                            resolution_source=strategy.name,
                            resolution_priority=strategy.priority,
                            session_id=session_id,
                            user_id=user_id,
                            metadata={
                                "strategy": strategy.name,
                                "had_explicit": bool(explicit_provider),
                            },
                        )

                        # Cache the result
                        if use_cache:
                            self._resolution_cache[cache_key] = resolution

                        logger.info(
                            f"Resolved provider/model: {provider}/{model_id} "
                            f"(source={strategy.name}, "
                            f"priority={strategy.priority})"
                        )

                        return resolution
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                continue

        # Should never reach here due to GlobalDefaultStrategy
        logger.error("All resolution strategies failed, using emergency fallback")
        return ResolutionResult(
            provider="openai",
            model_id="gpt-4",
            resolution_source="emergency_fallback",
            resolution_priority=99,
            session_id=session_id,
            user_id=user_id,
        )

    def _get_cache_key(self, context: dict[str, Any]) -> str:
        """Generate a cache key from context."""
        parts = [
            f"p:{context.get('explicit_provider', '')}",
            f"m:{context.get('explicit_model', '')}",
            f"s:{context.get('session_id', '')}",
            f"u:{context.get('user_id', '')}",
        ]
        return ":".join(parts)

    def invalidate_cache(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """
        Invalidate cached resolutions.

        Args:
            session_id: If provided, only invalidate for this session
            user_id: If provided, only invalidate for this user

        Returns:
            Number of cache entries invalidated
        """
        if not session_id and not user_id:
            count = len(self._resolution_cache)
            self._resolution_cache.clear()
            logger.info(f"Invalidated all {count} cached resolutions")
            return count

        # Selective invalidation
        keys_to_remove = []
        for key, result in self._resolution_cache.items():
            if (session_id and result.session_id == session_id) or (
                user_id and result.user_id == user_id
            ):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._resolution_cache[key]

        logger.info(f"Invalidated {len(keys_to_remove)} cached resolutions")
        return len(keys_to_remove)

    def get_resolution_from_context(self) -> tuple[str, str] | None:
        """
        Get provider/model from current request context (sync method).

        This is a convenience method for getting the selection from
        the request context without performing full resolution.
        Useful when you know the context has been set up by middleware.

        Returns:
            Tuple of (provider, model_id) if context exists, None otherwise
        """
        context = self._state.get_request_context()
        if context and context.selection:
            return (context.selection.provider, context.selection.model_id)
        return None

    async def get_current_selection_from_context(
        self,
        fallback_session_id: str | None = None,
        fallback_user_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Get the current selection from request context or resolve.

        This method first checks the request context (set by middleware),
        and falls back to full resolution if no context exists.

        Args:
            fallback_session_id: Session ID to use if context missing
            fallback_user_id: User ID to use if context missing

        Returns:
            Tuple of (provider, model_id)
        """
        # Try request context first (fastest)
        context_result = self.get_resolution_from_context()
        if context_result:
            provider, model_id = context_result
            logger.debug(f"Using context selection: {provider}/{model_id}")
            return context_result

        # Fall back to full resolution
        logger.debug("No request context, performing full resolution")
        return await self.resolve(
            session_id=fallback_session_id,
            user_id=fallback_user_id,
        )


# =============================================================================
# Singleton Instance and Factory
# =============================================================================


_provider_resolution_service: ProviderResolutionService | None = None


def get_provider_resolution_service() -> ProviderResolutionService:
    """
    Get the singleton ProviderResolutionService instance.

    Returns:
        The global ProviderResolutionService instance
    """
    global _provider_resolution_service
    if _provider_resolution_service is None:
        _provider_resolution_service = ProviderResolutionService()
    return _provider_resolution_service


async def get_provider_resolution_service_async() -> ProviderResolutionService:
    """
    Async factory for ProviderResolutionService.

    Use this as a FastAPI dependency:
        @router.get("/endpoint")
        async def endpoint(
            resolution: ProviderResolutionService = Depends(
                get_provider_resolution_service_async
            )
        ):
            provider, model = await resolution.resolve(session_id="...")

    Returns:
        The global ProviderResolutionService instance
    """
    return get_provider_resolution_service()


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "ExplicitParameterStrategy",
    "GlobalDefaultStrategy",
    "GlobalModelSelectionStrategy",
    # Main service
    "ProviderResolutionService",
    "RequestContextStrategy",
    # Result model
    "ResolutionResult",
    # Strategy classes (for extension)
    "ResolutionStrategy",
    "SessionSelectionStrategy",
    "UserDefaultStrategy",
    # Factory functions
    "get_provider_resolution_service",
    "get_provider_resolution_service_async",
]
