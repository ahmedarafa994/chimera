"""
Request Context Propagation - Immutable context for provider selection.

This module implements the immutable request context pattern using Python's contextvars.
The selection is resolved once at request entry and propagated through the entire call stack,
ensuring consistency and thread-safety for concurrent requests.

Key Features:
- Thread-safe context isolation using contextvars
- Immutable selection state (set once per request)
- Automatic propagation through async call chains
- Support for FastAPI dependency injection
- WebSocket context handling
- Context debugging and introspection
"""

import logging
from contextvars import ContextVar, Token
from typing import Any

from app.domain.models import Selection, SelectionScope

logger = logging.getLogger(__name__)

# Context variables for request-scoped selection state
_selection_context: ContextVar[Selection | None] = ContextVar("provider_selection", default=None)

_session_id_context: ContextVar[str | None] = ContextVar("session_id", default=None)

_user_id_context: ContextVar[str | None] = ContextVar("user_id", default=None)

_request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class SelectionContext:
    """
    Immutable request context for provider/model selection.

    This class manages the request-scoped selection state using Python's contextvars.
    Once set, the selection cannot be changed within the same request context, ensuring
    consistency across all downstream operations.

    Usage:
        >>> # At request entry (middleware or dependency)
        >>> SelectionContext.set_selection(selection, session_id="sess_123")
        >>>
        >>> # Anywhere in the call stack
        >>> current = SelectionContext.get_selection()
        >>> provider_id = SelectionContext.get_provider_id()
        >>> model_id = SelectionContext.get_model_id()
    """

    @staticmethod
    def set_selection(
        selection: Selection,
        session_id: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> tuple[Token, Token | None, Token | None, Token | None]:
        """
        Set the selection context for the current request.

        This should be called once at request entry (typically in middleware or
        dependency injection). Subsequent calls within the same context will log
        a warning but will not override the existing selection.

        Args:
            selection: The resolved Selection object
            session_id: Optional session identifier
            user_id: Optional user identifier
            request_id: Optional request identifier for tracing

        Returns:
            Tuple of tokens (selection_token, session_token, user_token, request_token)
            for potential context reset

        Raises:
            ValueError: If selection is None or invalid
        """
        if selection is None:
            raise ValueError("Selection cannot be None")

        # Check if selection is already set
        existing = _selection_context.get()
        if existing is not None:
            logger.warning(
                f"Selection context already set for request {request_id}. "
                f"Existing: {existing.provider_id}/{existing.model_id}, "
                f"New: {selection.provider_id}/{selection.model_id}. "
                "Using existing selection to maintain immutability."
            )
            # Return current tokens (no-op)
            return (
                _selection_context.set(existing),
                None,
                None,
                None,
            )

        # Set new selection context
        selection_token = _selection_context.set(selection)

        session_token = None
        if session_id is not None:
            session_token = _session_id_context.set(session_id)

        user_token = None
        if user_id is not None:
            user_token = _user_id_context.set(user_id)

        request_token = None
        if request_id is not None:
            request_token = _request_id_context.set(request_id)

        logger.debug(
            f"Set selection context: {selection.provider_id}/{selection.model_id} "
            f"(scope={selection.scope.value}, request={request_id})"
        )

        return selection_token, session_token, user_token, request_token

    @staticmethod
    def get_selection() -> Selection | None:
        """
        Get the current selection from context.

        Returns:
            The current Selection object, or None if not set
        """
        return _selection_context.get()

    @staticmethod
    def get_provider_id() -> str | None:
        """
        Get the current provider ID from context.

        Returns:
            The provider ID, or None if selection not set
        """
        selection = _selection_context.get()
        return selection.provider_id if selection else None

    @staticmethod
    def get_model_id() -> str | None:
        """
        Get the current model ID from context.

        Returns:
            The model ID, or None if selection not set
        """
        selection = _selection_context.get()
        return selection.model_id if selection else None

    @staticmethod
    def get_session_id() -> str | None:
        """
        Get the current session ID from context.

        Returns:
            The session ID, or None if not set
        """
        return _session_id_context.get()

    @staticmethod
    def get_user_id() -> str | None:
        """
        Get the current user ID from context.

        Returns:
            The user ID, or None if not set
        """
        return _user_id_context.get()

    @staticmethod
    def get_request_id() -> str | None:
        """
        Get the current request ID from context.

        Returns:
            The request ID, or None if not set
        """
        return _request_id_context.get()

    @staticmethod
    def is_set() -> bool:
        """
        Check if selection context is currently set.

        Returns:
            True if selection is set, False otherwise
        """
        return _selection_context.get() is not None

    @staticmethod
    def get_scope() -> SelectionScope | None:
        """
        Get the scope of the current selection.

        Returns:
            The SelectionScope, or None if selection not set
        """
        selection = _selection_context.get()
        return selection.scope if selection else None

    @staticmethod
    def reset(
        selection_token: Token | None = None,
        session_token: Token | None = None,
        user_token: Token | None = None,
        request_token: Token | None = None,
    ) -> None:
        """
        Reset context variables to their previous values.

        This is primarily for testing or explicit context cleanup. Normal
        request processing should rely on automatic context cleanup.

        Args:
            selection_token: Token from set_selection() call
            session_token: Session ID token
            user_token: User ID token
            request_token: Request ID token
        """
        if selection_token is not None:
            _selection_context.reset(selection_token)

        if session_token is not None:
            _session_id_context.reset(session_token)

        if user_token is not None:
            _user_id_context.reset(user_token)

        if request_token is not None:
            _request_id_context.reset(request_token)

        logger.debug("Reset selection context")

    @staticmethod
    def clear_all() -> None:
        """
        Clear all context variables (primarily for testing).

        WARNING: This clears context for the current execution context.
        Use with caution in production code.
        """
        # Set to None to clear
        _selection_context.set(None)
        _session_id_context.set(None)
        _user_id_context.set(None)
        _request_id_context.set(None)

        logger.debug("Cleared all selection context variables")

    @staticmethod
    def get_debug_info() -> dict[str, Any]:
        """
        Get debug information about the current context.

        Returns:
            Dictionary with context state information
        """
        selection = _selection_context.get()

        info = {
            "is_set": selection is not None,
            "provider_id": selection.provider_id if selection else None,
            "model_id": selection.model_id if selection else None,
            "scope": selection.scope.value if selection else None,
            "session_id": _session_id_context.get(),
            "user_id": _user_id_context.get(),
            "request_id": _request_id_context.get(),
        }

        if selection:
            info.update(
                {
                    "selection_user_id": selection.user_id,
                    "selection_session_id": selection.session_id,
                    "created_at": selection.created_at.isoformat(),
                    "updated_at": selection.updated_at.isoformat(),
                }
            )

        return info


class SelectionContextManager:
    """
    Context manager for selection context (useful for testing and explicit scoping).

    Usage:
        >>> with SelectionContextManager(selection, session_id="sess_123"):
        ...     # Selection is active within this block
        ...     provider = SelectionContext.get_provider_id()
        >>> # Selection is automatically cleaned up
    """

    def __init__(
        self,
        selection: Selection,
        session_id: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
    ):
        """
        Initialize context manager.

        Args:
            selection: The Selection object to set
            session_id: Optional session identifier
            user_id: Optional user identifier
            request_id: Optional request identifier
        """
        self.selection = selection
        self.session_id = session_id
        self.user_id = user_id
        self.request_id = request_id
        self.tokens: tuple[Token, ...] = ()

    def __enter__(self):
        """Enter context and set selection."""
        self.tokens = SelectionContext.set_selection(
            self.selection, self.session_id, self.user_id, self.request_id
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and reset selection."""
        if self.tokens:
            SelectionContext.reset(*self.tokens)
        return False  # Don't suppress exceptions


# Convenience function for FastAPI dependency injection
def get_current_selection() -> Selection | None:
    """
    FastAPI dependency to get the current selection from context.

    Usage in route:
        @app.get("/api/v1/something")
        async def endpoint(selection: Selection | None = Depends(get_current_selection)):
            if selection:
                provider_id = selection.provider_id
                ...
    """
    return SelectionContext.get_selection()


# Convenience function for getting required selection
def require_selection() -> Selection:
    """
    FastAPI dependency that requires a selection to be set.

    Raises HTTPException if no selection is in context.

    Usage in route:
        @app.get("/api/v1/something")
        async def endpoint(selection: Selection = Depends(require_selection)):
            # selection is guaranteed to be non-None
            provider_id = selection.provider_id
            ...
    """
    from fastapi import HTTPException

    selection = SelectionContext.get_selection()
    if selection is None:
        raise HTTPException(
            status_code=500,
            detail="Provider selection not found in request context. "
            "This is likely a configuration error.",
        )
    return selection
