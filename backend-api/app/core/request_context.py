"""Request Context Propagation.

Provides request-scoped context that flows through all application layers,
carrying provider/model selection and request metadata.

This module implements:
- RequestContext: Immutable data structure for request metadata
- ContextManager: Async context manager using AsyncLocalStorage pattern
- Context utilities for validation and error handling

Usage:
    # In middleware
    async with ContextManager.run_with_context(context):
        await handle_request()

    # In services
    context = ContextManager.get_context()
    provider = context.provider
    model = context.model
"""

import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, Field

# Type variable for generic async functions
T = TypeVar("T")

# Context variable for storing request context
_request_context: ContextVar[Optional["RequestContext"]] = ContextVar(
    "request_context",
    default=None,
)


class RequestContext(BaseModel):
    """Immutable request-scoped context carrying provider/model selection.

    This context is created at the API boundary and flows through all layers
    of the application, ensuring consistent provider/model usage for the entire
    request lifecycle.

    Attributes:
        request_id: Unique identifier for this request
        user_id: User making the request
        session_id: Session identifier
        provider: Selected AI provider (e.g., "openai", "anthropic")
        model: Selected model (e.g., "gpt-4o", "claude-3-5-sonnet")
        timestamp: When the context was created
        metadata: Additional request metadata (IP, user agent, etc.)

    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    provider: str
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True  # Make context immutable


class MissingContextError(Exception):
    """Raised when attempting to get context but none exists."""


class ContextManager:
    """Manages request-scoped context using Python's contextvars.

    Provides async context manager for running code with a specific context
    and utilities for getting/checking current context.
    """

    @staticmethod
    async def run_with_context(context: RequestContext, fn: Callable[[], Awaitable[T]]) -> T:
        """Run an async function with the given context.

        Args:
            context: RequestContext to use for this execution
            fn: Async function to run (no parameters)

        Returns:
            Result of the function

        Example:
            context = RequestContext(user_id="user123", ...)
            result = await ContextManager.run_with_context(
                context,
                lambda: my_service.do_work()
            )

        """
        token = _request_context.set(context)
        try:
            return await fn()
        finally:
            _request_context.reset(token)

    @staticmethod
    def get_context() -> RequestContext:
        """Get the current request context.

        Returns:
            Current RequestContext

        Raises:
            MissingContextError: If no context is set

        Example:
            context = ContextManager.get_context()
            print(f"Using provider: {context.provider}")

        """
        context = _request_context.get()
        if context is None:
            msg = (
                "No request context found. Ensure the request is running "
                "within a context created by ContextManager.run_with_context() "
                "or context propagation middleware."
            )
            raise MissingContextError(
                msg,
            )
        return context

    @staticmethod
    def get_context_or_none() -> RequestContext | None:
        """Get the current request context, or None if not set.

        Returns:
            Current RequestContext or None

        Example:
            context = ContextManager.get_context_or_none()
            if context:
                print(f"Provider: {context.provider}")

        """
        return _request_context.get()

    @staticmethod
    def has_context() -> bool:
        """Check if a request context is currently set.

        Returns:
            True if context exists, False otherwise

        Example:
            if ContextManager.has_context():
                context = ContextManager.get_context()

        """
        return _request_context.get() is not None

    @staticmethod
    def set_context(context: RequestContext) -> None:
        """Set the request context directly (use with caution).

        Warning: This should rarely be used directly. Prefer run_with_context()
        which properly handles cleanup.

        Args:
            context: RequestContext to set

        """
        _request_context.set(context)

    @staticmethod
    def clear_context() -> None:
        """Clear the current request context.

        Warning: This should rarely be used directly.
        """
        _request_context.set(None)


# Utility functions for common context operations


def get_current_provider() -> str:
    """Get the provider ID from current context."""
    return ContextManager.get_context().provider


def get_current_model() -> str:
    """Get the model ID from current context."""
    return ContextManager.get_context().model


def get_current_user_id() -> str:
    """Get the user ID from current context."""
    return ContextManager.get_context().user_id


def get_request_id() -> str:
    """Get the request ID from current context."""
    return ContextManager.get_context().request_id


def get_context_metadata(key: str, default: Any = None) -> Any:
    """Get a metadata value from current context."""
    context = ContextManager.get_context()
    return context.metadata.get(key, default)


def create_child_context(**updates) -> RequestContext:
    """Create a new context based on current context with updates.

    Useful for creating derived contexts (e.g., for sub-requests).

    Args:
        **updates: Fields to update in the new context

    Returns:
        New RequestContext with updates applied

    Example:
        # Create a child context with new request ID
        child_ctx = create_child_context(
            request_id=str(uuid.uuid4()),
            metadata={"parent_request_id": get_request_id()}
        )

    """
    current = ContextManager.get_context()
    current_dict = current.dict()
    current_dict.update(updates)
    return RequestContext(**current_dict)
