"""
Provider Boundary Guard for Service Boundaries.

This module provides a guard mechanism that enforces provider/model
consistency at service boundaries. It can be used as:
- A decorator for service methods
- A context manager for code blocks
- A validation utility

The guard ensures that the correct provider/model is being used within
a service call, preventing accidental mismatches during complex
request processing.

Usage:
    # As a decorator
    @ProviderBoundaryGuard.enforce("openai", "gpt-4")
    async def my_service_method(self):
        ...

    # As a context manager
    async with boundary_guard.validate_boundary("openai", "gpt-4"):
        await do_work()

    # Direct validation
    result = boundary_guard.check_boundary("openai", "gpt-4")
"""

import asyncio
import functools
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Optional, TypeVar, Union

from app.core.request_context import ContextManager
from app.core.selection_context import SelectionContext
from app.schemas.validation_errors import (
    BoundaryValidationResult,
    SelectionValidationError,
    ValidationErrorType,
)

logger = logging.getLogger(__name__)

# Type variable for generic function signatures
F = TypeVar('F', bound=Callable[..., Any])


class BoundaryValidationError(Exception):
    """
    Exception raised when provider/model boundary validation fails.

    Attributes:
        result: BoundaryValidationResult with failure details
        message: Human-readable error message
    """

    def __init__(
        self,
        result: BoundaryValidationResult,
        message: Optional[str] = None,
    ):
        self.result = result
        self.message = message or result.error_message or "Boundary validation failed"
        super().__init__(self.message)


class ProviderBoundaryGuard:
    """
    Guard that enforces provider/model consistency at service boundaries.

    This class provides multiple ways to validate that the current context
    matches expected provider/model values:

    1. Decorator: Validate at method entry
    2. Context manager: Validate for a code block
    3. Direct validation: Check without raising

    Thread Safety:
        This class is thread-safe as it only reads from contextvars.
    """

    def __init__(
        self,
        strict_mode: bool = False,
        log_mismatches: bool = True,
        allow_upgrades: bool = False,
    ):
        """
        Initialize the boundary guard.

        Args:
            strict_mode: Raise exception on mismatch (vs just log warning)
            log_mismatches: Whether to log boundary mismatches
            allow_upgrades: Allow "upgrade" mismatches (e.g., gpt-3.5 -> gpt-4)
        """
        self._strict_mode = strict_mode
        self._log_mismatches = log_mismatches
        self._allow_upgrades = allow_upgrades
        self._mismatch_count = 0
        self._validation_count = 0

    @staticmethod
    def enforce(
        expected_provider: str,
        expected_model: str,
        strict: bool = True,
        boundary_name: Optional[str] = None,
    ) -> Callable[[F], F]:
        """
        Decorator that validates provider/model matches expectation.

        Args:
            expected_provider: Expected provider ID
            expected_model: Expected model ID
            strict: Whether to raise on mismatch
            boundary_name: Name for this boundary (for logging)

        Returns:
            Decorated function

        Example:
            @ProviderBoundaryGuard.enforce("openai", "gpt-4")
            async def generate_completion(self, prompt: str):
                ...
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _boundary = boundary_name or func.__name__
                result = _check_boundary(
                    expected_provider, expected_model, _boundary
                )

                if not result.is_consistent:
                    if strict:
                        raise BoundaryValidationError(result)
                    else:
                        logger.warning(
                            f"Boundary mismatch in {_boundary}: "
                            f"expected={expected_provider}/{expected_model}, "
                            f"actual={result.actual_provider}/"
                            f"{result.actual_model}"
                        )

                return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                _boundary = boundary_name or func.__name__
                result = _check_boundary(
                    expected_provider, expected_model, _boundary
                )

                if not result.is_consistent:
                    if strict:
                        raise BoundaryValidationError(result)
                    else:
                        logger.warning(
                            f"Boundary mismatch in {_boundary}: "
                            f"expected={expected_provider}/{expected_model}, "
                            f"actual={result.actual_provider}/"
                            f"{result.actual_model}"
                        )

                return func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            else:
                return sync_wrapper  # type: ignore

        return decorator

    @staticmethod
    def require_context(
        strict: bool = True,
        boundary_name: Optional[str] = None,
    ) -> Callable[[F], F]:
        """
        Decorator that ensures context exists (any provider/model).

        Args:
            strict: Whether to raise on missing context
            boundary_name: Name for this boundary (for logging)

        Returns:
            Decorated function

        Example:
            @ProviderBoundaryGuard.require_context()
            async def process_request(self):
                ...
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _boundary = boundary_name or func.__name__

                if not _has_context():
                    if strict:
                        raise BoundaryValidationError(
                            BoundaryValidationResult(
                                is_consistent=False,
                                expected_provider="any",
                                expected_model="any",
                                boundary_name=_boundary,
                                error_message="No selection context available",
                            )
                        )
                    else:
                        logger.warning(
                            f"No context at boundary {_boundary}"
                        )

                return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                _boundary = boundary_name or func.__name__

                if not _has_context():
                    if strict:
                        raise BoundaryValidationError(
                            BoundaryValidationResult(
                                is_consistent=False,
                                expected_provider="any",
                                expected_model="any",
                                boundary_name=_boundary,
                                error_message="No selection context available",
                            )
                        )
                    else:
                        logger.warning(
                            f"No context at boundary {_boundary}"
                        )

                return func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            else:
                return sync_wrapper  # type: ignore

        return decorator

    @asynccontextmanager
    async def validate_boundary_async(
        self,
        provider: str,
        model: str,
        boundary_name: str = "async_block",
    ):
        """
        Async context manager for boundary validation.

        Args:
            provider: Expected provider ID
            model: Expected model ID
            boundary_name: Name for this boundary

        Yields:
            BoundaryValidationResult

        Raises:
            BoundaryValidationError: If strict mode and mismatch

        Example:
            async with guard.validate_boundary_async("openai", "gpt-4"):
                await do_work()
        """
        self._validation_count += 1
        result = _check_boundary(provider, model, boundary_name)

        if not result.is_consistent:
            self._mismatch_count += 1

            if self._log_mismatches:
                logger.warning(
                    f"Async boundary mismatch in {boundary_name}: "
                    f"expected={provider}/{model}, "
                    f"actual={result.actual_provider}/{result.actual_model}"
                )

            if self._strict_mode:
                raise BoundaryValidationError(result)

        try:
            yield result
        finally:
            pass  # Context cleanup if needed

    @contextmanager
    def validate_boundary(
        self,
        provider: str,
        model: str,
        boundary_name: str = "sync_block",
    ):
        """
        Sync context manager for boundary validation.

        Args:
            provider: Expected provider ID
            model: Expected model ID
            boundary_name: Name for this boundary

        Yields:
            BoundaryValidationResult

        Raises:
            BoundaryValidationError: If strict mode and mismatch

        Example:
            with guard.validate_boundary("openai", "gpt-4"):
                do_work()
        """
        self._validation_count += 1
        result = _check_boundary(provider, model, boundary_name)

        if not result.is_consistent:
            self._mismatch_count += 1

            if self._log_mismatches:
                logger.warning(
                    f"Sync boundary mismatch in {boundary_name}: "
                    f"expected={provider}/{model}, "
                    f"actual={result.actual_provider}/{result.actual_model}"
                )

            if self._strict_mode:
                raise BoundaryValidationError(result)

        try:
            yield result
        finally:
            pass

    def check_boundary(
        self,
        expected_provider: str,
        expected_model: str,
        boundary_name: str = "check",
    ) -> BoundaryValidationResult:
        """
        Check boundary without raising exception.

        Args:
            expected_provider: Expected provider ID
            expected_model: Expected model ID
            boundary_name: Name for this boundary

        Returns:
            BoundaryValidationResult with validation status
        """
        self._validation_count += 1
        result = _check_boundary(
            expected_provider, expected_model, boundary_name
        )

        if not result.is_consistent:
            self._mismatch_count += 1

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get guard statistics."""
        return {
            "validation_count": self._validation_count,
            "mismatch_count": self._mismatch_count,
            "mismatch_rate": (
                self._mismatch_count / self._validation_count
                if self._validation_count > 0 else 0
            ),
            "strict_mode": self._strict_mode,
        }


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _has_context() -> bool:
    """Check if any selection context is available."""
    # Try RequestContext first
    if ContextManager.has_context():
        return True

    # Try SelectionContext
    if SelectionContext.is_set():
        return True

    return False


def _get_current_selection() -> tuple[Optional[str], Optional[str]]:
    """
    Get current provider/model from context.

    Returns:
        Tuple of (provider, model) or (None, None)
    """
    # Try RequestContext first
    if ContextManager.has_context():
        ctx = ContextManager.get_context()
        return ctx.provider, ctx.model

    # Try SelectionContext
    if SelectionContext.is_set():
        selection = SelectionContext.get_selection()
        if selection:
            return selection.provider_id, selection.model_id

    return None, None


def _check_boundary(
    expected_provider: str,
    expected_model: str,
    boundary_name: str,
) -> BoundaryValidationResult:
    """
    Check if current context matches expected provider/model.

    Args:
        expected_provider: Expected provider ID
        expected_model: Expected model ID
        boundary_name: Name of the boundary being checked

    Returns:
        BoundaryValidationResult
    """
    actual_provider, actual_model = _get_current_selection()

    if actual_provider is None or actual_model is None:
        return BoundaryValidationResult(
            is_consistent=False,
            expected_provider=expected_provider,
            expected_model=expected_model,
            actual_provider=actual_provider,
            actual_model=actual_model,
            boundary_name=boundary_name,
            error_message="No selection context available",
        )

    # Normalize for comparison
    normalized_expected_provider = expected_provider.lower().strip()
    normalized_actual_provider = actual_provider.lower().strip()
    normalized_expected_model = expected_model.lower().strip()
    normalized_actual_model = actual_model.lower().strip()

    provider_matches = normalized_expected_provider == normalized_actual_provider
    model_matches = normalized_expected_model == normalized_actual_model

    is_consistent = provider_matches and model_matches

    error_message = None
    if not is_consistent:
        mismatches = []
        if not provider_matches:
            mismatches.append(
                f"provider: expected '{expected_provider}', "
                f"got '{actual_provider}'"
            )
        if not model_matches:
            mismatches.append(
                f"model: expected '{expected_model}', "
                f"got '{actual_model}'"
            )
        error_message = f"Boundary mismatch: {'; '.join(mismatches)}"

    return BoundaryValidationResult(
        is_consistent=is_consistent,
        expected_provider=expected_provider,
        expected_model=expected_model,
        actual_provider=actual_provider,
        actual_model=actual_model,
        boundary_name=boundary_name,
        error_message=error_message,
    )


# =============================================================================
# Module-Level Guard Instance
# =============================================================================


# Default guard instance for convenience
_default_guard = ProviderBoundaryGuard(
    strict_mode=False,
    log_mismatches=True,
)


def get_boundary_guard() -> ProviderBoundaryGuard:
    """Get the default boundary guard instance."""
    return _default_guard


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ProviderBoundaryGuard",
    "BoundaryValidationError",
    "get_boundary_guard",
]
