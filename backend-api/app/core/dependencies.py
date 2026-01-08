"""
Dependency Injection Module

Provides FastAPI dependencies for services with proper scoping:
- Application-scoped: Shared across all requests (singletons)
- Request-scoped: Created per request for isolation

SCALE-004: Refactored to support dependency injection for LLM service.
UNIFIED-PROVIDER: Added provider resolution dependencies for global
    provider/model selection propagation.
"""

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import Request

from meta_prompter.jailbreak_enhancer import JailbreakPromptEnhancer
from meta_prompter.prompt_enhancer import PromptEnhancer

if TYPE_CHECKING:
    from app.services.global_model_selection_state import (
        GlobalModelSelectionState,
    )
    from app.services.provider_resolution_service import (
        ProviderResolutionService,
    )

# =============================================================================
# Application-Scoped Dependencies (Singletons)
# =============================================================================


@lru_cache
def get_prompt_enhancer() -> PromptEnhancer:
    """
    Get or create a singleton instance of PromptEnhancer.
    Application-scoped - shared across all requests.
    """
    return PromptEnhancer()


@lru_cache
def get_jailbreak_enhancer() -> JailbreakPromptEnhancer:
    """
    Get or create a singleton instance of JailbreakPromptEnhancer.
    Application-scoped - shared across all requests.
    """
    return JailbreakPromptEnhancer()


# =============================================================================
# LLM Service Dependencies (SCALE-004)
# =============================================================================

# Global LLM service instance (application-scoped)
_llm_service_instance = None


def get_llm_service():
    """
    Get the application-scoped LLM service instance.

    This is the default dependency for most endpoints that need LLM access.
    The service maintains its own caching and circuit breaker state.

    Returns:
        LLMService: The shared LLM service instance
    """
    global _llm_service_instance

    if _llm_service_instance is None:
        from app.services.llm_service import LLMService

        _llm_service_instance = LLMService()

    return _llm_service_instance


def set_llm_service(service) -> None:
    """
    Set the global LLM service instance.

    Used during application startup to inject a configured service.

    Args:
        service: The LLMService instance to use
    """
    global _llm_service_instance
    _llm_service_instance = service


class RequestScopedLLMService:
    """
    Request-scoped wrapper for LLM service.

    SCALE-004: Provides request isolation while sharing the underlying
    provider connections. Each request gets its own:
    - Request ID tracking
    - Per-request metrics
    - Isolated error handling

    The underlying LLM providers and circuit breakers are still shared
    for efficiency.
    """

    def __init__(self, base_service, request_id: str | None = None):
        """
        Initialize request-scoped service.

        Args:
            base_service: The application-scoped LLMService
            request_id: Optional request ID for tracking
        """
        self._base_service = base_service
        self._request_id = request_id
        self._request_metrics = {
            "calls": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    @property
    def request_id(self) -> str | None:
        """Get the request ID."""
        return self._request_id

    @property
    def metrics(self) -> dict:
        """Get request-scoped metrics."""
        return self._request_metrics.copy()

    async def generate_text(self, request):
        """
        Generate text with request-scoped tracking.

        Args:
            request: PromptRequest object

        Returns:
            PromptResponse from the LLM
        """
        self._request_metrics["calls"] += 1

        try:
            response = await self._base_service.generate_text(request)
            return response
        except Exception:
            self._request_metrics["errors"] += 1
            raise

    async def generate(self, prompt: str, provider: str | None = None, **kwargs):
        """
        Generate text with simplified interface.

        Args:
            prompt: The text prompt
            provider: Optional provider name
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        self._request_metrics["calls"] += 1

        try:
            response = await self._base_service.generate(prompt, provider, **kwargs)
            return response
        except Exception:
            self._request_metrics["errors"] += 1
            raise

    def get_provider(self, name: str | None = None):
        """Get a provider by name."""
        return self._base_service.get_provider(name)

    async def list_providers(self):
        """List available providers."""
        return await self._base_service.list_providers()

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return self._base_service.get_available_providers()

    def get_provider_info(self, provider_name: str) -> dict:
        """Get information about a specific provider."""
        return self._base_service.get_provider_info(provider_name)

    @property
    def default_provider(self):
        """Get the default provider type."""
        return self._base_service.default_provider

    def get_performance_stats(self) -> dict:
        """Get performance statistics including request-scoped metrics."""
        base_stats = self._base_service.get_performance_stats()
        return {
            **base_stats,
            "request_metrics": self._request_metrics,
            "request_id": self._request_id,
        }


async def get_request_scoped_llm_service(
    request: Request,
) -> AsyncGenerator[RequestScopedLLMService, None]:
    """
    Get a request-scoped LLM service instance.

    SCALE-004: Creates a new RequestScopedLLMService for each request,
    providing isolation while sharing the underlying providers.

    Usage:
        @router.post("/generate")
        async def generate(
            llm: RequestScopedLLMService = Depends(get_request_scoped_llm_service)
        ):
            response = await llm.generate("Hello")
            return response

    Args:
        request: The FastAPI request object

    Yields:
        RequestScopedLLMService: A request-scoped LLM service wrapper
    """
    # Get request ID from headers or state
    request_id = request.headers.get("X-Request-ID") or getattr(request.state, "request_id", None)

    # Get the base service
    base_service = get_llm_service()

    # Create request-scoped wrapper
    scoped_service = RequestScopedLLMService(base_service, request_id)

    try:
        yield scoped_service
    finally:
        # Log request metrics if needed
        if scoped_service.metrics["calls"] > 0:
            from app.core.logging import logger

            logger.debug(f"Request {request_id} LLM metrics: {scoped_service.metrics}")


# =============================================================================
# Provider Registry Dependencies
# =============================================================================


def get_provider_registry():
    """
    Get the provider registry for managing LLM providers.

    Returns:
        The LLM service which acts as the provider registry
    """
    return get_llm_service()


# =============================================================================
# Session Service Dependencies
# =============================================================================

_session_service_instance = None


def get_session_service():
    """
    Get the session service instance.

    Returns:
        SessionService: The shared session service instance
    """
    global _session_service_instance

    if _session_service_instance is None:
        from app.services.session_service import SessionService

        _session_service_instance = SessionService()

    return _session_service_instance


def set_session_service(service) -> None:
    """
    Set the global session service instance.

    Args:
        service: The SessionService instance to use
    """
    global _session_service_instance
    _session_service_instance = service


# =============================================================================
# Transformation Service Dependencies
# =============================================================================

_transformation_service_instance = None


def get_transformation_service():
    """
    Get the transformation service instance.

    Returns:
        TransformationService: The shared transformation service instance
    """
    global _transformation_service_instance

    if _transformation_service_instance is None:
        from app.services.transformation_service import TransformationService

        _transformation_service_instance = TransformationService()

    return _transformation_service_instance


def set_transformation_service(service) -> None:
    """
    Set the global transformation service instance.

    Args:
        service: The TransformationService instance to use
    """
    global _transformation_service_instance
    _transformation_service_instance = service


# =============================================================================
# Intent-Aware Service Dependencies
# =============================================================================

_intent_aware_service_instance = None


def get_intent_aware_service():
    """
    Get the intent-aware jailbreak service instance.

    Returns:
        IntentAwareJailbreakService: The shared service instance
    """
    global _intent_aware_service_instance

    if _intent_aware_service_instance is None:
        from app.services.intent_aware_jailbreak_service import IntentAwareJailbreakService

        _intent_aware_service_instance = IntentAwareJailbreakService()

    return _intent_aware_service_instance


def set_intent_aware_service(service) -> None:
    """
    Set the global intent-aware service instance.

    Args:
        service: The IntentAwareJailbreakService instance to use
    """
    global _intent_aware_service_instance
    _intent_aware_service_instance = service


# =============================================================================
# Cleanup Functions
# =============================================================================


# Note: reset_all_services is defined at end of file with full cleanup


# =============================================================================
# Global Model Selection State Dependencies (Unified Provider System)
# =============================================================================

_global_model_selection_state_instance = None


def get_global_model_selection_state() -> "GlobalModelSelectionState":
    """
    Get the application-scoped GlobalModelSelectionState instance.

    This singleton manages the global provider/model selection state
    across all requests, including request-scoped context tracking.

    Returns:
        GlobalModelSelectionState: The shared selection state instance
    """
    global _global_model_selection_state_instance

    if _global_model_selection_state_instance is None:
        from app.services.global_model_selection_state import (
            GlobalModelSelectionState,
        )

        _global_model_selection_state_instance = (
            GlobalModelSelectionState.get_instance()
        )

    return _global_model_selection_state_instance


def set_global_model_selection_state(service) -> None:
    """
    Set the global model selection state instance.

    Used during application startup or testing.

    Args:
        service: The GlobalModelSelectionState instance to use
    """
    global _global_model_selection_state_instance
    _global_model_selection_state_instance = service


# =============================================================================
# Provider Resolution Service Dependencies (Unified Provider System)
# =============================================================================

_provider_resolution_service_instance = None


def get_provider_resolution_service() -> "ProviderResolutionService":
    """
    Get the application-scoped ProviderResolutionService instance.

    This singleton resolves the active provider/model for requests
    using a priority-based resolution strategy.

    Returns:
        ProviderResolutionService: The shared resolution service instance
    """
    global _provider_resolution_service_instance

    if _provider_resolution_service_instance is None:
        from app.services.provider_resolution_service import (
            get_provider_resolution_service as get_prs,
        )

        _provider_resolution_service_instance = get_prs()

    return _provider_resolution_service_instance


def set_provider_resolution_service(service) -> None:
    """
    Set the provider resolution service instance.

    Used during application startup or testing.

    Args:
        service: The ProviderResolutionService instance to use
    """
    global _provider_resolution_service_instance
    _provider_resolution_service_instance = service


async def get_resolved_provider_model(
    request: Request,
    session_id: str | None = None,
    explicit_provider: str | None = None,
    explicit_model: str | None = None,
) -> tuple[str, str]:
    """
    FastAPI dependency that resolves the current provider/model.

    Resolution priority:
    1. Explicit parameters (if provided)
    2. Request context (from GlobalModelSelectionState)
    3. Session selection (from database)
    4. User default (from user preferences)
    5. Global default (from config)

    Usage:
        @router.post("/generate")
        async def generate(
            provider_model: tuple[str, str] = Depends(
                get_resolved_provider_model
            )
        ):
            provider, model = provider_model
            # Use resolved provider/model

    Args:
        request: The FastAPI request object
        session_id: Optional session ID for session-based resolution
        explicit_provider: Optional explicit provider override
        explicit_model: Optional explicit model override

    Returns:
        tuple[str, str]: The resolved (provider, model) pair
    """
    resolution_service = get_provider_resolution_service()

    # Extract session_id from request if not provided
    if session_id is None:
        session_id = request.headers.get("X-Session-ID")

    # Extract user_id from request state if available
    user_id = getattr(request.state, "user_id", None)

    provider, model = await resolution_service.resolve(
        explicit_provider=explicit_provider,
        explicit_model=explicit_model,
        session_id=session_id,
        user_id=user_id,
    )

    return provider, model


class ResolvedProviderModel:
    """
    Request-scoped wrapper for resolved provider/model.

    Provides convenient access to both the provider/model pair
    and resolution metadata for debugging.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        resolution_source: str = "unknown",
        request_id: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.resolution_source = resolution_source
        self.request_id = request_id

    def as_tuple(self) -> tuple[str, str]:
        """Return as (provider, model) tuple."""
        return (self.provider, self.model)

    def __iter__(self):
        """Allow tuple unpacking."""
        return iter([self.provider, self.model])

    def __repr__(self) -> str:
        return (
            f"ResolvedProviderModel("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"source={self.resolution_source!r})"
        )


async def get_resolved_provider_model_with_metadata(
    request: Request,
    session_id: str | None = None,
    explicit_provider: str | None = None,
    explicit_model: str | None = None,
) -> ResolvedProviderModel:
    """
    FastAPI dependency with resolution metadata for debugging.

    Same as get_resolved_provider_model but returns a wrapper object
    with additional metadata about how the resolution was performed.

    Args:
        request: The FastAPI request object
        session_id: Optional session ID for session-based resolution
        explicit_provider: Optional explicit provider override
        explicit_model: Optional explicit model override

    Returns:
        ResolvedProviderModel: Wrapper with provider, model, and metadata
    """
    resolution_service = get_provider_resolution_service()

    # Extract session_id from request if not provided
    if session_id is None:
        session_id = request.headers.get("X-Session-ID")

    # Extract user_id and request_id from request
    user_id = getattr(request.state, "user_id", None)
    request_id = (
        request.headers.get("X-Request-ID")
        or getattr(request.state, "request_id", None)
    )

    result = await resolution_service.resolve_with_metadata(
        explicit_provider=explicit_provider,
        explicit_model=explicit_model,
        session_id=session_id,
        user_id=user_id,
    )

    return ResolvedProviderModel(
        provider=result.provider,
        model=result.model,
        resolution_source=result.source,
        request_id=request_id,
    )


# =============================================================================
# Extended Cleanup Functions
# =============================================================================


def reset_provider_services() -> None:
    """
    Reset provider-related service instances.

    Useful for testing to ensure clean state.
    """
    global _global_model_selection_state_instance
    global _provider_resolution_service_instance

    _global_model_selection_state_instance = None
    _provider_resolution_service_instance = None


def reset_all_services() -> None:
    """
    Reset all service instances.

    Useful for testing to ensure clean state between tests.
    """
    global _llm_service_instance
    global _session_service_instance
    global _transformation_service_instance
    global _intent_aware_service_instance
    global _global_model_selection_state_instance
    global _provider_resolution_service_instance

    _llm_service_instance = None
    _session_service_instance = None
    _transformation_service_instance = None
    _intent_aware_service_instance = None
    _global_model_selection_state_instance = None
    _provider_resolution_service_instance = None

    # Clear lru_cache for enhancers
    get_prompt_enhancer.cache_clear()
    get_jailbreak_enhancer.cache_clear()
