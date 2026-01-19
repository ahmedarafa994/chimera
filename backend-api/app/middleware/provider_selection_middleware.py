"""
Provider Selection Middleware

Middleware that injects provider/model context into requests based on:
1. HTTP headers (X-Provider, X-Model, X-Failover-Chain)
2. Query parameters (provider, model, failover_chain)
3. Default configuration from AIConfigManager

This middleware ensures consistent provider selection across the request pipeline.
"""

import logging
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class ProviderContext:
    """Context object for provider/model selection stored in request state."""

    def __init__(
        self,
        provider: str,
        model: str,
        failover_chain: str | None = None,
        source: str = "default",
    ):
        """
        Initialize provider context.

        Args:
            provider: Provider ID (e.g., "openai", "gemini")
            model: Model ID (e.g., "gpt-4", "gemini-2.0-flash-exp")
            failover_chain: Optional named failover chain
            source: Where the selection came from ("header", "query", "default")
        """
        self.provider = provider
        self.model = model
        self.failover_chain = failover_chain
        self.source = source

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "failover_chain": self.failover_chain,
            "source": self.source,
        }


class ProviderSelectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to inject provider/model context into requests.

    Checks for provider/model override in the following order:
    1. HTTP headers (X-Provider, X-Model, X-Failover-Chain)
    2. Query parameters (provider, model, failover_chain)
    3. Default from AI config manager

    The selected provider context is stored in request.state.provider_context
    and can be accessed by downstream handlers.

    Example:
        # Access in route handler
        @app.get("/api/v1/generate")
        async def generate(request: Request):
            context = get_request_provider(request)
            provider, model = context
            # Use provider and model...
    """

    # Headers for provider selection
    HEADER_PROVIDER = "X-Provider"
    HEADER_MODEL = "X-Model"
    HEADER_FAILOVER_CHAIN = "X-Failover-Chain"

    # Query params for provider selection
    QUERY_PROVIDER = "provider"
    QUERY_MODEL = "model"
    QUERY_FAILOVER_CHAIN = "failover_chain"

    # Paths to skip provider injection
    SKIP_PATHS = [
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
        "/api/v1/health",
        "/api/v1/auth",  # Exclude auth
    ]

    def __init__(
        self,
        app,
        *,
        allow_header_override: bool = True,
        allow_query_override: bool = True,
        validate_provider: bool = True,
    ):
        """
        Initialize provider selection middleware.

        Args:
            app: ASGI application
            allow_header_override: Allow provider override via headers
            allow_query_override: Allow provider override via query params
            validate_provider: Validate that selected provider exists
        """
        super().__init__(app)
        self._allow_header_override = allow_header_override
        self._allow_query_override = allow_query_override
        self._validate_provider = validate_provider
        self._config_manager = None

    def _get_config_manager(self):
        """Lazily get the AI config manager."""
        if self._config_manager is None:
            try:
                from app.core.ai_config_manager import get_ai_config_manager

                self._config_manager = get_ai_config_manager()
            except ImportError:
                logger.warning("AIConfigManager not available")
        return self._config_manager

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and inject provider context.

        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain

        Returns:
            Response from next handler
        """
        path = request.url.path

        # Skip provider injection for certain paths
        if self._should_skip(path):
            return await call_next(request)

        try:
            # Determine provider context
            context = self._resolve_provider_context(request)

            # Validate provider if enabled
            if self._validate_provider and context:
                validation_result = self._validate_provider_selection(context)
                if not validation_result["valid"]:
                    return self._create_error_response(
                        400,
                        "Invalid provider selection",
                        validation_result.get("reason", "Provider validation failed"),
                    )

            # Store context in request state
            request.state.provider_context = context

            # Log selection
            if context:
                logger.debug(
                    f"Provider context: provider={context.provider}, "
                    f"model={context.model}, source={context.source}"
                )

            # Add provider info to response headers
            response = await call_next(request)

            if context:
                response.headers["X-Provider-Used"] = context.provider
                response.headers["X-Model-Used"] = context.model

            return response

        except Exception as e:
            logger.error(f"Provider selection middleware error: {e}")
            # Continue without provider context on error
            return await call_next(request)

    def _should_skip(self, path: str) -> bool:
        """Check if path should skip provider injection."""
        return any(path.startswith(skip_path) for skip_path in self.SKIP_PATHS)

    def _resolve_provider_context(self, request: Request) -> ProviderContext | None:
        """
        Resolve provider context from request.

        Checks in order:
        1. HTTP headers
        2. Query parameters
        3. Default configuration

        Args:
            request: Incoming request

        Returns:
            ProviderContext or None if no config available
        """
        provider = None
        model = None
        failover_chain = None
        source = "default"

        # 1. Check headers (highest priority)
        if self._allow_header_override:
            header_provider = request.headers.get(self.HEADER_PROVIDER)
            header_model = request.headers.get(self.HEADER_MODEL)
            header_chain = request.headers.get(self.HEADER_FAILOVER_CHAIN)

            if header_provider:
                provider = header_provider
                source = "header"
            if header_model:
                model = header_model
                source = "header"
            if header_chain:
                failover_chain = header_chain
                source = "header"

        # 2. Check query parameters (if not already set by headers)
        if self._allow_query_override:
            query_params = request.query_params

            if not provider and self.QUERY_PROVIDER in query_params:
                provider = query_params[self.QUERY_PROVIDER]
                source = "query"
            if not model and self.QUERY_MODEL in query_params:
                model = query_params[self.QUERY_MODEL]
                source = "query"
            if not failover_chain and self.QUERY_FAILOVER_CHAIN in query_params:
                failover_chain = query_params[self.QUERY_FAILOVER_CHAIN]
                source = "query"

        # 3. Get defaults from config manager
        config_manager = self._get_config_manager()
        if config_manager and config_manager.is_loaded():
            config = config_manager.get_config()

            if not provider:
                provider = config.global_config.default_provider

            if not model:
                # Get default model for the provider
                provider_config = config.get_provider(provider) if provider else None
                if provider_config:
                    default_model = provider_config.get_default_model()
                    model = default_model.model_id if default_model else None
                else:
                    model = config.global_config.default_model

        # If still no provider/model, return None
        if not provider:
            logger.warning("No provider configured or specified")
            return None

        return ProviderContext(
            provider=provider,
            model=model or "",
            failover_chain=failover_chain,
            source=source,
        )

    def _validate_provider_selection(
        self,
        context: ProviderContext,
    ) -> dict:
        """
        Validate the provider selection.

        Args:
            context: Provider context to validate

        Returns:
            Dict with validation result
        """
        config_manager = self._get_config_manager()
        if not config_manager or not config_manager.is_loaded():
            # Cannot validate without config - allow
            return {"valid": True}

        config = config_manager.get_config()

        # Check provider exists
        provider = config.get_provider(context.provider)
        if not provider:
            return {
                "valid": False,
                "reason": f"Unknown provider: {context.provider}",
            }

        # Check provider is enabled
        if not provider.enabled:
            return {
                "valid": False,
                "reason": f"Provider '{context.provider}' is disabled",
            }

        # Check model exists if specified
        if context.model:
            model = provider.get_model(context.model)
            if not model:
                # Model not found in provider - check other providers
                found_elsewhere = False
                for p in config.providers.values():
                    if p.get_model(context.model):
                        found_elsewhere = True
                        break

                if found_elsewhere:
                    return {
                        "valid": False,
                        "reason": (
                            f"Model '{context.model}' not available for "
                            f"provider '{context.provider}'"
                        ),
                    }
                else:
                    return {
                        "valid": False,
                        "reason": f"Unknown model: {context.model}",
                    }

        # Check failover chain exists if specified
        if context.failover_chain and context.failover_chain not in config.failover_chains:
            return {
                "valid": False,
                "reason": f"Unknown failover chain: {context.failover_chain}",
            }

        return {"valid": True}

    def _create_error_response(
        self,
        status_code: int,
        error: str,
        detail: str,
    ) -> JSONResponse:
        """Create JSON error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "detail": detail,
                "status_code": status_code,
            },
        )


def get_request_provider(request: Request) -> tuple[str, str]:
    """
    Extract provider and model from request state.

    This is a helper function to access the provider context
    set by ProviderSelectionMiddleware.

    Args:
        request: Request object with provider context

    Returns:
        Tuple of (provider_id, model_id)

    Raises:
        ValueError: If no provider context is set
    """
    context = getattr(request.state, "provider_context", None)

    if not context:
        # Try to get from validated_provider (set by validation middleware)
        validated = getattr(request.state, "validated_provider", None)
        if validated:
            return (validated, "")

        raise ValueError("No provider context available in request")

    return (context.provider, context.model)


def get_request_provider_context(request: Request) -> ProviderContext | None:
    """
    Get the full provider context from request state.

    Args:
        request: Request object

    Returns:
        ProviderContext or None
    """
    return getattr(request.state, "provider_context", None)


def get_request_failover_chain(request: Request) -> str | None:
    """
    Get the failover chain name from request state.

    Args:
        request: Request object

    Returns:
        Failover chain name or None
    """
    context = getattr(request.state, "provider_context", None)
    return context.failover_chain if context else None


def create_provider_selection_middleware(
    allow_header_override: bool = True,
    allow_query_override: bool = True,
    validate_provider: bool = True,
) -> type:
    """
    Factory function to create middleware instance with custom configuration.

    Args:
        allow_header_override: Allow provider override via headers
        allow_query_override: Allow provider override via query params
        validate_provider: Validate that selected provider exists

    Returns:
        Configured ProviderSelectionMiddleware class

    Example:
        from fastapi import FastAPI
        from app.middleware.provider_selection_middleware import (
            create_provider_selection_middleware
        )

        app = FastAPI()
        app.add_middleware(
            create_provider_selection_middleware(
                allow_header_override=True,
                validate_provider=True,
            )
        )
    """

    class ConfiguredProviderSelectionMiddleware(ProviderSelectionMiddleware):
        def __init__(self, app):
            super().__init__(
                app,
                allow_header_override=allow_header_override,
                allow_query_override=allow_query_override,
                validate_provider=validate_provider,
            )

    return ConfiguredProviderSelectionMiddleware
