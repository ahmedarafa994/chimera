"""
Selection Validation Middleware for Provider/Model Selection System.

This middleware validates provider/model selections at each request,
ensuring that:
1. Provider exists in the registry
2. Model exists for the selected provider
3. API key is configured (if required)
4. Provider is healthy and available

The middleware integrates with GlobalModelSelectionState and the provider
plugin system to perform comprehensive validation.

Usage:
    from app.middleware.selection_validation_middleware import (
        SelectionValidationMiddleware
    )
    app.add_middleware(SelectionValidationMiddleware)
"""

import logging
import time
from typing import Callable, Optional, Set

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.selection_context import SelectionContext
from app.schemas.validation_errors import (
    SelectionValidationError,
    ValidationErrorType,
    ValidationResult,
    ValidationMetrics,
    create_invalid_provider_error,
    create_invalid_model_error,
    create_missing_api_key_error,
    create_provider_unhealthy_error,
    create_context_missing_error,
)

logger = logging.getLogger(__name__)


# Global metrics instance for validation tracking
_validation_metrics = ValidationMetrics()


def get_validation_metrics() -> ValidationMetrics:
    """Get the global validation metrics instance."""
    return _validation_metrics


class SelectionValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that validates provider/model selection at each request.

    Responsibilities:
    1. Extract session_id from request headers/cookies
    2. Load or resolve current selection
    3. Validate selection is valid (provider exists, model exists)
    4. Validate API key is configured for provider
    5. Check provider health status
    6. Set validated selection in request state
    7. Log validation result

    Configuration:
        bypass_paths: Set of paths to skip validation
        fail_on_invalid: Whether to return 400 on invalid selection
        check_health: Whether to check provider health
        check_api_key: Whether to verify API key exists
    """

    # Default paths that bypass validation
    DEFAULT_BYPASS_PATHS: Set[str] = {
        "/",
        "/health",
        "/health/ping",
        "/health/ready",
        "/health/live",
        "/health/full",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/health",
        "/api/v1/providers",
        "/api/v1/unified-providers",
        "/api/v1/admin",
    }

    # Path prefixes to bypass
    DEFAULT_BYPASS_PREFIXES: Set[str] = {
        "/api/v1/unified-providers/",
        "/api/v1/health/",
        "/api/v1/admin/",
    }

    def __init__(
        self,
        app,
        bypass_paths: Optional[list[str]] = None,
        bypass_prefixes: Optional[Set[str]] = None,
        fail_on_invalid: bool = False,
        enable_health_check: bool = True,
        check_api_key: bool = True,
        cache_ttl_seconds: float = 60.0,
        log_level: str = "DEBUG",
    ):
        """
        Initialize the selection validation middleware.

        Args:
            app: FastAPI application instance
            bypass_paths: Paths to skip validation (exact match)
            bypass_prefixes: Path prefixes to skip validation
            fail_on_invalid: Return 400 on invalid selection
            enable_health_check: Whether to check provider health status
            check_api_key: Whether to verify API key is configured
            cache_ttl_seconds: Cache TTL for provider/model lists
            log_level: Logging level for validation messages
        """
        super().__init__(app)
        self._bypass_paths = set(bypass_paths) if bypass_paths else self.DEFAULT_BYPASS_PATHS
        self._bypass_prefixes = bypass_prefixes or self.DEFAULT_BYPASS_PREFIXES
        self._fail_on_invalid = fail_on_invalid
        self._check_health = enable_health_check
        self._check_api_key = check_api_key
        self._cache_ttl_seconds = cache_ttl_seconds
        self._log_level = getattr(logging, log_level.upper(), logging.DEBUG)

        logger.info(
            f"SelectionValidationMiddleware initialized "
            f"(fail_on_invalid={fail_on_invalid}, "
            f"check_health={enable_health_check}, "
            f"check_api_key={check_api_key})"
        )

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Validate selection and process request.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response from handler or validation error response
        """
        # Check if validation should be bypassed
        if self._should_bypass(request.url.path):
            return await call_next(request)

        # Get request ID for logging
        request_id = getattr(request.state, "request_id", "unknown")

        # Perform validation
        start_time = time.time()
        validation_result = await self._validate_selection(request, request_id)
        validation_result.validation_time_ms = (
            (time.time() - start_time) * 1000
        )

        # Record metrics
        _validation_metrics.record_validation(validation_result)

        # Store validation result in request state
        request.state.validation_result = validation_result

        # Log validation result
        self._log_validation(request_id, validation_result)

        # Handle invalid selection
        if not validation_result.is_valid and self._fail_on_invalid:
            return self._create_error_response(validation_result)

        # Proceed with request
        response = await call_next(request)

        # Add validation headers to response
        for header, value in validation_result.to_headers().items():
            response.headers[header] = value

        return response

    def _should_bypass(self, path: str) -> bool:
        """Check if validation should be bypassed for this path."""
        # Exact path match
        if path in self._bypass_paths:
            return True

        # Prefix match
        for prefix in self._bypass_prefixes:
            if path.startswith(prefix):
                return True

        return False

    async def _validate_selection(
        self, request: Request, request_id: str
    ) -> ValidationResult:
        """
        Perform comprehensive selection validation.

        Args:
            request: FastAPI request
            request_id: Request identifier for logging

        Returns:
            ValidationResult with validation status and errors
        """
        errors: list[SelectionValidationError] = []
        warnings: list[str] = []

        # Get selection from context or request state
        selection = self._get_selection(request)

        if not selection:
            errors.append(create_context_missing_error(request_id))
            return ValidationResult(
                is_valid=False,
                provider="unknown",
                model="unknown",
                errors=errors,
                source="none",
            )

        provider_id = selection.provider_id
        model_id = selection.model_id
        source = selection.scope.value if selection.scope else "unknown"

        # Validate provider exists
        provider_valid, available_providers = await self._validate_provider(
            provider_id
        )
        if not provider_valid:
            errors.append(create_invalid_provider_error(
                provider_id, available_providers, request_id
            ))

        # Validate model exists for provider
        if provider_valid:
            model_valid, available_models = await self._validate_model(
                provider_id, model_id
            )
            if not model_valid:
                errors.append(create_invalid_model_error(
                    provider_id, model_id, available_models, request_id
                ))

        # Check API key if enabled
        api_key_configured = None
        if self._check_api_key and provider_valid:
            api_key_configured = await self._check_api_key_configured(
                provider_id
            )
            if not api_key_configured:
                errors.append(create_missing_api_key_error(
                    provider_id, request_id
                ))

        # Check provider health if enabled
        provider_health = None
        if self._check_health and provider_valid:
            provider_health = await self._check_provider_health(provider_id)
            if provider_health == "unhealthy":
                errors.append(create_provider_unhealthy_error(
                    provider_id, provider_health, request_id
                ))
            elif provider_health == "degraded":
                warnings.append(
                    f"Provider '{provider_id}' is degraded"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            provider=provider_id,
            model=model_id,
            errors=errors,
            warnings=warnings,
            source=source,
            provider_health=provider_health,
            api_key_configured=api_key_configured,
        )

    def _get_selection(self, request: Request):
        """Get selection from context or request state."""
        # Try SelectionContext first
        if SelectionContext.is_set():
            return SelectionContext.get_selection()

        # Try request state (set by SelectionMiddleware)
        if hasattr(request.state, "selection"):
            return request.state.selection

        return None

    async def _validate_provider(
        self, provider_id: str
    ) -> tuple[bool, list[str]]:
        """
        Validate that provider exists in registry.

        Returns:
            Tuple of (is_valid, available_providers)
        """
        try:
            from app.services.provider_plugins import (
                is_plugin_registered,
                get_all_provider_types,
            )

            is_valid = is_plugin_registered(provider_id)
            available = get_all_provider_types()
            return is_valid, available

        except ImportError:
            logger.warning("Provider plugin system not available")
            return True, []  # Allow if system unavailable

        except Exception as e:
            logger.error(f"Provider validation error: {e}")
            return True, []

    async def _validate_model(
        self, provider_id: str, model_id: str
    ) -> tuple[bool, list[str]]:
        """
        Validate that model exists for provider.

        Returns:
            Tuple of (is_valid, available_models)
        """
        try:
            from app.services.provider_plugins import get_plugin

            plugin = get_plugin(provider_id)
            if not plugin:
                return False, []

            # Try to get cached models
            if hasattr(plugin, "get_cached_models"):
                models = await plugin.get_cached_models()
            else:
                models = await plugin.list_models()

            model_ids = [m.model_id for m in models]
            is_valid = model_id in model_ids

            return is_valid, model_ids

        except ImportError:
            logger.warning("Provider plugin system not available")
            return True, []

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return True, []

    async def _check_api_key_configured(self, provider_id: str) -> bool:
        """Check if API key is configured for provider."""
        try:
            from app.services.provider_plugins import get_plugin
            import os

            plugin = get_plugin(provider_id)
            if not plugin:
                return False

            # Check if provider requires API key
            info = plugin.get_provider_info()
            if not info.requires_api_key:
                return True

            # Check if API key is set in environment
            env_var = info.api_key_env_var
            if env_var and os.environ.get(env_var):
                return True

            # Check common environment variable patterns
            common_patterns = [
                f"{provider_id.upper()}_API_KEY",
                f"{provider_id.upper().replace('-', '_')}_API_KEY",
            ]
            for pattern in common_patterns:
                if os.environ.get(pattern):
                    return True

            return False

        except Exception as e:
            logger.error(f"API key check error: {e}")
            return True  # Don't block on check errors

    async def _check_provider_health(
        self, provider_id: str
    ) -> Optional[str]:
        """
        Check provider health status.

        Returns:
            "healthy", "degraded", "unhealthy", or None
        """
        try:
            from app.core.health import health_checker

            # Try to get provider-specific health
            result = await health_checker.check_service(f"provider_{provider_id}")
            if result:
                return result.status.value

            return "healthy"  # Default to healthy if no check available

        except ImportError:
            return None

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return None

    def _log_validation(
        self, request_id: str, result: ValidationResult
    ) -> None:
        """Log validation result at appropriate level."""
        if result.is_valid:
            logger.log(
                self._log_level,
                f"[{request_id}] Selection validated: "
                f"{result.provider}/{result.model} (source={result.source})"
            )
        else:
            error_types = [e.error_type.value for e in result.errors]
            logger.warning(
                f"[{request_id}] Selection validation failed: "
                f"{result.provider}/{result.model} - errors={error_types}"
            )

    def _create_error_response(
        self, result: ValidationResult
    ) -> JSONResponse:
        """Create error response for invalid selection."""
        first_error = result.first_error
        if not first_error:
            first_error = SelectionValidationError(
                error_type=ValidationErrorType.INVALID_PROVIDER,
                message="Selection validation failed",
                provider=result.provider,
                model=result.model,
            )

        return JSONResponse(
            status_code=400,
            content=first_error.to_response_dict(),
            headers=result.to_headers(),
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SelectionValidationMiddleware",
    "get_validation_metrics",
]
