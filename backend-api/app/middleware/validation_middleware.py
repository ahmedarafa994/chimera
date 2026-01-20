"""Configuration Validation Middleware.

This middleware validates AI provider configuration at runtime,
ensuring providers are available and within rate limits before
processing requests.

Features:
- Pre-request validation of active provider availability
- Circuit breaker status checking
- Rate limit verification
- Request metrics collection
"""

import logging
import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class ConfigValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate AI provider configuration on requests.

    This middleware performs runtime validation checks before
    processing requests that interact with AI providers.

    Checks performed:
    - Active provider availability
    - Circuit breaker status
    - Rate limit compliance

    The middleware only applies to specific API paths that
    interact with AI providers.
    """

    # Paths that require provider validation
    PROVIDER_PATHS = [
        "/api/v1/llm",
        "/api/v1/generate",
        "/api/v1/chat",
        "/api/v1/completion",
        "/api/v1/adversarial",
        "/api/v1/jailbreak",
        "/api/v1/deepteam",
    ]

    # Paths to skip validation
    SKIP_PATHS = [
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]

    def __init__(
        self,
        app,
        *,
        enabled: bool = True,
        check_circuit_breaker: bool = True,
        check_rate_limits: bool = True,
        collect_metrics: bool = True,
    ) -> None:
        """Initialize validation middleware.

        Args:
            app: ASGI application
            enabled: Whether validation is enabled
            check_circuit_breaker: Check circuit breaker status
            check_rate_limits: Check rate limit compliance
            collect_metrics: Collect request metrics

        """
        super().__init__(app)
        self._enabled = enabled
        self._check_circuit_breaker = check_circuit_breaker
        self._check_rate_limits = check_rate_limits
        self._collect_metrics = collect_metrics
        self._request_count = 0
        self._blocked_count = 0
        self._fallback_count = 0

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with validation checks.

        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain

        Returns:
            Response from next handler or error response

        """
        if not self._enabled:
            return await call_next(request)

        path = request.url.path

        # Skip validation for non-provider paths
        if self._should_skip(path):
            return await call_next(request)

        # Skip validation for paths that don't need it
        if not self._needs_validation(path):
            return await call_next(request)

        start_time = time.perf_counter()
        self._request_count += 1

        try:
            # Check provider availability
            provider_check = await self._check_provider_availability()
            if not provider_check["available"]:
                self._blocked_count += 1
                return self._create_error_response(
                    503,
                    "AI provider unavailable",
                    provider_check.get("reason", "No available providers"),
                )

            # Check circuit breaker if enabled
            if self._check_circuit_breaker:
                cb_check = self._check_circuit_breaker_status(provider_check.get("provider"))
                if cb_check["open"]:
                    self._blocked_count += 1
                    return self._create_error_response(
                        503,
                        "Service temporarily unavailable",
                        f"Circuit breaker open for provider: {provider_check.get('provider')}",
                    )

            # Check rate limits if enabled
            if self._check_rate_limits:
                rate_check = await self._check_rate_limit_compliance(
                    request,
                    provider_check.get("provider"),
                )
                if rate_check["exceeded"]:
                    self._blocked_count += 1
                    return self._create_error_response(
                        429,
                        "Rate limit exceeded",
                        rate_check.get("message", "Too many requests"),
                        headers={"Retry-After": str(rate_check.get("retry_after", 60))},
                    )

            # Store provider info in request state for downstream use
            request.state.validated_provider = provider_check.get("provider")
            request.state.validation_time = time.perf_counter() - start_time

            # Process request
            response = await call_next(request)

            # Record success metrics
            if self._collect_metrics:
                self._record_success(
                    provider_check.get("provider"),
                    time.perf_counter() - start_time,
                )

            return response

        except Exception as e:
            logger.exception(f"Validation middleware error: {e}")
            # Allow request to proceed on validation errors
            # to avoid blocking legitimate requests
            return await call_next(request)

    def _should_skip(self, path: str) -> bool:
        """Check if path should skip validation entirely."""
        return any(path.startswith(skip_path) for skip_path in self.SKIP_PATHS)

    def _needs_validation(self, path: str) -> bool:
        """Check if path needs provider validation."""
        return any(path.startswith(provider_path) for provider_path in self.PROVIDER_PATHS)

    async def _check_provider_availability(self) -> dict:
        """Check if at least one AI provider is available.

        Returns:
            Dict with availability status and active provider

        """
        try:
            from app.core.fallback_manager import get_fallback_manager
            from app.core.startup_validator import StartupValidator

            # Check if system is ready
            if not StartupValidator.is_ready():
                return {
                    "available": False,
                    "reason": "System not ready - startup validation pending",
                }

            fallback_manager = get_fallback_manager()

            # Get default fallback chain
            chain = fallback_manager.get_fallback_chain("default")
            if not chain:
                return {
                    "available": False,
                    "reason": "No providers configured in fallback chain",
                }

            # Find first available provider
            for provider in chain:
                if not fallback_manager.should_skip_provider(provider):
                    return {
                        "available": True,
                        "provider": provider,
                    }

            return {
                "available": False,
                "reason": "All providers in fallback chain are unavailable",
            }

        except ImportError:
            # Fallback manager not available - assume available
            return {
                "available": True,
                "provider": "default",
            }
        except Exception as e:
            logger.warning(f"Provider availability check failed: {e}")
            # Assume available on check failure to avoid blocking
            return {
                "available": True,
                "provider": "unknown",
                "error": str(e),
            }

    def _check_circuit_breaker_status(
        self,
        provider: str | None,
    ) -> dict:
        """Check circuit breaker status for provider.

        Args:
            provider: Provider name to check

        Returns:
            Dict with circuit breaker status

        """
        if not provider:
            return {"open": False}

        try:
            from app.core.shared.circuit_breaker import CircuitBreakerRegistry

            status = CircuitBreakerRegistry.get_status(provider)
            if status is None:
                return {"open": False}

            state = status.get("state", "closed")
            return {
                "open": state == "open",
                "state": state,
                "failure_count": status.get("failure_count", 0),
            }

        except ImportError:
            return {"open": False}
        except Exception as e:
            logger.warning(f"Circuit breaker check failed: {e}")
            return {"open": False, "error": str(e)}

    async def _check_rate_limit_compliance(
        self,
        request: Request,
        provider: str | None,
    ) -> dict:
        """Check if request complies with rate limits.

        Args:
            request: Incoming request
            provider: Provider to check rate limit for

        Returns:
            Dict with rate limit status

        """
        if not provider:
            return {"exceeded": False}

        try:
            from app.core.ai_config_manager import get_ai_config_manager

            config_manager = get_ai_config_manager()
            if not config_manager.is_loaded():
                return {"exceeded": False}

            config = config_manager.get_config()
            provider_config = config.providers.get(provider)

            if not provider_config or not provider_config.rate_limit:
                return {"exceeded": False}

            rate_limit = provider_config.rate_limit

            # Simple in-memory rate limit check
            # In production, use Redis or similar
            # This is a placeholder for proper rate limiting
            client_id = self._get_client_id(request)
            current_rate = self._get_current_rate(client_id, provider)

            if current_rate >= rate_limit.requests_per_minute:
                return {
                    "exceeded": True,
                    "message": (
                        f"Rate limit exceeded: {current_rate}/{rate_limit.requests_per_minute} rpm"
                    ),
                    "retry_after": 60,
                }

            return {"exceeded": False}

        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return {"exceeded": False, "error": str(e)}

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key[:8]}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        if request.client:
            return f"ip:{request.client.host}"

        return "unknown"

    def _get_current_rate(
        self,
        client_id: str,
        provider: str,
    ) -> int:
        """Get current request rate for client/provider.

        This is a placeholder implementation.
        Production should use Redis or similar.
        """
        # TODO: Implement proper rate tracking
        return 0

    def _record_success(
        self,
        provider: str | None,
        latency: float,
    ) -> None:
        """Record successful request metrics."""
        try:
            from app.core.fallback_manager import get_fallback_manager

            if provider:
                fallback_manager = get_fallback_manager()
                fallback_manager.record_success(provider)

        except Exception:
            pass  # Metrics collection should not fail requests

    def _create_error_response(
        self,
        status_code: int,
        error: str,
        detail: str,
        headers: dict | None = None,
    ) -> JSONResponse:
        """Create JSON error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "detail": detail,
                "status_code": status_code,
            },
            headers=headers,
        )

    def get_stats(self) -> dict:
        """Get middleware statistics."""
        return {
            "enabled": self._enabled,
            "total_requests": self._request_count,
            "blocked_requests": self._blocked_count,
            "fallback_requests": self._fallback_count,
            "block_rate": (
                self._blocked_count / self._request_count if self._request_count > 0 else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset middleware statistics."""
        self._request_count = 0
        self._blocked_count = 0
        self._fallback_count = 0


def create_validation_middleware(
    enabled: bool = True,
    check_circuit_breaker: bool = True,
    check_rate_limits: bool = True,
    collect_metrics: bool = True,
) -> Callable:
    """Factory function to create validation middleware.

    Args:
        enabled: Whether validation is enabled
        check_circuit_breaker: Check circuit breaker status
        check_rate_limits: Check rate limit compliance
        collect_metrics: Collect request metrics

    Returns:
        Middleware class configured with options

    Example:
        from fastapi import FastAPI
        from app.middleware.validation_middleware import (
            create_validation_middleware
        )

        app = FastAPI()
        app.add_middleware(
            create_validation_middleware(
                enabled=True,
                check_circuit_breaker=True,
            )
        )

    """

    class ConfiguredValidationMiddleware(ConfigValidationMiddleware):
        def __init__(self, app) -> None:
            super().__init__(
                app,
                enabled=enabled,
                check_circuit_breaker=check_circuit_breaker,
                check_rate_limits=check_rate_limits,
                collect_metrics=collect_metrics,
            )

    return ConfiguredValidationMiddleware
