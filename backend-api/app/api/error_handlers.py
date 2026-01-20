"""Standardized API error handling for backend routes.

HIGH-002 FIX: Enhanced exception handling with comprehensive coverage
for all error types including circuit breaker, async, and provider errors.

This module provides decorators and utilities for consistent error handling
across all API endpoints, with proper error code mapping and sanitization.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import APIException, LLMProviderError, ValidationError
from app.core.shared.circuit_breaker import CircuitBreakerOpen
from app.core.unified_errors import ChimeraError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def api_error_handler(
    operation_name: str,
    default_error_message: str = "An unexpected error occurred",
):
    """Decorator for standardized API error handling.

    Features:
    - Maps known exceptions to appropriate HTTP status codes
    - Sanitizes error messages for external clients
    - Logs full error details internally
    - Preserves exception chains for debugging
    - Generates support reference IDs for tracking

    Usage:
        @router.post("/generate")
        @api_error_handler("generate_content")
        async def generate_content(request: PromptRequest):
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate a request ID for tracking
            request_id = f"req_{uuid.uuid4().hex[:12]}"

            try:
                return await func(*args, **kwargs)

            except ChimeraError:
                # Allow unified ChimeraErrors to bubble up
                # to the global exception handler
                raise

            except HTTPException:
                # Re-raise FastAPI HTTP exceptions as-is
                raise

            except APIException as e:
                # Our custom exceptions - map to appropriate HTTP response
                logger.warning(
                    f"[{request_id}] {operation_name} failed with {e.error_code}: {e.message}",
                    extra={"details": e.details, "request_id": request_id},
                )
                raise HTTPException(
                    status_code=e.status_code,
                    detail={
                        "error": e.error_code,
                        "message": e.message,
                        "details": e.details,
                        "request_id": request_id,
                    },
                ) from e

            except ValueError as e:
                # Input validation errors
                logger.warning(f"[{request_id}] {operation_name} validation error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "VALIDATION_ERROR",
                        "message": str(e),
                        "request_id": request_id,
                    },
                ) from e

            except TimeoutError as e:
                logger.exception(f"[{request_id}] {operation_name} timeout: {e}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "error": "TIMEOUT_ERROR",
                        "message": "The request timed out. Please try again.",
                        "request_id": request_id,
                    },
                ) from e

            except ConnectionError as e:
                logger.exception(f"[{request_id}] {operation_name} connection error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "SERVICE_UNAVAILABLE",
                        "message": "Unable to connect to required service. Please try again later.",
                        "request_id": request_id,
                    },
                ) from e

            except CircuitBreakerOpen as e:
                # HIGH-002 FIX: Handle circuit breaker open state
                logger.warning(
                    f"[{request_id}] {operation_name} circuit breaker open: "
                    f"{e.name}, retry after {e.retry_after:.1f}s",
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "CIRCUIT_BREAKER_OPEN",
                        "message": f"Service '{e.name}' is temporarily unavailable. "
                        f"Please retry after {int(e.retry_after)} seconds.",
                        "request_id": request_id,
                        "retry_after": int(e.retry_after),
                    },
                    headers={"Retry-After": str(int(e.retry_after))},
                ) from e

            except asyncio.CancelledError as e:
                # HIGH-002 FIX: Handle async cancellation gracefully
                logger.warning(f"[{request_id}] {operation_name} was cancelled")
                raise HTTPException(
                    status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
                    detail={
                        "error": "REQUEST_CANCELLED",
                        "message": "The request was cancelled.",
                        "request_id": request_id,
                    },
                ) from e

            except MemoryError as e:
                # HIGH-002 FIX: Handle memory errors
                logger.critical(f"[{request_id}] {operation_name} memory error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "RESOURCE_EXHAUSTED",
                        "message": "Server resources temporarily exhausted. "
                        "Please try again later.",
                        "request_id": request_id,
                    },
                ) from e

            except PermissionError as e:
                # HIGH-002 FIX: Handle permission errors
                logger.exception(f"[{request_id}] {operation_name} permission error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "PERMISSION_DENIED",
                        "message": "Access to the requested resource is denied.",
                        "request_id": request_id,
                    },
                ) from e

            except FileNotFoundError as e:
                # HIGH-002 FIX: Handle file not found errors
                logger.warning(f"[{request_id}] {operation_name} file not found: {e}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "RESOURCE_NOT_FOUND",
                        "message": "The requested resource was not found.",
                        "request_id": request_id,
                    },
                ) from e

            except Exception as e:
                # Unexpected errors - log full details, return sanitized message
                logger.exception(
                    f"[{request_id}] {operation_name} unexpected error: {type(e).__name__}: {e}",
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "INTERNAL_ERROR",
                        "message": default_error_message,
                        "request_id": request_id,
                    },
                ) from e

        return wrapper

    return decorator


class ErrorResponseBuilder:
    """Builder for consistent error responses.

    Usage:
        raise ErrorResponseBuilder.bad_request("Invalid input", field="prompt")
        raise ErrorResponseBuilder.not_found("Technique", technique_id)
        raise ErrorResponseBuilder.rate_limited(60, 100)
    """

    @staticmethod
    def bad_request(message: str, **details) -> HTTPException:
        """Create a 400 Bad Request response."""
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "BAD_REQUEST",
                "message": message,
                "details": details if details else None,
            },
        )

    @staticmethod
    def validation_error(message: str, field: str | None = None, **details) -> HTTPException:
        """Create a 400 validation error response."""
        error_details = {"field": field} if field else {}
        error_details.update(details)
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "VALIDATION_ERROR",
                "message": message,
                "details": error_details if error_details else None,
            },
        )

    @staticmethod
    def unauthorized(message: str = "Authentication required") -> HTTPException:
        """Create a 401 Unauthorized response."""
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "UNAUTHORIZED", "message": message},
        )

    @staticmethod
    def forbidden(message: str = "Access denied") -> HTTPException:
        """Create a 403 Forbidden response."""
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "FORBIDDEN", "message": message},
        )

    @staticmethod
    def not_found(resource: str, identifier: str | None = None) -> HTTPException:
        """Create a 404 Not Found response."""
        message = f"{resource} not found"
        details = {"identifier": identifier} if identifier else None
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "NOT_FOUND", "message": message, "details": details},
        )

    @staticmethod
    def rate_limited(
        retry_after: int,
        limit: int | None = None,
        window: str = "minute",
    ) -> HTTPException:
        """Create a 429 Too Many Requests response."""
        message = "Rate limit exceeded"
        if limit:
            message = f"Rate limit of {limit} requests per {window} exceeded"
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": message,
                "details": {"retry_after_seconds": retry_after},
            },
            headers={"Retry-After": str(retry_after)},
        )

    @staticmethod
    def conflict(message: str, **details) -> HTTPException:
        """Create a 409 Conflict response."""
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "CONFLICT",
                "message": message,
                "details": details if details else None,
            },
        )

    @staticmethod
    def payload_too_large(
        max_size: int | None = None,
        actual_size: int | None = None,
    ) -> HTTPException:
        """Create a 413 Payload Too Large response."""
        message = "Request payload is too large"
        details = {}
        if max_size:
            details["max_size_bytes"] = max_size
        if actual_size:
            details["actual_size_bytes"] = actual_size
        return HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "PAYLOAD_TOO_LARGE",
                "message": message,
                "details": details if details else None,
            },
        )

    @staticmethod
    def provider_unavailable(provider: str, reason: str | None = None) -> HTTPException:
        """Create a 503 provider unavailable response."""
        message = f"Provider '{provider}' is currently unavailable"
        if reason:
            message = f"{message}: {reason}"
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "PROVIDER_UNAVAILABLE",
                "message": message,
                "details": {"provider": provider},
            },
        )

    @staticmethod
    def service_unavailable(
        service: str | None = None,
        message: str | None = None,
    ) -> HTTPException:
        """Create a 503 Service Unavailable response."""
        error_message = message or "Service temporarily unavailable"
        if service:
            error_message = f"{service} is temporarily unavailable"
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "SERVICE_UNAVAILABLE", "message": error_message},
        )

    @staticmethod
    def gateway_timeout(operation: str | None = None) -> HTTPException:
        """Create a 504 Gateway Timeout response."""
        message = "The request timed out"
        if operation:
            message = f"{operation} timed out"
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"error": "GATEWAY_TIMEOUT", "message": message},
        )

    @staticmethod
    def internal_error(support_ref: str | None = None) -> HTTPException:
        """Create a 500 Internal Server Error response."""
        detail: dict[str, Any] = {
            "error": "INTERNAL_ERROR",
            "message": "An internal error occurred. Please try again later.",
        }
        if support_ref:
            detail["support_reference"] = support_ref
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


# Exception handlers for FastAPI app registration


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Global handler for APIException and subclasses."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.warning(
        f"[{request_id}] API Exception: {exc.error_code} - {exc.message}",
        extra={"path": request.url.path, "details": exc.details},
    )
    response_content = exc.to_dict()
    response_content["request_id"] = request_id
    return JSONResponse(status_code=exc.status_code, content=response_content)


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handler for validation errors."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.warning(
        f"[{request_id}] Validation Error: {exc.message}",
        extra={"path": request.url.path, "details": exc.details},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": request_id,
        },
    )


async def llm_exception_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
    """Handler for LLM provider errors."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.error(
        f"[{request_id}] LLM Provider Error: {exc.error_code} - {exc.message}",
        extra={"path": request.url.path, "details": exc.details},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": request_id,
        },
    )


async def circuit_breaker_exception_handler(
    request: Request,
    exc: CircuitBreakerOpen,
) -> JSONResponse:
    """Handler for circuit breaker open errors."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.warning(
        f"[{request_id}] Circuit Breaker Open: {exc.name}",
        extra={"path": request.url.path, "circuit_name": exc.name, "retry_after": exc.retry_after},
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "CIRCUIT_BREAKER_OPEN",
            "message": f"Service '{exc.name}' is temporarily unavailable. "
            f"Please retry after {int(exc.retry_after)} seconds.",
            "request_id": request_id,
            "retry_after": int(exc.retry_after),
        },
        headers={"Retry-After": str(int(exc.retry_after))},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global fallback handler for unhandled exceptions.

    HIGH-002 FIX: Ensures no unhandled exceptions leak internal details.
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.exception(
        f"[{request_id}] Unhandled exception: {type(exc).__name__}: {exc}",
        extra={"path": request.url.path},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request_id,
        },
    )


def register_exception_handlers(app) -> None:
    """Register all custom exception handlers with FastAPI app.

    HIGH-002 FIX: Enhanced to include circuit breaker and generic handlers.

    Usage:
        from app.api.error_handlers import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)
    """
    from app.core.exceptions import APIException, LLMProviderError, ValidationError

    # Register specific exception handlers
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(LLMProviderError, llm_exception_handler)
    app.add_exception_handler(CircuitBreakerOpen, circuit_breaker_exception_handler)

    # Register generic fallback handler for unhandled exceptions
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Registered custom exception handlers (including circuit breaker)")
