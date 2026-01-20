"""Comprehensive Error Handling System - FIXED VERSION.

This module provides:
1. Standardized error responses across all endpoints
2. Proper error logging and monitoring
3. User-friendly error messages
4. Security-aware error disclosure
5. Performance-optimized error handling
"""

import logging
import traceback
from datetime import datetime
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    error: str
    message: str
    details: dict[str, Any] | None = None
    timestamp: str
    request_id: str | None = None
    path: str | None = None


class ApiErrorType:
    """Standard API error types."""

    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    BUSINESS_LOGIC_ERROR = "BUSINESS_LOGIC_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    NOT_FOUND_ERROR = "NOT_FOUND_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"


class ErrorHandler:
    """Centralized error handling with logging and monitoring."""

    def __init__(self) -> None:
        self.error_counts = {}

    def log_error(self, error_type: str, error: Exception, request: Request = None) -> None:
        """Log error with context information."""
        # Count error occurrences
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Create log context
        context = {
            "error_type": error_type,
            "error_class": error.__class__.__name__,
            "error_message": str(error),
            "error_count": self.error_counts[error_type],
        }

        if request:
            context.update(
                {
                    "path": request.url.path,
                    "method": request.method,
                    "client_ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown"),
                    "request_id": getattr(request.state, "request_id", "unknown"),
                },
            )

        # Log based on error severity
        if error_type == ApiErrorType.INTERNAL_SERVER_ERROR:
            logger.error(f"Internal server error: {context}")
        elif error_type in [ApiErrorType.AUTHENTICATION_ERROR, ApiErrorType.AUTHORIZATION_ERROR]:
            logger.warning(f"Security error: {context}")
        else:
            logger.info(f"API error: {context}")

    def create_error_response(
        self,
        error_type: str,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
        request: Request = None,
    ) -> JSONResponse:
        """Create standardized error response."""
        error_response = ErrorResponse(
            error=error_type,
            message=message,
            details=details,
            timestamp=datetime.utcnow().isoformat(),
            request_id=getattr(request.state, "request_id", None) if request else None,
            path=request.url.path if request else None,
        )

        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(exclude_none=True),
        )


# Global error handler instance
error_handler = ErrorHandler()


# Exception handlers
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = exc.errors()

    # Format validation errors
    formatted_errors = []
    for error in errors:
        field_path = ".".join(str(loc) for loc in error.get("loc", []))
        formatted_errors.append(
            {
                "field": field_path,
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "validation_error"),
                "input": error.get("input"),
            },
        )

    error_handler.log_error(ApiErrorType.VALIDATION_ERROR, exc, request)

    return error_handler.create_error_response(
        error_type=ApiErrorType.VALIDATION_ERROR,
        message="Request validation failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={
            "validation_errors": formatted_errors,
            "error_count": len(formatted_errors),
        },
        request=request,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    # Map status codes to error types
    error_type_mapping = {
        400: ApiErrorType.VALIDATION_ERROR,
        401: ApiErrorType.AUTHENTICATION_ERROR,
        403: ApiErrorType.AUTHORIZATION_ERROR,
        404: ApiErrorType.NOT_FOUND_ERROR,
        429: ApiErrorType.RATE_LIMIT_ERROR,
        500: ApiErrorType.INTERNAL_SERVER_ERROR,
        502: ApiErrorType.EXTERNAL_SERVICE_ERROR,
        503: ApiErrorType.EXTERNAL_SERVICE_ERROR,
        504: ApiErrorType.TIMEOUT_ERROR,
    }

    error_type = error_type_mapping.get(exc.status_code, ApiErrorType.BUSINESS_LOGIC_ERROR)
    error_handler.log_error(error_type, exc, request)

    return error_handler.create_error_response(
        error_type=error_type,
        message=exc.detail,
        status_code=exc.status_code,
        details=getattr(exc, "details", None),
        request=request,
    )


async def internal_server_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle internal server errors."""
    error_handler.log_error(ApiErrorType.INTERNAL_SERVER_ERROR, exc, request)

    # In production, don't expose internal error details
    import os

    is_development = os.getenv("ENVIRONMENT", "development") == "development"

    details = None
    if is_development:
        details = {
            "exception_type": exc.__class__.__name__,
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)[
                -5:
            ],  # Last 5 lines
        }

    return error_handler.create_error_response(
        error_type=ApiErrorType.INTERNAL_SERVER_ERROR,
        message="An internal server error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details=details,
        request=request,
    )


# Custom exception classes for business logic
class ChimeraAPIError(HTTPException):
    """Base exception for Chimera API errors."""

    def __init__(
        self,
        error_type: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.error_type = error_type
        self.details = details


class ProviderError(ChimeraAPIError):
    """Exception for AI provider-related errors."""

    def __init__(self, provider: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            error_type=ApiErrorType.EXTERNAL_SERVICE_ERROR,
            message=f"Provider '{provider}' error: {message}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )
        self.provider = provider


class AuthenticationError(ChimeraAPIError):
    """Exception for authentication errors."""

    def __init__(
        self, message: str = "Authentication required", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            error_type=ApiErrorType.AUTHENTICATION_ERROR,
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class AuthorizationError(ChimeraAPIError):
    """Exception for authorization errors."""

    def __init__(
        self, message: str = "Insufficient permissions", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            error_type=ApiErrorType.AUTHORIZATION_ERROR,
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class ValidationError(ChimeraAPIError):
    """Exception for business validation errors."""

    def __init__(
        self, message: str, field: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        validation_details = details or {}
        if field:
            validation_details["field"] = field

        super().__init__(
            error_type=ApiErrorType.VALIDATION_ERROR,
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=validation_details,
        )


class RateLimitError(ChimeraAPIError):
    """Exception for rate limiting errors."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: int | None = None
    ) -> None:
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after

        super().__init__(
            error_type=ApiErrorType.RATE_LIMIT_ERROR,
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
        )


# Custom exception handler for Chimera API errors
async def chimera_api_error_handler(request: Request, exc: ChimeraAPIError) -> JSONResponse:
    """Handle custom Chimera API errors."""
    error_handler.log_error(exc.error_type, exc, request)

    return error_handler.create_error_response(
        error_type=exc.error_type,
        message=exc.detail,
        status_code=exc.status_code,
        details=exc.details,
        request=request,
    )


# Error monitoring and health metrics
def get_error_statistics() -> dict[str, Any]:
    """Get error statistics for monitoring."""
    total_errors = sum(error_handler.error_counts.values())

    return {
        "total_errors": total_errors,
        "error_counts_by_type": error_handler.error_counts.copy(),
        "error_types": list(error_handler.error_counts.keys()),
        "most_common_error": (
            max(error_handler.error_counts.items(), key=lambda x: x[1])[0]
            if error_handler.error_counts
            else None
        ),
    }


# Reset error statistics (for testing or periodic cleanup)
def reset_error_statistics() -> None:
    """Reset error statistics."""
    error_handler.error_counts.clear()
