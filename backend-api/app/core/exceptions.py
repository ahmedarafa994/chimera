"""Custom exceptions for the backend API.
Provides structured error handling with consistent error responses.
"""

from http import HTTPStatus
from typing import Any


class APIException(Exception):
    """Base exception class for all API exceptions."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"
    message: str = "An internal error occurred"
    details: dict[str, Any] | None = None

    def __init__(
        self,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        """Initialize the API exception."""
        if message:
            self.message = message
        if details:
            self.details = details
        if status_code:
            self.status_code = status_code
        if error_code:
            self.error_code = error_code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        response = {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }
        if self.details:
            response["details"] = self.details
        return response


class ValidationError(APIException):
    """Raised when request validation fails."""

    status_code = HTTPStatus.BAD_REQUEST
    error_code = "VALIDATION_ERROR"
    message = "Request validation failed"


class AuthenticationError(APIException):
    """Raised when authentication fails."""

    status_code = HTTPStatus.UNAUTHORIZED
    error_code = "AUTHENTICATION_ERROR"
    message = "Authentication failed"


class AuthorizationError(APIException):
    """Raised when authorization fails."""

    status_code = HTTPStatus.FORBIDDEN
    error_code = "AUTHORIZATION_ERROR"
    message = "You don't have permission to access this resource"


class NotFoundError(APIException):
    """Raised when a requested resource is not found."""

    status_code = HTTPStatus.NOT_FOUND
    error_code = "NOT_FOUND"
    message = "The requested resource was not found"


class ConflictError(APIException):
    """Raised when there's a conflict with the current state."""

    status_code = HTTPStatus.CONFLICT
    error_code = "CONFLICT_ERROR"
    message = "The request conflicts with the current state"


class RateLimitError(APIException):
    """Raised when rate limit is exceeded."""

    status_code = HTTPStatus.TOO_MANY_REQUESTS
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Rate limit exceeded. Please try again later"


class ServiceUnavailableError(APIException):
    """Raised when a service is temporarily unavailable."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    error_code = "SERVICE_UNAVAILABLE"
    message = "The service is temporarily unavailable"


class LLMProviderError(APIException):
    """Base exception for LLM provider errors."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "LLM_PROVIDER_ERROR"
    message = "LLM provider error occurred"


class LLMConnectionError(LLMProviderError):
    """Raised when connection to LLM provider fails."""

    error_code = "LLM_CONNECTION_ERROR"
    message = "Failed to connect to LLM provider"


class LLMTimeoutError(LLMProviderError):
    """Raised when LLM request times out."""

    status_code = HTTPStatus.REQUEST_TIMEOUT
    error_code = "LLM_TIMEOUT_ERROR"
    message = "LLM request timed out"


class LLMQuotaExceededError(LLMProviderError):
    """Raised when LLM quota is exceeded."""

    status_code = HTTPStatus.TOO_MANY_REQUESTS
    error_code = "LLM_QUOTA_EXCEEDED"
    message = "LLM API quota exceeded"

    def __init__(
        self, message: str | None = None, retry_after: int | None = None, **kwargs
    ) -> None:
        """Initialize with optional retry_after seconds."""
        super().__init__(message, **kwargs)
        self.retry_after = retry_after  # Seconds to wait before retry


class LLMInvalidResponseError(LLMProviderError):
    """Raised when LLM returns invalid response."""

    error_code = "LLM_INVALID_RESPONSE"
    message = "Invalid response from LLM provider"


class LLMContentBlockedError(LLMProviderError):
    """Raised when content is blocked by safety filters."""

    error_code = "LLM_CONTENT_BLOCKED"
    message = "Content blocked by safety filters"

    def __init__(
        self, message: str | None = None, block_reason: str | None = None, **kwargs
    ) -> None:
        """Initialize with optional block reason."""
        super().__init__(message, **kwargs)
        self.block_reason = block_reason


class TransformationError(APIException):
    """Raised when prompt transformation fails."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "TRANSFORMATION_ERROR"
    message = "Prompt transformation failed"


class InvalidPotencyError(ValidationError):
    """Raised when potency level is invalid."""

    error_code = "INVALID_POTENCY"
    message = "Invalid potency level. Must be between 1 and 10"


class InvalidTechniqueError(ValidationError):
    """Raised when technique suite is invalid."""

    error_code = "INVALID_TECHNIQUE"
    message = "Invalid technique suite"


class MissingFieldError(ValidationError):
    """Raised when required field is missing."""

    error_code = "MISSING_FIELD"
    message = "Required field is missing"

    def __init__(self, field_name: str) -> None:
        """Initialize with field name."""
        super().__init__(
            message=f"Required field '{field_name}' is missing",
            details={"missing_field": field_name},
        )


class InvalidFieldError(ValidationError):
    """Raised when field value is invalid."""

    error_code = "INVALID_FIELD"
    message = "Field value is invalid"

    def __init__(self, field_name: str, reason: str) -> None:
        """Initialize with field name and reason."""
        super().__init__(
            message=f"Invalid value for field '{field_name}': {reason}",
            details={"field": field_name, "reason": reason},
        )


class PayloadTooLargeError(APIException):
    """Raised when request payload is too large."""

    status_code = 413
    error_code = "PAYLOAD_TOO_LARGE"
    message = "Request payload is too large"


class ProviderNotConfiguredError(LLMProviderError):
    """Raised when LLM provider is not configured."""

    error_code = "PROVIDER_NOT_CONFIGURED"
    message = "LLM provider is not configured"

    def __init__(self, provider: str) -> None:
        """Initialize with provider name."""
        super().__init__(
            message=f"Provider '{provider}' is not configured. Please set API key.",
            details={"provider": provider},
        )


class ProviderNotAvailableError(LLMProviderError):
    """Raised when LLM provider is not available."""

    error_code = "PROVIDER_NOT_AVAILABLE"
    message = "LLM provider is not available"

    def __init__(self, provider: str, reason: str | None = None) -> None:
        """Initialize with provider name and optional reason."""
        details = {"provider": provider}
        message = f"Provider '{provider}' is not available"
        if reason:
            details["reason"] = reason
            message = f"{message}: {reason}"
        super().__init__(
            message=message,
            details=details,
        )


class CacheError(APIException):
    """Raised when cache operation fails."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "CACHE_ERROR"
    message = "Cache operation failed"


class ConfigurationError(APIException):
    """Raised when configuration is invalid."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "CONFIGURATION_ERROR"
    message = "Invalid configuration"


def handle_exception(exception: Exception) -> dict[str, Any]:
    """Convert any exception to a consistent error response.

    Args:
        exception: The exception to handle

    Returns:
        Dictionary with error details for JSON response

    """
    if isinstance(exception, APIException):
        return exception.to_dict()

    # Handle standard Python exceptions
    error_map = {
        ValueError: (HTTPStatus.BAD_REQUEST, "VALUE_ERROR"),
        KeyError: (HTTPStatus.BAD_REQUEST, "KEY_ERROR"),
        TypeError: (HTTPStatus.BAD_REQUEST, "TYPE_ERROR"),
        TimeoutError: (HTTPStatus.REQUEST_TIMEOUT, "TIMEOUT_ERROR"),
        ConnectionError: (HTTPStatus.SERVICE_UNAVAILABLE, "CONNECTION_ERROR"),
        MemoryError: (HTTPStatus.INTERNAL_SERVER_ERROR, "MEMORY_ERROR"),
        OSError: (HTTPStatus.INTERNAL_SERVER_ERROR, "OS_ERROR"),
    }

    for exc_type, (status_code, error_code) in error_map.items():
        if isinstance(exception, exc_type):
            return {
                "error": error_code,
                "message": str(exception),
                "status_code": status_code,
            }

    # Default error response
    return {
        "error": "INTERNAL_ERROR",
        "message": "An unexpected error occurred",
        "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
        "details": {"exception": type(exception).__name__},
    }
