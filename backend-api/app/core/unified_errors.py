"""
Unified Error Handling System

This module provides a consistent error hierarchy and handling strategy across
the entire backend. All errors inherit from ChimeraError and provide standardized
error responses with proper status codes and error details.
"""

import logging
from typing import Any

from fastapi import status

logger = logging.getLogger(__name__)


class ChimeraError(Exception):
    """Base error for all Chimera backend errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "CHIMERA_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for API response."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "status_code": self.status_code,
                "details": self.details,
            }
        }


# Service-specific errors
class ServiceError(ChimeraError):
    """Base error for service layer."""

    def __init__(self, message: str, service_name: str, **kwargs):
        # Extract status_code from kwargs if present, otherwise use default
        error_status_code = kwargs.pop("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        super().__init__(
            message,
            status_code=error_status_code,
            error_code=f"SERVICE_{service_name.upper()}_ERROR",
            **kwargs,
        )


class LLMProviderError(ServiceError):
    """LLM provider errors."""

    def __init__(self, message: str, provider: str, **kwargs):
        # Extract status_code from kwargs, default to 502 for provider errors
        error_status_code = kwargs.pop("status_code", status.HTTP_502_BAD_GATEWAY)
        # Merge provider into details
        existing_details = kwargs.pop("details", {})
        merged_details = {"provider": provider, **existing_details}
        super().__init__(
            message,
            service_name="llm_provider",
            status_code=error_status_code,
            details=merged_details,
            **kwargs,
        )


class ProviderNotAvailableError(LLMProviderError):
    """Provider not available error."""

    def __init__(
        self, provider: str, message: str | None = None, details: dict[str, Any] | None = None
    ):
        final_message = message or f"Provider {provider} is not available"
        super().__init__(
            final_message,
            provider=provider,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details or {},
        )


class TransformationError(ServiceError):
    """Transformation errors."""

    def __init__(self, message: str, **kwargs):
        # Extract status_code from kwargs, default to 400 for transformation errors
        error_status_code = kwargs.pop("status_code", status.HTTP_400_BAD_REQUEST)
        super().__init__(
            message, service_name="transformation", status_code=error_status_code, **kwargs
        )


class InvalidPotencyError(TransformationError):
    """Invalid potency level error."""

    pass


class InvalidTechniqueError(TransformationError):
    """Invalid technique suite error."""

    pass


# Generation Errors
class GenerationError(ServiceError):
    """Generation errors."""

    def __init__(self, message: str, **kwargs):
        # Extract status_code from kwargs, default to 500 for generation errors
        error_status_code = kwargs.pop("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        super().__init__(
            message, service_name="generation", status_code=error_status_code, **kwargs
        )


# Data layer errors
class DataError(ChimeraError):
    """Base error for data layer."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DATA_ERROR", **kwargs)


class RepositoryError(DataError):
    """Repository operation errors."""

    pass


class DatabaseConnectionError(DataError):
    """Database connection errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DATABASE_CONNECTION_ERROR",
            **kwargs,
        )


# Validation errors
class ValidationError(ChimeraError):
    """Input validation errors."""

    def __init__(self, message: str, field: str | None = None, **kwargs):
        details = {"field": field} if field else {}
        super().__init__(
            message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details,
            **kwargs,
        )


# Authentication and authorization errors
class AuthenticationError(ChimeraError):
    """Authentication errors."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
            **kwargs,
        )


class AuthorizationError(ChimeraError):
    """Authorization errors."""

    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(
            message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
            **kwargs,
        )


# Configuration errors
class ConfigurationError(ChimeraError):
    """Configuration errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CONFIGURATION_ERROR",
            **kwargs,
        )


# Rate limiting errors
class RateLimitError(ChimeraError):
    """Rate limit exceeded errors."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(
            message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_ERROR",
            **kwargs,
        )


# Circuit breaker errors
class CircuitBreakerError(ChimeraError):
    """Circuit breaker open errors."""

    def __init__(self, service: str, **kwargs):
        super().__init__(
            f"Circuit breaker open for service: {service}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="CIRCUIT_BREAKER_OPEN",
            details={"service": service},
            **kwargs,
        )


# =============================================================================
# Aegis Engine Errors
# =============================================================================


class AegisError(ServiceError):
    """Base error for Aegis engine operations."""

    def __init__(self, message: str, **kwargs):
        # Extract status_code from kwargs, default to 500
        error_status_code = kwargs.pop("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        super().__init__(message, service_name="aegis", status_code=error_status_code, **kwargs)


class AegisSynthesisError(AegisError):
    """Error during persona or scenario synthesis."""

    def __init__(self, message: str, **kwargs):
        # Default to 422 Unprocessable Entity for synthesis errors (often bad input/constraints)
        error_status_code = kwargs.pop("status_code", status.HTTP_422_UNPROCESSABLE_ENTITY)
        super().__init__(message, status_code=error_status_code, **kwargs)


class AegisTransformationError(AegisError):
    """Error during payload transformation."""

    def __init__(self, message: str, **kwargs):
        # Default to 400 Bad Request for transformation errors
        error_status_code = kwargs.pop("status_code", status.HTTP_400_BAD_REQUEST)
        super().__init__(message, status_code=error_status_code, **kwargs)


# =============================================================================
# Gemini-Specific Errors (Google GenAI SDK Enhancement)
# =============================================================================


class GeminiAPIError(LLMProviderError):
    """
    Gemini-specific API errors with detailed error code mapping.

    Maps Google GenAI SDK errors.APIError to appropriate HTTP status codes
    and provides structured error information.
    """

    def __init__(self, message: str, api_code: int, api_message: str | None = None, **kwargs):
        # Map Gemini API error codes to HTTP status codes
        status_code = self._map_api_code(api_code)

        details = kwargs.pop("details", {})
        details.update(
            {
                "api_code": api_code,
                "api_message": api_message or message,
            }
        )

        super().__init__(
            message, provider="gemini", status_code=status_code, details=details, **kwargs
        )
        self.api_code = api_code
        self.api_message = api_message

    @staticmethod
    def _map_api_code(api_code: int) -> int:
        """Map Gemini API error codes to HTTP status codes."""
        mapping = {
            400: status.HTTP_400_BAD_REQUEST,
            401: status.HTTP_401_UNAUTHORIZED,
            403: status.HTTP_403_FORBIDDEN,
            404: status.HTTP_404_NOT_FOUND,
            429: status.HTTP_429_TOO_MANY_REQUESTS,
            500: status.HTTP_502_BAD_GATEWAY,
            503: status.HTTP_503_SERVICE_UNAVAILABLE,
        }
        return mapping.get(api_code, status.HTTP_502_BAD_GATEWAY)


class GeminiRateLimitError(GeminiAPIError):
    """Gemini API rate limit exceeded."""

    def __init__(self, retry_after: int | None = None, **kwargs):
        details = kwargs.pop("details", {})
        if retry_after:
            details["retry_after_seconds"] = retry_after

        super().__init__(
            "Gemini API rate limit exceeded",
            api_code=429,
            api_message="Rate limit exceeded",
            details=details,
            **kwargs,
        )
        self.retry_after = retry_after


class GeminiStreamingError(GeminiAPIError):
    """Error during Gemini streaming generation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(f"Streaming error: {message}", api_code=500, api_message=message, **kwargs)


class GeminiTokenCountError(GeminiAPIError):
    """Error during Gemini token counting."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            f"Token counting error: {message}", api_code=500, api_message=message, **kwargs
        )
