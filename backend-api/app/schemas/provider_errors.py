"""Provider Error Schemas for Unified Provider System.

This module defines normalized error types and schemas for handling
provider-specific errors in a unified way across all AI providers.

Supports error normalization for:
- OpenAI, Anthropic, Google/Gemini, DeepSeek, Azure, and other providers
- Rate limiting, authentication, quota, and content policy errors
- Retry strategies and fallback suggestions
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProviderErrorCode(str, Enum):
    """Normalized error codes for provider-specific errors.

    These codes provide a unified way to categorize errors from
    different providers, enabling consistent error handling.
    """

    # Rate and quota errors
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"

    # Authentication and authorization
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    API_KEY_INVALID = "api_key_invalid"
    API_KEY_EXPIRED = "api_key_expired"

    # Model errors
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_OVERLOADED = "model_overloaded"
    MODEL_DEPRECATED = "model_deprecated"
    MODEL_UNAVAILABLE = "model_unavailable"

    # Content and request errors
    CONTENT_POLICY_VIOLATION = "content_policy_violation"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INVALID_REQUEST = "invalid_request"
    INVALID_PARAMETERS = "invalid_parameters"
    MALFORMED_REQUEST = "malformed_request"

    # Provider availability
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_MAINTENANCE = "provider_maintenance"
    SERVICE_UNAVAILABLE = "service_unavailable"

    # Network and timeout
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    NETWORK_ERROR = "network_error"

    # Streaming errors
    STREAM_INTERRUPTED = "stream_interrupted"
    STREAM_ERROR = "stream_error"

    # Internal errors
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Severity level of the error."""

    LOW = "low"  # Informational, may self-resolve
    MEDIUM = "medium"  # Requires attention but not critical
    HIGH = "high"  # Significant impact, requires action
    CRITICAL = "critical"  # Service impacting, immediate action needed


class RetryStrategy(BaseModel):
    """Strategy for retrying failed requests.

    Provides guidance on whether and how to retry a failed request,
    with configurable backoff and jitter settings.
    """

    should_retry: bool = Field(description="Whether the request should be retried")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retry attempts")
    base_delay_seconds: float = Field(
        default=1.0,
        ge=0,
        description="Initial delay before first retry in seconds",
    )
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Multiplier for exponential backoff",
    )
    max_delay_seconds: float = Field(
        default=60.0,
        ge=0,
        description="Maximum delay between retries in seconds",
    )
    jitter: bool = Field(default=True, description="Whether to add random jitter to delays")
    jitter_factor: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Factor for jitter (0.1 = ±10%)",
    )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate the delay for a specific retry attempt.

        Args:
            attempt: The retry attempt number (0-indexed)

        Returns:
            Delay in seconds before the next retry

        """
        import random

        delay = min(
            self.base_delay_seconds * (self.backoff_multiplier**attempt),
            self.max_delay_seconds,
        )

        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


class FallbackSuggestion(BaseModel):
    """Suggestion for fallback action when a provider/model fails.

    Provides alternative providers or models that may be used
    when the primary selection is unavailable.
    """

    alternative_provider: str | None = Field(
        default=None,
        description="Alternative provider to use",
    )
    alternative_model: str | None = Field(default=None, description="Alternative model to use")
    reason: str = Field(description="Reason for the fallback suggestion")
    confidence: float = Field(
        default=0.8,
        ge=0,
        le=1.0,
        description="Confidence that the fallback will work",
    )
    capability_match: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="How well the fallback matches required capabilities",
    )
    estimated_latency_change: float | None = Field(
        default=None,
        description="Estimated latency change in ms (positive = slower)",
    )
    estimated_cost_change: float | None = Field(
        default=None,
        description="Estimated cost change as a multiplier (1.0 = same cost)",
    )


class NormalizedProviderError(BaseModel):
    """Normalized error from any AI provider.

    This class provides a unified error format that normalizes
    provider-specific error codes and messages into a consistent
    structure for handling across the application.
    """

    # Error identification
    code: ProviderErrorCode = Field(description="Normalized error code")
    message: str = Field(description="Human-readable error message")

    # Provider context
    provider: str = Field(description="Provider that returned the error")
    model: str | None = Field(default=None, description="Model that was being used (if applicable)")

    # Original error details
    original_error: str = Field(description="Original error message from provider")
    original_code: str | None = Field(default=None, description="Original error code from provider")
    original_status: int | None = Field(default=None, description="Original HTTP status code")

    # Retry guidance
    retry_after_seconds: int | None = Field(
        default=None,
        description="Seconds to wait before retrying (from provider)",
    )
    is_retryable: bool = Field(default=False, description="Whether the error is retryable")
    retry_strategy: RetryStrategy | None = Field(
        default=None,
        description="Recommended retry strategy",
    )

    # User guidance
    suggested_action: str = Field(description="Suggested action for the user to take")
    user_message: str = Field(description="User-friendly error message")

    # Fallback options
    fallback_suggestion: FallbackSuggestion | None = Field(
        default=None,
        description="Suggested fallback if available",
    )

    # Metadata
    severity: ErrorSeverity = Field(
        default=ErrorSeverity.MEDIUM,
        description="Severity level of the error",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred",
    )
    request_id: str | None = Field(default=None, description="Request ID for tracing")
    trace_id: str | None = Field(default=None, description="Distributed trace ID")

    # Additional context
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional error metadata")

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for structured logging."""
        return {
            "error_code": self.code.value,
            "error_message": self.message,
            "provider": self.provider,
            "model": self.model,
            "original_error": self.original_error,
            "original_code": self.original_code,
            "original_status": self.original_status,
            "is_retryable": self.is_retryable,
            "retry_after_seconds": self.retry_after_seconds,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "trace_id": self.trace_id,
        }

    def to_user_response(self) -> dict[str, Any]:
        """Convert to a user-facing error response."""
        response = {
            "error": {
                "code": self.code.value,
                "message": self.user_message,
                "provider": self.provider,
                "is_retryable": self.is_retryable,
                "suggested_action": self.suggested_action,
            },
        }

        if self.retry_after_seconds:
            response["error"]["retry_after_seconds"] = self.retry_after_seconds

        if self.fallback_suggestion:
            response["error"]["fallback"] = {
                "provider": self.fallback_suggestion.alternative_provider,
                "model": self.fallback_suggestion.alternative_model,
                "reason": self.fallback_suggestion.reason,
            }

        if self.request_id:
            response["error"]["request_id"] = self.request_id

        return response


class ProviderErrorSummary(BaseModel):
    """Summary of errors for a provider over a time period."""

    provider: str
    time_range_start: datetime
    time_range_end: datetime
    total_errors: int
    errors_by_code: dict[str, int] = Field(default_factory=dict)
    errors_by_model: dict[str, int] = Field(default_factory=dict)
    retry_success_rate: float | None = None
    average_retry_count: float | None = None
    top_error_messages: list[dict[str, Any]] = Field(default_factory=list)


class ErrorMetrics(BaseModel):
    """Metrics about errors across all providers."""

    time_range_hours: int
    total_errors: int
    errors_by_provider: dict[str, int] = Field(default_factory=dict)
    errors_by_code: dict[str, int] = Field(default_factory=dict)
    errors_by_severity: dict[str, int] = Field(default_factory=dict)
    retry_stats: dict[str, Any] = Field(default_factory=dict)
    fallback_stats: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Provider Exception Classes
# =============================================================================


class ProviderException(Exception):
    """Base exception for provider-related errors.

    This exception wraps provider-specific errors and carries
    the normalized error information for consistent handling.
    """

    def __init__(
        self,
        normalized_error: NormalizedProviderError,
        original_exception: Exception | None = None,
    ) -> None:
        self.normalized_error = normalized_error
        self.original_exception = original_exception
        super().__init__(normalized_error.message)

    @property
    def code(self) -> ProviderErrorCode:
        return self.normalized_error.code

    @property
    def provider(self) -> str:
        return self.normalized_error.provider

    @property
    def model(self) -> str | None:
        return self.normalized_error.model

    @property
    def is_retryable(self) -> bool:
        return self.normalized_error.is_retryable

    @property
    def retry_strategy(self) -> RetryStrategy | None:
        return self.normalized_error.retry_strategy


class SelectionValidationException(Exception):
    """Exception raised when provider/model selection validation fails."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        validation_errors: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.validation_errors = validation_errors or []
        super().__init__(message)


class ProviderRateLimitException(ProviderException):
    """Exception raised when a provider rate limit is hit."""


class ProviderQuotaException(ProviderException):
    """Exception raised when a provider quota is exceeded."""


class ProviderAuthenticationException(ProviderException):
    """Exception raised for authentication failures."""


class ProviderModelException(ProviderException):
    """Exception raised for model-related errors."""


class ProviderContentPolicyException(ProviderException):
    """Exception raised for content policy violations."""


class ProviderTimeoutException(ProviderException):
    """Exception raised for timeout errors."""


class ProviderUnavailableException(ProviderException):
    """Exception raised when a provider is unavailable."""


# =============================================================================
# Error Code to HTTP Status Mapping
# =============================================================================


ERROR_CODE_TO_HTTP_STATUS: dict[ProviderErrorCode, int] = {
    # Rate and quota errors -> 429
    ProviderErrorCode.RATE_LIMITED: 429,
    ProviderErrorCode.QUOTA_EXCEEDED: 429,
    # Authentication errors -> 401
    ProviderErrorCode.AUTHENTICATION_FAILED: 401,
    ProviderErrorCode.API_KEY_INVALID: 401,
    ProviderErrorCode.API_KEY_EXPIRED: 401,
    # Authorization errors -> 403
    ProviderErrorCode.AUTHORIZATION_FAILED: 403,
    # Model not found -> 404
    ProviderErrorCode.MODEL_NOT_FOUND: 404,
    # Client errors -> 400
    ProviderErrorCode.CONTENT_POLICY_VIOLATION: 400,
    ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED: 400,
    ProviderErrorCode.INVALID_REQUEST: 400,
    ProviderErrorCode.INVALID_PARAMETERS: 400,
    ProviderErrorCode.MALFORMED_REQUEST: 400,
    # Service unavailable -> 503
    ProviderErrorCode.PROVIDER_UNAVAILABLE: 503,
    ProviderErrorCode.PROVIDER_MAINTENANCE: 503,
    ProviderErrorCode.SERVICE_UNAVAILABLE: 503,
    ProviderErrorCode.MODEL_OVERLOADED: 503,
    ProviderErrorCode.MODEL_UNAVAILABLE: 503,
    # Timeout -> 504
    ProviderErrorCode.TIMEOUT: 504,
    ProviderErrorCode.CONNECTION_ERROR: 502,
    ProviderErrorCode.NETWORK_ERROR: 502,
    # Streaming errors -> 500
    ProviderErrorCode.STREAM_INTERRUPTED: 500,
    ProviderErrorCode.STREAM_ERROR: 500,
    # Internal/Unknown -> 500
    ProviderErrorCode.MODEL_DEPRECATED: 400,
    ProviderErrorCode.INTERNAL_ERROR: 500,
    ProviderErrorCode.UNKNOWN: 500,
}


def get_http_status_for_error(error_code: ProviderErrorCode) -> int:
    """Get the appropriate HTTP status code for an error code."""
    return ERROR_CODE_TO_HTTP_STATUS.get(error_code, 500)


# =============================================================================
# Retryable Error Codes
# =============================================================================


RETRYABLE_ERROR_CODES: set[ProviderErrorCode] = {
    ProviderErrorCode.RATE_LIMITED,
    ProviderErrorCode.MODEL_OVERLOADED,
    ProviderErrorCode.PROVIDER_UNAVAILABLE,
    ProviderErrorCode.SERVICE_UNAVAILABLE,
    ProviderErrorCode.TIMEOUT,
    ProviderErrorCode.CONNECTION_ERROR,
    ProviderErrorCode.NETWORK_ERROR,
    ProviderErrorCode.STREAM_INTERRUPTED,
    ProviderErrorCode.INTERNAL_ERROR,  # May be transient
}


def is_retryable_error(error_code: ProviderErrorCode) -> bool:
    """Check if an error code indicates a retryable error."""
    return error_code in RETRYABLE_ERROR_CODES


# =============================================================================
# Default Retry Strategies
# =============================================================================


DEFAULT_RETRY_STRATEGIES: dict[ProviderErrorCode, RetryStrategy] = {
    ProviderErrorCode.RATE_LIMITED: RetryStrategy(
        should_retry=True,
        max_retries=5,
        base_delay_seconds=2.0,
        backoff_multiplier=2.0,
        max_delay_seconds=60.0,
        jitter=True,
    ),
    ProviderErrorCode.MODEL_OVERLOADED: RetryStrategy(
        should_retry=True,
        max_retries=3,
        base_delay_seconds=5.0,
        backoff_multiplier=2.0,
        max_delay_seconds=30.0,
        jitter=True,
    ),
    ProviderErrorCode.TIMEOUT: RetryStrategy(
        should_retry=True,
        max_retries=2,
        base_delay_seconds=1.0,
        backoff_multiplier=1.5,
        max_delay_seconds=10.0,
        jitter=False,
    ),
    ProviderErrorCode.CONNECTION_ERROR: RetryStrategy(
        should_retry=True,
        max_retries=3,
        base_delay_seconds=1.0,
        backoff_multiplier=2.0,
        max_delay_seconds=15.0,
        jitter=True,
    ),
    ProviderErrorCode.PROVIDER_UNAVAILABLE: RetryStrategy(
        should_retry=True,
        max_retries=3,
        base_delay_seconds=10.0,
        backoff_multiplier=2.0,
        max_delay_seconds=60.0,
        jitter=True,
    ),
    ProviderErrorCode.STREAM_INTERRUPTED: RetryStrategy(
        should_retry=True,
        max_retries=2,
        base_delay_seconds=0.5,
        backoff_multiplier=2.0,
        max_delay_seconds=5.0,
        jitter=False,
    ),
}


def get_default_retry_strategy(error_code: ProviderErrorCode) -> RetryStrategy | None:
    """Get the default retry strategy for an error code."""
    return DEFAULT_RETRY_STRATEGIES.get(error_code)


__all__ = [
    "DEFAULT_RETRY_STRATEGIES",
    # Mappings and utilities
    "ERROR_CODE_TO_HTTP_STATUS",
    "RETRYABLE_ERROR_CODES",
    "ErrorMetrics",
    "ErrorSeverity",
    "FallbackSuggestion",
    "NormalizedProviderError",
    "ProviderAuthenticationException",
    "ProviderContentPolicyException",
    # Enums
    "ProviderErrorCode",
    "ProviderErrorSummary",
    # Exceptions
    "ProviderException",
    "ProviderModelException",
    "ProviderQuotaException",
    "ProviderRateLimitException",
    "ProviderTimeoutException",
    "ProviderUnavailableException",
    # Models
    "RetryStrategy",
    "SelectionValidationException",
    "get_default_retry_strategy",
    "get_http_status_for_error",
    "is_retryable_error",
]
