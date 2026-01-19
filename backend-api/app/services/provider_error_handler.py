"""
Provider Error Handler for Unified Provider System

This module provides centralized error handling for all AI provider errors,
normalizing provider-specific error formats into a unified structure.

Features:
- Error normalization across 11+ providers
- Retry strategy determination
- Fallback suggestions
- Error metrics collection
"""

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

from app.schemas.provider_errors import (
    ErrorSeverity,
    FallbackSuggestion,
    NormalizedProviderError,
    ProviderErrorCode,
    ProviderException,
    RetryStrategy,
    get_default_retry_strategy,
    is_retryable_error,
)

logger = logging.getLogger(__name__)


class ProviderErrorHandler:
    """
    Normalizes errors from different AI providers into a unified format.

    Handles provider-specific:
    - Error codes and messages
    - Rate limiting responses
    - Authentication failures
    - Model availability issues
    - Quota exceeded errors
    - Content policy violations
    """

    def __init__(self):
        """Initialize the error handler with error mappers."""
        self._error_mappers: dict[str, BaseErrorMapper] = {}
        self._error_history: list[NormalizedProviderError] = []
        self._max_history_size = 10000
        self._fallback_mappings: dict[str, list[tuple[str, str]]] = {}

        # Initialize default fallback mappings
        self._init_fallback_mappings()

    def _init_fallback_mappings(self):
        """Initialize default fallback provider/model mappings."""
        self._fallback_mappings = {
            "openai": [
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("google", "gemini-1.5-pro"),
                ("deepseek", "deepseek-chat"),
            ],
            "anthropic": [
                ("openai", "gpt-4-turbo"),
                ("google", "gemini-1.5-pro"),
                ("deepseek", "deepseek-chat"),
            ],
            "google": [
                ("openai", "gpt-4-turbo"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("deepseek", "deepseek-chat"),
            ],
            "deepseek": [
                ("openai", "gpt-4-turbo"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("google", "gemini-1.5-pro"),
            ],
            "azure": [
                ("openai", "gpt-4-turbo"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
            ],
        }

    def register_mapper(self, provider: str, mapper: "BaseErrorMapper") -> None:
        """
        Register an error mapper for a provider.

        Args:
            provider: Provider identifier (lowercase)
            mapper: Error mapper instance
        """
        self._error_mappers[provider.lower()] = mapper
        logger.debug(f"Registered error mapper for provider: {provider}")

    def get_mapper(self, provider: str) -> Optional["BaseErrorMapper"]:
        """
        Get the error mapper for a provider.

        Args:
            provider: Provider identifier

        Returns:
            Error mapper if registered, None otherwise
        """
        return self._error_mappers.get(provider.lower())

    def normalize_error(
        self,
        provider: str,
        error: Exception,
        model: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> NormalizedProviderError:
        """
        Normalize a provider error into a unified format.

        Args:
            provider: Provider that returned the error
            error: The original exception
            model: Model that was being used
            request_id: Request ID for tracing
            trace_id: Distributed trace ID

        Returns:
            NormalizedProviderError with unified error information
        """
        provider = provider.lower()
        mapper = self._error_mappers.get(provider)

        if mapper:
            try:
                normalized = mapper.map_error(error, model, request_id, trace_id)
                self._record_error(normalized)
                return normalized
            except Exception as e:
                logger.warning(f"Error mapper failed for {provider}: {e}", exc_info=True)

        # Fall back to generic error mapping
        normalized = self._generic_normalize(provider, error, model, request_id, trace_id)
        self._record_error(normalized)
        return normalized

    def _generic_normalize(
        self,
        provider: str,
        error: Exception,
        model: str | None,
        request_id: str | None,
        trace_id: str | None,
    ) -> NormalizedProviderError:
        """
        Generic error normalization for unknown providers or mapper failures.
        """
        error_str = str(error)
        error_lower = error_str.lower()

        # Detect error type from message patterns
        code = self._detect_error_code(error, error_lower)
        severity = self._determine_severity(code)

        # Extract retry-after if present
        retry_after = self._extract_retry_after(error)

        # Build normalized error
        normalized = NormalizedProviderError(
            code=code,
            message=self._get_generic_message(code),
            provider=provider,
            model=model,
            original_error=error_str[:1000],
            original_code=getattr(error, "code", None),
            original_status=getattr(error, "status_code", None) or getattr(error, "status", None),
            retry_after_seconds=retry_after,
            is_retryable=is_retryable_error(code),
            retry_strategy=get_default_retry_strategy(code),
            suggested_action=self._get_suggested_action(code),
            user_message=self._get_user_message(code, provider),
            severity=severity,
            request_id=request_id,
            trace_id=trace_id,
        )

        return normalized

    def _detect_error_code(self, error: Exception, error_lower: str) -> ProviderErrorCode:
        """Detect the error code from exception and message."""
        # Check for HTTP status code
        status = getattr(error, "status_code", None) or getattr(error, "status", None)

        if status == 429:
            if "quota" in error_lower:
                return ProviderErrorCode.QUOTA_EXCEEDED
            return ProviderErrorCode.RATE_LIMITED

        if status == 401:
            return ProviderErrorCode.AUTHENTICATION_FAILED

        if status == 403:
            return ProviderErrorCode.AUTHORIZATION_FAILED

        if status == 404:
            if "model" in error_lower:
                return ProviderErrorCode.MODEL_NOT_FOUND
            return ProviderErrorCode.INVALID_REQUEST

        if status == 503:
            if "overload" in error_lower:
                return ProviderErrorCode.MODEL_OVERLOADED
            return ProviderErrorCode.PROVIDER_UNAVAILABLE

        if status == 504:
            return ProviderErrorCode.TIMEOUT

        # Pattern matching on error message
        patterns = [
            (r"rate.?limit", ProviderErrorCode.RATE_LIMITED),
            (r"quota", ProviderErrorCode.QUOTA_EXCEEDED),
            (r"auth", ProviderErrorCode.AUTHENTICATION_FAILED),
            (r"api.?key", ProviderErrorCode.API_KEY_INVALID),
            (r"model.*not.*found", ProviderErrorCode.MODEL_NOT_FOUND),
            (r"model.*unavailable", ProviderErrorCode.MODEL_UNAVAILABLE),
            (r"overload", ProviderErrorCode.MODEL_OVERLOADED),
            (r"content.?policy", ProviderErrorCode.CONTENT_POLICY_VIOLATION),
            (r"context.?(length|window)", ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED),
            (r"token.?limit", ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED),
            (r"timeout", ProviderErrorCode.TIMEOUT),
            (r"connect", ProviderErrorCode.CONNECTION_ERROR),
            (r"network", ProviderErrorCode.NETWORK_ERROR),
            (r"unavailable", ProviderErrorCode.PROVIDER_UNAVAILABLE),
            (r"maintenance", ProviderErrorCode.PROVIDER_MAINTENANCE),
        ]

        for pattern, code in patterns:
            if re.search(pattern, error_lower):
                return code

        return ProviderErrorCode.UNKNOWN

    def _determine_severity(self, code: ProviderErrorCode) -> ErrorSeverity:
        """Determine error severity based on error code."""
        critical_codes = {
            ProviderErrorCode.AUTHENTICATION_FAILED,
            ProviderErrorCode.API_KEY_INVALID,
            ProviderErrorCode.API_KEY_EXPIRED,
            ProviderErrorCode.AUTHORIZATION_FAILED,
        }

        high_codes = {
            ProviderErrorCode.QUOTA_EXCEEDED,
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            ProviderErrorCode.SERVICE_UNAVAILABLE,
        }

        low_codes = {
            ProviderErrorCode.RATE_LIMITED,
            ProviderErrorCode.MODEL_OVERLOADED,
        }

        if code in critical_codes:
            return ErrorSeverity.CRITICAL
        if code in high_codes:
            return ErrorSeverity.HIGH
        if code in low_codes:
            return ErrorSeverity.LOW
        return ErrorSeverity.MEDIUM

    def _extract_retry_after(self, error: Exception) -> int | None:
        """Extract retry-after value from error if present."""
        # Check for retry_after attribute
        if hasattr(error, "retry_after"):
            return int(error.retry_after)

        # Check in response headers
        if hasattr(error, "response"):
            response = error.response
            if hasattr(response, "headers"):
                retry = response.headers.get("Retry-After")
                if retry:
                    try:
                        return int(retry)
                    except ValueError:
                        pass

        # Check in error message
        error_str = str(error)
        match = re.search(r"retry.*?(\d+)\s*(?:second|sec|s)", error_str, re.I)
        if match:
            return int(match.group(1))

        return None

    def _get_generic_message(self, code: ProviderErrorCode) -> str:
        """Get a generic message for an error code."""
        messages = {
            ProviderErrorCode.RATE_LIMITED: "Request rate limit exceeded",
            ProviderErrorCode.QUOTA_EXCEEDED: "API quota exceeded",
            ProviderErrorCode.AUTHENTICATION_FAILED: "Authentication failed",
            ProviderErrorCode.AUTHORIZATION_FAILED: "Authorization failed",
            ProviderErrorCode.API_KEY_INVALID: "Invalid API key",
            ProviderErrorCode.API_KEY_EXPIRED: "API key expired",
            ProviderErrorCode.MODEL_NOT_FOUND: "Model not found",
            ProviderErrorCode.MODEL_OVERLOADED: "Model is overloaded",
            ProviderErrorCode.MODEL_DEPRECATED: "Model is deprecated",
            ProviderErrorCode.MODEL_UNAVAILABLE: "Model is unavailable",
            ProviderErrorCode.CONTENT_POLICY_VIOLATION: "Policy violation",
            ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED: "Context exceeded",
            ProviderErrorCode.INVALID_REQUEST: "Invalid request",
            ProviderErrorCode.INVALID_PARAMETERS: "Invalid parameters",
            ProviderErrorCode.MALFORMED_REQUEST: "Malformed request",
            ProviderErrorCode.PROVIDER_UNAVAILABLE: "Provider unavailable",
            ProviderErrorCode.PROVIDER_MAINTENANCE: "Provider maintenance",
            ProviderErrorCode.SERVICE_UNAVAILABLE: "Service unavailable",
            ProviderErrorCode.TIMEOUT: "Request timed out",
            ProviderErrorCode.CONNECTION_ERROR: "Connection error",
            ProviderErrorCode.NETWORK_ERROR: "Network error",
            ProviderErrorCode.STREAM_INTERRUPTED: "Stream interrupted",
            ProviderErrorCode.STREAM_ERROR: "Streaming error",
            ProviderErrorCode.INTERNAL_ERROR: "Internal error",
            ProviderErrorCode.UNKNOWN: "An unknown error occurred",
        }
        return messages.get(code, "An error occurred")

    def _get_suggested_action(self, code: ProviderErrorCode) -> str:
        """Get a suggested action for an error code."""
        actions = {
            ProviderErrorCode.RATE_LIMITED: (
                "Wait and retry the request, or reduce request frequency"
            ),
            ProviderErrorCode.QUOTA_EXCEEDED: ("Upgrade your plan or wait for quota reset"),
            ProviderErrorCode.AUTHENTICATION_FAILED: ("Check your API key and ensure it is valid"),
            ProviderErrorCode.API_KEY_INVALID: ("Verify your API key is correct and active"),
            ProviderErrorCode.API_KEY_EXPIRED: (
                "Generate a new API key from the provider dashboard"
            ),
            ProviderErrorCode.MODEL_NOT_FOUND: ("Check the model name or select a different model"),
            ProviderErrorCode.MODEL_OVERLOADED: ("Wait and retry, or try a different model"),
            ProviderErrorCode.MODEL_UNAVAILABLE: ("Select a different model or try again later"),
            ProviderErrorCode.CONTENT_POLICY_VIOLATION: (
                "Modify your prompt to comply with content policies"
            ),
            ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED: (
                "Reduce the length of your input or use a model with " "larger context"
            ),
            ProviderErrorCode.PROVIDER_UNAVAILABLE: (
                "Try a different provider or wait for service restoration"
            ),
            ProviderErrorCode.TIMEOUT: ("Retry the request or reduce prompt complexity"),
            ProviderErrorCode.CONNECTION_ERROR: ("Check your network connection and retry"),
        }
        return actions.get(code, "Try again or contact support if the issue persists")

    def _get_user_message(self, code: ProviderErrorCode, provider: str) -> str:
        """Get a user-friendly message for an error."""
        messages = {
            ProviderErrorCode.RATE_LIMITED: (
                f"Too many requests to {provider}. Please wait a moment " "before trying again."
            ),
            ProviderErrorCode.QUOTA_EXCEEDED: (
                f"Your {provider} API quota has been exceeded. Please check " "your plan limits."
            ),
            ProviderErrorCode.AUTHENTICATION_FAILED: (
                f"Unable to authenticate with {provider}. Please check your " "API key."
            ),
            ProviderErrorCode.MODEL_NOT_FOUND: (
                "The selected model is not available. Please choose a " "different model."
            ),
            ProviderErrorCode.MODEL_OVERLOADED: (
                f"The {provider} model is currently overloaded. Please try " "again shortly."
            ),
            ProviderErrorCode.CONTENT_POLICY_VIOLATION: (
                "Your request was flagged by content policies. Please modify " "your input."
            ),
            ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED: (
                "Your input is too long. Please reduce the text length and " "try again."
            ),
            ProviderErrorCode.PROVIDER_UNAVAILABLE: (
                f"{provider} is currently unavailable. Please try a different " "provider."
            ),
            ProviderErrorCode.TIMEOUT: (
                "The request took too long. Please try again with a shorter " "prompt."
            ),
        }
        return messages.get(code, "An error occurred. Please try again.")

    def _record_error(self, error: NormalizedProviderError) -> None:
        """Record an error in the history for metrics."""
        self._error_history.append(error)

        # Trim history if needed
        if len(self._error_history) > self._max_history_size:
            self._error_history = self._error_history[-self._max_history_size :]

    def get_retry_strategy(
        self,
        provider: str,
        error: NormalizedProviderError,
    ) -> RetryStrategy:
        """
        Get the retry strategy for an error.

        Args:
            provider: Provider that returned the error
            error: Normalized error

        Returns:
            RetryStrategy with retry configuration
        """
        # Use error's retry strategy if set
        if error.retry_strategy:
            return error.retry_strategy

        # Check for provider-specific strategy
        mapper = self._error_mappers.get(provider.lower())
        if mapper:
            strategy = mapper.get_retry_strategy(error)
            if strategy:
                return strategy

        # Use default strategy based on error code
        default = get_default_retry_strategy(error.code)
        if default:
            # Adjust for retry_after if present
            if error.retry_after_seconds:
                return RetryStrategy(
                    should_retry=True,
                    max_retries=default.max_retries,
                    base_delay_seconds=float(error.retry_after_seconds),
                    backoff_multiplier=1.0,
                    max_delay_seconds=default.max_delay_seconds,
                    jitter=False,
                )
            return default

        # Not retryable
        return RetryStrategy(should_retry=False, max_retries=0)

    def get_fallback_suggestion(
        self,
        provider: str,
        error: NormalizedProviderError,
        available_providers: list[str] | None = None,
    ) -> FallbackSuggestion | None:
        """
        Get a fallback suggestion for an error.

        Args:
            provider: Provider that returned the error
            error: Normalized error
            available_providers: List of available provider IDs

        Returns:
            FallbackSuggestion if a fallback is available
        """
        # Errors that should suggest fallback
        fallback_codes = {
            ProviderErrorCode.QUOTA_EXCEEDED,
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            ProviderErrorCode.SERVICE_UNAVAILABLE,
            ProviderErrorCode.MODEL_UNAVAILABLE,
            ProviderErrorCode.MODEL_OVERLOADED,
            ProviderErrorCode.AUTHENTICATION_FAILED,
            ProviderErrorCode.API_KEY_INVALID,
        }

        if error.code not in fallback_codes:
            return None

        # Get fallback options for provider
        fallbacks = self._fallback_mappings.get(provider.lower(), [])

        for alt_provider, alt_model in fallbacks:
            # Skip if not in available providers
            if available_providers and alt_provider not in available_providers:
                continue

            return FallbackSuggestion(
                alternative_provider=alt_provider,
                alternative_model=alt_model,
                reason=self._get_fallback_reason(error.code),
                confidence=0.8,
                capability_match=0.9,
            )

        return None

    def _get_fallback_reason(self, code: ProviderErrorCode) -> str:
        """Get the reason for suggesting a fallback."""
        reasons = {
            ProviderErrorCode.QUOTA_EXCEEDED: ("Primary provider quota exceeded"),
            ProviderErrorCode.PROVIDER_UNAVAILABLE: ("Primary provider is currently unavailable"),
            ProviderErrorCode.MODEL_UNAVAILABLE: ("Selected model is not available"),
            ProviderErrorCode.MODEL_OVERLOADED: ("Model is experiencing high demand"),
            ProviderErrorCode.AUTHENTICATION_FAILED: (
                "Authentication with primary provider failed"
            ),
        }
        return reasons.get(code, "Primary provider error")

    def get_error_metrics(
        self,
        provider: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get error metrics for monitoring.

        Args:
            provider: Optional provider to filter by
            time_range_hours: Number of hours to look back

        Returns:
            Dictionary of error metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=time_range_hours)

        # Filter errors
        errors = [
            e
            for e in self._error_history
            if e.timestamp >= cutoff and (provider is None or e.provider == provider.lower())
        ]

        if not errors:
            return {
                "time_range_hours": time_range_hours,
                "total_errors": 0,
                "errors_by_provider": {},
                "errors_by_code": {},
                "errors_by_severity": {},
            }

        # Aggregate metrics
        by_provider: dict[str, int] = defaultdict(int)
        by_code: dict[str, int] = defaultdict(int)
        by_severity: dict[str, int] = defaultdict(int)
        by_model: dict[str, int] = defaultdict(int)

        for error in errors:
            by_provider[error.provider] += 1
            by_code[error.code.value] += 1
            by_severity[error.severity.value] += 1
            if error.model:
                by_model[error.model] += 1

        return {
            "time_range_hours": time_range_hours,
            "total_errors": len(errors),
            "errors_by_provider": dict(by_provider),
            "errors_by_code": dict(by_code),
            "errors_by_severity": dict(by_severity),
            "errors_by_model": dict(by_model),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def clear_history(self) -> None:
        """Clear the error history."""
        self._error_history.clear()


class BaseErrorMapper:
    """
    Base class for provider-specific error mappers.

    Subclasses implement provider-specific error mapping logic.
    """

    provider_type: str = "base"

    def map_error(
        self,
        error: Exception,
        model: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> NormalizedProviderError:
        """
        Map a provider error to a normalized error.

        Args:
            error: The original exception
            model: Model that was being used
            request_id: Request ID for tracing
            trace_id: Distributed trace ID

        Returns:
            NormalizedProviderError
        """
        raise NotImplementedError

    def get_retry_strategy(self, error: NormalizedProviderError) -> RetryStrategy | None:
        """Get a provider-specific retry strategy."""
        return None

    def _extract_status_code(self, error: Exception) -> int | None:
        """Extract HTTP status code from various error types."""
        # Direct attribute
        if hasattr(error, "status_code"):
            return error.status_code
        if hasattr(error, "status"):
            return error.status

        # From response
        if hasattr(error, "response"):
            response = error.response
            if hasattr(response, "status_code"):
                return response.status_code
            if hasattr(response, "status"):
                return response.status

        return None

    def _extract_error_code(self, error: Exception) -> str | None:
        """Extract error code from various error types."""
        if hasattr(error, "code"):
            return str(error.code)
        if hasattr(error, "error_code"):
            return str(error.error_code)

        # From response body
        if hasattr(error, "body"):
            body = error.body
            if isinstance(body, dict):
                return body.get("error", {}).get("code")

        return None


# =============================================================================
# Global Error Handler Instance
# =============================================================================

_error_handler: ProviderErrorHandler | None = None


def get_error_handler() -> ProviderErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ProviderErrorHandler()
    return _error_handler


def normalize_provider_error(
    provider: str,
    error: Exception,
    model: str | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> NormalizedProviderError:
    """
    Convenience function to normalize a provider error.

    Args:
        provider: Provider that returned the error
        error: The original exception
        model: Model that was being used
        request_id: Request ID for tracing
        trace_id: Distributed trace ID

    Returns:
        NormalizedProviderError
    """
    return get_error_handler().normalize_error(provider, error, model, request_id, trace_id)


def create_provider_exception(
    provider: str,
    error: Exception,
    model: str | None = None,
    request_id: str | None = None,
) -> ProviderException:
    """
    Create a ProviderException from an error.

    Args:
        provider: Provider that returned the error
        error: The original exception
        model: Model that was being used
        request_id: Request ID for tracing

    Returns:
        ProviderException with normalized error
    """
    normalized = normalize_provider_error(provider, error, model, request_id)
    return ProviderException(normalized, error)


__all__ = [
    "BaseErrorMapper",
    "ProviderErrorHandler",
    "create_provider_exception",
    "get_error_handler",
    "normalize_provider_error",
]
