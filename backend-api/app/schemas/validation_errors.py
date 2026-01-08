"""
Validation Error Schemas for Provider/Model Selection System.

This module defines standardized error response schemas for selection validation
failures, ensuring consistent error handling across all API endpoints.

Usage:
    from app.schemas.validation_errors import SelectionValidationError, ValidationErrorType

    error = SelectionValidationError(
        error_type=ValidationErrorType.INVALID_PROVIDER,
        message="Provider 'unknown' is not registered",
        provider="unknown",
        available_providers=["openai", "anthropic", "google"]
    )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ValidationErrorType(str, Enum):
    """Types of selection validation errors."""

    INVALID_PROVIDER = "invalid_provider"
    INVALID_MODEL = "invalid_model"
    MISSING_API_KEY = "missing_api_key"
    PROVIDER_UNHEALTHY = "provider_unhealthy"
    PROVIDER_DISABLED = "provider_disabled"
    MODEL_DISABLED = "model_disabled"
    RATE_LIMITED = "rate_limited"
    SELECTION_MISMATCH = "selection_mismatch"
    CONTEXT_MISSING = "context_missing"
    VALIDATION_TIMEOUT = "validation_timeout"


class SelectionValidationError(BaseModel):
    """
    Standardized error response for selection validation failures.

    This schema provides comprehensive error information including:
    - Error type for programmatic handling
    - Human-readable message
    - Current selection values
    - Available alternatives when applicable
    - Metadata for debugging
    """

    error_type: ValidationErrorType = Field(
        ...,
        description="Type of validation error"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    provider: Optional[str] = Field(
        None,
        description="The provider that failed validation"
    )
    model: Optional[str] = Field(
        None,
        description="The model that failed validation"
    )
    available_providers: Optional[list[str]] = Field(
        None,
        description="List of available provider IDs"
    )
    available_models: Optional[list[str]] = Field(
        None,
        description="List of available model IDs for the provider"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracing"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error metadata"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_response_dict(self) -> dict[str, Any]:
        """Convert to API response dictionary."""
        return {
            "error": {
                "type": self.error_type.value,
                "message": self.message,
                "provider": self.provider,
                "model": self.model,
                "available_providers": self.available_providers,
                "available_models": self.available_models,
                "request_id": self.request_id,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "metadata": self.metadata,
            }
        }


class ValidationResult(BaseModel):
    """
    Result of a selection validation operation.

    Captures whether validation passed, any errors encountered,
    and diagnostic information.
    """

    is_valid: bool = Field(
        ...,
        description="Whether the selection passed validation"
    )
    provider: str = Field(
        ...,
        description="Provider ID that was validated"
    )
    model: str = Field(
        ...,
        description="Model ID that was validated"
    )
    errors: list[SelectionValidationError] = Field(
        default_factory=list,
        description="List of validation errors (if any)"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings about the selection"
    )
    validation_time_ms: float = Field(
        default=0.0,
        description="Time taken to validate (milliseconds)"
    )
    source: str = Field(
        default="unknown",
        description="Source of the selection (request, session, global)"
    )
    provider_health: Optional[str] = Field(
        None,
        description="Health status of the provider (healthy, degraded, unhealthy)"
    )
    api_key_configured: Optional[bool] = Field(
        None,
        description="Whether API key is configured for the provider"
    )

    @property
    def first_error(self) -> Optional[SelectionValidationError]:
        """Get the first error if any exist."""
        return self.errors[0] if self.errors else None

    def to_headers(self) -> dict[str, str]:
        """Convert validation result to HTTP headers for debugging."""
        headers = {
            "X-Selection-Validated": str(self.is_valid).lower(),
            "X-Validation-Provider": self.provider,
            "X-Validation-Model": self.model,
            "X-Validation-Source": self.source,
        }

        if self.provider_health:
            headers["X-Provider-Health"] = self.provider_health

        if self.api_key_configured is not None:
            headers["X-API-Key-Configured"] = str(self.api_key_configured).lower()

        if self.validation_time_ms > 0:
            headers["X-Validation-Time-Ms"] = f"{self.validation_time_ms:.2f}"

        if self.errors:
            headers["X-Validation-Error-Type"] = self.first_error.error_type.value

        return headers


class BoundaryValidationResult(BaseModel):
    """
    Result of provider boundary validation.

    Used by ProviderBoundaryGuard to track selection consistency
    at service boundaries.
    """

    is_consistent: bool = Field(
        ...,
        description="Whether the provider/model matches expectations"
    )
    expected_provider: str = Field(
        ...,
        description="Expected provider ID"
    )
    expected_model: str = Field(
        ...,
        description="Expected model ID"
    )
    actual_provider: Optional[str] = Field(
        None,
        description="Actual provider ID found"
    )
    actual_model: Optional[str] = Field(
        None,
        description="Actual model ID found"
    )
    boundary_name: str = Field(
        default="unknown",
        description="Name of the service boundary"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if validation failed"
    )


class ValidationMetrics(BaseModel):
    """
    Aggregated validation metrics.

    Used for monitoring and observability.
    """

    total_validations: int = Field(
        default=0,
        description="Total number of validations performed"
    )
    successful_validations: int = Field(
        default=0,
        description="Number of successful validations"
    )
    failed_validations: int = Field(
        default=0,
        description="Number of failed validations"
    )
    error_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each error type"
    )
    source_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of selection sources"
    )
    provider_health_at_validation: dict[str, int] = Field(
        default_factory=dict,
        description="Provider health status counts at validation time"
    )
    avg_validation_time_ms: float = Field(
        default=0.0,
        description="Average validation time in milliseconds"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When metrics were last updated"
    )

    def record_validation(self, result: ValidationResult) -> None:
        """Record a validation result in the metrics."""
        self.total_validations += 1

        if result.is_valid:
            self.successful_validations += 1
        else:
            self.failed_validations += 1
            for error in result.errors:
                error_type = error.error_type.value
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Track source distribution
        self.source_distribution[result.source] = (
            self.source_distribution.get(result.source, 0) + 1
        )

        # Track provider health
        if result.provider_health:
            self.provider_health_at_validation[result.provider_health] = (
                self.provider_health_at_validation.get(result.provider_health, 0) + 1
            )

        # Update average validation time (running average)
        if self.total_validations == 1:
            self.avg_validation_time_ms = result.validation_time_ms
        else:
            self.avg_validation_time_ms = (
                (self.avg_validation_time_ms * (self.total_validations - 1) +
                 result.validation_time_ms) / self.total_validations
            )

        self.last_updated = datetime.utcnow()

    def get_success_rate(self) -> float:
        """Get the validation success rate."""
        if self.total_validations == 0:
            return 0.0
        return self.successful_validations / self.total_validations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "total_validations": self.total_validations,
            "successful_validations": self.successful_validations,
            "failed_validations": self.failed_validations,
            "success_rate": self.get_success_rate(),
            "error_counts": self.error_counts,
            "source_distribution": self.source_distribution,
            "provider_health_at_validation": self.provider_health_at_validation,
            "avg_validation_time_ms": self.avg_validation_time_ms,
            "last_updated": self.last_updated.isoformat(),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_invalid_provider_error(
    provider: str,
    available_providers: list[str],
    request_id: Optional[str] = None,
) -> SelectionValidationError:
    """Create an error for invalid provider."""
    return SelectionValidationError(
        error_type=ValidationErrorType.INVALID_PROVIDER,
        message=f"Provider '{provider}' is not registered or available",
        provider=provider,
        available_providers=available_providers,
        request_id=request_id,
    )


def create_invalid_model_error(
    provider: str,
    model: str,
    available_models: list[str],
    request_id: Optional[str] = None,
) -> SelectionValidationError:
    """Create an error for invalid model."""
    return SelectionValidationError(
        error_type=ValidationErrorType.INVALID_MODEL,
        message=f"Model '{model}' is not available for provider '{provider}'",
        provider=provider,
        model=model,
        available_models=available_models,
        request_id=request_id,
    )


def create_missing_api_key_error(
    provider: str,
    request_id: Optional[str] = None,
) -> SelectionValidationError:
    """Create an error for missing API key."""
    return SelectionValidationError(
        error_type=ValidationErrorType.MISSING_API_KEY,
        message=f"API key is not configured for provider '{provider}'",
        provider=provider,
        request_id=request_id,
        metadata={"action": "Configure API key in settings or environment variables"},
    )


def create_provider_unhealthy_error(
    provider: str,
    health_status: str,
    request_id: Optional[str] = None,
) -> SelectionValidationError:
    """Create an error for unhealthy provider."""
    return SelectionValidationError(
        error_type=ValidationErrorType.PROVIDER_UNHEALTHY,
        message=f"Provider '{provider}' is currently unhealthy (status: {health_status})",
        provider=provider,
        request_id=request_id,
        metadata={"health_status": health_status},
    )


def create_context_missing_error(
    request_id: Optional[str] = None,
) -> SelectionValidationError:
    """Create an error for missing selection context."""
    return SelectionValidationError(
        error_type=ValidationErrorType.CONTEXT_MISSING,
        message="Selection context is not available for this request",
        request_id=request_id,
        metadata={"action": "Ensure SelectionMiddleware is configured"},
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ValidationErrorType",
    # Models
    "SelectionValidationError",
    "ValidationResult",
    "BoundaryValidationResult",
    "ValidationMetrics",
    # Factory functions
    "create_invalid_provider_error",
    "create_invalid_model_error",
    "create_missing_api_key_error",
    "create_provider_unhealthy_error",
    "create_context_missing_error",
]
