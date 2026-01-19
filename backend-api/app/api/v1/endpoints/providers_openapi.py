"""
OpenAPI/REST API Contract Specifications for Model Selection

This module defines the complete API contract for the multi-provider
AI model selection feature, including request/response schemas,
error responses, and endpoint documentation.
"""

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

# ============================================================================
# Enums
# ============================================================================


class ProviderEnum(str, Enum):
    """Supported AI providers."""

    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


class ModelTierEnum(str, Enum):
    """Model pricing/capability tiers."""

    STANDARD = "standard"
    PREMIUM = "premium"
    EXPERIMENTAL = "experimental"


class ProviderStatusEnum(str, Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class UserTierEnum(str, Enum):
    """User subscription tiers for rate limiting."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# ============================================================================
# Request Schemas
# ============================================================================


class SelectProviderRequest(BaseModel):
    """Request to select a provider and model for the session."""

    provider: ProviderEnum = Field(..., description="AI provider identifier", example="gemini")
    model: str = Field(
        ...,
        description="Model identifier within the provider",
        example="gemini-3-pro-preview",
        min_length=1,
        max_length=100,
    )

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {"provider": "gemini", "model": "gemini-3-pro-preview"}
        }


class GenerateWithModelRequest(BaseModel):
    """Request to generate text with a specific model."""

    prompt: str = Field(
        ..., description="The input prompt for generation", min_length=1, max_length=50000
    )
    system_instruction: str | None = Field(
        None, description="System instruction to guide the model", max_length=10000
    )
    provider: ProviderEnum | None = Field(
        None, description="Override provider (uses session default if not specified)"
    )
    model: str | None = Field(
        None, description="Override model (uses session default if not specified)"
    )
    config: dict[str, Any] | None = Field(
        None, description="Generation configuration (temperature, top_p, etc.)"
    )

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "system_instruction": "You are a helpful science teacher",
                "provider": "gemini",
                "model": "gemini-3-pro-preview",
                "config": {"temperature": 0.7, "max_output_tokens": 2048},
            }
        }


# ============================================================================
# Response Schemas
# ============================================================================


class ModelInfoResponse(BaseModel):
    """Detailed information about a model."""

    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: str | None = Field(None, description="Model description")
    max_tokens: int = Field(4096, description="Maximum output tokens")
    is_default: bool = Field(False, description="Whether this is the default model")
    tier: ModelTierEnum = Field(ModelTierEnum.STANDARD, description="Model tier")
    supports_streaming: bool = Field(True, description="Supports streaming responses")
    supports_vision: bool = Field(False, description="Supports image inputs")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "id": "gemini-3-pro-preview",
                "name": "Gemini 3 Pro Preview",
                "description": "Latest Gemini model with advanced reasoning",
                "max_tokens": 8192,
                "is_default": True,
                "tier": "premium",
                "supports_streaming": True,
                "supports_vision": False,
            }
        }


class ProviderInfoResponse(BaseModel):
    """Information about a provider with its models."""

    provider: str = Field(..., description="Provider identifier")
    display_name: str = Field(..., description="Human-readable provider name")
    status: ProviderStatusEnum = Field(..., description="Current health status")
    is_healthy: bool = Field(..., description="Whether provider is operational")
    models: list[str] = Field(..., description="List of available model IDs")
    default_model: str | None = Field(None, description="Default model for this provider")
    latency_ms: float | None = Field(None, description="Average response latency in ms")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "provider": "gemini",
                "display_name": "Google Gemini",
                "status": "healthy",
                "is_healthy": True,
                "models": ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
                "default_model": "gemini-3-pro-preview",
                "latency_ms": 245.5,
            }
        }


class ProvidersListResponse(BaseModel):
    """Response containing all available providers."""

    providers: list[ProviderInfoResponse] = Field(..., description="List of providers")
    count: int = Field(..., description="Total number of providers")
    default_provider: str = Field(..., description="System default provider")
    default_model: str = Field(..., description="System default model")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "providers": [
                    {
                        "provider": "gemini",
                        "display_name": "Google Gemini",
                        "status": "healthy",
                        "is_healthy": True,
                        "models": ["gemini-3-pro-preview", "gemini-2.5-pro"],
                        "default_model": "gemini-3-pro-preview",
                        "latency_ms": 245.5,
                    },
                    {
                        "provider": "deepseek",
                        "display_name": "DeepSeek",
                        "status": "healthy",
                        "is_healthy": True,
                        "models": ["deepseek-chat", "deepseek-reasoner"],
                        "default_model": "deepseek-chat",
                        "latency_ms": 180.2,
                    },
                ],
                "count": 2,
                "default_provider": "deepseek",
                "default_model": "deepseek-chat",
            }
        }


class ProviderModelsResponse(BaseModel):
    """Response containing models for a specific provider."""

    provider: str = Field(..., description="Provider identifier")
    display_name: str = Field(..., description="Human-readable provider name")
    models: list[ModelInfoResponse] = Field(..., description="Detailed model information")
    default_model: str | None = Field(None, description="Default model ID")
    count: int = Field(..., description="Total number of models")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "provider": "gemini",
                "display_name": "Google Gemini",
                "models": [
                    {
                        "id": "gemini-3-pro-preview",
                        "name": "Gemini 3 Pro Preview",
                        "tier": "premium",
                        "max_tokens": 8192,
                        "is_default": True,
                    }
                ],
                "default_model": "gemini-3-pro-preview",
                "count": 3,
            }
        }


class SelectProviderResponse(BaseModel):
    """Response after selecting a provider/model."""

    success: bool = Field(..., description="Whether selection was successful")
    message: str = Field(..., description="Status message")
    provider: str = Field(..., description="Selected provider")
    model: str = Field(..., description="Selected model")
    session_id: str = Field(..., description="Session identifier")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "success": True,
                "message": "Model selection updated successfully",
                "provider": "gemini",
                "model": "gemini-3-pro-preview",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }


class CurrentSelectionResponse(BaseModel):
    """Response containing current model selection."""

    provider: str = Field(..., description="Currently selected provider")
    model: str = Field(..., description="Currently selected model")
    display_name: str = Field(..., description="Provider display name")
    session_id: str | None = Field(None, description="Session identifier")
    is_default: bool = Field(..., description="Whether using default selection")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "provider": "gemini",
                "model": "gemini-3-pro-preview",
                "display_name": "Google Gemini",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "is_default": False,
            }
        }


class ProviderHealthResponse(BaseModel):
    """Response containing provider health information."""

    providers: list[dict[str, Any]] = Field(..., description="Health data for each provider")
    timestamp: str = Field(..., description="Timestamp of health check")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "providers": [
                    {
                        "provider": "gemini",
                        "is_healthy": True,
                        "last_check": "2025-01-15T10:30:00Z",
                        "latency_ms": 245.5,
                        "consecutive_failures": 0,
                    }
                ],
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }


class RateLimitInfoResponse(BaseModel):
    """Response containing rate limit information."""

    allowed: bool = Field(..., description="Whether request is allowed")
    remaining_requests: int = Field(..., description="Remaining requests in window")
    remaining_tokens: int = Field(..., description="Remaining tokens in window")
    reset_at: str = Field(..., description="When limits reset (ISO timestamp)")
    retry_after_seconds: int | None = Field(None, description="Seconds until retry allowed")
    limit_type: str | None = Field(None, description="Type of limit hit")
    tier: str = Field(..., description="User's rate limit tier")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "allowed": True,
                "remaining_requests": 45,
                "remaining_tokens": 85000,
                "reset_at": "2025-01-15T10:31:00Z",
                "tier": "pro",
            }
        }


# ============================================================================
# Error Response Schemas
# ============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: str | None = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: str | None = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    status_code: int = Field(..., description="HTTP status code")
    details: list[ErrorDetail] | None = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: str | None = Field(None, description="Request identifier for debugging")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "error_code": "PROVIDER_UNAVAILABLE",
                "message": "The requested provider is currently unavailable",
                "status_code": 503,
                "details": [
                    {
                        "field": "provider",
                        "message": "Provider 'gemini' is experiencing issues",
                        "code": "CIRCUIT_BREAKER_OPEN",
                    }
                ],
                "timestamp": "2025-01-15T10:30:00Z",
                "request_id": "req_abc123",
            }
        }


class RateLimitErrorResponse(ErrorResponse):
    """Error response for rate limit exceeded."""

    retry_after: int = Field(..., description="Seconds until retry is allowed")
    fallback_provider: str | None = Field(None, description="Suggested fallback provider")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": "Rate limit exceeded for provider 'gemini'",
                "status_code": 429,
                "retry_after": 45,
                "fallback_provider": "deepseek",
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }


# ============================================================================
# WebSocket Message Schemas
# ============================================================================


class WebSocketMessageBase(BaseModel):
    """Base schema for WebSocket messages."""

    type: str = Field(..., description="Message type")
    timestamp: str = Field(..., description="Message timestamp")


class SelectionChangeMessage(WebSocketMessageBase):
    """WebSocket message for model selection changes."""

    type: str = Field("selection_change", const=True)
    data: dict[str, Any] = Field(..., description="Selection change data")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "type": "selection_change",
                "data": {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "previous_provider": "gemini",
                    "previous_model": "gemini-3-pro-preview",
                },
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }


class HealthUpdateMessage(WebSocketMessageBase):
    """WebSocket message for provider health updates."""

    type: str = Field("health_update", const=True)
    data: dict[str, Any] = Field(..., description="Health update data")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "type": "health_update",
                "data": {
                    "providers": [
                        {"provider": "gemini", "is_healthy": True, "latency_ms": 250},
                        {"provider": "deepseek", "is_healthy": True, "latency_ms": 180},
                    ]
                },
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }


# ============================================================================
# API Contract Summary
# ============================================================================

API_CONTRACT = """
# Model Selection API Contract

## Base URL
`/api/v1/providers`

## Authentication
All endpoints require authentication via:
- `Authorization: Bearer <token>` header, OR
- `X-API-Key: <api_key>` header

## Endpoints

### GET /available
List all available providers with health status.

**Response:** `ProvidersListResponse`
**Rate Limit:** 100 requests/minute

### GET /{provider}/models
Get detailed model information for a specific provider.

**Parameters:**
- `provider` (path): Provider identifier (gemini, deepseek)

**Response:** `ProviderModelsResponse`
**Rate Limit:** 100 requests/minute

### POST /select
Select a provider and model for the current session.

**Request Body:** `SelectProviderRequest`
**Response:** `SelectProviderResponse`
**Rate Limit:** 30 requests/minute

### GET /current
Get the current model selection for the session.

**Response:** `CurrentSelectionResponse`
**Rate Limit:** 100 requests/minute

### GET /health
Get health status for all providers.

**Response:** `ProviderHealthResponse`
**Rate Limit:** 60 requests/minute

### WebSocket /ws/selection
Real-time model selection synchronization.

**Messages:**
- `selection_change`: Broadcast when model selection changes
- `health_update`: Broadcast when provider health changes
- `heartbeat`: Periodic keepalive (every 30s)

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 400 | Invalid request parameters |
| AUTHENTICATION_ERROR | 401 | Missing or invalid credentials |
| PROVIDER_NOT_FOUND | 404 | Requested provider doesn't exist |
| MODEL_NOT_FOUND | 404 | Requested model doesn't exist |
| RATE_LIMIT_EXCEEDED | 429 | Rate limit exceeded |
| PROVIDER_UNAVAILABLE | 503 | Provider is temporarily unavailable |
| CIRCUIT_BREAKER_OPEN | 503 | Circuit breaker is open |

## Rate Limit Headers

All responses include:
- `X-RateLimit-Remaining-Requests`: Remaining requests in window
- `X-RateLimit-Remaining-Tokens`: Remaining tokens in window
- `X-RateLimit-Reset`: Window reset timestamp
- `X-RateLimit-Provider`: Provider for this limit

When rate limited (429):
- `Retry-After`: Seconds until retry allowed
"""
