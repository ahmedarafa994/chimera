"""
Provider Domain Models

Comprehensive models for dynamic AI provider management including:
- Provider configuration and metadata
- Provider status and health tracking
- Provider capabilities and features
- Secure API key handling
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderStatus(str, Enum):
    """Provider availability status"""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    INITIALIZING = "initializing"
    ERROR = "error"


class ProviderTier(str, Enum):
    """Provider pricing/capability tier"""

    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ProviderType(str, Enum):
    """Supported provider types"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    XAI = "xai"
    CURSOR = "cursor"
    LOCAL = "local"
    CUSTOM = "custom"


class ProviderCapabilities(BaseModel):
    """Provider capability flags"""

    supports_streaming: bool = True
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_system_prompt: bool = True
    supports_token_counting: bool = False
    max_context_length: int = 4096
    max_output_tokens: int = 4096
    supports_embeddings: bool = False
    supports_fine_tuning: bool = False


class ModelInfo(BaseModel):
    """Model information within a provider"""

    id: str
    name: str
    description: str | None = None
    max_tokens: int = 4096
    max_output_tokens: int = 4096
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_function_calling: bool = False
    is_default: bool = False
    tier: ProviderTier = ProviderTier.STANDARD
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None

    model_config = ConfigDict(protected_namespaces=())


class ProviderHealthInfo(BaseModel):
    """Provider health and performance metrics"""

    status: ProviderStatus = ProviderStatus.INITIALIZING
    last_check: datetime | None = None
    last_success: datetime | None = None
    last_error: str | None = None
    error_count: int = 0
    success_count: int = 0
    avg_latency_ms: float | None = None
    uptime_percentage: float | None = None
    rate_limit_remaining: int | None = None
    rate_limit_reset: datetime | None = None


class ProviderConfig(BaseModel):
    """Provider configuration for registration"""

    provider_type: ProviderType
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str | None = None
    api_key: str | None = None
    api_base_url: str | None = None
    organization_id: str | None = None
    default_model: str | None = None
    models: list[ModelInfo] = Field(default_factory=list)
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    priority: int = Field(default=100, ge=0, le=1000)
    enabled: bool = True
    is_fallback: bool = False
    timeout_seconds: int = Field(default=60, ge=5, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    custom_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(protected_namespaces=())

    @field_validator("name")
    def validate_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Provider name must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v.lower()

    @field_validator("api_key")
    def validate_api_key(cls, v):
        if v is not None and len(v) < 10:
            raise ValueError("API key must be at least 10 characters")
        return v

    @field_validator("api_base_url")
    def validate_api_base_url(cls, v):
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("API base URL must start with http:// or https://")
        return v.rstrip("/") if v else v


class ProviderInfo(BaseModel):
    """Complete provider information for API responses"""

    id: str
    provider_type: ProviderType
    name: str
    display_name: str
    status: ProviderStatus = ProviderStatus.INITIALIZING
    enabled: bool = True
    is_default: bool = False
    is_fallback: bool = False
    priority: int = 100
    default_model: str | None = None
    models: list[ModelInfo] = Field(default_factory=list)
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    health: ProviderHealthInfo = Field(default_factory=ProviderHealthInfo)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Exclude sensitive data from serialization
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "id": "openai-default",
                "provider_type": "openai",
                "name": "openai",
                "display_name": "OpenAI",
                "status": "available",
                "enabled": True,
                "is_default": True,
                "priority": 100,
                "default_model": "gpt-4o",
                "models": [{"id": "gpt-4o", "name": "GPT-4o", "max_tokens": 128000}],
            }
        },
    )


class ProviderRegistration(BaseModel):
    """Request model for registering a new provider"""

    provider_type: ProviderType
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str | None = None
    api_key: str = Field(..., min_length=10)
    api_base_url: str | None = None
    organization_id: str | None = None
    default_model: str | None = None
    priority: int = Field(default=100, ge=0, le=1000)
    enabled: bool = True
    is_fallback: bool = False
    timeout_seconds: int = Field(default=60, ge=5, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    custom_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(protected_namespaces=())

    @field_validator("name")
    def validate_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Provider name must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v.lower()


class ProviderUpdate(BaseModel):
    """Request model for updating a provider"""

    display_name: str | None = None
    api_key: str | None = None
    api_base_url: str | None = None
    organization_id: str | None = None
    default_model: str | None = None
    priority: int | None = Field(default=None, ge=0, le=1000)
    enabled: bool | None = None
    is_fallback: bool | None = None
    timeout_seconds: int | None = Field(default=None, ge=5, le=300)
    max_retries: int | None = Field(default=None, ge=0, le=10)
    custom_headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(protected_namespaces=())


class ProviderTestResult(BaseModel):
    """Result of testing a provider connection"""

    success: bool
    provider_id: str
    latency_ms: float | None = None
    error: str | None = None
    models_discovered: list[str] = Field(default_factory=list)
    capabilities_detected: ProviderCapabilities | None = None
    tested_at: datetime = Field(default_factory=datetime.utcnow)


class ProviderSelectionRequest(BaseModel):
    """Request to select/switch active provider"""

    provider_id: str
    model_id: str | None = None
    session_id: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class ProviderSelectionResponse(BaseModel):
    """Response after selecting a provider"""

    success: bool
    provider_id: str
    provider_name: str
    model_id: str | None = None
    previous_provider_id: str | None = None
    session_id: str | None = None
    message: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class ProviderListResponse(BaseModel):
    """Response containing list of providers"""

    providers: list[ProviderInfo]
    total: int
    default_provider_id: str | None = None
    active_provider_id: str | None = None


class ProviderHealthResponse(BaseModel):
    """Response containing provider health information"""

    provider_id: str
    health: ProviderHealthInfo
    recommendations: list[str] = Field(default_factory=list)


class ProviderFallbackConfig(BaseModel):
    """Configuration for provider fallback behavior"""

    enabled: bool = True
    max_attempts: int = Field(default=3, ge=1, le=10)
    fallback_order: list[str] = Field(default_factory=list)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    exponential_backoff: bool = True
    preserve_model_preference: bool = False


class ProviderRoutingConfig(BaseModel):
    """Configuration for provider routing/load balancing"""

    strategy: str = Field(
        default="priority", pattern="^(priority|round_robin|least_latency|cost_optimized)$"
    )
    health_check_interval_seconds: int = Field(default=30, ge=10, le=300)
    circuit_breaker_threshold: int = Field(default=5, ge=1, le=20)
    circuit_breaker_timeout_seconds: int = Field(default=60, ge=10, le=600)
    fallback: ProviderFallbackConfig = Field(default_factory=ProviderFallbackConfig)
