"""
AI Provider Configuration Settings

Pydantic models for type-safe AI provider configuration management.
Provides validation, serialization, and hot-reload support.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Enumerations
# =============================================================================


class ProviderType(str, Enum):
    """Supported AI provider types."""

    GOOGLE = "google"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    BIGMODEL = "bigmodel"
    CURSOR = "cursor"
    ROUTEWAY = "routeway"
    CUSTOM = "custom"


class ModelTier(str, Enum):
    """Model pricing/capability tier."""

    ECONOMY = "economy"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ProviderStatus(str, Enum):
    """Provider operational status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RATE_LIMITED = "rate_limited"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class CircuitState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# Rate Limit Configuration
# =============================================================================


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for a provider."""

    requests_per_minute: int | None = Field(
        default=60, ge=1, description="Maximum requests per minute"
    )
    tokens_per_minute: int | None = Field(
        default=100000, ge=1, description="Maximum tokens per minute"
    )
    requests_per_day: int | None = Field(
        default=None, ge=1, description="Maximum requests per day (null for unlimited)"
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Circuit Breaker Configuration
# =============================================================================


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for provider resilience."""

    enabled: bool = Field(default=True, description="Whether circuit breaker is enabled")
    failure_threshold: int = Field(
        default=5, ge=1, le=50, description="Number of failures before opening circuit"
    )
    recovery_timeout_seconds: int = Field(
        default=60, ge=10, le=600, description="Time in seconds before attempting recovery"
    )
    half_open_max_requests: int = Field(
        default=3, ge=1, le=10, description="Max requests to allow in half-open state"
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Pricing Configuration
# =============================================================================


class ModelPricingConfig(BaseModel):
    """Pricing information for a model (per 1K tokens)."""

    input_cost_per_1k: float = Field(
        default=0.0, ge=0.0, description="Cost per 1K input tokens in USD"
    )
    output_cost_per_1k: float = Field(
        default=0.0, ge=0.0, description="Cost per 1K output tokens in USD"
    )
    cached_input_cost_per_1k: float | None = Field(
        default=None, ge=0.0, description="Cost per 1K cached input tokens (if supported)"
    )
    reasoning_cost_per_1k: float | None = Field(
        default=None, ge=0.0, description="Cost per 1K reasoning tokens (for o1-like models)"
    )

    model_config = ConfigDict(frozen=True)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate total cost for a request."""
        cost = (input_tokens / 1000) * self.input_cost_per_1k
        cost += (output_tokens / 1000) * self.output_cost_per_1k

        if reasoning_tokens > 0 and self.reasoning_cost_per_1k:
            cost += (reasoning_tokens / 1000) * self.reasoning_cost_per_1k

        if cached_tokens > 0 and self.cached_input_cost_per_1k:
            cost += (cached_tokens / 1000) * self.cached_input_cost_per_1k

        return cost


# =============================================================================
# Model Configuration
# =============================================================================


class ModelConfig(BaseModel):
    """Configuration for a specific AI model."""

    # Model identification
    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: str | None = Field(
        default=None, description="Model description and use case"
    )

    # Context and token limits
    context_length: int = Field(
        default=4096, ge=1, description="Maximum context window size in tokens"
    )
    max_output_tokens: int = Field(
        default=4096, ge=1, description="Maximum output tokens per request"
    )

    # Capabilities
    supports_streaming: bool = Field(
        default=True, description="Whether model supports streaming responses"
    )
    supports_vision: bool = Field(
        default=False, description="Whether model supports image inputs"
    )
    supports_function_calling: bool = Field(
        default=False, description="Whether model supports function/tool calling"
    )

    # Classification
    is_default: bool = Field(
        default=False, description="Whether this is the default model for the provider"
    )
    tier: ModelTier = Field(
        default=ModelTier.STANDARD, description="Model pricing/capability tier"
    )

    # Pricing
    pricing: ModelPricingConfig | None = Field(
        default=None, description="Token pricing information"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific metadata"
    )

    model_config = ConfigDict(protected_namespaces=(), frozen=True)


# =============================================================================
# Provider Capabilities
# =============================================================================


class ProviderCapabilities(BaseModel):
    """Capability flags for a provider."""

    supports_streaming: bool = Field(
        default=True, description="Whether provider supports streaming responses"
    )
    supports_vision: bool = Field(
        default=False, description="Whether provider supports image inputs"
    )
    supports_function_calling: bool = Field(
        default=False, description="Whether provider supports function/tool calling"
    )
    supports_json_mode: bool = Field(
        default=False, description="Whether provider supports JSON output mode"
    )
    supports_system_prompt: bool = Field(
        default=True, description="Whether provider supports system prompts"
    )
    supports_token_counting: bool = Field(
        default=False, description="Whether provider supports token counting API"
    )
    supports_embeddings: bool = Field(
        default=False, description="Whether provider supports embeddings API"
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# API Configuration
# =============================================================================


class ProviderAPIConfig(BaseModel):
    """API connection configuration for a provider."""

    base_url: str = Field(..., description="Base URL for API requests")
    key_env_var: str = Field(
        ..., description="Environment variable name for API key"
    )
    timeout_seconds: int = Field(
        default=120, ge=5, le=600, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    custom_headers: dict[str, str] = Field(
        default_factory=dict, description="Additional headers to include in requests"
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Provider Configuration
# =============================================================================


class ProviderConfig(BaseModel):
    """Complete configuration for an AI provider."""

    # Provider identification
    provider_id: str = Field(..., description="Unique provider identifier")
    type: ProviderType = Field(..., description="Provider type")
    name: str = Field(..., description="Human-readable provider name")
    description: str | None = Field(default=None, description="Provider description")
    enabled: bool = Field(default=True, description="Whether provider is enabled")

    # API configuration
    api: ProviderAPIConfig = Field(..., description="API connection configuration")

    # Priority and failover
    priority: int = Field(
        default=50,
        ge=0,
        le=1000,
        description="Priority for failover ordering (higher = tried first)",
    )
    failover_chain: list[str] = Field(
        default_factory=list, description="Ordered list of provider IDs for failover"
    )

    # Capabilities and limits
    capabilities: ProviderCapabilities = Field(
        default_factory=ProviderCapabilities, description="Provider capability flags"
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig, description="Circuit breaker configuration"
    )
    rate_limits: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting configuration"
    )

    # Models
    models: dict[str, ModelConfig] = Field(
        default_factory=dict, description="Available models for this provider"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific metadata"
    )

    model_config = ConfigDict(protected_namespaces=(), frozen=True)

    @model_validator(mode="after")
    def validate_provider(self) -> "ProviderConfig":
        """Ensure at least one model is marked as default if models exist."""
        if self.models:
            has_default = any(m.is_default for m in self.models.values())
            if not has_default:
                # Log warning but don't fail - first model will be used as default
                pass
        return self

    def get_default_model(self) -> ModelConfig | None:
        """Get the default model for this provider."""
        for model in self.models.values():
            if model.is_default:
                return model
        return next(iter(self.models.values()), None) if self.models else None

    def get_model(self, model_id: str) -> ModelConfig | None:
        """Get a specific model by ID."""
        return self.models.get(model_id)

    def get_models_by_tier(self, tier: ModelTier) -> list[ModelConfig]:
        """Get all models of a specific tier."""
        return [m for m in self.models.values() if m.tier == tier]


# =============================================================================
# Cost Tracking Configuration
# =============================================================================


class CostTrackingConfig(BaseModel):
    """Configuration for cost tracking and budgets."""

    enabled: bool = Field(default=True, description="Whether cost tracking is enabled")
    daily_budget_usd: float | None = Field(
        default=None, ge=0.0, description="Daily budget limit in USD"
    )
    alert_threshold_percent: int = Field(
        default=80, ge=0, le=100, description="Percentage of budget to trigger alerts"
    )
    hard_limit_enabled: bool = Field(
        default=False, description="Whether to enforce hard budget limits"
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Global Rate Limiting Configuration
# =============================================================================


class GlobalRateLimitConfig(BaseModel):
    """Global rate limiting configuration."""

    enabled: bool = Field(default=True, description="Whether rate limiting is enabled")
    default_requests_per_minute: int = Field(
        default=60, ge=1, description="Default requests per minute"
    )
    default_tokens_per_minute: int = Field(
        default=100000, ge=1, description="Default tokens per minute"
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Global Configuration
# =============================================================================


class GlobalConfig(BaseModel):
    """Global configuration settings."""

    default_provider: str = Field(default="gemini", description="Default provider ID")
    default_model: str = Field(
        default="gemini-2.0-flash-exp", description="Default model ID"
    )
    failover_enabled: bool = Field(
        default=True, description="Enable automatic provider failover"
    )
    max_failover_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum failover attempts"
    )
    health_check_interval: int = Field(
        default=60, ge=10, le=600, description="Health check interval in seconds"
    )
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(
        default=300, ge=0, le=3600, description="Cache TTL in seconds"
    )
    cost_tracking: CostTrackingConfig = Field(
        default_factory=CostTrackingConfig, description="Cost tracking configuration"
    )
    rate_limiting: GlobalRateLimitConfig = Field(
        default_factory=GlobalRateLimitConfig,
        description="Default rate limiting configuration",
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Failover Chain Configuration
# =============================================================================


class FailoverChainConfig(BaseModel):
    """Named failover chain configuration."""

    name: str = Field(..., description="Chain name")
    description: str | None = Field(default=None, description="Chain description")
    providers: list[str] = Field(
        ..., min_length=1, description="Ordered list of provider IDs"
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Root AI Providers Configuration
# =============================================================================


class AIProvidersConfig(BaseModel):
    """
    Root configuration binding for the AI provider system.

    This is the top-level configuration model that loads from providers.yaml
    and provides access to all provider configurations.
    """

    # Schema version
    schema_version: str = Field(default="1.0.0", description="Configuration schema version")

    # Global settings
    global_config: GlobalConfig = Field(
        default_factory=GlobalConfig,
        alias="global",
        description="Global configuration settings",
    )

    # Provider configurations
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict, description="Provider configurations keyed by provider ID"
    )

    # Aliases
    aliases: dict[str, str] = Field(
        default_factory=dict, description="Provider name aliases"
    )

    # Named failover chains
    failover_chains: dict[str, FailoverChainConfig] = Field(
        default_factory=dict, description="Named failover chain configurations"
    )

    # Metadata
    loaded_at: datetime | None = Field(
        default=None, description="Timestamp when config was loaded"
    )
    config_hash: str | None = Field(
        default=None, description="Hash of config for change detection"
    )

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    @model_validator(mode="after")
    def validate_config(self) -> "AIProvidersConfig":
        """Validate configuration consistency."""
        # Validate default provider exists
        if self.providers and self.global_config.default_provider not in self.providers:
            raise ValueError(
                f"Default provider '{self.global_config.default_provider}' "
                "not found in providers"
            )

        # Validate failover chains reference valid providers
        for provider in self.providers.values():
            for failover_id in provider.failover_chain:
                resolved_id = self.aliases.get(failover_id, failover_id)
                if resolved_id not in self.providers:
                    raise ValueError(
                        f"Failover provider '{failover_id}' in "
                        f"'{provider.provider_id}' chain not found"
                    )

        # Validate named failover chains
        for chain_name, chain in self.failover_chains.items():
            for provider_id in chain.providers:
                resolved_id = self.aliases.get(provider_id, provider_id)
                if resolved_id not in self.providers:
                    raise ValueError(
                        f"Provider '{provider_id}' in failover chain "
                        f"'{chain_name}' not found"
                    )

        return self

    def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """Get provider by ID or alias."""
        resolved_id = self.aliases.get(provider_id.lower(), provider_id.lower())
        return self.providers.get(resolved_id)

    def get_default_provider(self) -> ProviderConfig | None:
        """Get the default provider configuration."""
        return self.providers.get(self.global_config.default_provider)

    def get_active_provider(self) -> ProviderConfig | None:
        """Get the currently active provider (alias for get_default_provider)."""
        return self.get_default_provider()

    def get_active_model(self) -> ModelConfig | None:
        """Get the currently active model from the default provider."""
        provider = self.get_default_provider()
        if not provider:
            return None

        # First try to find the specified default model
        model = provider.get_model(self.global_config.default_model)
        if model:
            return model

        # Fall back to provider's default model
        return provider.get_default_model()

    def get_enabled_providers(self) -> list[ProviderConfig]:
        """Get all enabled providers sorted by priority."""
        enabled = [p for p in self.providers.values() if p.enabled]
        return sorted(enabled, key=lambda p: -p.priority)

    def get_failover_chain(
        self, provider_id: str, chain_name: str | None = None
    ) -> list[str]:
        """Get failover chain for a provider."""
        if chain_name and chain_name in self.failover_chains:
            return self.failover_chains[chain_name].providers

        provider = self.get_provider(provider_id)
        if provider:
            return provider.failover_chain

        return []

    def resolve_provider_alias(self, name: str) -> str:
        """Resolve a provider alias to its canonical name."""
        return self.aliases.get(name.lower(), name.lower())

    def calculate_cost(
        self,
        provider_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost for a request."""
        provider = self.get_provider(provider_id)
        if not provider:
            return 0.0

        model = provider.get_model(model_id)
        if not model or not model.pricing:
            return 0.0

        return model.pricing.calculate_cost(
            input_tokens, output_tokens, reasoning_tokens, cached_tokens
        )


# =============================================================================
# Configuration Snapshot (for versioning)
# =============================================================================


class ConfigSnapshot(BaseModel):
    """Snapshot of configuration for versioning and rollback."""

    version: int = Field(..., description="Snapshot version number")
    config: AIProvidersConfig = Field(..., description="Configuration snapshot")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Snapshot creation timestamp"
    )
    created_by: str | None = Field(
        default=None, description="User/process that created the snapshot"
    )
    description: str | None = Field(
        default=None, description="Description of changes"
    )

    model_config = ConfigDict(frozen=True)
