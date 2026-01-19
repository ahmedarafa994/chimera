# =============================================================================
# Chimera - Configuration Schema Models
# =============================================================================
# Pydantic models for configuration validation and type safety
# Part of Story 1.1: Provider Configuration Management
# =============================================================================

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from ..domain.provider_models import ProviderType


class ConnectionMode(str, Enum):
    """API connection modes"""

    DIRECT = "direct"
    PROXY = "proxy"


class ProviderConnectionConfig(BaseModel):
    """Configuration for a single provider connection"""

    provider_type: ProviderType
    name: str = Field(..., min_length=1, max_length=50)
    display_name: str | None = None
    api_key: str | None = None
    api_key_encrypted: bool = False
    base_url: str | None = None
    organization_id: str | None = None
    default_model: str | None = None
    models: list[str] = Field(default_factory=list)
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1000)
    timeout_seconds: int = Field(default=60, ge=5, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    custom_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate provider name format"""
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Provider name must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v.lower()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v):
        """Validate base URL format"""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v.rstrip("/") if v else v


class ProxyModeConfig(BaseModel):
    """Configuration for proxy mode (AIClient-2-API Server)"""

    enabled: bool = Field(default=False, description="Enable proxy mode")
    endpoint: str = Field(
        default="http://localhost:8080", description="AIClient-2-API Server endpoint"
    )
    health_check_enabled: bool = Field(
        default=True, description="Enable health check for proxy endpoint"
    )
    health_check_interval: int = Field(
        default=30, ge=10, le=300, description="Health check interval in seconds"
    )
    timeout_seconds: int = Field(
        default=30, ge=5, le=120, description="Request timeout for proxy calls"
    )
    fallback_to_direct: bool = Field(
        default=True, description="Fallback to direct mode if proxy unavailable"
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v):
        """Validate proxy endpoint URL"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Proxy endpoint must be a valid URL")
        return v.rstrip("/")


class ConfigurationValidationResult(BaseModel):
    """Result of configuration validation"""

    is_valid: bool
    provider: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    connectivity_test_passed: bool | None = None
    validated_at: datetime = Field(default_factory=datetime.now)


class ConfigurationReloadResult(BaseModel):
    """Result of configuration hot-reload"""

    status: str  # "success" or "error"
    reloaded_at: datetime = Field(default_factory=datetime.now)
    elapsed_ms: float
    changes: dict[str, list[str]] = Field(default_factory=dict)
    validation_results: list[ConfigurationValidationResult] = Field(default_factory=list)
    error: str | None = None
    callbacks_executed: int = 0
    callbacks_failed: int = 0


class ProviderConfigurationSummary(BaseModel):
    """Summary of provider configuration status"""

    total_providers: int
    configured_providers: list[str]
    valid_providers: list[str] = Field(default_factory=list)
    invalid_providers: list[str] = Field(default_factory=list)
    connection_mode: ConnectionMode
    proxy_enabled: bool = False
    encryption_enabled: bool = False
    last_validated: datetime | None = None
    health_status: dict[str, str] = Field(default_factory=dict)


class EncryptionConfig(BaseModel):
    """Configuration for API key encryption"""

    enabled: bool = Field(default=True, description="Enable API key encryption at rest")
    algorithm: str = Field(default="AES-256", description="Encryption algorithm")
    key_rotation_enabled: bool = Field(default=False, description="Enable automatic key rotation")
    key_rotation_days: int = Field(
        default=90, ge=30, le=365, description="Days between key rotations"
    )


class ChimeraConfiguration(BaseModel):
    """Complete Chimera configuration model"""

    # Basic settings
    project_name: str = "Chimera Backend"
    version: str = "1.0.0"
    environment: str = "development"
    api_v1_str: str = "/api/v1"

    # Connection mode
    connection_mode: ConnectionMode = ConnectionMode.DIRECT

    # Provider configurations
    providers: dict[str, ProviderConnectionConfig] = Field(default_factory=dict)

    # Proxy configuration
    proxy_mode: ProxyModeConfig = Field(default_factory=ProxyModeConfig)

    # Encryption configuration
    encryption: EncryptionConfig = Field(default_factory=EncryptionConfig)

    # Validation settings
    validate_on_startup: bool = Field(default=True, description="Validate configuration on startup")
    validate_connectivity: bool = Field(
        default=True, description="Test provider connectivity during validation"
    )

    # Hot-reload settings
    hot_reload_enabled: bool = Field(default=True, description="Enable configuration hot-reload")
    config_file_watching: bool = Field(default=False, description="Watch config files for changes")

    # Rate limiting and timeouts
    default_timeout_seconds: int = Field(default=60, ge=5, le=300)
    default_max_retries: int = Field(default=3, ge=0, le=10)

    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_api_keys: bool = Field(
        default=False, description="Log API keys (NEVER enable in production)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_name": "Chimera Backend",
                "version": "1.0.0",
                "environment": "development",
                "connection_mode": "direct",
                "providers": {
                    "openai": {
                        "provider_type": "openai",
                        "name": "openai",
                        "display_name": "OpenAI",
                        "api_key": "sk-...",
                        "base_url": "https://api.openai.com/v1",
                        "default_model": "gpt-4o",
                        "models": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                        "enabled": True,
                    }
                },
                "proxy_mode": {"enabled": False, "endpoint": "http://localhost:8080"},
                "encryption": {"enabled": True, "algorithm": "AES-256"},
            }
        }
    }


class ProviderTestRequest(BaseModel):
    """Request to test a provider configuration"""

    provider_name: str
    api_key: str | None = None
    base_url: str | None = None
    test_connectivity: bool = True
    test_model: str | None = None


class ProviderTestResponse(BaseModel):
    """Response from provider configuration test"""

    success: bool
    provider_name: str
    latency_ms: float | None = None
    error: str | None = None
    models_discovered: list[str] = Field(default_factory=list)
    tested_at: datetime = Field(default_factory=datetime.now)
    recommendations: list[str] = Field(default_factory=list)


class ConfigurationUpdateRequest(BaseModel):
    """Request to update configuration"""

    providers: dict[str, ProviderConnectionConfig] | None = None
    connection_mode: ConnectionMode | None = None
    proxy_mode: ProxyModeConfig | None = None
    encryption: EncryptionConfig | None = None
    reload_immediately: bool = Field(default=True, description="Apply changes immediately")
    validate_before_apply: bool = Field(
        default=True, description="Validate configuration before applying"
    )
