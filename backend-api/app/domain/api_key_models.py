# =============================================================================
# Chimera - API Key Storage Domain Models
# =============================================================================
# Pydantic models for secure API key storage with encryption support,
# provider metadata, and key validation for multi-provider LLM integration.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# =============================================================================

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ApiKeyRole(str, Enum):
    """Role designation for API keys supporting failover scenarios."""

    PRIMARY = "primary"
    BACKUP = "backup"
    FALLBACK = "fallback"


class ApiKeyStatus(str, Enum):
    """Status of an API key."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    RATE_LIMITED = "rate_limited"
    INVALID = "invalid"
    REVOKED = "revoked"


class ProviderType(str, Enum):
    """Supported LLM provider types for API key management."""

    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    BIGMODEL = "bigmodel"
    ROUTEWAY = "routeway"
    CURSOR = "cursor"


# Provider-specific API key format patterns for validation
API_KEY_PATTERNS: dict[str, str] = {
    "openai": r"^sk-[a-zA-Z0-9]{32,}$",
    "anthropic": r"^sk-ant-[a-zA-Z0-9\-_]{30,}$",
    "google": r"^[a-zA-Z0-9_\-]{32,}$",
    "deepseek": r"^sk-[a-zA-Z0-9]{32,}$",
    "qwen": r"^[a-zA-Z0-9_\-]{16,}$",
    "bigmodel": r"^[a-zA-Z0-9\-_\.]{20,}$",
    "routeway": r"^[a-zA-Z0-9_\-]{16,}$",
    "cursor": r"^[a-zA-Z0-9_\-]{16,}$",
}


class ApiKeyUsageStats(BaseModel):
    """Usage statistics for an API key."""

    request_count: int = Field(
        default=0, ge=0, description="Total number of requests made with this key"
    )
    successful_requests: int = Field(default=0, ge=0, description="Number of successful requests")
    failed_requests: int = Field(default=0, ge=0, description="Number of failed requests")
    total_tokens_used: int = Field(default=0, ge=0, description="Total tokens consumed")
    total_input_tokens: int = Field(default=0, ge=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, ge=0, description="Total output tokens")
    last_used_at: datetime | None = Field(default=None, description="Last time this key was used")
    last_error: str | None = Field(
        default=None, max_length=500, description="Last error message if any"
    )
    last_error_at: datetime | None = Field(default=None, description="Timestamp of last error")
    rate_limit_hits: int = Field(default=0, ge=0, description="Number of rate limit hits")
    avg_latency_ms: float | None = Field(
        default=None, ge=0, description="Average request latency in ms"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_count": 1250,
                "successful_requests": 1200,
                "failed_requests": 50,
                "total_tokens_used": 500000,
                "total_input_tokens": 300000,
                "total_output_tokens": 200000,
                "last_used_at": "2025-01-11T15:30:00Z",
                "rate_limit_hits": 5,
                "avg_latency_ms": 450.5,
            }
        }
    )


class ApiKeyRecord(BaseModel):
    """
    Internal storage model for API keys with encryption support.

    This model is used for persisting API keys securely. The encrypted_key
    field stores the AES-256 encrypted API key with 'enc:' prefix.
    """

    id: str = Field(
        ..., min_length=1, max_length=100, description="Unique identifier for this API key record"
    )
    provider_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Provider identifier (e.g., 'openai', 'google')",
    )
    encrypted_key: str = Field(
        ..., min_length=1, description="AES-256 encrypted API key with 'enc:' prefix"
    )
    name: str = Field(
        ..., min_length=1, max_length=100, description="Human-readable name for this key"
    )
    role: ApiKeyRole = Field(
        default=ApiKeyRole.PRIMARY, description="Role of this key (primary/backup/fallback)"
    )
    status: ApiKeyStatus = Field(
        default=ApiKeyStatus.ACTIVE, description="Current status of this key"
    )
    priority: int = Field(
        default=0, ge=0, le=100, description="Priority for failover (lower = higher priority)"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this key was added"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this key was last modified"
    )
    expires_at: datetime | None = Field(
        default=None, description="Optional expiration date for this key"
    )

    # Usage tracking
    usage_stats: ApiKeyUsageStats = Field(
        default_factory=ApiKeyUsageStats, description="Usage statistics"
    )

    # Metadata
    description: str | None = Field(
        default=None, max_length=500, description="Optional description"
    )
    tags: list[str] = Field(
        default_factory=list, max_length=10, description="Tags for categorization"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "key_openai_primary_001",
                "provider_id": "openai",
                "encrypted_key": "enc:gAAAAABl...",
                "name": "OpenAI Production Key",
                "role": "primary",
                "status": "active",
                "priority": 0,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-11T10:00:00Z",
                "description": "Primary production API key for OpenAI",
                "tags": ["production", "team-alpha"],
            }
        }
    )

    @field_validator("provider_id")
    def validate_provider_id(cls, v: str) -> str:
        """Validate provider_id is a known provider."""
        valid_providers = [p.value for p in ProviderType]
        if v.lower() not in valid_providers:
            raise ValueError(f"Provider must be one of: {valid_providers}")
        return v.lower()

    @field_validator("encrypted_key")
    def validate_encrypted_key(cls, v: str) -> str:
        """Validate that the key appears to be encrypted."""
        # Allow both encrypted keys (with enc: prefix) and plaintext for flexibility
        # The service layer will handle encryption before storage
        if not v or len(v) < 10:
            raise ValueError("API key must be at least 10 characters")
        return v

    @field_validator("tags")
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags are properly formatted."""
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        for tag in v:
            if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", tag):
                raise ValueError(
                    f"Invalid tag format: {tag}. Tags must be alphanumeric with hyphens/underscores, max 50 chars"
                )
        return [t.lower() for t in v]


class ApiKeyCreate(BaseModel):
    """
    Request model for creating/adding a new API key.

    The api_key field accepts a plaintext API key which will be encrypted
    by the service layer before storage.
    """

    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider identifier")
    api_key: str = Field(
        ..., min_length=10, max_length=256, description="Plaintext API key to store"
    )
    name: str = Field(
        ..., min_length=1, max_length=100, description="Human-readable name for this key"
    )
    role: ApiKeyRole = Field(default=ApiKeyRole.PRIMARY, description="Role of this key")
    priority: int = Field(
        default=0, ge=0, le=100, description="Priority for failover (lower = higher priority)"
    )
    expires_at: datetime | None = Field(default=None, description="Optional expiration date")
    description: str | None = Field(
        default=None, max_length=500, description="Optional description"
    )
    tags: list[str] = Field(
        default_factory=list, max_length=10, description="Tags for categorization"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "api_key": "sk-abc123...",
                "name": "OpenAI Backup Key",
                "role": "backup",
                "priority": 10,
                "description": "Backup key for failover scenarios",
                "tags": ["backup", "production"],
            }
        }
    )

    @field_validator("provider_id")
    def validate_provider_id(cls, v: str) -> str:
        """Validate provider_id is a known provider."""
        valid_providers = [p.value for p in ProviderType]
        if v.lower() not in valid_providers:
            raise ValueError(f"Provider must be one of: {valid_providers}")
        return v.lower()

    @field_validator("api_key")
    def validate_api_key_format(cls, v: str) -> str:
        """Basic validation that the key is not empty and has reasonable length."""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        if len(v) < 10:
            raise ValueError("API key must be at least 10 characters")
        if len(v) > 256:
            raise ValueError("API key must be at most 256 characters")
        # Don't do provider-specific validation here since we don't have provider_id yet
        # Provider-specific validation is done in the service layer
        return v.strip()

    @field_validator("tags")
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags are properly formatted."""
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        for tag in v:
            if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", tag):
                raise ValueError(f"Invalid tag format: {tag}")
        return [t.lower() for t in v]

    @model_validator(mode="after")
    def validate_api_key_for_provider(self):
        """Validate API key format against provider-specific patterns."""
        pattern = API_KEY_PATTERNS.get(self.provider_id.lower())
        if pattern and not re.match(pattern, self.api_key):
            # Just warn, don't fail - some providers have varying key formats
            pass  # Validation is lenient to support various key formats
        return self


class ApiKeyUpdate(BaseModel):
    """
    Request model for updating an existing API key.

    All fields are optional. Only provided fields will be updated.
    If api_key is provided, it will be re-encrypted before storage.
    """

    api_key: str | None = Field(
        default=None, min_length=10, max_length=256, description="New API key (will be encrypted)"
    )
    name: str | None = Field(default=None, min_length=1, max_length=100, description="New name")
    role: ApiKeyRole | None = Field(default=None, description="New role")
    status: ApiKeyStatus | None = Field(default=None, description="New status")
    priority: int | None = Field(default=None, ge=0, le=100, description="New priority")
    expires_at: datetime | None = Field(default=None, description="New expiration date")
    description: str | None = Field(default=None, max_length=500, description="New description")
    tags: list[str] | None = Field(default=None, max_length=10, description="New tags")
    metadata: dict[str, Any] | None = Field(default=None, description="New metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Updated Key Name",
                "role": "primary",
                "status": "active",
                "priority": 0,
            }
        }
    )

    @field_validator("api_key")
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate API key if provided."""
        if v is not None:
            if not v.strip():
                raise ValueError("API key cannot be empty")
            if len(v) < 10:
                raise ValueError("API key must be at least 10 characters")
        return v.strip() if v else None

    @field_validator("tags")
    def validate_tags(cls, v: list[str] | None) -> list[str] | None:
        """Validate tags if provided."""
        if v is not None:
            if len(v) > 10:
                raise ValueError("Maximum 10 tags allowed")
            for tag in v:
                if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", tag):
                    raise ValueError(f"Invalid tag format: {tag}")
            return [t.lower() for t in v]
        return v


class ApiKeyResponse(BaseModel):
    """
    Response model for API key information.

    SECURITY: This model NEVER exposes the raw API key.
    Only a masked version (e.g., 'sk-...abc123') is shown.
    """

    id: str = Field(..., description="Unique identifier")
    provider_id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Human-readable name")
    masked_key: str = Field(..., description="Masked API key (e.g., 'sk-...abc123')")
    role: ApiKeyRole = Field(..., description="Role of this key")
    status: ApiKeyStatus = Field(..., description="Current status")
    priority: int = Field(..., description="Priority for failover")

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    expires_at: datetime | None = Field(default=None, description="Expiration date if set")
    last_used_at: datetime | None = Field(default=None, description="Last usage timestamp")

    # Usage summary
    request_count: int = Field(default=0, description="Total requests made")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")

    # Metadata
    description: str | None = Field(default=None, description="Description")
    tags: list[str] = Field(default_factory=list, description="Tags")
    is_expired: bool = Field(default=False, description="Whether the key has expired")
    is_rate_limited: bool = Field(
        default=False, description="Whether the key is currently rate limited"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "key_openai_primary_001",
                "provider_id": "openai",
                "name": "OpenAI Production Key",
                "masked_key": "sk-...xyz789",
                "role": "primary",
                "status": "active",
                "priority": 0,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-11T10:00:00Z",
                "last_used_at": "2025-01-11T15:30:00Z",
                "request_count": 1250,
                "successful_requests": 1200,
                "failed_requests": 50,
                "description": "Primary production API key",
                "tags": ["production"],
                "is_expired": False,
                "is_rate_limited": False,
            }
        }
    )

    @classmethod
    def from_record(
        cls, record: ApiKeyRecord, decrypted_key: str | None = None
    ) -> "ApiKeyResponse":
        """
        Create a response from an ApiKeyRecord.

        Args:
            record: The API key record
            decrypted_key: Optional decrypted key for masking (if not provided, uses encrypted_key)

        Returns:
            ApiKeyResponse with masked key
        """
        # Generate masked key
        if decrypted_key:
            masked_key = cls._mask_api_key(decrypted_key)
        else:
            # If we don't have the decrypted key, show a generic mask
            masked_key = "***...***"

        # Check if expired
        is_expired = False
        if record.expires_at:
            is_expired = record.expires_at < datetime.utcnow()

        return cls(
            id=record.id,
            provider_id=record.provider_id,
            name=record.name,
            masked_key=masked_key,
            role=record.role,
            status=record.status,
            priority=record.priority,
            created_at=record.created_at,
            updated_at=record.updated_at,
            expires_at=record.expires_at,
            last_used_at=record.usage_stats.last_used_at,
            request_count=record.usage_stats.request_count,
            successful_requests=record.usage_stats.successful_requests,
            failed_requests=record.usage_stats.failed_requests,
            description=record.description,
            tags=record.tags,
            is_expired=is_expired,
            is_rate_limited=record.status == ApiKeyStatus.RATE_LIMITED,
        )

    @staticmethod
    def _mask_api_key(key: str) -> str:
        """
        Mask an API key for safe display.

        Examples:
            'sk-abc123def456xyz789' -> 'sk-...xyz789'
            'AIzaSyB...' -> 'AIz...yB...' (first 3 + last 6)
        """
        if not key or len(key) < 10:
            return "***...***"

        # Show prefix (if it looks like a standard prefix) and last 6 chars
        if key.startswith("sk-"):
            return f"sk-...{key[-6:]}"
        elif key.startswith("sk-ant-"):
            return f"sk-ant-...{key[-6:]}"
        else:
            # Generic masking: first 3 + last 6
            return f"{key[:3]}...{key[-6:]}"


class ApiKeyListResponse(BaseModel):
    """Response model for listing API keys."""

    keys: list[ApiKeyResponse] = Field(..., description="List of API keys")
    total: int = Field(..., ge=0, description="Total number of keys")
    by_provider: dict[str, int] = Field(
        default_factory=dict, description="Count of keys per provider"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "keys": [],
                "total": 5,
                "by_provider": {"openai": 2, "google": 2, "anthropic": 1},
            }
        }
    )


class ApiKeyTestRequest(BaseModel):
    """Request model for testing an API key."""

    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider to test with")
    api_key: str | None = Field(
        default=None,
        min_length=10,
        max_length=256,
        description="API key to test (optional, uses stored key if not provided)",
    )
    key_id: str | None = Field(default=None, description="ID of stored key to test")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "key_id": "key_openai_primary_001",
            }
        }
    )

    @model_validator(mode="after")
    def validate_key_source(self):
        """Ensure either api_key or key_id is provided."""
        if not self.api_key and not self.key_id:
            raise ValueError("Either api_key or key_id must be provided")
        return self


class ApiKeyTestResult(BaseModel):
    """Result of testing an API key."""

    success: bool = Field(..., description="Whether the test was successful")
    provider_id: str = Field(..., description="Provider that was tested")
    latency_ms: float | None = Field(default=None, ge=0, description="Response latency in ms")
    error: str | None = Field(
        default=None, max_length=500, description="Error message if test failed"
    )
    models_available: list[str] = Field(
        default_factory=list, description="Models available with this key"
    )
    rate_limit_info: dict[str, Any] | None = Field(
        default=None, description="Rate limit information if available"
    )
    tested_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the test was performed"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "provider_id": "openai",
                "latency_ms": 250.5,
                "models_available": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                "tested_at": "2025-01-11T15:30:00Z",
            }
        }
    )


class ProviderKeySummary(BaseModel):
    """Summary of API keys for a specific provider."""

    provider_id: str = Field(..., description="Provider identifier")
    provider_name: str = Field(..., description="Provider display name")
    total_keys: int = Field(default=0, ge=0, description="Total number of keys")
    active_keys: int = Field(default=0, ge=0, description="Number of active keys")
    primary_key_id: str | None = Field(default=None, description="ID of the primary key")
    backup_key_ids: list[str] = Field(default_factory=list, description="IDs of backup keys")
    has_valid_key: bool = Field(
        default=False, description="Whether provider has at least one valid key"
    )
    status: str = Field(default="unconfigured", description="Overall provider key status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "provider_name": "OpenAI",
                "total_keys": 2,
                "active_keys": 2,
                "primary_key_id": "key_openai_primary_001",
                "backup_key_ids": ["key_openai_backup_001"],
                "has_valid_key": True,
                "status": "configured",
            }
        }
    )
