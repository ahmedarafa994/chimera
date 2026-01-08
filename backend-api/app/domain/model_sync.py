"""
Domain models for model synchronization system.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ModelStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    UNAVAILABLE = "unavailable"


class ModelInfo(BaseModel):
    """Model information for API responses."""

    id: str = Field(..., description="Canonical model identifier")
    name: str = Field(..., description="Display name for the model")
    description: str = Field(..., description="Model description")
    capabilities: list[str] = Field(default_factory=list, description="Model capabilities")
    max_tokens: int = Field(..., description="Maximum token limit")
    is_active: bool = Field(..., description="Whether model is currently active")
    provider: str = Field(..., description="Provider identifier")
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AvailableModelsResponse(BaseModel):
    """Response for available models endpoint."""

    models: list[ModelInfo] = Field(default_factory=list)
    count: int = Field(..., description="Total number of models")
    timestamp: str = Field(..., description="Response timestamp")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")


class ModelSelectionRequest(BaseModel):
    """Request model selection payload."""

    model_id: str = Field(..., description="Selected model identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp")

    model_config = ConfigDict(protected_namespaces=())


class ModelSelectionResponse(BaseModel):
    """Response for model selection endpoint."""

    success: bool = Field(..., description="Selection success status")
    model_id: str = Field(..., description="Selected model ID")
    message: str = Field(..., description="Response message")
    timestamp: str = Field(..., description="Response timestamp")
    fallback_model: str | None = Field(None, description="Fallback model if selection failed")

    model_config = ConfigDict(protected_namespaces=())


class UserSession(BaseModel):
    """User session model for database."""

    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    selected_model_id: str | None = Field(None, description="Selected model ID")
    last_updated: datetime = Field(..., description="Last update timestamp")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str | None = Field(None, description="Client user agent")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(None, description="Session expiration")


class ModelChangeLog(BaseModel):
    """Model change audit log."""

    id: str | None = None
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    old_model_id: str | None = Field(None, description="Previous model ID")
    new_model_id: str = Field(..., description="New model ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str | None = Field(None, description="Client user agent")
    success: bool = Field(..., description="Whether change was successful")
    error_message: str | None = Field(None, description="Error message if failed")


class ModelAvailabilityUpdate(BaseModel):
    """WebSocket update for model availability."""

    type: str = Field(default="model_availability", description="Update type")
    model_id: str = Field(..., description="Model identifier")
    is_available: bool = Field(..., description="Model availability status")
    timestamp: str = Field(..., description="Update timestamp")
    message: str | None = Field(None, description="Status message")

    model_config = ConfigDict(protected_namespaces=())


class ModelSyncError(BaseModel):
    """Standardized error response for model sync operations."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error description")
    fallback_model: str | None = Field(None, description="Fallback model suggestion")
    retry_after: int | None = Field(None, description="Retry delay in seconds")
    timestamp: str = Field(..., description="Error timestamp")


class ModelValidationError(Exception):
    """Custom exception for model validation errors."""

    def __init__(self, message: str, model_id: str | None = None):
        self.message = message
        self.model_id = model_id
        super().__init__(message)


# Database table definitions (SQLAlchemy models would go here)
# These would be defined in a separate models.py file for SQLAlchemy
