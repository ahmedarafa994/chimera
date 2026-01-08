"""
Synchronization Domain Models

Models for provider/model synchronization between backend and frontend including:
- Sync state and metadata
- Version tracking for conflict resolution
- Event types for real-time updates
- Error handling and recovery
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SyncEventType(str, Enum):
    """Types of synchronization events"""

    # Full sync events
    FULL_SYNC = "full_sync"
    INITIAL_STATE = "initial_state"

    # Provider events
    PROVIDER_ADDED = "provider_added"
    PROVIDER_UPDATED = "provider_updated"
    PROVIDER_REMOVED = "provider_removed"
    PROVIDER_STATUS_CHANGED = "provider_status_changed"
    PROVIDER_HEALTH_UPDATED = "provider_health_updated"

    # Model events
    MODEL_ADDED = "model_added"
    MODEL_UPDATED = "model_updated"
    MODEL_REMOVED = "model_removed"
    MODEL_DEPRECATED = "model_deprecated"

    # Active selection events
    ACTIVE_PROVIDER_CHANGED = "active_provider_changed"
    ACTIVE_MODEL_CHANGED = "active_model_changed"

    # System events
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    RECONNECT_REQUIRED = "reconnect_required"


class SyncStatus(str, Enum):
    """Synchronization status"""

    SYNCED = "synced"
    SYNCING = "syncing"
    STALE = "stale"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class ModelDeprecationStatus(str, Enum):
    """Model deprecation status"""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"  # Will be removed soon
    REMOVED = "removed"


class ModelSpecification(BaseModel):
    """Detailed model specification for sync"""

    id: str
    name: str
    provider_id: str
    description: str | None = None

    # Context and token limits
    context_window: int = 4096
    max_input_tokens: int = 4096
    max_output_tokens: int = 4096

    # Capabilities
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_system_prompt: bool = True
    supports_tools: bool = False

    # Pricing (per 1K tokens)
    input_price_per_1k: float | None = None
    output_price_per_1k: float | None = None
    pricing_tier: str = "standard"

    # Status
    is_default: bool = False
    is_available: bool = True
    deprecation_status: ModelDeprecationStatus = ModelDeprecationStatus.ACTIVE
    deprecation_date: datetime | None = None
    sunset_date: datetime | None = None
    replacement_model_id: str | None = None

    # Metadata
    version: str = "1.0"
    release_date: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(protected_namespaces=())


class ProviderSyncInfo(BaseModel):
    """Provider information optimized for sync"""

    id: str
    provider_type: str
    name: str
    display_name: str

    # Status
    status: str = "initializing"
    is_available: bool = True
    enabled: bool = True
    is_default: bool = False
    is_fallback: bool = False

    # Configuration
    priority: int = 100
    default_model_id: str | None = None

    # Capabilities summary
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_function_calling: bool = False
    max_context_length: int = 4096

    # Models
    models: list[ModelSpecification] = Field(default_factory=list)
    model_count: int = 0

    # Health
    health_status: str = "unknown"
    last_health_check: datetime | None = None
    avg_latency_ms: float | None = None
    error_rate: float | None = None

    # Sync metadata
    version: int = 1
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(protected_namespaces=())


class SyncMetadata(BaseModel):
    """Metadata for synchronization state"""

    version: int = 1
    last_sync: datetime = Field(default_factory=datetime.utcnow)
    last_full_sync: datetime | None = None
    sync_status: SyncStatus = SyncStatus.SYNCED

    # Checksums for conflict detection
    providers_checksum: str | None = None
    models_checksum: str | None = None

    # Server info
    server_version: str = "2.0.0"
    server_time: datetime = Field(default_factory=datetime.utcnow)


class SyncState(BaseModel):
    """Complete synchronization state"""

    metadata: SyncMetadata = Field(default_factory=SyncMetadata)

    # Provider data
    providers: list[ProviderSyncInfo] = Field(default_factory=list)
    provider_count: int = 0

    # Active selections
    active_provider_id: str | None = None
    active_model_id: str | None = None
    default_provider_id: str | None = None

    # All models (flattened for easy lookup)
    all_models: list[ModelSpecification] = Field(default_factory=list)
    model_count: int = 0

    # Deprecated models (for UI warnings)
    deprecated_models: list[str] = Field(default_factory=list)

    model_config = ConfigDict(protected_namespaces=())


class SyncEvent(BaseModel):
    """Event for real-time synchronization updates"""

    type: SyncEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1

    # Event data
    provider_id: str | None = None
    model_id: str | None = None
    data: dict[str, Any] | None = None

    # For incremental updates
    previous_version: int | None = None
    changes: dict[str, Any] | None = None

    # Error info
    error_code: str | None = None
    error_message: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class SyncRequest(BaseModel):
    """Request for sync operations"""

    client_version: int = 0
    last_sync: datetime | None = None
    force_full_sync: bool = False
    include_deprecated: bool = False
    provider_ids: list[str] | None = None  # Filter to specific providers


class SyncResponse(BaseModel):
    """Response for sync operations"""

    success: bool
    sync_type: str = "full"  # "full" or "incremental"
    state: SyncState | None = None
    events: list[SyncEvent] = Field(default_factory=list)

    # For incremental sync
    has_changes: bool = False
    changes_since_version: int | None = None

    # Error handling
    error: str | None = None
    retry_after_seconds: int | None = None


class ProviderAvailabilityInfo(BaseModel):
    """Provider availability information for UI"""

    provider_id: str
    is_available: bool
    status: str
    reason: str | None = None

    # Fallback info
    fallback_provider_id: str | None = None
    fallback_available: bool = False

    # Recovery info
    estimated_recovery: datetime | None = None
    retry_after_seconds: int | None = None


class ModelAvailabilityInfo(BaseModel):
    """Model availability information for UI"""

    model_id: str
    provider_id: str
    is_available: bool
    status: str
    reason: str | None = None

    # Deprecation info
    is_deprecated: bool = False
    deprecation_message: str | None = None
    replacement_model_id: str | None = None

    # Alternative suggestions
    alternative_models: list[str] = Field(default_factory=list)

    model_config = ConfigDict(protected_namespaces=())
