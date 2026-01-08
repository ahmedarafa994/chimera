"""
Provider Synchronization Service

Handles synchronization of provider and model configurations between
backend and frontend with:
- Full and incremental sync support
- Version-based conflict resolution
- Event broadcasting for real-time updates
- Checksum-based change detection
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from app.domain.provider_models import (
    ModelInfo,
    ProviderInfo,
    ProviderStatus,
)
from app.domain.sync_models import (
    ModelAvailabilityInfo,
    ModelDeprecationStatus,
    ModelSpecification,
    ProviderAvailabilityInfo,
    ProviderSyncInfo,
    SyncEvent,
    SyncEventType,
    SyncMetadata,
    SyncRequest,
    SyncResponse,
    SyncState,
    SyncStatus,
)

logger = logging.getLogger(__name__)


class ProviderSyncService:
    """
    Service for synchronizing provider and model configurations.

    Provides:
    - Full sync for initial load
    - Incremental sync for updates
    - Event broadcasting for real-time updates
    - Version tracking for conflict resolution
    """

    def __init__(self):
        self._version = 1
        self._last_update = datetime.utcnow()
        self._event_callbacks: list[Callable[[SyncEvent], Any]] = []
        self._sync_state: SyncState | None = None
        self._lock = asyncio.Lock()

    @property
    def version(self) -> int:
        return self._version

    def register_event_callback(self, callback: Callable[[SyncEvent], Any]) -> None:
        """Register a callback for sync events"""
        if callback not in self._event_callbacks:
            self._event_callbacks.append(callback)

    def unregister_event_callback(self, callback: Callable[[SyncEvent], Any]) -> None:
        """Unregister an event callback"""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    async def _broadcast_event(self, event: SyncEvent) -> None:
        """Broadcast an event to all registered callbacks"""
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in sync event callback: {e}")

    def _compute_checksum(self, data: Any) -> str:
        """Compute a checksum for change detection"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()

    def _convert_model_to_spec(self, model: ModelInfo, provider_id: str) -> ModelSpecification:
        """Convert ModelInfo to ModelSpecification"""
        return ModelSpecification(
            id=model.id,
            name=model.name,
            provider_id=provider_id,
            description=model.description,
            context_window=model.max_tokens,
            max_input_tokens=model.max_tokens,
            max_output_tokens=model.max_output_tokens,
            supports_streaming=model.supports_streaming,
            supports_vision=model.supports_vision,
            supports_function_calling=model.supports_function_calling,
            input_price_per_1k=model.cost_per_1k_input,
            output_price_per_1k=model.cost_per_1k_output,
            pricing_tier=model.tier.value,
            is_default=model.is_default,
            is_available=True,
            deprecation_status=ModelDeprecationStatus.ACTIVE,
        )

    def _convert_provider_to_sync_info(self, provider: ProviderInfo) -> ProviderSyncInfo:
        """Convert ProviderInfo to ProviderSyncInfo"""
        models = [self._convert_model_to_spec(m, provider.id) for m in provider.models]

        return ProviderSyncInfo(
            id=provider.id,
            provider_type=provider.provider_type.value,
            name=provider.name,
            display_name=provider.display_name,
            status=provider.status.value if provider.status else "initializing",
            is_available=provider.status == ProviderStatus.AVAILABLE,
            enabled=provider.enabled,
            is_default=provider.is_default,
            is_fallback=provider.is_fallback,
            priority=provider.priority,
            default_model_id=provider.default_model,
            supports_streaming=provider.capabilities.supports_streaming,
            supports_vision=provider.capabilities.supports_vision,
            supports_function_calling=provider.capabilities.supports_function_calling,
            max_context_length=provider.capabilities.max_context_length,
            models=models,
            model_count=len(models),
            health_status=provider.health.status.value if provider.health else "unknown",
            last_health_check=provider.health.last_check if provider.health else None,
            avg_latency_ms=provider.health.avg_latency_ms if provider.health else None,
            error_rate=(
                provider.health.error_count
                / max(provider.health.success_count + provider.health.error_count, 1)
                * 100
                if provider.health
                else None
            ),
            version=self._version,
            updated_at=provider.updated_at or datetime.utcnow(),
        )

    async def get_full_sync_state(
        self,
        providers: list[ProviderInfo],
        active_provider_id: str | None = None,
        active_model_id: str | None = None,
        default_provider_id: str | None = None,
    ) -> SyncState:
        """Build complete sync state from provider list"""
        async with self._lock:
            # Convert providers
            sync_providers = [self._convert_provider_to_sync_info(p) for p in providers]

            # Flatten all models
            all_models: list[ModelSpecification] = []
            deprecated_models: list[str] = []

            for provider in sync_providers:
                for model in provider.models:
                    all_models.append(model)
                    if model.deprecation_status != ModelDeprecationStatus.ACTIVE:
                        deprecated_models.append(model.id)

            # Compute checksums
            providers_data = [p.model_dump() for p in sync_providers]
            models_data = [m.model_dump() for m in all_models]

            metadata = SyncMetadata(
                version=self._version,
                last_sync=datetime.utcnow(),
                last_full_sync=datetime.utcnow(),
                sync_status=SyncStatus.SYNCED,
                providers_checksum=self._compute_checksum(providers_data),
                models_checksum=self._compute_checksum(models_data),
                server_version="2.0.0",
                server_time=datetime.utcnow(),
            )

            state = SyncState(
                metadata=metadata,
                providers=sync_providers,
                provider_count=len(sync_providers),
                active_provider_id=active_provider_id,
                active_model_id=active_model_id,
                default_provider_id=default_provider_id,
                all_models=all_models,
                model_count=len(all_models),
                deprecated_models=deprecated_models,
            )

            self._sync_state = state
            return state

    async def handle_sync_request(
        self,
        request: SyncRequest,
        providers: list[ProviderInfo],
        active_provider_id: str | None = None,
        active_model_id: str | None = None,
        default_provider_id: str | None = None,
    ) -> SyncResponse:
        """Handle a sync request from the frontend"""
        try:
            # Check if full sync is needed
            needs_full_sync = (
                request.force_full_sync
                or request.client_version == 0
                or request.client_version < self._version - 10  # Too far behind
            )

            if needs_full_sync:
                state = await self.get_full_sync_state(
                    providers=providers,
                    active_provider_id=active_provider_id,
                    active_model_id=active_model_id,
                    default_provider_id=default_provider_id,
                )

                return SyncResponse(
                    success=True,
                    sync_type="full",
                    state=state,
                    has_changes=True,
                )

            # Incremental sync - check for changes since client version
            if self._sync_state and request.client_version >= self._version:
                # Client is up to date
                return SyncResponse(
                    success=True,
                    sync_type="incremental",
                    has_changes=False,
                    changes_since_version=request.client_version,
                )

            # Client needs updates - send full state for simplicity
            # In production, you'd track individual changes
            state = await self.get_full_sync_state(
                providers=providers,
                active_provider_id=active_provider_id,
                active_model_id=active_model_id,
                default_provider_id=default_provider_id,
            )

            return SyncResponse(
                success=True,
                sync_type="full",
                state=state,
                has_changes=True,
                changes_since_version=request.client_version,
            )

        except Exception as e:
            logger.error(f"Sync request failed: {e}")
            return SyncResponse(
                success=False,
                error=str(e),
                retry_after_seconds=5,
            )

    async def notify_provider_added(self, provider: ProviderInfo) -> None:
        """Notify clients that a provider was added"""
        async with self._lock:
            self._version += 1
            self._last_update = datetime.utcnow()

        sync_info = self._convert_provider_to_sync_info(provider)

        event = SyncEvent(
            type=SyncEventType.PROVIDER_ADDED,
            version=self._version,
            provider_id=provider.id,
            data=sync_info.model_dump(),
        )

        await self._broadcast_event(event)

    async def notify_provider_updated(self, provider: ProviderInfo) -> None:
        """Notify clients that a provider was updated"""
        async with self._lock:
            self._version += 1
            self._last_update = datetime.utcnow()

        sync_info = self._convert_provider_to_sync_info(provider)

        event = SyncEvent(
            type=SyncEventType.PROVIDER_UPDATED,
            version=self._version,
            provider_id=provider.id,
            data=sync_info.model_dump(),
        )

        await self._broadcast_event(event)

    async def notify_provider_removed(self, provider_id: str) -> None:
        """Notify clients that a provider was removed"""
        async with self._lock:
            self._version += 1
            self._last_update = datetime.utcnow()

        event = SyncEvent(
            type=SyncEventType.PROVIDER_REMOVED,
            version=self._version,
            provider_id=provider_id,
        )

        await self._broadcast_event(event)

    async def notify_provider_status_changed(
        self, provider_id: str, new_status: str, old_status: str | None = None
    ) -> None:
        """Notify clients that a provider's status changed"""
        async with self._lock:
            self._version += 1
            self._last_update = datetime.utcnow()

        event = SyncEvent(
            type=SyncEventType.PROVIDER_STATUS_CHANGED,
            version=self._version,
            provider_id=provider_id,
            data={
                "new_status": new_status,
                "old_status": old_status,
            },
        )

        await self._broadcast_event(event)

    async def notify_active_provider_changed(
        self,
        new_provider_id: str,
        new_model_id: str | None = None,
        old_provider_id: str | None = None,
    ) -> None:
        """Notify clients that the active provider changed"""
        async with self._lock:
            self._version += 1
            self._last_update = datetime.utcnow()

        event = SyncEvent(
            type=SyncEventType.ACTIVE_PROVIDER_CHANGED,
            version=self._version,
            provider_id=new_provider_id,
            model_id=new_model_id,
            data={
                "previous_provider_id": old_provider_id,
            },
        )

        await self._broadcast_event(event)

    async def notify_model_deprecated(
        self,
        model_id: str,
        provider_id: str,
        deprecation_date: datetime | None = None,
        replacement_model_id: str | None = None,
    ) -> None:
        """Notify clients that a model has been deprecated"""
        async with self._lock:
            self._version += 1
            self._last_update = datetime.utcnow()

        event = SyncEvent(
            type=SyncEventType.MODEL_DEPRECATED,
            version=self._version,
            provider_id=provider_id,
            model_id=model_id,
            data={
                "deprecation_date": deprecation_date.isoformat() if deprecation_date else None,
                "replacement_model_id": replacement_model_id,
            },
        )

        await self._broadcast_event(event)

    def get_provider_availability(
        self,
        provider: ProviderInfo,
        fallback_provider: ProviderInfo | None = None,
    ) -> ProviderAvailabilityInfo:
        """Get availability information for a provider"""
        is_available = provider.enabled and provider.status == ProviderStatus.AVAILABLE

        reason = None
        if not provider.enabled:
            reason = "Provider is disabled"
        elif provider.status == ProviderStatus.UNAVAILABLE:
            reason = "Provider is unavailable"
        elif provider.status == ProviderStatus.RATE_LIMITED:
            reason = "Provider is rate limited"
        elif provider.status == ProviderStatus.ERROR:
            reason = provider.health.last_error if provider.health else "Unknown error"

        fallback_available = False
        fallback_id = None
        if fallback_provider:
            fallback_available = (
                fallback_provider.enabled and fallback_provider.status == ProviderStatus.AVAILABLE
            )
            fallback_id = fallback_provider.id

        return ProviderAvailabilityInfo(
            provider_id=provider.id,
            is_available=is_available,
            status=provider.status.value if provider.status else "unknown",
            reason=reason,
            fallback_provider_id=fallback_id,
            fallback_available=fallback_available,
            retry_after_seconds=60 if provider.status == ProviderStatus.RATE_LIMITED else None,
        )

    def get_model_availability(
        self,
        model: ModelSpecification,
        provider_available: bool = True,
        alternative_models: list[str] | None = None,
    ) -> ModelAvailabilityInfo:
        """Get availability information for a model"""
        is_available = (
            model.is_available
            and provider_available
            and model.deprecation_status == ModelDeprecationStatus.ACTIVE
        )

        reason = None
        if not provider_available:
            reason = "Provider is unavailable"
        elif not model.is_available:
            reason = "Model is not available"
        elif model.deprecation_status == ModelDeprecationStatus.DEPRECATED:
            reason = "Model is deprecated"
        elif model.deprecation_status == ModelDeprecationStatus.SUNSET:
            reason = "Model will be removed soon"
        elif model.deprecation_status == ModelDeprecationStatus.REMOVED:
            reason = "Model has been removed"

        deprecation_message = None
        if model.deprecation_status != ModelDeprecationStatus.ACTIVE:
            if model.sunset_date:
                deprecation_message = (
                    f"This model will be removed on {model.sunset_date.strftime('%Y-%m-%d')}"
                )
            elif model.replacement_model_id:
                deprecation_message = f"Please migrate to {model.replacement_model_id}"
            else:
                deprecation_message = "This model is deprecated"

        return ModelAvailabilityInfo(
            model_id=model.id,
            provider_id=model.provider_id,
            is_available=is_available,
            status=model.deprecation_status.value,
            reason=reason,
            is_deprecated=model.deprecation_status != ModelDeprecationStatus.ACTIVE,
            deprecation_message=deprecation_message,
            replacement_model_id=model.replacement_model_id,
            alternative_models=alternative_models or [],
        )

    async def send_heartbeat(self) -> None:
        """Send a heartbeat event to all clients"""
        event = SyncEvent(
            type=SyncEventType.HEARTBEAT,
            version=self._version,
            data={
                "server_time": datetime.utcnow().isoformat(),
                "last_update": self._last_update.isoformat(),
            },
        )

        await self._broadcast_event(event)


# Global singleton instance
provider_sync_service = ProviderSyncService()
