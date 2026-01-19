"""
Provider Management Service

Comprehensive service for dynamic AI provider management including:
- Runtime provider registration and deregistration
- Provider health monitoring and status tracking
- Secure API key storage and encryption
- Provider fallback and routing logic
- Real-time provider status broadcasting

AIConfigManager Integration:
- Config-driven provider listings
- Validation against config definitions
- Database state sync with config state
"""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from base64 import urlsafe_b64encode
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional

from cryptography.fernet import Fernet

from app.core.config import settings
from app.domain.interfaces import LLMProvider
from app.domain.provider_models import (
    ProviderCapabilities,
    ProviderHealthInfo,
    ProviderInfo,
    ProviderListResponse,
    ProviderRegistration,
    ProviderRoutingConfig,
    ProviderSelectionRequest,
    ProviderSelectionResponse,
    ProviderStatus,
    ProviderTestResult,
    ProviderType,
    ProviderUpdate,
)

logger = logging.getLogger(__name__)


def _get_config_manager():
    """
    Get AIConfigManager instance with graceful fallback.

    Returns None if config manager is not available.
    """
    try:
        from app.core.service_registry import get_ai_config_manager

        config_manager = get_ai_config_manager()
        if config_manager.is_loaded():
            return config_manager
    except Exception as e:
        logger.debug(f"Config manager not available: {e}")
    return None


class ProviderEncryption:
    """Handles secure encryption/decryption of API keys"""

    def __init__(self, secret_key: str | None = None):
        # Use provided key or derive from settings
        # Fallback chain: explicit key -> env var -> CHIMERA_API_KEY -> GOOGLE_API_KEY -> default
        key = (
            secret_key
            or os.getenv("PROVIDER_ENCRYPTION_KEY")
            or settings.CHIMERA_API_KEY
            or settings.GOOGLE_API_KEY
            or "chimera-default-encryption-key-change-in-production"
        )
        # Derive a valid Fernet key from the secret
        derived_key = hashlib.sha256(key.encode()).digest()
        self._fernet = Fernet(urlsafe_b64encode(derived_key))

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string value"""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt an encrypted string"""
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def mask_key(self, api_key: str) -> str:
        """Return a masked version of the API key for display"""
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


class ProviderHealthMonitor:
    """Monitors provider health and manages circuit breakers"""

    def __init__(
        self,
        check_interval: int = 30,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
    ):
        self.check_interval = check_interval
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self._health_data: dict[str, ProviderHealthInfo] = {}
        self._circuit_breakers: dict[str, dict] = {}
        self._check_tasks: dict[str, asyncio.Task] = {}
        self._callbacks: list[Callable] = []

    def register_callback(self, callback: Callable):
        """Register a callback for health status changes"""
        self._callbacks.append(callback)

    async def _notify_callbacks(self, provider_id: str, health: ProviderHealthInfo):
        """Notify all registered callbacks of health changes"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(provider_id, health)
                else:
                    callback(provider_id, health)
            except Exception as e:
                logger.error(f"Health callback error: {e}")

    def get_health(self, provider_id: str) -> ProviderHealthInfo:
        """Get current health info for a provider"""
        return self._health_data.get(provider_id, ProviderHealthInfo())

    def is_circuit_open(self, provider_id: str) -> bool:
        """Check if circuit breaker is open (provider should not be used)"""
        breaker = self._circuit_breakers.get(provider_id)
        if not breaker:
            return False

        if breaker["state"] == "open":
            # Check if timeout has passed
            if time.time() - breaker["opened_at"] > self.circuit_breaker_timeout:
                breaker["state"] = "half-open"
                return False
            return True
        return False

    def record_success(self, provider_id: str, latency_ms: float):
        """Record a successful provider call"""
        health = self._health_data.setdefault(provider_id, ProviderHealthInfo())
        health.success_count += 1
        health.last_success = datetime.utcnow()
        health.status = ProviderStatus.AVAILABLE

        # Update average latency
        if health.avg_latency_ms is None:
            health.avg_latency_ms = latency_ms
        else:
            # Exponential moving average
            health.avg_latency_ms = 0.9 * health.avg_latency_ms + 0.1 * latency_ms

        # Reset circuit breaker on success
        if provider_id in self._circuit_breakers:
            self._circuit_breakers[provider_id]["failure_count"] = 0
            self._circuit_breakers[provider_id]["state"] = "closed"

    def record_failure(self, provider_id: str, error: str):
        """Record a failed provider call"""
        health = self._health_data.setdefault(provider_id, ProviderHealthInfo())
        health.error_count += 1
        health.last_error = error
        health.last_check = datetime.utcnow()

        # Update circuit breaker
        breaker = self._circuit_breakers.setdefault(
            provider_id, {"failure_count": 0, "state": "closed", "opened_at": None}
        )
        breaker["failure_count"] += 1

        if breaker["failure_count"] >= self.circuit_breaker_threshold:
            breaker["state"] = "open"
            breaker["opened_at"] = time.time()
            health.status = ProviderStatus.UNAVAILABLE
            logger.warning(f"Circuit breaker opened for provider {provider_id}")
        elif breaker["failure_count"] >= self.circuit_breaker_threshold // 2:
            health.status = ProviderStatus.DEGRADED

    async def start_monitoring(self, provider_id: str, check_func: Callable):
        """Start periodic health monitoring for a provider"""
        if provider_id in self._check_tasks:
            return

        async def monitor_loop():
            while True:
                try:
                    start = time.time()
                    is_healthy = await check_func()
                    latency = (time.time() - start) * 1000

                    if is_healthy:
                        self.record_success(provider_id, latency)
                    else:
                        self.record_failure(provider_id, "Health check returned false")

                    health = self.get_health(provider_id)
                    health.last_check = datetime.utcnow()
                    await self._notify_callbacks(provider_id, health)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.record_failure(provider_id, str(e))
                    logger.error(f"Health check failed for {provider_id}: {e}")

                await asyncio.sleep(self.check_interval)

        self._check_tasks[provider_id] = asyncio.create_task(monitor_loop())

    def stop_monitoring(self, provider_id: str):
        """Stop health monitoring for a provider"""
        if provider_id in self._check_tasks:
            self._check_tasks[provider_id].cancel()
            del self._check_tasks[provider_id]
        if provider_id in self._health_data:
            del self._health_data[provider_id]
        if provider_id in self._circuit_breakers:
            del self._circuit_breakers[provider_id]


class ProviderManagementService:
    """
    Central service for managing AI providers dynamically.

    Features:
    - Runtime provider registration/deregistration
    - Secure API key storage
    - Health monitoring and circuit breakers
    - Provider fallback and routing
    - Real-time status updates

    AIConfigManager Integration:
    - Config-driven provider listings
    - CRUD validation against config definitions
    - Database/config state synchronization
    """

    _instance: Optional["ProviderManagementService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._providers: dict[str, ProviderInfo] = {}
        self._provider_instances: dict[str, LLMProvider] = {}
        self._encrypted_keys: dict[str, str] = {}
        self._default_provider_id: str | None = None
        self._active_provider_id: str | None = None
        self._encryption = ProviderEncryption()
        self._health_monitor = ProviderHealthMonitor()
        self._routing_config = ProviderRoutingConfig()
        self._event_callbacks: list[Callable] = []
        self._lock = asyncio.Lock()
        self._initialized = True

        # Load config-driven settings
        self._sync_with_config()

        logger.info("ProviderManagementService initialized")

    def _sync_with_config(self) -> None:
        """
        Synchronize provider state with AIConfigManager configuration.

        This ensures the service reflects the current config state.
        """
        config_manager = _get_config_manager()
        if not config_manager:
            return

        try:
            config = config_manager.get_config()

            # Update health monitor settings from global config
            global_cfg = config.global_config
            self._health_monitor.check_interval = global_cfg.health_check_interval

            # Set default provider from config
            if not self._default_provider_id:
                self._default_provider_id = global_cfg.default_provider

            logger.info(
                f"ProviderManagementService synced with config: "
                f"default={global_cfg.default_provider}"
            )
        except Exception as e:
            logger.warning(f"Failed to sync with config: {e}")

    def register_event_callback(self, callback: Callable):
        """Register callback for provider events (add, remove, status change)"""
        self._event_callbacks.append(callback)

    async def _emit_event(self, event_type: str, data: dict):
        """Emit an event to all registered callbacks"""
        event = {"type": event_type, "data": data, "timestamp": datetime.utcnow().isoformat()}
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def _generate_provider_id(self, name: str, provider_type: ProviderType) -> str:
        """Generate a unique provider ID"""
        return f"{provider_type.value}-{name}-{uuid.uuid4().hex[:8]}"

    async def register_provider(
        self,
        registration: ProviderRegistration,
        provider_instance: LLMProvider | None = None,
    ) -> ProviderInfo:
        """
        Register a new provider dynamically.

        Args:
            registration: Provider registration details
            provider_instance: Optional pre-created provider instance

        Returns:
            ProviderInfo with the registered provider details

        Note:
            Validates against config if available. Logs warnings for
            providers not defined in config.
        """
        async with self._lock:
            # Check for duplicate name
            for existing in self._providers.values():
                if existing.name == registration.name:
                    raise ValueError(f"Provider with name '{registration.name}' exists")

            # Validate against config
            self._validate_against_config(
                registration.name,
                registration.provider_type,
            )

            provider_id = self._generate_provider_id(registration.name, registration.provider_type)

            # Encrypt and store API key
            encrypted_key = self._encryption.encrypt(registration.api_key)
            self._encrypted_keys[provider_id] = encrypted_key

            # Create provider info
            provider_info = ProviderInfo(
                id=provider_id,
                provider_type=registration.provider_type,
                name=registration.name,
                display_name=registration.display_name or registration.name.title(),
                status=ProviderStatus.INITIALIZING,
                enabled=registration.enabled,
                is_default=False,
                is_fallback=registration.is_fallback,
                priority=registration.priority,
                default_model=registration.default_model,
                models=[],
                capabilities=ProviderCapabilities(),
                health=ProviderHealthInfo(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=registration.metadata,
            )

            # Store provider instance if provided
            if provider_instance:
                self._provider_instances[provider_id] = provider_instance

            self._providers[provider_id] = provider_info

            # Set as default if first provider or explicitly requested
            if not self._default_provider_id:
                self._default_provider_id = provider_id
                provider_info.is_default = True

            # Start health monitoring if we have an instance
            if provider_instance:
                await self._health_monitor.start_monitoring(
                    provider_id, provider_instance.check_health
                )

            logger.info(f"Registered provider: {provider_id} ({registration.name})")

            await self._emit_event(
                "provider_registered",
                {
                    "provider_id": provider_id,
                    "name": registration.name,
                    "type": registration.provider_type.value,
                },
            )

            return provider_info

    async def deregister_provider(self, provider_id: str) -> bool:
        """
        Remove a provider from the system.

        Args:
            provider_id: ID of the provider to remove

        Returns:
            True if successfully removed
        """
        async with self._lock:
            if provider_id not in self._providers:
                raise ValueError(f"Provider '{provider_id}' not found")

            provider = self._providers[provider_id]

            # Stop health monitoring
            self._health_monitor.stop_monitoring(provider_id)

            # Remove from all registries
            del self._providers[provider_id]
            self._encrypted_keys.pop(provider_id, None)
            self._provider_instances.pop(provider_id, None)

            # Update default if needed
            if self._default_provider_id == provider_id:
                self._default_provider_id = next(iter(self._providers.keys()), None)
                if self._default_provider_id:
                    self._providers[self._default_provider_id].is_default = True

            # Update active if needed
            if self._active_provider_id == provider_id:
                self._active_provider_id = self._default_provider_id

            logger.info(f"Deregistered provider: {provider_id}")

            await self._emit_event(
                "provider_deregistered", {"provider_id": provider_id, "name": provider.name}
            )

            return True

    async def update_provider(self, provider_id: str, update: ProviderUpdate) -> ProviderInfo:
        """
        Update an existing provider's configuration.

        Args:
            provider_id: ID of the provider to update
            update: Fields to update

        Returns:
            Updated ProviderInfo
        """
        async with self._lock:
            if provider_id not in self._providers:
                raise ValueError(f"Provider '{provider_id}' not found")

            provider = self._providers[provider_id]

            # Update fields if provided
            if update.display_name is not None:
                provider.display_name = update.display_name
            if update.api_key is not None:
                self._encrypted_keys[provider_id] = self._encryption.encrypt(update.api_key)
            if update.default_model is not None:
                provider.default_model = update.default_model
            if update.priority is not None:
                provider.priority = update.priority
            if update.enabled is not None:
                provider.enabled = update.enabled
            if update.is_fallback is not None:
                provider.is_fallback = update.is_fallback
            if update.metadata is not None:
                provider.metadata.update(update.metadata)

            provider.updated_at = datetime.utcnow()

            logger.info(f"Updated provider: {provider_id}")

            await self._emit_event(
                "provider_updated", {"provider_id": provider_id, "name": provider.name}
            )

            return provider

    def get_provider(self, provider_id: str) -> ProviderInfo | None:
        """Get provider info by ID"""
        return self._providers.get(provider_id)

    def get_provider_by_name(self, name: str) -> ProviderInfo | None:
        """Get provider info by name"""
        for provider in self._providers.values():
            if provider.name == name:
                return provider
        return None

    def get_provider_instance(self, provider_id: str) -> LLMProvider | None:
        """Get the actual provider instance for making API calls"""
        return self._provider_instances.get(provider_id)

    def get_api_key(self, provider_id: str) -> str | None:
        """Get decrypted API key for a provider"""
        encrypted = self._encrypted_keys.get(provider_id)
        if encrypted:
            return self._encryption.decrypt(encrypted)
        return None

    def list_providers(
        self,
        enabled_only: bool = False,
        include_health: bool = True,
        include_config_providers: bool = True,
    ) -> ProviderListResponse:
        """
        List all registered providers.

        Args:
            enabled_only: Only return enabled providers
            include_health: Include health information
            include_config_providers: Include providers from config

        Returns:
            ProviderListResponse with provider list
        """
        providers = list(self._providers.values())

        # Optionally include config-defined providers not yet registered
        if include_config_providers:
            config_providers = self._get_config_providers()
            for cfg_provider in config_providers:
                if not any(p.name == cfg_provider["name"] for p in providers):
                    # Create a stub ProviderInfo from config
                    providers.append(self._create_provider_info_from_config(cfg_provider))

        if enabled_only:
            providers = [p for p in providers if p.enabled]

        if include_health:
            for provider in providers:
                health = self._health_monitor.get_health(provider.id)
                provider.health = health

        # Sort by priority
        providers.sort(key=lambda p: p.priority)

        return ProviderListResponse(
            providers=providers,
            total=len(providers),
            default_provider_id=self._default_provider_id,
            active_provider_id=self._active_provider_id,
        )

    def _get_config_providers(self) -> list[dict[str, Any]]:
        """Get provider definitions from AIConfigManager."""
        config_manager = _get_config_manager()
        if not config_manager:
            return []

        try:
            config = config_manager.get_config()
            result = []
            for provider_id, provider_cfg in config.providers.items():
                result.append(
                    {
                        "id": provider_id,
                        "name": provider_cfg.name,
                        "type": provider_cfg.type.value,
                        "enabled": provider_cfg.enabled,
                        "priority": provider_cfg.priority,
                        "default_model": (
                            provider_cfg.get_default_model().model_id
                            if provider_cfg.get_default_model()
                            else None
                        ),
                        "models": list(provider_cfg.models.keys()),
                        "capabilities": {
                            "supports_streaming": (provider_cfg.capabilities.supports_streaming),
                            "supports_vision": (provider_cfg.capabilities.supports_vision),
                            "supports_function_calling": (
                                provider_cfg.capabilities.supports_function_calling
                            ),
                        },
                    }
                )
            return result
        except Exception as e:
            logger.warning(f"Failed to get config providers: {e}")
            return []

    def _create_provider_info_from_config(self, cfg: dict[str, Any]) -> ProviderInfo:
        """Create a ProviderInfo from config data."""
        # Map config type to ProviderType enum
        try:
            provider_type = ProviderType(cfg["type"])
        except ValueError:
            provider_type = ProviderType.CUSTOM

        return ProviderInfo(
            id=cfg["id"],
            provider_type=provider_type,
            name=cfg["name"],
            display_name=cfg["name"].title(),
            status=ProviderStatus.UNKNOWN,
            enabled=cfg["enabled"],
            is_default=(cfg["id"] == self._default_provider_id),
            is_fallback=False,
            priority=cfg["priority"],
            default_model=cfg.get("default_model"),
            models=[],
            capabilities=ProviderCapabilities(
                supports_streaming=cfg.get("capabilities", {}).get("supports_streaming", True),
                supports_vision=cfg.get("capabilities", {}).get("supports_vision", False),
                supports_function_calling=cfg.get("capabilities", {}).get(
                    "supports_function_calling", False
                ),
            ),
            health=ProviderHealthInfo(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={"source": "config"},
        )

    def _validate_against_config(
        self,
        provider_name: str,
        provider_type: ProviderType,
    ) -> None:
        """
        Validate provider registration against config definitions.

        Logs warnings if provider is not defined in config.
        """
        config_manager = _get_config_manager()
        if not config_manager:
            return

        try:
            # Check if provider exists in config
            provider_config = config_manager.get_provider(provider_name)

            if not provider_config:
                logger.warning(
                    f"Provider '{provider_name}' not defined in config. "
                    "Consider adding it to providers.yaml"
                )
                return

            # Validate type matches
            config_type = provider_config.type.value
            if config_type != provider_type.value:
                logger.warning(
                    f"Provider type mismatch for '{provider_name}': "
                    f"registration={provider_type.value}, "
                    f"config={config_type}"
                )

            # Check if enabled in config
            if not provider_config.enabled:
                logger.warning(f"Provider '{provider_name}' is disabled in config")

        except Exception as e:
            logger.debug(f"Config validation failed: {e}")

    def validate_provider_config(self, provider_id: str) -> dict[str, Any]:
        """
        Validate a provider's state against config.

        Returns validation results with any mismatches.
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return {
                "valid": False,
                "error": f"Provider '{provider_id}' not found",
            }

        config_manager = _get_config_manager()
        if not config_manager:
            return {
                "valid": True,
                "warning": "Config manager not available",
            }

        try:
            provider_config = config_manager.get_provider(provider.name)
            if not provider_config:
                return {
                    "valid": True,
                    "warning": f"Provider '{provider.name}' not in config",
                }

            mismatches = []

            # Check enabled status
            if provider.enabled != provider_config.enabled:
                mismatches.append(
                    {
                        "field": "enabled",
                        "local": provider.enabled,
                        "config": provider_config.enabled,
                    }
                )

            # Check priority
            if provider.priority != provider_config.priority:
                mismatches.append(
                    {
                        "field": "priority",
                        "local": provider.priority,
                        "config": provider_config.priority,
                    }
                )

            return {
                "valid": len(mismatches) == 0,
                "mismatches": mismatches,
                "config_provider_id": provider_config.provider_id,
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
            }

    async def sync_provider_with_config(self, provider_id: str) -> bool:
        """
        Sync a provider's state with config definitions.

        Updates local provider to match config values.
        Returns True if sync was successful.
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return False

        config_manager = _get_config_manager()
        if not config_manager:
            return False

        try:
            provider_config = config_manager.get_provider(provider.name)
            if not provider_config:
                return False

            # Update from config
            provider.enabled = provider_config.enabled
            provider.priority = provider_config.priority
            provider.default_model = (
                provider_config.get_default_model().model_id
                if provider_config.get_default_model()
                else provider.default_model
            )
            provider.updated_at = datetime.utcnow()

            # Update health monitor circuit breaker config
            cb = provider_config.circuit_breaker
            self._health_monitor.circuit_breaker_threshold = cb.failure_threshold
            self._health_monitor.circuit_breaker_timeout = cb.recovery_timeout_seconds

            logger.info(f"Synced provider '{provider_id}' with config")

            await self._emit_event(
                "provider_synced",
                {"provider_id": provider_id, "name": provider.name},
            )

            return True

        except Exception as e:
            logger.error(f"Failed to sync provider with config: {e}")
            return False

    async def set_default_provider(self, provider_id: str) -> bool:
        """Set the default provider"""
        async with self._lock:
            if provider_id not in self._providers:
                raise ValueError(f"Provider '{provider_id}' not found")

            # Clear previous default
            if self._default_provider_id and self._default_provider_id in self._providers:
                self._providers[self._default_provider_id].is_default = False

            self._default_provider_id = provider_id
            self._providers[provider_id].is_default = True

            await self._emit_event("default_provider_changed", {"provider_id": provider_id})

            return True

    async def select_provider(self, request: ProviderSelectionRequest) -> ProviderSelectionResponse:
        """
        Select/switch the active provider.

        Args:
            request: Selection request with provider and optional model

        Returns:
            ProviderSelectionResponse with result
        """
        provider = self._providers.get(request.provider_id)
        if not provider:
            return ProviderSelectionResponse(
                success=False,
                provider_id=request.provider_id,
                provider_name="",
                message=f"Provider '{request.provider_id}' not found",
            )

        if not provider.enabled:
            return ProviderSelectionResponse(
                success=False,
                provider_id=request.provider_id,
                provider_name=provider.name,
                message=f"Provider '{provider.name}' is disabled",
            )

        # Check health
        if self._health_monitor.is_circuit_open(request.provider_id):
            return ProviderSelectionResponse(
                success=False,
                provider_id=request.provider_id,
                provider_name=provider.name,
                message=f"Provider '{provider.name}' is currently unavailable",
            )

        previous_id = self._active_provider_id
        self._active_provider_id = request.provider_id

        await self._emit_event(
            "provider_selected",
            {
                "provider_id": request.provider_id,
                "previous_provider_id": previous_id,
                "model_id": request.model_id,
            },
        )

        return ProviderSelectionResponse(
            success=True,
            provider_id=request.provider_id,
            provider_name=provider.name,
            model_id=request.model_id or provider.default_model,
            previous_provider_id=previous_id,
            session_id=request.session_id,
        )

    def get_active_provider(self) -> ProviderInfo | None:
        """Get the currently active provider"""
        provider_id = self._active_provider_id or self._default_provider_id
        if provider_id:
            return self._providers.get(provider_id)
        return None

    async def test_provider(self, provider_id: str) -> ProviderTestResult:
        """
        Test a provider's connectivity and capabilities.

        Args:
            provider_id: ID of the provider to test

        Returns:
            ProviderTestResult with test results
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return ProviderTestResult(
                success=False, provider_id=provider_id, error="Provider not found"
            )

        instance = self._provider_instances.get(provider_id)
        if not instance:
            return ProviderTestResult(
                success=False, provider_id=provider_id, error="Provider instance not available"
            )

        try:
            start = time.time()
            is_healthy = await instance.check_health()
            latency = (time.time() - start) * 1000

            if is_healthy:
                self._health_monitor.record_success(provider_id, latency)
                provider.status = ProviderStatus.AVAILABLE

                return ProviderTestResult(
                    success=True,
                    provider_id=provider_id,
                    latency_ms=latency,
                    models_discovered=[m.id for m in provider.models],
                    capabilities_detected=provider.capabilities,
                )
            else:
                self._health_monitor.record_failure(provider_id, "Health check failed")
                return ProviderTestResult(
                    success=False,
                    provider_id=provider_id,
                    latency_ms=latency,
                    error="Health check returned false",
                )

        except Exception as e:
            self._health_monitor.record_failure(provider_id, str(e))
            return ProviderTestResult(success=False, provider_id=provider_id, error=str(e))

    async def get_fallback_provider(
        self, exclude_ids: list[str] | None = None
    ) -> ProviderInfo | None:
        """
        Get the next available fallback provider.

        Args:
            exclude_ids: Provider IDs to exclude from consideration

        Returns:
            Next available provider or None
        """
        exclude_ids = exclude_ids or []

        # Get providers sorted by priority
        candidates = [
            p
            for p in self._providers.values()
            if p.id not in exclude_ids
            and p.enabled
            and not self._health_monitor.is_circuit_open(p.id)
        ]

        # Prefer fallback providers first
        fallbacks = [p for p in candidates if p.is_fallback]
        if fallbacks:
            fallbacks.sort(key=lambda p: p.priority)
            return fallbacks[0]

        # Otherwise return highest priority available
        if candidates:
            candidates.sort(key=lambda p: p.priority)
            return candidates[0]

        return None

    def get_health(self, provider_id: str) -> ProviderHealthInfo:
        """Get health information for a provider"""
        return self._health_monitor.get_health(provider_id)

    def set_routing_config(self, config: ProviderRoutingConfig):
        """Update the routing configuration"""
        self._routing_config = config
        self._health_monitor.check_interval = config.health_check_interval_seconds
        self._health_monitor.circuit_breaker_threshold = config.circuit_breaker_threshold
        self._health_monitor.circuit_breaker_timeout = config.circuit_breaker_timeout_seconds

    def get_routing_config(self) -> ProviderRoutingConfig:
        """Get current routing configuration"""
        return self._routing_config

    async def shutdown(self):
        """Cleanup resources on shutdown"""
        for provider_id in list(self._providers.keys()):
            self._health_monitor.stop_monitoring(provider_id)

        self._providers.clear()
        self._provider_instances.clear()
        self._encrypted_keys.clear()

        logger.info("ProviderManagementService shutdown complete")


# Singleton instance
provider_management_service = ProviderManagementService()
