"""
Dynamic Provider Registry Service for Project Chimera

This module provides a centralized registry for managing AI/LLM providers with:
- Dynamic registration and deregistration of providers at runtime
- Hot-swapping of providers without server restart
- Provider health monitoring and automatic failover
- Secure API key storage with encryption
- Provider configuration management
- Plugin-based provider system integration
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from base64 import b64decode, b64encode
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

if TYPE_CHECKING:
    from app.services.provider_plugins import ProviderPlugin

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ProviderStatus(str, Enum):
    """Provider availability status."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    INITIALIZING = "initializing"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states for provider resilience."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


class ProviderType(str, Enum):
    """Supported provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GEMINI = "gemini"  # Alias for google
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    BIGMODEL = "bigmodel"  # ZhiPu AI GLM models
    ROUTEWAY = "routeway"  # Unified AI gateway
    OLLAMA = "ollama"
    AZURE = "azure"  # Azure OpenAI
    LOCAL = "local"
    CUSTOM = "custom"

    @classmethod
    def normalize(cls, value: str) -> "ProviderType":
        """
        Normalize provider type, handling aliases.

        Args:
            value: Provider type string

        Returns:
            Normalized ProviderType
        """
        # Handle aliases
        aliases = {
            "gemini": cls.GOOGLE,
            "google-ai": cls.GOOGLE,
            "claude": cls.ANTHROPIC,
            "zhipu": cls.BIGMODEL,
            "glm": cls.BIGMODEL,
            "azure-openai": cls.AZURE,
        }

        normalized = value.lower().strip()
        if normalized in aliases:
            return aliases[normalized]

        try:
            return cls(normalized)
        except ValueError:
            return cls.CUSTOM


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""

    provider_id: str
    provider_type: ProviderType
    display_name: str
    api_key: str | None = None
    api_key_encrypted: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    organization_id: str | None = None
    project_id: str | None = None
    is_enabled: bool = True
    is_default: bool = False
    priority: int = 100  # Lower = higher priority for fallback
    max_retries: int = 3
    timeout_seconds: float = 60.0
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
    supported_models: list[str] = field(default_factory=list)
    default_model: str | None = None
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self, include_sensitive: bool = False) -> dict[str, Any]:
        """Convert to dictionary, optionally excluding sensitive data."""
        data = {
            "provider_id": self.provider_id,
            "provider_type": self.provider_type.value,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "organization_id": self.organization_id,
            "project_id": self.project_id,
            "is_enabled": self.is_enabled,
            "is_default": self.is_default,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "rate_limit_rpm": self.rate_limit_rpm,
            "rate_limit_tpm": self.rate_limit_tpm,
            "supported_models": self.supported_models,
            "default_model": self.default_model,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_sensitive:
            data["api_key"] = self.api_key
            data["api_key_encrypted"] = self.api_key_encrypted
        else:
            data["has_api_key"] = bool(self.api_key or self.api_key_encrypted)
        return data


@dataclass
class ProviderHealthStatus:
    """Health status for a provider."""

    provider_id: str
    status: ProviderStatus
    circuit_breaker_state: CircuitBreakerState
    last_success: datetime | None = None
    last_failure: datetime | None = None
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    avg_latency_ms: float | None = None
    error_rate: float = 0.0
    last_error_message: str | None = None
    checked_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_id": self.provider_id,
            "status": self.status.value,
            "circuit_breaker_state": self.circuit_breaker_state.value,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "error_rate": self.error_rate,
            "last_error_message": self.last_error_message,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class ModelInfo:
    """Information about a model."""

    model_id: str
    provider_id: str
    display_name: str
    model_type: str = "chat"  # chat, completion, embedding, etc.
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    input_price_per_1k: float | None = None
    output_price_per_1k: float | None = None
    currency: str = "USD"
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "name": self.display_name,
            "display_name": self.display_name,
            "model_type": self.model_type,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_streaming": self.supports_streaming,
            "supports_function_calling": self.supports_function_calling,
            "supports_vision": self.supports_vision,
            "pricing": (
                {
                    "input_per_1k_tokens": self.input_price_per_1k,
                    "output_per_1k_tokens": self.output_price_per_1k,
                    "currency": self.currency,
                }
                if self.input_price_per_1k
                else None
            ),
            "capabilities": self.capabilities,
            "metadata": self.metadata,
        }


# =============================================================================
# Encryption Service
# =============================================================================


class EncryptionService:
    """Service for encrypting and decrypting sensitive data like API keys."""

    def __init__(self, secret_key: str | None = None):
        """Initialize encryption service with a secret key."""
        self._secret_key = secret_key or os.environ.get(
            "CHIMERA_ENCRYPTION_KEY", "chimera-default-key-change-in-production"
        )
        self._fernet = self._create_fernet()

    def _create_fernet(self) -> Fernet:
        """Create Fernet instance from secret key."""
        # Derive a proper key from the secret
        salt = b"chimera_salt_v1"  # Fixed salt for consistency
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = b64encode(kdf.derive(self._secret_key.encode()))
        return Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext."""
        if not plaintext:
            return ""
        encrypted = self._fernet.encrypt(plaintext.encode())
        return b64encode(encrypted).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64-encoded ciphertext and return plaintext."""
        if not ciphertext:
            return ""
        try:
            encrypted = b64decode(ciphertext.encode())
            decrypted = self._fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt: {e}")
            return ""

    def hash_key(self, api_key: str) -> str:
        """Create a hash of an API key for comparison without storing the key."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """Circuit breaker for provider resilience."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    async def can_execute(self) -> bool:
        """Check if a request can be executed."""
        async with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True

            if self._state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time and (
                    time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
                return False

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._success_count += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._success_count >= self.half_open_max_calls:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker transitioning to CLOSED")

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                logger.warning("Circuit breaker transitioning to OPEN from HALF_OPEN")
            elif (
                self._state == CircuitBreakerState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker transitioning to OPEN after {self._failure_count} failures"
                )

    async def reset(self) -> None:
        """Reset the circuit breaker."""
        async with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# =============================================================================
# Provider Registry
# =============================================================================


class ProviderRegistry:
    """
    Central registry for managing AI/LLM providers.

    Features:
    - Dynamic registration and deregistration
    - Hot-swapping without restart
    - Health monitoring with circuit breakers
    - Automatic failover
    - Secure API key storage
    """

    _instance: Optional["ProviderRegistry"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "ProviderRegistry":
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the provider registry."""
        if self._initialized:
            return

        self._providers: dict[str, ProviderConfig] = {}
        self._health_status: dict[str, ProviderHealthStatus] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._models: dict[str, dict[str, ModelInfo]] = {}  # provider_id -> model_id -> ModelInfo
        self._active_provider_id: str | None = None
        self._encryption_service = EncryptionService()
        self._event_handlers: dict[str, list[Callable]] = {
            "provider_registered": [],
            "provider_deregistered": [],
            "provider_updated": [],
            "provider_status_changed": [],
            "active_provider_changed": [],
            "model_added": [],
            "model_removed": [],
        }
        self._config_file_path = Path("data/provider_configs.json")
        self._health_check_interval = 30.0  # seconds
        self._health_check_task: asyncio.Task | None = None
        self._initialized = True

        logger.info("ProviderRegistry initialized")

    # =========================================================================
    # Provider Management
    # =========================================================================

    async def register_provider(
        self,
        config: ProviderConfig,
        encrypt_api_key: bool = True,
    ) -> bool:
        """
        Register a new provider or update an existing one.

        Args:
            config: Provider configuration
            encrypt_api_key: Whether to encrypt the API key

        Returns:
            True if registration was successful
        """
        async with self._lock:
            try:
                # Encrypt API key if provided
                if encrypt_api_key and config.api_key:
                    config.api_key_encrypted = self._encryption_service.encrypt(config.api_key)
                    config.api_key = None  # Clear plaintext

                # Check if updating existing provider
                is_update = config.provider_id in self._providers

                # Store provider config
                self._providers[config.provider_id] = config

                # Initialize health status
                if config.provider_id not in self._health_status:
                    self._health_status[config.provider_id] = ProviderHealthStatus(
                        provider_id=config.provider_id,
                        status=ProviderStatus.INITIALIZING,
                        circuit_breaker_state=CircuitBreakerState.CLOSED,
                    )

                # Initialize circuit breaker
                if config.provider_id not in self._circuit_breakers:
                    self._circuit_breakers[config.provider_id] = CircuitBreaker()

                # Initialize models dict
                if config.provider_id not in self._models:
                    self._models[config.provider_id] = {}

                # Set as default if specified or if first provider
                if config.is_default or len(self._providers) == 1:
                    await self._set_default_provider(config.provider_id)

                # Emit event
                event = "provider_updated" if is_update else "provider_registered"
                await self._emit_event(event, config.provider_id, config.to_dict())

                logger.info(
                    f"Provider {'updated' if is_update else 'registered'}: {config.provider_id}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to register provider {config.provider_id}: {e}")
                return False

    async def deregister_provider(self, provider_id: str) -> bool:
        """
        Remove a provider from the registry.

        Args:
            provider_id: ID of the provider to remove

        Returns:
            True if deregistration was successful
        """
        async with self._lock:
            if provider_id not in self._providers:
                logger.warning(f"Provider not found: {provider_id}")
                return False

            try:
                # Remove provider
                config = self._providers.pop(provider_id)
                self._health_status.pop(provider_id, None)
                self._circuit_breakers.pop(provider_id, None)
                self._models.pop(provider_id, None)

                # Update active provider if needed
                if self._active_provider_id == provider_id:
                    await self._select_next_available_provider()

                # Emit event
                await self._emit_event("provider_deregistered", provider_id, config.to_dict())

                logger.info(f"Provider deregistered: {provider_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to deregister provider {provider_id}: {e}")
                return False

    async def update_provider(
        self,
        provider_id: str,
        updates: dict[str, Any],
        encrypt_api_key: bool = True,
    ) -> bool:
        """
        Update an existing provider's configuration.

        Args:
            provider_id: ID of the provider to update
            updates: Dictionary of fields to update
            encrypt_api_key: Whether to encrypt API key if provided

        Returns:
            True if update was successful
        """
        async with self._lock:
            if provider_id not in self._providers:
                logger.warning(f"Provider not found: {provider_id}")
                return False

            try:
                config = self._providers[provider_id]

                # Handle API key update
                if updates.get("api_key"):
                    if encrypt_api_key:
                        config.api_key_encrypted = self._encryption_service.encrypt(
                            updates["api_key"]
                        )
                        config.api_key = None
                    else:
                        config.api_key = updates["api_key"]
                    del updates["api_key"]

                # Update other fields
                for key, value in updates.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

                config.updated_at = datetime.utcnow()

                # Handle default provider change
                if updates.get("is_default"):
                    await self._set_default_provider(provider_id)

                # Emit event
                await self._emit_event("provider_updated", provider_id, config.to_dict())

                logger.info(f"Provider updated: {provider_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to update provider {provider_id}: {e}")
                return False

    async def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """Get a provider's configuration."""
        return self._providers.get(provider_id)

    async def get_all_providers(self) -> list[ProviderConfig]:
        """Get all registered providers."""
        return list(self._providers.values())

    async def get_enabled_providers(self) -> list[ProviderConfig]:
        """Get all enabled providers."""
        return [p for p in self._providers.values() if p.is_enabled]

    async def get_provider_api_key(self, provider_id: str) -> str | None:
        """Get decrypted API key for a provider."""
        config = self._providers.get(provider_id)
        if not config:
            return None

        if config.api_key:
            return config.api_key

        if config.api_key_encrypted:
            return self._encryption_service.decrypt(config.api_key_encrypted)

        return None

    # =========================================================================
    # Active Provider Management
    # =========================================================================

    async def get_active_provider(self) -> ProviderConfig | None:
        """Get the currently active provider."""
        if self._active_provider_id:
            return self._providers.get(self._active_provider_id)
        return None

    async def set_active_provider(self, provider_id: str) -> bool:
        """
        Set the active provider for requests.

        Args:
            provider_id: ID of the provider to activate

        Returns:
            True if activation was successful
        """
        async with self._lock:
            if provider_id not in self._providers:
                logger.warning(f"Provider not found: {provider_id}")
                return False

            config = self._providers[provider_id]
            if not config.is_enabled:
                logger.warning(f"Provider is disabled: {provider_id}")
                return False

            old_provider_id = self._active_provider_id
            self._active_provider_id = provider_id

            # Emit event
            await self._emit_event(
                "active_provider_changed",
                provider_id,
                {"old_provider_id": old_provider_id, "new_provider_id": provider_id},
            )

            logger.info(f"Active provider changed: {old_provider_id} -> {provider_id}")
            return True

    async def _set_default_provider(self, provider_id: str) -> None:
        """Set a provider as the default."""
        # Clear default flag from other providers
        for pid, config in self._providers.items():
            if pid != provider_id:
                config.is_default = False

        # Set default flag
        if provider_id in self._providers:
            self._providers[provider_id].is_default = True

            # Also set as active if no active provider
            if not self._active_provider_id:
                self._active_provider_id = provider_id

    async def _select_next_available_provider(self) -> None:
        """Select the next available provider when current one is unavailable."""
        # Sort by priority (lower = higher priority)
        sorted_providers = sorted(
            [p for p in self._providers.values() if p.is_enabled],
            key=lambda p: p.priority,
        )

        for provider in sorted_providers:
            health = self._health_status.get(provider.provider_id)
            if health and health.status in [ProviderStatus.AVAILABLE, ProviderStatus.DEGRADED]:
                self._active_provider_id = provider.provider_id
                logger.info(f"Switched to fallback provider: {provider.provider_id}")
                return

        # No available provider
        self._active_provider_id = None
        logger.warning("No available providers for fallback")

    # =========================================================================
    # Model Management
    # =========================================================================

    async def register_model(self, model: ModelInfo) -> bool:
        """Register a model for a provider."""
        async with self._lock:
            if model.provider_id not in self._providers:
                logger.warning(f"Provider not found: {model.provider_id}")
                return False

            if model.provider_id not in self._models:
                self._models[model.provider_id] = {}

            self._models[model.provider_id][model.model_id] = model

            # Update provider's supported models list
            config = self._providers[model.provider_id]
            if model.model_id not in config.supported_models:
                config.supported_models.append(model.model_id)

            # Emit event
            await self._emit_event("model_added", model.provider_id, model.to_dict())

            logger.debug(f"Model registered: {model.model_id} for {model.provider_id}")
            return True

    async def deregister_model(self, provider_id: str, model_id: str) -> bool:
        """Remove a model from a provider."""
        async with self._lock:
            if provider_id not in self._models:
                return False

            if model_id not in self._models[provider_id]:
                return False

            self._models[provider_id].pop(model_id)

            # Update provider's supported models list
            if provider_id in self._providers:
                config = self._providers[provider_id]
                if model_id in config.supported_models:
                    config.supported_models.remove(model_id)

            # Emit event
            await self._emit_event("model_removed", provider_id, {"model_id": model_id})

            logger.debug(f"Model deregistered: {model_id} from {provider_id}")
            return True

    async def get_models(self, provider_id: str) -> list[ModelInfo]:
        """Get all models for a provider."""
        if provider_id not in self._models:
            return []
        return list(self._models[provider_id].values())

    async def get_model(self, provider_id: str, model_id: str) -> ModelInfo | None:
        """Get a specific model."""
        if provider_id not in self._models:
            return None
        return self._models[provider_id].get(model_id)

    async def get_all_models(self) -> dict[str, list[ModelInfo]]:
        """Get all models grouped by provider."""
        return {pid: list(models.values()) for pid, models in self._models.items()}

    # =========================================================================
    # Health Management
    # =========================================================================

    async def get_health_status(self, provider_id: str) -> ProviderHealthStatus | None:
        """Get health status for a provider."""
        return self._health_status.get(provider_id)

    async def get_all_health_status(self) -> dict[str, ProviderHealthStatus]:
        """Get health status for all providers."""
        return dict(self._health_status)

    async def update_health_status(
        self,
        provider_id: str,
        status: ProviderStatus,
        latency_ms: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update health status for a provider."""
        async with self._lock:
            if provider_id not in self._health_status:
                self._health_status[provider_id] = ProviderHealthStatus(
                    provider_id=provider_id,
                    status=status,
                    circuit_breaker_state=CircuitBreakerState.CLOSED,
                )

            health = self._health_status[provider_id]
            old_status = health.status
            health.status = status
            health.checked_at = datetime.utcnow()
            health.total_requests += 1

            if status == ProviderStatus.AVAILABLE:
                health.success_count += 1
                health.last_success = datetime.utcnow()
                if latency_ms:
                    # Update rolling average
                    if health.avg_latency_ms:
                        health.avg_latency_ms = (health.avg_latency_ms * 0.9) + (latency_ms * 0.1)
                    else:
                        health.avg_latency_ms = latency_ms
            else:
                health.failure_count += 1
                health.last_failure = datetime.utcnow()
                health.last_error_message = error_message

            # Update error rate
            if health.total_requests > 0:
                health.error_rate = health.failure_count / health.total_requests

            # Update circuit breaker state
            cb = self._circuit_breakers.get(provider_id)
            if cb:
                health.circuit_breaker_state = cb.state

            # Emit event if status changed
            if old_status != status:
                await self._emit_event(
                    "provider_status_changed",
                    provider_id,
                    {"old_status": old_status.value, "new_status": status.value},
                )

    async def record_request_success(self, provider_id: str, latency_ms: float) -> None:
        """Record a successful request for a provider."""
        cb = self._circuit_breakers.get(provider_id)
        if cb:
            await cb.record_success()

        await self.update_health_status(
            provider_id, ProviderStatus.AVAILABLE, latency_ms=latency_ms
        )

    async def record_request_failure(self, provider_id: str, error_message: str) -> None:
        """Record a failed request for a provider."""
        cb = self._circuit_breakers.get(provider_id)
        if cb:
            await cb.record_failure()

        # Determine status based on circuit breaker
        status = ProviderStatus.DEGRADED
        if cb and cb.state == CircuitBreakerState.OPEN:
            status = ProviderStatus.UNAVAILABLE

        await self.update_health_status(provider_id, status, error_message=error_message)

    async def can_use_provider(self, provider_id: str) -> bool:
        """Check if a provider can be used (circuit breaker check)."""
        cb = self._circuit_breakers.get(provider_id)
        if not cb:
            return True
        return await cb.can_execute()

    async def check_provider_health(self, provider_id: str) -> ProviderStatus:
        """
        Perform a health check on a provider.

        This should be overridden or extended to perform actual health checks.
        """
        config = self._providers.get(provider_id)
        if not config:
            return ProviderStatus.UNKNOWN

        if not config.is_enabled:
            return ProviderStatus.DISABLED

        # Check circuit breaker
        cb = self._circuit_breakers.get(provider_id)
        if cb and cb.state == CircuitBreakerState.OPEN:
            return ProviderStatus.UNAVAILABLE

        # Default to current status or available
        health = self._health_status.get(provider_id)
        if health:
            return health.status

        return ProviderStatus.AVAILABLE

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_check_task and not self._health_check_task.done():
            return

        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Health monitoring started")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring task."""
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task
            self._health_check_task = None
            logger.info("Health monitoring stopped")

    async def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)

                for provider_id in list(self._providers.keys()):
                    try:
                        status = await self.check_provider_health(provider_id)
                        await self.update_health_status(provider_id, status)
                    except Exception as e:
                        logger.error(f"Health check failed for {provider_id}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    # =========================================================================
    # Event System
    # =========================================================================

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if event in self._event_handlers and handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)

    async def _emit_event(self, event: str, provider_id: str, data: dict[str, Any]) -> None:
        """Emit an event to all registered handlers."""
        if event not in self._event_handlers:
            return

        for handler in self._event_handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(provider_id, data)
                else:
                    handler(provider_id, data)
            except Exception as e:
                logger.error(f"Event handler error for {event}: {e}")

    # =========================================================================
    # Persistence
    # =========================================================================

    async def save_to_file(self, file_path: Path | None = None) -> bool:
        """Save provider configurations to a JSON file."""
        path = file_path or self._config_file_path

        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data (exclude sensitive info)
            data = {
                "providers": {
                    pid: config.to_dict(include_sensitive=True)
                    for pid, config in self._providers.items()
                },
                "active_provider_id": self._active_provider_id,
                "saved_at": datetime.utcnow().isoformat(),
            }

            # Write to file
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Provider configurations saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
            return False

    async def load_from_file(self, file_path: Path | None = None) -> bool:
        """Load provider configurations from a JSON file."""
        path = file_path or self._config_file_path

        if not path.exists():
            logger.info(f"No configuration file found at {path}")
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            # Load providers
            for _pid, config_data in data.get("providers", {}).items():
                config = ProviderConfig(
                    provider_id=config_data["provider_id"],
                    provider_type=ProviderType(config_data["provider_type"]),
                    display_name=config_data["display_name"],
                    api_key_encrypted=config_data.get("api_key_encrypted"),
                    base_url=config_data.get("base_url"),
                    api_version=config_data.get("api_version"),
                    organization_id=config_data.get("organization_id"),
                    project_id=config_data.get("project_id"),
                    is_enabled=config_data.get("is_enabled", True),
                    is_default=config_data.get("is_default", False),
                    priority=config_data.get("priority", 100),
                    max_retries=config_data.get("max_retries", 3),
                    timeout_seconds=config_data.get("timeout_seconds", 60.0),
                    rate_limit_rpm=config_data.get("rate_limit_rpm"),
                    rate_limit_tpm=config_data.get("rate_limit_tpm"),
                    supported_models=config_data.get("supported_models", []),
                    default_model=config_data.get("default_model"),
                    capabilities=config_data.get("capabilities", []),
                    metadata=config_data.get("metadata", {}),
                )
                await self.register_provider(config, encrypt_api_key=False)

            # Set active provider
            if data.get("active_provider_id"):
                await self.set_active_provider(data["active_provider_id"])

            logger.info(f"Provider configurations loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            return False

    # =========================================================================
    # Fallback and Routing
    # =========================================================================

    async def get_fallback_provider(
        self, exclude_providers: set[str] | None = None
    ) -> ProviderConfig | None:
        """
        Get the next available fallback provider.

        Args:
            exclude_providers: Set of provider IDs to exclude

        Returns:
            Next available provider or None
        """
        exclude = exclude_providers or set()

        # Sort by priority
        sorted_providers = sorted(
            [p for p in self._providers.values() if p.is_enabled and p.provider_id not in exclude],
            key=lambda p: p.priority,
        )

        for provider in sorted_providers:
            # Check health
            health = self._health_status.get(provider.provider_id)
            if health and health.status in [
                ProviderStatus.AVAILABLE,
                ProviderStatus.DEGRADED,
            ]:
                # Check circuit breaker
                if await self.can_use_provider(provider.provider_id):
                    return provider

        return None

    async def get_provider_for_model(self, model_id: str) -> ProviderConfig | None:
        """
        Find a provider that supports a specific model.

        Args:
            model_id: The model ID to find

        Returns:
            Provider that supports the model or None
        """
        for provider in self._providers.values():
            if not provider.is_enabled:
                continue

            if model_id in provider.supported_models:
                if await self.can_use_provider(provider.provider_id):
                    return provider

            # Also check registered models
            if provider.provider_id in self._models:
                if model_id in self._models[provider.provider_id]:
                    if await self.can_use_provider(provider.provider_id):
                        return provider

        return None

    # =========================================================================
    # Plugin System Integration
    # =========================================================================

    async def register_plugin(
        self,
        plugin: "ProviderPlugin",
        auto_discover_models: bool = True,
    ) -> bool:
        """
        Register a provider plugin and optionally discover its models.

        Args:
            plugin: The provider plugin to register
            auto_discover_models: Whether to auto-discover models

        Returns:
            True if registration was successful
        """
        try:
            # Create provider config from plugin
            provider_type = ProviderType.normalize(plugin.provider_type)

            config = ProviderConfig(
                provider_id=plugin.provider_type,
                provider_type=provider_type,
                display_name=plugin.display_name,
                is_enabled=True,
                capabilities=[cap.value for cap in plugin.capabilities],
                default_model=plugin.get_default_model(),
            )

            # Register the provider
            success = await self.register_provider(config, encrypt_api_key=False)

            if success and auto_discover_models:
                # Discover and register models
                await self._discover_plugin_models(plugin)

            logger.info(f"Plugin registered: {plugin.provider_type}")
            return success

        except Exception as e:
            logger.error(f"Failed to register plugin {plugin.provider_type}: {e}")
            return False

    async def _discover_plugin_models(
        self,
        plugin: "ProviderPlugin",
    ) -> None:
        """
        Discover and register models from a plugin.

        Args:
            plugin: The plugin to discover models from
        """
        try:
            models = await plugin.list_models()

            for model in models:
                # Convert plugin ModelInfo to registry ModelInfo
                model_info = ModelInfo(
                    model_id=model.model_id,
                    provider_id=plugin.provider_type,
                    display_name=model.display_name,
                    context_window=model.context_window,
                    max_output_tokens=model.max_output_tokens,
                    supports_streaming=model.supports_streaming,
                    supports_function_calling=model.supports_function_calling,
                    supports_vision=model.supports_vision,
                    input_price_per_1k=model.input_price_per_1k,
                    output_price_per_1k=model.output_price_per_1k,
                    capabilities=model.capabilities or [],
                )
                await self.register_model(model_info)

            logger.info(f"Discovered {len(models)} models for {plugin.provider_type}")

        except Exception as e:
            logger.error(f"Failed to discover models for {plugin.provider_type}: {e}")

    async def get_models_for_provider(
        self,
        provider_type: str,
    ) -> list[ModelInfo]:
        """
        Get all models for a specific provider type.

        Uses plugin system for dynamic discovery if available.

        Args:
            provider_type: The provider type (e.g., "openai", "anthropic")

        Returns:
            List of ModelInfo objects
        """
        # First check locally registered models
        normalized_type = ProviderType.normalize(provider_type).value

        if normalized_type in self._models:
            models = list(self._models[normalized_type].values())
            if models:
                return models

        # Try to discover from plugin
        try:
            from app.services.provider_plugins import get_plugin

            plugin = get_plugin(provider_type)
            if plugin:
                await self._discover_plugin_models(plugin)
                return list(self._models.get(plugin.provider_type, {}).values())
        except ImportError:
            pass

        return []

    async def get_all_available_providers(self) -> list[dict[str, Any]]:
        """
        Get all available providers with their status and capabilities.

        Returns:
            List of provider info dictionaries
        """
        providers = []

        for config in self._providers.values():
            health = self._health_status.get(config.provider_id)

            provider_info = {
                "provider_id": config.provider_id,
                "provider_type": config.provider_type.value,
                "display_name": config.display_name,
                "is_enabled": config.is_enabled,
                "is_default": config.is_default,
                "status": health.status.value if health else "unknown",
                "capabilities": config.capabilities,
                "default_model": config.default_model,
                "models_count": len(self._models.get(config.provider_id, {})),
                "has_api_key": bool(config.api_key or config.api_key_encrypted),
            }
            providers.append(provider_info)

        # Also check for available plugins not yet registered
        try:
            from app.services.provider_plugins import get_all_plugins

            plugins = get_all_plugins()
            registered_types = {p.provider_type.value for p in self._providers.values()}

            for plugin in plugins.values():
                if plugin.provider_type not in registered_types:
                    providers.append(
                        {
                            "provider_id": plugin.provider_type,
                            "provider_type": plugin.provider_type,
                            "display_name": plugin.display_name,
                            "is_enabled": False,
                            "is_default": False,
                            "status": "not_registered",
                            "capabilities": [c.value for c in plugin.capabilities],
                            "default_model": plugin.get_default_model(),
                            "models_count": 0,
                            "has_api_key": False,
                        }
                    )
        except ImportError:
            pass

        return providers

    async def initialize_from_plugins(self) -> None:
        """
        Initialize the registry with all available plugins.

        This should be called at application startup to auto-register
        all available provider plugins.
        """
        try:
            from app.services.provider_plugins import get_all_plugins, register_all_plugins

            # Register all available plugins
            register_all_plugins()

            plugins = get_all_plugins()
            logger.info(f"Found {len(plugins)} provider plugins")

            # Register each plugin with the registry
            for plugin in plugins.values():
                await self.register_plugin(plugin, auto_discover_models=False)

            logger.info("Provider registry initialized from plugins")

        except ImportError as e:
            logger.warning(f"Plugin system not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize from plugins: {e}")

    async def route_request(
        self,
        preferred_provider_id: str | None = None,
        preferred_model_id: str | None = None,
        require_capability: str | None = None,
    ) -> tuple[ProviderConfig, str | None] | None:
        """
        Route a request to the best available provider.

        Args:
            preferred_provider_id: Preferred provider ID
            preferred_model_id: Preferred model ID
            require_capability: Required capability

        Returns:
            Tuple of (provider, model_id) or None
        """
        # Try preferred provider first
        if preferred_provider_id:
            provider = self._providers.get(preferred_provider_id)
            if provider and provider.is_enabled:
                if await self.can_use_provider(preferred_provider_id):
                    model = preferred_model_id or provider.default_model
                    return (provider, model)

        # Try to find provider for preferred model
        if preferred_model_id:
            provider = await self.get_provider_for_model(preferred_model_id)
            if provider:
                return (provider, preferred_model_id)

        # Try active provider
        if self._active_provider_id:
            provider = self._providers.get(self._active_provider_id)
            if provider and provider.is_enabled:
                if await self.can_use_provider(self._active_provider_id):
                    # Check capability if required
                    if require_capability and require_capability not in provider.capabilities:
                        pass  # Skip this provider
                    else:
                        return (provider, provider.default_model)

        # Fall back to any available provider
        fallback = await self.get_fallback_provider()
        if fallback:
            return (fallback, fallback.default_model)

        return None


# =============================================================================
# Global Registry Instance
# =============================================================================


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    return ProviderRegistry()


# Convenience alias
provider_registry = ProviderRegistry()
