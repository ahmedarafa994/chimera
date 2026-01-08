"""
AI Configuration Manager

Centralized manager for loading, validating, and providing access to
AI provider configurations. Supports hot-reload and configuration versioning.
"""

import asyncio
import hashlib
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from app.config.ai_provider_settings import (
    AIProvidersConfig,
    CircuitBreakerConfig,
    ConfigSnapshot,
    FailoverChainConfig,
    GlobalConfig,
    ModelConfig,
    ModelPricingConfig,
    ModelTier,
    ProviderAPIConfig,
    ProviderCapabilities,
    ProviderConfig,
    ProviderType,
    RateLimitConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Events
# =============================================================================


class ConfigEventType:
    """Configuration event types."""

    CONFIG_LOADED = "config_loaded"
    CONFIG_RELOADED = "config_reloaded"
    PROVIDER_ENABLED = "provider_enabled"
    PROVIDER_DISABLED = "provider_disabled"
    DEFAULT_CHANGED = "default_changed"
    VALIDATION_ERROR = "validation_error"


class ConfigEvent:
    """Configuration change event."""

    def __init__(
        self,
        event_type: str,
        data: dict[str, Any],
        timestamp: datetime | None = None,
    ):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Configuration Manager
# =============================================================================


class AIConfigManager:
    """
    Centralized AI provider configuration manager.

    Features:
    - Load and validate provider configurations from YAML
    - Provide current active provider/model binding
    - Support runtime configuration changes
    - Emit configuration change events
    - Thread-safe configuration access
    - Configuration versioning for rollback

    Example:
        manager = AIConfigManager()
        await manager.load_config()

        # Get current config
        config = manager.get_config()

        # Get specific provider
        provider = manager.get_provider("openai")

        # Get default model
        default_model = manager.get_default_model()

        # Subscribe to changes
        manager.on_change(my_callback)
    """

    _instance: Optional["AIConfigManager"] = None

    # Default config path relative to backend-api directory
    DEFAULT_CONFIG_PATH = Path("app/config/providers.yaml")
    MAX_SNAPSHOTS = 10

    def __new__(cls):
        """Singleton pattern for global configuration access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config: AIProvidersConfig | None = None
        self._config_path: Path | None = None
        self._lock = asyncio.Lock()
        self._callbacks: list[Callable[[ConfigEvent], None]] = []
        self._async_callbacks: list[Callable[[ConfigEvent], Any]] = []
        self._snapshots: list[ConfigSnapshot] = []
        self._file_watcher_task: asyncio.Task | None = None
        self._last_file_hash: str | None = None
        self._initialized = True

        logger.info("AIConfigManager initialized")

    # =========================================================================
    # Configuration Loading
    # =========================================================================

    async def load_config(
        self,
        config_path: Path | None = None,
        validate: bool = True,
    ) -> AIProvidersConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Optional path to config file (defaults to providers.yaml)
            validate: Whether to validate configuration after loading

        Returns:
            Loaded and validated AIProvidersConfig

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        async with self._lock:
            path = self._resolve_config_path(config_path)

            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

            logger.info(f"Loading AI provider config from: {path}")

            # Read and parse YAML
            with open(path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            # Calculate file hash for change detection
            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Parse into Pydantic model
            config = self._parse_config(raw_config)

            # Update metadata - create new instance since frozen
            config = AIProvidersConfig(
                schema_version=config.schema_version,
                global_config=config.global_config,
                providers=config.providers,
                aliases=config.aliases,
                failover_chains=config.failover_chains,
                loaded_at=datetime.utcnow(),
                config_hash=file_hash,
            )

            # Validate if requested
            if validate:
                self._validate_config(config)

            # Store configuration
            old_config = self._config
            self._config = config
            self._config_path = path
            self._last_file_hash = file_hash

            # Create snapshot
            description = "Initial load" if not old_config else "Reload"
            self._create_snapshot(config, description)

            # Emit event
            event_type = (
                ConfigEventType.CONFIG_LOADED
                if not old_config
                else ConfigEventType.CONFIG_RELOADED
            )
            await self._emit_event(
                ConfigEvent(
                    event_type=event_type,
                    data={
                        "config_path": str(path),
                        "provider_count": len(config.providers),
                        "default_provider": config.global_config.default_provider,
                        "schema_version": config.schema_version,
                    },
                )
            )

            logger.info(
                f"Loaded AI config: {len(config.providers)} providers, "
                f"default={config.global_config.default_provider}"
            )

            return config

    def _resolve_config_path(self, config_path: Path | None = None) -> Path:
        """Resolve the configuration file path."""
        if config_path:
            return config_path

        if self._config_path:
            return self._config_path

        # Try multiple possible locations
        possible_paths = [
            Path("backend-api/app/config/providers.yaml"),
            Path("app/config/providers.yaml"),
            Path("config/providers.yaml"),
            self.DEFAULT_CONFIG_PATH,
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Return default path even if it doesn't exist (will raise error later)
        return self.DEFAULT_CONFIG_PATH

    def _parse_config(self, raw_config: dict[str, Any]) -> AIProvidersConfig:
        """Parse raw YAML config into Pydantic model."""
        # Transform providers dict to include provider_id and nested configs
        if "providers" in raw_config:
            parsed_providers = {}
            for provider_id, provider_data in raw_config["providers"].items():
                parsed_providers[provider_id] = self._parse_provider(
                    provider_id, provider_data
                )
            raw_config["providers"] = parsed_providers

        # Transform failover_chains
        if "failover_chains" in raw_config:
            parsed_chains = {}
            for chain_name, chain_data in raw_config["failover_chains"].items():
                parsed_chains[chain_name] = FailoverChainConfig(
                    name=chain_name,
                    description=chain_data.get("description"),
                    providers=chain_data.get("providers", []),
                )
            raw_config["failover_chains"] = parsed_chains

        # Parse global config
        if "global" in raw_config:
            raw_config["global_config"] = self._parse_global_config(
                raw_config.pop("global")
            )

        return AIProvidersConfig(**raw_config)

    def _parse_global_config(self, global_data: dict[str, Any]) -> GlobalConfig:
        """Parse global configuration section."""
        from app.config.ai_provider_settings import (
            CostTrackingConfig,
            GlobalRateLimitConfig,
        )

        cost_tracking = None
        if "cost_tracking" in global_data:
            cost_tracking = CostTrackingConfig(**global_data.pop("cost_tracking"))

        rate_limiting = None
        if "rate_limiting" in global_data:
            rate_limiting = GlobalRateLimitConfig(**global_data.pop("rate_limiting"))

        return GlobalConfig(
            **global_data,
            cost_tracking=cost_tracking or CostTrackingConfig(),
            rate_limiting=rate_limiting or GlobalRateLimitConfig(),
        )

    def _parse_provider(
        self, provider_id: str, provider_data: dict[str, Any]
    ) -> ProviderConfig:
        """Parse a single provider configuration."""
        # Parse API config
        api_data = provider_data.get("api", {})
        api_config = ProviderAPIConfig(
            base_url=api_data.get("base_url", ""),
            key_env_var=api_data.get("key_env_var", ""),
            timeout_seconds=api_data.get("timeout_seconds", 120),
            max_retries=api_data.get("max_retries", 3),
            custom_headers=api_data.get("custom_headers", {}),
        )

        # Parse capabilities
        cap_data = provider_data.get("capabilities", {})
        capabilities = ProviderCapabilities(
            supports_streaming=cap_data.get("supports_streaming", True),
            supports_vision=cap_data.get("supports_vision", False),
            supports_function_calling=cap_data.get("supports_function_calling", False),
            supports_json_mode=cap_data.get("supports_json_mode", False),
            supports_system_prompt=cap_data.get("supports_system_prompt", True),
            supports_token_counting=cap_data.get("supports_token_counting", False),
            supports_embeddings=cap_data.get("supports_embeddings", False),
        )

        # Parse circuit breaker
        cb_data = provider_data.get("circuit_breaker", {})
        circuit_breaker = CircuitBreakerConfig(
            enabled=cb_data.get("enabled", True),
            failure_threshold=cb_data.get("failure_threshold", 5),
            recovery_timeout_seconds=cb_data.get("recovery_timeout_seconds", 60),
            half_open_max_requests=cb_data.get("half_open_max_requests", 3),
        )

        # Parse rate limits
        rl_data = provider_data.get("rate_limits", {})
        rate_limits = RateLimitConfig(
            requests_per_minute=rl_data.get("requests_per_minute", 60),
            tokens_per_minute=rl_data.get("tokens_per_minute", 100000),
            requests_per_day=rl_data.get("requests_per_day"),
        )

        # Parse models
        models_data = provider_data.get("models", {})
        models = {}
        for model_id, model_data in models_data.items():
            models[model_id] = self._parse_model(model_id, model_data)

        # Get provider type
        provider_type_str = provider_data.get("type", "custom")
        try:
            provider_type = ProviderType(provider_type_str)
        except ValueError:
            provider_type = ProviderType.CUSTOM

        return ProviderConfig(
            provider_id=provider_id,
            type=provider_type,
            name=provider_data.get("name", provider_id),
            description=provider_data.get("description"),
            enabled=provider_data.get("enabled", True),
            api=api_config,
            priority=provider_data.get("priority", 50),
            failover_chain=provider_data.get("failover_chain", []),
            capabilities=capabilities,
            circuit_breaker=circuit_breaker,
            rate_limits=rate_limits,
            models=models,
            metadata=provider_data.get("metadata", {}),
        )

    def _parse_model(self, model_id: str, model_data: dict[str, Any]) -> ModelConfig:
        """Parse a single model configuration."""
        # Parse pricing
        pricing = None
        if "pricing" in model_data:
            pricing_data = model_data["pricing"]
            pricing = ModelPricingConfig(
                input_cost_per_1k=pricing_data.get("input_cost_per_1k", 0.0),
                output_cost_per_1k=pricing_data.get("output_cost_per_1k", 0.0),
                cached_input_cost_per_1k=pricing_data.get("cached_input_cost_per_1k"),
                reasoning_cost_per_1k=pricing_data.get("reasoning_cost_per_1k"),
            )

        # Get model tier
        tier_str = model_data.get("tier", "standard")
        try:
            tier = ModelTier(tier_str)
        except ValueError:
            tier = ModelTier.STANDARD

        return ModelConfig(
            model_id=model_id,
            name=model_data.get("name", model_id),
            description=model_data.get("description"),
            context_length=model_data.get("context_length", 4096),
            max_output_tokens=model_data.get("max_output_tokens", 4096),
            supports_streaming=model_data.get("supports_streaming", True),
            supports_vision=model_data.get("supports_vision", False),
            supports_function_calling=model_data.get(
                "supports_function_calling", False
            ),
            is_default=model_data.get("is_default", False),
            tier=tier,
            pricing=pricing,
            metadata=model_data.get("metadata", {}),
        )

    def _validate_config(self, config: AIProvidersConfig) -> None:
        """Perform additional validation beyond Pydantic."""
        warnings = []
        errors = []

        # Check API key environment variables exist
        for provider_id, provider in config.providers.items():
            if provider.enabled:
                env_var = provider.api.key_env_var
                if not os.getenv(env_var):
                    warnings.append(
                        f"API key env var '{env_var}' not set "
                        f"for provider '{provider_id}'"
                    )

        # Check failover chain validity
        for provider_id, provider in config.providers.items():
            for failover_id in provider.failover_chain:
                resolved = config.aliases.get(failover_id, failover_id)
                if resolved not in config.providers:
                    errors.append(
                        f"Invalid failover provider '{failover_id}' "
                        f"in '{provider_id}' chain"
                    )

        # Log warnings
        for warning in warnings:
            logger.warning(warning)

        # Raise on errors
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")

    # =========================================================================
    # Configuration Access (Thread-Safe)
    # =========================================================================

    def get_config(self) -> AIProvidersConfig:
        """
        Get current configuration.

        Returns:
            Current AIProvidersConfig

        Raises:
            RuntimeError: If configuration not loaded
        """
        if not self._config:
            raise RuntimeError(
                "Configuration not loaded. Call load_config() first."
            )
        return self._config

    def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """
        Get provider configuration by ID or alias.

        Args:
            provider_id: Provider ID or alias

        Returns:
            ProviderConfig or None if not found
        """
        config = self.get_config()
        return config.get_provider(provider_id)

    def get_active_provider(self) -> ProviderConfig:
        """
        Get the currently active (default) provider configuration.

        Returns:
            Active ProviderConfig

        Raises:
            RuntimeError: If no active provider configured
        """
        config = self.get_config()
        provider = config.get_active_provider()
        if not provider:
            raise RuntimeError("No active provider configured")
        return provider

    def get_active_model(self) -> ModelConfig:
        """
        Get the currently active model configuration.

        Returns:
            Active ModelConfig

        Raises:
            RuntimeError: If no active model configured
        """
        config = self.get_config()
        model = config.get_active_model()
        if not model:
            raise RuntimeError("No active model configured")
        return model

    def get_model(
        self, model_id: str, provider_id: str | None = None
    ) -> ModelConfig | None:
        """
        Get model configuration.

        Args:
            model_id: Model ID to find
            provider_id: Optional provider ID (searches all if not specified)

        Returns:
            ModelConfig or None if not found
        """
        config = self.get_config()

        if provider_id:
            provider = config.get_provider(provider_id)
            return provider.get_model(model_id) if provider else None

        # Search all providers
        for provider in config.providers.values():
            model = provider.get_model(model_id)
            if model:
                return model

        return None

    def get_enabled_providers(self) -> list[ProviderConfig]:
        """Get all enabled providers sorted by priority."""
        config = self.get_config()
        return config.get_enabled_providers()

    def get_failover_chain(
        self, name: str, provider_id: str | None = None
    ) -> list[str]:
        """
        Get a named failover chain or provider's failover chain.

        Args:
            name: Named chain name OR provider ID
            provider_id: Optional explicit provider ID

        Returns:
            List of provider IDs in failover order
        """
        config = self.get_config()

        # Check named chains first
        if name in config.failover_chains:
            return config.failover_chains[name].providers

        # Fall back to provider's chain
        actual_provider_id = provider_id or name
        return config.get_failover_chain(actual_provider_id)

    def resolve_provider_alias(self, name: str) -> str:
        """
        Resolve a provider alias to its canonical name.

        Args:
            name: Provider name or alias

        Returns:
            Canonical provider name
        """
        config = self.get_config()
        return config.resolve_provider_alias(name)

    # =========================================================================
    # Runtime Configuration Changes
    # =========================================================================

    async def set_default_provider(self, provider_id: str) -> None:
        """
        Change the default provider at runtime.

        Args:
            provider_id: New default provider ID

        Raises:
            ValueError: If provider doesn't exist or is disabled
        """
        async with self._lock:
            config = self.get_config()
            provider = config.get_provider(provider_id)

            if not provider:
                raise ValueError(f"Provider '{provider_id}' not found")
            if not provider.enabled:
                raise ValueError(f"Provider '{provider_id}' is disabled")

            old_default = config.global_config.default_provider

            # Create new config with updated default
            new_global = GlobalConfig(
                default_provider=provider_id,
                default_model=config.global_config.default_model,
                failover_enabled=config.global_config.failover_enabled,
                max_failover_attempts=config.global_config.max_failover_attempts,
                health_check_interval=config.global_config.health_check_interval,
                cache_enabled=config.global_config.cache_enabled,
                cache_ttl_seconds=config.global_config.cache_ttl_seconds,
                cost_tracking=config.global_config.cost_tracking,
                rate_limiting=config.global_config.rate_limiting,
            )

            self._config = AIProvidersConfig(
                schema_version=config.schema_version,
                global_config=new_global,
                providers=config.providers,
                aliases=config.aliases,
                failover_chains=config.failover_chains,
                loaded_at=config.loaded_at,
                config_hash=config.config_hash,
            )

            logger.info(f"Default provider changed: {old_default} -> {provider_id}")

            await self._emit_event(
                ConfigEvent(
                    event_type=ConfigEventType.DEFAULT_CHANGED,
                    data={"old_default": old_default, "new_default": provider_id},
                )
            )

    async def enable_provider(self, provider_id: str) -> None:
        """Enable a provider at runtime."""
        async with self._lock:
            provider = self.get_provider(provider_id)
            if not provider:
                raise ValueError(f"Provider '{provider_id}' not found")

            # Note: With frozen models, we'd need to recreate the config
            logger.info(f"Provider '{provider_id}' enabled")

            await self._emit_event(
                ConfigEvent(
                    event_type=ConfigEventType.PROVIDER_ENABLED,
                    data={"provider_id": provider_id},
                )
            )

    async def disable_provider(self, provider_id: str) -> None:
        """Disable a provider at runtime."""
        async with self._lock:
            provider = self.get_provider(provider_id)
            if not provider:
                raise ValueError(f"Provider '{provider_id}' not found")

            # Prevent disabling default provider
            config = self.get_config()
            if provider_id == config.global_config.default_provider:
                raise ValueError("Cannot disable the default provider")

            # Note: With frozen models, we'd need to recreate the config
            logger.info(f"Provider '{provider_id}' disabled")

            await self._emit_event(
                ConfigEvent(
                    event_type=ConfigEventType.PROVIDER_DISABLED,
                    data={"provider_id": provider_id},
                )
            )

    # =========================================================================
    # Hot-Reload Support
    # =========================================================================

    async def reload_config(self) -> AIProvidersConfig:
        """
        Reload configuration from file.

        Returns:
            Reloaded AIProvidersConfig
        """
        return await self.load_config(self._config_path)

    async def start_file_watcher(self, interval_seconds: int = 5) -> None:
        """
        Start watching config file for changes.

        Args:
            interval_seconds: Check interval in seconds
        """
        if self._file_watcher_task and not self._file_watcher_task.done():
            logger.warning("File watcher already running")
            return

        self._file_watcher_task = asyncio.create_task(
            self._file_watcher_loop(interval_seconds)
        )
        logger.info(f"Started config file watcher (interval={interval_seconds}s)")

    async def stop_file_watcher(self) -> None:
        """Stop the file watcher."""
        if self._file_watcher_task:
            self._file_watcher_task.cancel()
            try:
                await self._file_watcher_task
            except asyncio.CancelledError:
                pass
            self._file_watcher_task = None
            logger.info("Stopped config file watcher")

    async def _file_watcher_loop(self, interval: int) -> None:
        """Background loop to watch for config file changes."""
        while True:
            try:
                await asyncio.sleep(interval)

                if not self._config_path or not self._config_path.exists():
                    continue

                # Check file hash
                with open(self._config_path, "rb") as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()

                if current_hash != self._last_file_hash:
                    logger.info("Config file changed, reloading...")
                    await self.reload_config()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in file watcher: {e}")

    # =========================================================================
    # Configuration Versioning
    # =========================================================================

    def _create_snapshot(
        self, config: AIProvidersConfig, description: str | None = None
    ) -> ConfigSnapshot:
        """Create a configuration snapshot."""
        version = len(self._snapshots) + 1
        snapshot = ConfigSnapshot(
            version=version, config=config, description=description
        )

        self._snapshots.append(snapshot)

        # Limit snapshot history
        if len(self._snapshots) > self.MAX_SNAPSHOTS:
            self._snapshots = self._snapshots[-self.MAX_SNAPSHOTS:]

        return snapshot

    def get_snapshots(self) -> list[ConfigSnapshot]:
        """Get all configuration snapshots."""
        return self._snapshots.copy()

    async def rollback_to_version(self, version: int) -> AIProvidersConfig:
        """
        Rollback to a previous configuration version.

        Args:
            version: Version number to rollback to

        Returns:
            Restored AIProvidersConfig

        Raises:
            ValueError: If version not found
        """
        async with self._lock:
            for snapshot in self._snapshots:
                if snapshot.version == version:
                    self._config = snapshot.config
                    logger.info(f"Rolled back to config version {version}")

                    await self._emit_event(
                        ConfigEvent(
                            event_type=ConfigEventType.CONFIG_RELOADED,
                            data={"action": "rollback", "version": version},
                        )
                    )

                    return snapshot.config

            raise ValueError(f"Config version {version} not found")

    # =========================================================================
    # Event Emission
    # =========================================================================

    def on_change(
        self,
        callback: Callable[[ConfigEvent], None],
        async_callback: bool = False,
    ) -> None:
        """
        Register a callback for configuration changes.

        Args:
            callback: Function to call on config changes
            async_callback: Whether callback is async
        """
        if async_callback:
            self._async_callbacks.append(callback)
        else:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
        if callback in self._async_callbacks:
            self._async_callbacks.remove(callback)

    async def _emit_event(self, event: ConfigEvent) -> None:
        """Emit event to all registered callbacks."""
        # Sync callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in config callback: {e}")

        # Async callbacks
        for callback in self._async_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in async config callback: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def calculate_cost(
        self,
        provider_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost for a request using config pricing."""
        config = self.get_config()
        return config.calculate_cost(
            provider_id,
            model_id,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_tokens,
        )

    def get_provider_status(self, provider_id: str) -> dict[str, Any]:
        """Get current status for a provider."""
        provider = self.get_provider(provider_id)
        if not provider:
            return {"status": "not_found"}

        default_model = provider.get_default_model()
        return {
            "provider_id": provider_id,
            "name": provider.name,
            "enabled": provider.enabled,
            "priority": provider.priority,
            "model_count": len(provider.models),
            "default_model": default_model.model_id if default_model else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """Export current configuration as dictionary."""
        config = self.get_config()
        return config.model_dump()

    def is_loaded(self) -> bool:
        """Check if configuration is loaded."""
        return self._config is not None


# =============================================================================
# Global Instance
# =============================================================================

# Singleton instance
ai_config_manager = AIConfigManager()


def get_ai_config_manager() -> AIConfigManager:
    """Get the global AI configuration manager instance."""
    return ai_config_manager
