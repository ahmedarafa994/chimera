"""
Provider Plugin System for Project Chimera

This module implements a plugin-based architecture for AI provider integrations.
Each provider implements the ProviderPlugin protocol, enabling:
- Dynamic model discovery from provider APIs
- Unified interface for all providers (OpenAI, Anthropic, Google/Gemini, DeepSeek, etc.)
- API key validation
- Streaming and non-streaming generation
- Consistent error handling across providers

Usage:
    from app.services.provider_plugins import ProviderPlugin, get_plugin, register_plugin

    # Register a custom plugin
    register_plugin(MyCustomPlugin())

    # Get a plugin by provider type
    plugin = get_plugin("openai")
    models = await plugin.list_models()
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


class ProviderCapability(str, Enum):
    """Capabilities that providers may support."""

    STREAMING = "streaming"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    EMBEDDINGS = "embeddings"
    CODE_COMPLETION = "code_completion"
    REASONING = "reasoning"  # Extended thinking/reasoning mode
    JSON_MODE = "json_mode"
    SYSTEM_MESSAGES = "system_messages"


@dataclass
class ModelInfo:
    """Information about a model available from a provider."""

    model_id: str
    provider_id: str
    display_name: str
    model_type: str = "chat"  # chat, completion, embedding, etc.
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    input_price_per_1k: float | None = None
    output_price_per_1k: float | None = None
    currency: str = "USD"
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_available: bool = True
    deprecated: bool = False
    deprecation_message: str | None = None

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
            "supports_reasoning": self.supports_reasoning,
            "pricing": {
                "input_per_1k_tokens": self.input_price_per_1k,
                "output_per_1k_tokens": self.output_price_per_1k,
                "currency": self.currency,
            }
            if self.input_price_per_1k is not None
            else None,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "is_available": self.is_available,
            "deprecated": self.deprecated,
            "deprecation_message": self.deprecation_message,
        }


@dataclass
class ProviderInfo:
    """Information about an AI provider."""

    provider_id: str
    display_name: str
    provider_type: str
    aliases: list[str] = field(default_factory=list)
    is_enabled: bool = True
    requires_api_key: bool = True
    api_key_env_var: str | None = None
    base_url: str | None = None
    capabilities: list[str] = field(default_factory=list)
    default_model: str | None = None
    model_count: int = 0
    status: str = "available"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "provider_type": self.provider_type,
            "aliases": self.aliases,
            "is_enabled": self.is_enabled,
            "requires_api_key": self.requires_api_key,
            "base_url": self.base_url,
            "capabilities": self.capabilities,
            "default_model": self.default_model,
            "model_count": self.model_count,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    model: str | None = None
    system_instruction: str | None = None
    messages: list[dict[str, str]] | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    api_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    model_used: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    latency_ms: float | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model_used": self.model_used,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """A chunk of streamed generation response."""

    text: str
    is_final: bool = False
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Provider Plugin Protocol
# =============================================================================


@runtime_checkable
class ProviderPlugin(Protocol):
    """
    Protocol for AI provider plugins.

    All provider plugins must implement this protocol to be compatible
    with the unified provider system.
    """

    @property
    def provider_type(self) -> str:
        """Unique identifier for this provider type (e.g., 'openai', 'anthropic')."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for this provider."""
        ...

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """
        List all available models from this provider.

        Args:
            api_key: Optional API key to use for the request.
                    If not provided, uses environment variable.

        Returns:
            List of ModelInfo objects describing available models.
        """
        ...

    async def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key for this provider.

        Args:
            api_key: The API key to validate.

        Returns:
            True if the API key is valid, False otherwise.
        """
        ...

    async def generate(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> GenerationResponse:
        """
        Generate text using this provider.

        Args:
            request: The generation request parameters.
            api_key: Optional API key to use. If not provided, uses environment variable.

        Returns:
            GenerationResponse with the generated text and metadata.
        """
        ...

    async def generate_stream(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream text generation from this provider.

        Args:
            request: The generation request parameters.
            api_key: Optional API key to use. If not provided, uses environment variable.

        Yields:
            StreamChunk objects containing incremental text.
        """
        ...

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming generation."""
        ...

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        ...

    def get_provider_info(self) -> ProviderInfo:
        """Get information about this provider."""
        ...


# =============================================================================
# Base Provider Plugin Implementation
# =============================================================================


class BaseProviderPlugin(ABC):
    """
    Abstract base class for provider plugins.

    Provides common functionality and ensures all plugins implement
    the required methods.
    """

    def __init__(self):
        self._models_cache: list[ModelInfo] | None = None
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_seconds: int = 3600  # 1 hour default

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Unique identifier for this provider type."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for this provider."""
        pass

    @property
    def aliases(self) -> list[str]:
        """Alternative names for this provider (e.g., 'gemini' -> 'google')."""
        return []

    @property
    def api_key_env_var(self) -> str | None:
        """Environment variable name for the API key."""
        return None

    @property
    def base_url(self) -> str | None:
        """Base URL for the provider API."""
        return None

    @property
    def capabilities(self) -> list[ProviderCapability]:
        """List of capabilities this provider supports."""
        return [ProviderCapability.STREAMING]

    def is_configured(self) -> bool:
        """
        Check if this provider is properly configured (API key available).

        Returns True if:
        - No API key is required, OR
        - The required API key environment variable is set
        """
        if not self.api_key_env_var:
            return True  # No API key required
        import os
        return bool(os.environ.get(self.api_key_env_var))

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming generation."""
        return ProviderCapability.STREAMING in self.capabilities

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @abstractmethod
    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """List all available models from this provider."""
        pass

    @abstractmethod
    async def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key for this provider."""
        pass

    @abstractmethod
    async def generate(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> GenerationResponse:
        """Generate text using this provider."""
        pass

    @abstractmethod
    async def generate_stream(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from this provider."""
        pass

    def get_provider_info(self) -> ProviderInfo:
        """Get information about this provider."""
        return ProviderInfo(
            provider_id=self.provider_type,
            display_name=self.display_name,
            provider_type=self.provider_type,
            aliases=self.aliases,
            is_enabled=True,
            requires_api_key=self.api_key_env_var is not None,
            api_key_env_var=self.api_key_env_var,
            base_url=self.base_url,
            capabilities=[c.value for c in self.capabilities],
            default_model=self.get_default_model(),
            model_count=len(self._models_cache) if self._models_cache else 0,
            status="available",
        )

    def _get_api_key(self, api_key: str | None = None) -> str | None:
        """Get API key from parameter or environment."""
        import os

        if api_key:
            return api_key
        if self.api_key_env_var:
            return os.environ.get(self.api_key_env_var)
        return None

    def _is_cache_valid(self) -> bool:
        """Check if the models cache is still valid."""
        if self._models_cache is None or self._cache_timestamp is None:
            return False
        elapsed = (datetime.utcnow() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_ttl_seconds

    async def get_cached_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """Get models from cache or fetch if cache is invalid."""
        if self._is_cache_valid() and self._models_cache is not None:
            return self._models_cache

        models = await self.list_models(api_key)
        self._models_cache = models
        self._cache_timestamp = datetime.utcnow()
        return models

    def invalidate_cache(self) -> None:
        """Invalidate the models cache."""
        self._models_cache = None
        self._cache_timestamp = None


# =============================================================================
# Plugin Registry
# =============================================================================


class PluginRegistry:
    """
    Registry for provider plugins.

    Manages registration, lookup, and lifecycle of provider plugins.
    """

    _instance: Optional["PluginRegistry"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "PluginRegistry":
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the plugin registry."""
        if self._initialized:
            return

        self._plugins: dict[str, ProviderPlugin] = {}
        self._aliases: dict[str, str] = {}  # alias -> provider_type
        self._initialized = True
        logger.info("PluginRegistry initialized")

    def register(self, plugin: ProviderPlugin) -> None:
        """
        Register a provider plugin.

        Args:
            plugin: The plugin to register.
        """
        provider_type = plugin.provider_type.lower()

        if provider_type in self._plugins:
            logger.warning(f"Overwriting existing plugin for provider: {provider_type}")

        self._plugins[provider_type] = plugin

        # Register aliases if the plugin has them
        if hasattr(plugin, "aliases"):
            for alias in plugin.aliases:
                alias_lower = alias.lower()
                self._aliases[alias_lower] = provider_type
                logger.debug(f"Registered alias '{alias_lower}' -> '{provider_type}'")

        logger.info(f"Registered plugin for provider: {provider_type} ({plugin.display_name})")

    def unregister(self, provider_type: str) -> bool:
        """
        Unregister a provider plugin.

        Args:
            provider_type: The provider type to unregister.

        Returns:
            True if the plugin was unregistered, False if not found.
        """
        provider_type = provider_type.lower()

        if provider_type not in self._plugins:
            return False

        plugin = self._plugins.pop(provider_type)

        # Remove any aliases
        if hasattr(plugin, "aliases"):
            for alias in plugin.aliases:
                self._aliases.pop(alias.lower(), None)

        logger.info(f"Unregistered plugin for provider: {provider_type}")
        return True

    def get(self, provider_type: str) -> ProviderPlugin | None:
        """
        Get a provider plugin by type or alias.

        Args:
            provider_type: The provider type or alias.

        Returns:
            The plugin if found, None otherwise.
        """
        provider_type = provider_type.lower()

        # Check direct match first
        if provider_type in self._plugins:
            return self._plugins[provider_type]

        # Check aliases
        if provider_type in self._aliases:
            canonical = self._aliases[provider_type]
            return self._plugins.get(canonical)

        return None

    def get_all(self) -> list[ProviderPlugin]:
        """Get all registered plugins."""
        return list(self._plugins.values())

    def get_all_provider_types(self) -> list[str]:
        """Get all registered provider types."""
        return list(self._plugins.keys())

    def resolve_alias(self, provider_type: str) -> str:
        """
        Resolve a provider alias to its canonical type.

        Args:
            provider_type: The provider type or alias.

        Returns:
            The canonical provider type.
        """
        provider_type = provider_type.lower()
        return self._aliases.get(provider_type, provider_type)

    def is_registered(self, provider_type: str) -> bool:
        """Check if a provider type or alias is registered."""
        provider_type = provider_type.lower()
        return provider_type in self._plugins or provider_type in self._aliases

    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()
        self._aliases.clear()
        logger.info("Cleared all registered plugins")


# =============================================================================
# Convenience Functions
# =============================================================================


_registry = PluginRegistry()


def register_plugin(plugin: ProviderPlugin) -> None:
    """Register a provider plugin with the global registry."""
    _registry.register(plugin)


def unregister_plugin(provider_type: str) -> bool:
    """Unregister a provider plugin from the global registry."""
    return _registry.unregister(provider_type)


def get_plugin(provider_type: str) -> ProviderPlugin | None:
    """Get a provider plugin from the global registry."""
    return _registry.get(provider_type)


def get_all_plugins() -> list[ProviderPlugin]:
    """Get all registered plugins from the global registry."""
    return _registry.get_all()


def get_all_provider_types() -> list[str]:
    """Get all registered provider types from the global registry."""
    return _registry.get_all_provider_types()


def is_plugin_registered(provider_type: str) -> bool:
    """Check if a provider type is registered in the global registry."""
    return _registry.is_registered(provider_type)


def resolve_provider_alias(provider_type: str) -> str:
    """Resolve a provider alias to its canonical type."""
    return _registry.resolve_alias(provider_type)


def get_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    return _registry


# =============================================================================
# Plugin Registration
# =============================================================================


def register_all_plugins() -> int:
    """
    Register all available provider plugins.

    This function imports and registers all built-in provider plugins.
    Should be called during application startup.

    Returns:
        Number of plugins successfully registered.
    """
    registered_count = 0

    # Import and register each plugin
    plugin_modules = [
        ("openai_plugin", "OpenAIPlugin"),
        ("anthropic_plugin", "AnthropicPlugin"),
        ("google_plugin", "GooglePlugin"),
        ("deepseek_plugin", "DeepSeekPlugin"),
        ("qwen_plugin", "QwenPlugin"),
        ("bigmodel_plugin", "BigModelPlugin"),
        ("routeway_plugin", "RoutewayPlugin"),
        ("ollama_plugin", "OllamaPlugin"),
        ("azure_plugin", "AzurePlugin"),
        ("local_plugin", "LocalPlugin"),
        ("custom_plugin", "CustomPlugin"),
    ]

    for module_name, class_name in plugin_modules:
        try:
            # Dynamic import
            module = __import__(
                f"app.services.provider_plugins.{module_name}",
                fromlist=[class_name]
            )
            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class()
            register_plugin(plugin_instance)
            registered_count += 1
            logger.debug(f"Registered plugin: {class_name}")
        except ImportError as e:
            logger.warning(
                f"Could not import plugin {module_name}: {e}"
            )
        except AttributeError as e:
            logger.warning(
                f"Plugin class {class_name} not found in {module_name}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Error registering plugin {class_name}: {e}"
            )

    logger.info(
        f"Registered {registered_count}/{len(plugin_modules)} provider plugins"
    )
    return registered_count


async def initialize_plugins() -> None:
    """
    Initialize the plugin system asynchronously.

    This function registers all plugins and performs any async initialization.
    """
    logger.info("Initializing provider plugin system...")
    count = register_all_plugins()
    logger.info(f"Provider plugin system initialized with {count} plugins")


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Data classes
    "ModelInfo",
    "ProviderInfo",
    "GenerationRequest",
    "GenerationResponse",
    "StreamChunk",
    "ProviderCapability",
    # Protocol and base class
    "ProviderPlugin",
    "BaseProviderPlugin",
    # Registry
    "PluginRegistry",
    "get_registry",
    # Convenience functions
    "register_plugin",
    "unregister_plugin",
    "get_plugin",
    "get_all_plugins",
    "get_all_provider_types",
    "is_plugin_registered",
    "resolve_provider_alias",
    # Initialization
    "register_all_plugins",
    "initialize_plugins",
]
