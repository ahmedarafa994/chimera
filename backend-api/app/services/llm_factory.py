"""
LLM Factory for Project Chimera.

Factory class for creating and managing LLM adapter instances.
Supports both legacy adapter-based approach and new plugin system.
"""

import logging
from typing import Any, ClassVar

from app.schemas.api_schemas import LLMProvider
from app.services.llm_adapters.base_adapter import BaseLLMAdapter
from app.services.llm_adapters.gemini_adapter import GeminiAdapter
from app.services.llm_adapters.openai_adapter import OpenAIAdapter

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory class to create and retrieve instances of LLM adapters.

    Supports both:
    - Legacy adapter-based approach (OpenAI, Gemini adapters)
    - New plugin-based approach for all 11 provider types
    """

    # Cache for instantiated adapters, keyed by DB model ID
    _adapters: ClassVar[dict[int, BaseLLMAdapter]] = {}

    # Cache for plugin instances, keyed by provider type
    _plugins: ClassVar[dict[str, Any]] = {}

    # Flag to track if plugins have been initialized
    _plugins_initialized: ClassVar[bool] = False

    @classmethod
    def get_adapter(
        cls,
        model_id: int,
        provider: LLMProvider,
        model_name: str,
        api_key: str,
        config: dict[str, Any],
    ) -> BaseLLMAdapter:
        """
        Retrieves an LLM adapter instance. Caches adapters to avoid
        re-initialization.

        Args:
            model_id: Database model ID for caching
            provider: LLM provider type
            model_name: Name of the model to use
            api_key: API key for authentication
            config: Additional configuration options

        Returns:
            BaseLLMAdapter instance

        Raises:
            ValueError: If provider is not supported
        """
        if model_id not in cls._adapters:
            adapter = cls._create_adapter(provider, model_name, api_key, config)
            cls._adapters[model_id] = adapter
        return cls._adapters[model_id]

    @classmethod
    def _create_adapter(
        cls,
        provider: LLMProvider,
        model_name: str,
        api_key: str,
        config: dict[str, Any],
    ) -> BaseLLMAdapter:
        """
        Create a new adapter instance for the given provider.

        Args:
            provider: LLM provider type
            model_name: Name of the model
            api_key: API key
            config: Configuration options

        Returns:
            BaseLLMAdapter instance
        """
        # Try legacy adapters first for backward compatibility
        if provider == LLMProvider.OPENAI:
            return OpenAIAdapter(model_name=model_name, api_key=api_key, config=config)
        elif provider == LLMProvider.GEMINI:
            return GeminiAdapter(model_name=model_name, api_key=api_key, config=config)

        # Try plugin-based approach for other providers
        plugin = cls.get_plugin_for_provider(provider.value)
        if plugin:
            # Create a plugin-based adapter wrapper
            return PluginAdapterWrapper(
                plugin=plugin, model_name=model_name, api_key=api_key, config=config
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def get_plugin_for_provider(cls, provider_type: str) -> Any:
        """
        Get a plugin instance for the given provider type.

        Args:
            provider_type: Provider type string (e.g., "openai", "anthropic")

        Returns:
            Plugin instance or None
        """
        # Initialize plugins if not done
        if not cls._plugins_initialized:
            cls._initialize_plugins()

        # Check cache first
        if provider_type in cls._plugins:
            return cls._plugins[provider_type]

        # Try to get from plugin registry
        try:
            from app.services.provider_plugins import get_plugin

            plugin = get_plugin(provider_type)
            if plugin:
                cls._plugins[provider_type] = plugin
                return plugin
        except ImportError:
            logger.warning("Plugin system not available")

        return None

    @classmethod
    def _initialize_plugins(cls) -> None:
        """Initialize all available plugins."""
        try:
            from app.services.provider_plugins import register_all_plugins

            register_all_plugins()
            cls._plugins_initialized = True
            logger.info("LLM Factory plugins initialized")
        except ImportError:
            logger.warning("Plugin system not available for initialization")
        except Exception as e:
            logger.error(f"Failed to initialize plugins: {e}")

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of all available provider types.

        Returns:
            List of provider type strings
        """
        providers = set()

        # Add legacy providers
        providers.add("openai")
        providers.add("gemini")

        # Add plugin-based providers
        try:
            from app.services.provider_plugins import get_all_plugins

            if not cls._plugins_initialized:
                cls._initialize_plugins()

            plugins = get_all_plugins()
            for plugin in plugins.values():
                providers.add(plugin.provider_type)
        except ImportError:
            pass

        return sorted(providers)

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clears the adapter cache.
        Useful for testing or when configurations change dynamically.
        """
        cls._adapters = {}
        logger.debug("LLM Factory adapter cache cleared")

    @classmethod
    def clear_plugin_cache(cls) -> None:
        """Clear the plugin cache."""
        cls._plugins = {}
        cls._plugins_initialized = False
        logger.debug("LLM Factory plugin cache cleared")

    @classmethod
    def clear_all_caches(cls) -> None:
        """Clear all caches (adapters and plugins)."""
        cls.clear_cache()
        cls.clear_plugin_cache()


class PluginAdapterWrapper(BaseLLMAdapter):
    """
    Wrapper that adapts a ProviderPlugin to the BaseLLMAdapter interface.

    This allows using the new plugin system with code that expects
    the legacy adapter interface.
    """

    def __init__(
        self,
        plugin: Any,
        model_name: str,
        api_key: str,
        config: dict[str, Any],
    ):
        """
        Initialize the wrapper.

        Args:
            plugin: ProviderPlugin instance
            model_name: Name of the model to use
            api_key: API key for authentication
            config: Additional configuration
        """
        self.plugin = plugin
        self.model_name = model_name
        self.api_key = api_key
        self.config = config or {}

    async def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using the plugin.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        from app.services.provider_plugins import GenerationRequest

        # Build request from legacy parameters
        request = GenerationRequest(
            prompt=prompt,
            model=kwargs.get("model", self.model_name),
            temperature=kwargs.get("temperature", self.config.get("temp")),
            max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens")),
            top_p=kwargs.get("top_p"),
            stop_sequences=kwargs.get("stop"),
            system_instruction=kwargs.get("system_prompt"),
        )

        response = await self.plugin.generate(request, api_key=self.api_key)
        return response.text

    async def generate_stream(
        self,
        prompt: str,
        **kwargs: Any,
    ):
        """
        Stream text generation using the plugin.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks
        """
        from app.services.provider_plugins import GenerationRequest

        request = GenerationRequest(
            prompt=prompt,
            model=kwargs.get("model", self.model_name),
            temperature=kwargs.get("temperature", self.config.get("temp")),
            max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens")),
            top_p=kwargs.get("top_p"),
            stop_sequences=kwargs.get("stop"),
            system_instruction=kwargs.get("system_prompt"),
        )

        async for chunk in self.plugin.generate_stream(request, api_key=self.api_key):
            if chunk.text:
                yield chunk.text

    def supports_streaming(self) -> bool:
        """Check if the plugin supports streaming."""
        return self.plugin.supports_streaming()

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        This is a basic implementation; providers may override.
        """
        # Basic estimation: ~4 chars per token for English
        return len(text) // 4

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        try:
            models = await self.plugin.list_models(api_key=self.api_key)
            for model in models:
                if model.model_id == self.model_name:
                    return {
                        "model_id": model.model_id,
                        "display_name": model.display_name,
                        "context_window": model.context_window,
                        "max_output_tokens": model.max_output_tokens,
                        "supports_streaming": model.supports_streaming,
                        "supports_vision": model.supports_vision,
                        "capabilities": model.capabilities,
                    }
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")

        return {
            "model_id": self.model_name,
            "provider": self.plugin.provider_type,
        }


# Convenience function for getting adapters
def get_llm_adapter(
    model_id: int,
    provider: str | LLMProvider,
    model_name: str,
    api_key: str,
    config: dict[str, Any] | None = None,
) -> BaseLLMAdapter:
    """
    Convenience function to get an LLM adapter.

    Args:
        model_id: Database model ID
        provider: Provider type (string or LLMProvider enum)
        model_name: Name of the model
        api_key: API key
        config: Optional configuration

    Returns:
        BaseLLMAdapter instance
    """
    if isinstance(provider, str):
        try:
            provider = LLMProvider(provider.lower())
        except ValueError:
            # Handle as plugin-based provider
            provider = LLMProvider.OPENAI  # Fallback for type
            plugin = LLMFactory.get_plugin_for_provider(provider)
            if plugin:
                return PluginAdapterWrapper(
                    plugin=plugin, model_name=model_name, api_key=api_key, config=config or {}
                )
            raise ValueError(f"Unknown provider: {provider}")

    return LLMFactory.get_adapter(
        model_id=model_id,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        config=config or {},
    )
