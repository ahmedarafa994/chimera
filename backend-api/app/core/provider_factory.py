"""
Unified Provider Factory for LLM clients.

All providers use the modern async implementation pattern with:
- Native async support
- Streaming generation
- Token counting
- Proper error handling

Integration with AIConfigManager for centralized configuration:
- Config-driven provider settings (timeout, retries)
- Alias resolution from configuration
- Provider capability lookup
"""

from typing import Any

from app.core.logging import logger
from app.domain.interfaces import LLMProvider as LLMProviderInterface
from app.infrastructure.anthropic_client import AnthropicClient
from app.infrastructure.bigmodel_client import BigModelClient
from app.infrastructure.cursor_client import CursorClient
from app.infrastructure.deepseek_client import DeepSeekClient
from app.infrastructure.gemini_client import GeminiClient
from app.infrastructure.openai_client import OpenAIClient
from app.infrastructure.qwen_client import QwenClient
from app.infrastructure.routeway_client import RoutewayClient

# Provider name aliases for flexible configuration
# (fallback when config not loaded)
PROVIDER_ALIASES: dict[str, str] = {
    "google": "gemini",
    "gpt": "openai",
    "gpt-4": "openai",
    "gpt-3.5": "openai",
    "claude": "anthropic",
    "claude-3": "anthropic",
    "dashscope": "qwen",
    "alibaba": "qwen",
    "zhipu": "bigmodel",
    "glm": "bigmodel",
}


def _get_config_manager():
    """
    Get AIConfigManager instance with graceful fallback.

    Returns None if config manager is not available (e.g., during startup).
    """
    try:
        from app.core.service_registry import get_ai_config_manager
        config_manager = get_ai_config_manager()
        if config_manager.is_loaded():
            return config_manager
    except Exception as e:
        logger.debug(f"Config manager not available: {e}")
    return None


class ProviderFactory:
    """
    Factory for creating LLM provider clients.

    Integrates with AIConfigManager for:
    - Config-driven provider settings
    - Centralized alias resolution
    - Timeout and retry configuration
    """

    _providers: dict[str, type[LLMProviderInterface]] = {
        "gemini": GeminiClient,
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "deepseek": DeepSeekClient,
        "qwen": QwenClient,
        "cursor": CursorClient,
        "bigmodel": BigModelClient,
        "routeway": RoutewayClient,
    }

    @classmethod
    def resolve_provider_alias(cls, provider_name: str) -> str:
        """
        Resolve a provider alias to its canonical name.

        Uses AIConfigManager if available, falls back to hardcoded aliases.

        Args:
            provider_name: Provider name or alias

        Returns:
            Canonical provider name
        """
        normalized = provider_name.lower().strip()

        # Try config manager first
        config_manager = _get_config_manager()
        if config_manager:
            try:
                return config_manager.resolve_provider_alias(normalized)
            except Exception as e:
                logger.debug(f"Config alias resolution failed: {e}")

        # Fallback to hardcoded aliases
        return PROVIDER_ALIASES.get(normalized, normalized)

    @classmethod
    def get_provider_config(
        cls, provider_name: str
    ) -> dict[str, Any] | None:
        """
        Get configuration for a provider from AIConfigManager.

        Args:
            provider_name: Provider name or alias

        Returns:
            Provider configuration dict or None if not available
        """
        config_manager = _get_config_manager()
        if not config_manager:
            logger.debug(
                f"Config manager not available for provider: {provider_name}"
            )
            return None

        try:
            resolved = cls.resolve_provider_alias(provider_name)
            provider_config = config_manager.get_provider(resolved)

            if provider_config:
                caps = provider_config.capabilities
                cb = provider_config.circuit_breaker
                rl = provider_config.rate_limits

                return {
                    "provider_id": provider_config.provider_id,
                    "name": provider_config.name,
                    "enabled": provider_config.enabled,
                    "priority": provider_config.priority,
                    "api": {
                        "base_url": provider_config.api.base_url,
                        "timeout_seconds": provider_config.api.timeout_seconds,
                        "max_retries": provider_config.api.max_retries,
                    },
                    "capabilities": {
                        "supports_streaming": caps.supports_streaming,
                        "supports_vision": caps.supports_vision,
                        "supports_function_calling": (
                            caps.supports_function_calling
                        ),
                        "supports_json_mode": caps.supports_json_mode,
                    },
                    "circuit_breaker": {
                        "enabled": cb.enabled,
                        "failure_threshold": cb.failure_threshold,
                        "recovery_timeout_seconds": (
                            cb.recovery_timeout_seconds
                        ),
                    },
                    "rate_limits": {
                        "requests_per_minute": rl.requests_per_minute,
                        "tokens_per_minute": rl.tokens_per_minute,
                    },
                    "models": list(provider_config.models.keys()),
                    "failover_chain": provider_config.failover_chain,
                }

            logger.warning(f"Provider '{provider_name}' not found in config")
            return None

        except Exception as e:
            logger.warning(
                f"Failed to get provider config for '{provider_name}': {e}"
            )
            return None

    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        use_config: bool = True,
    ) -> LLMProviderInterface:
        """
        Create a provider instance by name.

        Args:
            provider_name: Provider name or alias
                (e.g., 'openai', 'google', 'claude')
            use_config: Whether to use AIConfigManager for settings
                (default True)

        Returns:
            LLMProviderInterface: Configured provider instance

        Raises:
            ValueError: If provider name is not recognized
        """
        # Resolve alias using config or fallback
        resolved = cls.resolve_provider_alias(provider_name)

        # Check if provider is registered
        provider_class = cls._providers.get(resolved)
        if not provider_class:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {available}"
            )

        # Get config-driven settings if available
        provider_config = None
        if use_config:
            provider_config = cls.get_provider_config(resolved)

            if provider_config:
                # Validate provider is enabled in config
                if not provider_config.get("enabled", True):
                    logger.warning(
                        f"Provider '{resolved}' is disabled in config, "
                        "creating anyway for backward compatibility"
                    )

                api_cfg = provider_config["api"]
                logger.info(
                    f"Creating provider: {resolved} "
                    f"(timeout={api_cfg['timeout_seconds']}s, "
                    f"retries={api_cfg['max_retries']})"
                )
            else:
                logger.info(
                    f"Creating provider: {resolved} "
                    "(using defaults, config not available)"
                )
        else:
            logger.info(f"Creating provider: {resolved} (config disabled)")

        # Create provider instance
        # Individual clients may use config internally
        return provider_class()

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Return list of available provider names.

        Uses config if available, otherwise returns registered providers.
        """
        config_manager = _get_config_manager()
        if config_manager:
            try:
                # Get enabled providers from config
                enabled = config_manager.get_enabled_providers()
                return [p.provider_id for p in enabled]
            except Exception as e:
                logger.debug(f"Failed to get providers from config: {e}")

        # Fallback to registered providers
        return list(cls._providers.keys())

    @classmethod
    def is_provider_enabled(cls, provider_name: str) -> bool:
        """
        Check if a provider is enabled in configuration.

        Args:
            provider_name: Provider name or alias

        Returns:
            True if enabled or config not available (backward compatibility)
        """
        config = cls.get_provider_config(provider_name)
        if config is None:
            # Default to enabled if config not available
            return True
        return config.get("enabled", True)

    @classmethod
    def get_provider_timeout(cls, provider_name: str) -> int:
        """
        Get timeout setting for a provider.

        Args:
            provider_name: Provider name or alias

        Returns:
            Timeout in seconds (default 120 if not configured)
        """
        config = cls.get_provider_config(provider_name)
        if config:
            return config.get("api", {}).get("timeout_seconds", 120)
        return 120

    @classmethod
    def get_provider_retries(cls, provider_name: str) -> int:
        """
        Get max retries setting for a provider.

        Args:
            provider_name: Provider name or alias

        Returns:
            Max retries (default 3 if not configured)
        """
        config = cls.get_provider_config(provider_name)
        if config:
            return config.get("api", {}).get("max_retries", 3)
        return 3

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[LLMProviderInterface]
    ) -> None:
        """
        Register a custom provider.

        Args:
            name: Provider name for lookup
            provider_class: Class implementing LLMProviderInterface
        """
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered custom provider: {name}")
