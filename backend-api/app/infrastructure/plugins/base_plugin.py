"""
Base Provider Plugin - Abstract base class for provider implementations.

This module provides the BaseProviderPlugin abstract class that all provider
plugins inherit from. It implements common functionality and defines the
interface that concrete providers must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.domain.interfaces import BaseLLMClient, ProviderPlugin
from app.domain.models import Capability, Model, Provider

logger = logging.getLogger(__name__)


class BaseProviderPlugin(ABC, ProviderPlugin):
    """
    Abstract base class for provider plugin implementations.

    Provides common functionality and defines the contract that all
    provider plugins must implement. Concrete providers should inherit
    from this class and implement the abstract methods.

    Features:
    - Configuration validation helpers
    - Default health check implementation
    - Model metadata management
    - Capability checking
    - Logging and error handling
    """

    def __init__(
        self,
        provider_id: str,
        provider_name: str,
        api_key_env_var: str,
        base_url: Optional[str] = None,
        is_enabled: bool = True,
    ):
        """
        Initialize base provider plugin.

        Args:
            provider_id: Unique identifier for the provider (e.g., 'openai')
            provider_name: Human-readable provider name (e.g., 'OpenAI')
            api_key_env_var: Environment variable name for API key
            base_url: Optional base URL for API endpoint
            is_enabled: Whether provider is enabled (default: True)
        """
        self._provider_id = provider_id
        self._provider_name = provider_name
        self._api_key_env_var = api_key_env_var
        self._base_url = base_url
        self._is_enabled = is_enabled
        self._models_cache: Optional[list[Model]] = None

        logger.debug(f"Initialized {provider_name} provider plugin")

    @property
    def provider_id(self) -> str:
        """Get the provider identifier."""
        return self._provider_id

    @property
    def provider_metadata(self) -> Provider:
        """
        Get complete provider metadata.

        Returns a Provider model with all configuration details.
        Must be implemented by concrete providers to include
        provider-specific capabilities and pricing information.
        """
        return Provider(
            id=self._provider_id,
            name=self._provider_name,
            is_enabled=self._is_enabled,
            base_url=self._base_url,
            capabilities=self._get_provider_capabilities(),
            models=self.get_available_models(),
        )

    @abstractmethod
    def _get_provider_capabilities(self) -> set[Capability]:
        """
        Get the capabilities supported by this provider.

        Must be implemented by concrete providers to specify
        which capabilities they support (streaming, function calling, etc.).

        Returns:
            Set of Capability enums
        """
        pass

    @abstractmethod
    def _get_api_key(self) -> Optional[str]:
        """
        Get the API key for this provider.

        Must be implemented by concrete providers to retrieve
        the API key from environment or configuration.

        Returns:
            API key string or None if not configured
        """
        pass

    @abstractmethod
    def _build_model_list(self) -> list[Model]:
        """
        Build the list of available models for this provider.

        Must be implemented by concrete providers to return
        their specific model catalog.

        Returns:
            List of Model instances
        """
        pass

    def get_available_models(self) -> list[Model]:
        """
        Get list of models available from this provider.

        Implements caching to avoid rebuilding the model list
        on every call. Cache is cleared when provider is re-registered.

        Returns:
            List of Model instances
        """
        if self._models_cache is None:
            try:
                self._models_cache = self._build_model_list()
                logger.debug(
                    f"Built model list for {self._provider_name}: "
                    f"{len(self._models_cache)} models"
                )
            except Exception as e:
                logger.error(f"Failed to build model list for {self._provider_name}: {e}")
                self._models_cache = []

        return self._models_cache

    def clear_models_cache(self) -> None:
        """
        Clear the cached model list.

        Useful when models need to be refreshed (e.g., after
        provider configuration changes).
        """
        self._models_cache = None
        logger.debug(f"Cleared model cache for {self._provider_name}")

    @abstractmethod
    def create_client(self, model_id: str, **kwargs: Any) -> BaseLLMClient:
        """
        Factory method to create a client for a specific model.

        Must be implemented by concrete providers to create
        provider-specific client instances.

        Args:
            model_id: The model to create a client for
            **kwargs: Additional configuration parameters

        Returns:
            BaseLLMClient instance configured for the specified model

        Raises:
            ValueError: If model_id is not available from this provider
            RuntimeError: If client creation fails
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate that the provider is properly configured.

        Checks for API keys and other required configuration.
        Can be overridden by concrete providers for additional checks.

        Returns:
            True if configuration is valid, False otherwise
        """
        api_key = self._get_api_key()

        if not api_key:
            logger.warning(
                f"{self._provider_name} API key not configured. "
                f"Set {self._api_key_env_var} environment variable."
            )
            return False

        if len(api_key) < 10:
            logger.warning(f"{self._provider_name} API key appears to be invalid (too short)")
            return False

        logger.debug(f"{self._provider_name} configuration is valid")
        return True

    async def health_check(self) -> bool:
        """
        Perform a health check on the provider's API.

        Default implementation validates configuration. Can be overridden
        by concrete providers to make actual API calls.

        Returns:
            True if provider is healthy and accessible, False otherwise
        """
        try:
            # Basic validation
            if not self.validate_config():
                return False

            # Check if any models are available
            models = self.get_available_models()
            if not models:
                logger.warning(f"{self._provider_name} has no available models")
                return False

            logger.debug(f"{self._provider_name} health check passed")
            return True

        except Exception as e:
            logger.error(f"{self._provider_name} health check failed: {e}")
            return False

    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """
        Get a specific model by ID.

        Utility method for finding a model in the provider's catalog.

        Args:
            model_id: The model identifier to search for

        Returns:
            Model instance if found, None otherwise
        """
        models = self.get_available_models()

        for model in models:
            if model.id == model_id:
                return model

        return None

    def validate_model_id(self, model_id: str) -> bool:
        """
        Validate that a model ID is available from this provider.

        Args:
            model_id: The model identifier to validate

        Returns:
            True if model is available, False otherwise
        """
        return self.get_model_by_id(model_id) is not None

    def get_models_with_capabilities(
        self, capabilities: set[Capability]
    ) -> list[Model]:
        """
        Get models that support specific capabilities.

        Args:
            capabilities: Set of required capabilities

        Returns:
            List of models that support all specified capabilities
        """
        models = self.get_available_models()

        return [
            model for model in models
            if capabilities.issubset(model.capabilities)
        ]

    def __repr__(self) -> str:
        """String representation of the plugin."""
        return (
            f"<{self.__class__.__name__} "
            f"provider={self._provider_id} "
            f"enabled={self._is_enabled} "
            f"models={len(self.get_available_models())}>"
        )
