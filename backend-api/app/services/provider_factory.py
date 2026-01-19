"""
Provider Factory Service - Creates LLM clients based on selection context.

This module implements the factory pattern for creating BaseLLMClient instances
based on the current selection context. It integrates with UnifiedProviderRegistry
and SelectionContext to provide a clean interface for obtaining configured clients.

Key Features:
- Context-aware client creation
- Selection validation and fallback
- Client caching for performance
- Support for selection overrides
- Automatic cleanup and resource management
"""

import logging
from typing import Any

from app.core.selection_context import SelectionContext
from app.domain.interfaces import BaseLLMClient
from app.domain.models import Selection
from app.services.unified_provider_registry import unified_registry

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating LLM clients based on provider/model selection.

    This factory integrates with SelectionContext and UnifiedProviderRegistry
    to provide a unified interface for client creation across the application.

    Usage:
        >>> factory = ProviderFactory()
        >>>
        >>> # Create client from current context
        >>> client = factory.get_client_from_context()
        >>>
        >>> # Create client with override
        >>> client = factory.get_client("openai", "gpt-4-turbo")
        >>>
        >>> # Create client with fallback
        >>> client = factory.get_client_with_fallback("openai", "gpt-4")
    """

    def __init__(self, enable_caching: bool = True):
        """
        Initialize the provider factory.

        Args:
            enable_caching: Whether to cache client instances (default: True)
        """
        self._client_cache: dict[str, BaseLLMClient] = {}
        self._enable_caching = enable_caching
        logger.debug(
            f"ProviderFactory initialized (caching={'enabled' if enable_caching else 'disabled'})"
        )

    def get_client_from_context(self, **kwargs: Any) -> BaseLLMClient:
        """
        Create a client based on the current selection context.

        This is the primary method for creating clients within request handlers.
        It reads the selection from SelectionContext and creates an appropriate client.

        Args:
            **kwargs: Additional configuration parameters for client creation

        Returns:
            BaseLLMClient instance configured for the current selection

        Raises:
            ValueError: If no selection is set in context
            KeyError: If provider or model is not registered
            RuntimeError: If client creation fails

        Example:
            >>> # Within a request handler (selection already set by middleware)
            >>> factory = ProviderFactory()
            >>> client = factory.get_client_from_context()
            >>> response = await client.generate("Hello, world!")
        """
        selection = SelectionContext.get_selection()
        if selection is None:
            raise ValueError(
                "No provider selection found in context. "
                "Ensure selection middleware is configured correctly."
            )

        logger.debug(
            f"Creating client from context: {selection.provider_id}/{selection.model_id} "
            f"(scope={selection.scope.value})"
        )

        return self._create_client_internal(
            provider_id=selection.provider_id,
            model_id=selection.model_id,
            selection=selection,
            **kwargs,
        )

    def get_client(self, provider_id: str, model_id: str, **kwargs: Any) -> BaseLLMClient:
        """
        Create a client for a specific provider and model.

        This method bypasses the selection context and creates a client directly.
        Useful for administrative tasks or background jobs.

        Args:
            provider_id: The provider ID or alias
            model_id: The model identifier
            **kwargs: Additional configuration parameters

        Returns:
            BaseLLMClient instance configured for the specified provider/model

        Raises:
            KeyError: If provider or model is not registered
            ValueError: If provider/model combination is invalid
            RuntimeError: If client creation fails

        Example:
            >>> factory = ProviderFactory()
            >>> client = factory.get_client("openai", "gpt-4-turbo")
            >>> response = await client.generate("Hello, world!")
        """
        logger.debug(f"Creating client directly: {provider_id}/{model_id}")

        # Validate selection
        if not unified_registry.validate_selection(provider_id, model_id):
            raise ValueError(
                f"Invalid provider/model combination: {provider_id}/{model_id}. "
                "Model may not be available for this provider."
            )

        return self._create_client_internal(
            provider_id=provider_id, model_id=model_id, selection=None, **kwargs
        )

    def get_client_with_fallback(
        self,
        provider_id: str,
        model_id: str,
        fallback_provider: str | None = None,
        fallback_model: str | None = None,
        **kwargs: Any,
    ) -> BaseLLMClient:
        """
        Create a client with automatic fallback if primary selection fails.

        Attempts to create a client for the primary provider/model. If that fails,
        attempts to create a client for the fallback provider/model.

        Args:
            provider_id: Primary provider ID
            model_id: Primary model ID
            fallback_provider: Fallback provider ID (optional)
            fallback_model: Fallback model ID (optional)
            **kwargs: Additional configuration parameters

        Returns:
            BaseLLMClient instance (either primary or fallback)

        Raises:
            RuntimeError: If both primary and fallback creation fail

        Example:
            >>> factory = ProviderFactory()
            >>> client = factory.get_client_with_fallback(
            ...     "openai", "gpt-4-turbo",
            ...     fallback_provider="anthropic",
            ...     fallback_model="claude-3-5-sonnet-20241022"
            ... )
        """
        try:
            return self.get_client(provider_id, model_id, **kwargs)
        except Exception as e:
            logger.warning(
                f"Failed to create client for {provider_id}/{model_id}: {e}. "
                f"Attempting fallback..."
            )

            if not fallback_provider or not fallback_model:
                raise RuntimeError(
                    f"Failed to create client for {provider_id}/{model_id} and no fallback specified"
                ) from e

            try:
                logger.info(f"Using fallback: {fallback_provider}/{fallback_model}")
                return self.get_client(fallback_provider, fallback_model, **kwargs)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to create both primary ({provider_id}/{model_id}) "
                    f"and fallback ({fallback_provider}/{fallback_model}) clients"
                ) from fallback_error

    def _create_client_internal(
        self, provider_id: str, model_id: str, selection: Selection | None, **kwargs: Any
    ) -> BaseLLMClient:
        """
        Internal method to create a client with caching support.

        Args:
            provider_id: The provider ID
            model_id: The model ID
            selection: Optional Selection object for context
            **kwargs: Additional configuration parameters

        Returns:
            BaseLLMClient instance

        Raises:
            KeyError: If provider is not registered
            RuntimeError: If client creation fails
        """
        # Generate cache key
        cache_key = self._generate_cache_key(provider_id, model_id, kwargs)

        # Check cache
        if self._enable_caching and cache_key in self._client_cache:
            logger.debug(f"Using cached client: {cache_key}")
            return self._client_cache[cache_key]

        # Create new client
        try:
            client = unified_registry.create_client(provider_id, model_id, **kwargs)

            # Cache the client
            if self._enable_caching:
                self._client_cache[cache_key] = client
                logger.debug(f"Cached new client: {cache_key}")

            return client

        except KeyError as e:
            logger.error(f"Provider '{provider_id}' not registered: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create client for {provider_id}/{model_id}: {e}")
            raise RuntimeError(f"Client creation failed for {provider_id}/{model_id}: {e!s}") from e

    def _generate_cache_key(self, provider_id: str, model_id: str, kwargs: dict[str, Any]) -> str:
        """
        Generate a cache key for client instances.

        Args:
            provider_id: Provider ID
            model_id: Model ID
            kwargs: Additional parameters

        Returns:
            Cache key string
        """
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = ",".join(f"{k}={v}" for k, v in sorted_kwargs)
        return f"{provider_id}:{model_id}:{kwargs_str}"

    async def cleanup(self) -> None:
        """
        Clean up all cached clients.

        Calls close() on all cached clients to release resources.
        Should be called at application shutdown.
        """
        logger.info(f"Cleaning up {len(self._client_cache)} cached clients")

        for cache_key, client in self._client_cache.items():
            try:
                await client.close()
                logger.debug(f"Closed client: {cache_key}")
            except Exception as e:
                logger.error(f"Failed to close client {cache_key}: {e}")

        self._client_cache.clear()
        logger.info("Client cache cleared")

    def clear_cache(self) -> None:
        """
        Clear the client cache without closing connections.

        Use this for testing or when you want to force recreation of clients.
        For production cleanup, use cleanup() instead.
        """
        count = len(self._client_cache)
        self._client_cache.clear()
        logger.info(f"Cleared {count} clients from cache (connections not closed)")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get statistics about the client cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_enabled": self._enable_caching,
            "cached_clients": len(self._client_cache),
            "cache_keys": list(self._client_cache.keys()) if self._enable_caching else [],
        }


# Global factory instance
provider_factory = ProviderFactory()


# Convenience function for FastAPI dependency injection
def get_llm_client(**kwargs: Any) -> BaseLLMClient:
    """
    FastAPI dependency to get an LLM client from context.

    Usage in route:
        @app.post("/api/v1/generate")
        async def generate(
            request: PromptRequest,
            client: BaseLLMClient = Depends(get_llm_client)
        ):
            response = await client.generate(request.prompt)
            return response

    Args:
        **kwargs: Additional configuration parameters

    Returns:
        BaseLLMClient instance from current context

    Raises:
        ValueError: If no selection is set in context
    """
    return provider_factory.get_client_from_context(**kwargs)
