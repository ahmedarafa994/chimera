from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from app.domain.models import (
    Capability,
    Model,
    PromptRequest,
    PromptResponse,
    Provider,
    StreamChunk,
)


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Generate text based on the prompt request."""

    @abstractmethod
    async def check_health(self) -> bool:
        """Check if the provider is available."""

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation token by token.

        This is an optional method - providers that don't support streaming
        should not override this method (it will raise NotImplementedError).

        Yields:
            StreamChunk: Individual chunks of generated text with metadata.

        """
        msg = "Streaming not supported by this provider"
        raise NotImplementedError(msg)

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count the number of tokens in the given text.

        This is an optional method - providers that don't support token counting
        should not override this method (it will raise NotImplementedError).

        Args:
            text: The text to count tokens for.
            model: Optional model name for model-specific tokenization.

        Returns:
            int: The number of tokens in the text.

        """
        msg = "Token counting not supported by this provider"
        raise NotImplementedError(msg)


# =============================================================================
# Unified Provider System Interfaces
# =============================================================================


@runtime_checkable
class BaseLLMClient(Protocol):
    """Abstract interface that all LLM provider clients must implement.

    This protocol defines the standard contract for interacting with any LLM provider,
    allowing the system to work with different providers in a uniform way.

    The @runtime_checkable decorator enables isinstance() checks at runtime.
    """

    @property
    def provider_id(self) -> str:
        """Get the provider identifier (e.g., 'openai', 'anthropic')."""
        ...

    @property
    def model_id(self) -> str:
        """Get the model identifier (e.g., 'gpt-4-turbo', 'claude-3-5-sonnet')."""
        ...

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        **kwargs,
    ) -> PromptResponse:
        """Generate text completion from the LLM.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (None for provider default)
            system_instruction: Optional system instruction/prompt
            **kwargs: Provider-specific parameters

        Returns:
            PromptResponse with generated text and metadata

        Raises:
            ProviderException: If the provider API call fails
            ValidationException: If parameters are invalid

        """
        ...

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream text completion from the LLM.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (None for provider default)
            system_instruction: Optional system instruction/prompt
            **kwargs: Provider-specific parameters

        Yields:
            Text chunks as they are generated

        Raises:
            ProviderException: If the provider API call fails
            ValidationException: If parameters are invalid

        """
        ...

    def get_capabilities(self) -> set[Capability]:
        """Get the capabilities supported by this client's model.

        Returns:
            Set of Capability enums

        """
        ...

    async def close(self) -> None:
        """Clean up resources (close connections, clear caches, etc.).

        This is called when the client is no longer needed, typically at
        application shutdown or when switching providers.
        """
        ...


@runtime_checkable
class ProviderPlugin(Protocol):
    """Plugin interface for provider implementations.

    Each provider (OpenAI, Anthropic, Google, etc.) implements this protocol
    to register itself with the ProviderRegistry and provide factory methods
    for creating clients.
    """

    @property
    def provider_id(self) -> str:
        """Get the unique provider identifier.

        Returns:
            Provider ID (e.g., 'openai', 'anthropic', 'google')

        """
        ...

    @property
    def provider_metadata(self) -> Provider:
        """Get complete provider metadata.

        Returns:
            Provider model with all configuration details

        """
        ...

    def get_available_models(self) -> list[Model]:
        """Get list of models available from this provider.

        Returns:
            List of Model instances representing available models

        """
        ...

    def create_client(self, model_id: str, **kwargs) -> BaseLLMClient:
        """Factory method to create a client for a specific model.

        Args:
            model_id: The model to create a client for
            **kwargs: Additional configuration parameters

        Returns:
            BaseLLMClient instance configured for the specified model

        Raises:
            ValueError: If model_id is not available from this provider
            ConfigurationException: If required configuration is missing

        """
        ...

    def validate_config(self) -> bool:
        """Validate that the provider is properly configured.

        Checks for API keys, endpoints, and other required configuration.

        Returns:
            True if configuration is valid, False otherwise

        """
        ...

    async def health_check(self) -> bool:
        """Perform a health check on the provider's API.

        Makes a lightweight API call to verify connectivity and credentials.

        Returns:
            True if provider is healthy and accessible, False otherwise

        """
        ...
