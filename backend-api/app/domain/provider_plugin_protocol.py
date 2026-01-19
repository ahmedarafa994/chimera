"""
Provider Plugin Protocol

Standardized interface that all AI provider implementations must follow.
This protocol enables consistent integration of all providers (OpenAI, Anthropic,
Google/Gemini, DeepSeek, Azure, local models, etc.) with the unified system.

Features:
- Standard interface for provider metadata and capabilities
- Unified client creation and configuration
- Normalized completion and streaming methods
- Consistent error handling across providers
- Standard response format

Usage:
    class MyProviderPlugin(IProviderPlugin):
        def get_provider_id(self) -> str:
            return "my-provider"

        async def get_available_models(self) -> List[Model]:
            return [...]

        async def handle_completion(self, client, request) -> CompletionResponse:
            ...
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel


class ProviderCapabilities(BaseModel):
    """Provider capability flags"""

    supports_streaming: bool = True
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_system_prompt: bool = True
    supports_token_counting: bool = False
    max_context_length: int = 4096
    max_output_tokens: int = 4096
    supports_embeddings: bool = False
    supports_fine_tuning: bool = False


class Model(BaseModel):
    """Model information"""

    id: str
    provider_id: str
    display_name: str
    description: str | None = None
    capabilities: dict[str, Any] = {}
    parameters: dict[str, Any] = {}


class CompletionRequest(BaseModel):
    """Unified completion request format"""

    model: str
    messages: list[dict[str, str]]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # Additional provider-specific parameters
    extra_params: dict[str, Any] = {}


class Choice(BaseModel):
    """Completion choice"""

    index: int
    message: dict[str, str]
    finish_reason: str | None = None


class Usage(BaseModel):
    """Token usage information"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Unified completion response format"""

    id: str
    provider: str
    model: str
    choices: list[Choice]
    usage: Usage
    created_at: int


class StreamChunk(BaseModel):
    """Streaming response chunk"""

    id: str
    provider: str
    model: str
    choices: list[dict[str, Any]]
    created_at: int


class NormalizedError(BaseModel):
    """Standardized error response"""

    code: str
    message: str
    provider: str
    model: str
    is_retryable: bool
    user_message: str
    original_error: Any = None


class IProviderPlugin(ABC):
    """
    Interface that all provider plugins must implement.

    This protocol ensures consistent behavior across all AI providers,
    enabling the unified selection system to work seamlessly.
    """

    @abstractmethod
    def get_provider_id(self) -> str:
        """
        Get the canonical provider ID.

        Returns:
            Provider ID (e.g., "openai", "anthropic", "google")

        Example:
            return "openai"
        """
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """
        Get the user-facing provider display name.

        Returns:
            Display name (e.g., "OpenAI", "Google AI (Gemini)")

        Example:
            return "OpenAI"
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """
        Get provider capabilities and limits.

        Returns:
            ProviderCapabilities object

        Example:
            return ProviderCapabilities(
                supports_streaming=True,
                supports_vision=True,
                max_context_length=128000
            )
        """
        pass

    @abstractmethod
    async def validate_config(self, api_key: str, config: dict[str, Any] | None = None) -> bool:
        """
        Validate provider configuration and API key.

        Args:
            api_key: API key to validate
            config: Additional configuration parameters

        Returns:
            True if valid, False otherwise

        Example:
            try:
                client = OpenAI(api_key=api_key)
                await client.models.list()
                return True
            except:
                return False
        """
        pass

    @abstractmethod
    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        """
        Get list of available models for this provider.

        Args:
            api_key: Optional API key for fetching live models

        Returns:
            List of Model objects

        Example:
            return [
                Model(id="gpt-4o", provider_id="openai", display_name="GPT-4o"),
                Model(id="gpt-4-turbo", provider_id="openai", display_name="GPT-4 Turbo")
            ]
        """
        pass

    @abstractmethod
    async def create_client(self, api_key: str, config: dict[str, Any] | None = None) -> Any:
        """
        Create a provider-specific client instance.

        Args:
            api_key: API key for authentication
            config: Additional configuration (base_url, timeout, etc.)

        Returns:
            Provider-specific client object

        Example:
            return OpenAI(api_key=api_key, base_url=config.get("base_url"))
        """
        pass

    @abstractmethod
    async def handle_completion(
        self, client: Any, request: CompletionRequest
    ) -> CompletionResponse:
        """
        Handle a completion request.

        Args:
            client: Provider client (from create_client)
            request: Unified completion request

        Returns:
            Unified completion response

        Example:
            response = await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature
            )
            return self.normalize_response(response)
        """
        pass

    @abstractmethod
    async def handle_streaming(
        self, client: Any, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Handle a streaming completion request.

        Args:
            client: Provider client (from create_client)
            request: Unified completion request

        Returns:
            Async iterator of stream chunks

        Example:
            stream = await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                stream=True
            )
            async for chunk in stream:
                yield self.normalize_chunk(chunk)
        """
        pass

    @abstractmethod
    def normalize_error(self, error: Exception) -> NormalizedError:
        """
        Convert provider-specific error to standardized format.

        Args:
            error: Provider-specific exception

        Returns:
            NormalizedError with standard fields

        Example:
            is_retryable = "rate_limit" in str(error).lower() or error.status_code == 429
            return NormalizedError(
                code="RATE_LIMIT" if is_retryable else "API_ERROR",
                message=str(error),
                provider=self.get_provider_id(),
                model="unknown",
                is_retryable=is_retryable,
                user_message="Rate limit exceeded. Please try again later."
            )
        """
        pass

    @abstractmethod
    def is_retryable(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: Exception to check

        Returns:
            True if should retry, False otherwise

        Example:
            error_str = str(error).lower()
            return ("rate_limit" in error_str or
                    "timeout" in error_str or
                    "503" in error_str)
        """
        pass

    @abstractmethod
    def normalize_response(self, response: Any) -> CompletionResponse:
        """
        Convert provider-specific response to standard format.

        Args:
            response: Provider-specific response object

        Returns:
            Unified CompletionResponse

        Example:
            return CompletionResponse(
                id=response.id,
                provider=self.get_provider_id(),
                model=response.model,
                choices=[Choice(
                    index=0,
                    message={"role": "assistant", "content": response.choices[0].message.content},
                    finish_reason=response.choices[0].finish_reason
                )],
                usage=Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                ),
                created_at=response.created
            )
        """
        pass


class BaseProviderPlugin(IProviderPlugin):
    """
    Base class providing common functionality for provider plugins.

    Subclasses should override abstract methods but can use the
    utility methods provided here.
    """

    def __init__(self):
        self._capabilities: ProviderCapabilities | None = None

    def _get_error_code(self, error: Exception) -> str:
        """Helper to extract error code from exception."""
        error_str = str(error).lower()

        if "rate_limit" in error_str or "429" in error_str:
            return "RATE_LIMIT"
        elif "unauthorized" in error_str or "401" in error_str:
            return "UNAUTHORIZED"
        elif "forbidden" in error_str or "403" in error_str:
            return "FORBIDDEN"
        elif "not_found" in error_str or "404" in error_str:
            return "NOT_FOUND"
        elif "timeout" in error_str:
            return "TIMEOUT"
        elif "503" in error_str:
            return "SERVICE_UNAVAILABLE"
        else:
            return "API_ERROR"

    def _get_user_friendly_message(self, error: Exception) -> str:
        """Helper to generate user-friendly error message."""
        error_code = self._get_error_code(error)

        messages = {
            "RATE_LIMIT": "Rate limit exceeded. Please wait a moment and try again.",
            "UNAUTHORIZED": "Invalid API key. Please check your configuration.",
            "FORBIDDEN": "Access forbidden. Please check your permissions.",
            "NOT_FOUND": "Resource not found. The model may not be available.",
            "TIMEOUT": "Request timed out. Please try again.",
            "SERVICE_UNAVAILABLE": "Service temporarily unavailable. Please try again later.",
        }

        return messages.get(error_code, "An error occurred. Please try again later.")
