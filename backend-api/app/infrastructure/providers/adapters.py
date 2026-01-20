"""Provider Adapters.

Adapts existing BaseProvider implementations to the IProviderPlugin interface.
This enables backward compatibility while integrating with the unified system.

Features:
- Adapts existing providers (OpenAI, Anthropic, Google, DeepSeek, etc.)
- Implements IProviderPlugin interface
- Bridges old and new architectures
- Maintains provider-specific functionality

Usage:
    adapter = OpenAIProviderAdapter()
    models = await adapter.get_available_models()
    client = await adapter.create_client(api_key)
"""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import get_settings
from app.domain.models import GenerationConfig, PromptRequest
from app.domain.provider_plugin_protocol import (
    BaseProviderPlugin,
    Choice,
    CompletionRequest,
    CompletionResponse,
    Model,
    NormalizedError,
    ProviderCapabilities,
    StreamChunk,
    Usage,
)
from app.infrastructure.providers.anthropic_provider import AnthropicProvider
from app.infrastructure.providers.bigmodel_client import BigModelClient
from app.infrastructure.providers.cursor_client import CursorClient
from app.infrastructure.providers.deepseek_client import DeepSeekClient
from app.infrastructure.providers.google_provider import GoogleProvider
from app.infrastructure.providers.openai_provider import OpenAIProvider
from app.infrastructure.providers.qwen_client import QwenClient
from app.infrastructure.providers.routeway_client import RoutewayClient


class ProviderAdapter(BaseProviderPlugin):
    """Base adapter class for wrapping existing BaseProvider implementations."""

    def __init__(self, base_provider: Any, provider_id: str, display_name: str) -> None:
        """Initialize the adapter.

        Args:
            base_provider: Existing BaseProvider instance
            provider_id: Canonical provider ID
            display_name: Display name for UI

        """
        super().__init__()
        self._base_provider = base_provider
        self._provider_id = provider_id
        self._display_name = display_name
        self._settings = get_settings()

    def get_provider_id(self) -> str:
        return self._provider_id

    def get_display_name(self) -> str:
        return self._display_name

    async def validate_config(self, api_key: str, config: dict[str, Any] | None = None) -> bool:
        """Validate configuration by checking health with the API key."""
        try:
            # Create temporary provider instance with API key
            return self._base_provider.validate_api_key(api_key)
        except Exception:
            return False

    async def create_client(self, api_key: str, config: dict[str, Any] | None = None) -> Any:
        """Create client - delegates to base provider's client initialization."""
        return self._base_provider._get_client(api_key)

    async def handle_completion(
        self,
        client: Any,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Handle completion request by converting to base provider format."""
        # Convert CompletionRequest to PromptRequest
        prompt_request = self._convert_to_prompt_request(request)

        # Call base provider
        response = await self._base_provider.generate(prompt_request)

        # Convert response to CompletionResponse
        return CompletionResponse(
            id=str(uuid.uuid4()),
            provider=self._provider_id,
            model=response.model,
            choices=[
                Choice(
                    index=0,
                    message={"role": "assistant", "content": response.text},
                    finish_reason=response.finish_reason,
                ),
            ],
            usage=Usage(
                prompt_tokens=response.input_tokens or 0,
                completion_tokens=response.output_tokens or 0,
                total_tokens=response.total_tokens or 0,
            ),
            created_at=int(time.time()),
        )

    async def handle_streaming(
        self,
        client: Any,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Handle streaming request."""
        # Convert to PromptRequest
        prompt_request = self._convert_to_prompt_request(request)

        # Stream from base provider
        async for chunk in self._base_provider.generate_stream(prompt_request):
            yield StreamChunk(
                id=str(uuid.uuid4()),
                provider=self._provider_id,
                model=request.model,
                choices=[
                    {
                        "delta": {"content": chunk.text},
                        "finish_reason": chunk.finish_reason,
                        "index": 0,
                    },
                ],
                created_at=int(time.time()),
            )

    def normalize_error(self, error: Exception) -> NormalizedError:
        """Normalize error using base class helper methods."""
        error_code = self._get_error_code(error)
        return NormalizedError(
            code=error_code,
            message=str(error),
            provider=self._provider_id,
            model="unknown",
            is_retryable=self.is_retryable(error),
            user_message=self._get_user_friendly_message(error),
            original_error=error,
        )

    def is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable."""
        from app.infrastructure.providers.base import is_retryable_error

        return is_retryable_error(error)

    def normalize_response(self, response: Any) -> CompletionResponse:
        """Normalize response - base providers already return normalized format."""
        # This is handled in handle_completion

    def _convert_to_prompt_request(self, request: CompletionRequest) -> PromptRequest:
        """Convert CompletionRequest to PromptRequest for base provider."""
        # Extract system instruction and user prompt from messages
        system_instruction = None
        prompt = ""

        for msg in request.messages:
            if msg.get("role") == "system":
                system_instruction = msg.get("content", "")
            elif msg.get("role") == "user":
                prompt += msg.get("content", "") + "\n"

        # Build generation config
        config = None
        if request.temperature is not None or request.max_tokens is not None:
            config = GenerationConfig(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.extra_params.get("top_p"),
            )

        return PromptRequest(
            prompt=prompt.strip(),
            model=request.model,
            system_instruction=system_instruction,
            config=config,
            stream=request.stream,
        )


# ============================================================================
# Concrete Provider Adapters
# ============================================================================


class OpenAIProviderAdapter(ProviderAdapter):
    """Adapter for OpenAI provider."""

    def __init__(self) -> None:
        super().__init__(
            base_provider=OpenAIProvider(get_settings()),
            provider_id="openai",
            display_name="OpenAI",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_system_prompt=True,
            supports_token_counting=True,
            max_context_length=128000,
            max_output_tokens=16384,
            supports_embeddings=True,
            supports_fine_tuning=True,
        )

    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        return [
            Model(
                id="gpt-4o",
                provider_id="openai",
                display_name="GPT-4o",
                description="Most capable GPT-4 model",
                parameters={"max_tokens": 128000, "context_window": 128000},
            ),
            Model(
                id="gpt-4-turbo",
                provider_id="openai",
                display_name="GPT-4 Turbo",
                description="Optimized GPT-4",
                parameters={"max_tokens": 128000, "context_window": 128000},
            ),
            Model(
                id="gpt-4",
                provider_id="openai",
                display_name="GPT-4",
                parameters={"max_tokens": 8192, "context_window": 8192},
            ),
            Model(
                id="gpt-3.5-turbo",
                provider_id="openai",
                display_name="GPT-3.5 Turbo",
                description="Fast and efficient",
                parameters={"max_tokens": 16384, "context_window": 16384},
            ),
        ]


class AnthropicProviderAdapter(ProviderAdapter):
    """Adapter for Anthropic provider."""

    def __init__(self) -> None:
        super().__init__(
            base_provider=AnthropicProvider(get_settings()),
            provider_id="anthropic",
            display_name="Anthropic",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_json_mode=False,
            supports_system_prompt=True,
            supports_token_counting=True,
            max_context_length=200000,
            max_output_tokens=8192,
            supports_embeddings=False,
            supports_fine_tuning=False,
        )

    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        return [
            Model(
                id="claude-3-5-sonnet-20241022",
                provider_id="anthropic",
                display_name="Claude 3.5 Sonnet",
                description="Most intelligent Claude model",
                parameters={"max_tokens": 200000, "context_window": 200000},
            ),
            Model(
                id="claude-3-opus-20240229",
                provider_id="anthropic",
                display_name="Claude 3 Opus",
                description="Powerful reasoning",
                parameters={"max_tokens": 200000, "context_window": 200000},
            ),
            Model(
                id="claude-3-sonnet-20240229",
                provider_id="anthropic",
                display_name="Claude 3 Sonnet",
                parameters={"max_tokens": 200000, "context_window": 200000},
            ),
        ]


class GoogleProviderAdapter(ProviderAdapter):
    """Adapter for Google (Gemini) provider."""

    def __init__(self) -> None:
        super().__init__(
            base_provider=GoogleProvider(get_settings()),
            provider_id="google",
            display_name="Google AI (Gemini)",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_system_prompt=True,
            supports_token_counting=False,
            max_context_length=1000000,
            max_output_tokens=8192,
            supports_embeddings=True,
            supports_fine_tuning=False,
        )

    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        return [
            Model(
                id="gemini-2.5-pro",
                provider_id="google",
                display_name="Gemini 2.5 Pro",
                description="Latest Gemini with reasoning",
                parameters={"max_tokens": 1000000, "context_window": 1000000},
            ),
            Model(
                id="gemini-1.5-pro",
                provider_id="google",
                display_name="Gemini 1.5 Pro",
                description="Advanced Gemini",
                parameters={"max_tokens": 1000000, "context_window": 1000000},
            ),
            Model(
                id="gemini-1.5-flash",
                provider_id="google",
                display_name="Gemini 1.5 Flash",
                description="Fast Gemini",
                parameters={"max_tokens": 1000000, "context_window": 1000000},
            ),
        ]


class DeepSeekProviderAdapter(ProviderAdapter):
    """Adapter for DeepSeek provider."""

    def __init__(self) -> None:
        super().__init__(
            base_provider=DeepSeekClient(get_settings()),
            provider_id="deepseek",
            display_name="DeepSeek",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_system_prompt=True,
            supports_token_counting=False,
            max_context_length=64000,
            max_output_tokens=8192,
            supports_embeddings=False,
            supports_fine_tuning=False,
        )

    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        return [
            Model(
                id="deepseek-chat",
                provider_id="deepseek",
                display_name="DeepSeek Chat",
                description="General chat model",
                parameters={"max_tokens": 64000, "context_window": 64000},
            ),
            Model(
                id="deepseek-reasoner",
                provider_id="deepseek",
                display_name="DeepSeek Reasoner",
                description="Reasoning-focused model",
                parameters={"max_tokens": 64000, "context_window": 64000},
            ),
        ]


class QwenProviderAdapter(ProviderAdapter):
    """Adapter for Qwen provider."""

    def __init__(self) -> None:
        super().__init__(
            base_provider=QwenClient(get_settings()),
            provider_id="qwen",
            display_name="Qwen (Alibaba)",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_json_mode=False,
            supports_system_prompt=True,
            supports_token_counting=False,
            max_context_length=32000,
            max_output_tokens=8192,
        )

    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        return [
            Model(id="qwen-max", provider_id="qwen", display_name="Qwen Max"),
            Model(id="qwen-plus", provider_id="qwen", display_name="Qwen Plus"),
        ]


class CursorProviderAdapter(ProviderAdapter):
    """Adapter for Cursor provider."""

    def __init__(self) -> None:
        super().__init__(
            base_provider=CursorClient(get_settings()),
            provider_id="cursor",
            display_name="Cursor",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_system_prompt=True,
            supports_token_counting=False,
            max_context_length=8192,
            max_output_tokens=4096,
        )

    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        return [
            Model(id="cursor-default", provider_id="cursor", display_name="Cursor Default"),
        ]


class BigModelProviderAdapter(ProviderAdapter):
    """Adapter for BigModel provider."""

    def __init__(self) -> None:
        super().__init__(
            base_provider=BigModelClient(get_settings()),
            provider_id="bigmodel",
            display_name="BigModel (ZhipuAI)",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_json_mode=False,
            supports_system_prompt=True,
            max_context_length=128000,
            max_output_tokens=8192,
        )

    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        return [
            Model(id="glm-4", provider_id="bigmodel", display_name="GLM-4"),
            Model(id="glm-4v", provider_id="bigmodel", display_name="GLM-4V (Vision)"),
        ]


class RoutewayProviderAdapter(ProviderAdapter):
    """Adapter for Routeway provider."""

    def __init__(self) -> None:
        super().__init__(
            base_provider=RoutewayClient(get_settings()),
            provider_id="routeway",
            display_name="Routeway",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_system_prompt=True,
            max_context_length=16000,
            max_output_tokens=4096,
        )

    async def get_available_models(self, api_key: str | None = None) -> list[Model]:
        return [
            Model(id="routeway-default", provider_id="routeway", display_name="Routeway Default"),
        ]
