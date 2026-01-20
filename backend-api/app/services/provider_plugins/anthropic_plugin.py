"""Anthropic Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for Anthropic API integration,
supporting Claude 3.5, Claude 3, and future Claude models.
"""

import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from app.services.provider_plugins import (
    BaseProviderPlugin,
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderCapability,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class AnthropicPlugin(BaseProviderPlugin):
    """Provider plugin for Anthropic API.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, etc.
    """

    @property
    def provider_type(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic"

    @property
    def aliases(self) -> list[str]:
        return ["claude"]

    @property
    def api_key_env_var(self) -> str:
        return "ANTHROPIC_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get("DIRECT_ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.SYSTEM_MESSAGES,
        ]

    def get_default_model(self) -> str:
        return "claude-3-5-sonnet-20241022"

    def _get_static_models(self) -> list[ModelInfo]:
        """Return static model definitions for Anthropic."""
        return [
            # Claude 3.5 Models
            ModelInfo(
                model_id="claude-3-5-sonnet-20241022",
                provider_id="anthropic",
                display_name="Claude 3.5 Sonnet",
                context_window=200000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.003,
                output_price_per_1k=0.015,
                capabilities=["chat", "vision", "function_calling"],
            ),
            ModelInfo(
                model_id="claude-3-5-haiku-20241022",
                provider_id="anthropic",
                display_name="Claude 3.5 Haiku",
                context_window=200000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.0008,
                output_price_per_1k=0.004,
                capabilities=["chat", "vision", "function_calling"],
            ),
            # Claude 3 Models
            ModelInfo(
                model_id="claude-3-opus-20240229",
                provider_id="anthropic",
                display_name="Claude 3 Opus",
                context_window=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.015,
                output_price_per_1k=0.075,
                capabilities=["chat", "vision", "function_calling"],
            ),
            ModelInfo(
                model_id="claude-3-sonnet-20240229",
                provider_id="anthropic",
                display_name="Claude 3 Sonnet",
                context_window=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.003,
                output_price_per_1k=0.015,
                capabilities=["chat", "vision", "function_calling"],
            ),
            ModelInfo(
                model_id="claude-3-haiku-20240307",
                provider_id="anthropic",
                display_name="Claude 3 Haiku",
                context_window=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.00025,
                output_price_per_1k=0.00125,
                capabilities=["chat", "vision", "function_calling"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """List available Anthropic models.

        Returns static list as Anthropic doesn't have a models list API.
        """
        return self._get_static_models()

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate Anthropic API key by making a minimal test request."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=self.base_url if self.base_url else None,
            )

            # Make a minimal request to validate the key
            await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic API key validation failed: {e}")
            return False

    async def generate(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> GenerationResponse:
        """Generate text using Anthropic API."""
        import anthropic

        key = self._get_api_key(api_key)
        if not key:
            msg = "Anthropic API key is required"
            raise ValueError(msg)

        client = anthropic.AsyncAnthropic(
            api_key=key,
            base_url=self.base_url if self.base_url else None,
        )
        model = request.model or self.get_default_model()

        start_time = time.time()

        # Build messages
        messages: list[dict[str, Any]] = []

        if request.messages:
            messages.extend(request.messages)
        else:
            messages.append({"role": "user", "content": request.prompt})

        # Build request parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }

        if request.system_instruction:
            params["system"] = request.system_instruction
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.top_k is not None:
            params["top_k"] = request.top_k
        if request.stop_sequences:
            params["stop_sequences"] = request.stop_sequences

        try:
            response = await client.messages.create(**params)

            latency_ms = (time.time() - start_time) * 1000

            # Extract text from response
            text = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text

            return GenerationResponse(
                text=text,
                model_used=response.model,
                provider="anthropic",
                finish_reason=response.stop_reason,
                usage=(
                    {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": (
                            response.usage.input_tokens + response.usage.output_tokens
                        ),
                    }
                    if response.usage
                    else None
                ),
                latency_ms=latency_ms,
                request_id=response.id,
            )
        except Exception as e:
            logger.exception(f"Anthropic generation error: {e}")
            raise

    async def generate_stream(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Anthropic API."""
        import anthropic

        key = self._get_api_key(api_key)
        if not key:
            msg = "Anthropic API key is required"
            raise ValueError(msg)

        client = anthropic.AsyncAnthropic(
            api_key=key,
            base_url=self.base_url if self.base_url else None,
        )
        model = request.model or self.get_default_model()

        # Build messages
        messages: list[dict[str, Any]] = []

        if request.messages:
            messages.extend(request.messages)
        else:
            messages.append({"role": "user", "content": request.prompt})

        # Build request parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }

        if request.system_instruction:
            params["system"] = request.system_instruction
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.top_k is not None:
            params["top_k"] = request.top_k
        if request.stop_sequences:
            params["stop_sequences"] = request.stop_sequences

        try:
            async with client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(
                        text=text,
                        is_final=False,
                    )

                # Get final message for metadata
                final_message = await stream.get_final_message()
                yield StreamChunk(
                    text="",
                    is_final=True,
                    finish_reason=final_message.stop_reason,
                    usage=(
                        {
                            "prompt_tokens": final_message.usage.input_tokens,
                            "completion_tokens": final_message.usage.output_tokens,
                            "total_tokens": (
                                final_message.usage.input_tokens + final_message.usage.output_tokens
                            ),
                        }
                        if final_message.usage
                        else None
                    ),
                )
        except Exception as e:
            logger.exception(f"Anthropic streaming error: {e}")
            raise
