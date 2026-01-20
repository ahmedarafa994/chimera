"""OpenAI Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for OpenAI API integration,
supporting models like GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, and o1.
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


class OpenAIPlugin(BaseProviderPlugin):
    """Provider plugin for OpenAI API.

    Supports GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, o1 reasoning models, etc.
    """

    @property
    def provider_type(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    @property
    def aliases(self) -> list[str]:
        return ["gpt", "chatgpt"]

    @property
    def api_key_env_var(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get("DIRECT_OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.JSON_MODE,
            ProviderCapability.SYSTEM_MESSAGES,
            ProviderCapability.REASONING,
        ]

    def get_default_model(self) -> str:
        return "gpt-4o"

    def _get_static_models(self) -> list[ModelInfo]:
        """Return static model definitions for OpenAI."""
        return [
            # GPT-4o Family
            ModelInfo(
                model_id="gpt-4o",
                provider_id="openai",
                display_name="GPT-4o",
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.0025,
                output_price_per_1k=0.01,
                capabilities=["chat", "vision", "function_calling", "json_mode"],
            ),
            ModelInfo(
                model_id="gpt-4o-mini",
                provider_id="openai",
                display_name="GPT-4o Mini",
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.00015,
                output_price_per_1k=0.0006,
                capabilities=["chat", "vision", "function_calling", "json_mode"],
            ),
            ModelInfo(
                model_id="gpt-4o-2024-11-20",
                provider_id="openai",
                display_name="GPT-4o (2024-11-20)",
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.0025,
                output_price_per_1k=0.01,
                capabilities=["chat", "vision", "function_calling", "json_mode"],
            ),
            # GPT-4 Turbo
            ModelInfo(
                model_id="gpt-4-turbo",
                provider_id="openai",
                display_name="GPT-4 Turbo",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.01,
                output_price_per_1k=0.03,
                capabilities=["chat", "vision", "function_calling", "json_mode"],
            ),
            ModelInfo(
                model_id="gpt-4-turbo-preview",
                provider_id="openai",
                display_name="GPT-4 Turbo Preview",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=False,
                input_price_per_1k=0.01,
                output_price_per_1k=0.03,
                capabilities=["chat", "function_calling", "json_mode"],
            ),
            # GPT-4 Original
            ModelInfo(
                model_id="gpt-4",
                provider_id="openai",
                display_name="GPT-4",
                context_window=8192,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=False,
                input_price_per_1k=0.03,
                output_price_per_1k=0.06,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="gpt-4-32k",
                provider_id="openai",
                display_name="GPT-4 32K",
                context_window=32768,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=False,
                input_price_per_1k=0.06,
                output_price_per_1k=0.12,
                capabilities=["chat", "function_calling"],
            ),
            # GPT-3.5 Turbo
            ModelInfo(
                model_id="gpt-3.5-turbo",
                provider_id="openai",
                display_name="GPT-3.5 Turbo",
                context_window=16385,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=False,
                input_price_per_1k=0.0005,
                output_price_per_1k=0.0015,
                capabilities=["chat", "function_calling", "json_mode"],
            ),
            ModelInfo(
                model_id="gpt-3.5-turbo-16k",
                provider_id="openai",
                display_name="GPT-3.5 Turbo 16K",
                context_window=16385,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=False,
                input_price_per_1k=0.003,
                output_price_per_1k=0.004,
                capabilities=["chat", "function_calling"],
                deprecated=True,
                deprecation_message="Use gpt-3.5-turbo instead",
            ),
            # o1 Reasoning Models
            ModelInfo(
                model_id="o1",
                provider_id="openai",
                display_name="o1",
                context_window=200000,
                max_output_tokens=100000,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_reasoning=True,
                input_price_per_1k=0.015,
                output_price_per_1k=0.06,
                capabilities=["chat", "reasoning", "vision", "function_calling"],
            ),
            ModelInfo(
                model_id="o1-preview",
                provider_id="openai",
                display_name="o1 Preview",
                context_window=128000,
                max_output_tokens=32768,
                supports_streaming=False,
                supports_function_calling=False,
                supports_vision=False,
                supports_reasoning=True,
                input_price_per_1k=0.015,
                output_price_per_1k=0.06,
                capabilities=["chat", "reasoning"],
            ),
            ModelInfo(
                model_id="o1-mini",
                provider_id="openai",
                display_name="o1 Mini",
                context_window=128000,
                max_output_tokens=65536,
                supports_streaming=False,
                supports_function_calling=False,
                supports_vision=False,
                supports_reasoning=True,
                input_price_per_1k=0.003,
                output_price_per_1k=0.012,
                capabilities=["chat", "reasoning"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """List available OpenAI models.

        Attempts to fetch from API, falls back to static list on failure.
        """
        key = self._get_api_key(api_key)

        # Try to fetch from API
        if key:
            try:
                return await self._fetch_models_from_api(key)
            except Exception as e:
                logger.warning(f"Failed to fetch OpenAI models from API: {e}")

        # Fallback to static list
        return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str) -> list[ModelInfo]:
        """Fetch models from OpenAI API."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
            response = await client.models.list()

            # Filter to chat models and map to ModelInfo
            static_models = {m.model_id: m for m in self._get_static_models()}
            models = []

            for model in response.data:
                model_id = model.id

                # Skip non-chat models
                if not any(prefix in model_id for prefix in ["gpt-4", "gpt-3.5", "o1"]):
                    continue

                # Use static info if available, otherwise create basic info
                if model_id in static_models:
                    models.append(static_models[model_id])
                else:
                    models.append(
                        ModelInfo(
                            model_id=model_id,
                            provider_id="openai",
                            display_name=model_id,
                            supports_streaming=True,
                            capabilities=["chat"],
                        ),
                    )

            # Add any static models not in API response
            api_model_ids = {m.model_id for m in models}
            for static_model in static_models.values():
                if static_model.model_id not in api_model_ids:
                    models.append(static_model)

            return models

        except Exception as e:
            logger.exception(f"Error fetching OpenAI models: {e}")
            raise

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate OpenAI API key by making a test request."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
            # Try to list models as a simple validation
            await client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI API key validation failed: {e}")
            return False

    async def generate(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> GenerationResponse:
        """Generate text using OpenAI API."""
        from openai import AsyncOpenAI

        key = self._get_api_key(api_key)
        if not key:
            msg = "OpenAI API key is required"
            raise ValueError(msg)

        client = AsyncOpenAI(api_key=key, base_url=self.base_url)
        model = request.model or self.get_default_model()

        start_time = time.time()

        # Build messages
        messages: list[dict[str, Any]] = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})

        if request.messages:
            messages.extend(request.messages)
        else:
            messages.append({"role": "user", "content": request.prompt})

        # Build request parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop_sequences:
            params["stop"] = request.stop_sequences

        try:
            response = await client.chat.completions.create(**params)

            latency_ms = (time.time() - start_time) * 1000

            return GenerationResponse(
                text=response.choices[0].message.content or "",
                model_used=response.model,
                provider="openai",
                finish_reason=response.choices[0].finish_reason,
                usage=(
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None
                ),
                latency_ms=latency_ms,
                request_id=response.id,
            )
        except Exception as e:
            logger.exception(f"OpenAI generation error: {e}")
            raise

    async def generate_stream(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from OpenAI API."""
        from openai import AsyncOpenAI

        key = self._get_api_key(api_key)
        if not key:
            msg = "OpenAI API key is required"
            raise ValueError(msg)

        client = AsyncOpenAI(api_key=key, base_url=self.base_url)
        model = request.model or self.get_default_model()

        # Build messages
        messages: list[dict[str, Any]] = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})

        if request.messages:
            messages.extend(request.messages)
        else:
            messages.append({"role": "user", "content": request.prompt})

        # Build request parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop_sequences:
            params["stop"] = request.stop_sequences

        try:
            stream = await client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        text=chunk.choices[0].delta.content,
                        is_final=False,
                    )

                if chunk.choices and chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        text="",
                        is_final=True,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
        except Exception as e:
            logger.exception(f"OpenAI streaming error: {e}")
            raise
