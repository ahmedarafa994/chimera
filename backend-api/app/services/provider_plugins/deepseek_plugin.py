"""DeepSeek Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for DeepSeek AI integration,
supporting DeepSeek Chat, DeepSeek Coder, and DeepSeek Reasoner models.
"""

import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.services.provider_plugins import (
    BaseProviderPlugin,
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderCapability,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class DeepSeekPlugin(BaseProviderPlugin):
    """Provider plugin for DeepSeek AI API.

    DeepSeek uses an OpenAI-compatible API format.
    Supports DeepSeek Chat, Coder, and Reasoner models.
    """

    @property
    def provider_type(self) -> str:
        return "deepseek"

    @property
    def display_name(self) -> str:
        return "DeepSeek"

    @property
    def aliases(self) -> list[str]:
        return ["deep-seek", "deepseek-ai"]

    @property
    def api_key_env_var(self) -> str:
        return "DEEPSEEK_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.CODE_COMPLETION,
            ProviderCapability.REASONING,
            ProviderCapability.SYSTEM_MESSAGES,
            ProviderCapability.JSON_MODE,
        ]

    def get_default_model(self) -> str:
        return "deepseek-chat"

    def _get_static_models(self) -> list[ModelInfo]:
        """Return static model definitions for DeepSeek."""
        return [
            # DeepSeek Chat Models
            ModelInfo(
                model_id="deepseek-chat",
                provider_id="deepseek",
                display_name="DeepSeek Chat",
                context_window=128000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.00014,
                output_price_per_1k=0.00028,
                capabilities=["chat", "function_calling"],
            ),
            # DeepSeek Reasoner (R1)
            ModelInfo(
                model_id="deepseek-reasoner",
                provider_id="deepseek",
                display_name="DeepSeek Reasoner (R1)",
                context_window=128000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_reasoning=True,
                input_price_per_1k=0.00055,
                output_price_per_1k=0.00219,
                capabilities=["chat", "reasoning"],
            ),
            # DeepSeek Coder
            ModelInfo(
                model_id="deepseek-coder",
                provider_id="deepseek",
                display_name="DeepSeek Coder",
                context_window=128000,
                max_output_tokens=8192,
                supports_streaming=True,
                input_price_per_1k=0.00014,
                output_price_per_1k=0.00028,
                capabilities=["chat", "code"],
            ),
            # V3 Models
            ModelInfo(
                model_id="deepseek-v3",
                provider_id="deepseek",
                display_name="DeepSeek V3",
                context_window=128000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.00014,
                output_price_per_1k=0.00028,
                capabilities=["chat", "function_calling"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """List available DeepSeek models.

        Attempts to fetch from API, falls back to static list on failure.
        """
        key = self._get_api_key(api_key)

        # Try to fetch from API
        if key:
            try:
                return await self._fetch_models_from_api(key)
            except Exception as e:
                logger.warning(f"Failed to fetch DeepSeek models: {e}")

        # Fallback to static list
        return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str) -> list[ModelInfo]:
        """Fetch models from DeepSeek API (OpenAI-compatible format)."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        static_models = {m.model_id: m for m in self._get_static_models()}
        models = []

        for model in data.get("data", []):
            model_id = model.get("id", "")

            # Use static info if available
            if model_id in static_models:
                models.append(static_models[model_id])
            else:
                # Create basic info
                models.append(
                    ModelInfo(
                        model_id=model_id,
                        provider_id="deepseek",
                        display_name=model.get("id", model_id),
                        context_window=128000,
                        supports_streaming=True,
                        capabilities=["chat"],
                    ),
                )

        # Add any static models not in API response
        api_ids = {m.model_id for m in models}
        for static_model in static_models.values():
            if static_model.model_id not in api_ids:
                models.append(static_model)

        return models

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate DeepSeek API key."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"DeepSeek API key validation failed: {e}")
            return False

    async def generate(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> GenerationResponse:
        """Generate text using DeepSeek API (OpenAI-compatible)."""
        key = self._get_api_key(api_key)
        if not key:
            msg = "DeepSeek API key is required"
            raise ValueError(msg)

        model = request.model or self.get_default_model()
        start_time = time.time()

        # Build messages
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})

        if request.messages:
            messages.extend(request.messages)
        elif request.prompt:
            messages.append({"role": "user", "content": request.prompt})

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        if request.response_format:
            payload["response_format"] = request.response_format

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})

            return GenerationResponse(
                text=message.get("content", ""),
                model_used=data.get("model", model),
                provider="deepseek",
                finish_reason=choice.get("finish_reason"),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            logger.exception(f"DeepSeek API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.exception(f"DeepSeek generation error: {e}")
            raise

    async def generate_stream(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from DeepSeek API."""
        key = self._get_api_key(api_key)
        if not key:
            msg = "DeepSeek API key is required"
            raise ValueError(msg)

        model = request.model or self.get_default_model()

        # Build messages
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})

        if request.messages:
            messages.extend(request.messages)
        elif request.prompt:
            messages.append({"role": "user", "content": request.prompt})

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        try:
            async with (
                httpx.AsyncClient(timeout=120.0) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response,
            ):
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str == "[DONE]":
                            yield StreamChunk(text="", is_final=True)
                            break

                        try:
                            import json

                            data = json.loads(data_str)
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                yield StreamChunk(
                                    text=content,
                                    is_final=False,
                                )
                        except Exception as e:
                            logger.warning(f"Parse error: {e}")
                            continue
        except Exception as e:
            logger.exception(f"DeepSeek streaming error: {e}")
            raise
