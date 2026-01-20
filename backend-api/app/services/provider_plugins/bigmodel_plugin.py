"""BigModel (Zhipu AI) Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for Zhipu AI BigModel integration,
supporting GLM-4, GLM-3, and CodeGeeX models.
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


class BigModelPlugin(BaseProviderPlugin):
    """Provider plugin for Zhipu AI BigModel API.

    Supports GLM-4, GLM-3-Turbo, CodeGeeX, and other Zhipu models.
    Uses the OpenAI-compatible API format.
    """

    @property
    def provider_type(self) -> str:
        return "bigmodel"

    @property
    def display_name(self) -> str:
        return "BigModel (Zhipu AI)"

    @property
    def aliases(self) -> list[str]:
        return ["zhipu", "zhipuai", "glm", "chatglm"]

    @property
    def api_key_env_var(self) -> str:
        return "ZHIPU_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.CODE_COMPLETION,
            ProviderCapability.SYSTEM_MESSAGES,
        ]

    def get_default_model(self) -> str:
        return "glm-4-plus"

    def _get_static_models(self) -> list[ModelInfo]:
        """Return static model definitions for BigModel."""
        return [
            # GLM-4 Series (Most Capable)
            ModelInfo(
                model_id="glm-4-plus",
                provider_id="bigmodel",
                display_name="GLM-4 Plus",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.007,
                output_price_per_1k=0.007,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="glm-4",
                provider_id="bigmodel",
                display_name="GLM-4",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.014,
                output_price_per_1k=0.014,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="glm-4-0520",
                provider_id="bigmodel",
                display_name="GLM-4 (0520)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.014,
                output_price_per_1k=0.014,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="glm-4-air",
                provider_id="bigmodel",
                display_name="GLM-4 Air",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0007,
                output_price_per_1k=0.0007,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="glm-4-airx",
                provider_id="bigmodel",
                display_name="GLM-4 AirX",
                context_window=8192,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0014,
                output_price_per_1k=0.0014,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="glm-4-flash",
                provider_id="bigmodel",
                display_name="GLM-4 Flash",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0001,
                output_price_per_1k=0.0001,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="glm-4-flashx",
                provider_id="bigmodel",
                display_name="GLM-4 FlashX",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0001,
                output_price_per_1k=0.0001,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="glm-4-long",
                provider_id="bigmodel",
                display_name="GLM-4 Long",
                context_window=1000000,
                max_output_tokens=4096,
                supports_streaming=True,
                input_price_per_1k=0.0007,
                output_price_per_1k=0.0007,
                capabilities=["chat", "long_context"],
            ),
            # GLM-4V (Vision)
            ModelInfo(
                model_id="glm-4v",
                provider_id="bigmodel",
                display_name="GLM-4V (Vision)",
                context_window=2048,
                max_output_tokens=1024,
                supports_streaming=True,
                supports_vision=True,
                input_price_per_1k=0.007,
                output_price_per_1k=0.007,
                capabilities=["chat", "vision"],
            ),
            ModelInfo(
                model_id="glm-4v-plus",
                provider_id="bigmodel",
                display_name="GLM-4V Plus",
                context_window=8192,
                max_output_tokens=1024,
                supports_streaming=True,
                supports_vision=True,
                input_price_per_1k=0.014,
                output_price_per_1k=0.014,
                capabilities=["chat", "vision"],
            ),
            # GLM-3 Series (Legacy)
            ModelInfo(
                model_id="glm-3-turbo",
                provider_id="bigmodel",
                display_name="GLM-3 Turbo",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                input_price_per_1k=0.0007,
                output_price_per_1k=0.0007,
                capabilities=["chat"],
            ),
            # CodeGeeX Series
            ModelInfo(
                model_id="codegeex-4",
                provider_id="bigmodel",
                display_name="CodeGeeX-4",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                input_price_per_1k=0.0001,
                output_price_per_1k=0.0001,
                capabilities=["chat", "code"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """List available BigModel models.

        Attempts to fetch from API, falls back to static list.
        """
        key = self._get_api_key(api_key)

        if key:
            try:
                return await self._fetch_models_from_api(key)
            except Exception as e:
                logger.warning(f"Failed to fetch BigModel models: {e}")

        return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str) -> list[ModelInfo]:
        """Fetch models from BigModel API."""
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

            if model_id in static_models:
                models.append(static_models[model_id])
            else:
                models.append(
                    ModelInfo(
                        model_id=model_id,
                        provider_id="bigmodel",
                        display_name=model.get("id", model_id),
                        context_window=128000,
                        supports_streaming=True,
                        capabilities=["chat"],
                    ),
                )

        api_ids = {m.model_id for m in models}
        for static_model in static_models.values():
            if static_model.model_id not in api_ids:
                models.append(static_model)

        return models

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate BigModel API key."""
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
            logger.warning(f"BigModel API key validation failed: {e}")
            return False

    async def generate(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> GenerationResponse:
        """Generate text using BigModel API."""
        key = self._get_api_key(api_key)
        if not key:
            msg = "BigModel API key is required"
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
                provider="bigmodel",
                finish_reason=choice.get("finish_reason"),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            logger.exception(f"BigModel API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.exception(f"BigModel generation error: {e}")
            raise

    async def generate_stream(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from BigModel API."""
        key = self._get_api_key(api_key)
        if not key:
            msg = "BigModel API key is required"
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
            logger.exception(f"BigModel streaming error: {e}")
            raise
