"""
Qwen Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for Alibaba Cloud Qwen AI integration,
supporting Qwen-Max, Qwen-Plus, Qwen-Turbo, and specialized Qwen models.
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


class QwenPlugin(BaseProviderPlugin):
    """
    Provider plugin for Alibaba Cloud Qwen API.

    Qwen uses its own API format (DashScope) but has OpenAI-compatible
    endpoint available as well. This implementation uses the OpenAI-
    compatible endpoint for consistency.
    """

    @property
    def provider_type(self) -> str:
        return "qwen"

    @property
    def display_name(self) -> str:
        return "Qwen (Alibaba)"

    @property
    def aliases(self) -> list[str]:
        return ["alibaba", "dashscope", "tongyi", "qwen-ai"]

    @property
    def api_key_env_var(self) -> str:
        return "QWEN_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.CODE_COMPLETION,
            ProviderCapability.SYSTEM_MESSAGES,
            ProviderCapability.JSON_MODE,
        ]

    def get_default_model(self) -> str:
        return "qwen-max"

    def _get_static_models(self) -> list[ModelInfo]:
        """Return static model definitions for Qwen."""
        return [
            # Qwen Max Series (Most Capable)
            ModelInfo(
                model_id="qwen-max",
                provider_id="qwen",
                display_name="Qwen Max",
                context_window=32768,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.004,
                output_price_per_1k=0.012,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="qwen-max-latest",
                provider_id="qwen",
                display_name="Qwen Max Latest",
                context_window=32768,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.004,
                output_price_per_1k=0.012,
                capabilities=["chat", "function_calling"],
            ),
            # Qwen Plus Series (Balanced)
            ModelInfo(
                model_id="qwen-plus",
                provider_id="qwen",
                display_name="Qwen Plus",
                context_window=131072,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0008,
                output_price_per_1k=0.002,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="qwen-plus-latest",
                provider_id="qwen",
                display_name="Qwen Plus Latest",
                context_window=131072,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0008,
                output_price_per_1k=0.002,
                capabilities=["chat", "function_calling"],
            ),
            # Qwen Turbo Series (Fast)
            ModelInfo(
                model_id="qwen-turbo",
                provider_id="qwen",
                display_name="Qwen Turbo",
                context_window=131072,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0003,
                output_price_per_1k=0.0006,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="qwen-turbo-latest",
                provider_id="qwen",
                display_name="Qwen Turbo Latest",
                context_window=131072,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0003,
                output_price_per_1k=0.0006,
                capabilities=["chat", "function_calling"],
            ),
            # Qwen Long (Extended Context)
            ModelInfo(
                model_id="qwen-long",
                provider_id="qwen",
                display_name="Qwen Long",
                context_window=1000000,
                max_output_tokens=8192,
                supports_streaming=True,
                input_price_per_1k=0.0005,
                output_price_per_1k=0.002,
                capabilities=["chat", "long_context"],
            ),
            # Qwen VL (Vision-Language)
            ModelInfo(
                model_id="qwen-vl-max",
                provider_id="qwen",
                display_name="Qwen VL Max",
                context_window=32768,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_vision=True,
                input_price_per_1k=0.003,
                output_price_per_1k=0.009,
                capabilities=["chat", "vision"],
            ),
            ModelInfo(
                model_id="qwen-vl-plus",
                provider_id="qwen",
                display_name="Qwen VL Plus",
                context_window=8192,
                max_output_tokens=2048,
                supports_streaming=True,
                supports_vision=True,
                input_price_per_1k=0.002,
                output_price_per_1k=0.006,
                capabilities=["chat", "vision"],
            ),
            # Qwen Coder
            ModelInfo(
                model_id="qwen-coder-plus",
                provider_id="qwen",
                display_name="Qwen Coder Plus",
                context_window=131072,
                max_output_tokens=8192,
                supports_streaming=True,
                input_price_per_1k=0.0035,
                output_price_per_1k=0.007,
                capabilities=["chat", "code"],
            ),
            ModelInfo(
                model_id="qwen-coder-turbo",
                provider_id="qwen",
                display_name="Qwen Coder Turbo",
                context_window=131072,
                max_output_tokens=8192,
                supports_streaming=True,
                input_price_per_1k=0.002,
                output_price_per_1k=0.006,
                capabilities=["chat", "code"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """
        List available Qwen models.

        Attempts to fetch from API, falls back to static list on failure.
        """
        key = self._get_api_key(api_key)

        # Try to fetch from API
        if key:
            try:
                return await self._fetch_models_from_api(key)
            except Exception as e:
                logger.warning(f"Failed to fetch Qwen models: {e}")

        # Fallback to static list
        return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str) -> list[ModelInfo]:
        """Fetch models from Qwen API (OpenAI-compatible format)."""
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

            # Filter to Qwen models
            if not model_id.startswith("qwen"):
                continue

            # Use static info if available
            if model_id in static_models:
                models.append(static_models[model_id])
            else:
                models.append(ModelInfo(
                    model_id=model_id,
                    provider_id="qwen",
                    display_name=model.get("id", model_id),
                    context_window=32768,
                    supports_streaming=True,
                    capabilities=["chat"],
                ))

        # Add static models not in API response
        api_ids = {m.model_id for m in models}
        for static_model in static_models.values():
            if static_model.model_id not in api_ids:
                models.append(static_model)

        return models

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate Qwen API key."""
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
            logger.warning(f"Qwen API key validation failed: {e}")
            return False

    async def generate(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> GenerationResponse:
        """Generate text using Qwen API (OpenAI-compatible)."""
        key = self._get_api_key(api_key)
        if not key:
            raise ValueError("Qwen API key is required")

        model = request.model or self.get_default_model()
        start_time = time.time()

        # Build messages
        messages = []
        if request.system_instruction:
            messages.append({
                "role": "system",
                "content": request.system_instruction
            })

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
                provider="qwen",
                finish_reason=choice.get("finish_reason"),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Qwen API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Qwen generation error: {e}")
            raise

    async def generate_stream(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Qwen API."""
        key = self._get_api_key(api_key)
        if not key:
            raise ValueError("Qwen API key is required")

        model = request.model or self.get_default_model()

        # Build messages
        messages = []
        if request.system_instruction:
            messages.append({
                "role": "system",
                "content": request.system_instruction
            })

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
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
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
            logger.error(f"Qwen streaming error: {e}")
            raise
