"""
Routeway Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for Routeway AI gateway integration,
supporting unified access to multiple model providers through a single endpoint.
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


class RoutewayPlugin(BaseProviderPlugin):
    """
    Provider plugin for Routeway AI gateway.

    Routeway is an AI gateway that provides unified access to multiple
    model providers through a single OpenAI-compatible API endpoint.
    It supports load balancing, failover, and cost optimization.
    """

    @property
    def provider_type(self) -> str:
        return "routeway"

    @property
    def display_name(self) -> str:
        return "Routeway Gateway"

    @property
    def aliases(self) -> list[str]:
        return ["routeway-ai", "route-way"]

    @property
    def api_key_env_var(self) -> str:
        return "ROUTEWAY_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get(
            "ROUTEWAY_BASE_URL",
            "https://api.routeway.ai/v1"
        )

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.EMBEDDINGS,
            ProviderCapability.CODE_COMPLETION,
            ProviderCapability.JSON_MODE,
            ProviderCapability.SYSTEM_MESSAGES,
        ]

    def get_default_model(self) -> str:
        return os.environ.get("ROUTEWAY_DEFAULT_MODEL", "gpt-4o")

    def _get_static_models(self) -> list[ModelInfo]:
        """
        Return static model definitions for Routeway.

        Routeway proxies to multiple providers, so it supports
        a wide variety of models from different providers.
        """
        return [
            # OpenAI Models via Routeway
            ModelInfo(
                model_id="gpt-4o",
                provider_id="routeway",
                display_name="GPT-4o (via Routeway)",
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                capabilities=[
                    "chat", "vision", "function_calling"
                ],
            ),
            ModelInfo(
                model_id="gpt-4o-mini",
                provider_id="routeway",
                display_name="GPT-4o Mini (via Routeway)",
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                capabilities=[
                    "chat", "vision", "function_calling"
                ],
            ),
            ModelInfo(
                model_id="gpt-4-turbo",
                provider_id="routeway",
                display_name="GPT-4 Turbo (via Routeway)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                capabilities=[
                    "chat", "vision", "function_calling"
                ],
            ),
            # Claude Models via Routeway
            ModelInfo(
                model_id="claude-3-5-sonnet-20241022",
                provider_id="routeway",
                display_name="Claude 3.5 Sonnet (via Routeway)",
                context_window=200000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_vision=True,
                capabilities=["chat", "vision"],
            ),
            ModelInfo(
                model_id="claude-3-opus-20240229",
                provider_id="routeway",
                display_name="Claude 3 Opus (via Routeway)",
                context_window=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_vision=True,
                capabilities=["chat", "vision"],
            ),
            # Gemini Models via Routeway
            ModelInfo(
                model_id="gemini-pro",
                provider_id="routeway",
                display_name="Gemini Pro (via Routeway)",
                context_window=32768,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="gemini-pro-vision",
                provider_id="routeway",
                display_name="Gemini Pro Vision (via Routeway)",
                context_window=16384,
                max_output_tokens=2048,
                supports_streaming=True,
                supports_vision=True,
                capabilities=["chat", "vision"],
            ),
            # DeepSeek via Routeway
            ModelInfo(
                model_id="deepseek-chat",
                provider_id="routeway",
                display_name="DeepSeek Chat (via Routeway)",
                context_window=128000,
                max_output_tokens=8192,
                supports_streaming=True,
                capabilities=["chat"],
            ),
            # Mixtral via Routeway
            ModelInfo(
                model_id="mixtral-8x7b-instruct",
                provider_id="routeway",
                display_name="Mixtral 8x7B (via Routeway)",
                context_window=32768,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """
        List available Routeway models.

        Attempts to fetch from API, falls back to static list.
        """
        key = self._get_api_key(api_key)

        if key:
            try:
                return await self._fetch_models_from_api(key)
            except Exception as e:
                logger.warning(f"Failed to fetch Routeway models: {e}")

        return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str) -> list[ModelInfo]:
        """Fetch models from Routeway API (OpenAI-compatible format)."""
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
                models.append(ModelInfo(
                    model_id=model_id,
                    provider_id="routeway",
                    display_name=f"{model_id} (via Routeway)",
                    context_window=model.get("context_length", 8192),
                    supports_streaming=True,
                    capabilities=["chat"],
                ))

        api_ids = {m.model_id for m in models}
        for static_model in static_models.values():
            if static_model.model_id not in api_ids:
                models.append(static_model)

        return models

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate Routeway API key."""
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
            logger.warning(f"Routeway API key validation failed: {e}")
            return False

    async def generate(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> GenerationResponse:
        """Generate text using Routeway API (OpenAI-compatible)."""
        key = self._get_api_key(api_key)
        if not key:
            raise ValueError("Routeway API key is required")

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
                provider="routeway",
                finish_reason=choice.get("finish_reason"),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Routeway API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Routeway generation error: {e}")
            raise

    async def generate_stream(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Routeway API."""
        key = self._get_api_key(api_key)
        if not key:
            raise ValueError("Routeway API key is required")

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
            logger.error(f"Routeway streaming error: {e}")
            raise
