"""
Local Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for local/self-hosted models,
supporting various local inference servers with OpenAI-compatible APIs.
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


class LocalPlugin(BaseProviderPlugin):
    """
    Provider plugin for local/self-hosted inference servers.

    Supports any OpenAI-compatible local server including:
    - vLLM
    - LocalAI
    - LM Studio
    - text-generation-inference
    - llama.cpp server
    - Custom OpenAI-compatible servers
    """

    @property
    def provider_type(self) -> str:
        return "local"

    @property
    def display_name(self) -> str:
        return "Local Models"

    @property
    def aliases(self) -> list[str]:
        return ["local-llm", "vllm", "localai", "lmstudio", "tgi", "llama-cpp"]

    @property
    def api_key_env_var(self) -> str:
        return "LOCAL_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get("LOCAL_BASE_URL", "http://localhost:8001/v1")

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.CODE_COMPLETION,
            ProviderCapability.SYSTEM_MESSAGES,
        ]

    def get_default_model(self) -> str:
        return os.environ.get("LOCAL_DEFAULT_MODEL", "default")

    def _get_static_models(self) -> list[ModelInfo]:
        """
        Return placeholder model definitions.

        Actual models depend on what's loaded in the local server.
        """
        return [
            ModelInfo(
                model_id="default",
                provider_id="local",
                display_name="Default Local Model",
                context_window=4096,
                max_output_tokens=2048,
                supports_streaming=True,
                capabilities=["chat"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """
        List available models from local server.

        Queries the local server for available models.
        """
        try:
            return await self._fetch_models_from_api(api_key)
        except Exception as e:
            logger.warning(f"Failed to fetch local models: {e}")
            return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str | None = None) -> list[ModelInfo]:
        """Fetch models from local OpenAI-compatible server."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = self._get_api_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            models.append(
                ModelInfo(
                    model_id=model_id,
                    provider_id="local",
                    display_name=model.get("id", model_id),
                    context_window=model.get("context_length", 4096),
                    max_output_tokens=model.get("max_tokens", 2048),
                    supports_streaming=True,
                    capabilities=["chat"],
                )
            )

        return models if models else self._get_static_models()

    async def validate_api_key(self, api_key: str) -> bool:
        """
        Validate connection to local server.

        For local servers, this checks if the server is reachable.
        """
        try:
            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Local server connection check failed: {e}")
            return False

    async def generate(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> GenerationResponse:
        """Generate text using local server (OpenAI-compatible)."""
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

        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = self._get_api_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
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
                provider="local",
                finish_reason=choice.get("finish_reason"),
                usage=(
                    {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                    if usage
                    else None
                ),
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Local server API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Local server generation error: {e}")
            raise

    async def generate_stream(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from local server."""
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

        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = self._get_api_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        try:
            async with (
                httpx.AsyncClient(timeout=300.0) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
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
            logger.error(f"Local server streaming error: {e}")
            raise
