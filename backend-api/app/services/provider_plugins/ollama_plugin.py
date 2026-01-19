"""
Ollama Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for local Ollama integration,
supporting locally-hosted open-source models like Llama, Mistral, etc.
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


class OllamaPlugin(BaseProviderPlugin):
    """
    Provider plugin for Ollama local inference.

    Ollama allows running open-source LLMs locally. This plugin
    connects to a local Ollama server for inference.
    """

    @property
    def provider_type(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama (Local)"

    @property
    def aliases(self) -> list[str]:
        return ["local-ollama", "ollama-local"]

    @property
    def api_key_env_var(self) -> str:
        # Ollama doesn't require an API key by default
        return "OLLAMA_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
            ProviderCapability.CODE_COMPLETION,
            ProviderCapability.SYSTEM_MESSAGES,
        ]

    def get_default_model(self) -> str:
        return "llama3.2"

    def _get_static_models(self) -> list[ModelInfo]:
        """
        Return common Ollama model definitions.

        Note: Actual available models depend on what's pulled locally.
        """
        return [
            # Llama 3 Series
            ModelInfo(
                model_id="llama3.2",
                provider_id="ollama",
                display_name="Llama 3.2 (3B)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_vision=True,
                capabilities=["chat", "vision"],
            ),
            ModelInfo(
                model_id="llama3.2:1b",
                provider_id="ollama",
                display_name="Llama 3.2 (1B)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat"],
            ),
            ModelInfo(
                model_id="llama3.1",
                provider_id="ollama",
                display_name="Llama 3.1 (8B)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat"],
            ),
            ModelInfo(
                model_id="llama3.1:70b",
                provider_id="ollama",
                display_name="Llama 3.1 (70B)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat"],
            ),
            # Mistral Series
            ModelInfo(
                model_id="mistral",
                provider_id="ollama",
                display_name="Mistral (7B)",
                context_window=32768,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat"],
            ),
            ModelInfo(
                model_id="mistral-nemo",
                provider_id="ollama",
                display_name="Mistral Nemo (12B)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat"],
            ),
            # Code Models
            ModelInfo(
                model_id="codellama",
                provider_id="ollama",
                display_name="Code Llama (7B)",
                context_window=16384,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat", "code"],
            ),
            ModelInfo(
                model_id="deepseek-coder-v2",
                provider_id="ollama",
                display_name="DeepSeek Coder V2",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat", "code"],
            ),
            ModelInfo(
                model_id="qwen2.5-coder",
                provider_id="ollama",
                display_name="Qwen 2.5 Coder",
                context_window=32768,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat", "code"],
            ),
            # Vision Models
            ModelInfo(
                model_id="llava",
                provider_id="ollama",
                display_name="LLaVA (Vision)",
                context_window=4096,
                max_output_tokens=2048,
                supports_streaming=True,
                supports_vision=True,
                capabilities=["chat", "vision"],
            ),
            # Reasoning Models
            ModelInfo(
                model_id="deepseek-r1",
                provider_id="ollama",
                display_name="DeepSeek R1",
                context_window=128000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_reasoning=True,
                capabilities=["chat", "reasoning"],
            ),
            ModelInfo(
                model_id="qwq",
                provider_id="ollama",
                display_name="QwQ (Reasoning)",
                context_window=32768,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_reasoning=True,
                capabilities=["chat", "reasoning"],
            ),
            # General Purpose
            ModelInfo(
                model_id="phi3",
                provider_id="ollama",
                display_name="Phi-3 (3.8B)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat"],
            ),
            ModelInfo(
                model_id="gemma2",
                provider_id="ollama",
                display_name="Gemma 2 (9B)",
                context_window=8192,
                max_output_tokens=4096,
                supports_streaming=True,
                capabilities=["chat"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """
        List available Ollama models from the local server.

        Queries the Ollama server for actually installed models.
        """
        try:
            return await self._fetch_models_from_api(api_key)
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str | None = None) -> list[ModelInfo]:
        """Fetch models from local Ollama server."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = self._get_api_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.base_url}/api/tags",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        static_models = {m.model_id: m for m in self._get_static_models()}
        models = []

        for model in data.get("models", []):
            model_name = model.get("name", "")
            # Strip tag suffix for matching (e.g., "llama3.2:latest" -> "llama3.2")
            base_name = model_name.split(":")[0] if ":" in model_name else model_name

            # Check if we have static info for this model
            if base_name in static_models:
                model_info = static_models[base_name]
                # Update model_id to actual name with tag
                models.append(
                    ModelInfo(
                        model_id=model_name,
                        provider_id="ollama",
                        display_name=model_info.display_name,
                        context_window=model_info.context_window,
                        max_output_tokens=model_info.max_output_tokens,
                        supports_streaming=model_info.supports_streaming,
                        supports_vision=model_info.supports_vision,
                        supports_reasoning=model_info.supports_reasoning,
                        capabilities=model_info.capabilities,
                    )
                )
            elif model_name in static_models:
                models.append(static_models[model_name])
            else:
                # Create basic info for unknown models
                size_gb = model.get("size", 0) / (1024**3)
                models.append(
                    ModelInfo(
                        model_id=model_name,
                        provider_id="ollama",
                        display_name=f"{model_name} ({size_gb:.1f}GB)",
                        context_window=4096,
                        max_output_tokens=2048,
                        supports_streaming=True,
                        capabilities=["chat"],
                    )
                )

        return models if models else self._get_static_models()

    async def validate_api_key(self, api_key: str) -> bool:
        """
        Validate Ollama connection.

        For Ollama, this just checks if the server is reachable.
        """
        try:
            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/tags",
                    headers=headers,
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
            return False

    async def generate(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> GenerationResponse:
        """Generate text using Ollama API."""
        model = request.model or self.get_default_model()
        start_time = time.time()

        # Build messages for chat format
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

        # Add options
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.top_k is not None:
            options["top_k"] = request.top_k
        if request.stop_sequences:
            options["stop"] = request.stop_sequences

        if options:
            payload["options"] = options

        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = self._get_api_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            message = data.get("message", {})

            # Ollama provides eval_count and prompt_eval_count
            usage = None
            if "eval_count" in data or "prompt_eval_count" in data:
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
                }

            return GenerationResponse(
                text=message.get("content", ""),
                model_used=data.get("model", model),
                provider="ollama",
                finish_reason="stop" if data.get("done") else None,
                usage=usage,
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def generate_stream(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Ollama API."""
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

        # Add options
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.top_k is not None:
            options["top_k"] = request.top_k
        if request.stop_sequences:
            options["stop"] = request.stop_sequences

        if options:
            payload["options"] = options

        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = self._get_api_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        try:
            async with (
                httpx.AsyncClient(timeout=300.0) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    headers=headers,
                    json=payload,
                ) as response,
            ):
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        import json

                        data = json.loads(line)

                        message = data.get("message", {})
                        content = message.get("content", "")

                        if content:
                            yield StreamChunk(
                                text=content,
                                is_final=False,
                            )

                        if data.get("done"):
                            yield StreamChunk(text="", is_final=True)
                            break
                    except Exception as e:
                        logger.warning(f"Parse error: {e}")
                        continue
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
