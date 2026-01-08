"""
Custom Provider Plugin for Project Chimera.

Provides a base class for creating custom provider plugins with
configurable endpoints and behaviors.
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


class CustomPlugin(BaseProviderPlugin):
    """
    Configurable custom provider plugin.

    This plugin allows dynamic configuration of custom AI providers
    through environment variables or runtime configuration.

    Configuration via environment variables:
    - CUSTOM_PROVIDER_NAME: Provider identifier
    - CUSTOM_PROVIDER_DISPLAY_NAME: Human-readable name
    - CUSTOM_API_KEY: API key for the provider
    - CUSTOM_BASE_URL: Base URL for the API
    - CUSTOM_DEFAULT_MODEL: Default model to use
    """

    def __init__(
        self,
        provider_name: str | None = None,
        display_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        custom_capabilities: list[ProviderCapability] | None = None,
    ):
        """
        Initialize custom provider with optional configuration.

        Args:
            provider_name: Unique identifier for this provider
            display_name: Human-readable display name
            base_url: Base URL for the API endpoint
            api_key: API key for authentication
            default_model: Default model identifier
            custom_capabilities: List of supported capabilities
        """
        super().__init__()
        self._provider_name = provider_name
        self._display_name = display_name
        self._base_url = base_url
        self._api_key = api_key
        self._default_model = default_model
        self._custom_capabilities = custom_capabilities

    @property
    def provider_type(self) -> str:
        return self._provider_name or os.environ.get(
            "CUSTOM_PROVIDER_NAME", "custom"
        )

    @property
    def display_name(self) -> str:
        return self._display_name or os.environ.get(
            "CUSTOM_PROVIDER_DISPLAY_NAME", "Custom Provider"
        )

    @property
    def aliases(self) -> list[str]:
        return ["custom-provider"]

    @property
    def api_key_env_var(self) -> str:
        return "CUSTOM_API_KEY"

    @property
    def base_url(self) -> str:
        return self._base_url or os.environ.get(
            "CUSTOM_BASE_URL",
            "http://localhost:8000/v1"
        )

    @property
    def capabilities(self) -> list[ProviderCapability]:
        if self._custom_capabilities:
            return self._custom_capabilities
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.SYSTEM_MESSAGES,
        ]

    def get_default_model(self) -> str:
        return self._default_model or os.environ.get(
            "CUSTOM_DEFAULT_MODEL", "default"
        )

    def _get_api_key(self, api_key: str | None = None) -> str | None:
        """Get API key from parameter, instance, or environment."""
        if api_key:
            return api_key
        if self._api_key:
            return self._api_key
        return os.environ.get(self.api_key_env_var)

    def _get_static_models(self) -> list[ModelInfo]:
        """Return default model definition."""
        return [
            ModelInfo(
                model_id=self.get_default_model(),
                provider_id=self.provider_type,
                display_name=f"{self.display_name} - Default",
                context_window=4096,
                max_output_tokens=2048,
                supports_streaming=True,
                capabilities=["chat"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """List available models from custom provider."""
        try:
            return await self._fetch_models_from_api(api_key)
        except Exception as e:
            logger.warning(f"Failed to fetch custom models: {e}")
            return self._get_static_models()

    async def _fetch_models_from_api(
        self, api_key: str | None = None
    ) -> list[ModelInfo]:
        """Fetch models from custom server (OpenAI-compatible format)."""
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
            models.append(ModelInfo(
                model_id=model_id,
                provider_id=self.provider_type,
                display_name=model.get("id", model_id),
                context_window=model.get("context_length", 4096),
                max_output_tokens=model.get("max_tokens", 2048),
                supports_streaming=True,
                capabilities=["chat"],
            ))

        return models if models else self._get_static_models()

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate connection to custom server."""
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
            logger.warning(f"Custom server connection check failed: {e}")
            return False

    async def generate(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> GenerationResponse:
        """Generate text using custom provider (OpenAI-compatible)."""
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
                provider=self.provider_type,
                finish_reason=choice.get("finish_reason"),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                } if usage else None,
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Custom provider error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Custom provider generation error: {e}")
            raise

    async def generate_stream(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from custom provider."""
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

        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = self._get_api_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
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
            logger.error(f"Custom provider streaming error: {e}")
            raise


def create_custom_plugin(
    provider_name: str,
    display_name: str,
    base_url: str,
    api_key: str | None = None,
    default_model: str = "default",
    capabilities: list[ProviderCapability] | None = None,
) -> CustomPlugin:
    """
    Factory function to create a configured custom plugin.

    Args:
        provider_name: Unique identifier for the provider
        display_name: Human-readable display name
        base_url: Base URL for the API
        api_key: Optional API key
        default_model: Default model to use
        capabilities: List of supported capabilities

    Returns:
        Configured CustomPlugin instance
    """
    return CustomPlugin(
        provider_name=provider_name,
        display_name=display_name,
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        custom_capabilities=capabilities,
    )
