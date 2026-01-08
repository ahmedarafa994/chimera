"""
Azure OpenAI Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for Azure OpenAI Service integration,
supporting Azure-hosted OpenAI models with enterprise features.
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


class AzurePlugin(BaseProviderPlugin):
    """
    Provider plugin for Azure OpenAI Service.

    Azure OpenAI provides enterprise-grade access to OpenAI models
    with additional security, compliance, and regional deployment options.
    """

    @property
    def provider_type(self) -> str:
        return "azure"

    @property
    def display_name(self) -> str:
        return "Azure OpenAI"

    @property
    def aliases(self) -> list[str]:
        return ["azure-openai", "azureopenai", "azure-ai"]

    @property
    def api_key_env_var(self) -> str:
        return "AZURE_OPENAI_API_KEY"

    @property
    def base_url(self) -> str:
        """
        Azure OpenAI requires a resource endpoint.
        Format: https://{resource-name}.openai.azure.com
        """
        return os.environ.get(
            "AZURE_OPENAI_ENDPOINT",
            ""
        )

    @property
    def api_version(self) -> str:
        """Azure OpenAI API version."""
        return os.environ.get(
            "AZURE_OPENAI_API_VERSION",
            "2024-08-01-preview"
        )

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.EMBEDDINGS,
            ProviderCapability.JSON_MODE,
            ProviderCapability.SYSTEM_MESSAGES,
        ]

    def get_default_model(self) -> str:
        # Azure uses deployment names, not model names
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    def _get_deployment_name(self, model: str | None) -> str:
        """
        Get the Azure deployment name for a model.

        In Azure, you deploy models to named deployments.
        This mapping can be customized via environment variables.
        """
        if not model:
            return self.get_default_model()

        # Check for deployment name mapping
        env_key = f"AZURE_DEPLOYMENT_{model.upper().replace('-', '_')}"
        return os.environ.get(env_key, model)

    def _get_static_models(self) -> list[ModelInfo]:
        """
        Return static model definitions for Azure OpenAI.

        Note: Actual available models depend on Azure deployment.
        """
        return [
            # GPT-4o Series
            ModelInfo(
                model_id="gpt-4o",
                provider_id="azure",
                display_name="GPT-4o (Azure)",
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.0025,
                output_price_per_1k=0.01,
                capabilities=[
                    "chat", "vision", "function_calling"
                ],
            ),
            ModelInfo(
                model_id="gpt-4o-mini",
                provider_id="azure",
                display_name="GPT-4o Mini (Azure)",
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.00015,
                output_price_per_1k=0.0006,
                capabilities=[
                    "chat", "vision", "function_calling"
                ],
            ),
            # GPT-4 Turbo
            ModelInfo(
                model_id="gpt-4-turbo",
                provider_id="azure",
                display_name="GPT-4 Turbo (Azure)",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.01,
                output_price_per_1k=0.03,
                capabilities=[
                    "chat", "vision", "function_calling"
                ],
            ),
            # GPT-4 (Standard)
            ModelInfo(
                model_id="gpt-4",
                provider_id="azure",
                display_name="GPT-4 (Azure)",
                context_window=8192,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.03,
                output_price_per_1k=0.06,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="gpt-4-32k",
                provider_id="azure",
                display_name="GPT-4 32K (Azure)",
                context_window=32768,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.06,
                output_price_per_1k=0.12,
                capabilities=["chat", "function_calling"],
            ),
            # GPT-3.5 Turbo
            ModelInfo(
                model_id="gpt-35-turbo",
                provider_id="azure",
                display_name="GPT-3.5 Turbo (Azure)",
                context_window=16384,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.0005,
                output_price_per_1k=0.0015,
                capabilities=["chat", "function_calling"],
            ),
            ModelInfo(
                model_id="gpt-35-turbo-16k",
                provider_id="azure",
                display_name="GPT-3.5 Turbo 16K (Azure)",
                context_window=16384,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                input_price_per_1k=0.003,
                output_price_per_1k=0.004,
                capabilities=["chat", "function_calling"],
            ),
            # Embedding Models
            ModelInfo(
                model_id="text-embedding-ada-002",
                provider_id="azure",
                display_name="Ada Embedding V2 (Azure)",
                context_window=8191,
                supports_streaming=False,
                input_price_per_1k=0.0001,
                capabilities=["embeddings"],
            ),
            ModelInfo(
                model_id="text-embedding-3-small",
                provider_id="azure",
                display_name="Embedding 3 Small (Azure)",
                context_window=8191,
                supports_streaming=False,
                input_price_per_1k=0.00002,
                capabilities=["embeddings"],
            ),
            ModelInfo(
                model_id="text-embedding-3-large",
                provider_id="azure",
                display_name="Embedding 3 Large (Azure)",
                context_window=8191,
                supports_streaming=False,
                input_price_per_1k=0.00013,
                capabilities=["embeddings"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """
        List available Azure OpenAI models.

        Attempts to fetch from Azure API, falls back to static list.
        """
        key = self._get_api_key(api_key)

        if key and self.base_url:
            try:
                return await self._fetch_models_from_api(key)
            except Exception as e:
                logger.warning(f"Failed to fetch Azure models: {e}")

        return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str) -> list[ModelInfo]:
        """Fetch deployments from Azure OpenAI."""
        if not self.base_url:
            raise ValueError("Azure OpenAI endpoint not configured")

        url = f"{self.base_url}/openai/deployments"
        params = {"api-version": self.api_version}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                params=params,
                headers={
                    "api-key": api_key,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        static_models = {m.model_id: m for m in self._get_static_models()}
        models = []

        for deployment in data.get("data", []):
            deployment_id = deployment.get("id", "")
            model_name = deployment.get("model", "")

            # Map Azure model names to our static models
            # Azure uses slightly different naming (e.g., gpt-35-turbo)
            if model_name in static_models:
                base_model = static_models[model_name]
                models.append(ModelInfo(
                    model_id=deployment_id,
                    provider_id="azure",
                    display_name=f"{base_model.display_name}",
                    context_window=base_model.context_window,
                    max_output_tokens=base_model.max_output_tokens,
                    supports_streaming=base_model.supports_streaming,
                    supports_function_calling=(
                        base_model.supports_function_calling
                    ),
                    supports_vision=base_model.supports_vision,
                    input_price_per_1k=base_model.input_price_per_1k,
                    output_price_per_1k=base_model.output_price_per_1k,
                    capabilities=base_model.capabilities,
                ))
            else:
                models.append(ModelInfo(
                    model_id=deployment_id,
                    provider_id="azure",
                    display_name=f"{deployment_id} ({model_name})",
                    context_window=8192,
                    supports_streaming=True,
                    capabilities=["chat"],
                ))

        return models if models else self._get_static_models()

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate Azure OpenAI API key and endpoint."""
        if not self.base_url:
            logger.warning("Azure OpenAI endpoint not configured")
            return False

        try:
            url = f"{self.base_url}/openai/deployments"
            params = {"api-version": self.api_version}

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers={"api-key": api_key},
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Azure OpenAI validation failed: {e}")
            return False

    async def generate(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> GenerationResponse:
        """Generate text using Azure OpenAI API."""
        key = self._get_api_key(api_key)
        if not key:
            raise ValueError("Azure OpenAI API key is required")
        if not self.base_url:
            raise ValueError("Azure OpenAI endpoint is required")

        deployment = self._get_deployment_name(request.model)
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

        url = (
            f"{self.base_url}/openai/deployments/{deployment}"
            f"/chat/completions"
        )
        params = {"api-version": self.api_version}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    params=params,
                    headers={
                        "api-key": key,
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
                model_used=data.get("model", deployment),
                provider="azure",
                finish_reason=choice.get("finish_reason"),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Azure OpenAI API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Azure OpenAI generation error: {e}")
            raise

    async def generate_stream(
        self, request: GenerationRequest, api_key: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Azure OpenAI API."""
        key = self._get_api_key(api_key)
        if not key:
            raise ValueError("Azure OpenAI API key is required")
        if not self.base_url:
            raise ValueError("Azure OpenAI endpoint is required")

        deployment = self._get_deployment_name(request.model)

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

        url = (
            f"{self.base_url}/openai/deployments/{deployment}"
            f"/chat/completions"
        )
        params = {"api-version": self.api_version}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    url,
                    params=params,
                    headers={
                        "api-key": key,
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
            logger.error(f"Azure OpenAI streaming error: {e}")
            raise
