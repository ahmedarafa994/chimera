"""
OpenAI Provider Plugin - Implementation for OpenAI/GPT models.

This module implements the OpenAI provider plugin, supporting GPT-4, GPT-3.5,
and other OpenAI models through the unified provider interface.
"""

import logging
import os
from typing import Any

from openai import AsyncOpenAI

from app.domain.interfaces import BaseLLMClient
from app.domain.models import Capability, Model, PromptResponse
from app.infrastructure.plugins.base_plugin import BaseProviderPlugin

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    OpenAI-specific LLM client implementation.

    Wraps the OpenAI Python SDK to implement the BaseLLMClient interface.
    """

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI client.

        Args:
            provider_id: Provider identifier ('openai')
            model_id: Model identifier (e.g., 'gpt-4-turbo')
            api_key: OpenAI API key
            base_url: Optional custom base URL
            **kwargs: Additional OpenAI client configuration
        """
        self._provider_id = provider_id
        self._model_id = model_id

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)
        logger.debug(f"Initialized OpenAI client for model: {model_id}")

    @property
    def provider_id(self) -> str:
        """Get the provider identifier."""
        return self._provider_id

    @property
    def model_id(self) -> str:
        """Get the model identifier."""
        return self._model_id

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        """
        Generate text completion using OpenAI API.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            system_instruction: Optional system message
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            PromptResponse with generated text and metadata
        """
        # Build messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # Prepare parameters
        params = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        # Merge additional kwargs
        params.update(kwargs)

        try:
            # Call OpenAI API
            response = await self._client.chat.completions.create(**params)

            # Extract response
            content = response.choices[0].message.content or ""

            # Build PromptResponse
            return PromptResponse(
                text=content,
                provider=self._provider_id,
                model=self._model_id,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        **kwargs: Any,
    ):
        """
        Stream text completion from OpenAI API.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_instruction: Optional system message
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        # Build messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # Prepare parameters
        params = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        params.update(kwargs)

        try:
            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise

    def get_capabilities(self) -> set[Capability]:
        """Get the capabilities supported by this model."""
        capabilities = {
            Capability.CHAT,
            Capability.STREAMING,
        }

        # Add function calling for GPT-4 and GPT-3.5-turbo
        if "gpt-4" in self._model_id or "gpt-3.5-turbo" in self._model_id:
            capabilities.add(Capability.FUNCTION_CALLING)

        # Add vision for GPT-4V models
        if "vision" in self._model_id or "gpt-4-turbo" in self._model_id:
            capabilities.add(Capability.VISION)

        return capabilities

    async def close(self) -> None:
        """Clean up resources."""
        await self._client.close()
        logger.debug(f"Closed OpenAI client for model: {self._model_id}")


class OpenAIPlugin(BaseProviderPlugin):
    """
    OpenAI provider plugin implementation.

    Provides factory methods and metadata for OpenAI models.
    """

    def __init__(self):
        """Initialize OpenAI plugin."""
        super().__init__(
            provider_id="openai",
            provider_name="OpenAI",
            api_key_env_var="OPENAI_API_KEY",
            base_url=os.getenv("OPENAI_BASE_URL"),
            is_enabled=True,
        )

    def _get_provider_capabilities(self) -> set[Capability]:
        """Get capabilities supported by OpenAI."""
        return {
            Capability.CHAT,
            Capability.STREAMING,
            Capability.FUNCTION_CALLING,
            Capability.VISION,
        }

    def _get_api_key(self) -> str | None:
        """Get OpenAI API key from environment."""
        return os.getenv(self._api_key_env_var)

    def _build_model_list(self) -> list[Model]:
        """Build list of available OpenAI models (January 2026)."""
        models = [
            Model(
                id="gpt-5.2",
                name="GPT-5.2 (Flagship)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.FUNCTION_CALLING,
                    Capability.VISION,
                },
                context_window=200000,
                max_output_tokens=16384,
                pricing_input_per_1k=0.005,
                pricing_output_per_1k=0.015,
                is_enabled=True,
            ),
            Model(
                id="gpt-5.2-codex",
                name="GPT-5.2 Codex",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.FUNCTION_CALLING,
                },
                context_window=200000,
                max_output_tokens=16384,
                pricing_input_per_1k=0.005,
                pricing_output_per_1k=0.015,
                is_enabled=True,
            ),
            Model(
                id="o3-mini",
                name="o3-mini (Reasoning)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=128000,
                max_output_tokens=100000,
                pricing_input_per_1k=0.0011,
                pricing_output_per_1k=0.0044,
                is_enabled=True,
            ),
            Model(
                id="gpt-4.5",
                name="GPT-4.5",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.FUNCTION_CALLING,
                    Capability.VISION,
                },
                context_window=128000,
                max_output_tokens=16384,
                pricing_input_per_1k=0.003,
                pricing_output_per_1k=0.012,
                is_enabled=True,
            ),
            Model(
                id="gpt-4o",
                name="GPT-4o",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.FUNCTION_CALLING,
                    Capability.VISION,
                },
                context_window=128000,
                max_output_tokens=16384,
                pricing_input_per_1k=0.0025,
                pricing_output_per_1k=0.01,
                is_enabled=True,
            ),
        ]

        return models

    def create_client(self, model_id: str, **kwargs: Any) -> BaseLLMClient:
        """
        Create an OpenAI client for the specified model.

        Args:
            model_id: The model identifier
            **kwargs: Additional configuration

        Returns:
            OpenAIClient instance

        Raises:
            ValueError: If model is not available or API key missing
        """
        # Validate model
        if not self.validate_model_id(model_id):
            available = [m.id for m in self.get_available_models()]
            raise ValueError(
                f"Model '{model_id}' not available from OpenAI. "
                f"Available models: {', '.join(available)}"
            )

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(
                f"OpenAI API key not configured. "
                f"Set {self._api_key_env_var} environment variable."
            )

        # Create client
        return OpenAIClient(
            provider_id=self._provider_id,
            model_id=model_id,
            api_key=api_key,
            base_url=self._base_url,
            **kwargs,
        )
