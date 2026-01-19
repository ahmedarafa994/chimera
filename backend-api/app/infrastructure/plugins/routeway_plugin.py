"""
Routeway Provider Plugin - Unified API access to 70+ AI models.

Routeway is a unified API aggregator providing seamless access to top AI models
from OpenAI, Anthropic, DeepSeek, Meta, and others at 40% lower cost than official providers.
API: https://api.routeway.ai
"""

import logging
import os
from typing import Any

from app.domain.interfaces import BaseLLMClient
from app.domain.models import Capability, Model
from app.infrastructure.plugins.base_plugin import BaseProviderPlugin
from app.infrastructure.routeway_client import RoutewayClient as ExistingRoutewayClient

logger = logging.getLogger(__name__)


class RoutewayClient(BaseLLMClient):
    """
    Routeway-specific LLM client implementation.

    Wraps the existing Routeway client to implement the BaseLLMClient interface.
    """

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        api_key: str,
        **kwargs: Any,
    ):
        """
        Initialize Routeway client.

        Args:
            provider_id: Provider identifier ('routeway')
            model_id: Model identifier
            api_key: Routeway API key
            **kwargs: Additional configuration
        """
        self._provider_id = provider_id
        self._model_id = model_id
        self._underlying_client = ExistingRoutewayClient(api_key=api_key, model=model_id)
        logger.debug(f"Initialized Routeway client for model: {model_id}")

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
    ):
        """Generate text completion using Routeway API."""
        return await self._underlying_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
            **kwargs,
        )

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        **kwargs: Any,
    ):
        """Stream text completion from Routeway API."""
        async for chunk in self._underlying_client.stream(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
            **kwargs,
        ):
            yield chunk

    def get_capabilities(self) -> set[Capability]:
        """Get the capabilities supported by this model."""
        return {
            Capability.CHAT,
            Capability.STREAMING,
        }

    async def close(self) -> None:
        """Clean up resources."""
        await self._underlying_client.close()
        logger.debug(f"Closed Routeway client for model: {self._model_id}")


class RoutewayPlugin(BaseProviderPlugin):
    """
    Routeway provider plugin implementation.

    Provides factory methods and metadata for Routeway models.
    """

    def __init__(self):
        """Initialize Routeway plugin."""
        super().__init__(
            provider_id="routeway",
            provider_name="Routeway AI",
            api_key_env_var="ROUTEWAY_API_KEY",
            base_url=os.getenv("ROUTEWAY_BASE_URL"),
            is_enabled=True,
        )

    def _get_provider_capabilities(self) -> set[Capability]:
        """Get capabilities supported by Routeway."""
        return {
            Capability.CHAT,
            Capability.STREAMING,
        }

    def _get_api_key(self) -> str | None:
        """Get Routeway API key from environment."""
        return os.getenv(self._api_key_env_var)

    def _build_model_list(self) -> list[Model]:
        """
        Build list of available Routeway models (January 2026).

        Note: Routeway is a unified API providing access to 70+ models from
        multiple providers (OpenAI, Anthropic, DeepSeek, Meta, etc.) at 40% lower cost.
        Listed here are the most popular models available through Routeway.
        """
        models = [
            # OpenAI models via Routeway
            Model(
                id="gpt-5.2",
                name="GPT-5.2 (via Routeway)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=200000,
                max_output_tokens=16384,
                pricing_input_per_1k=0.003,  # 40% discount
                pricing_output_per_1k=0.009,
                is_enabled=True,
            ),
            Model(
                id="o3-mini",
                name="o3-mini (via Routeway)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=128000,
                max_output_tokens=100000,
                pricing_input_per_1k=0.00066,  # 40% discount
                pricing_output_per_1k=0.00264,
                is_enabled=True,
            ),
            # Anthropic models via Routeway
            Model(
                id="claude-opus-4.5",
                name="Claude Opus 4.5 (via Routeway)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=200000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.009,  # 40% discount
                pricing_output_per_1k=0.045,
                is_enabled=True,
            ),
            # DeepSeek models via Routeway
            Model(
                id="deepseek-v4",
                name="DeepSeek-V4 (via Routeway)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=128000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.000084,  # 40% discount
                pricing_output_per_1k=0.000168,
                is_enabled=True,
            ),
            # Meta models via Routeway
            Model(
                id="llama-4-scout-17b-16e-instruct",
                name="Meta: Llama 4 Scout (via Routeway)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=128000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.0001,
                pricing_output_per_1k=0.000225,
                is_enabled=True,
            ),
        ]

        return models

    def create_client(self, model_id: str, **kwargs: Any) -> BaseLLMClient:
        """
        Create a Routeway client for the specified model.

        Args:
            model_id: The model identifier
            **kwargs: Additional configuration

        Returns:
            RoutewayClient instance

        Raises:
            ValueError: If model is not available or API key missing
        """
        # Validate model
        if not self.validate_model_id(model_id):
            available = [m.id for m in self.get_available_models()]
            raise ValueError(
                f"Model '{model_id}' not available from Routeway. "
                f"Available models: {', '.join(available)}"
            )

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(
                f"Routeway API key not configured. "
                f"Set {self._api_key_env_var} environment variable."
            )

        # Create client
        return RoutewayClient(
            provider_id=self._provider_id,
            model_id=model_id,
            api_key=api_key,
            **kwargs,
        )
