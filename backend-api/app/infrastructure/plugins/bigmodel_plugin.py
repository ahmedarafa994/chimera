"""
BigModel Provider Plugin - Implementation for BigModel (Zhipu AI) models.

This module implements the BigModel provider plugin, supporting GLM models
through the unified provider interface.
"""

import logging
import os
from typing import Any, Optional

from app.domain.interfaces import BaseLLMClient
from app.domain.models import Capability, Model
from app.infrastructure.bigmodel_client import BigModelClient as ExistingBigModelClient
from app.infrastructure.plugins.base_plugin import BaseProviderPlugin

logger = logging.getLogger(__name__)


class BigModelClient(BaseLLMClient):
    """
    BigModel-specific LLM client implementation.

    Wraps the existing BigModel client to implement the BaseLLMClient interface.
    """

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        api_key: str,
        **kwargs: Any,
    ):
        """
        Initialize BigModel client.

        Args:
            provider_id: Provider identifier ('bigmodel')
            model_id: Model identifier (e.g., 'glm-4')
            api_key: BigModel API key
            **kwargs: Additional configuration
        """
        self._provider_id = provider_id
        self._model_id = model_id
        self._underlying_client = ExistingBigModelClient(api_key=api_key, model=model_id)
        logger.debug(f"Initialized BigModel client for model: {model_id}")

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
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        """Generate text completion using BigModel API."""
        return await self._underlying_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
            **kwargs
        )

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        """Stream text completion from BigModel API."""
        async for chunk in self._underlying_client.stream(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
            **kwargs
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
        logger.debug(f"Closed BigModel client for model: {self._model_id}")


class BigModelPlugin(BaseProviderPlugin):
    """
    BigModel provider plugin implementation.

    Provides factory methods and metadata for BigModel/GLM models.
    """

    def __init__(self):
        """Initialize BigModel plugin."""
        super().__init__(
            provider_id="bigmodel",
            provider_name="BigModel (Zhipu AI)",
            api_key_env_var="BIGMODEL_API_KEY",
            base_url=os.getenv("BIGMODEL_BASE_URL"),
            is_enabled=True,
        )

    def _get_provider_capabilities(self) -> set[Capability]:
        """Get capabilities supported by BigModel."""
        return {
            Capability.CHAT,
            Capability.STREAMING,
        }

    def _get_api_key(self) -> Optional[str]:
        """Get BigModel API key from environment."""
        return os.getenv(self._api_key_env_var)

    def _build_model_list(self) -> list[Model]:
        """Build list of available BigModel models (January 2026)."""
        models = [
            Model(
                id="glm-4.7",
                name="GLM-4.7 (Flagship)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=128000,
                max_output_tokens=4096,
                pricing_input_per_1k=0.05,
                pricing_output_per_1k=0.05,
                is_enabled=True,
            ),
            Model(
                id="glm-4.6v",
                name="GLM-4.6V (Vision, 108B)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=128000,
                max_output_tokens=4096,
                pricing_input_per_1k=0.05,
                pricing_output_per_1k=0.05,
                is_enabled=True,
            ),
            Model(
                id="glm-4-plus",
                name="GLM-4-Plus",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=128000,
                max_output_tokens=4096,
                pricing_input_per_1k=0.05,
                pricing_output_per_1k=0.05,
                is_enabled=True,
            ),
            Model(
                id="glm-4-flash",
                name="GLM-4 Flash",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=128000,
                max_output_tokens=4096,
                pricing_input_per_1k=0.0001,
                pricing_output_per_1k=0.0001,
                is_enabled=True,
            ),
        ]

        return models

    def create_client(self, model_id: str, **kwargs: Any) -> BaseLLMClient:
        """
        Create a BigModel client for the specified model.

        Args:
            model_id: The model identifier
            **kwargs: Additional configuration

        Returns:
            BigModelClient instance

        Raises:
            ValueError: If model is not available or API key missing
        """
        # Validate model
        if not self.validate_model_id(model_id):
            available = [m.id for m in self.get_available_models()]
            raise ValueError(
                f"Model '{model_id}' not available from BigModel. "
                f"Available models: {', '.join(available)}"
            )

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(
                f"BigModel API key not configured. "
                f"Set {self._api_key_env_var} environment variable."
            )

        # Create client
        return BigModelClient(
            provider_id=self._provider_id,
            model_id=model_id,
            api_key=api_key,
            **kwargs,
        )
