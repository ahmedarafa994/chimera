"""
DeepSeek Provider Plugin - Implementation for DeepSeek AI models.

This module implements the DeepSeek provider plugin, supporting DeepSeek models
through the unified provider interface.
"""

import logging
import os
from typing import Any, Optional

from app.domain.interfaces import BaseLLMClient
from app.domain.models import Capability, Model
from app.infrastructure.deepseek_client import DeepSeekClient as ExistingDeepSeekClient
from app.infrastructure.plugins.base_plugin import BaseProviderPlugin

logger = logging.getLogger(__name__)


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek-specific LLM client implementation.

    Wraps the existing DeepSeek client to implement the BaseLLMClient interface.
    """

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        api_key: str,
        **kwargs: Any,
    ):
        """
        Initialize DeepSeek client.

        Args:
            provider_id: Provider identifier ('deepseek')
            model_id: Model identifier (e.g., 'deepseek-chat')
            api_key: DeepSeek API key
            **kwargs: Additional configuration
        """
        self._provider_id = provider_id
        self._model_id = model_id
        self._underlying_client = ExistingDeepSeekClient(api_key=api_key, model=model_id)
        logger.debug(f"Initialized DeepSeek client for model: {model_id}")

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
        """Generate text completion using DeepSeek API."""
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
        """Stream text completion from DeepSeek API."""
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
        logger.debug(f"Closed DeepSeek client for model: {self._model_id}")


class DeepSeekPlugin(BaseProviderPlugin):
    """
    DeepSeek provider plugin implementation.

    Provides factory methods and metadata for DeepSeek models.
    """

    def __init__(self):
        """Initialize DeepSeek plugin."""
        super().__init__(
            provider_id="deepseek",
            provider_name="DeepSeek AI",
            api_key_env_var="DEEPSEEK_API_KEY",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            is_enabled=True,
        )

    def _get_provider_capabilities(self) -> set[Capability]:
        """Get capabilities supported by DeepSeek."""
        return {
            Capability.CHAT,
            Capability.STREAMING,
        }

    def _get_api_key(self) -> Optional[str]:
        """Get DeepSeek API key from environment."""
        return os.getenv(self._api_key_env_var)

    def _build_model_list(self) -> list[Model]:
        """Build list of available DeepSeek models (January 2026)."""
        models = [
            Model(
                id="deepseek-v4",
                name="DeepSeek-V4 (Flagship, 600B)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=128000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.00014,
                pricing_output_per_1k=0.00028,
                is_enabled=True,
            ),
            Model(
                id="deepseek-chat",
                name="DeepSeek V3 (Chat)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=64000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.00014,
                pricing_output_per_1k=0.00028,
                is_enabled=True,
            ),
            Model(
                id="deepseek-reasoner",
                name="DeepSeek R1 (Reasoning)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                },
                context_window=64000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.00055,
                pricing_output_per_1k=0.00219,
                is_enabled=True,
            ),
        ]

        return models

    def create_client(self, model_id: str, **kwargs: Any) -> BaseLLMClient:
        """
        Create a DeepSeek client for the specified model.

        Args:
            model_id: The model identifier
            **kwargs: Additional configuration

        Returns:
            DeepSeekClient instance

        Raises:
            ValueError: If model is not available or API key missing
        """
        # Validate model
        if not self.validate_model_id(model_id):
            available = [m.id for m in self.get_available_models()]
            raise ValueError(
                f"Model '{model_id}' not available from DeepSeek. "
                f"Available models: {', '.join(available)}"
            )

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(
                f"DeepSeek API key not configured. "
                f"Set {self._api_key_env_var} environment variable."
            )

        # Create client
        return DeepSeekClient(
            provider_id=self._provider_id,
            model_id=model_id,
            api_key=api_key,
            **kwargs,
        )
