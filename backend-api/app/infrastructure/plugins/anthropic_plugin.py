"""
Anthropic Provider Plugin - Implementation for Claude models.

This module implements the Anthropic provider plugin, supporting Claude 3.5,
Claude 3, and other Anthropic models through the unified provider interface.
"""

import logging
import os
from typing import Any, Optional

from anthropic import AsyncAnthropic

from app.domain.interfaces import BaseLLMClient
from app.domain.models import Capability, Model, PromptResponse
from app.infrastructure.plugins.base_plugin import BaseProviderPlugin

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """
    Anthropic-specific LLM client implementation.

    Wraps the Anthropic Python SDK to implement the BaseLLMClient interface.
    """

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize Anthropic client.

        Args:
            provider_id: Provider identifier ('anthropic')
            model_id: Model identifier (e.g., 'claude-3-5-sonnet-20241022')
            api_key: Anthropic API key
            base_url: Optional custom base URL
            **kwargs: Additional Anthropic client configuration
        """
        self._provider_id = provider_id
        self._model_id = model_id

        # Initialize Anthropic client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncAnthropic(**client_kwargs)
        logger.debug(f"Initialized Anthropic client for model: {model_id}")

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
    ) -> PromptResponse:
        """
        Generate text completion using Anthropic API.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (default: 4096)
            system_instruction: Optional system prompt
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            PromptResponse with generated text and metadata
        """
        # Prepare parameters
        params = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }

        if system_instruction:
            params["system"] = system_instruction

        # Merge additional kwargs
        params.update(kwargs)

        try:
            # Call Anthropic API
            response = await self._client.messages.create(**params)

            # Extract response
            content = response.content[0].text if response.content else ""

            # Build PromptResponse
            return PromptResponse(
                text=content,
                provider=self._provider_id,
                model=self._model_id,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
            )

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Stream text completion from Anthropic API.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_instruction: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        # Prepare parameters
        params = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": True,
        }

        if system_instruction:
            params["system"] = system_instruction

        params.update(kwargs)

        try:
            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise

    def get_capabilities(self) -> set[Capability]:
        """Get the capabilities supported by this model."""
        capabilities = {
            Capability.CHAT,
            Capability.STREAMING,
        }

        # Add vision for Claude 3 Opus and Sonnet
        if "claude-3" in self._model_id and ("opus" in self._model_id or "sonnet" in self._model_id):
            capabilities.add(Capability.VISION)

        return capabilities

    async def close(self) -> None:
        """Clean up resources."""
        await self._client.close()
        logger.debug(f"Closed Anthropic client for model: {self._model_id}")


class AnthropicPlugin(BaseProviderPlugin):
    """
    Anthropic provider plugin implementation.

    Provides factory methods and metadata for Claude models.
    """

    def __init__(self):
        """Initialize Anthropic plugin."""
        super().__init__(
            provider_id="anthropic",
            provider_name="Anthropic",
            api_key_env_var="ANTHROPIC_API_KEY",
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            is_enabled=True,
        )

    def _get_provider_capabilities(self) -> set[Capability]:
        """Get capabilities supported by Anthropic."""
        return {
            Capability.CHAT,
            Capability.STREAMING,
            Capability.VISION,
        }

    def _get_api_key(self) -> Optional[str]:
        """Get Anthropic API key from environment."""
        return os.getenv(self._api_key_env_var)

    def _build_model_list(self) -> list[Model]:
        """Build list of available Anthropic models (January 2026)."""
        models = [
            Model(
                id="claude-opus-4.5",
                name="Claude Opus 4.5 (Flagship)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=200000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.015,
                pricing_output_per_1k=0.075,
                is_enabled=True,
            ),
            Model(
                id="claude-sonnet-4.5",
                name="Claude Sonnet 4.5",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=200000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.003,
                pricing_output_per_1k=0.015,
                is_enabled=True,
            ),
            Model(
                id="claude-haiku-4.5",
                name="Claude Haiku 4.5",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=200000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.0008,
                pricing_output_per_1k=0.004,
                is_enabled=True,
            ),
            Model(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet (Legacy)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=200000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.003,
                pricing_output_per_1k=0.015,
                is_enabled=True,
            ),
        ]

        return models

    def create_client(self, model_id: str, **kwargs: Any) -> BaseLLMClient:
        """
        Create an Anthropic client for the specified model.

        Args:
            model_id: The model identifier
            **kwargs: Additional configuration

        Returns:
            AnthropicClient instance

        Raises:
            ValueError: If model is not available or API key missing
        """
        # Validate model
        if not self.validate_model_id(model_id):
            available = [m.id for m in self.get_available_models()]
            raise ValueError(
                f"Model '{model_id}' not available from Anthropic. "
                f"Available models: {', '.join(available)}"
            )

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(
                f"Anthropic API key not configured. "
                f"Set {self._api_key_env_var} environment variable."
            )

        # Create client
        return AnthropicClient(
            provider_id=self._provider_id,
            model_id=model_id,
            api_key=api_key,
            base_url=self._base_url,
            **kwargs,
        )
