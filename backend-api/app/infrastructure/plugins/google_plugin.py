"""
Google Provider Plugin - Implementation for Gemini models.

This module implements the Google provider plugin, supporting Gemini Pro,
Gemini Flash, and other Google AI models through the unified provider interface.
"""

import logging
import os
from typing import Any

import google.generativeai as genai

from app.domain.interfaces import BaseLLMClient
from app.domain.models import Capability, Model, PromptResponse
from app.infrastructure.plugins.base_plugin import BaseProviderPlugin

logger = logging.getLogger(__name__)


class GoogleClient(BaseLLMClient):
    """
    Google-specific LLM client implementation.

    Wraps the Google Generative AI SDK to implement the BaseLLMClient interface.
    """

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        api_key: str,
        **kwargs: Any,
    ):
        """
        Initialize Google client.

        Args:
            provider_id: Provider identifier ('google')
            model_id: Model identifier (e.g., 'gemini-1.5-pro')
            api_key: Google API key
            **kwargs: Additional configuration
        """
        self._provider_id = provider_id
        self._model_id = model_id

        # Configure API
        genai.configure(api_key=api_key)

        # Initialize model
        self._model = genai.GenerativeModel(model_id)
        logger.debug(f"Initialized Google client for model: {model_id}")

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
        Generate text completion using Google Gemini API.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            system_instruction: Optional system instruction
            **kwargs: Additional Google-specific parameters

        Returns:
            PromptResponse with generated text and metadata
        """
        # Build generation config
        generation_config = {
            "temperature": temperature,
        }

        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        # Merge additional kwargs
        generation_config.update(kwargs)

        try:
            # Combine system instruction with prompt if provided
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"

            # Call Google API
            response = await self._model.generate_content_async(
                full_prompt,
                generation_config=generation_config,
            )

            # Extract response
            content = response.text if response.text else ""

            # Build usage info (Google API doesn't always provide detailed usage)
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            # Build PromptResponse
            return PromptResponse(
                text=content,
                provider=self._provider_id,
                model=self._model_id,
                usage=usage,
                finish_reason=(
                    response.candidates[0].finish_reason.name if response.candidates else None
                ),
            )

        except Exception as e:
            logger.error(f"Google generation failed: {e}")
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
        Stream text completion from Google Gemini API.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_instruction: Optional system instruction
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        # Build generation config
        generation_config = {
            "temperature": temperature,
        }

        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        generation_config.update(kwargs)

        try:
            # Combine system instruction with prompt if provided
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"

            # Stream response
            response = await self._model.generate_content_async(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Google streaming failed: {e}")
            raise

    def get_capabilities(self) -> set[Capability]:
        """Get the capabilities supported by this model."""
        capabilities = {
            Capability.CHAT,
            Capability.STREAMING,
        }

        # Add vision for Gemini Pro Vision and 1.5 Pro
        if "vision" in self._model_id or "1.5-pro" in self._model_id:
            capabilities.add(Capability.VISION)

        return capabilities

    async def close(self) -> None:
        """Clean up resources."""
        # Google SDK doesn't require explicit cleanup
        logger.debug(f"Closed Google client for model: {self._model_id}")


class GooglePlugin(BaseProviderPlugin):
    """
    Google provider plugin implementation.

    Provides factory methods and metadata for Gemini models.
    Supports aliases: 'google' and 'gemini'.
    """

    def __init__(self):
        """Initialize Google plugin."""
        super().__init__(
            provider_id="google",
            provider_name="Google",
            api_key_env_var="GOOGLE_API_KEY",
            base_url=None,  # Google SDK handles endpoints
            is_enabled=True,
        )

    def _get_provider_capabilities(self) -> set[Capability]:
        """Get capabilities supported by Google/Gemini."""
        return {
            Capability.CHAT,
            Capability.STREAMING,
            Capability.VISION,
        }

    def _get_api_key(self) -> str | None:
        """Get Google API key from environment."""
        return os.getenv(self._api_key_env_var)

    def _build_model_list(self) -> list[Model]:
        """Build list of available Google/Gemini models (January 2026)."""
        models = [
            Model(
                id="gemini-3-pro",
                name="Gemini 3 Pro (Flagship)",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=1000000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.00125,
                pricing_output_per_1k=0.005,
                is_enabled=True,
            ),
            Model(
                id="gemini-3-flash",
                name="Gemini 3 Flash",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=1000000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.00001875,
                pricing_output_per_1k=0.000075,
                is_enabled=True,
            ),
            Model(
                id="gemini-2.5-pro-latest",
                name="Gemini 2.5 Pro",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=1000000,
                max_output_tokens=8192,
                pricing_input_per_1k=0.00125,
                pricing_output_per_1k=0.005,
                is_enabled=True,
            ),
            Model(
                id="gemini-1.5-pro",
                name="Gemini 1.5 Pro",
                provider_id=self._provider_id,
                capabilities={
                    Capability.CHAT,
                    Capability.STREAMING,
                    Capability.VISION,
                },
                context_window=2000000,  # 2M tokens!
                max_output_tokens=8192,
                pricing_input_per_1k=0.00125,
                pricing_output_per_1k=0.00375,
                is_enabled=True,
            ),
        ]

        return models

    def create_client(self, model_id: str, **kwargs: Any) -> BaseLLMClient:
        """
        Create a Google client for the specified model.

        Args:
            model_id: The model identifier
            **kwargs: Additional configuration

        Returns:
            GoogleClient instance

        Raises:
            ValueError: If model is not available or API key missing
        """
        # Validate model
        if not self.validate_model_id(model_id):
            available = [m.id for m in self.get_available_models()]
            raise ValueError(
                f"Model '{model_id}' not available from Google. "
                f"Available models: {', '.join(available)}"
            )

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(
                f"Google API key not configured. "
                f"Set {self._api_key_env_var} environment variable."
            )

        # Create client
        return GoogleClient(
            provider_id=self._provider_id,
            model_id=model_id,
            api_key=api_key,
            **kwargs,
        )
