"""
Routeway AI Gateway Client implementing LLMProvider interface.

This client wraps the RoutewayProvider for use with the ProviderFactory.
"""

import time
from collections.abc import AsyncIterator

import tiktoken
from fastapi import status

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import logger
from app.domain.interfaces import LLMProvider
from app.domain.models import PromptRequest, PromptResponse, StreamChunk
from app.infrastructure.providers.routeway_provider import RoutewayProvider


class RoutewayClient(LLMProvider):
    """Routeway AI Gateway client implementing LLMProvider interface."""

    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        api_key = self.config.ROUTEWAY_API_KEY
        if not api_key:
            logger.warning("Routeway API Key not set")
            self._provider = None
        else:
            self._provider = RoutewayProvider(config=self.config)
            logger.info("RoutewayClient configured in DIRECT mode")

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        # Routeway API keys start with 'clsk-'
        return api_key.startswith("clsk-") and len(api_key) > 10

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Generate text using Routeway."""
        if not request.prompt:
            raise AppError("Prompt is required", status_code=status.HTTP_400_BAD_REQUEST)

        if not self._provider:
            raise AppError(
                "Routeway client not initialized. Check API key.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        start_time = time.time()
        model_name = request.model or self.config.ROUTEWAY_MODEL

        try:
            async with self._provider as provider:
                response = await provider.generate(request, model_name)

            latency = (time.time() - start_time) * 1000

            # Update latency if not already set
            if response.latency_ms is None:
                response.latency_ms = latency

            return response

        except AppError:
            raise
        except Exception as e:
            logger.error(f"Routeway generation failed: {e!s}", exc_info=True)
            raise AppError(
                f"Routeway generation failed: {e!s}", status_code=status.HTTP_502_BAD_GATEWAY
            )

    async def check_health(self) -> bool:
        """Check Routeway API health."""
        try:
            if not self._provider:
                return False

            async with self._provider as provider:
                return await provider.check_health()
        except Exception as e:
            logger.error(f"Routeway health check failed: {e!s}")
            return False

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation using Routeway."""
        if not request.prompt:
            raise AppError("Prompt is required", status_code=status.HTTP_400_BAD_REQUEST)

        if not self._provider:
            raise AppError(
                "Routeway client not initialized. Check API key.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        model_name = request.model or self.config.ROUTEWAY_MODEL

        try:
            async with self._provider as provider:
                async for chunk in provider.generate_stream(request, model_name):
                    yield chunk

        except AppError:
            raise
        except Exception as e:
            logger.error(f"Routeway streaming failed: {e!s}", exc_info=True)
            raise AppError(f"Streaming failed: {e!s}", status_code=status.HTTP_502_BAD_GATEWAY)

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens using tiktoken.

        Routeway proxies to various models, so we use tiktoken
        with cl100k_base encoding as a reasonable approximation.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting failed: {e!s}", exc_info=True)
            raise AppError(f"Token counting failed: {e!s}", status_code=status.HTTP_502_BAD_GATEWAY)
