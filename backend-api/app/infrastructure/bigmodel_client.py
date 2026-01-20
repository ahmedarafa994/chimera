"""BigModel (ZhiPu AI) Client implementing LLMProvider interface.

This client wraps the BigModelProvider for use with the ProviderFactory.
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
from app.infrastructure.providers.bigmodel_provider import BigModelProvider


class BigModelClient(LLMProvider):
    """BigModel (ZhiPu AI) client implementing LLMProvider interface."""

    def __init__(self, config: Settings = None) -> None:
        self.config = config or get_settings()
        api_key = self.config.BIGMODEL_API_KEY
        if not api_key:
            logger.warning("BigModel API Key not set")
            self._provider = None
        else:
            self._provider = BigModelProvider(config=self.config)
            logger.info("BigModelClient configured in DIRECT mode")

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        # BigModel API keys have variable formats
        return len(api_key) > 10

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Generate text using BigModel."""
        if not request.prompt:
            msg = "Prompt is required"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        if not self._provider:
            msg = "BigModel client not initialized. Check API key."
            raise AppError(
                msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        start_time = time.time()
        model_name = request.model or self.config.BIGMODEL_MODEL

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
            logger.error(f"BigModel generation failed: {e!s}", exc_info=True)
            msg = f"BigModel generation failed: {e!s}"
            raise AppError(
                msg,
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    async def check_health(self) -> bool:
        """Check BigModel API health."""
        try:
            if not self._provider:
                return False

            async with self._provider as provider:
                return await provider.check_health()
        except Exception as e:
            logger.error(f"BigModel health check failed: {e!s}")
            return False

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation using BigModel."""
        if not request.prompt:
            msg = "Prompt is required"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        if not self._provider:
            msg = "BigModel client not initialized. Check API key."
            raise AppError(
                msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        model_name = request.model or self.config.BIGMODEL_MODEL

        try:
            async with self._provider as provider:
                async for chunk in provider.generate_stream(request, model_name):
                    yield chunk

        except AppError:
            raise
        except Exception as e:
            logger.error(f"BigModel streaming failed: {e!s}", exc_info=True)
            msg = f"Streaming failed: {e!s}"
            raise AppError(msg, status_code=status.HTTP_502_BAD_GATEWAY)

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens using tiktoken.

        BigModel uses a tokenizer similar to OpenAI's, so we use tiktoken
        with cl100k_base encoding as a reasonable approximation.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting failed: {e!s}", exc_info=True)
            msg = f"Token counting failed: {e!s}"
            raise AppError(msg, status_code=status.HTTP_502_BAD_GATEWAY)
