"""
Modern Cursor client using OpenAI-compatible API.

Cursor uses an OpenAI-compatible API interface.
Implements the LLMProvider interface with native async support.
"""

import re
import time
from collections.abc import AsyncIterator

import tiktoken
from fastapi import status
from openai import APIError, AsyncOpenAI

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import logger
from app.domain.interfaces import LLMProvider
from app.domain.models import PromptRequest, PromptResponse, StreamChunk


class CursorClient(LLMProvider):
    """
    Cursor client using OpenAI-compatible API.

    Note: Cursor's API endpoint and authentication may vary.
    Configure via CURSOR_API_KEY and provider endpoint settings.
    """

    DEFAULT_ENDPOINT = "https://api.cursor.sh/v1"
    DEFAULT_MODEL = "cursor-small"

    def __init__(self, config: Settings | None = None):
        self.config = config or get_settings()
        self.endpoint = self.config.get_provider_endpoint("cursor") or self.DEFAULT_ENDPOINT
        api_key = getattr(self.config, "CURSOR_API_KEY", None)

        if not api_key:
            logger.warning("Cursor API Key not set for Direct mode")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=api_key, base_url=self.endpoint)
            logger.info("CursorClient configured in DIRECT mode")

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        return len(api_key) > 10

    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize user input."""
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        prompt = re.sub(r"<script.*?</script>", "", prompt, flags=re.IGNORECASE | re.DOTALL)
        prompt = re.sub(r"javascript:", "", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"on\w+\s*=", "", prompt, flags=re.IGNORECASE)

        max_len = getattr(self.config, "JAILBREAK_MAX_PROMPT_LENGTH", 10000)
        if len(prompt) > max_len:
            logger.warning(f"Prompt truncated from {len(prompt)} to {max_len}")
            prompt = prompt[:max_len]

        return prompt.strip()

    def _get_client(self, request: PromptRequest) -> AsyncOpenAI:
        """Get client, using override API key if provided."""
        if request.api_key:
            if not self._validate_api_key(request.api_key):
                raise AppError(
                    "Invalid API key format",
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            return AsyncOpenAI(api_key=request.api_key, base_url=self.endpoint)

        if not self.client:
            raise AppError(
                "Cursor client not initialized. Check API key.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return self.client

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Generate a response from Cursor."""
        if not request.prompt:
            raise AppError(
                "Prompt is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        sanitized_prompt = self._sanitize_prompt(request.prompt)
        client = self._get_client(request)
        model_name = request.model or getattr(self.config, "CURSOR_MODEL", self.DEFAULT_MODEL)
        start_time = time.time()

        try:
            messages = []
            if request.system_instruction:
                sanitized_system = self._sanitize_prompt(request.system_instruction)
                messages.append({"role": "system", "content": sanitized_system})
            messages.append({"role": "user", "content": sanitized_prompt})

            logger.info(f"Generating content with Cursor model: {model_name}")

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=(request.config.temperature if request.config else 0.7),
                max_tokens=(request.config.max_output_tokens if request.config else 2048),
                top_p=request.config.top_p if request.config else 0.95,
                stop=request.config.stop_sequences if request.config else None,
                stream=False,
            )

            latency = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "unknown"

            usage_metadata = {}
            if response.usage:
                usage_metadata = {
                    "prompt_token_count": response.usage.prompt_tokens,
                    "candidates_token_count": response.usage.completion_tokens,
                    "total_token_count": response.usage.total_tokens,
                }

            return PromptResponse(
                text=content,
                model_used=model_name,
                provider="cursor",
                usage_metadata=usage_metadata,
                finish_reason=finish_reason.upper(),
                latency_ms=latency,
            )

        except APIError as e:
            logger.error(f"Cursor API error: {e.message}")
            raise AppError(
                f"Cursor API error: {e.message}",
                status_code=self._map_error_code(e.status_code),
            )
        except Exception as e:
            logger.error(f"Cursor generation failed: {e!s}", exc_info=True)
            raise AppError(
                f"Cursor generation failed: {e!s}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    def _map_error_code(self, code: int | None) -> int:
        """Map API error codes to HTTP status codes."""
        if code is None:
            return status.HTTP_502_BAD_GATEWAY
        mapping = {
            400: status.HTTP_400_BAD_REQUEST,
            401: status.HTTP_401_UNAUTHORIZED,
            403: status.HTTP_403_FORBIDDEN,
            404: status.HTTP_404_NOT_FOUND,
            429: status.HTTP_429_TOO_MANY_REQUESTS,
            500: status.HTTP_502_BAD_GATEWAY,
            503: status.HTTP_503_SERVICE_UNAVAILABLE,
        }
        return mapping.get(code, status.HTTP_502_BAD_GATEWAY)

    async def check_health(self) -> bool:
        """Check if the Cursor API is healthy."""
        try:
            if not self.client:
                return False
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Cursor health check failed: {e!s}")
            return False

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Cursor."""
        if not request.prompt:
            raise AppError(
                "Prompt is required",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        sanitized_prompt = self._sanitize_prompt(request.prompt)
        client = self._get_client(request)
        model_name = request.model or getattr(self.config, "CURSOR_MODEL", self.DEFAULT_MODEL)

        try:
            messages = []
            if request.system_instruction:
                sanitized_system = self._sanitize_prompt(request.system_instruction)
                messages.append({"role": "system", "content": sanitized_system})
            messages.append({"role": "user", "content": sanitized_prompt})

            logger.info(f"Streaming content with Cursor model: {model_name}")

            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=(request.config.temperature if request.config else 0.7),
                max_tokens=(request.config.max_output_tokens if request.config else 2048),
                top_p=request.config.top_p if request.config else 0.95,
                stop=request.config.stop_sequences if request.config else None,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        text=chunk.choices[0].delta.content,
                        is_final=False,
                        finish_reason=None,
                    )

                if chunk.choices and chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        text="",
                        is_final=False,
                        finish_reason=chunk.choices[0].finish_reason.upper(),
                    )

            yield StreamChunk(text="", is_final=True, finish_reason="STOP")

        except APIError as e:
            logger.error(f"Cursor streaming error: {e.message}")
            raise AppError(
                f"Streaming failed: {e.message}",
                status_code=self._map_error_code(e.status_code),
            )
        except Exception as e:
            logger.error(f"Cursor streaming failed: {e!s}", exc_info=True)
            raise AppError(
                f"Streaming failed: {e!s}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens using tiktoken (cl100k_base approximation)."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting failed: {e!s}", exc_info=True)
            raise AppError(
                f"Token counting failed: {e!s}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )
