"""Modern Qwen client using OpenAI-compatible API via DashScope.

Implements the LLMProvider interface with:
- Native async via AsyncOpenAI (OpenAI-compatible endpoint)
- Streaming generation
- Token counting via tiktoken
- Proper error handling
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
from app.domain.models import LLMProviderType, PromptRequest, PromptResponse, StreamChunk


class QwenClient(LLMProvider):
    """Qwen client using DashScope's OpenAI-compatible API.

    DashScope (Alibaba Cloud) provides Qwen models via an OpenAI-compatible interface.
    Default endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen-turbo"

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or get_settings()
        self.endpoint = self.config.get_provider_endpoint("qwen") or self.DEFAULT_ENDPOINT
        api_key = getattr(self.config, "QWEN_API_KEY", None)

        if not api_key:
            logger.warning("Qwen API Key not set for Direct mode")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=api_key, base_url=self.endpoint)
            logger.info("QwenClient configured in DIRECT mode via OpenAI-compatible SDK")

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        return len(api_key) > 10

    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize user input."""
        if not prompt:
            msg = "Prompt cannot be empty"
            raise ValueError(msg)

        prompt = re.sub(r"<script.*?</script>", "", prompt, flags=re.IGNORECASE | re.DOTALL)
        prompt = re.sub(r"javascript:", "", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"on\w+\s*=", "", prompt, flags=re.IGNORECASE)

        max_len = getattr(self.config, "JAILBREAK_MAX_PROMPT_LENGTH", 10000)
        if len(prompt) > max_len:
            logger.warning(f"Prompt truncated from {len(prompt)} to {max_len} chars")
            prompt = prompt[:max_len]

        return prompt.strip()

    def _get_client(self, request: PromptRequest) -> AsyncOpenAI:
        """Get client, using override API key if provided."""
        if request.api_key:
            if not self._validate_api_key(request.api_key):
                msg = "Invalid API key format"
                raise AppError(
                    msg,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            return AsyncOpenAI(api_key=request.api_key, base_url=self.endpoint)

        if not self.client:
            msg = "Qwen client not initialized. Check API key."
            raise AppError(
                msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return self.client

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Generate a response from Qwen."""
        if not request.prompt:
            msg = "Prompt is required"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        sanitized_prompt = self._sanitize_prompt(request.prompt)
        client = self._get_client(request)
        model_name = request.model or getattr(self.config, "QWEN_MODEL", self.DEFAULT_MODEL)
        start_time = time.time()

        try:
            messages = []
            if request.system_instruction:
                sanitized_system = self._sanitize_prompt(request.system_instruction)
                messages.append({"role": "system", "content": sanitized_system})
            messages.append({"role": "user", "content": sanitized_prompt})

            logger.info(f"Generating content with Qwen model: {model_name}")

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=request.config.temperature if request.config else 0.7,
                max_tokens=request.config.max_output_tokens if request.config else 2048,
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

            # Get provider type, fallback to string if not in enum
            try:
                provider_value = LLMProviderType.QWEN.value
            except AttributeError:
                provider_value = "qwen"

            return PromptResponse(
                text=content,
                model_used=model_name,
                provider=provider_value,
                usage_metadata=usage_metadata,
                finish_reason=finish_reason.upper(),
                latency_ms=latency,
            )

        except APIError as e:
            logger.error(f"Qwen API error: {e.message}")
            msg = f"Qwen API error: {e.message}"
            raise AppError(
                msg,
                status_code=self._map_error_code(e.status_code),
            )
        except Exception as e:
            logger.error(f"Qwen generation failed: {e!s}", exc_info=True)
            msg = f"Qwen generation failed: {e!s}"
            raise AppError(
                msg,
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
        """Check if the Qwen API is healthy."""
        try:
            if not self.client:
                return False
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Qwen health check failed: {e!s}")
            return False

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Qwen."""
        if not request.prompt:
            msg = "Prompt is required"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        sanitized_prompt = self._sanitize_prompt(request.prompt)
        client = self._get_client(request)
        model_name = request.model or getattr(self.config, "QWEN_MODEL", self.DEFAULT_MODEL)

        try:
            messages = []
            if request.system_instruction:
                sanitized_system = self._sanitize_prompt(request.system_instruction)
                messages.append({"role": "system", "content": sanitized_system})
            messages.append({"role": "user", "content": sanitized_prompt})

            logger.info(f"Streaming content with Qwen model: {model_name}")

            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=request.config.temperature if request.config else 0.7,
                max_tokens=request.config.max_output_tokens if request.config else 2048,
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
            logger.error(f"Qwen streaming error: {e.message}")
            msg = f"Streaming failed: {e.message}"
            raise AppError(
                msg,
                status_code=self._map_error_code(e.status_code),
            )
        except Exception as e:
            logger.error(f"Qwen streaming failed: {e!s}", exc_info=True)
            msg = f"Streaming failed: {e!s}"
            raise AppError(
                msg,
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens using tiktoken (cl100k_base approximation)."""
        try:
            # Qwen uses a similar tokenizer to OpenAI
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting failed: {e!s}", exc_info=True)
            msg = f"Token counting failed: {e!s}"
            raise AppError(
                msg,
                status_code=status.HTTP_502_BAD_GATEWAY,
            )
