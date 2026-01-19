"""
Modern Anthropic Claude client using the official anthropic SDK with native async support.

Implements the LLMProvider interface with:
- Native async via AsyncAnthropic
- Streaming generation
- Token counting via the API
- Proper error handling
"""

import re
import time
from collections.abc import AsyncIterator

from anthropic import APIError, AsyncAnthropic
from fastapi import status

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import logger
from app.domain.interfaces import LLMProvider
from app.domain.models import LLMProviderType, PromptRequest, PromptResponse, StreamChunk


class AnthropicClient(LLMProvider):
    """
    Modern Anthropic Claude client using the official SDK with:
    - Native async support via AsyncAnthropic
    - Streaming generation
    - Token counting
    - Proper error handling with x-api-key header
    """

    def __init__(self, config: Settings | None = None):
        self.config = config or get_settings()
        self.endpoint = self.config.get_provider_endpoint("anthropic")
        api_key = self.config.ANTHROPIC_API_KEY

        if not api_key:
            logger.warning("Anthropic API Key not set for Direct mode")
            self.client = None
        else:
            self.client = AsyncAnthropic(api_key=api_key, base_url=self.endpoint)
            logger.info("AnthropicClient configured in DIRECT mode via SDK")

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        # Anthropic keys are typically prefixed with 'sk-ant-'
        return len(api_key) > 10 and re.match(r"^[a-zA-Z0-9\-_]+$", api_key) is not None

    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        prompt = re.sub(r"<script.*?</script>", "", prompt, flags=re.IGNORECASE | re.DOTALL)
        prompt = re.sub(r"javascript:", "", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"on\w+\s*=", "", prompt, flags=re.IGNORECASE)

        max_prompt_length = getattr(self.config, "JAILBREAK_MAX_PROMPT_LENGTH", 10000)
        if len(prompt) > max_prompt_length:
            logger.warning(f"Prompt truncated from {len(prompt)} to {max_prompt_length} chars")
            prompt = prompt[:max_prompt_length]

        return prompt.strip()

    def _get_client(self, request: PromptRequest) -> AsyncAnthropic:
        """Get client, using override API key if provided."""
        if request.api_key:
            if not self._validate_api_key(request.api_key):
                raise AppError("Invalid API key format", status_code=status.HTTP_400_BAD_REQUEST)
            return AsyncAnthropic(api_key=request.api_key, base_url=self.endpoint)

        if not self.client:
            raise AppError(
                "Anthropic client not initialized. Check API key.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return self.client

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Generate a response from Anthropic Claude."""
        if not request.prompt:
            raise AppError("Prompt is required", status_code=status.HTTP_400_BAD_REQUEST)

        sanitized_prompt = self._sanitize_prompt(request.prompt)
        client = self._get_client(request)
        model_name = request.model or getattr(
            self.config, "ANTHROPIC_MODEL", "claude-3-opus-20240229"
        )
        start_time = time.time()

        try:
            # Build system prompt if provided
            system_content = None
            if request.system_instruction:
                system_content = self._sanitize_prompt(request.system_instruction)

            logger.info(f"Generating content with model: {model_name} (Direct)")

            # Anthropic uses a different message format
            response = await client.messages.create(
                model=model_name,
                max_tokens=request.config.max_output_tokens if request.config else 2048,
                system=system_content or "",
                messages=[{"role": "user", "content": sanitized_prompt}],
                temperature=request.config.temperature if request.config else 0.7,
                top_p=request.config.top_p if request.config else 0.95,
                stop_sequences=request.config.stop_sequences if request.config else None,
            )

            latency = (time.time() - start_time) * 1000

            # Extract content from response
            content = ""
            if response.content and len(response.content) > 0:
                content = response.content[0].text

            finish_reason = response.stop_reason or "unknown"

            usage_metadata = {}
            if response.usage:
                usage_metadata = {
                    "prompt_token_count": response.usage.input_tokens,
                    "candidates_token_count": response.usage.output_tokens,
                    "total_token_count": response.usage.input_tokens + response.usage.output_tokens,
                }

            return PromptResponse(
                text=content,
                model_used=model_name,
                provider=LLMProviderType.ANTHROPIC.value,
                usage_metadata=usage_metadata,
                finish_reason=finish_reason.upper() if finish_reason else "UNKNOWN",
                latency_ms=latency,
            )

        except APIError as e:
            logger.error(f"Anthropic API error: {e.message}")
            raise AppError(
                f"Anthropic API error: {e.message}",
                status_code=self._map_error_code(e.status_code),
            )
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e!s}", exc_info=True)
            raise AppError(
                f"Anthropic generation failed: {e!s}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    def _map_error_code(self, code: int | None) -> int:
        """Map Anthropic API error codes to HTTP status codes."""
        if code is None:
            return status.HTTP_502_BAD_GATEWAY
        mapping = {
            400: status.HTTP_400_BAD_REQUEST,
            401: status.HTTP_401_UNAUTHORIZED,
            403: status.HTTP_403_FORBIDDEN,
            404: status.HTTP_404_NOT_FOUND,
            429: status.HTTP_429_TOO_MANY_REQUESTS,
            500: status.HTTP_502_BAD_GATEWAY,
            529: status.HTTP_503_SERVICE_UNAVAILABLE,  # Anthropic overloaded
        }
        return mapping.get(code, status.HTTP_502_BAD_GATEWAY)

    async def check_health(self) -> bool:
        """
        Check if the Anthropic API is healthy.

        Uses a lightweight token counting request instead of a full generation.
        This should complete in <100ms as required by the PRD.
        The SDK handles connection pooling internally via httpx.
        """
        try:
            if not self.client:
                return False

            # Use token counting as a lightweight health check
            # This is faster and cheaper than a full message request
            await self.client.messages.count_tokens(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e!s}")
            return False

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Anthropic Claude."""
        if not request.prompt:
            raise AppError("Prompt is required", status_code=status.HTTP_400_BAD_REQUEST)

        sanitized_prompt = self._sanitize_prompt(request.prompt)
        client = self._get_client(request)
        model_name = request.model or getattr(
            self.config, "ANTHROPIC_MODEL", "claude-3-opus-20240229"
        )

        try:
            system_content = None
            if request.system_instruction:
                system_content = self._sanitize_prompt(request.system_instruction)

            logger.info(f"Streaming content with model: {model_name}")

            async with client.messages.stream(
                model=model_name,
                max_tokens=request.config.max_output_tokens if request.config else 2048,
                system=system_content or "",
                messages=[{"role": "user", "content": sanitized_prompt}],
                temperature=request.config.temperature if request.config else 0.7,
                top_p=request.config.top_p if request.config else 0.95,
                stop_sequences=request.config.stop_sequences if request.config else None,
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(text=text, is_final=False, finish_reason=None)

            yield StreamChunk(text="", is_final=True, finish_reason="STOP")

        except APIError as e:
            logger.error(f"Anthropic streaming error: {e.message}")
            raise AppError(
                f"Streaming failed: {e.message}",
                status_code=self._map_error_code(e.status_code),
            )
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e!s}", exc_info=True)
            raise AppError(
                f"Streaming failed: {e!s}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens using Anthropic's token counting API.

        Note: Anthropic provides token counting via the messages API.
        """
        try:
            if not self.client:
                raise AppError(
                    "Anthropic client not initialized",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            model_name = model or getattr(self.config, "ANTHROPIC_MODEL", "claude-3-opus-20240229")

            # Use the count_tokens endpoint
            result = await self.client.messages.count_tokens(
                model=model_name,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens

        except Exception as e:
            logger.error(f"Token counting failed: {e!s}", exc_info=True)
            raise AppError(
                f"Token counting failed: {e!s}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )
