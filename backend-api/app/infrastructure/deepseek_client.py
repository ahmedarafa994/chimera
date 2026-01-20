import time
from collections.abc import AsyncIterator

import tiktoken
from fastapi import status
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import logger
from app.domain.interfaces import LLMProvider
from app.domain.models import LLMProviderType, PromptRequest, PromptResponse, StreamChunk


class DeepSeekClient(LLMProvider):
    def __init__(self, config: Settings = None) -> None:
        self.config = config or get_settings()
        self.endpoint = self.config.get_provider_endpoint("deepseek")
        api_key = self.config.DEEPSEEK_API_KEY
        if not api_key:
            logger.warning("DeepSeek API Key not set for Direct mode")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=api_key, base_url=self.endpoint)
            logger.info("DeepSeekClient configured in DIRECT mode via OpenAI SDK")

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        # Basic validation - assuming standard format or at least non-empty
        return len(api_key) > 5

    async def generate(self, request: PromptRequest) -> PromptResponse:
        # Input validation
        if not request.prompt:
            msg = "Prompt is required"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        # API key override if provided
        client = self.client
        if request.api_key:
            if not self._validate_api_key(request.api_key):
                msg = "Invalid API key format"
                raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)
            client = AsyncOpenAI(api_key=request.api_key, base_url=self.endpoint)

        model_name = request.model or self.config.DEEPSEEK_MODEL
        start_time = time.time()

        # Direct Mode
        if not client:
            msg = "DeepSeek client not initialized. Check API key."
            raise AppError(
                msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        try:
            messages = []
            if request.system_instruction:
                messages.append({"role": "system", "content": request.system_instruction})
            messages.append({"role": "user", "content": request.prompt})

            logger.info(f"Generating content with model: {model_name} (Direct)")
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

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            usage_metadata = {}
            if response.usage:
                usage_metadata = {
                    "prompt_token_count": response.usage.prompt_tokens,
                    "candidates_token_count": response.usage.completion_tokens,
                    "total_token_count": response.usage.total_tokens,
                }

            # Map DeepSeek provider type if it exists in LLMProviderType, otherwise use OPENAI or a generic one
            # Assuming LLMProviderType needs update or we use a string compatible value
            provider_value = "deepseek"
            try:
                provider_value = LLMProviderType.DEEPSEEK.value
            except AttributeError:
                # Fallback if DEEPSEEK not in enum yet
                provider_value = "deepseek"

            return PromptResponse(
                text=content,
                model_used=model_name,
                provider=provider_value,
                usage_metadata=usage_metadata,
                finish_reason=finish_reason,
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e!s}", exc_info=True)
            msg = f"DeepSeek generation failed: {e!s}"
            raise AppError(
                msg,
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    async def check_health(self) -> bool:
        try:
            # Direct API health check
            if not self.client:
                return False

            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {e!s}")
            return False

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation using DeepSeek's OpenAI-compatible streaming API.

        Yields:
            StreamChunk: Individual chunks of generated text.

        """
        # Input validation
        if not request.prompt:
            msg = "Prompt is required"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        client = self.client
        if request.api_key:
            if not self._validate_api_key(request.api_key):
                msg = "Invalid API key format"
                raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)
            client = AsyncOpenAI(api_key=request.api_key, base_url=self.endpoint)

        if not client:
            msg = "DeepSeek client not initialized. Check API key."
            raise AppError(
                msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        model_name = request.model or self.config.DEEPSEEK_MODEL

        try:
            messages = []
            if request.system_instruction:
                messages.append({"role": "system", "content": request.system_instruction})
            messages.append({"role": "user", "content": request.prompt})

            logger.info(f"Streaming content with model: {model_name}")

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

            # Send final chunk
            yield StreamChunk(text="", is_final=True, finish_reason="STOP")

        except Exception as e:
            logger.error(f"DeepSeek streaming failed: {e!s}", exc_info=True)
            msg = f"Streaming failed: {e!s}"
            raise AppError(msg, status_code=status.HTTP_502_BAD_GATEWAY)

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens using tiktoken for DeepSeek models.

        DeepSeek uses a tokenizer similar to OpenAI's, so we use tiktoken
        with cl100k_base encoding as a reasonable approximation.

        Args:
            text: The text to count tokens for.
            model: Optional model name (not used, but kept for interface compatibility).

        Returns:
            int: The number of tokens in the text.

        """
        try:
            # DeepSeek uses a tokenizer similar to OpenAI's cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)

        except Exception as e:
            logger.error(f"Token counting failed: {e!s}", exc_info=True)
            msg = f"Token counting failed: {e!s}"
            raise AppError(msg, status_code=status.HTTP_502_BAD_GATEWAY)
