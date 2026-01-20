import asyncio
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import status
from google import genai
from google.genai import errors, types

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import logger
from app.domain.interfaces import LLMProvider
from app.domain.models import LLMProviderType, PromptRequest, PromptResponse, StreamChunk


class GeminiClient(LLMProvider):
    """Modern Gemini client using python-genai SDK with:
    - Native async support via client.aio
    - Streaming generation
    - Token counting
    - Proper error handling with errors.APIError.
    """

    def __init__(self, config: Settings = None) -> None:
        self.config = config or get_settings()
        self._client: genai.Client | None = None

        self.endpoint = self.config.get_provider_endpoint("google")
        api_key = self.config.get_effective_api_key()
        if not api_key:
            logger.warning("Google API Key not set for Direct mode")
        else:
            self._client = genai.Client(api_key=api_key)
        logger.info("GeminiClient configured in DIRECT mode via SDK")

    def _get_safety_settings(self) -> list:
        """Get safety settings configured to allow all content for research/testing.
        BLOCK_NONE allows all content including hate speech, harassment, and complex content.
        """
        from google.genai import types

        return [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]

    @property
    def client(self) -> genai.Client:
        """Get the sync client, raising if not initialized."""
        if not self._client:
            msg = "Gemini client not initialized - API key may be missing"
            raise AppError(
                msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return self._client

    async def close(self) -> None:
        """Close the client and cleanup resources. Call on application shutdown."""
        if self._client is not None:
            with suppress(Exception):
                await asyncio.wait_for(self._client.aclose(), timeout=5.0)
            self._client = None

    def _ensure_client(self) -> genai.Client:
        """Ensure a client instance exists and return it.
        Reuses the existing client to avoid repeated cleanup issues with aiohttp.
        """
        if self._client is None:
            api_key = self.config.get_effective_api_key()
            if not api_key:
                msg = "Gemini client not initialized - API key missing"
                raise AppError(
                    msg,
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    @asynccontextmanager
    async def _async_client(self):
        """Context manager for async client operations.
        Uses client.aio pattern from SDK for native async support.

        Creates a fresh client for each context to avoid event loop binding issues
        when called from different threads/loops (e.g., AutoDAN executor threads).
        """
        api_key = self.config.get_effective_api_key()
        if not api_key:
            msg = "Gemini client not initialized - API key missing"
            raise AppError(
                msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Create a fresh client for this async context to ensure proper event loop binding
        local_client = genai.Client(api_key=api_key)
        try:
            yield local_client.aio
        finally:
            # Suppress cleanup errors - the client will be garbage collected
            # This avoids "Task was destroyed" warnings on shutdown
            pass

    def _build_generation_config(self, request: PromptRequest) -> types.GenerateContentConfig:
        """Build GenerateContentConfig from request parameters with safety disabled."""
        config_params = {
            "temperature": request.config.temperature if request.config else 0.7,
            "top_p": request.config.top_p if request.config else 0.95,
            "max_output_tokens": request.config.max_output_tokens if request.config else 2048,
            "safety_settings": self._get_safety_settings(),
        }

        if request.config and request.config.stop_sequences:
            config_params["stop_sequences"] = request.config.stop_sequences

        return types.GenerateContentConfig(**config_params)

    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not prompt:
            msg = "Prompt cannot be empty"
            raise ValueError(msg)

        # Remove potential code injection patterns
        prompt = re.sub(r"<script.*?</script>", "", prompt, flags=re.IGNORECASE | re.DOTALL)
        prompt = re.sub(r"javascript:", "", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"on\w+\s*=", "", prompt, flags=re.IGNORECASE)

        # Limit prompt length to prevent DoS (using config limit if available)
        max_prompt_length = getattr(self.config, "JAILBREAK_MAX_PROMPT_LENGTH", 10000)
        if len(prompt) > max_prompt_length:
            logger.warning(f"Prompt truncated from {len(prompt)} to {max_prompt_length} characters")
            prompt = prompt[:max_prompt_length]

        return prompt.strip()

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and prevent injection."""
        if not api_key:
            return False
        # Basic validation - alphanumeric with common characters
        if not re.match(r"^[a-zA-Z0-9\-_\.]+$", api_key):
            logger.warning("Invalid API key format detected")
            return False
        return True

    async def generate(self, request: PromptRequest) -> PromptResponse:
        # Input validation and sanitization
        if not request.prompt:
            msg = "Prompt is required"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        sanitized_prompt = self._sanitize_prompt(request.prompt)
        sanitized_system_instruction = None
        if request.system_instruction:
            sanitized_system_instruction = self._sanitize_prompt(request.system_instruction)

        # API key validation
        if request.api_key and not self._validate_api_key(request.api_key):
            msg = "Invalid API key format"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        model_name = request.model or self.config.GOOGLE_MODEL
        start_time = time.time()

        # Direct Gemini API call using native async SDK
        try:
            generation_config = self._build_generation_config(request)

            # Combine system instruction with prompt if provided
            if sanitized_system_instruction:
                contents = f"System: {sanitized_system_instruction}\n\nUser: {sanitized_prompt}"
            else:
                contents = sanitized_prompt

            logger.info(f"Generating content with model: {model_name} (Direct/Async)")

            # Use native async client for better performance
            async with self._async_client() as aclient:
                response = await aclient.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )

            latency = (time.time() - start_time) * 1000

            return self._build_response(response, model_name, latency)

        except errors.APIError as e:
            logger.error(f"Gemini API error: code={e.code}, message={e.message}")
            msg = f"Gemini API error: {e.message}"
            raise AppError(
                msg,
                status_code=self._map_error_code(e.code),
            )
        except Exception as e:
            logger.error(f"Gemini generation failed: {e!s}", exc_info=True)
            msg = f"Gemini generation failed: {e!s}"
            raise AppError(
                msg,
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    def _build_response(self, response, model_name: str, latency: float) -> PromptResponse:
        """Build PromptResponse from SDK response."""
        if not response.candidates:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(
                    f"Content blocked by safety filters: {response.prompt_feedback.block_reason}",
                )
                msg = f"Content blocked by safety filters: {response.prompt_feedback.block_reason}"
                raise AppError(
                    msg,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            return PromptResponse(
                text="",
                model_used=model_name,
                provider=LLMProviderType.GOOGLE.value,
                finish_reason="BLOCKED_OR_EMPTY",
                latency_ms=latency,
            )

        usage_metadata = {}
        if response.usage_metadata:
            try:
                if hasattr(response.usage_metadata, "to_dict"):
                    usage_metadata = response.usage_metadata.to_dict()
                else:
                    usage_metadata = {
                        "prompt_token_count": getattr(
                            response.usage_metadata,
                            "prompt_token_count",
                            0,
                        ),
                        "candidates_token_count": getattr(
                            response.usage_metadata,
                            "candidates_token_count",
                            0,
                        ),
                        "total_token_count": getattr(
                            response.usage_metadata,
                            "total_token_count",
                            0,
                        ),
                    }
            except Exception as e:
                logger.warning(f"Failed to parse usage_metadata: {e}")

        return PromptResponse(
            text=response.text or "",  # Handle None response text
            model_used=model_name,
            provider=LLMProviderType.GOOGLE.value,
            usage_metadata=usage_metadata,
            finish_reason=(
                response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            ),
            latency_ms=latency,
        )

    def _map_error_code(self, code: int) -> int:
        """Map Gemini API error codes to HTTP status codes."""
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

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation using generate_content_stream.

        Yields:
            StreamChunk: Individual chunks of generated text.

        """
        # Input validation
        if not request.prompt:
            msg = "Prompt is required"
            raise AppError(msg, status_code=status.HTTP_400_BAD_REQUEST)

        sanitized_prompt = self._sanitize_prompt(request.prompt)
        sanitized_system_instruction = None
        if request.system_instruction:
            sanitized_system_instruction = self._sanitize_prompt(request.system_instruction)

        model_name = request.model or self.config.GOOGLE_MODEL
        generation_config = self._build_generation_config(request)

        # Combine system instruction with prompt if provided
        if sanitized_system_instruction:
            contents = f"System: {sanitized_system_instruction}\n\nUser: {sanitized_prompt}"
        else:
            contents = sanitized_prompt

        logger.info(f"Streaming content with model: {model_name}")

        try:
            async with self._async_client() as aclient:
                stream = await aclient.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )

                async for chunk in stream:
                    chunk_text = chunk.text if chunk.text else ""
                    finish_reason = None

                    if chunk.candidates and chunk.candidates[0].finish_reason:
                        finish_reason = chunk.candidates[0].finish_reason.name

                    yield StreamChunk(text=chunk_text, is_final=False, finish_reason=finish_reason)

            # Send final chunk to signal completion
            yield StreamChunk(text="", is_final=True, finish_reason="STOP")

        except errors.APIError as e:
            logger.error(f"Gemini streaming error: code={e.code}, message={e.message}")
            msg = f"Streaming failed: {e.message}"
            raise AppError(
                msg,
                status_code=self._map_error_code(e.code),
            )
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e!s}", exc_info=True)
            msg = f"Streaming failed: {e!s}"
            raise AppError(msg, status_code=status.HTTP_502_BAD_GATEWAY)

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: The text to count tokens for.
            model: Optional model name for model-specific tokenization.

        Returns:
            int: The number of tokens in the text.

        """
        model_name = model or self.config.GOOGLE_MODEL

        try:
            async with self._async_client() as aclient:
                result = await aclient.models.count_tokens(model=model_name, contents=text)
                return result.total_tokens

        except errors.APIError as e:
            logger.error(f"Token counting error: code={e.code}, message={e.message}")
            msg = f"Token counting failed: {e.message}"
            raise AppError(
                msg,
                status_code=self._map_error_code(e.code),
            )
        except Exception as e:
            logger.error(f"Token counting failed: {e!s}", exc_info=True)
            msg = f"Token counting failed: {e!s}"
            raise AppError(msg, status_code=status.HTTP_502_BAD_GATEWAY)

    async def check_health(self) -> bool:
        try:
            # Direct API health check using async client
            async with self._async_client() as aclient:
                await aclient.models.list()
                # If we can list models, the API is healthy
                return True
        except Exception as e:
            logger.error(f"Gemini health check failed: {e!s}")
            return False
