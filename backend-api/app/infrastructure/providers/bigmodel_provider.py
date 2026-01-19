"""
BigModel (ZhiPu AI) Provider Implementation for Direct API Integration

Story 1.2: Direct API Integration
Implements native API format for BigModel (智谱AI) with streaming support.

Provider Details:
- API: https://open.bigmodel.cn/api/paas/v4/chat/completions
- Authentication: Bearer token (API key)
- Default Models: glm-4.7 (flagship), glm-4.6v (vision), glm-4-plus
- Streaming: Server-Sent Events (SSE)
- Features: Thinking mode (reasoning_content), tool calls, web search,  multimodal

Documentation: https://docs.bigmodel.cn/api-reference/模型-api/对话补全
"""

from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import status

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import logger
from app.domain.models import PromptRequest, PromptResponse, StreamChunk
from app.infrastructure.providers.base import BaseProvider, is_retryable_error


class BigModelProvider(BaseProvider):
    """
    BigModel (ZhiPu AI) provider implementation using native Chat Completions API.

    Features:
    - Direct API integration with BigModel
    - Streaming support with Server-Sent Events
    - Thinking mode (reasoning chain) support for GLM-4.5+
    - Automatic retry with exponential backoff
    - Rate limit tracking and handling
    """

    # BigModel API endpoints
    _BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    _DEFAULT_MODEL = "glm-4.7"

    # Available models
    _AVAILABLE_MODELS = [
        # GLM-4.7 (Latest flagship)
        "glm-4.7",
        # GLM-4.6 Series
        "glm-4.6",
        # GLM-4.5 Series
        "glm-4.5",
        "glm-4.5-air",
        "glm-4.5-x",
        "glm-4.5-airx",
        "glm-4.5-flash",
        # GLM-4 Series
        "glm-4-plus",
        "glm-4-air-250414",
        "glm-4-airx",
        "glm-4-flashx",
        "glm-4-flashx-250414",
    ]

    def __init__(self, config: Settings | None = None):
        """
        Initialize the BigModel provider.

        Args:
            config: Optional configuration settings.
        """
        super().__init__("bigmodel", config)
        self._http_client: httpx.AsyncClient | None = None

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _initialize_client(self, api_key: str) -> httpx.AsyncClient:
        """
        Initialize the BigModel API HTTP client.

        Args:
            api_key: The API key for authentication.

        Returns:
            The initialized httpx async client.
        """
        return httpx.AsyncClient(
            base_url=self._BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def _get_default_model(self) -> str:
        """
        Get the default model name for BigModel.

        Returns:
            The default model name.
        """
        model = getattr(self.config, "bigmodel_model", None)
        return model or self._DEFAULT_MODEL

    def _get_api_key(self) -> str | None:
        """
        Get the BigModel API key from configuration.

        Returns:
            The API key, or None if not configured.
        """
        return getattr(self.config, "BIGMODEL_API_KEY", None)

    async def _generate_impl(
        self,
        client: httpx.AsyncClient,
        request: PromptRequest,
        model_name: str,
    ) -> PromptResponse:
        """
        Generate text using BigModel Chat Completions API.

        Args:
            client: The httpx client instance.
            request: The prompt request.
            model_name: The model name to use.

        Returns:
            The generated response.

        Raises:
            AppError: If generation fails.
        """
        # Build messages array
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.prompt})

        # Build request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            **self._build_generation_config(request),
        }

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()

            # Parse response
            if "choices" not in data or not data["choices"]:
                raise AppError(
                    "No choices in BigModel response",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                )

            choice = data["choices"][0]
            message = choice.get("message", {})
            text = message.get("content", "")

            # Handle reasoning content (thinking mode for GLM-4.5+)
            reasoning_content = message.get("reasoning_content", "")
            if reasoning_content:
                logger.debug(f"[bigmodel] Reasoning content: {len(reasoning_content)} chars")

            # Extract usage metadata
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Get finish reason
            finish_reason = choice.get("finish_reason", "UNKNOWN")

            return PromptResponse(
                text=text,
                model_used=model_name,
                provider="bigmodel",
                usage_metadata={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"[bigmodel] HTTP error: {e.response.status_code} - {e.response.text}")

            status_code = e.response.status_code
            if status_code == 401:
                raise AppError(
                    "Invalid BigModel API key",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                ) from e
            elif status_code == 429:
                raise AppError(
                    "BigModel rate limit exceeded",
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                ) from e
            else:
                raise AppError(
                    f"BigModel API error: {e.response.text}",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                ) from e

        except httpx.RequestError as e:
            if is_retryable_error(e):
                raise AppError(
                    f"BigModel API connection error: {e}",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                ) from e
            raise

    async def _check_health_impl(self, client: httpx.AsyncClient) -> bool:
        """
        Check BigModel API health with a minimal request.

        Args:
            client: The httpx client instance.

        Returns:
            True if the provider is healthy.
        """
        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": "glm-4-flash",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                },
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"[bigmodel] Health check failed: {e}")
            return False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_generation_config(self, request: PromptRequest) -> dict[str, Any]:
        """
        Build generation config from request parameters.

        Args:
            request: The prompt request.

        Returns:
            Generation config dict for BigModel API.
        """
        config = {}

        if request.config:
            # Temperature: 0.0-1.0 for BigModel
            if request.config.temperature is not None:
                config["temperature"] = max(0.0, min(1.0, request.config.temperature))

            # Top P: 0.01-1.0 for BigModel
            if request.config.top_p is not None:
                config["top_p"] = max(0.01, min(1.0, request.config.top_p))

            # Max output tokens
            if request.config.max_output_tokens is not None:
                config["max_tokens"] = request.config.max_output_tokens

        # Set defaults based on BigModel documentation
        config.setdefault("temperature", 1.0)  # Default for GLM-4.7
        config.setdefault("top_p", 0.95)
        config.setdefault("max_tokens", 4096)

        return config

    # =========================================================================
    # Streaming Support
    # =========================================================================

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream text generation from BigModel API.

        Args:
            request: The prompt request.

        Yields:
            StreamChunk: Individual chunks of generated text.

        Raises:
            AppError: If streaming fails.
        """
        client = self._get_client(request.api_key)
        model_name = request.model or self._get_default_model()

        # Build messages
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.prompt})

        # Build request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            **self._build_generation_config(request),
        }

        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix

                    # Check for [DONE] sentinel
                    if data_str.strip() == "[DONE]":
                        yield StreamChunk(text="", is_final=True, finish_reason="stop")
                        break

                    try:
                        import json

                        chunk_data = json.loads(data_str)

                        # Extract delta from choices
                        if chunk_data.get("choices"):
                            choice = chunk_data["choices"][0]
                            delta = choice.get("delta", {})
                            text = delta.get("content", "")

                            # Also check for reasoning content in stream
                            reasoning_text = delta.get("reasoning_content", "")
                            if reasoning_text:
                                logger.debug(
                                    f"[bigmodel] Stream reasoning chunk: {len(reasoning_text)} chars"
                                )

                            # Check finish reason
                            finish_reason = choice.get("finish_reason")
                            is_final = finish_reason is not None

                            # Get usage from final chunk
                            usage = chunk_data.get("usage", {})
                            token_count = usage.get("completion_tokens", None)

                            yield StreamChunk(
                                text=text,
                                is_final=is_final,
                                finish_reason=finish_reason,
                                token_count=token_count,
                            )

                    except json.JSONDecodeError:
                        logger.warning(f"[bigmodel] Failed to parse streaming chunk: {data_str}")
                        continue

        except httpx.HTTPStatusError as e:
            raise AppError(
                f"BigModel streaming error: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise AppError(
                f"BigModel streaming failed: {e}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from e

    # =========================================================================
    # Thinking Mode Support (GLM-4.5+)
    # =========================================================================

    async def generate_with_thinking(
        self,
        request: PromptRequest,
        thinking_enabled: bool = True,
        clear_thinking: bool = True,
    ) -> PromptResponse:
        """
        Generate text with thinking mode enabled.

        Only GLM-4.5+ models support this feature. The model will provide
        reasoning_content in addition to the regular content.

        Args:
            request: The prompt request.
            thinking_enabled: Whether to enable thinking mode.
            clear_thinking: Whether to clear thinking from previous turns.

        Returns:
            The generated response with reasoning content.
        """
        client = self._get_client(request.api_key)
        model_name = request.model or self._get_default_model()

        # Build messages
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.prompt})

        # Build request payload with thinking configuration
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "thinking": {
                "type": "enabled" if thinking_enabled else "disabled",
                "clear_thinking": clear_thinking,
            },
            **self._build_generation_config(request),
        }

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()

            if "choices" not in data or not data["choices"]:
                raise AppError(
                    "No choices in BigModel response",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                )

            choice = data["choices"][0]
            message = choice.get("message", {})
            text = message.get("content", "")
            reasoning_content = message.get("reasoning_content", "")

            # Log reasoning for debugging
            if reasoning_content:
                logger.info(f"[bigmodel] Reasoning content length: {len(reasoning_content)} chars")

            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            finish_reason = choice.get("finish_reason", "UNKNOWN")

            # Include reasoning in metadata
            usage_metadata = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

            # Store reasoning content in metadata if available
            if reasoning_content:
                usage_metadata["reasoning_content"] = reasoning_content

            response_obj = PromptResponse(
                text=text,
                model_used=model_name,
                provider="bigmodel",
                usage_metadata=usage_metadata,
                finish_reason=finish_reason,
            )

            return response_obj

        except httpx.HTTPStatusError as e:
            raise AppError(
                f"BigModel API error: {e.response.text}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from e

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
