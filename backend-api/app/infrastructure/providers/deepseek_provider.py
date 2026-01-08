"""
DeepSeek Provider Implementation for Direct API Integration

Story 1.2: Direct API Integration
Implements native API format for DeepSeek with streaming support.

Provider Details:
- API: https://api.deepseek.com/v1/chat/completions
- Authentication: Bearer token (API key)
- Default Models: deepseek-chat, deepseek-coder
- Streaming: Server-Sent Events (SSE)
- Rate Limits: Varies by tier
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


class DeepSeekProvider(BaseProvider):
    """
    DeepSeek provider implementation using native Chat Completions API.

    Features:
    - Direct API integration with DeepSeek
    - Streaming support with Server-Sent Events
    - Automatic retry with exponential backoff
    - Rate limit tracking and handling
    """

    # DeepSeek API endpoints (OpenAI-compatible)
    _BASE_URL = "https://api.deepseek.com/v1"
    _DEFAULT_MODEL = "deepseek-v4"

    # Available models (January 2026)
    _AVAILABLE_MODELS = [
        "deepseek-v4",
        "deepseek-chat",
        "deepseek-reasoner",
    ]

    def __init__(self, config: Settings | None = None):
        """
        Initialize the DeepSeek provider.

        Args:
            config: Optional configuration settings.
        """
        super().__init__("deepseek", config)
        self._http_client: httpx.AsyncClient | None = None

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _initialize_client(self, api_key: str) -> httpx.AsyncClient:
        """
        Initialize the DeepSeek API HTTP client.

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
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def _get_default_model(self) -> str:
        """
        Get the default model name for DeepSeek.

        Returns:
            The default model name.
        """
        model = getattr(self.config, "deepseek_model", None)
        return model or self._DEFAULT_MODEL

    def _get_api_key(self) -> str | None:
        """
        Get the DeepSeek API key from configuration.

        Returns:
            The API key, or None if not configured.
        """
        return getattr(self.config, "deepseek_api_key", None)

    async def _generate_impl(
        self,
        client: httpx.AsyncClient,
        request: PromptRequest,
        model_name: str,
    ) -> PromptResponse:
        """
        Generate text using DeepSeek Chat Completions API.

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

        # Build request payload (OpenAI-compatible format)
        payload = {
            "model": model_name,
            "messages": messages,
            **self._build_generation_config(request),
        }

        try:
            response = await client.post("chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()

            # Parse response
            if "choices" not in data or not data["choices"]:
                raise AppError(
                    "No choices in DeepSeek response",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                )

            choice = data["choices"][0]
            text = choice.get("message", {}).get("content", "")

            # Extract usage metadata
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Get finish reason
            finish_reason = choice.get("finish_reason", "UNKNOWN")

            return PromptResponse(
                text=text,
                model=model_name,
                provider="deepseek",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"[deepseek] HTTP error: {e.response.status_code} - {e.response.text}")

            status_code = e.response.status_code
            if status_code == 401:
                raise AppError(
                    "Invalid DeepSeek API key",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                ) from e
            elif status_code == 429:
                raise AppError(
                    "DeepSeek rate limit exceeded",
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                ) from e
            else:
                raise AppError(
                    f"DeepSeek API error: {e.response.text}",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                ) from e

        except httpx.RequestError as e:
            if is_retryable_error(e):
                raise AppError(
                    f"DeepSeek API connection error: {e}",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                ) from e
            raise

    async def _check_health_impl(self, client: httpx.AsyncClient) -> bool:
        """
        Check DeepSeek API health with a minimal request.

        Args:
            client: The httpx client instance.

        Returns:
            True if the provider is healthy.
        """
        try:
            response = await client.post(
                "chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                },
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"[deepseek] Health check failed: {e}")
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
            Generation config dict for DeepSeek API.
        """
        config = {}

        if request.config:
            if request.config.temperature is not None:
                config["temperature"] = max(0.0, min(2.0, request.config.temperature))

            if request.config.top_p is not None:
                config["top_p"] = max(0.0, min(1.0, request.config.top_p))

            if request.config.max_tokens is not None:
                config["max_tokens"] = request.config.max_tokens

        # Set defaults
        config.setdefault("temperature", 0.7)
        config.setdefault("top_p", 1.0)
        config.setdefault("max_tokens", 4096)

        return config

    # =========================================================================
    # Streaming Support
    # =========================================================================

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream text generation from DeepSeek API.

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
            async with client.stream("POST", "chat/completions", json=payload) as response:
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
                        logger.warning(f"[deepseek] Failed to parse streaming chunk: {data_str}")
                        continue

        except httpx.HTTPStatusError as e:
            raise AppError(
                f"DeepSeek streaming error: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise AppError(
                f"DeepSeek streaming failed: {e}",
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
