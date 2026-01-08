"""
Anthropic Claude Provider Implementation for Direct API Integration

Story 1.2: Direct API Integration
Implements native API format for Anthropic Messages API with streaming support.

Provider Details:
- API: https://api.anthropic.com/v1/messages
- Authentication: x-api-key header (not Bearer token)
- Default Models: claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-3-sonnet-20240229
- Streaming: Server-Sent Events (SSE)
- Rate Limits: 5 requests/second (tier-dependent)
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


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude provider implementation using native Messages API.

    Features:
    - Direct API integration with Anthropic
    - Streaming support with Server-Sent Events
    - Automatic retry with exponential backoff
    - Rate limit tracking and handling
    """

    # Anthropic API endpoints
    _BASE_URL = "https://api.anthropic.com/v1"
    _DEFAULT_MODEL = "claude-opus-4.5"

    # Available models (January 2026)
    _AVAILABLE_MODELS = [
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ]

    # Anthropic API version
    _API_VERSION = "2023-06-01"

    def __init__(self, config: Settings | None = None):
        """
        Initialize the Anthropic provider.

        Args:
            config: Optional configuration settings.
        """
        super().__init__("anthropic", config)
        self._http_client: httpx.AsyncClient | None = None

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _initialize_client(self, api_key: str) -> httpx.AsyncClient:
        """
        Initialize the Anthropic API HTTP client.

        Args:
            api_key: The API key for authentication.

        Returns:
            The initialized httpx async client.
        """
        return httpx.AsyncClient(
            base_url=self._BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self._API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def _get_default_model(self) -> str:
        """
        Get the default model name for Anthropic.

        Returns:
            The default model name.
        """
        model = getattr(self.config, "anthropic_model", None)
        return model or self._DEFAULT_MODEL

    def _get_api_key(self) -> str | None:
        """
        Get the Anthropic API key from configuration.

        Returns:
            The API key, or None if not configured.
        """
        return getattr(self.config, "anthropic_api_key", None)

    async def _generate_impl(
        self,
        client: httpx.AsyncClient,
        request: PromptRequest,
        model_name: str,
    ) -> PromptResponse:
        """
        Generate text using Anthropic Messages API.

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
        messages = [{"role": "user", "content": request.prompt}]

        # Build request payload
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 4096,  # Anthropic requires max_tokens
            **self._build_generation_config(request),
        }

        # Add system instruction if provided
        if request.system_instruction:
            payload["system"] = request.system_instruction

        try:
            response = await client.post("messages", json=payload)
            response.raise_for_status()

            data = response.json()

            # Parse response
            if "content" not in data or not data["content"]:
                raise AppError(
                    "No content in Anthropic response",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                )

            # Extract text from content blocks
            text_blocks = [block.get("text", "") for block in data["content"] if block.get("type") == "text"]
            text = "".join(text_blocks)

            # Extract usage metadata
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens

            # Get stop reason
            finish_reason = data.get("stop_reason", "UNKNOWN")

            return PromptResponse(
                text=text,
                model=model_name,
                provider="anthropic",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"[anthropic] HTTP error: {e.response.status_code} - {e.response.text}")

            status_code = e.response.status_code
            if status_code == 401:
                raise AppError(
                    "Invalid Anthropic API key",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                ) from e
            elif status_code == 429:
                raise AppError(
                    "Anthropic rate limit exceeded",
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                ) from e
            else:
                raise AppError(
                    f"Anthropic API error: {e.response.text}",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                ) from e

        except httpx.RequestError as e:
            if is_retryable_error(e):
                raise AppError(
                    f"Anthropic API connection error: {e}",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                ) from e
            raise

    async def _check_health_impl(self, client: httpx.AsyncClient) -> bool:
        """
        Check Anthropic API health with a minimal request.

        Args:
            client: The httpx client instance.

        Returns:
            True if the provider is healthy.
        """
        try:
            # Use a minimal message request for health check
            response = await client.post(
                "messages",
                json={
                    "model": "claude-3-haiku-20240307",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                },
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"[anthropic] Health check failed: {e}")
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
            Generation config dict for Anthropic API.
        """
        config = {}

        if request.config:
            if request.config.temperature is not None:
                config["temperature"] = max(0.0, min(1.0, request.config.temperature))

            if request.config.top_p is not None:
                config["top_p"] = max(0.0, min(1.0, request.config.top_p))

            if request.config.top_k is not None:
                config["top_k"] = max(0, min(100, request.config.top_k))

            if request.config.max_tokens is not None:
                config["max_tokens"] = request.config.max_tokens

        # Set defaults
        config.setdefault("temperature", 0.7)
        config.setdefault("top_p", 1.0)
        config.setdefault("top_k", 40)
        config.setdefault("max_tokens", 4096)

        return config

    # =========================================================================
    # Streaming Support
    # =========================================================================

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream text generation from Anthropic API.

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
        messages = [{"role": "user", "content": request.prompt}]

        # Build request payload
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
            **self._build_generation_config(request),
        }

        if request.system_instruction:
            payload["system"] = request.system_instruction

        try:
            async with client.stream("POST", "messages", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix

                    # Check for event_type
                    if data_str.startswith("event: "):
                        continue

                    try:
                        import json

                        chunk_data = json.loads(data_str)

                        # Extract text from content blocks
                        if "delta" in chunk_data:
                            delta = chunk_data["delta"]
                            text = delta.get("text", "")

                            # Check for message_stop event
                            event_type = chunk_data.get("type", "")
                            is_final = event_type == "message_stop"

                            # Get usage from message_delta event
                            usage = chunk_data.get("usage", {})
                            token_count = usage.get("output_tokens", None)

                            # Get stop reason from message stop
                            finish_reason = None
                            if is_final:
                                finish_reason = chunk_data.get("stop_reason", "UNKNOWN")

                            yield StreamChunk(
                                text=text,
                                is_final=is_final,
                                finish_reason=finish_reason,
                                token_count=token_count,
                            )

                    except json.JSONDecodeError:
                        logger.warning(f"[anthropic] Failed to parse streaming chunk: {data_str}")
                        continue

        except httpx.HTTPStatusError as e:
            raise AppError(
                f"Anthropic streaming error: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise AppError(
                f"Anthropic streaming failed: {e}",
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
