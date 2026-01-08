"""
Google Gemini Provider Implementation for Direct API Integration

Story 1.2: Direct API Integration
Implements native API format for Google Gemini with streaming support.

Provider Details:
- API: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
- Authentication: Bearer token (API key)
- Default Models: gemini-1.5-pro, gemini-1.5-flash, gemini-pro
- Streaming: Server-Sent Events (SSE)
- Rate Limits: 60 requests/minute (free), 1500 requests/minute (paid)
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


class GoogleProvider(BaseProvider):
    """
    Google Gemini provider implementation using native API.

    Features:
    - Direct API integration with Google Generative Language API
    - Streaming support with Server-Sent Events
    - Automatic retry with exponential backoff
    - Rate limit tracking and handling
    """

    # Google API endpoints
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    _DEFAULT_MODEL = "gemini-3-pro"

    # Available models (January 2026)
    _AVAILABLE_MODELS = [
        "gemini-3-pro",
        "gemini-3-flash",
        "gemini-2.5-pro-latest",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    def __init__(self, config: Settings | None = None):
        """
        Initialize the Google Gemini provider.

        Args:
            config: Optional configuration settings.
        """
        super().__init__("google", config)
        self._http_client: httpx.AsyncClient | None = None

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _initialize_client(self, api_key: str) -> httpx.AsyncClient:
        """
        Initialize the Google API HTTP client.

        Args:
            api_key: The API key for authentication.

        Returns:
            The initialized httpx async client.
        """
        return httpx.AsyncClient(
            base_url=self._BASE_URL,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def _get_default_model(self) -> str:
        """
        Get the default model name for Google Gemini.

        Returns:
            The default model name.
        """
        # Check config first, then use class default
        model = getattr(self.config, "google_model", None)
        return model or self._DEFAULT_MODEL

    def _get_api_key(self) -> str | None:
        """
        Get the Google API key from configuration.

        Returns:
            The API key, or None if not configured.
        """
        return getattr(self.config, "google_api_key", None)

    async def _generate_impl(
        self,
        client: httpx.AsyncClient,
        request: PromptRequest,
        model_name: str,
    ) -> PromptResponse:
        """
        Generate text using Google Gemini API.

        Args:
            client: The httpx client instance.
            request: The prompt request.
            model_name: The model name to use.

        Returns:
            The generated response.

        Raises:
            AppError: If generation fails.
        """
        # Build request payload following Google's API format
        contents = []
        if request.system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"System: {request.system_instruction}"}]})

        contents.append({"role": "user", "parts": [{"text": request.prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": self._build_generation_config(request),
        }

        # Add safety settings if needed
        payload["safetySettings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        endpoint = f"models/{model_name}:generateContent"

        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()

            data = response.json()

            # Parse response
            if "candidates" not in data or not data["candidates"]:
                raise AppError(
                    "No candidates in Google response",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                )

            candidate = data["candidates"][0]
            text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")

            # Extract usage metadata
            usage = candidate.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)

            # Get finish reason
            finish_reason = candidate.get("finishReason", "UNKNOWN")

            return PromptResponse(
                text=text,
                model=model_name,
                provider="google",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"[google] HTTP error: {e.response.status_code} - {e.response.text}")

            # Map error status codes
            status_code = e.response.status_code
            if status_code == 401:
                raise AppError(
                    "Invalid Google API key",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                ) from e
            elif status_code == 429:
                raise AppError(
                    "Google rate limit exceeded",
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                ) from e
            else:
                raise AppError(
                    f"Google API error: {e.response.text}",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                ) from e

        except httpx.RequestError as e:
            if is_retryable_error(e):
                raise AppError(
                    f"Google API connection error: {e}",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                ) from e
            raise

    async def _check_health_impl(self, client: httpx.AsyncClient) -> bool:
        """
        Check Google API health with a minimal request.

        Args:
            client: The httpx client instance.

        Returns:
            True if the provider is healthy.
        """
        try:
            # Use a simple models list endpoint for health check
            response = await client.get("models", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"[google] Health check failed: {e}")
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
            Generation config dict for Google API.
        """
        config = {}

        if request.config:
            # Temperature (0.0 - 2.0 for Google)
            if request.config.temperature is not None:
                config["temperature"] = max(0.0, min(2.0, request.config.temperature))

            # Top P (0.0 - 1.0)
            if request.config.top_p is not None:
                config["topP"] = max(0.0, min(1.0, request.config.top_p))

            # Top K (1 - 100 for Google)
            if request.config.top_k is not None:
                config["topK"] = max(1, min(100, request.config.top_k))

            # Max output tokens
            if request.config.max_tokens is not None:
                config["maxOutputTokens"] = request.config.max_tokens

        # Set defaults if not provided
        config.setdefault("temperature", 0.7)
        config.setdefault("topP", 0.95)
        config.setdefault("topK", 40)
        config.setdefault("maxOutputTokens", 4096)

        return config

    # =========================================================================
    # Streaming Support
    # =========================================================================

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream text generation from Google Gemini API.

        Args:
            request: The prompt request.

        Yields:
            StreamChunk: Individual chunks of generated text.

        Raises:
            AppError: If streaming fails.
        """
        client = self._get_client(request.api_key)
        model_name = request.model or self._get_default_model()

        # Build request payload
        contents = []
        if request.system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"System: {request.system_instruction}"}]})
        contents.append({"role": "user", "parts": [{"text": request.prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": self._build_generation_config(request),
        }

        endpoint = f"models/{model_name}:streamGenerateContent"

        try:
            async with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # Parse SSE format (data: {...})
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        try:
                            import json

                            chunk_data = json.loads(data_str)

                            # Extract text from candidates
                            if chunk_data.get("candidates"):
                                candidate = chunk_data["candidates"][0]
                                text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")

                                # Check if this is the final chunk
                                finish_reason = candidate.get("finishReason", None)
                                is_final = finish_reason is not None

                                # Get usage metadata from final chunk
                                usage = candidate.get("usageMetadata", {})
                                token_count = usage.get("candidatesTokenCount", None)

                                yield StreamChunk(
                                    text=text,
                                    is_final=is_final,
                                    finish_reason=finish_reason,
                                    token_count=token_count,
                                )

                        except json.JSONDecodeError:
                            logger.warning(f"[google] Failed to parse streaming chunk: {data_str}")
                            continue

        except httpx.HTTPStatusError as e:
            raise AppError(
                f"Google streaming error: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise AppError(
                f"Google streaming failed: {e}",
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
