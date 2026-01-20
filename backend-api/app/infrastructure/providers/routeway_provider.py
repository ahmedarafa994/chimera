"""Routeway Provider Implementation.

A unified AI gateway that provides access to multiple LLM providers through
a single OpenAI-compatible API. Supports both free and premium models.

Story 1.2: Direct API Integration
Based on: https://docs.routeway.ai/getting-started/quickstart

Key features:
- OpenAI-compatible chat completions API
- Access to multiple model providers (OpenAI, Anthropic, Meta, etc.)
- Free tier with `:free` suffix models
- Premium tier with usage-based pricing
- Bearer token authentication
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Self

import httpx
from fastapi import status

from app.core.errors import AppError
from app.domain.models import PromptRequest, PromptResponse, StreamChunk
from app.infrastructure.providers.base import BaseProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.core.config import Settings

logger = logging.getLogger(__name__)


class RoutewayProvider(BaseProvider):
    """Routeway Provider for multi-model AI gateway.

    Routeway provides a unified API gateway to access multiple LLM providers
    including OpenAI, Anthropic, Meta (Llama), and others through a single
    OpenAI-compatible endpoint.

    Features:
    - OpenAI-compatible API format
    - Access to 100+ models from various providers
    - Free tier models (append :free to model name)
    - Premium tier with usage-based pricing
    - Easy model switching without code changes
    """

    _BASE_URL = "https://api.routeway.ai/v1"
    _DEFAULT_MODEL = "gpt-5.2"  # Default to latest flagship model

    # Popular models available through Routeway (January 2026)
    _AVAILABLE_MODELS = [
        # OpenAI models (via Routeway)
        "gpt-5.2",
        "gpt-5.2-codex",
        "o3-mini",
        "gpt-4.5",
        "gpt-4o",
        "gpt-4o-mini",
        # Anthropic models
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-3-5-sonnet-20241022",
        # Meta Llama models
        "llama-4-scout-17b-16e-instruct",
        "llama-4-maverick-17b-128e-instruct",
        "llama-3.3-70b-instruct",
        # DeepSeek models
        "deepseek-v4",
        "deepseek-chat",
        "deepseek-r1",
        # Google models
        "gemini-3-pro",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        # Free tier models (append :free to any model)
        "gpt-4o-mini:free",
        "llama-3.1-8b-instruct:free",
        "gemini-1.5-flash:free",
    ]

    def __init__(self, config: Settings | None = None) -> None:
        """Initialize Routeway provider."""
        super().__init__("routeway", config)
        self._http_client: httpx.AsyncClient | None = None

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _initialize_client(self, api_key: str) -> httpx.AsyncClient:
        """Initialize HTTP client for Routeway API.

        Uses Bearer token authentication as specified in the documentation.
        """
        return httpx.AsyncClient(
            base_url=self._BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),  # Long timeout for complex models
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def _get_default_model(self) -> str:
        """Get default model from config or fallback."""
        if hasattr(self.config, "routeway_model") and self.config.routeway_model:
            return self.config.routeway_model
        return self._DEFAULT_MODEL

    def _get_api_key(self) -> str | None:
        """Get API key from config."""
        if hasattr(self.config, "ROUTEWAY_API_KEY"):
            return self.config.ROUTEWAY_API_KEY
        return None

    async def _generate_impl(
        self,
        client: httpx.AsyncClient,
        request: PromptRequest,
        model_name: str,
    ) -> PromptResponse:
        """Generate text using Routeway's chat completions API.

        Uses OpenAI-compatible format for requests and responses.
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
            **self._build_generation_config(request),
        }

        logger.debug(f"[routeway] Sending request to model {model_name}")

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse response (OpenAI format)
            if "choices" not in data or not data["choices"]:
                msg = "No choices in Routeway response"
                raise AppError(
                    msg,
                    status_code=status.HTTP_502_BAD_GATEWAY,
                )

            choice = data["choices"][0]
            text = choice.get("message", {}).get("content", "")

            # Extract usage
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            finish_reason = choice.get("finish_reason", "unknown")

            return PromptResponse(
                text=text,
                model_used=model_name,
                provider="routeway",
                usage_metadata={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            await self._handle_http_error(e)
        except httpx.RequestError as e:
            logger.exception(f"[routeway] Request error: {e}")
            msg = f"Routeway request failed: {e!s}"
            raise AppError(
                msg,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    async def _check_health_impl(self, client: httpx.AsyncClient) -> bool:
        """Check health by making a minimal request to the models endpoint.

        Routeway provides a GET /models endpoint that can be used for health checks.
        """
        try:
            response = await client.get("/models")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"[routeway] Health check failed: {e}")
            return False

    # =========================================================================
    # Streaming Support
    # =========================================================================

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Generate streaming response using Routeway's SSE endpoint.

        Uses Server-Sent Events (SSE) format compatible with OpenAI streaming.
        """
        client = self._get_client(request.api_key)
        model_name = request.model or self._get_default_model()

        # Build messages
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            **self._build_generation_config(request),
        }

        logger.debug(f"[routeway] Starting stream for model {model_name}")

        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # Skip empty lines
                    if not line.strip():
                        continue

                    # Skip non-data lines
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix

                    # Handle [DONE] sentinel
                    if data_str.strip() == "[DONE]":
                        yield StreamChunk(text="", is_final=True, finish_reason="stop")
                        break

                    try:
                        chunk_data = json.loads(data_str)

                        if chunk_data.get("choices"):
                            choice = chunk_data["choices"][0]
                            delta = choice.get("delta", {})
                            text = delta.get("content", "")
                            finish_reason = choice.get("finish_reason")
                            is_final = finish_reason is not None

                            # Extract usage if present (some models include it in final chunk)
                            usage = chunk_data.get("usage", {})
                            token_count = usage.get("completion_tokens")

                            yield StreamChunk(
                                text=text,
                                is_final=is_final,
                                finish_reason=finish_reason,
                                token_count=token_count,
                            )

                    except json.JSONDecodeError:
                        logger.warning(f"[routeway] Failed to parse chunk: {data_str}")
                        continue

        except httpx.HTTPStatusError as e:
            await self._handle_http_error(e)
        except Exception as e:
            logger.exception(f"[routeway] Stream error: {e}")
            msg = f"Routeway streaming failed: {e!s}"
            raise AppError(
                msg,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    # =========================================================================
    # Configuration Helpers
    # =========================================================================

    def _build_generation_config(self, request: PromptRequest) -> dict[str, Any]:
        """Build generation configuration for Routeway API.

        Maps PromptRequest parameters to OpenAI-compatible parameters.
        """
        config: dict[str, Any] = {}

        if request.config:
            # Temperature (0.0-2.0 for OpenAI-compatible APIs)
            if request.config.temperature is not None:
                config["temperature"] = max(0.0, min(2.0, request.config.temperature))

            # Top-p
            if request.config.top_p is not None:
                config["top_p"] = max(0.0, min(1.0, request.config.top_p))

            # Max tokens
            if request.config.max_output_tokens is not None:
                config["max_tokens"] = request.config.max_output_tokens

            # Stop sequences
            if request.config.stop_sequences:
                config["stop"] = request.config.stop_sequences

        # Apply defaults if not set
        config.setdefault("temperature", 0.7)
        config.setdefault("max_tokens", 4096)

        return config

    async def _handle_http_error(self, e: httpx.HTTPStatusError) -> None:
        """Handle HTTP errors from Routeway API."""
        status_code = e.response.status_code

        try:
            error_data = e.response.json()
            error_message = error_data.get("error", {}).get("message", str(e))
            error_type = error_data.get("error", {}).get("type", "error")
        except Exception:
            error_message = str(e)
            error_type = "error"

        logger.error(f"[routeway] HTTP {status_code}: {error_message} (type: {error_type})")

        if status_code == 401:
            msg = "Routeway authentication failed. Check your API key."
            raise AppError(
                msg,
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        if status_code == 403:
            msg = "Routeway access forbidden. Your API key may not have access to this resource."
            raise AppError(
                msg,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        if status_code == 429:
            msg = f"Routeway rate limit exceeded: {error_message}"
            raise AppError(
                msg,
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        if status_code == 400:
            msg = f"Routeway bad request: {error_message}"
            raise AppError(
                msg,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if status_code == 404:
            msg = f"Routeway resource not found: {error_message}"
            raise AppError(
                msg,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        if status_code == 422:
            msg = f"Routeway unprocessable entity: {error_message}"
            raise AppError(
                msg,
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        msg = f"Routeway API error ({status_code}): {error_message}"
        raise AppError(
            msg,
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    # =========================================================================
    # Model Discovery
    # =========================================================================

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models from Routeway API.

        Returns a list of model information including pricing and availability.
        """
        client = self._get_client()

        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.exception(f"[routeway] Failed to list models: {e}")
            return []

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
