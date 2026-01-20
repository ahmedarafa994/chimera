"""Google/Gemini Provider Plugin for Project Chimera.

Implements the ProviderPlugin protocol for Google AI (Gemini) integration,
supporting Gemini 3, Gemini 2.5, Gemini 2.0, and Gemini 1.5 models.

Handles both "google" and "gemini" aliases.
"""

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from app.services.provider_plugins import (
    BaseProviderPlugin,
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderCapability,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class GooglePlugin(BaseProviderPlugin):
    """Provider plugin for Google AI (Gemini) API.

    Supports Gemini 3 Pro, Gemini 2.5 Pro/Flash, Gemini 2.0, Gemini 1.5, etc.
    Handles both "google" and "gemini" as aliases.
    """

    @property
    def provider_type(self) -> str:
        return "google"

    @property
    def display_name(self) -> str:
        return "Google AI (Gemini)"

    @property
    def aliases(self) -> list[str]:
        return ["gemini", "google-ai", "google-gemini"]

    @property
    def api_key_env_var(self) -> str:
        return "GOOGLE_API_KEY"

    @property
    def base_url(self) -> str:
        return os.environ.get(
            "DIRECT_GOOGLE_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        )

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.JSON_MODE,
            ProviderCapability.SYSTEM_MESSAGES,
            ProviderCapability.REASONING,
        ]

    def get_default_model(self) -> str:
        return "gemini-3-pro-preview"

    def _get_static_models(self) -> list[ModelInfo]:
        """Return static model definitions for Google AI."""
        return [
            # Gemini 3 Models (Latest)
            ModelInfo(
                model_id="gemini-3-pro-preview",
                provider_id="google",
                display_name="Gemini 3 Pro Preview",
                context_window=1000000,
                max_output_tokens=65536,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_reasoning=True,
                input_price_per_1k=0.00125,
                output_price_per_1k=0.005,
                capabilities=["chat", "vision", "function_calling", "reasoning"],
            ),
            ModelInfo(
                model_id="gemini-3-pro-image-preview",
                provider_id="google",
                display_name="Gemini 3 Pro Image Preview",
                context_window=1000000,
                max_output_tokens=65536,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_reasoning=True,
                input_price_per_1k=0.00125,
                output_price_per_1k=0.005,
                capabilities=[
                    "chat",
                    "vision",
                    "function_calling",
                    "reasoning",
                    "image_generation",
                ],
            ),
            # Gemini 2.5 Models
            ModelInfo(
                model_id="gemini-2.5-pro",
                provider_id="google",
                display_name="Gemini 2.5 Pro",
                context_window=1000000,
                max_output_tokens=65536,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_reasoning=True,
                input_price_per_1k=0.00125,
                output_price_per_1k=0.005,
                capabilities=["chat", "vision", "function_calling", "reasoning"],
            ),
            ModelInfo(
                model_id="gemini-2.5-pro-preview-06-05",
                provider_id="google",
                display_name="Gemini 2.5 Pro (2025-06-05)",
                context_window=1000000,
                max_output_tokens=65536,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_reasoning=True,
                input_price_per_1k=0.00125,
                output_price_per_1k=0.005,
                capabilities=["chat", "vision", "function_calling", "reasoning"],
            ),
            ModelInfo(
                model_id="gemini-2.5-flash",
                provider_id="google",
                display_name="Gemini 2.5 Flash",
                context_window=1000000,
                max_output_tokens=65536,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_reasoning=True,
                input_price_per_1k=0.000075,
                output_price_per_1k=0.0003,
                capabilities=["chat", "vision", "function_calling", "reasoning"],
            ),
            ModelInfo(
                model_id="gemini-2.5-flash-lite",
                provider_id="google",
                display_name="Gemini 2.5 Flash Lite",
                context_window=1000000,
                max_output_tokens=65536,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.000075,
                output_price_per_1k=0.0003,
                capabilities=["chat", "vision", "function_calling"],
            ),
            # Gemini 2.0 Models
            ModelInfo(
                model_id="gemini-2.0-flash",
                provider_id="google",
                display_name="Gemini 2.0 Flash",
                context_window=1000000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.0001,
                output_price_per_1k=0.0004,
                capabilities=["chat", "vision", "function_calling"],
            ),
            ModelInfo(
                model_id="gemini-2.0-flash-lite",
                provider_id="google",
                display_name="Gemini 2.0 Flash Lite",
                context_window=1000000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.000075,
                output_price_per_1k=0.0003,
                capabilities=["chat", "vision", "function_calling"],
            ),
            # Gemini 1.5 Models (Stable)
            ModelInfo(
                model_id="gemini-1.5-pro",
                provider_id="google",
                display_name="Gemini 1.5 Pro",
                context_window=2000000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.00125,
                output_price_per_1k=0.005,
                capabilities=["chat", "vision", "function_calling"],
            ),
            ModelInfo(
                model_id="gemini-1.5-flash",
                provider_id="google",
                display_name="Gemini 1.5 Flash",
                context_window=1000000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.000075,
                output_price_per_1k=0.0003,
                capabilities=["chat", "vision", "function_calling"],
            ),
            ModelInfo(
                model_id="gemini-1.5-flash-8b",
                provider_id="google",
                display_name="Gemini 1.5 Flash 8B",
                context_window=1000000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_1k=0.0000375,
                output_price_per_1k=0.00015,
                capabilities=["chat", "vision", "function_calling"],
            ),
        ]

    async def list_models(self, api_key: str | None = None) -> list[ModelInfo]:
        """List available Google AI models.

        Attempts to fetch from API, falls back to static list on failure.
        """
        key = self._get_api_key(api_key)

        # Try to fetch from API
        if key:
            try:
                return await self._fetch_models_from_api(key)
            except Exception as e:
                logger.warning(f"Failed to fetch Google models from API: {e}")

        # Fallback to static list
        return self._get_static_models()

    async def _fetch_models_from_api(self, api_key: str) -> list[ModelInfo]:
        """Fetch models from Google AI API."""
        try:
            from google import genai

            client = genai.Client(api_key=api_key)

            # Fetch models in a thread to avoid event loop issues
            response = await asyncio.to_thread(client.models.list)

            # Filter to Gemini models and map to ModelInfo
            static_models = {m.model_id: m for m in self._get_static_models()}
            models = []

            for model in response:
                model_id = model.name.replace("models/", "")

                # Skip non-Gemini models
                if not model_id.startswith("gemini"):
                    continue

                # Use static info if available
                if model_id in static_models:
                    models.append(static_models[model_id])
                else:
                    # Create basic info from API response
                    models.append(
                        ModelInfo(
                            model_id=model_id,
                            provider_id="google",
                            display_name=model.display_name or model_id,
                            context_window=getattr(model, "input_token_limit", None),
                            max_output_tokens=getattr(model, "output_token_limit", None),
                            supports_streaming=True,
                            capabilities=["chat"],
                        ),
                    )

            # Add any static models not in API response
            api_model_ids = {m.model_id for m in models}
            for static_model in static_models.values():
                if static_model.model_id not in api_model_ids:
                    models.append(static_model)

            return models

        except Exception as e:
            logger.exception(f"Error fetching Google models: {e}")
            raise

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate Google API key by making a test request."""
        try:
            from google import genai

            client = genai.Client(api_key=api_key)

            # Try to list models as a simple validation
            await asyncio.to_thread(client.models.list)
            return True
        except Exception as e:
            logger.warning(f"Google API key validation failed: {e}")
            return False

    async def generate(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> GenerationResponse:
        """Generate text using Google AI API."""
        from google import genai
        from google.genai import types

        key = self._get_api_key(api_key)
        if not key:
            msg = "Google API key is required"
            raise ValueError(msg)

        client = genai.Client(api_key=key)
        model = request.model or self.get_default_model()

        start_time = time.time()

        # Build content
        if request.messages:
            # Convert messages to Gemini format
            contents = []
            for msg in request.messages:
                role = msg.get("role", "user")
                # Gemini uses "user" and "model" roles
                if role == "assistant":
                    role = "model"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        else:
            contents = request.prompt

        # Build generation config
        config_params: dict[str, Any] = {}

        if request.temperature is not None:
            config_params["temperature"] = request.temperature
        if request.max_tokens is not None:
            config_params["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            config_params["top_p"] = request.top_p
        if request.top_k is not None:
            config_params["top_k"] = request.top_k
        if request.stop_sequences:
            config_params["stop_sequences"] = request.stop_sequences
        if request.system_instruction:
            config_params["system_instruction"] = request.system_instruction

        config = types.GenerateContentConfig(**config_params) if config_params else None

        try:
            # Use sync API in thread to avoid event loop issues
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage metadata
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(
                        response.usage_metadata,
                        "candidates_token_count",
                        0,
                    ),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }

            return GenerationResponse(
                text=response.text or "",
                model_used=model,
                provider="google",
                finish_reason=getattr(
                    response.candidates[0] if response.candidates else None,
                    "finish_reason",
                    None,
                ),
                usage=usage,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.exception(f"Google generation error: {e}")
            raise

    async def generate_stream(
        self,
        request: GenerationRequest,
        api_key: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text generation from Google AI API."""
        from google import genai
        from google.genai import types

        key = self._get_api_key(api_key)
        if not key:
            msg = "Google API key is required"
            raise ValueError(msg)

        client = genai.Client(api_key=key)
        model = request.model or self.get_default_model()

        # Build content
        if request.messages:
            contents = []
            for msg in request.messages:
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "model"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        else:
            contents = request.prompt

        # Build generation config
        config_params: dict[str, Any] = {}

        if request.temperature is not None:
            config_params["temperature"] = request.temperature
        if request.max_tokens is not None:
            config_params["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            config_params["top_p"] = request.top_p
        if request.top_k is not None:
            config_params["top_k"] = request.top_k
        if request.stop_sequences:
            config_params["stop_sequences"] = request.stop_sequences
        if request.system_instruction:
            config_params["system_instruction"] = request.system_instruction

        config = types.GenerateContentConfig(**config_params) if config_params else None

        try:
            # Use sync streaming API in thread
            def stream_content():
                return client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                )

            stream = await asyncio.to_thread(stream_content)

            for chunk in stream:
                if chunk.text:
                    yield StreamChunk(
                        text=chunk.text,
                        is_final=False,
                    )

            # Final chunk with metadata
            yield StreamChunk(
                text="",
                is_final=True,
            )
        except Exception as e:
            logger.exception(f"Google streaming error: {e}")
            raise
