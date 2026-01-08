"""
ProxyProviderAdapter for routing LLM requests through proxy server.

STORY-1.3: Base adapter that implements the LLMProvider protocol
for proxy mode communication.

Features:
- Implements LLMProvider interface for seamless integration
- Routes requests through ProxyClient to AIClient-2-API server
- Supports fallback to direct mode on proxy failure
- Streaming support through proxy (when available)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.domain.interfaces import LLMProvider
from app.domain.models import (
    PromptRequest,
    PromptResponse,
    StreamChunk,
)
from app.infrastructure.proxy.proxy_client import (
    ProxyClient,
    ProxyConnectionError,
    ProxyError,
    ProxyTimeoutError,
    get_proxy_client,
)


class ProxyProviderAdapter(LLMProvider):
    """
    Adapter that routes LLM requests through the proxy server.

    This adapter implements the LLMProvider interface, allowing it to be
    used interchangeably with direct provider implementations. When proxy
    mode is enabled, requests are routed through the AIClient-2-API server.
    """

    def __init__(
        self,
        provider_name: str,
        default_model: str | None = None,
        fallback_provider: LLMProvider | None = None,
        proxy_client: ProxyClient | None = None,
    ):
        """
        Initialize the ProxyProviderAdapter.

        Args:
            provider_name: Name of the target provider (e.g., 'openai')
            default_model: Default model to use for this provider
            fallback_provider: Optional direct provider for fallback
            proxy_client: Optional custom ProxyClient instance
        """
        self._provider_name = provider_name
        self._default_model = default_model
        self._fallback_provider = fallback_provider
        self._proxy_client = proxy_client or get_proxy_client()

        self._request_count = 0
        self._fallback_count = 0
        self._error_count = 0

        logger.info(
            f"ProxyProviderAdapter initialized: provider={provider_name}, "
            f"default_model={default_model}, "
            f"has_fallback={fallback_provider is not None}"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def supports_streaming(self) -> bool:
        """Check if streaming is supported through proxy."""
        # Streaming through proxy depends on proxy server capability
        return True

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """
        Generate text through the proxy server.

        Args:
            request: The prompt request

        Returns:
            PromptResponse with generated text

        Raises:
            ProviderNotAvailableError: When proxy and fallback both fail
        """
        start_time = time.time()
        self._request_count += 1

        model = request.model or self._default_model
        config = self._build_config(request)

        try:
            response = await self._proxy_client.send_request(
                provider=self._provider_name,
                model=model,
                prompt=request.prompt,
                config=config,
            )

            latency_ms = (time.time() - start_time) * 1000

            return PromptResponse(
                text=response.get("text", response.get("content", "")),
                model_used=response.get("model", model or "unknown"),
                provider=f"proxy:{self._provider_name}",
                usage_metadata=response.get("usage", {}),
                latency_ms=latency_ms,
            )

        except (ProxyConnectionError, ProxyTimeoutError) as e:
            self._error_count += 1
            logger.warning(
                f"Proxy request failed for {self._provider_name}: {e}"
            )

            if self._should_fallback():
                return await self._execute_fallback(request, start_time)

            raise

        except ProxyError as e:
            self._error_count += 1
            logger.error(f"Proxy error for {self._provider_name}: {e}")

            if self._should_fallback() and e.status_code in (502, 503, 504):
                return await self._execute_fallback(request, start_time)

            raise

    def _build_config(self, request: PromptRequest) -> dict[str, Any]:
        """Build configuration dict from request."""
        config: dict[str, Any] = {}

        if request.config:
            if request.config.temperature is not None:
                config["temperature"] = request.config.temperature
            if request.config.max_output_tokens is not None:
                config["max_tokens"] = request.config.max_output_tokens
            if request.config.top_p is not None:
                config["top_p"] = request.config.top_p
            if request.config.top_k is not None:
                config["top_k"] = request.config.top_k

        if request.system_instruction:
            config["system"] = request.system_instruction

        return config

    def _should_fallback(self) -> bool:
        """Check if fallback to direct mode should be attempted."""
        return (
            settings.PROXY_MODE_FALLBACK_TO_DIRECT
            and self._fallback_provider is not None
        )

    async def _execute_fallback(
        self,
        request: PromptRequest,
        start_time: float,
    ) -> PromptResponse:
        """Execute request using fallback provider."""
        logger.info(
            f"Falling back to direct mode for {self._provider_name}"
        )
        self._fallback_count += 1

        if self._fallback_provider is None:
            raise ProxyError("No fallback provider configured")

        response = await self._fallback_provider.generate(request)

        # Update latency to include fallback time
        total_latency = (time.time() - start_time) * 1000
        response.latency_ms = total_latency
        response.provider = f"fallback:{self._provider_name}"

        return response

    async def generate_stream(
        self,
        request: PromptRequest,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream text generation through proxy.

        Note: Streaming through proxy may have additional latency.
        Falls back to buffered generation if proxy doesn't support streaming.
        """
        model = request.model or self._default_model
        config = self._build_config(request)

        try:
            # Attempt streaming through proxy
            response = await self._proxy_client.send_request(
                provider=self._provider_name,
                model=model,
                prompt=request.prompt,
                config={**config, "stream": True},
            )

            # If proxy returns full response, simulate streaming
            text = response.get("text", response.get("content", ""))
            chunk_size = 50

            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                yield StreamChunk(
                    text=chunk_text,
                    is_final=(i + chunk_size >= len(text)),
                    model=response.get("model", model),
                    provider=f"proxy:{self._provider_name}",
                )

        except ProxyError as e:
            if self._should_fallback():
                async for chunk in self._fallback_provider.generate_stream(
                    request
                ):
                    yield chunk
            else:
                raise NotImplementedError(
                    f"Streaming through proxy failed: {e}"
                )

    async def check_health(self) -> bool:
        """Check if the proxy connection is healthy."""
        health = await self._proxy_client.check_health()
        return health.is_healthy

    async def count_tokens(
        self,
        text: str,
        model: str | None = None,
    ) -> int:
        """
        Count tokens using proxy or fallback.

        Token counting through proxy may not be available for all providers.
        Falls back to direct provider if available.
        """
        if self._fallback_provider is not None:
            return await self._fallback_provider.count_tokens(text, model)

        # Rough estimation if no fallback available
        return len(text) // 4

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            "provider": self._provider_name,
            "default_model": self._default_model,
            "request_count": self._request_count,
            "fallback_count": self._fallback_count,
            "error_count": self._error_count,
            "fallback_rate": (
                self._fallback_count / self._request_count
                if self._request_count > 0
                else 0.0
            ),
            "has_fallback": self._fallback_provider is not None,
            "proxy_healthy": self._proxy_client.is_healthy,
        }


def create_proxy_adapter(
    provider_name: str,
    direct_provider: LLMProvider | None = None,
    default_model: str | None = None,
) -> ProxyProviderAdapter:
    """
    Factory function to create a ProxyProviderAdapter.

    Args:
        provider_name: Name of the target provider
        direct_provider: Optional direct provider for fallback
        default_model: Default model to use

    Returns:
        Configured ProxyProviderAdapter instance
    """
    return ProxyProviderAdapter(
        provider_name=provider_name,
        default_model=default_model,
        fallback_provider=direct_provider,
    )
