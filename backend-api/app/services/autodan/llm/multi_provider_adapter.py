"""
Multi-Provider LLM Adapter for AutoDAN

This module provides a unified interface for multiple LLM providers:
- Google (Gemini)
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- DeepSeek
- Local models (via vLLM, Ollama)

Features:
- Automatic provider selection based on availability
- Fallback chains for reliability
- Rate limiting and quota management
- Unified response format
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM providers."""

    GOOGLE = "google"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    VLLM = "vllm"
    GROQ = "groq"
    TOGETHER = "together"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    provider_type: ProviderType
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    max_tokens: int = 4096
    timeout_seconds: float | None = None  # No timeout
    rate_limit_rpm: int | None = None  # Requests per minute
    enabled: bool = True
    priority: int = 0  # Higher = preferred

    # Model mappings
    available_models: list[str] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Unified response format from LLM providers."""

    content: str
    model: str
    provider: ProviderType
    latency_ms: float
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMRequest:
    """Unified request format for LLM providers."""

    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._last_request_time = 0.0
        self._request_count = 0

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

    async def _rate_limit(self):
        """Apply rate limiting if configured."""
        if self.config.rate_limit_rpm:
            min_interval = 60.0 / self.config.rate_limit_rpm
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1


class GoogleProvider(BaseLLMProvider):
    """Google/Gemini LLM provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Get or create the Gemini client."""
        if self._client is None:
            from google import genai

            api_key = self.config.api_key or settings.GOOGLE_API_KEY
            if api_key:
                self._client = genai.Client(api_key=api_key)
            else:
                self._client = None
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Gemini."""
        await self._rate_limit()
        start_time = time.time()

        try:
            from google.genai import types

            client = await self._get_client()
            if not client:
                raise RuntimeError("Gemini client not available")

            model_name = request.model or self.config.default_model or "gemini-3-pro-preview"

            # Configure generation
            generation_config_params = {
                "temperature": request.temperature,
                "max_output_tokens": request.max_tokens,
                "top_p": request.top_p,
            }

            # Gemini 3 Pro support: Add thinking_config for enhanced reasoning
            thinking_level = request.metadata.get("thinking_level") if request.metadata else None
            if "gemini-3" in model_name and thinking_level:
                generation_config_params["thinking_config"] = {"thinking_budget": thinking_level}

            config = types.GenerateContentConfig(**generation_config_params)

            # Combine system instruction with prompt if provided
            if request.system_prompt:
                contents = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
            else:
                contents = request.prompt

            # Generate using new SDK
            response = client.models.generate_content(
                model=model_name, contents=contents, config=config
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.text if response.text else "",
                model=model_name,
                provider=ProviderType.GOOGLE,
                latency_ms=latency_ms,
                metadata={"finish_reason": "stop"},
            )

        except Exception as e:
            logger.error(f"Google provider error: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Google provider is available."""
        try:
            api_key = self.config.api_key or settings.GOOGLE_API_KEY
            return bool(api_key)
        except Exception:
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                api_key = self.config.api_key or getattr(settings, "OPENAI_API_KEY", None)
                base_url = self.config.base_url
                self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                logger.warning("OpenAI package not installed")
                raise
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI."""
        await self._rate_limit()
        start_time = time.time()

        try:
            client = await self._get_client()
            model_name = request.model or self.config.default_model or "gpt-4"

            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model_name,
                provider=ProviderType.OPENAI,
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            logger.error(f"OpenAI provider error: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        try:
            api_key = self.config.api_key or getattr(settings, "OPENAI_API_KEY", None)
            return bool(api_key)
        except Exception:
            return False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) LLM provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                api_key = self.config.api_key or getattr(settings, "ANTHROPIC_API_KEY", None)
                self._client = AsyncAnthropic(api_key=api_key)
            except ImportError:
                logger.warning("Anthropic package not installed")
                raise
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Claude."""
        await self._rate_limit()
        start_time = time.time()

        try:
            client = await self._get_client()
            model_name = request.model or self.config.default_model or "claude-3-sonnet-20240229"

            response = await client.messages.create(
                model=model_name,
                max_tokens=request.max_tokens,
                system=request.system_prompt or "",
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                top_p=request.top_p,
            )

            latency_ms = (time.time() - start_time) * 1000

            content = ""
            if response.content:
                content = response.content[0].text if response.content[0].type == "text" else ""

            return LLMResponse(
                content=content,
                model=model_name,
                provider=ProviderType.ANTHROPIC,
                latency_ms=latency_ms,
                tokens_used=(
                    response.usage.input_tokens + response.usage.output_tokens
                    if response.usage
                    else 0
                ),
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                metadata={"stop_reason": response.stop_reason},
            )

        except Exception as e:
            logger.error(f"Anthropic provider error: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Anthropic provider is available."""
        try:
            api_key = self.config.api_key or getattr(settings, "ANTHROPIC_API_KEY", None)
            return bool(api_key)
        except Exception:
            return False


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek LLM provider (OpenAI-compatible API)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Get or create the DeepSeek client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                api_key = self.config.api_key or getattr(settings, "DEEPSEEK_API_KEY", None)
                base_url = self.config.base_url or "https://api.deepseek.com/v1"
                self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                logger.warning("OpenAI package not installed (required for DeepSeek)")
                raise
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using DeepSeek."""
        await self._rate_limit()
        start_time = time.time()

        try:
            client = await self._get_client()
            model_name = request.model or self.config.default_model or "deepseek-chat"

            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model_name,
                provider=ProviderType.DEEPSEEK,
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            logger.error(f"DeepSeek provider error: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if DeepSeek provider is available."""
        try:
            api_key = self.config.api_key or getattr(settings, "DEEPSEEK_API_KEY", None)
            return bool(api_key)
        except Exception:
            return False


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._base_url = config.base_url or "http://localhost:11434"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama."""
        await self._rate_limit()
        start_time = time.time()

        try:
            import httpx

            model_name = request.model or self.config.default_model or "llama2"

            prompt = request.prompt
            if request.system_prompt:
                prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"

            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": request.temperature,
                            "top_p": request.top_p,
                            "num_predict": request.max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=data.get("response", ""),
                model=model_name,
                provider=ProviderType.OLLAMA,
                latency_ms=latency_ms,
                metadata={
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                },
            )

        except Exception as e:
            logger.error(f"Ollama provider error: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


class MultiProviderAdapter:
    """
    Multi-provider LLM adapter with automatic fallback.

    Features:
    - Automatic provider selection based on availability and priority
    - Fallback chains for reliability
    - Unified interface for all providers
    - Rate limiting and quota management
    """

    def __init__(self):
        self._providers: dict[ProviderType, BaseLLMProvider] = {}
        self._fallback_chain: list[ProviderType] = []
        self._default_provider: ProviderType | None = None

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_count": 0,
            "provider_usage": {},
        }

    def register_provider(
        self, provider_type: ProviderType, config: ProviderConfig, is_default: bool = False
    ):
        """Register a provider with the adapter."""
        provider_class = self._get_provider_class(provider_type)
        if provider_class:
            self._providers[provider_type] = provider_class(config)
            self._stats["provider_usage"][provider_type.value] = 0

            if is_default:
                self._default_provider = provider_type

            logger.info(f"Registered provider: {provider_type.value} (default={is_default})")

    def _get_provider_class(self, provider_type: ProviderType) -> type:
        """Get the provider class for a provider type."""
        mapping = {
            ProviderType.GOOGLE: GoogleProvider,
            ProviderType.GEMINI: GoogleProvider,
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.ANTHROPIC: AnthropicProvider,
            ProviderType.DEEPSEEK: DeepSeekProvider,
            ProviderType.OLLAMA: OllamaProvider,
        }
        return mapping.get(provider_type)

    def set_fallback_chain(self, chain: list[ProviderType]):
        """Set the fallback chain for provider selection."""
        self._fallback_chain = chain
        logger.info(f"Fallback chain set: {[p.value for p in chain]}")

    async def generate(
        self, request: LLMRequest, provider: ProviderType | None = None, use_fallback: bool = True
    ) -> LLMResponse:
        """
        Generate a response using the specified or default provider.

        Args:
            request: The LLM request
            provider: Specific provider to use (optional)
            use_fallback: Whether to use fallback chain on failure

        Returns:
            LLMResponse from the provider
        """
        self._stats["total_requests"] += 1

        # Determine provider order
        providers_to_try = []
        if provider:
            providers_to_try.append(provider)
        elif self._default_provider:
            providers_to_try.append(self._default_provider)

        if use_fallback:
            providers_to_try.extend(p for p in self._fallback_chain if p not in providers_to_try)

        last_error = None
        for ptype in providers_to_try:
            if ptype not in self._providers:
                continue

            provider_instance = self._providers[ptype]

            # Check availability
            if not await provider_instance.is_available():
                logger.warning(f"Provider {ptype.value} not available, trying next")
                continue

            try:
                response = await provider_instance.generate(request)
                self._stats["successful_requests"] += 1
                self._stats["provider_usage"][ptype.value] += 1

                if ptype != providers_to_try[0]:
                    self._stats["fallback_count"] += 1

                return response

            except Exception as e:
                logger.warning(f"Provider {ptype.value} failed: {e}")
                last_error = e
                continue

        self._stats["failed_requests"] += 1
        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    async def generate_with_retry(
        self, request: LLMRequest, max_retries: int = 3, retry_delay: float = 1.0
    ) -> LLMResponse:
        """Generate with automatic retry on failure."""
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.generate(request)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))

        raise last_error

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        total = self._stats["total_requests"]
        successful = self._stats["successful_requests"]

        return {
            **self._stats,
            "success_rate": successful / total if total > 0 else 0.0,
            "fallback_rate": self._stats["fallback_count"] / successful if successful > 0 else 0.0,
        }

    async def get_available_providers(self) -> list[ProviderType]:
        """Get list of currently available providers."""
        available = []
        for ptype, provider in self._providers.items():
            if await provider.is_available():
                available.append(ptype)
        return available


# Global adapter instance
multi_provider_adapter = MultiProviderAdapter()


def setup_default_providers():
    """Setup default providers based on available API keys."""
    # Google/Gemini
    if hasattr(settings, "GOOGLE_API_KEY") and settings.GOOGLE_API_KEY:
        multi_provider_adapter.register_provider(
            ProviderType.GOOGLE,
            ProviderConfig(
                provider_type=ProviderType.GOOGLE,
                api_key=settings.GOOGLE_API_KEY,
                default_model=getattr(settings, "GOOGLE_MODEL", "gemini-3-pro-preview"),
                rate_limit_rpm=60,
                priority=10,
            ),
            is_default=True,
        )

    # OpenAI
    if hasattr(settings, "OPENAI_API_KEY") and settings.OPENAI_API_KEY:
        multi_provider_adapter.register_provider(
            ProviderType.OPENAI,
            ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key=settings.OPENAI_API_KEY,
                default_model="gpt-4",
                rate_limit_rpm=60,
                priority=8,
            ),
        )

    # DeepSeek
    if hasattr(settings, "DEEPSEEK_API_KEY") and settings.DEEPSEEK_API_KEY:
        multi_provider_adapter.register_provider(
            ProviderType.DEEPSEEK,
            ProviderConfig(
                provider_type=ProviderType.DEEPSEEK,
                api_key=settings.DEEPSEEK_API_KEY,
                default_model="deepseek-chat",
                rate_limit_rpm=60,
                priority=5,
            ),
        )

    # Set fallback chain
    multi_provider_adapter.set_fallback_chain(
        [
            ProviderType.GOOGLE,
            ProviderType.OPENAI,
            ProviderType.DEEPSEEK,
            ProviderType.OLLAMA,
        ]
    )

    logger.info("Default providers configured")
