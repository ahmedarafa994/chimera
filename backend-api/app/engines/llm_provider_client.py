#!/usr/bin/env python3
"""
LLM Provider Client Architecture
Multi-provider support with authentication, rate limiting, and error handling

PERF-001 FIX: Integrated connection pooling for improved HTTP performance.
"""

import hashlib
import json
import os
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import requests

from app.core.config import get_settings
from app.core.connection_pool import get_pooled_session
from app.core.logging import logger

# Configure logging (using app logger instead of basicConfig if available, else fallback)
# logger imported from app.core.logging


class LLMProvider(Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"


@dataclass
class ProviderConfig:
    """Configuration for LLM provider"""

    provider: LLMProvider
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int | None = None  # No timeout
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    cost_per_1k_tokens: float = 0.0


@dataclass
class TokenUsage:
    """Token usage tracking"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


@dataclass
class LLMResponse:
    """Standardized LLM response"""

    content: str
    provider: LLMProvider
    model: str
    usage: TokenUsage
    latency_ms: int
    timestamp: datetime
    cached: bool = False
    etag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate_per_minute: int):
        self.rate = rate_per_minute
        self.tokens = rate_per_minute
        self.last_update = time.time()
        self.lock = None  # Thread lock for production

    def acquire(self) -> bool:
        """Acquire a token, return False if rate limited"""
        current = time.time()
        elapsed = current - self.last_update

        # Refill tokens based on elapsed time
        self.tokens = min(self.rate, self.tokens + (elapsed * self.rate / 60.0))
        self.last_update = current

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def wait_time(self) -> float:
        """Calculate wait time until next token available"""
        if self.tokens >= 1:
            return 0.0
        tokens_needed = 1 - self.tokens
        return (tokens_needed * 60.0) / self.rate


class ResponseCache:
    """Simple in-memory cache with ETag support"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: dict[str, tuple[LLMResponse, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)

    def _generate_key(self, prompt: str, config: dict) -> str:
        """Generate cache key from prompt and config"""
        data = f"{prompt}:{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, prompt: str, config: dict) -> LLMResponse | None:
        """Get cached response if valid"""
        key = self._generate_key(prompt, config)

        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.info(f"Cache HIT for key: {key[:16]}...")
                response.cached = True
                return response
            else:
                # Expired
                del self.cache[key]

        logger.info(f"Cache MISS for key: {key[:16]}...")
        return None

    def set(self, prompt: str, config: dict, response: LLMResponse):
        """Cache a response"""
        key = self._generate_key(prompt, config)

        # Simple LRU: remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]

        self.cache[key] = (response, datetime.now())
        logger.info(f"Cached response for key: {key[:16]}...")

    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        logger.info("Cache cleared")


class LLMProviderClient:
    """Base client for LLM providers"""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self.cache = ResponseCache()
        self.session = self._create_session()
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def _create_session(self) -> requests.Session:
        """
        Create requests session with retry logic and connection pooling.

        PERF-001 FIX: Uses centralized connection pool manager for efficient
        connection reuse and improved throughput.
        """
        # Use connection pool manager for session creation
        provider_name = self.config.provider.value
        return get_pooled_session(provider_name, timeout=self.config.timeout)

    def _wait_for_rate_limit(self):
        """Wait if rate limited (synchronous)"""
        while not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            logger.warning(f"Rate limited. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)

    def _handle_error(self, response: requests.Response) -> str:
        """Handle API error responses"""
        status_code = response.status_code

        error_messages = {
            400: "Bad Request - Invalid parameters",
            401: "Unauthorized - Invalid API key",
            403: "Forbidden - Access denied",
            404: "Not Found - Invalid endpoint",
            429: "Rate Limited - Too many requests",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
        }

        error_msg = error_messages.get(status_code, f"Unknown error (status {status_code})")

        try:
            error_data = response.json()
            if "error" in error_data:
                error_msg += f": {error_data['error']}"
        except BaseException:
            pass

        logger.error(f"API Error: {error_msg}")
        return error_msg

    def generate(
        self, prompt: str, stream: bool = False, use_cache: bool = True, **kwargs
    ) -> LLMResponse:
        """Generate response from LLM"""
        raise NotImplementedError("Subclass must implement generate()")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from LLM"""
        raise NotImplementedError("Subclass must implement generate_stream()")


class OpenAIClient(LLMProviderClient):
    """OpenAI API client"""

    def generate(
        self, prompt: str, stream: bool = False, use_cache: bool = True, **kwargs
    ) -> LLMResponse:
        """Generate response from OpenAI"""

        # Check cache first
        if use_cache:
            cached = self.cache.get(
                prompt, {"model": self.config.model, "temperature": self.config.temperature}
            )
            if cached:
                return cached

        # Rate limiting
        self._wait_for_rate_limit()

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": kwargs.get("model") or self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": stream,
        }

        start_time = time.time()

        try:
            # Ensure proper URL construction (avoid double /v1 if present)
            base = self.config.base_url.rstrip("/")
            if base.endswith("/chat/completions"):
                url = base
            elif "/v1" in base:
                url = f"{base}/chat/completions"
            else:
                url = f"{base}/chat/completions"  # Default assumption

            response = self.session.post(
                url, headers=headers, json=payload, timeout=self.config.timeout
            )

            if response.status_code != 200:
                raise Exception(self._handle_error(response))

            data = response.json()
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            content = data["choices"][0]["message"]["content"]
            usage_data = data.get("usage", {})

            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                estimated_cost=usage_data.get("total_tokens", 0)
                * self.config.cost_per_1k_tokens
                / 1000,
            )

            # Update metrics
            self.request_count += 1
            self.total_tokens += usage.total_tokens
            self.total_cost += usage.estimated_cost

            llm_response = LLMResponse(
                content=content,
                provider=self.config.provider,  # Use configured provider
                model=self.config.model,
                usage=usage,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                cached=False,
            )

            # Cache the response
            if use_cache:
                self.cache.set(
                    prompt,
                    {"model": self.config.model, "temperature": self.config.temperature},
                    llm_response,
                )

            logger.info(
                f"OpenAI request completed: "
                f"{usage.total_tokens} tokens, "
                f"${usage.estimated_cost:.4f}, "
                f"{latency_ms}ms"
            )

            return llm_response

        except Exception as e:
            logger.error(f"OpenAI API error: {e!s}")
            raise


class AnthropicClient(LLMProviderClient):
    """Anthropic Claude API client"""

    def generate(
        self, prompt: str, stream: bool = False, use_cache: bool = True, **kwargs
    ) -> LLMResponse:
        """Generate response from Anthropic Claude"""

        # Check cache first
        if use_cache:
            cached = self.cache.get(
                prompt, {"model": self.config.model, "temperature": self.config.temperature}
            )
            if cached:
                return cached

        # Rate limiting
        self._wait_for_rate_limit()

        # Prepare request
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": kwargs.get("model") or self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": stream,
        }

        start_time = time.time()

        try:
            base = self.config.base_url.rstrip("/")
            url = f"{base}/messages"

            response = self.session.post(
                url, headers=headers, json=payload, timeout=self.config.timeout
            )

            if response.status_code != 200:
                raise Exception(self._handle_error(response))

            data = response.json()
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            content = data["content"][0]["text"]
            usage_data = data.get("usage", {})

            usage = TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
                estimated_cost=(
                    usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
                )
                * self.config.cost_per_1k_tokens
                / 1000,
            )

            # Update metrics
            self.request_count += 1
            self.total_tokens += usage.total_tokens
            self.total_cost += usage.estimated_cost

            llm_response = LLMResponse(
                content=content,
                provider=self.config.provider,  # Use configured provider
                model=self.config.model,
                usage=usage,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                cached=False,
            )

            # Cache the response
            if use_cache:
                self.cache.set(
                    prompt,
                    {"model": self.config.model, "temperature": self.config.temperature},
                    llm_response,
                )

            logger.info(
                f"Anthropic request completed: "
                f"{usage.total_tokens} tokens, "
                f"${usage.estimated_cost:.4f}, "
                f"{latency_ms}ms"
            )

            return llm_response

        except Exception as e:
            logger.error(f"Anthropic API error: {e!s}")
            raise


class GoogleGeminiClient(LLMProviderClient):
    """Google Gemini API client"""

    def generate(
        self, prompt: str, stream: bool = False, use_cache: bool = True, **kwargs
    ) -> LLMResponse:
        """Generate response from Google Gemini"""

        # Check cache first
        if use_cache:
            cached = self.cache.get(
                prompt, {"model": self.config.model, "temperature": self.config.temperature}
            )
            if cached:
                return cached

        # Rate limiting
        self._wait_for_rate_limit()

        # Prepare request
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "topP": kwargs.get("top_p", 0.95),
                "topK": kwargs.get("top_k", 40),
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        start_time = time.time()

        try:
            # HIGH-002 FIX: Use header instead of query parameter to prevent API key exposure in logs
            model = kwargs.get("model") or self.config.model
            base = self.config.base_url.rstrip("/")
            url = f"{base}/models/{model}:generateContent"

            # Add API key as header instead of query param for security
            if self.config.api_key:
                headers["x-goog-api-key"] = self.config.api_key

            response = self.session.post(
                url, headers=headers, json=payload, timeout=self.config.timeout
            )

            if response.status_code != 200:
                raise Exception(self._handle_error(response))

            data = response.json()
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            if "candidates" not in data or not data["candidates"]:
                # Check for prompt blocking at the top level
                if data.get("promptFeedback", {}).get("blockReason"):
                    block_reason = data["promptFeedback"]["blockReason"]
                    raise Exception(f"Gemini blocked the prompt. Reason: {block_reason}")
                raise Exception("No response candidates generated from Gemini")

            candidate = data["candidates"][0]
            content_obj = candidate.get("content", {})
            parts = content_obj.get("parts", [])

            if not parts:
                finish_reason = candidate.get("finishReason", "UNKNOWN")
                if finish_reason == "SAFETY":
                    raise Exception("Gemini response blocked by safety filters.")
                elif finish_reason == "RECITATION":
                    raise Exception("Gemini response blocked: Recitation check failed.")
                elif finish_reason in ("STOP", "MAX_TOKENS"):
                    # MAX_TOKENS with no content means prompt was too long or max_tokens too small
                    logger.warning(
                        f"Gemini {finish_reason} reason with no parts. Full candidate: {candidate}"
                    )
                    content = ""
                else:
                    raise Exception(
                        f"Gemini generated empty response. Finish Reason: {finish_reason}"
                    )
            else:
                content = parts[0].get("text", "")
                if not content:
                    logger.warning(f"Gemini parts found but no text. Parts: {parts}")

            # Extract token usage
            usage_metadata = data.get("usageMetadata", {})
            prompt_tokens = usage_metadata.get("promptTokenCount", 0)
            completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
            total_tokens = usage_metadata.get("totalTokenCount", prompt_tokens + completion_tokens)

            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=total_tokens * self.config.cost_per_1k_tokens / 1000,
            )

            # Update metrics
            self.request_count += 1
            self.total_tokens += usage.total_tokens
            self.total_cost += usage.estimated_cost

            llm_response = LLMResponse(
                content=content,
                provider=self.config.provider,  # Use configured provider
                model=self.config.model,
                usage=usage,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                cached=False,
            )

            # Cache the response
            if use_cache:
                self.cache.set(
                    prompt,
                    {"model": self.config.model, "temperature": self.config.temperature},
                    llm_response,
                )

            logger.info(
                f"Google Gemini request completed: "
                f"{usage.total_tokens} tokens, "
                f"${usage.estimated_cost:.4f}, "
                f"{latency_ms}ms"
            )

            return llm_response

        except Exception as e:
            logger.error(f"Google Gemini API error: {e!s}")
            raise


class CustomModelClient(LLMProviderClient):
    """Custom model endpoint client"""

    def generate(
        self, prompt: str, stream: bool = False, use_cache: bool = True, **kwargs
    ) -> LLMResponse:
        """Generate response from custom model endpoint"""

        # Check cache first
        if use_cache:
            cached = self.cache.get(
                prompt, {"model": self.config.model, "temperature": self.config.temperature}
            )
            if cached:
                return cached

        # Rate limiting
        self._wait_for_rate_limit()

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": stream,
        }

        start_time = time.time()

        try:
            response = self.session.post(
                f"{self.config.base_url}/generate",
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )

            if response.status_code != 200:
                raise Exception(self._handle_error(response))

            data = response.json()
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data (flexible schema)
            content = data.get("text", data.get("response", data.get("output", "")))

            # Estimate tokens if not provided
            tokens_used = data.get("tokens", len(content.split()) * 1.3)

            usage = TokenUsage(
                prompt_tokens=int(len(prompt.split()) * 1.3),
                completion_tokens=int(tokens_used),
                total_tokens=int(len(prompt.split()) * 1.3 + tokens_used),
                estimated_cost=tokens_used * self.config.cost_per_1k_tokens / 1000,
            )

            # Update metrics
            self.request_count += 1
            self.total_tokens += usage.total_tokens
            self.total_cost += usage.estimated_cost

            llm_response = LLMResponse(
                content=content,
                provider=self.config.provider,  # Use configured provider
                model=self.config.model,
                usage=usage,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                cached=False,
                metadata=data,
            )

            # Cache the response
            if use_cache:
                self.cache.set(
                    prompt,
                    {"model": self.config.model, "temperature": self.config.temperature},
                    llm_response,
                )

            logger.info(
                f"Custom model request completed: {usage.total_tokens} tokens, {latency_ms}ms"
            )

            return llm_response

        except Exception as e:
            logger.error(f"Custom model API error: {e!s}")
            raise


class LLMClientFactory:
    """Factory for creating LLM clients"""

    @staticmethod
    def create_client(config: ProviderConfig) -> LLMProviderClient:
        """Create appropriate client based on provider"""
        clients = {
            LLMProvider.OPENAI: OpenAIClient,
            LLMProvider.ANTHROPIC: AnthropicClient,
            LLMProvider.GOOGLE: GoogleGeminiClient,
            LLMProvider.CUSTOM: CustomModelClient,
        }

        client_class = clients.get(config.provider)
        if not client_class:
            raise ValueError(f"Unsupported provider: {config.provider}")

        return client_class(config)

    @staticmethod
    def from_env(provider: LLMProvider) -> LLMProviderClient:
        """Create client from centralized settings"""
        settings = get_settings()

        def get_prov_key(prov_enum: LLMProvider):
            # Helper to map enum to string for settings
            if prov_enum == LLMProvider.OPENAI:
                return "openai"
            if prov_enum == LLMProvider.ANTHROPIC:
                return "anthropic"
            if prov_enum == LLMProvider.GOOGLE:
                return "google"
            return "custom"

        # Determine configuration
        prov_key = get_prov_key(provider)

        # Base URL resolution (direct only)
        base_url = settings.get_provider_endpoint(prov_key)

        config = None
        if provider == LLMProvider.OPENAI:
            config = ProviderConfig(
                provider=LLMProvider.OPENAI,
                api_key=settings.OPENAI_API_KEY,
                base_url=base_url,
                model=settings.OPENAI_MODEL or "gpt-4",
                cost_per_1k_tokens=0.03,
            )
        elif provider == LLMProvider.ANTHROPIC:
            config = ProviderConfig(
                provider=LLMProvider.ANTHROPIC,
                api_key=settings.ANTHROPIC_API_KEY,
                base_url=base_url,
                model=settings.ANTHROPIC_MODEL or "claude-3-opus-20240229",
                cost_per_1k_tokens=0.015,
            )
        elif provider == LLMProvider.GOOGLE:
            config = ProviderConfig(
                provider=LLMProvider.GOOGLE,
                api_key=settings.GOOGLE_API_KEY,
                base_url=base_url,
                model=settings.GOOGLE_MODEL or "gemini-1.5-pro",
                cost_per_1k_tokens=0.00125,
            )
        elif provider == LLMProvider.CUSTOM:
            config = ProviderConfig(
                provider=LLMProvider.CUSTOM,
                api_key=os.getenv("CUSTOM_API_KEY", ""),
                base_url=base_url,
                model=os.getenv("CUSTOM_MODEL", "custom-model"),
                cost_per_1k_tokens=0.0,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return LLMClientFactory.create_client(config)


# Example usage
if __name__ == "__main__":
    # Create OpenAI client from environment
    try:
        from app.core.config import get_settings  # Ensure loaded

        client = LLMClientFactory.from_env(LLMProvider.OPENAI)

        prompt = "Explain quantum computing in simple terms."
        response = client.generate(prompt)

        print(f"Response: {response.content[:200]}...")
        print(f"Tokens: {response.usage.total_tokens}")
        print(f"Cost: ${response.usage.estimated_cost:.4f}")
        print(f"Latency: {response.latency_ms}ms")
        print(f"Cached: {response.cached}")

    except Exception as e:
        print(f"Error: {e}")
