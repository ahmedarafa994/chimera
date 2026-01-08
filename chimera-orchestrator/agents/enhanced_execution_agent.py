"""
Enhanced Execution Agent - Advanced LLM Deployment with Multi-Provider Support
Provides streaming, batching, circuit breaker, and comprehensive error handling
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp

try:
    from core.config import Config, LLMProviderConfig
    from core.enhanced_models import EnhancedPrompt, EventType, LLMResponse, SystemEvent
    from core.event_bus import EventBus, EventHandler
    from core.message_queue import MessageQueue
    from core.models import AgentType, Message, MessageType

    from agents.base_agent import BaseAgent
except ImportError:
    from ..core.config import Config, LLMProviderConfig
    from ..core.enhanced_models import EventType, LLMResponse, SystemEvent
    from ..core.event_bus import EventBus
    from ..core.message_queue import MessageQueue
    from ..core.models import AgentType, Message, MessageType
    from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for provider resilience."""

    provider: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    half_open_calls: int = 0

    def record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker for {self.provider} closed")
        else:
            self.failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
            logger.warning(f"Circuit breaker for {self.provider} re-opened")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker for {self.provider} opened after {self.failure_count} failures"
            )

    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker for {self.provider} half-open")
                    return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def execute(self, prompt: str, model: str | None = None, **kwargs) -> LLMResponse:
        """Execute a prompt and return the response."""
        pass

    @abstractmethod
    async def execute_streaming(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        """Execute a prompt with streaming response."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: LLMProviderConfig, session: aiohttp.ClientSession):
        super().__init__(config)
        self.session = session
        self.base_url = config.base_url or "https://api.openai.com/v1"

    async def execute(self, prompt: str, model: str | None = None, **kwargs) -> LLMResponse:
        """Execute a prompt using OpenAI API."""
        start_time = time.time()

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.config.model or "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        try:
            async with self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", self.config.timeout)),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                response_time = int((time.time() - start_time) * 1000)

                return LLMResponse(
                    provider=self.name,
                    model=data.get("model", payload["model"]),
                    response_text=data["choices"][0]["message"]["content"],
                    response_time_ms=response_time,
                    tokens_input=data.get("usage", {}).get("prompt_tokens", 0),
                    tokens_output=data.get("usage", {}).get("completion_tokens", 0),
                    tokens_total=data.get("usage", {}).get("total_tokens", 0),
                    finish_reason=data["choices"][0].get("finish_reason", ""),
                    success=True,
                    raw_response=data,
                )

        except aiohttp.ClientError as e:
            return LLMResponse(
                provider=self.name,
                model=model or self.config.model,
                response_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e),
            )

    async def execute_streaming(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        """Execute with streaming response."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.config.model or "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        async with self.session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", 120)),
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.config.api_key}"}

            async with self.session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
        except Exception:
            return False


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, config: LLMProviderConfig, session: aiohttp.ClientSession):
        super().__init__(config)
        self.session = session
        self.base_url = config.base_url or "https://api.anthropic.com"

    async def execute(self, prompt: str, model: str | None = None, **kwargs) -> LLMResponse:
        """Execute a prompt using Anthropic API."""
        start_time = time.time()

        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": model or self.config.model or "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        try:
            async with self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", self.config.timeout)),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                response_time = int((time.time() - start_time) * 1000)

                return LLMResponse(
                    provider=self.name,
                    model=data.get("model", payload["model"]),
                    response_text=data["content"][0]["text"],
                    response_time_ms=response_time,
                    tokens_input=data.get("usage", {}).get("input_tokens", 0),
                    tokens_output=data.get("usage", {}).get("output_tokens", 0),
                    tokens_total=data.get("usage", {}).get("input_tokens", 0)
                    + data.get("usage", {}).get("output_tokens", 0),
                    finish_reason=data.get("stop_reason", ""),
                    success=True,
                    raw_response=data,
                )

        except aiohttp.ClientError as e:
            return LLMResponse(
                provider=self.name,
                model=model or self.config.model,
                response_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e),
            )

    async def execute_streaming(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        """Execute with streaming response."""
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": model or self.config.model or "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        async with self.session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", 120)),
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            text = data.get("delta", {}).get("text", "")
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        """Check Anthropic API health."""
        try:
            # Anthropic doesn't have a dedicated health endpoint
            # We'll try a minimal request
            return True  # Assume healthy if configured
        except Exception:
            return False


class OllamaProvider(LLMProvider):
    """Local Ollama provider."""

    def __init__(self, config: LLMProviderConfig, session: aiohttp.ClientSession):
        super().__init__(config)
        self.session = session
        self.base_url = config.base_url or "http://localhost:11434"

    async def execute(self, prompt: str, model: str | None = None, **kwargs) -> LLMResponse:
        """Execute a prompt using Ollama API."""
        start_time = time.time()

        # Try OpenAI-compatible endpoint first
        try:
            return await self._execute_openai_compatible(prompt, model, start_time, **kwargs)
        except Exception:
            pass

        # Fall back to native Ollama API
        return await self._execute_native(prompt, model, start_time, **kwargs)

    async def _execute_openai_compatible(
        self, prompt: str, model: str | None, start_time: float, **kwargs
    ) -> LLMResponse:
        """Execute using OpenAI-compatible endpoint."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": model or self.config.model or "llama2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        async with self.session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", self.config.timeout)),
        ) as response:
            response.raise_for_status()
            data = await response.json()

            response_time = int((time.time() - start_time) * 1000)

            return LLMResponse(
                provider=self.name,
                model=data.get("model", payload["model"]),
                response_text=data["choices"][0]["message"]["content"],
                response_time_ms=response_time,
                tokens_total=data.get("usage", {}).get("total_tokens", 0),
                success=True,
                raw_response=data,
            )

    async def _execute_native(
        self, prompt: str, model: str | None, start_time: float, **kwargs
    ) -> LLMResponse:
        """Execute using native Ollama API."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model or self.config.model or "llama2",
            "prompt": prompt,
            "stream": False,
        }

        try:
            async with self.session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", self.config.timeout)),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                response_time = int((time.time() - start_time) * 1000)

                return LLMResponse(
                    provider=self.name,
                    model=data.get("model", payload["model"]),
                    response_text=data.get("response", ""),
                    response_time_ms=response_time,
                    tokens_total=data.get("eval_count", 0),
                    success=True,
                    raw_response=data,
                )

        except aiohttp.ClientError as e:
            return LLMResponse(
                provider=self.name,
                model=model or self.config.model,
                response_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e),
            )

    async def execute_streaming(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        """Execute with streaming response."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model or self.config.model or "llama2",
            "prompt": prompt,
            "stream": True,
        }

        async with self.session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", 120))
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                try:
                    data = json.loads(line.decode("utf-8"))
                    text = data.get("response", "")
                    if text:
                        yield text
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> bool:
        """Check Ollama health."""
        try:
            url = f"{self.base_url}/api/tags"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
        except Exception:
            return False


class GoogleProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, config: LLMProviderConfig, session: aiohttp.ClientSession):
        super().__init__(config)
        self.session = session
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"

    async def execute(self, prompt: str, model: str | None = None, **kwargs) -> LLMResponse:
        """Execute a prompt using Google Gemini API."""
        start_time = time.time()

        model_name = model or self.config.model or "gemini-pro"
        url = f"{self.base_url}/models/{model_name}:generateContent?key={self.config.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }

        try:
            async with self.session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", self.config.timeout)),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                response_time = int((time.time() - start_time) * 1000)

                # Extract text from response
                text = ""
                if data.get("candidates"):
                    content = data["candidates"][0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")

                return LLMResponse(
                    provider=self.name,
                    model=model_name,
                    response_text=text,
                    response_time_ms=response_time,
                    success=True,
                    raw_response=data,
                )

        except aiohttp.ClientError as e:
            return LLMResponse(
                provider=self.name,
                model=model_name,
                response_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e),
            )

    async def execute_streaming(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        """Execute with streaming response."""
        model_name = model or self.config.model or "gemini-pro"
        url = f"{self.base_url}/models/{model_name}:streamGenerateContent?key={self.config.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }

        async with self.session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", 120))
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if data.get("candidates"):
                        content = data["candidates"][0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            if text:
                                yield text
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> bool:
        """Check Google API health."""
        try:
            url = f"{self.base_url}/models?key={self.config.api_key}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                return response.status == 200
        except Exception:
            return False


class EnhancedExecutionAgent(BaseAgent):
    """
    Enhanced Execution Agent with multi-provider support.

    Features:
    - Multiple LLM provider support (OpenAI, Anthropic, Ollama, Google)
    - Circuit breaker pattern for resilience
    - Rate limiting per provider
    - Streaming support
    - Batch execution
    - Automatic retries with exponential backoff
    - Provider health monitoring
    """

    def __init__(
        self,
        config: Config,
        message_queue: MessageQueue,
        event_bus: EventBus | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(
            agent_type=AgentType.EXECUTOR,
            config=config,
            message_queue=message_queue,
            agent_id=agent_id,
        )

        self.event_bus = event_bus

        # HTTP session
        self._session: aiohttp.ClientSession | None = None

        # Providers
        self._providers: dict[str, LLMProvider] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Rate limiting
        self._rate_limiters: dict[str, asyncio.Semaphore] = {}
        self._last_request_time: dict[str, float] = {}

        # Metrics
        self._execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_tokens": 0,
            "total_response_time_ms": 0,
            "average_response_time_ms": 0,
            "executions_by_provider": {},
            "errors_by_provider": {},
        }

    async def on_start(self):
        """Initialize providers and HTTP session."""
        self._session = aiohttp.ClientSession()

        # Initialize providers
        for name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue

            provider = self._create_provider(name, provider_config)
            if provider:
                self._providers[name] = provider
                self._circuit_breakers[name] = CircuitBreaker(provider=name)

                # Initialize rate limiter
                rate_per_second = max(1, provider_config.rate_limit // 60)
                self._rate_limiters[name] = asyncio.Semaphore(rate_per_second)
                self._last_request_time[name] = 0

                # Initialize metrics
                self._execution_metrics["executions_by_provider"][name] = 0
                self._execution_metrics["errors_by_provider"][name] = 0

        logger.info(f"Initialized {len(self._providers)} LLM providers")

    async def on_stop(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()

    def _create_provider(self, name: str, config: LLMProviderConfig) -> LLMProvider | None:
        """Create a provider instance based on name."""
        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "local": OllamaProvider,
            "ollama": OllamaProvider,
            "google": GoogleProvider,
            "gemini": GoogleProvider,
        }

        provider_class = provider_map.get(name.lower())
        if provider_class:
            return provider_class(config, self._session)

        # Default to OpenAI-compatible
        return OpenAIProvider(config, self._session)

    async def process_message(self, message: Message):
        """Process incoming execution requests."""
        if message.type == MessageType.EXECUTE_REQUEST:
            await self._handle_execute_request(message)
        elif message.type == MessageType.STATUS_UPDATE:
            await self.send_message(
                MessageType.STATUS_UPDATE,
                target=message.source,
                job_id=message.job_id,
                payload={
                    "status": self.status.to_dict(),
                    "execution_metrics": self._execution_metrics,
                    "provider_health": await self._get_provider_health(),
                },
            )

    async def _handle_execute_request(self, message: Message):
        """Handle an execution request."""
        job_id = message.job_id
        self.add_active_job(job_id)

        try:
            prompt_id = message.payload.get("prompt_id", "")
            prompt_text = message.payload.get("prompt_text", "")
            provider_name = message.payload.get(
                "provider", message.payload.get("target_llm", "local")
            )
            model = message.payload.get("model")
            timeout = message.payload.get("timeout", 60)
            retry_count = message.payload.get("retry_count", 3)

            # Map target_llm to provider name
            provider_name = self._map_target_to_provider(provider_name)

            # Execute prompt
            response = await self.execute_prompt(
                prompt_text=prompt_text,
                provider_name=provider_name,
                model=model,
                timeout=timeout,
                max_retries=retry_count,
            )
            response.prompt_id = prompt_id

            # Emit event
            if self.event_bus:
                await self.event_bus.publish(
                    SystemEvent(
                        type=EventType.EXECUTION_COMPLETED,
                        source=self.agent_id,
                        job_id=job_id,
                        data={"response_id": response.id, "success": response.success},
                    )
                )

            # Send response
            await self.send_message(
                MessageType.EXECUTE_RESPONSE,
                target=AgentType.ORCHESTRATOR,
                job_id=job_id,
                payload=response.to_dict(),
                priority=7,
            )

        except Exception as e:
            logger.error(f"Execution error for job {job_id}: {e}")

            error_response = LLMResponse(
                prompt_id=message.payload.get("prompt_id", ""),
                provider=message.payload.get("provider", "unknown"),
                success=False,
                error_message=str(e),
            )

            await self.send_message(
                MessageType.EXECUTE_RESPONSE,
                target=AgentType.ORCHESTRATOR,
                job_id=job_id,
                payload=error_response.to_dict(),
                priority=7,
            )

        finally:
            self.remove_active_job(job_id)

    def _map_target_to_provider(self, target: str) -> str:
        """Map target LLM identifier to provider name."""
        mapping = {
            "openai_gpt4": "openai",
            "openai_gpt35": "openai",
            "anthropic_claude": "anthropic",
            "local_ollama": "local",
            "google_gemini": "google",
            "custom": "local",
        }
        return mapping.get(target, target)

    async def execute_prompt(
        self,
        prompt_text: str,
        provider_name: str = "local",
        model: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs,
    ) -> LLMResponse:
        """
        Execute a prompt against a provider.

        Args:
            prompt_text: The prompt to execute
            provider_name: Name of the provider to use
            model: Specific model to use (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with the result
        """
        self._execution_metrics["total_executions"] += 1

        # Get provider
        provider = self._providers.get(provider_name)
        if not provider:
            # Try to find any available provider
            for name, p in self._providers.items():
                if self._circuit_breakers[name].can_execute():
                    provider = p
                    provider_name = name
                    break

        if not provider:
            self._execution_metrics["failed_executions"] += 1
            return LLMResponse(
                provider=provider_name,
                success=False,
                error_message=f"Provider {provider_name} not available",
            )

        # Check circuit breaker
        circuit_breaker = self._circuit_breakers.get(provider_name)
        if circuit_breaker and not circuit_breaker.can_execute():
            self._execution_metrics["failed_executions"] += 1
            return LLMResponse(
                provider=provider_name,
                success=False,
                error_message=f"Provider {provider_name} circuit breaker is open",
            )

        # Execute with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                await self._apply_rate_limit(provider_name)

                # Execute
                response = await provider.execute(
                    prompt_text, model=model, timeout=timeout, **kwargs
                )

                if response.success:
                    # Record success
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    self._execution_metrics["successful_executions"] += 1
                    self._execution_metrics["executions_by_provider"][provider_name] = (
                        self._execution_metrics["executions_by_provider"].get(provider_name, 0) + 1
                    )
                    self._execution_metrics["total_tokens"] += response.tokens_total
                    self._execution_metrics["total_response_time_ms"] += response.response_time_ms
                    self._update_average_response_time()

                    response.retry_count = attempt
                    return response
                else:
                    last_error = response.error_message

            except TimeoutError:
                last_error = "Request timed out"
                logger.warning(f"Timeout on attempt {attempt + 1} for provider {provider_name}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error on attempt {attempt + 1}: {e}")

            # Exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

        # All retries failed
        if circuit_breaker:
            circuit_breaker.record_failure()

        self._execution_metrics["failed_executions"] += 1
        self._execution_metrics["errors_by_provider"][provider_name] = (
            self._execution_metrics["errors_by_provider"].get(provider_name, 0) + 1
        )

        return LLMResponse(
            provider=provider_name,
            model=model,
            success=False,
            error_message=last_error,
            retry_count=max_retries,
        )

    async def execute_streaming(
        self,
        prompt_text: str,
        provider_name: str = "local",
        model: str | None = None,
        callback: Callable[[str], None] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Execute a prompt with streaming response.

        Args:
            prompt_text: The prompt to execute
            provider_name: Name of the provider to use
            model: Specific model to use
            callback: Callback function for each chunk
            **kwargs: Additional arguments

        Returns:
            LLMResponse with complete response
        """
        provider = self._providers.get(provider_name)
        if not provider:
            return LLMResponse(
                provider=provider_name,
                success=False,
                error_message=f"Provider {provider_name} not available",
            )

        start_time = time.time()
        full_response = []

        try:
            async for chunk in provider.execute_streaming(prompt_text, model, **kwargs):
                full_response.append(chunk)
                if callback:
                    callback(chunk)

            response_time = int((time.time() - start_time) * 1000)

            return LLMResponse(
                provider=provider_name,
                model=model,
                response_text="".join(full_response),
                response_time_ms=response_time,
                success=True,
                streaming=True,
            )

        except Exception as e:
            return LLMResponse(
                provider=provider_name,
                model=model,
                response_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e),
                streaming=True,
            )

    async def execute_batch(
        self,
        prompts: list[tuple[str, str]],  # List of (prompt_id, prompt_text)
        provider_name: str = "local",
        model: str | None = None,
        max_concurrent: int = 5,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Execute multiple prompts concurrently.

        Args:
            prompts: List of (prompt_id, prompt_text) tuples
            provider_name: Provider to use
            model: Model to use
            max_concurrent: Maximum concurrent executions
            **kwargs: Additional arguments

        Returns:
            List of LLMResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(prompt_id: str, prompt_text: str) -> LLMResponse:
            async with semaphore:
                response = await self.execute_prompt(
                    prompt_text=prompt_text, provider_name=provider_name, model=model, **kwargs
                )
                response.prompt_id = prompt_id
                return response

        tasks = [
            execute_with_semaphore(prompt_id, prompt_text) for prompt_id, prompt_text in prompts
        ]

        return await asyncio.gather(*tasks)

    async def _apply_rate_limit(self, provider_name: str):
        """Apply rate limiting for a provider."""
        if provider_name not in self._rate_limiters:
            return

        semaphore = self._rate_limiters[provider_name]
        provider_config = self.config.providers.get(provider_name)

        if not provider_config:
            return

        async with semaphore:
            # Ensure minimum time between requests
            last_time = self._last_request_time.get(provider_name, 0)
            min_interval = 60.0 / max(1, provider_config.rate_limit)

            elapsed = time.time() - last_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_request_time[provider_name] = time.time()

    def _update_average_response_time(self):
        """Update the average response time metric."""
        successful = self._execution_metrics["successful_executions"]
        if successful > 0:
            self._execution_metrics["average_response_time_ms"] = (
                self._execution_metrics["total_response_time_ms"] / successful
            )

    async def _get_provider_health(self) -> dict[str, Any]:
        """Get health status of all providers."""
        health = {}

        for name, provider in self._providers.items():
            circuit_breaker = self._circuit_breakers.get(name)

            health[name] = {
                "available": circuit_breaker.can_execute() if circuit_breaker else True,
                "circuit_state": circuit_breaker.state.value if circuit_breaker else "unknown",
                "failure_count": circuit_breaker.failure_count if circuit_breaker else 0,
                "healthy": await provider.health_check(),
            }

        return health

    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        available = []
        for name in self._providers:
            circuit_breaker = self._circuit_breakers.get(name)
            if circuit_breaker and circuit_breaker.can_execute():
                available.append(name)
        return available

    def get_execution_metrics(self) -> dict[str, Any]:
        """Get execution metrics."""
        return self._execution_metrics.copy()
