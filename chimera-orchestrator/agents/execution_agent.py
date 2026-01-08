"""
Execution Agent - Deploys prompts against target LLMs
"""

import asyncio
import logging
import time
from typing import Any

import aiohttp

try:
    from core.config import Config, LLMProviderConfig
    from core.message_queue import MessageQueue
    from core.models import (
        AgentType,
        ExecutionRequest,
        ExecutionResult,
        Message,
        MessageType,
        TargetLLM,
    )

    from agents.base_agent import BaseAgent
except ImportError:
    from ..core.config import Config, LLMProviderConfig
    from ..core.message_queue import MessageQueue
    from ..core.models import (
        AgentType,
        ExecutionRequest,
        ExecutionResult,
        Message,
        MessageType,
        TargetLLM,
    )
    from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """
    Agent responsible for executing prompts against target LLMs.

    Features:
    - Multi-provider support (OpenAI, Anthropic, Local)
    - Async execution with rate limiting
    - Automatic retries with exponential backoff
    - Response capture and timing
    """

    def __init__(self, config: Config, message_queue: MessageQueue, agent_id: str | None = None):
        super().__init__(
            agent_type=AgentType.EXECUTOR,
            config=config,
            message_queue=message_queue,
            agent_id=agent_id,
        )

        # Rate limiting
        self._rate_limiters: dict[str, asyncio.Semaphore] = {}
        self._last_request_time: dict[str, float] = {}

        # HTTP session
        self._session: aiohttp.ClientSession | None = None

        # Execution metrics
        self._execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_tokens": 0,
            "average_response_time_ms": 0,
        }

    async def on_start(self):
        """Initialize HTTP session and rate limiters."""
        self._session = aiohttp.ClientSession()

        # Initialize rate limiters for each provider
        for name, provider in self.config.providers.items():
            if provider.enabled:
                # Semaphore for concurrent requests
                self._rate_limiters[name] = asyncio.Semaphore(
                    provider.rate_limit // 60 or 1  # requests per second
                )
                self._last_request_time[name] = 0

    async def on_stop(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()

    async def process_message(self, message: Message):
        """Process incoming execution requests."""
        if message.type == MessageType.EXECUTE_REQUEST:
            await self._handle_execute_request(message)
        elif message.type == MessageType.STATUS_UPDATE:
            # Handle status queries
            await self.send_message(
                MessageType.STATUS_UPDATE,
                target=message.source,
                job_id=message.job_id,
                payload={
                    "status": self.status.to_dict(),
                    "execution_metrics": self._execution_metrics,
                },
            )

    async def _handle_execute_request(self, message: Message):
        """Handle an execution request."""
        job_id = message.job_id
        self.add_active_job(job_id)

        try:
            # Parse request
            request = ExecutionRequest(
                prompt_id=message.payload.get("prompt_id", ""),
                prompt_text=message.payload.get("prompt_text", ""),
                target_llm=TargetLLM(message.payload.get("target_llm", "openai_gpt4")),
                model_config=message.payload.get("model_config", {}),
                timeout=message.payload.get("timeout", 60),
                retry_count=message.payload.get("retry_count", 3),
            )

            # Execute prompt
            result = await self.execute_prompt(request)

            # Send response
            await self.send_message(
                MessageType.EXECUTE_RESPONSE,
                target=AgentType.ORCHESTRATOR,
                job_id=job_id,
                payload=result.to_dict(),
                priority=7,
            )

        except Exception as e:
            logger.error(f"Execution error for job {job_id}: {e}")

            # Send error response
            error_result = ExecutionResult(
                prompt_id=message.payload.get("prompt_id", ""),
                target_llm=message.payload.get("target_llm", ""),
                success=False,
                error_message=str(e),
            )

            await self.send_message(
                MessageType.EXECUTE_RESPONSE,
                target=AgentType.ORCHESTRATOR,
                job_id=job_id,
                payload=error_result.to_dict(),
                priority=7,
            )

        finally:
            self.remove_active_job(job_id)

    async def execute_prompt(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute a prompt against the target LLM.

        Args:
            request: The execution request

        Returns:
            ExecutionResult with response and metrics
        """
        self._execution_metrics["total_executions"] += 1
        start_time = time.time()

        # Get provider config
        provider_name = self._get_provider_name(request.target_llm)
        provider = self.config.providers.get(provider_name)

        if not provider or not provider.enabled:
            return ExecutionResult(
                prompt_id=request.prompt_id,
                target_llm=request.target_llm.value,
                success=False,
                error_message=f"Provider {provider_name} not configured or disabled",
            )

        # Execute with retries
        last_error = None
        for attempt in range(request.retry_count):
            try:
                # Rate limiting
                await self._apply_rate_limit(provider_name)

                # Execute based on provider type
                if request.target_llm in [TargetLLM.OPENAI_GPT4, TargetLLM.OPENAI_GPT35]:
                    response = await self._execute_openai(request, provider)
                elif request.target_llm == TargetLLM.ANTHROPIC_CLAUDE:
                    response = await self._execute_anthropic(request, provider)
                elif request.target_llm == TargetLLM.LOCAL_OLLAMA:
                    response = await self._execute_local(request, provider)
                else:
                    response = await self._execute_custom(request, provider)

                # Calculate metrics
                response_time_ms = int((time.time() - start_time) * 1000)

                self._execution_metrics["successful_executions"] += 1
                self._execution_metrics["total_tokens"] += response.get("tokens", 0)
                self._update_average_response_time(response_time_ms)

                return ExecutionResult(
                    prompt_id=request.prompt_id,
                    target_llm=request.target_llm.value,
                    response_text=response.get("text", ""),
                    response_time_ms=response_time_ms,
                    tokens_used=response.get("tokens", 0),
                    success=True,
                    raw_response=response,
                )

            except TimeoutError:
                last_error = "Request timed out"
                logger.warning(f"Timeout on attempt {attempt + 1} for {request.prompt_id}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error on attempt {attempt + 1}: {e}")

            # Exponential backoff
            if attempt < request.retry_count - 1:
                await asyncio.sleep(2**attempt)

        # All retries failed
        self._execution_metrics["failed_executions"] += 1

        return ExecutionResult(
            prompt_id=request.prompt_id,
            target_llm=request.target_llm.value,
            response_time_ms=int((time.time() - start_time) * 1000),
            success=False,
            error_message=last_error,
        )

    async def _execute_openai(
        self, request: ExecutionRequest, provider: LLMProviderConfig
    ) -> dict[str, Any]:
        """Execute prompt using OpenAI API."""
        url = f"{provider.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        model = request.model_config.get("model", provider.model)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt_text}],
            "max_tokens": request.model_config.get("max_tokens", provider.max_tokens),
            "temperature": request.model_config.get("temperature", provider.temperature),
        }

        async with self._session.post(
            url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=request.timeout)
        ) as response:
            response.raise_for_status()
            data = await response.json()

            return {
                "text": data["choices"][0]["message"]["content"],
                "tokens": data.get("usage", {}).get("total_tokens", 0),
                "model": data.get("model", model),
                "finish_reason": data["choices"][0].get("finish_reason"),
            }

    async def _execute_anthropic(
        self, request: ExecutionRequest, provider: LLMProviderConfig
    ) -> dict[str, Any]:
        """Execute prompt using Anthropic API."""
        url = f"{provider.base_url}/v1/messages"

        headers = {
            "x-api-key": provider.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        model = request.model_config.get("model", provider.model)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt_text}],
            "max_tokens": request.model_config.get("max_tokens", provider.max_tokens),
        }

        async with self._session.post(
            url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=request.timeout)
        ) as response:
            response.raise_for_status()
            data = await response.json()

            return {
                "text": data["content"][0]["text"],
                "tokens": data.get("usage", {}).get("input_tokens", 0)
                + data.get("usage", {}).get("output_tokens", 0),
                "model": data.get("model", model),
                "stop_reason": data.get("stop_reason"),
            }

    async def _execute_local(
        self, request: ExecutionRequest, provider: LLMProviderConfig
    ) -> dict[str, Any]:
        """Execute prompt using local/Ollama API."""
        # Try OpenAI-compatible endpoint first
        url = f"{provider.base_url}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        model = request.model_config.get("model", provider.model)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt_text}],
            "max_tokens": request.model_config.get("max_tokens", provider.max_tokens),
            "temperature": request.model_config.get("temperature", provider.temperature),
        }

        try:
            async with self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # Handle OpenAI-compatible response
                if "choices" in data:
                    return {
                        "text": data["choices"][0]["message"]["content"],
                        "tokens": data.get("usage", {}).get("total_tokens", 0),
                        "model": data.get("model", model),
                    }
                # Handle direct response
                return {
                    "text": data.get("response", data.get("content", str(data))),
                    "tokens": 0,
                    "model": model,
                }
        except aiohttp.ClientError:
            # Try Ollama native endpoint
            return await self._execute_ollama_native(request, provider)

    async def _execute_ollama_native(
        self, request: ExecutionRequest, provider: LLMProviderConfig
    ) -> dict[str, Any]:
        """Execute prompt using Ollama native API."""
        url = f"{provider.base_url}/api/generate"

        payload = {
            "model": request.model_config.get("model", provider.model),
            "prompt": request.prompt_text,
            "stream": False,
        }

        async with self._session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=request.timeout)
        ) as response:
            response.raise_for_status()
            data = await response.json()

            return {
                "text": data.get("response", ""),
                "tokens": data.get("eval_count", 0),
                "model": data.get("model", ""),
            }

    async def _execute_custom(
        self, request: ExecutionRequest, provider: LLMProviderConfig
    ) -> dict[str, Any]:
        """Execute prompt using custom API endpoint."""
        url = provider.base_url

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": request.prompt_text,
            "model": request.model_config.get("model", provider.model),
            "max_tokens": request.model_config.get("max_tokens", provider.max_tokens),
            **request.model_config,
        }

        async with self._session.post(
            url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=request.timeout)
        ) as response:
            response.raise_for_status()
            data = await response.json()

            # Try to extract response from common formats
            text = (
                data.get("response")
                or data.get("text")
                or data.get("content")
                or data.get("output")
                or str(data)
            )

            return {
                "text": text,
                "tokens": data.get("tokens", 0),
                "model": data.get("model", "custom"),
            }

    def _get_provider_name(self, target_llm: TargetLLM) -> str:
        """Map target LLM to provider name."""
        mapping = {
            TargetLLM.OPENAI_GPT4: "openai",
            TargetLLM.OPENAI_GPT35: "openai",
            TargetLLM.ANTHROPIC_CLAUDE: "anthropic",
            TargetLLM.LOCAL_OLLAMA: "local",
            TargetLLM.CUSTOM: "custom",
        }
        return mapping.get(target_llm, "local")

    async def _apply_rate_limit(self, provider_name: str):
        """Apply rate limiting for a provider."""
        if provider_name not in self._rate_limiters:
            return

        semaphore = self._rate_limiters[provider_name]

        async with semaphore:
            # Ensure minimum time between requests
            last_time = self._last_request_time.get(provider_name, 0)
            min_interval = 1.0 / (self.config.providers[provider_name].rate_limit / 60)

            elapsed = time.time() - last_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_request_time[provider_name] = time.time()

    def _update_average_response_time(self, new_time_ms: int):
        """Update rolling average response time."""
        current_avg = self._execution_metrics["average_response_time_ms"]
        total = self._execution_metrics["successful_executions"]

        if total == 1:
            self._execution_metrics["average_response_time_ms"] = new_time_ms
        else:
            # Rolling average
            self._execution_metrics["average_response_time_ms"] = (
                current_avg * (total - 1) + new_time_ms
            ) / total

    async def execute_batch(
        self, requests: list[ExecutionRequest], max_concurrent: int = 5
    ) -> list[ExecutionResult]:
        """
        Execute multiple prompts concurrently.

        Args:
            requests: List of execution requests
            max_concurrent: Maximum concurrent executions

        Returns:
            List of execution results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(request: ExecutionRequest) -> ExecutionResult:
            async with semaphore:
                return await self.execute_prompt(request)

        tasks = [execute_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks)
