"""
Provider Manager with Failover Support

Story 1.2: Direct API Integration
Provides centralized management of LLM providers with automatic failover.

Features:
- Provider registration and health tracking
- Automatic failover to backup providers
- Provider priority ordering
- Rate limit tracking per provider
- Health check scheduling
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.config import get_settings
from app.domain.interfaces import LLMProvider
from app.domain.models import PromptRequest, PromptResponse
from app.infrastructure.retry_handler import (
    RetryConfig,
    RetryExhaustedError,
    RetryHandler,
    get_provider_retry_config,
)

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider availability status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


@dataclass
class ProviderState:
    """Tracks the state of a provider."""

    name: str
    status: ProviderStatus = ProviderStatus.UNKNOWN
    priority: int = 0
    last_health_check: float | None = None
    last_success: float | None = None
    last_failure: float | None = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    rate_limit_reset: float | None = None
    average_latency_ms: float = 0

    # Rate limit tracking
    requests_this_window: int = 0
    window_start_time: float = field(default_factory=time.time)
    rate_limit_window: int = 60  # seconds
    rate_limit_max_requests: int = 60  # per window


@dataclass
class FailoverConfig:
    """Configuration for provider failover behavior."""

    enabled: bool = True
    max_failover_attempts: int = 3
    failover_on_rate_limit: bool = True
    failover_on_timeout: bool = True
    failover_on_error: bool = True

    # Health check settings
    health_check_interval: int = 60  # seconds
    unhealthy_threshold: int = 3  # consecutive failures
    recovery_threshold: int = 2  # consecutive successes to recover

    # Rate limit settings
    rate_limit_cooldown: int = 30  # seconds before retrying rate-limited


class ProviderManager:
    """
    Centralized provider management with failover support.

    Manages multiple LLM providers, tracks their health,
    and provides automatic failover when primary providers fail.
    """

    def __init__(self, failover_config: FailoverConfig | None = None):
        self.config = failover_config or FailoverConfig()
        self._providers: dict[str, LLMProvider] = {}
        self._provider_states: dict[str, ProviderState] = {}
        self._retry_handlers: dict[str, RetryHandler] = {}
        self._default_provider: str | None = None
        self._provider_order: list[str] = []
        self._lock = asyncio.Lock()
        self._health_check_task: asyncio.Task | None = None

    def register_provider(
        self,
        name: str,
        provider: LLMProvider,
        priority: int = 0,
        is_default: bool = False,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """
        Register a provider with the manager.

        Args:
            name: Unique name for the provider.
            provider: The LLMProvider implementation.
            priority: Priority for failover (higher = tried first).
            is_default: Whether this is the default provider.
            retry_config: Custom retry configuration.
        """
        self._providers[name] = provider
        self._provider_states[name] = ProviderState(
            name=name,
            priority=priority,
        )

        # Set up retry handler
        config = retry_config or get_provider_retry_config(name)
        self._retry_handlers[name] = RetryHandler(config)

        if is_default:
            self._default_provider = name

        # Update provider order based on priority
        self._update_provider_order()

        logger.info(
            f"Registered provider '{name}' with priority {priority} "
            f"(default={is_default})"
        )

    def _update_provider_order(self) -> None:
        """Update the provider order based on priority and status."""
        self._provider_order = sorted(
            self._providers.keys(),
            key=lambda n: (
                -self._provider_states[n].priority,
                n != self._default_provider,
            ),
        )

    def get_provider(self, name: str | None = None) -> LLMProvider | None:
        """Get a specific provider or the default."""
        if name:
            return self._providers.get(name)
        return self._providers.get(self._default_provider or "")

    def get_provider_state(self, name: str) -> ProviderState | None:
        """Get the current state of a provider."""
        return self._provider_states.get(name)

    def get_available_providers(self) -> list[str]:
        """Get list of available (healthy) providers in priority order."""
        available = []

        for name in self._provider_order:
            state = self._provider_states[name]
            if state.status in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED):
                if not self._is_rate_limited(name):
                    available.append(name)

        return available

    def _is_rate_limited(self, name: str) -> bool:
        """Check if a provider is currently rate limited."""
        state = self._provider_states.get(name)
        if not state:
            return False

        if state.status == ProviderStatus.RATE_LIMITED:
            if state.rate_limit_reset and time.time() < state.rate_limit_reset:
                return True
            # Rate limit has expired
            state.status = ProviderStatus.DEGRADED

        return False

    def _should_failover(self, error: Exception) -> bool:
        """Determine if we should failover based on the error."""
        if not self.config.enabled:
            return False

        error_str = str(error).lower()

        if "rate limit" in error_str or "429" in error_str:
            return self.config.failover_on_rate_limit

        if "timeout" in error_str:
            return self.config.failover_on_timeout

        return self.config.failover_on_error

    async def _record_success(self, name: str, latency_ms: float) -> None:
        """Record a successful request for a provider."""
        async with self._lock:
            state = self._provider_states.get(name)
            if not state:
                return

            state.last_success = time.time()
            state.total_requests += 1
            state.consecutive_failures = 0
            state.requests_this_window += 1

            # Update average latency
            if state.average_latency_ms == 0:
                state.average_latency_ms = latency_ms
            else:
                # Exponential moving average
                state.average_latency_ms = (
                    0.9 * state.average_latency_ms + 0.1 * latency_ms
                )

            # Check if healthy
            if state.status != ProviderStatus.HEALTHY:
                state.status = ProviderStatus.HEALTHY
                logger.info(f"Provider '{name}' recovered to HEALTHY status")

    async def _record_failure(
        self,
        name: str,
        error: Exception,
        is_rate_limit: bool = False,
    ) -> None:
        """Record a failed request for a provider."""
        async with self._lock:
            state = self._provider_states.get(name)
            if not state:
                return

            state.last_failure = time.time()
            state.total_requests += 1
            state.total_failures += 1
            state.consecutive_failures += 1

            if is_rate_limit:
                state.status = ProviderStatus.RATE_LIMITED
                state.rate_limit_reset = (
                    time.time() + self.config.rate_limit_cooldown
                )
                logger.warning(
                    f"Provider '{name}' rate limited. "
                    f"Cooldown until {state.rate_limit_reset}"
                )
            elif state.consecutive_failures >= self.config.unhealthy_threshold:
                state.status = ProviderStatus.UNHEALTHY
                logger.error(
                    f"Provider '{name}' marked UNHEALTHY after "
                    f"{state.consecutive_failures} consecutive failures"
                )
            else:
                state.status = ProviderStatus.DEGRADED

    async def generate_with_failover(
        self,
        request: PromptRequest,
        preferred_provider: str | None = None,
    ) -> PromptResponse:
        """
        Generate text with automatic failover to backup providers.

        Args:
            request: The prompt request.
            preferred_provider: Preferred provider to try first.

        Returns:
            PromptResponse from the first successful provider.

        Raises:
            RetryExhaustedError: When all providers fail.
        """
        # Build provider order for this request
        providers_to_try = []

        if preferred_provider and preferred_provider in self._providers:
            providers_to_try.append(preferred_provider)

        # Add available providers in priority order
        for name in self.get_available_providers():
            if name not in providers_to_try:
                providers_to_try.append(name)

        # Add remaining providers as last resort
        for name in self._provider_order:
            if name not in providers_to_try:
                providers_to_try.append(name)

        if not providers_to_try:
            raise RetryExhaustedError(
                message="No providers available",
                attempts=0,
            )

        # Limit failover attempts
        providers_to_try = providers_to_try[:self.config.max_failover_attempts]

        last_error: Exception | None = None

        for provider_name in providers_to_try:
            provider = self._providers[provider_name]
            retry_handler = self._retry_handlers.get(
                provider_name,
                RetryHandler(),
            )

            try:
                logger.info(f"Attempting generation with provider '{provider_name}'")
                start_time = time.time()

                response = await retry_handler.execute_with_retry(
                    provider.generate,
                    request,
                    provider=provider_name,
                )

                latency_ms = (time.time() - start_time) * 1000
                await self._record_success(provider_name, latency_ms)

                return response

            except Exception as e:
                last_error = e
                is_rate_limit = "rate limit" in str(e).lower() or "429" in str(e)
                await self._record_failure(provider_name, e, is_rate_limit)

                if not self._should_failover(e):
                    raise

                logger.warning(
                    f"Provider '{provider_name}' failed: {e!s}. "
                    f"Trying next provider..."
                )

        raise RetryExhaustedError(
            message=f"All {len(providers_to_try)} providers failed",
            attempts=len(providers_to_try),
            last_error=last_error,
        )

    async def check_provider_health(self, name: str) -> ProviderStatus:
        """
        Check the health of a specific provider.

        Args:
            name: The provider name.

        Returns:
            The current health status of the provider.
        """
        provider = self._providers.get(name)
        if not provider:
            return ProviderStatus.UNKNOWN

        state = self._provider_states[name]

        try:
            is_healthy = await provider.check_health()
            state.last_health_check = time.time()

            if is_healthy:
                if state.status == ProviderStatus.UNHEALTHY:
                    state.status = ProviderStatus.DEGRADED
                elif state.status in (
                    ProviderStatus.DEGRADED,
                    ProviderStatus.UNKNOWN,
                ):
                    state.status = ProviderStatus.HEALTHY
                state.consecutive_failures = 0
            else:
                state.consecutive_failures += 1
                if state.consecutive_failures >= self.config.unhealthy_threshold:
                    state.status = ProviderStatus.UNHEALTHY
                else:
                    state.status = ProviderStatus.DEGRADED

            return state.status

        except Exception as e:
            logger.error(f"Health check failed for '{name}': {e!s}")
            state.status = ProviderStatus.UNHEALTHY
            state.last_health_check = time.time()
            return state.status

    async def check_all_providers_health(self) -> dict[str, ProviderStatus]:
        """Check health of all registered providers."""
        results = {}

        for name in self._providers:
            results[name] = await self.check_provider_health(name)

        return results

    async def start_health_checks(self) -> None:
        """Start periodic health checks for all providers."""
        if self._health_check_task and not self._health_check_task.done():
            return

        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )
        logger.info("Started provider health check loop")

    async def stop_health_checks(self) -> None:
        """Stop the health check loop."""
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task
            self._health_check_task = None
            logger.info("Stopped provider health check loop")

    async def _health_check_loop(self) -> None:
        """Background loop for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.check_all_providers_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e!s}")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all providers."""
        stats = {
            "providers": {},
            "default_provider": self._default_provider,
            "provider_order": self._provider_order,
            "failover_enabled": self.config.enabled,
        }

        for name, state in self._provider_states.items():
            stats["providers"][name] = {
                "status": state.status.value,
                "priority": state.priority,
                "total_requests": state.total_requests,
                "total_failures": state.total_failures,
                "consecutive_failures": state.consecutive_failures,
                "average_latency_ms": round(state.average_latency_ms, 2),
                "last_success": state.last_success,
                "last_failure": state.last_failure,
                "rate_limited": self._is_rate_limited(name),
            }

        return stats


# Global provider manager instance
_provider_manager: ProviderManager | None = None


def get_provider_manager() -> ProviderManager:
    """Get or create the global provider manager."""
    global _provider_manager

    if _provider_manager is None:
        _provider_manager = ProviderManager()

    return _provider_manager


async def initialize_provider_manager() -> ProviderManager:
    """
    Initialize the provider manager with all configured providers.

    This should be called during application startup.
    """
    manager = get_provider_manager()
    settings = get_settings()

    # Import providers dynamically to avoid circular imports
    from app.infrastructure.anthropic_client import AnthropicClient
    from app.infrastructure.cursor_client import CursorClient
    from app.infrastructure.deepseek_client import DeepSeekClient
    from app.infrastructure.gemini_client import GeminiClient
    from app.infrastructure.openai_client import OpenAIClient
    from app.infrastructure.qwen_client import QwenClient

    # Register providers based on configuration
    providers_config = [
        ("gemini", GeminiClient, settings.GOOGLE_API_KEY, 100, True),
        ("openai", OpenAIClient, settings.OPENAI_API_KEY, 90, False),
        ("anthropic", AnthropicClient, settings.ANTHROPIC_API_KEY, 80, False),
        ("deepseek", DeepSeekClient, settings.DEEPSEEK_API_KEY, 70, False),
        ("qwen", QwenClient, settings.QWEN_API_KEY, 60, False),
        ("cursor", CursorClient, settings.CURSOR_API_KEY, 50, False),
    ]

    for name, client_class, api_key, priority, is_default in providers_config:
        if api_key:
            try:
                provider = client_class(settings)
                manager.register_provider(
                    name=name,
                    provider=provider,
                    priority=priority,
                    is_default=is_default,
                )
            except Exception as e:
                logger.warning(f"Failed to register provider '{name}': {e!s}")

    # Start health checks
    await manager.start_health_checks()

    return manager
