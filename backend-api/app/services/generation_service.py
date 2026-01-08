import logging
import time
from typing import Any

from app.core.service_registry import get_ai_config_manager
from app.core.unified_errors import GenerationError
from app.domain.interfaces import LLMProvider
from app.domain.models import PromptRequest, PromptResponse

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Core generation service with AI config integration.

    Integrates with AIConfigManager for:
    - Config-driven provider/model selection
    - Streaming support validation from config capabilities
    - Config-driven retry logic
    - Generation cost tracking
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._config_manager = None
        self._cost_tracker: dict[str, float] = {}
        self._generation_count = 0
        self._retry_count = 0

    def _get_config_manager(self):
        """Get AI config manager with lazy initialization."""
        if self._config_manager is None:
            try:
                self._config_manager = get_ai_config_manager()
            except Exception as e:
                logger.warning(f"Failed to get AI config manager: {e}")
                return None
        return self._config_manager

    def _get_retry_config(self) -> tuple[int, float]:
        """
        Get retry configuration from AI config.

        Returns:
            Tuple of (max_retries, base_delay_seconds)
        """
        config_manager = self._get_config_manager()
        if not config_manager:
            return 3, 1.0  # Default retry config

        try:
            provider_name = getattr(self.provider, "name", None)
            if provider_name:
                provider_config = config_manager.get_provider(provider_name)
                if provider_config:
                    return (
                        provider_config.api.max_retries,
                        1.0  # Base delay, could be added to config
                    )
        except Exception as e:
            logger.debug(f"Could not get retry config: {e}")

        return 3, 1.0

    def validate_streaming_support(self) -> bool:
        """
        Validate that the current provider supports streaming.

        Returns:
            True if streaming is supported
        """
        config_manager = self._get_config_manager()
        if not config_manager:
            return True  # Assume supported if no config

        try:
            provider_name = getattr(self.provider, "name", None)
            if provider_name:
                provider_config = config_manager.get_provider(provider_name)
                if provider_config:
                    return provider_config.capabilities.supports_streaming
        except Exception as e:
            logger.debug(f"Could not validate streaming: {e}")

        return True

    def _track_generation_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        provider_name: str | None = None
    ) -> float | None:
        """
        Track cost for a generation.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider_name: Provider name (uses current if None)

        Returns:
            Cost in USD or None if pricing unavailable
        """
        config_manager = self._get_config_manager()
        if not config_manager:
            return None

        try:
            provider = provider_name or getattr(self.provider, "name", None)
            if not provider:
                return None

            provider_config = config_manager.get_provider(provider)
            if not provider_config:
                return None

            default_model = provider_config.get_default_model()
            if not default_model:
                return None

            cost = config_manager.calculate_cost(
                provider_id=provider,
                model_id=default_model.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Track accumulated cost
            if cost:
                self._cost_tracker[provider] = (
                    self._cost_tracker.get(provider, 0.0) + cost
                )

            return cost
        except Exception as e:
            logger.debug(f"Could not track cost: {e}")
            return None

    async def generate_text(self, request: PromptRequest) -> PromptResponse:
        """
        Generate text with config-driven retry logic and cost tracking.
        """
        self._generation_count += 1
        max_retries, base_delay = self._get_retry_config()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = await self.provider.generate(request)
                generation_time = time.time() - start_time

                # Track cost (estimate tokens from response)
                if hasattr(response, 'text') and response.text:
                    # Rough token estimation
                    input_tokens = len(request.prompt.split()) * 1.3
                    output_tokens = len(response.text.split()) * 1.3
                    cost = self._track_generation_cost(
                        int(input_tokens), int(output_tokens)
                    )
                    if cost:
                        logger.debug(
                            f"Generation cost: ${cost:.6f}, "
                            f"time: {generation_time:.2f}s"
                        )

                return response

            except Exception as e:
                last_error = e
                self._retry_count += 1

                if attempt < max_retries:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Generation attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await self._async_sleep(delay)
                else:
                    logger.error(
                        f"Generation failed after {max_retries + 1} attempts"
                    )

        raise GenerationError(
            message=f"Generation failed after retries: {last_error!s}",
            details={
                "provider": getattr(self.provider, "name", "unknown"),
                "error": str(last_error),
                "attempts": max_retries + 1,
                "config_driven": True,
            },
        )

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep helper for retry delays."""
        import asyncio
        await asyncio.sleep(seconds)

    async def check_provider_health(self) -> bool:
        """Check provider health status."""
        return await self.provider.check_health()

    def get_generation_stats(self) -> dict[str, Any]:
        """
        Get generation statistics including cost tracking.

        Returns:
            Dict with generation statistics
        """
        return {
            "total_generations": self._generation_count,
            "total_retries": self._retry_count,
            "total_cost_usd": sum(self._cost_tracker.values()),
            "cost_by_provider": dict(self._cost_tracker),
            "provider": getattr(self.provider, "name", "unknown"),
            "streaming_supported": self.validate_streaming_support(),
        }

    def reset_stats(self) -> None:
        """Reset generation statistics."""
        self._cost_tracker.clear()
        self._generation_count = 0
        self._retry_count = 0

    def get_provider_capabilities(self) -> dict[str, bool]:
        """
        Get capabilities of the current provider from config.

        Returns:
            Dict mapping capability names to availability
        """
        config_manager = self._get_config_manager()

        capabilities = {
            "streaming": True,
            "vision": False,
            "function_calling": False,
            "json_mode": False,
            "system_prompt": True,
            "token_counting": False,
        }

        if not config_manager:
            return capabilities

        try:
            provider_name = getattr(self.provider, "name", None)
            if provider_name:
                provider_config = config_manager.get_provider(provider_name)
                if provider_config:
                    caps = provider_config.capabilities
                    capabilities["streaming"] = caps.supports_streaming
                    capabilities["vision"] = caps.supports_vision
                    capabilities["function_calling"] = (
                        caps.supports_function_calling
                    )
                    capabilities["json_mode"] = caps.supports_json_mode
                    capabilities["system_prompt"] = caps.supports_system_prompt
                    capabilities["token_counting"] = (
                        caps.supports_token_counting
                    )
        except Exception as e:
            logger.debug(f"Could not get capabilities: {e}")

        return capabilities
