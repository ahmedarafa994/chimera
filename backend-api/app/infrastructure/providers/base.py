"""
Base Provider Implementation for Direct API Integration

Story 1.2: Direct API Integration
Provides common functionality for all LLM provider implementations.

Features:
- Common API key validation
- Input sanitization and validation
- Error handling patterns
- Logging and metrics
- Abstract methods for provider-specific implementation
"""

import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from fastapi import status

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import logger
from app.domain.interfaces import LLMProvider
from app.domain.models import PromptRequest, PromptResponse, StreamChunk


class BaseProvider(LLMProvider, ABC):
    """
    Base implementation for LLM providers.

    Provides common functionality including:
    - Input validation and sanitization
    - API key validation
    - Error handling patterns
    - Logging and metrics
    """

    def __init__(self, provider_name: str, config: Settings | None = None):
        """
        Initialize the base provider.

        Args:
            provider_name: The name of the provider (e.g., 'google', 'openai').
            config: Optional configuration settings.
        """
        self.provider_name = provider_name.lower()
        self.config = config or get_settings()
        self._client: Any = None
        self._metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "avg_latency_ms": 0.0,
        }

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _initialize_client(self, api_key: str) -> Any:
        """
        Initialize the provider-specific client.

        Args:
            api_key: The API key for authentication.

        Returns:
            The initialized client instance.
        """
        pass

    @abstractmethod
    async def _generate_impl(
        self,
        client: Any,
        request: PromptRequest,
        model_name: str,
    ) -> PromptResponse:
        """
        Provider-specific implementation of text generation.

        Args:
            client: The initialized client instance.
            request: The prompt request.
            model_name: The model name to use.

        Returns:
            The generated response.
        """
        pass

    @abstractmethod
    async def _check_health_impl(self, client: Any) -> bool:
        """
        Provider-specific health check implementation.

        Args:
            client: The initialized client instance.

        Returns:
            True if the provider is healthy.
        """
        pass

    @abstractmethod
    def _get_default_model(self) -> str:
        """
        Get the default model name for this provider.

        Returns:
            The default model name.
        """
        pass

    @abstractmethod
    def _get_api_key(self) -> str | None:
        """
        Get the API key for this provider from configuration.

        Returns:
            The API key, or None if not configured.
        """
        pass

    # =========================================================================
    # Common Validation Methods
    # =========================================================================

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key format.

        Args:
            api_key: The API key to validate.

        Returns:
            True if the API key format is valid.
        """
        if not api_key or not isinstance(api_key, str):
            return False

        # Basic format validation
        if not re.match(r"^[a-zA-Z0-9\-_\.]+$", api_key):
            logger.warning(f"[{self.provider_name}] Invalid API key format detected")
            return False

        if len(api_key) < 16 or len(api_key) > 256:
            logger.warning(f"[{self.provider_name}] API key length invalid")
            return False

        return True

    def sanitize_prompt(self, prompt: str, max_length: int = 50000) -> str:
        """
        Sanitize user input to prevent injection attacks.

        Args:
            prompt: The input prompt to sanitize.
            max_length: Maximum allowed prompt length.

        Returns:
            The sanitized prompt.

        Raises:
            AppError: If the prompt is invalid.
        """
        if not prompt:
            raise AppError("Prompt cannot be empty", status_code=status.HTTP_400_BAD_REQUEST)

        # Remove potential code injection patterns
        prompt = re.sub(r"<script.*?</script>", "", prompt, flags=re.IGNORECASE | re.DOTALL)
        prompt = re.sub(r"javascript:", "", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"on\w+\s*=", "", prompt, flags=re.IGNORECASE)

        # Limit prompt length
        if len(prompt) > max_length:
            logger.warning(
                f"[{self.provider_name}] Prompt truncated from {len(prompt)} to {max_length} characters"
            )
            prompt = prompt[:max_length]

        return prompt.strip()

    def validate_request(self, request: PromptRequest) -> None:
        """
        Validate the prompt request.

        Args:
            request: The request to validate.

        Raises:
            AppError: If the request is invalid.
        """
        if not request.prompt:
            raise AppError("Prompt is required", status_code=status.HTTP_400_BAD_REQUEST)

        # Validate API key override if provided
        if request.api_key and not self.validate_api_key(request.api_key):
            raise AppError("Invalid API key format", status_code=status.HTTP_400_BAD_REQUEST)

        # Validate model name if provided
        if request.model and not re.match(r"^[a-zA-Z0-9\-_\.]+$", request.model):
            raise AppError("Invalid model name format", status_code=status.HTTP_400_BAD_REQUEST)

    # =========================================================================
    # Client Management
    # =========================================================================

    def _get_client(self, api_key_override: str | None = None) -> Any:
        """
        Get or create a client instance.

        Args:
            api_key_override: Optional API key override.

        Returns:
            The client instance.

        Raises:
            AppError: If client initialization fails.
        """
        api_key = api_key_override or self._get_api_key()

        if not api_key:
            raise AppError(
                f"{self.provider_name.title()} API key not configured",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if not self.validate_api_key(api_key):
            raise AppError(
                "Invalid API key format",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Return existing client if no override and already initialized
        if not api_key_override and self._client is not None:
            return self._client

        try:
            client = self._initialize_client(api_key)

            # Cache client only if no override
            if not api_key_override:
                self._client = client

            return client
        except Exception as e:
            logger.error(f"[{self.provider_name}] Client initialization failed: {e}")
            raise AppError(
                f"Failed to initialize {self.provider_name} client",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    # =========================================================================
    # Metrics and Logging
    # =========================================================================

    def _record_request_start(self) -> float:
        """
        Record the start of a request.

        Returns:
            The start timestamp.
        """
        self._metrics["requests_total"] += 1
        return time.time()

    def _record_request_success(self, start_time: float) -> None:
        """
        Record a successful request.

        Args:
            start_time: The request start timestamp.
        """
        latency_ms = (time.time() - start_time) * 1000
        self._metrics["requests_success"] += 1

        # Update average latency (exponential moving average)
        if self._metrics["avg_latency_ms"] == 0:
            self._metrics["avg_latency_ms"] = latency_ms
        else:
            self._metrics["avg_latency_ms"] = (
                0.9 * self._metrics["avg_latency_ms"] + 0.1 * latency_ms
            )

        logger.debug(f"[{self.provider_name}] Request completed in {latency_ms:.1f}ms")

    def _record_request_failure(self, start_time: float, error: Exception) -> None:
        """
        Record a failed request.

        Args:
            start_time: The request start timestamp.
            error: The exception that occurred.
        """
        latency_ms = (time.time() - start_time) * 1000
        self._metrics["requests_failed"] += 1

        logger.error(
            f"[{self.provider_name}] Request failed after {latency_ms:.1f}ms: {error}",
            exc_info=True,
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        Get provider metrics.

        Returns:
            Dictionary containing provider metrics.
        """
        total_requests = self._metrics["requests_total"]
        success_rate = (
            self._metrics["requests_success"] / total_requests if total_requests > 0 else 0
        )

        return {
            "provider": self.provider_name,
            "requests_total": total_requests,
            "requests_success": self._metrics["requests_success"],
            "requests_failed": self._metrics["requests_failed"],
            "success_rate": round(success_rate, 3),
            "avg_latency_ms": round(self._metrics["avg_latency_ms"], 2),
        }

    # =========================================================================
    # LLMProvider Interface Implementation
    # =========================================================================

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """
        Generate text using the provider's API.

        Args:
            request: The prompt request.

        Returns:
            The generated response.

        Raises:
            AppError: If generation fails.
        """
        start_time = self._record_request_start()

        try:
            # Validate and sanitize input
            self.validate_request(request)

            # Get client (with optional API key override)
            client = self._get_client(request.api_key)

            # Get model name
            model_name = request.model or self._get_default_model()

            logger.info(f"[{self.provider_name}] Generating with model: {model_name}")

            # Call provider-specific implementation
            response = await self._generate_impl(client, request, model_name)

            self._record_request_success(start_time)
            return response

        except AppError:
            # Re-raise AppError without modification
            self._record_request_failure(start_time, AppError)
            raise
        except Exception as e:
            self._record_request_failure(start_time, e)
            raise AppError(
                f"{self.provider_name.title()} generation failed: {e}",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from e

    async def check_health(self) -> bool:
        """
        Check if the provider is available.

        Returns:
            True if the provider is healthy.
        """
        try:
            api_key = self._get_api_key()
            if not api_key:
                logger.debug(f"[{self.provider_name}] No API key configured")
                return False

            client = self._get_client()
            return await self._check_health_impl(client)

        except Exception as e:
            logger.error(f"[{self.provider_name}] Health check failed: {e}")
            return False

    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream text generation (default implementation raises NotImplementedError).

        Subclasses should override this method to provide streaming support.

        Args:
            request: The prompt request.

        Yields:
            StreamChunk: Individual chunks of generated text.

        Raises:
            NotImplementedError: If streaming is not supported.
        """
        raise NotImplementedError(f"Streaming not supported by {self.provider_name} provider")

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens in text (default implementation raises NotImplementedError).

        Subclasses should override this method to provide token counting.

        Args:
            text: The text to count tokens for.
            model: Optional model name for model-specific tokenization.

        Returns:
            int: The number of tokens in the text.

        Raises:
            NotImplementedError: If token counting is not supported.
        """
        raise NotImplementedError(f"Token counting not supported by {self.provider_name} provider")


# ============================================================================
# Common Error Handling Utilities
# ============================================================================


def map_http_status_code(status_code: int) -> int:
    """
    Map provider API status codes to appropriate HTTP status codes.

    Args:
        status_code: The provider's API status code.

    Returns:
        The appropriate HTTP status code.
    """
    mapping = {
        400: status.HTTP_400_BAD_REQUEST,
        401: status.HTTP_401_UNAUTHORIZED,
        403: status.HTTP_403_FORBIDDEN,
        404: status.HTTP_404_NOT_FOUND,
        429: status.HTTP_429_TOO_MANY_REQUESTS,
        500: status.HTTP_502_BAD_GATEWAY,
        502: status.HTTP_502_BAD_GATEWAY,
        503: status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    return mapping.get(status_code, status.HTTP_502_BAD_GATEWAY)


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.

    Args:
        error: The exception that occurred.

    Returns:
        True if the error is retryable.
    """
    error_str = str(error).lower()

    # Retryable conditions
    retryable_patterns = [
        "timeout",
        "connection",
        "503",  # Service unavailable
        "502",  # Bad gateway
        "500",  # Internal server error
    ]

    return any(pattern in error_str for pattern in retryable_patterns)


def is_rate_limit_error(error: Exception) -> bool:
    """
    Determine if an error is due to rate limiting.

    Args:
        error: The exception that occurred.

    Returns:
        True if the error is due to rate limiting.
    """
    error_str = str(error).lower()
    return "rate limit" in error_str or "429" in error_str
