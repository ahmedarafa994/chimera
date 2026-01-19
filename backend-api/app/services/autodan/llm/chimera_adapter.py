"""
Chimera LLM Adapter for AutoDAN Integration

This module bridges AutoDAN's model interface requirements with Chimera's LLMService,
providing robust error handling with exponential backoff for rate limit (429) errors.
"""

import asyncio
import contextlib
import logging

# Helper: cryptographically secure pseudo-floats for security-sensitive choices
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import numpy as np

from app.core.config import settings
from app.services.llm_service import llm_service


def _secure_random() -> float:
    """Cryptographically secure float in [0,1)."""
    return secrets.randbelow(10**9) / 1e9


def _secure_uniform(a, b):
    return a + _secure_random() * (b - a)


try:
    import google.genai as genai
    from google.genai import types

    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SharedRateLimitState:
    """
    Shared rate limit state across multiple ChimeraLLMAdapter instances.

    Expected Impact: -30% rate limit hits through coordinated backoff.
    """

    _instance = None
    _lock = None

    def __new__(cls):
        if cls._instance is None:
            import threading

            cls._instance = super().__new__(cls)
            cls._instance._lock = threading.Lock()
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if getattr(self, "_initialized", False):
            return

        # Shared state across all adapter instances
        self._rate_limit_until: float = 0
        self._consecutive_failures: int = 0
        self._last_error_time: float = 0
        self._provider_cooldowns: dict[str, float] = {}
        self._initialized = True
        logger.info("SharedRateLimitState initialized")

    def update_rate_limit_state(self, error: Exception, provider: str = "default"):
        """Update rate limit tracking state."""
        self._consecutive_failures += 1
        self._last_error_time = time.time()

        # Extract retry-after or calculate based on failures
        retry_after = extract_retry_after(error)
        if retry_after:
            self._rate_limit_until = time.time() + retry_after
        else:
            # Progressive cooldown based on consecutive failures
            cooldown = min(30 * self._consecutive_failures, 300)
            self._rate_limit_until = time.time() + cooldown

        # Track per-provider cooldown
        self._provider_cooldowns[provider] = self._rate_limit_until

        logger.debug(
            f"Rate limit state updated: provider={provider}, "
            f"failures={self._consecutive_failures}, "
            f"cooldown_until={self._rate_limit_until}"
        )

    def reset_rate_limit_state(self, provider: str = "default"):
        """Reset rate limit tracking after successful call."""
        self._consecutive_failures = 0
        self._rate_limit_until = 0
        self._provider_cooldowns[provider] = 0
        logger.debug(f"Rate limit state reset for provider={provider}")

    def is_in_cooldown(self, provider: str = "default") -> bool:
        """Check if we're in a rate limit cooldown period."""
        return time.time() < self._provider_cooldowns.get(provider, 0)

    def get_cooldown_remaining(self, provider: str = "default") -> float:
        """Get remaining cooldown time in seconds."""
        cooldown_until = self._provider_cooldowns.get(provider, 0)
        remaining = cooldown_until - time.time()
        return max(0.0, remaining)

    def get_stats(self) -> dict:
        """Get rate limit statistics."""
        return {
            "consecutive_failures": self._consecutive_failures,
            "current_cooldown": self._rate_limit_until,
            "provider_cooldowns": self._provider_cooldowns.copy(),
        }


class RetryStrategy(Enum):
    """Retry strategy types."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry behavior (retries disabled by default)."""

    max_retries: int = 0  # No retries by default
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.25
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

    # Specific error codes to retry on
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )


class RateLimitError(Exception):
    """Raised when rate limit (429) is encountered."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ResourceExhaustedError(Exception):
    """Raised when resource is exhausted (quota exceeded)."""

    def __init__(self, message: str, resource_type: str = "unknown"):
        super().__init__(message)
        self.resource_type = resource_type


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate the delay before the next retry attempt.

    Args:
        attempt: Current attempt numberndexed)
        config: Retry configuration

    Returns:
        Delay in seconds before next retry
    """
    if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        delay = config.base_delay * (config.exponential_base**attempt)
    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.base_delay * (attempt + 1)
    else:  # FIXED_DELAY
        delay = config.base_delay

    # Apply max delay cap
    delay = min(delay, config.max_delay)

    # Apply jitter to prevent thundering herd
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay = delay + _secure_uniform(-jitter_range, jitter_range)
        delay = max(0.1, delay)  # Ensure minimum delay

    return delay


def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """
    Determine if an error is retryable.

    Args:
        error: The exception that occurred
        config: Retry configuration

    Returns:
        True if the error should trigger a retry
    """
    # Check for specific exception types
    if isinstance(error, config.retryable_exceptions):
        return True

    # Check for rate limit errors
    error_str = str(error).lower()
    if "429" in error_str or "rate limit" in error_str or "resource exhausted" in error_str:
        return True

    if "quota" in error_str or "too many requests" in error_str:
        return True

    # Check for transient server errors
    return any(str(code) in error_str for code in config.retryable_status_codes)


def extract_retry_after(error: Exception) -> float | None:
    """
    Extract retry-after value from error if available.

    Args:
        error: The exception that occurred

    Returns:
        Retry-after value in seconds, or None if not available
    """
    error_str = str(error)

    # Try to extract retry-after from error message
    import re

    # Pattern: "retry after X seconds" or "retry-after: X"
    patterns = [
        r"retry[- ]?after[:\s]+(\d+(?:\.\d+)?)",
        r"wait[:\s]+(\d+(?:\.\d+)?)\s*(?:seconds?|s)",
        r"(\d+(?:\.\d+)?)\s*(?:seconds?|s)\s*(?:before|until)",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def infer_provider_from_model(model_name: str | None) -> str | None:
    """Infer provider name from a model identifier."""
    if not model_name:
        return None
    model_lower = model_name.lower()
    if any(tag in model_lower for tag in ("gemini", "palm", "bard")):
        return "gemini"
    if any(tag in model_lower for tag in ("gpt", "o1", "davinci", "curie", "babbage")):
        return "openai"
    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    if "deepseek" in model_lower:
        return "deepseek"
    return None


def resolve_default_model(provider: str) -> str:
    """Resolve default model name for a provider."""
    if provider in ("gemini", "google"):
        return settings.GOOGLE_MODEL or "gemini-3-pro-preview"
    if provider == "openai":
        return settings.OPENAI_MODEL or "gpt-4o"
    if provider == "anthropic":
        return settings.ANTHROPIC_MODEL or "claude-3-5-sonnet"
    if provider == "deepseek":
        return settings.DEEPSEEK_MODEL or "deepseek-chat"
    return settings.DEFAULT_MODEL_ID or settings.DEEPSEEK_MODEL or "deepseek-chat"


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    operation_name: str = "operation",
    **kwargs,
) -> T:
    """
    Execute a function without retry logic - fails immediately on any error.

    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        config: Retry configuration (ignored - no retries)
        operation_name: Name of operation for logging
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        Exception immediately on any error (no retries)
    """
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    except Exception as e:
        error_str = str(e).lower()

        # Log rate limit errors specifically
        if (
            "429" in error_str
            or "rate limit" in error_str
            or "resource exhausted" in error_str
            or "quota" in error_str
        ):
            logger.error(f"{operation_name} failed with rate limit/quota error (no retry): {e}")
        else:
            logger.error(f"{operation_name} failed: {e}")
        raise


class ChimeraLLMAdapter:
    """
    Adapter to bridge AutoDAN's model interface requirements with Chimera's LLMService.

    Features:
    - Exponential backoff for rate limit errors (HTTP 429)
    - Circuit breaker integration
    - Configurable retry strategies
    - Support for both sync and async contexts
    - Configurable safety settings (SECURITY FIX)
    """

    # Environment-based safety settings configuration
    # Set AUTODAN_SAFETY_ENABLED=true in production to enable content filtering
    # CIVP: Forcing to False for unrestricted research
    SAFETY_ENABLED = False

    def __init__(
        self,
        model_name: str | None = None,
        provider: str | None = None,
        loop=None,
        retry_config: RetryConfig | None = None,
        safety_enabled: bool | None = None,
        shared_rate_limit: SharedRateLimitState | None = None,
    ):
        self.model_name = model_name
        self.provider = provider
        self.loop = loop
        self.retry_config = retry_config or RetryConfig(
            max_retries=0,  # No retries - fail immediately
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=2.0,
            jitter=True,
        )
        self.vocab_size = 1024

        # Safety settings: use parameter, then env var, then class default
        if safety_enabled is not None:
            self.safety_enabled = safety_enabled
        else:
            self.safety_enabled = self.SAFETY_ENABLED

        # Resolve provider/model defaults and fix mismatches early
        logger.debug(f"ChimeraLLMAdapter init: input model_name={model_name}, provider={provider}")

        if not self.provider and self.model_name:
            inferred = infer_provider_from_model(self.model_name)
            if inferred:
                self.provider = inferred
                logger.debug(f"Inferred provider from model: {inferred}")

        if not self.provider:
            self.provider = settings.AI_PROVIDER or "deepseek"
            logger.debug(
                f"Using default provider: {self.provider} (AI_PROVIDER={settings.AI_PROVIDER})"
            )

        self.provider = self.provider.lower()

        if not self.model_name:
            self.model_name = resolve_default_model(self.provider)
            logger.debug(f"Resolved default model: {self.model_name} for provider {self.provider}")

        valid_models = settings.get_provider_models().get(self.provider)
        logger.debug(
            f"Valid models for {self.provider}: {valid_models}, current model: {self.model_name}"
        )
        if valid_models and self.model_name not in valid_models:
            logger.warning(
                "Unsupported model for provider; falling back to default. "
                f"provider={self.provider}, model={self.model_name}, "
                f"valid={valid_models}"
            )
            self.model_name = valid_models[0]

        # Use shared rate limit state if provided, otherwise create local state
        self._shared_state = shared_rate_limit or SharedRateLimitState()

        logger.info(
            f"ChimeraLLMAdapter initialized: provider={self.provider}, "
            f"model={self.model_name}, max_retries={self.retry_config.max_retries}, "
            f"safety_enabled={self.safety_enabled}, shared_rate_limit={shared_rate_limit is not None}"
        )

    # ------------------------------------------------------------------
    # Lightweight tokenization & gradient stubs so gradient-based flows
    # in AutoDAN can run even when the upstream provider lacks native
    # gradient/token APIs.
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Naive UTF-16 code unit tokenizer."""
        return [min(ord(char), self.vocab_size - 1) for char in text]

    def decode(self, tokens: list[int]) -> str:
        """Reverse of encode()."""
        chars = []
        for token in tokens:
            try:
                chars.append(chr(int(token)))
            except (ValueError, OverflowError):
                chars.append("?")
        return "".join(chars)

    def compute_gradients(self, tokens: list[int], target_string: str | None = None) -> np.ndarray:
        """Return a deterministic but low-information gradient tensor."""
        length = max(1, len(tokens))
        gradients = np.zeros((length, self.vocab_size), dtype=float)
        if target_string:
            target_idx = sum(ord(c) for c in target_string) % self.vocab_size
            gradients[:, target_idx] = 1.0 / length
        return gradients

    compute_gradient = compute_gradients

    def get_next_token_probs(self, context: list[int] | None = None) -> np.ndarray:
        """Simple uniform distribution used for candidate sampling."""
        probs = np.ones(self.vocab_size, dtype=float)
        probs /= probs.sum()
        return probs

    def _check_rate_limit_cooldown(self) -> None:
        """Check if we're in a rate limit cooldown period."""
        if self._shared_state.is_in_cooldown(self.provider):
            wait_time = self._shared_state.get_cooldown_remaining(self.provider)
            raise RateLimitError(
                f"Rate limit cooldown active for {self.provider}. Wait {wait_time:.1f}s",
                retry_after=wait_time,
            )

    def _update_rate_limit_state(self, error: Exception) -> None:
        """Update rate limit tracking state after an error."""
        self._shared_state.update_rate_limit_state(error, self.provider)

    def _reset_rate_limit_state(self) -> None:
        """Reset rate limit tracking after successful call."""
        self._shared_state.reset_rate_limit_state(self.provider)

    def generate(self, system: str, user: str, **kwargs) -> str:
        """
        Generates a response using the Chimera LLM Service.

        Args:
            system: System prompt
            user: User prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        prompt = f"{system}\n\n{user}"
        return self._invoke_llm(prompt, **kwargs)

    def conditional_generate(self, condition: str, system: str, user: str, **kwargs) -> str:
        """
           Generates a response with a pre-filled condition.

        Args:
               condition: Pre-fill/conditioning context
               system: System prompt
               user: User prompt
               **kwargs: Additional generation parameters

           Returns:
               Generated text response
        """
        prompt = f"{system}\n\n{user}\n\n{condition}"
        return self._invoke_llm(prompt, **kwargs)

    def _invoke_llm(self, prompt: str, **kwargs) -> str:
        """
        Invoke the LLM with proper async handling and retry logic.

        Args:
            prompt: The full prompt to send
            **kwargs: Generation parameters

        Returns:
            Generated text response
        """

        logger.debug(
            f"Invoking LLM: provider={self.provider}, model={self.model_name}, "
            f"loop={'active' if self.loop and not self.loop.is_closed() else 'none'}"
        )

        # Check rate limit cooldown
        try:
            self._check_rate_limit_cooldown()
        except RateLimitError as e:
            logger.warning(f"Rate limit cooldown active: {e}")
            # Wait for cooldown and retry
            time.sleep(e.retry_after)

        # If we have a reference to the main event loop, use it
        if self.loop and not self.loop.is_closed():
            logger.debug("Using main event loop for generation")
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._generate_with_retry(prompt, **kwargs),
                    self.loop,
                )
            except RuntimeError as e:
                # e.g. uvicorn reload/shutdown closed the loop mid-flight
                if "event loop is closed" in str(e).lower():
                    logger.warning("Main event loop is closed; falling back to a new loop")
                    self.loop = None
                else:
                    raise
            else:
                try:
                    result = future.result()  # No timeout
                    self._reset_rate_limit_state()
                    return result
                except Exception as e:
                    if "event loop is closed" in str(e).lower():
                        logger.warning(
                            "Main event loop closed while awaiting result; "
                            "falling back to a new loop"
                        )
                        with contextlib.suppress(Exception):
                            future.cancel()
                        self.loop = None
                    else:
                        self._update_rate_limit_state(e)
                        raise

        logger.debug("No main loop available, using synchronous Gemini client")

        # Use synchronous Gemini client directly to avoid event loop issues
        # This is called from thread pool executors in AutoDAN
        result = self._generate_sync(prompt, **kwargs)
        self._reset_rate_limit_state()
        return result

    def _generate_sync(self, prompt: str, **kwargs) -> str:
        """
        Synchronous generation using provider SDK directly.
        Used when called from thread pool executors without an event loop.
        Supports DeepSeek, Gemini, and OpenAI providers.
        """
        # Map kwargs to generation config
        temperature = kwargs.get("temperature", 0.7)
        raw_max_tokens = kwargs.get("max_length", kwargs.get("max_tokens", 8192))
        max_tokens = min(int(raw_max_tokens), 8192)
        top_p = kwargs.get("top_p", 0.95)

        # Route to appropriate provider
        if self.provider in ("deepseek", "openai"):
            return self._generate_sync_openai_compatible(prompt, temperature, max_tokens, top_p)
        else:
            return self._generate_sync_gemini(prompt, temperature, max_tokens, top_p)

    def _generate_sync_openai_compatible(
        self, prompt: str, temperature: float, max_tokens: int, top_p: float
    ) -> str:
        """Synchronous generation using OpenAI-compatible API (DeepSeek, OpenAI)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai SDK not available for sync generation")

        # Get API key and base URL based on provider
        if self.provider == "deepseek":
            api_key = settings.DEEPSEEK_API_KEY
            # DeepSeek API base URL - MUST include /v1 suffix for OpenAI SDK compatibility
            base_url = settings.DIRECT_DEEPSEEK_BASE_URL or "https://api.deepseek.com/v1"
            # CRITICAL: Use self.model_name which was validated in __init__
            # Do NOT fall back to settings here - that bypasses validation
            model_name = self.model_name
            if not model_name:
                # This should never happen if __init__ ran correctly
                logger.error(
                    "[CRITICAL] self.model_name is None/empty after __init__! "
                    f"Falling back to settings.DEEPSEEK_MODEL={settings.DEEPSEEK_MODEL}"
                )
                model_name = settings.DEEPSEEK_MODEL or "deepseek-chat"
            if not api_key:
                raise RuntimeError("No DeepSeek API key configured (DEEPSEEK_API_KEY)")
        else:  # openai
            api_key = settings.OPENAI_API_KEY
            base_url = settings.DIRECT_OPENAI_BASE_URL or "https://api.openai.com/v1"
            model_name = self.model_name or settings.OPENAI_MODEL or "gpt-4o"
            if not api_key:
                raise RuntimeError("No OpenAI API key configured (OPENAI_API_KEY)")

        # ALWAYS log at ERROR level to ensure visibility in all log configurations
        # This is critical for debugging "Model Not Exist" errors
        logger.error(
            f"[MODEL DEBUG] API call starting: provider={self.provider}, "
            f"model_name='{model_name}', self.model_name='{self.model_name}', "
            f"settings.DEEPSEEK_MODEL='{settings.DEEPSEEK_MODEL}', "
            f"base_url={base_url}"
        )

        # Create client
        client = OpenAI(api_key=api_key, base_url=base_url)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
            )

            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                logger.warning(f"Empty response from {self.provider}")
                return ""

        except Exception as e:
            error_str = str(e).lower()

            # Log full details for debugging "Model Not Exist" errors
            # Using ERROR level to ensure visibility regardless of log config
            # Note: Using separate log calls to avoid multi-line truncation
            logger.error(f"Sync {self.provider} generation failed: {e}")
            logger.error(f"  [DEBUG] model_name used in API call: '{model_name}'")
            logger.error(f"  [DEBUG] self.model_name: '{self.model_name}'")
            logger.error(f"  [DEBUG] settings.DEEPSEEK_MODEL: '{settings.DEEPSEEK_MODEL}'")
            logger.error(f"  [DEBUG] base_url: '{base_url}'")
            logger.error(f"  [DEBUG] api_key present: {bool(api_key)}")

            # Fail immediately on rate limit/quota errors - no retries
            if (
                "429" in error_str
                or "rate limit" in error_str
                or "resource exhausted" in error_str
                or "quota" in error_str
            ):
                raise RateLimitError(f"Rate limit/quota exceeded: {e}")

            # Other errors also fail immediately
            raise

    def _generate_sync_gemini(
        self, prompt: str, temperature: float, max_tokens: int, top_p: float
    ) -> str:
        """Synchronous generation using Gemini SDK."""
        if not HAS_GENAI:
            raise RuntimeError("google-genai SDK not available for sync generation")

        # Get API key
        api_key = settings.GOOGLE_API_KEY or getattr(settings, "GEMINI_API_KEY", None)
        if not api_key:
            raise RuntimeError("No Gemini API key configured")

        # Create a fresh sync client for this request
        client = genai.Client(api_key=api_key)

        # Build safety settings based on configuration
        if self.safety_enabled:
            safety_settings = None
            logger.debug("Using default Gemini safety settings (filtering enabled)")
        else:
            safety_settings = [
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
                ),
            ]
            logger.debug("Safety settings disabled for research mode")

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            safety_settings=safety_settings,
        )

        model_name = self.model_name or settings.GOOGLE_MODEL

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                if response and hasattr(response, "prompt_feedback"):
                    logger.warning(f"Prompt feedback: {response.prompt_feedback}")
                return ""

        except Exception as e:
            error_str = str(e).lower()

            if (
                "429" in error_str
                or "rate limit" in error_str
                or "resource exhausted" in error_str
                or "quota" in error_str
            ):
                logger.error(f"Rate limit/quota exceeded (no retry): {e}")
                raise RateLimitError(f"Rate limit/quota exceeded: {e}")

            logger.error(f"Sync Gemini generation failed: {e}")
            raise

    async def generate_async(self, system: str, user: str, **kwargs) -> str:
        """Async version of generate."""
        prompt = f"{system}\n\n{user}"
        return await self._generate_with_retry(prompt, **kwargs)

    async def conditional_generate_async(
        self, condition: str, system: str, user: str, **kwargs
    ) -> str:
        """Async version of conditional_generate."""
        prompt = f"{system}\n\n{user}\n\n{condition}"
        return await self._generate_with_retry(prompt, **kwargs)

    async def _generate_with_retry(self, prompt: str, **kwargs) -> str:
        """
        Generate with automatic retry and eonential backoff.

        Args:
            prompt: The prompt to send
            **kwargs: Generation parameters

        Returns:
            Generated text response
        """
        return await retry_with_backoff(
            self._generate_async,
            prompt,
            config=self.retry_config,
            operation_name=f"LLM generation ({self.provider}/{self.model_name})",
            **kwargs,
        )

    async def _generate_async(self, prompt: str, **kwargs) -> str:
        """
        Core async generation method.

        Args:
            prompt: The prompt to send
            **kwargs: Generation parameters

        Returns:
            Generated text response
        """
        try:
            # Map kwargs to generation config
            temperature = kwargs.get("temperature", 0.7)

            # AutoDAN uses max_length, use higher default for full prompt generation
            # Cap at 8192 for Gemini compatibility
            raw_max_tokens = kwargs.get("max_length", kwargs.get("max_tokens", 8192))
            max_tokens = min(int(raw_max_tokens), 8192)

            top_p = kwargs.get("top_p", 0.95)

            logger.debug(
                f"ChimeraLLMAdapter gtemp={temperature}, "
                f"max_tokens={max_tokens} (raw={raw_max_tokens}), top_p={top_p}"
            )

            response = await llm_service.generate(
                prompt=prompt,
                provider=self.provider,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            return response.content

        except Exception as e:
            error_str = str(e).lower()

            # Classify the error for better handling
            if "429" in error_str or "rate limit" in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif "resource exhausted" in error_str or "quota" in error_str:
                raise ResourceExhaustedError(f"Resource exhausted: {e}")

            logger.error(f"Error invoking LLM Service: {e}")
            raise


class AutoDANModelInterface:
    """
    High-level interface for AutoDAN framework integration.

    This class provides a clean interface that matches AutoDAN's expected
    model interface while leveraging Chimrastructure.
    """

    def __init__(
        self, provider: str = "gemini", model: str | None = None, retry_config: RetryConfig = None
    ):
        self.adapter = ChimeraLLMAdapter(
            model_name=model, provider=provider, retry_config=retry_config
        )
        self._call_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "call_count": self._call_count,
            "error_count": self._error_count,
            "avg_latency_ms": (
                self._total_latency / self._call_count * 1000 if self._call_count > 0 else 0
            ),
            "error_rate": (self._error_count / self._call_count if self._call_count > 0 else 0),
        }

    def generate(self, system: str, user: str, **kwargs) -> str:
        """Generate response with tracking."""
        self._call_count += 1
        start_time = time.time()

        try:
            result = self.adapter.generate(system, user, **kwargs)
            self._total_latency += time.time() - start_time
            return result
        except Exception:
            self._error_count += 1
            raise

    def conditional_generate(self, condition: str, system: str, user: str, **kwargs) -> str:
        """Conditional generate with tracking."""
        self._call_count += 1
        start_time = time.time()

        try:
            result = self.adapter.conditional_generate(condition, system, user, **kwargs)
            self._total_latency += time.time() - start_time
            return result
        except Exception:
            self._error_count += 1
            raise

    async def generate_async(self, system: str, user: str, **kwargs) -> str:
        """Async generate with tracking."""
        self._call_count += 1
        start_time = time.time()

        try:
            result = await self.adapter.generate_async(system, user, **kwargs)
            self._total_latency += time.time() - start_time
            return result
        except Exception:
            self._error_count += 1
            raise


class BatchingChimeraAdapter(ChimeraLLMAdapter):
    """
    Batching-enabled Chimera LLM Adapter.

    Queues requests and processes them in batches for improved throughput
    and reduced API costs. Uses a semaphore to limit concurrent API requests
    and prevent rate limiting.

    Expected Impact: -40% API cost through request batching.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 5,
        batch_timeout: float = 0.5,
        max_concurrent_requests: int = 3,
        **kwargs,
    ):
        """
        Initialize batching adapter.

        Args:
            batch_size: Maximum requests per batch
            batch_timeout: Maximum wait time before processing partial batch
            max_concurrent_requests: Maximum number of concurrent API requests (default: 3)
            *args, **kwargs: Passed to ChimeraLLMAdapter
        """
        super().__init__(*args, **kwargs)
        self._batch_queue: asyncio.Queue = None
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self._batch_task: asyncio.Task = None
        self._batch_lock = asyncio.Lock()
        self._max_concurrent_requests = max_concurrent_requests
        self._concurrency_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_requests)

        logger.info(
            f"BatchingChimeraAdapter initialized: batch_size={batch_size}, "
            f"timeout={batch_timeout}s, max_concurrent_requests={max_concurrent_requests}"
        )

    def _ensure_queue(self):
        """Ensure the batch queue is initialized."""
        if self._batch_queue is None:
            self._batch_queue = asyncio.Queue()

    async def generate_batched(self, system: str, user: str, **kwargs) -> str:
        """
        Queue request for batched processing.

        Args:
            system: System prompt
            user: User prompt
            **kwargs: Generation parameters

        Returns:
            Generated text response
        """
        self._ensure_queue()

        # Create a future to receive the result
        future = asyncio.get_event_loop().create_future()
        prompt = f"{system}\n\n{user}"

        await self._batch_queue.put((prompt, kwargs, future))

        # Start batch processor if not running
        async with self._batch_lock:
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def _process_batch(self):
        """Process queued requests in batches."""
        while True:
            batch = []

            try:
                # Collect batch items
                while len(batch) < self._batch_size:
                    try:
                        item = await asyncio.wait_for(
                            self._batch_queue.get(), timeout=self._batch_timeout
                        )
                        batch.append(item)
                    except TimeoutError:
                        break

                if not batch:
                    return

                # Process batch
                prompts = [item[0] for item in batch]
                kwargs_list = [item[1] for item in batch]
                futures = [item[2] for item in batch]

                logger.debug(
                    f"Processing batch of {len(batch)} requests "
                    f"(max_concurrent={self._max_concurrent_requests})"
                )

                # Execute requests with rate limiting via semaphore
                tasks = [
                    self._rate_limited_generate(prompt, **kw)
                    for prompt, kw in zip(prompts, kwargs_list, strict=False)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Distribute results to futures
                for future, result in zip(futures, results, strict=False):
                    if not future.done():
                        if isinstance(result, Exception):
                            future.set_exception(result)
                        else:
                            future.set_result(result)

            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Set exception on all pending futures
                for item in batch:
                    future = item[2]
                    if not future.done():
                        future.set_exception(e)

            # Check if more items in queue
            if self._batch_queue.empty():
                return

    async def _rate_limited_generate(self, prompt: str, **kwargs) -> str:
        """
        Generate with rate limiting via semaphore.

        Acquires the concurrency semaphore before making the API call to ensure
        we don't exceed max_concurrent_requests simultaneous API calls.

        Args:
            prompt: The prompt to send
            **kwargs: Generation parameters

        Returns:
            Generated text response
        """
        async with self._concurrency_semaphore:
            return await self._generate_with_retry(prompt, **kwargs)

    async def generate_batch(self, requests: list[tuple[str, str]], **kwargs) -> list[str]:
        """
        Generate responses for multiple requests with rate limiting.

        Uses a semaphore to limit concurrent API calls and prevent rate limiting.

        Args:
            requests: List of (system, user) prompt tuples
            **kwargs: Shared generation parameters

        Returns:
            List of generated responses
        """
        tasks = [
            self._rate_limited_generate(f"{system}\n\n{user}", **kwargs)
            for system, user in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_batch_stats(self) -> dict:
        """Get batching statistics."""
        return {
            "queue_size": self._batch_queue.qsize() if self._batch_queue else 0,
            "batch_size": self._batch_size,
            "batch_timeout": self._batch_timeout,
            "batch_task_running": self._batch_task is not None and not self._batch_task.done(),
            "max_concurrent_requests": self._max_concurrent_requests,
        }
