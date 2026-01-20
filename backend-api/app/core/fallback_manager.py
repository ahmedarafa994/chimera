"""Fallback Manager for AI Provider Integration.

Manages fallback logic for AI provider operations including:
- Chain fallback through provider failover chains
- Capability-based fallback
- Cost-based fallback
- Circuit breaker integration
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Fallback Enums and Data Classes
# =============================================================================


class FallbackStrategy(Enum):
    """Available fallback strategies."""

    CHAIN = "chain"  # Try next provider in failover chain
    CAPABILITY = "capability"  # Find provider with required capability
    COST = "cost"  # Fall back to cheaper provider
    PRIORITY = "priority"  # Try providers by priority


class FallbackReason(Enum):
    """Reasons for falling back to another provider."""

    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    API_ERROR = "api_error"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNKNOWN = "unknown"


@dataclass
class FallbackResult:
    """Result of a fallback operation."""

    success: bool
    provider_used: str
    model_used: str
    attempts: int
    fallback_chain: list[str]
    error_history: list[str]
    total_time_ms: float = 0.0
    final_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "provider_used": self.provider_used,
            "model_used": self.model_used,
            "attempts": self.attempts,
            "fallback_chain": self.fallback_chain,
            "error_history": self.error_history,
            "total_time_ms": self.total_time_ms,
            "final_error": self.final_error,
        }


@dataclass
class ProviderFailureRecord:
    """Record of provider failures for tracking."""

    failure_count: int = 0
    last_failure_time: datetime | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    success_count: int = 0
    total_requests: int = 0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests

    def should_skip(
        self,
        skip_threshold: int = 3,
        skip_window_seconds: int = 300,
    ) -> bool:
        """Determine if provider should be skipped.

        Args:
            skip_threshold: Number of consecutive failures to skip
            skip_window_seconds: Time window for considering failures

        Returns:
            True if provider should be skipped

        """
        if self.consecutive_failures < skip_threshold:
            return False

        if self.last_failure_time is None:
            return False

        window = timedelta(seconds=skip_window_seconds)
        return not datetime.utcnow() - self.last_failure_time > window


@dataclass
class FallbackStats:
    """Statistics about fallback operations."""

    total_fallbacks: int = 0
    successful_fallbacks: int = 0
    failed_fallbacks: int = 0
    provider_failures: dict[str, ProviderFailureRecord] = field(default_factory=dict)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    avg_attempts_per_request: float = 0.0
    total_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_fallbacks": self.total_fallbacks,
            "successful_fallbacks": self.successful_fallbacks,
            "failed_fallbacks": self.failed_fallbacks,
            "fallback_success_rate": (
                self.successful_fallbacks / self.total_fallbacks
                if self.total_fallbacks > 0
                else 0.0
            ),
            "provider_failures": {
                k: {
                    "failure_count": v.failure_count,
                    "consecutive_failures": v.consecutive_failures,
                    "success_count": v.success_count,
                    "failure_rate": v.failure_rate,
                }
                for k, v in self.provider_failures.items()
            },
            "fallback_reasons": self.fallback_reasons,
            "avg_attempts_per_request": self.avg_attempts_per_request,
            "total_requests": self.total_requests,
        }


# =============================================================================
# Fallback Manager
# =============================================================================


class FallbackManager:
    """Manages fallback logic for AI provider operations.

    Features:
    - Execute operations with automatic fallback through provider chain
    - Track provider failures and successes
    - Skip providers based on recent failure patterns
    - Support multiple fallback strategies
    - Integration with circuit breaker system

    Example:
        manager = FallbackManager()

        result, fallback_result = await manager.execute_with_fallback(
            operation=my_llm_call,
            chain_name="premium",
            max_attempts=3,
        )

    """

    def __init__(
        self,
        skip_threshold: int = 3,
        skip_window_seconds: int = 300,
        reset_window_seconds: int = 600,
    ) -> None:
        """Initialize the FallbackManager.

        Args:
            skip_threshold: Consecutive failures before skipping provider
            skip_window_seconds: Time window for failure tracking
            reset_window_seconds: Time after which failure counts reset

        """
        self._config_manager = None
        self._stats = FallbackStats()
        self._skip_threshold = skip_threshold
        self._skip_window = skip_window_seconds
        self._reset_window = reset_window_seconds
        self._lock = asyncio.Lock()

        logger.info(
            f"FallbackManager initialized (skip_threshold={skip_threshold}, "
            f"skip_window={skip_window_seconds}s)",
        )

    def _get_config_manager(self):
        """Lazily get the AI config manager."""
        if self._config_manager is None:
            from app.core.ai_config_manager import get_ai_config_manager

            self._config_manager = get_ai_config_manager()
        return self._config_manager

    # =========================================================================
    # Main Fallback Execution
    # =========================================================================

    async def execute_with_fallback(
        self,
        operation: Callable[..., Any],
        chain_name: str | None = None,
        provider_id: str | None = None,
        max_attempts: int = 3,
        strategy: FallbackStrategy = FallbackStrategy.CHAIN,
        required_capabilities: list[str] | None = None,
        **kwargs,
    ) -> tuple[Any, FallbackResult]:
        """Execute operation with automatic fallback through provider chain.

        Args:
            operation: Async callable to execute (should accept provider_id)
            chain_name: Named failover chain to use
            provider_id: Starting provider (uses default if not specified)
            max_attempts: Maximum number of providers to try
            strategy: Fallback strategy to use
            required_capabilities: List of required capability names
            **kwargs: Additional arguments to pass to operation

        Returns:
            Tuple of (operation result, FallbackResult metadata)

        """
        start_time = time.perf_counter()
        error_history = []
        providers_tried = []

        # Get the fallback chain
        chain = self._get_provider_chain(
            chain_name=chain_name,
            provider_id=provider_id,
            strategy=strategy,
            required_capabilities=required_capabilities,
        )

        if not chain:
            return None, FallbackResult(
                success=False,
                provider_used="",
                model_used="",
                attempts=0,
                fallback_chain=[],
                error_history=["No providers available in fallback chain"],
                final_error="No providers available",
            )

        # Limit to max_attempts
        chain = chain[:max_attempts]

        result = None
        final_error = None
        current_provider = ""
        current_model = ""

        for attempt, current_provider in enumerate(chain, 1):
            providers_tried.append(current_provider)

            # Check if provider should be skipped
            if self.should_skip_provider(current_provider):
                error_msg = f"Skipping {current_provider} (recent failures)"
                error_history.append(error_msg)
                logger.info(error_msg)
                continue

            # Get provider config for model info
            config_manager = self._get_config_manager()
            if config_manager.is_loaded():
                provider_config = config_manager.get_provider(current_provider)
                if provider_config:
                    default_model = provider_config.get_default_model()
                    current_model = default_model.model_id if default_model else ""

            try:
                logger.info(
                    f"Attempt {attempt}/{len(chain)}: Trying provider '{current_provider}'",
                )

                # Execute the operation
                result = await operation(
                    provider_id=current_provider,
                    **kwargs,
                )

                # Success - record it
                self.record_success(current_provider)

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Update stats
                async with self._lock:
                    self._stats.total_requests += 1
                    if attempt > 1:
                        self._stats.total_fallbacks += 1
                        self._stats.successful_fallbacks += 1

                return result, FallbackResult(
                    success=True,
                    provider_used=current_provider,
                    model_used=current_model,
                    attempts=attempt,
                    fallback_chain=providers_tried,
                    error_history=error_history,
                    total_time_ms=elapsed_ms,
                )

            except Exception as e:
                error_msg = f"Provider '{current_provider}' failed: {e}"
                error_history.append(error_msg)
                final_error = str(e)
                logger.warning(error_msg)

                # Record the failure
                reason = self._classify_error(e)
                self.record_failure(current_provider, str(e), reason)

        # All attempts failed
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        async with self._lock:
            self._stats.total_requests += 1
            if len(providers_tried) > 1:
                self._stats.total_fallbacks += 1
                self._stats.failed_fallbacks += 1

        return None, FallbackResult(
            success=False,
            provider_used=current_provider,
            model_used=current_model,
            attempts=len(providers_tried),
            fallback_chain=providers_tried,
            error_history=error_history,
            total_time_ms=elapsed_ms,
            final_error=final_error,
        )

    def _get_provider_chain(
        self,
        chain_name: str | None = None,
        provider_id: str | None = None,
        strategy: FallbackStrategy = FallbackStrategy.CHAIN,
        required_capabilities: list[str] | None = None,
    ) -> list[str]:
        """Get the provider chain based on strategy.

        Args:
            chain_name: Named failover chain
            provider_id: Starting provider
            strategy: Fallback strategy
            required_capabilities: Required capability names

        Returns:
            List of provider IDs in fallback order

        """
        config_manager = self._get_config_manager()
        if not config_manager.is_loaded():
            return []

        config = config_manager.get_config()

        if strategy == FallbackStrategy.CHAIN:
            return self.get_fallback_chain(chain_name, provider_id)

        if strategy == FallbackStrategy.PRIORITY:
            # Get all enabled providers sorted by priority
            enabled = config.get_enabled_providers()
            return [p.provider_id for p in enabled]

        if strategy == FallbackStrategy.CAPABILITY:
            # Filter by required capabilities
            if not required_capabilities:
                return self.get_fallback_chain(chain_name, provider_id)

            matching = []
            for provider in config.get_enabled_providers():
                caps = provider.capabilities
                has_all = True
                for cap in required_capabilities:
                    if not getattr(caps, f"supports_{cap}", False):
                        has_all = False
                        break
                if has_all:
                    matching.append(provider.provider_id)
            return matching

        if strategy == FallbackStrategy.COST:
            # Sort by cost (economy tier first)
            from app.config.ai_provider_settings import ModelTier

            tier_order = {
                ModelTier.ECONOMY: 0,
                ModelTier.STANDARD: 1,
                ModelTier.PREMIUM: 2,
                ModelTier.ENTERPRISE: 3,
            }

            providers_with_tier = []
            for provider in config.get_enabled_providers():
                default_model = provider.get_default_model()
                tier = default_model.tier if default_model else ModelTier.STANDARD
                providers_with_tier.append((provider.provider_id, tier))

            providers_with_tier.sort(key=lambda x: tier_order.get(x[1], 1))
            return [p[0] for p in providers_with_tier]

        return []

    def _classify_error(self, error: Exception) -> FallbackReason:
        """Classify an error to determine the fallback reason."""
        error_str = str(error).lower()

        if "timeout" in error_str:
            return FallbackReason.TIMEOUT
        if "rate" in error_str or "429" in error_str:
            return FallbackReason.RATE_LIMITED
        if "circuit" in error_str:
            return FallbackReason.CIRCUIT_OPEN
        if "auth" in error_str or "401" in error_str or "403" in error_str:
            return FallbackReason.AUTHENTICATION
        if "quota" in error_str:
            return FallbackReason.QUOTA_EXCEEDED
        if "model" in error_str and "not found" in error_str:
            return FallbackReason.MODEL_UNAVAILABLE
        if "api" in error_str or "500" in error_str or "502" in error_str:
            return FallbackReason.API_ERROR
        return FallbackReason.UNKNOWN

    # =========================================================================
    # Fallback Chain Management
    # =========================================================================

    def get_fallback_chain(
        self,
        name: str | None = None,
        provider_id: str | None = None,
    ) -> list[str]:
        """Get fallback chain from config or use default.

        Args:
            name: Named chain name (e.g., "premium", "cost_optimized")
            provider_id: Provider ID to get chain for

        Returns:
            List of provider IDs in fallback order

        """
        config_manager = self._get_config_manager()
        if not config_manager.is_loaded():
            return []

        config = config_manager.get_config()

        # Check named chains first
        if name and name in config.failover_chains:
            return config.failover_chains[name].providers

        # Get provider-specific chain
        if provider_id:
            provider = config.get_provider(provider_id)
            if provider:
                # Start with the provider itself, then its chain
                return [provider_id, *list(provider.failover_chain)]

        # Use default provider's chain
        default_provider_id = config.global_config.default_provider
        default_provider = config.get_provider(default_provider_id)
        if default_provider:
            return [default_provider_id, *list(default_provider.failover_chain)]

        # Last resort: all enabled providers by priority
        return [p.provider_id for p in config.get_enabled_providers()]

    # =========================================================================
    # Failure Tracking
    # =========================================================================

    def record_failure(
        self,
        provider: str,
        error: str,
        reason: FallbackReason = FallbackReason.UNKNOWN,
    ) -> None:
        """Record a provider failure for statistics.

        Args:
            provider: Provider ID that failed
            error: Error message
            reason: Classified reason for failure

        """
        if provider not in self._stats.provider_failures:
            self._stats.provider_failures[provider] = ProviderFailureRecord()

        record = self._stats.provider_failures[provider]
        record.failure_count += 1
        record.consecutive_failures += 1
        record.last_failure_time = datetime.utcnow()
        record.last_error = error
        record.total_requests += 1

        # Track reason
        reason_key = reason.value
        self._stats.fallback_reasons[reason_key] = (
            self._stats.fallback_reasons.get(reason_key, 0) + 1
        )

        logger.debug(
            f"Recorded failure for {provider}: "
            f"consecutive={record.consecutive_failures}, "
            f"reason={reason.value}",
        )

    def record_success(self, provider: str) -> None:
        """Record a successful provider operation.

        Args:
            provider: Provider ID that succeeded

        """
        if provider not in self._stats.provider_failures:
            self._stats.provider_failures[provider] = ProviderFailureRecord()

        record = self._stats.provider_failures[provider]
        record.success_count += 1
        record.consecutive_failures = 0  # Reset consecutive failures
        record.total_requests += 1

        logger.debug(f"Recorded success for {provider}")

    def should_skip_provider(self, provider: str) -> bool:
        """Check if provider should be skipped based on recent failures.

        Args:
            provider: Provider ID to check

        Returns:
            True if provider should be skipped

        """
        if provider not in self._stats.provider_failures:
            return False

        record = self._stats.provider_failures[provider]
        return record.should_skip(
            skip_threshold=self._skip_threshold,
            skip_window_seconds=self._skip_window,
        )

    def reset_provider_status(self, provider: str) -> None:
        """Reset failure count for a provider.

        Args:
            provider: Provider ID to reset

        """
        if provider in self._stats.provider_failures:
            record = self._stats.provider_failures[provider]
            record.consecutive_failures = 0
            record.last_failure_time = None
            logger.info(f"Reset failure status for provider: {provider}")

    def reset_all_provider_status(self) -> None:
        """Reset failure counts for all providers."""
        for provider in self._stats.provider_failures:
            self.reset_provider_status(provider)
        logger.info("Reset failure status for all providers")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_fallback_stats(self) -> dict[str, Any]:
        """Get statistics about fallback operations.

        Returns:
            Dictionary with fallback statistics

        """
        return self._stats.to_dict()

    def get_provider_health(self, provider: str) -> dict[str, Any]:
        """Get health information for a specific provider.

        Args:
            provider: Provider ID

        Returns:
            Dictionary with provider health information

        """
        if provider not in self._stats.provider_failures:
            return {
                "provider": provider,
                "status": "healthy",
                "failure_count": 0,
                "success_count": 0,
                "consecutive_failures": 0,
                "should_skip": False,
            }

        record = self._stats.provider_failures[provider]
        should_skip = record.should_skip(
            skip_threshold=self._skip_threshold,
            skip_window_seconds=self._skip_window,
        )

        status = "healthy"
        if should_skip:
            status = "unhealthy"
        elif record.consecutive_failures > 0:
            status = "degraded"

        return {
            "provider": provider,
            "status": status,
            "failure_count": record.failure_count,
            "success_count": record.success_count,
            "consecutive_failures": record.consecutive_failures,
            "failure_rate": record.failure_rate,
            "should_skip": should_skip,
            "last_failure": (
                record.last_failure_time.isoformat() if record.last_failure_time else None
            ),
            "last_error": record.last_error,
        }

    def get_all_providers_health(self) -> dict[str, dict[str, Any]]:
        """Get health information for all tracked providers.

        Returns:
            Dictionary mapping provider IDs to health information

        """
        config_manager = self._get_config_manager()
        health = {}

        # Get all providers from config
        if config_manager.is_loaded():
            config = config_manager.get_config()
            for provider_id in config.providers:
                health[provider_id] = self.get_provider_health(provider_id)

        # Add any providers we've tracked but might not be in config
        for provider_id in self._stats.provider_failures:
            if provider_id not in health:
                health[provider_id] = self.get_provider_health(provider_id)

        return health


# =============================================================================
# Global Instance
# =============================================================================

# Singleton instance
_fallback_manager: FallbackManager | None = None


def get_fallback_manager() -> FallbackManager:
    """Get the global FallbackManager instance."""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = FallbackManager()
    return _fallback_manager


# Convenience alias for service registry
fallback_manager = get_fallback_manager()
