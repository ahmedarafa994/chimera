"""
Timeout Configuration for External Service Calls

PERF-001 FIX: Implements proper timeout hierarchy to prevent hanging requests
and improve overall application reliability and performance.

Timeout Strategy:
- Standard API calls: 30s (most operations)
- Long-running operations: 5min (AutoDAN, GPTFuzz)
- LLM provider calls: 2min (external API latency)
- Extended optimization: 10min (AutoDAN lifelong learning)

Usage:
    from app.core.timeouts import TimeoutConfig

    # Get timeout for operation type
    timeout = TimeoutConfig.get_timeout("standard")
    timeout = TimeoutConfig.get_timeout("llm")
    timeout = TimeoutConfig.get_timeout("autodan")

    # Use with async operations
    import asyncio
    try:
        result = await asyncio.wait_for(api_call(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TimeoutType(Enum):
    """Predefined timeout types for different operation categories."""

    # Standard timeouts
    STANDARD = "standard"  # 30s - Regular API calls
    FAST = "fast"  # 10s - Quick operations
    EXTENDED = "extended"  # 5min - Long-running operations

    # LLM-specific timeouts
    LLM = "llm"  # 2min - LLM provider calls
    LLM_STREAM = "llm_stream"  # 5min - Streaming LLM calls

    # Specialized operation timeouts
    AUTODAN = "autodan"  # 10min - AutoDAN optimization
    GPTFUZZ = "gptfuzz"  # 10min - GPTFuzz testing
    GRADIENT = "gradient"  # 5min - Gradient optimization

    # Health check timeouts
    HEALTH_CHECK = "health_check"  # 5s - Health checks
    CONNECTION_TEST = "connection"  # 10s - Connection testing


@dataclass
class TimeoutConfig:
    """
    Timeout configuration for different operation types.

    Provides centralized timeout management to prevent configuration
    inconsistencies across the application.
    """

    # Standard API operations (milliseconds)
    standard_timeout_ms: int = 30000  # 30 seconds
    fast_timeout_ms: int = 10000  # 10 seconds
    extended_timeout_ms: int = 300000  # 5 minutes

    # LLM operations (milliseconds)
    llm_timeout_ms: int = 120000  # 2 minutes
    llm_stream_timeout_ms: int = 300000  # 5 minutes

    # Specialized operations (milliseconds)
    autodan_timeout_ms: int = 600000  # 10 minutes
    gptfuzz_timeout_ms: int = 600000  # 10 minutes
    gradient_timeout_ms: int = 300000  # 5 minutes

    # Health checks (milliseconds)
    health_check_timeout_ms: int = 5000  # 5 seconds
    connection_test_timeout_ms: int = 10000  # 10 seconds

    @classmethod
    def get_timeout(cls, operation: str | TimeoutType) -> float:
        """
        Get timeout value for an operation type in seconds.

        Args:
            operation: Operation type as string or TimeoutType enum

        Returns:
            Timeout in seconds (float)

        Raises:
            ValueError: If operation type is unknown
        """
        # Convert string to enum if needed
        if isinstance(operation, str):
            try:
                operation = TimeoutType(operation)
            except ValueError:
                valid_types = [t.value for t in TimeoutType]
                raise ValueError(
                    f"Unknown operation type: {operation}. "
                    f"Valid types: {', '.join(valid_types)}"
                )

        config = cls()

        # Map operation types to timeout values (in seconds)
        timeout_map = {
            TimeoutType.STANDARD: config.standard_timeout_ms / 1000,
            TimeoutType.FAST: config.fast_timeout_ms / 1000,
            TimeoutType.EXTENDED: config.extended_timeout_ms / 1000,
            TimeoutType.LLM: config.llm_timeout_ms / 1000,
            TimeoutType.LLM_STREAM: config.llm_stream_timeout_ms / 1000,
            TimeoutType.AUTODAN: config.autodan_timeout_ms / 1000,
            TimeoutType.GPTFUZZ: config.gptfuzz_timeout_ms / 1000,
            TimeoutType.GRADIENT: config.gradient_timeout_ms / 1000,
            TimeoutType.HEALTH_CHECK: config.health_check_timeout_ms / 1000,
            TimeoutType.CONNECTION_TEST: config.connection_test_timeout_ms / 1000,
        }

        if operation not in timeout_map:
            valid_types = [t.value for t in TimeoutType]
            raise ValueError(
                f"Timeout not configured for operation: {operation}. "
                f"Valid types: {', '.join(valid_types)}"
            )

        return timeout_map[operation]

    @classmethod
    def get_timeout_ms(cls, operation: str | TimeoutType) -> int:
        """
        Get timeout value for an operation type in milliseconds.

        Args:
            operation: Operation type as string or TimeoutType enum

        Returns:
            Timeout in milliseconds (int)
        """
        return int(cls.get_timeout(operation) * 1000)

    @classmethod
    def get_all_timeouts(cls) -> dict[str, float]:
        """
        Get all configured timeouts for monitoring/documentation.

        Returns:
            Dictionary mapping operation names to timeout values (in seconds)
        """
        config = cls()
        return {
            "standard": config.standard_timeout_ms / 1000,
            "fast": config.fast_timeout_ms / 1000,
            "extended": config.extended_timeout_ms / 1000,
            "llm": config.llm_timeout_ms / 1000,
            "llm_stream": config.llm_stream_timeout_ms / 1000,
            "autodan": config.autodan_timeout_ms / 1000,
            "gptfuzz": config.gptfuzz_timeout_ms / 1000,
            "gradient": config.gradient_timeout_ms / 1000,
            "health_check": config.health_check_timeout_ms / 1000,
            "connection_test": config.connection_test_timeout_ms / 1000,
        }


# Convenience functions for common timeout operations


def get_standard_timeout() -> float:
    """Get standard timeout for regular API calls (30s)."""
    return TimeoutConfig.get_timeout(TimeoutType.STANDARD)


def get_fast_timeout() -> float:
    """Get fast timeout for quick operations (10s)."""
    return TimeoutConfig.get_timeout(TimeoutType.FAST)


def get_llm_timeout() -> float:
    """Get timeout for LLM provider calls (2min)."""
    return TimeoutConfig.get_timeout(TimeoutType.LLM)


def get_extended_timeout() -> float:
    """Get timeout for long-running operations (5min)."""
    return TimeoutConfig.get_timeout(TimeoutType.EXTENDED)


def get_autodan_timeout() -> float:
    """Get timeout for AutoDAN optimization (10min)."""
    return TimeoutConfig.get_timeout(TimeoutType.AUTODAN)


def get_health_check_timeout() -> float:
    """Get timeout for health check operations (5s)."""
    return TimeoutConfig.get_timeout(TimeoutType.HEALTH_CHECK)


__all__ = [
    "TimeoutConfig",
    "TimeoutType",
    "get_autodan_timeout",
    "get_extended_timeout",
    "get_fast_timeout",
    "get_health_check_timeout",
    "get_llm_timeout",
    "get_standard_timeout",
]
