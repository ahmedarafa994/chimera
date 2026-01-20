"""Shared core utilities that can be used across Chimera components.

This module contains reusable patterns and utilities that are designed
to be consistent across backend-api, chimera-orchestrator, and chimera-agent.

Includes:
- Circuit Breaker pattern for resilience
- Structured logging with correlation IDs
- Request context tracking
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
    is_circuit_open,
)
from .structured_logging import (
    HumanReadableFormatter,
    RequestContextMiddleware,
    StructuredLogFormatter,
    configure_logging,
    generate_correlation_id,
    get_correlation_id,
    get_logger,
    get_request_context,
    log_execution_time,
    log_with_context,
    set_correlation_id,
    set_request_context,
    update_request_context,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitBreakerRegistry",
    "CircuitState",
    "HumanReadableFormatter",
    "RequestContextMiddleware",
    "StructuredLogFormatter",
    "circuit_breaker",
    # Structured Logging
    "configure_logging",
    "generate_correlation_id",
    "get_correlation_id",
    "get_logger",
    "get_request_context",
    "is_circuit_open",
    "log_execution_time",
    "log_with_context",
    "set_correlation_id",
    "set_request_context",
    "update_request_context",
]
