"""Circuit Breaker Pattern Implementation
Resilience pattern for external service calls (AI providers, APIs).

HIGH-001 FIX: This module now re-exports from the shared implementation
to maintain backward compatibility while using the consolidated code.

Usage:
    from app.core.circuit_breaker import circuit_breaker, CircuitState

    @circuit_breaker("gemini", failure_threshold=3, recovery_timeout=60)
    async def call_gemini_api(prompt: str):
        ...
"""

# Re-export from shared implementation for backward compatibility
from app.core.shared.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
    is_circuit_open,
)

# Legacy alias for backward compatibility
CircuitBreakerState = CircuitBreaker

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitBreakerRegistry",
    "CircuitBreakerState",  # Legacy alias
    "CircuitState",
    "circuit_breaker",
    "is_circuit_open",
]
