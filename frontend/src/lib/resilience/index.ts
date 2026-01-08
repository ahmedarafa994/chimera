/**
 * Resilience Module for Project Chimera Frontend
 * Barrel export for circuit breaker and retry utilities
 * 
 * Usage:
 *   import { 
 *     CircuitBreakerRegistry, 
 *     withRetry, 
 *     withCircuitBreaker 
 *   } from '@/lib/resilience';
 */

// ============================================================================
// Circuit Breaker
// ============================================================================

export {
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    withCircuitBreaker,
    isCircuitBreakerError,
    getCircuitBreakerSnapshot,
    type CircuitBreakerConfig,
    type CircuitBreakerStats,
} from "./circuit-breaker";

// ============================================================================
// Retry Logic
// ============================================================================

export {
    withRetry,
    withRetryResult,
    withImmediateRetry,
    withAggressiveBackoff,
    withLinearBackoff,
    createQueryRetry,
    createQueryRetryDelay,
    type RetryConfig,
    type RetryResult,
} from "./retry";
