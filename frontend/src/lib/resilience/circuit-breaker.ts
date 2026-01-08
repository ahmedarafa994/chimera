/**
 * Circuit Breaker Pattern Implementation
 *
 * Prevents cascading failures by temporarily blocking requests to
 * failing services, allowing them time to recover.
 *
 * States:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Failures exceeded threshold, requests are blocked
 * - HALF_OPEN: Testing if service has recovered
 *
 * @module lib/resilience/circuit-breaker
 */

import { CircuitBreakerOpenError } from '../errors';

// ============================================================================
// Types
// ============================================================================

export enum CircuitState {
    CLOSED = 'CLOSED',
    OPEN = 'OPEN',
    HALF_OPEN = 'HALF_OPEN',
}

export interface CircuitBreakerConfig {
    /** Name/identifier for this circuit */
    name: string;
    /** Number of failures before opening circuit */
    failureThreshold: number;
    /** Time in ms before attempting to close circuit */
    resetTimeout: number;
    /** Number of successful requests needed to close circuit from half-open */
    successThreshold: number;
    /** Number of requests allowed in half-open state */
    halfOpenRequests: number;
    /** Time window for counting failures (ms) */
    failureWindow: number;
    /** Callback when state changes */
    onStateChange?: (from: CircuitState, to: CircuitState, name: string) => void;
    /** Callback when circuit opens */
    onOpen?: (name: string, failures: number) => void;
    /** Callback when circuit closes */
    onClose?: (name: string) => void;
}

export interface CircuitBreakerStats {
    name: string;
    state: CircuitState;
    failures: number;
    successes: number;
    totalRequests: number;
    lastFailureTime: number | null;
    lastSuccessTime: number | null;
    openedAt: number | null;
    halfOpenAt: number | null;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: Omit<CircuitBreakerConfig, 'name'> = {
    failureThreshold: 5,
    resetTimeout: 30000, // 30 seconds
    successThreshold: 3,
    halfOpenRequests: 3,
    failureWindow: 60000, // 1 minute
};

// ============================================================================
// Circuit Breaker Implementation
// ============================================================================

export class CircuitBreaker {
    private state: CircuitState = CircuitState.CLOSED;
    private failures: number = 0;
    private successes: number = 0;
    private totalRequests: number = 0;
    private halfOpenRequests: number = 0;
    private lastFailureTime: number | null = null;
    private lastSuccessTime: number | null = null;
    private openedAt: number | null = null;
    private halfOpenAt: number | null = null;
    private failureTimestamps: number[] = [];
    private config: CircuitBreakerConfig;

    constructor(config: Partial<CircuitBreakerConfig> & { name: string }) {
        this.config = {
            ...DEFAULT_CONFIG,
            ...config,
        };
    }

    /**
     * Get current circuit state
     */
    getState(): CircuitState {
        this.checkStateTransition();
        return this.state;
    }

    /**
     * Get circuit statistics
     */
    getStats(): CircuitBreakerStats {
        return {
            name: this.config.name,
            state: this.state,
            failures: this.failures,
            successes: this.successes,
            totalRequests: this.totalRequests,
            lastFailureTime: this.lastFailureTime,
            lastSuccessTime: this.lastSuccessTime,
            openedAt: this.openedAt,
            halfOpenAt: this.halfOpenAt,
        };
    }

    /**
   * Check if request is allowed
   */
    canRequest(): boolean {
        this.checkStateTransition();

        switch (this.state) {
            case CircuitState.CLOSED:
                return true;

            case CircuitState.OPEN:
                return false;

            case CircuitState.HALF_OPEN:
                return this.halfOpenRequests < this.config.halfOpenRequests;

            default:
                return false;
        }
    }

    /**
     * Check if request is allowed (alias for compatibility)
     */
    canExecute(): boolean {
        return this.canRequest();
    }

    /**
     * Execute a function through the circuit breaker
     */
    async execute<T>(fn: () => Promise<T>): Promise<T> {
        this.checkStateTransition();

        if (!this.canRequest()) {
            const retryAfter = this.getRetryAfter();
            throw new CircuitBreakerOpenError(this.config.name, retryAfter);
        }

        this.totalRequests++;

        if (this.state === CircuitState.HALF_OPEN) {
            this.halfOpenRequests++;
        }

        try {
            const result = await fn();
            this.recordSuccess();
            return result;
        } catch (error) {
            this.recordFailure();
            throw error;
        }
    }

    /**
     * Record a successful request
     */
    recordSuccess(): void {
        this.successes++;
        this.lastSuccessTime = Date.now();

        if (this.state === CircuitState.HALF_OPEN) {
            // Check if we've had enough successes to close
            if (this.successes >= this.config.successThreshold) {
                this.transitionTo(CircuitState.CLOSED);
            }
        } else if (this.state === CircuitState.CLOSED) {
            // Reset failure count on success in closed state
            this.failures = 0;
            this.failureTimestamps = [];
        }
    }

    /**
     * Record a failed request
     */
    recordFailure(): void {
        const now = Date.now();
        this.failures++;
        this.lastFailureTime = now;
        this.failureTimestamps.push(now);

        // Clean up old failure timestamps
        this.cleanupFailureTimestamps();

        if (this.state === CircuitState.HALF_OPEN) {
            // Any failure in half-open state opens the circuit
            this.transitionTo(CircuitState.OPEN);
        } else if (this.state === CircuitState.CLOSED) {
            // Check if failures in window exceed threshold
            const recentFailures = this.failureTimestamps.length;
            if (recentFailures >= this.config.failureThreshold) {
                this.transitionTo(CircuitState.OPEN);
            }
        }
    }

    /**
     * Manually reset the circuit breaker
     */
    reset(): void {
        this.transitionTo(CircuitState.CLOSED);
        this.failures = 0;
        this.successes = 0;
        this.halfOpenRequests = 0;
        this.failureTimestamps = [];
        this.openedAt = null;
        this.halfOpenAt = null;
    }

    /**
     * Force circuit to open state
     */
    trip(): void {
        this.transitionTo(CircuitState.OPEN);
    }

    /**
     * Get time until circuit might close (in seconds)
     */
    getRetryAfter(): number {
        if (this.state !== CircuitState.OPEN || !this.openedAt) {
            return 0;
        }

        const elapsed = Date.now() - this.openedAt;
        const remaining = Math.max(0, this.config.resetTimeout - elapsed);
        return Math.ceil(remaining / 1000);
    }

    // ============================================================================
    // Private Methods
    // ============================================================================

    private checkStateTransition(): void {
        if (this.state === CircuitState.OPEN && this.openedAt) {
            const elapsed = Date.now() - this.openedAt;
            if (elapsed >= this.config.resetTimeout) {
                this.transitionTo(CircuitState.HALF_OPEN);
            }
        }
    }

    private transitionTo(newState: CircuitState): void {
        if (this.state === newState) return;

        const oldState = this.state;
        this.state = newState;

        // Update timestamps
        if (newState === CircuitState.OPEN) {
            this.openedAt = Date.now();
            this.halfOpenAt = null;
            this.config.onOpen?.(this.config.name, this.failures);
        } else if (newState === CircuitState.HALF_OPEN) {
            this.halfOpenAt = Date.now();
            this.halfOpenRequests = 0;
            this.successes = 0;
        } else if (newState === CircuitState.CLOSED) {
            this.openedAt = null;
            this.halfOpenAt = null;
            this.failures = 0;
            this.successes = 0;
            this.halfOpenRequests = 0;
            this.failureTimestamps = [];
            this.config.onClose?.(this.config.name);
        }

        this.config.onStateChange?.(oldState, newState, this.config.name);
    }

    private cleanupFailureTimestamps(): void {
        const cutoff = Date.now() - this.config.failureWindow;
        this.failureTimestamps = this.failureTimestamps.filter(ts => ts > cutoff);
    }
}

// ============================================================================
// Circuit Breaker Registry
// ============================================================================

export class CircuitBreakerRegistryClass {
    private circuits: Map<string, CircuitBreaker> = new Map();
    private globalConfig: Partial<CircuitBreakerConfig> = {};
    private listeners: Set<(name: string, stats: CircuitBreakerStats) => void> = new Set();

    /**
     * Set global configuration for all new circuits
     */
    setGlobalConfig(config: Partial<CircuitBreakerConfig>): void {
        this.globalConfig = config;
    }

    /**
     * Get or create a circuit breaker
     */
    getCircuit(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
        let circuit = this.circuits.get(name);

        if (!circuit) {
            circuit = new CircuitBreaker({
                name,
                ...this.globalConfig,
                ...config,
            });
            this.circuits.set(name, circuit);
        }

        return circuit;
    }

    /**
     * Alias for getCircuit to maintain compatibility
     */
    get(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
        return this.getCircuit(name, config);
    }

    /**
     * Check if a circuit exists
     */
    hasCircuit(name: string): boolean {
        return this.circuits.has(name);
    }

    /**
     * Remove a circuit
     */
    removeCircuit(name: string): boolean {
        return this.circuits.delete(name);
    }

    /**
     * Get all circuit statistics
     */
    getAllStats(): CircuitBreakerStats[] {
        return Array.from(this.circuits.values()).map(c => c.getStats());
    }

    /**
     * Get stats for a specific circuit
     */
    getStats(name: string): CircuitBreakerStats | null {
        return this.circuits.get(name)?.getStats() ?? null;
    }

    /**
     * Reset all circuits
     */
    resetAll(): void {
        this.circuits.forEach(circuit => circuit.reset());
    }

    /**
     * Add listener for state changes
     */
    addListener(listener: (name: string, stats: CircuitBreakerStats) => void): () => void {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    /**
     * Force notify listeners (internal)
     */
    notifyListeners(name: string, stats: CircuitBreakerStats): void {
        this.listeners.forEach(l => l(name, stats));
    }
}

export const circuitBreakerRegistry = new CircuitBreakerRegistryClass();
export const CircuitBreakerRegistry = circuitBreakerRegistry;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Wrap a function with circuit breaker protection
 */
export function withCircuitBreaker<T extends unknown[], R>(
    circuitName: string,
    fn: (...args: T) => Promise<R>,
    config?: Partial<CircuitBreakerConfig>
): (...args: T) => Promise<R> {
    return async (...args: T): Promise<R> => {
        const circuit = circuitBreakerRegistry.getCircuit(circuitName, config);
        return circuit.execute(() => fn(...args));
    };
}

/**
 * Get snapshot of all circuit breaker stats
 */
export function getCircuitBreakerSnapshot(): CircuitBreakerStats[] {
    return circuitBreakerRegistry.getAllStats();
}

/**
 * Check if an error is a circuit breaker error
 */
export function isCircuitBreakerError(error: unknown): error is CircuitBreakerOpenError {
    return error instanceof CircuitBreakerOpenError;
}
