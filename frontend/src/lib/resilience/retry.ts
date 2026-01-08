/**
 * Retry Logic with Exponential Backoff
 *
 * Provides configurable retry mechanisms for API requests with
 * exponential backoff, jitter, and circuit breaker integration.
 *
 * @module lib/resilience/retry
 */

import { APIError, isRetryableError } from '../errors';

// ============================================================================
// Types
// ============================================================================

export interface RetryConfig {
    /** Maximum number of retry attempts */
    maxRetries: number;
    /** Base delay in milliseconds */
    baseDelay: number;
    /** Maximum delay in milliseconds */
    maxDelay: number;
    /** Backoff multiplier (default: 2 for exponential) */
    backoffMultiplier: number;
    /** Add random jitter to prevent thundering herd */
    useJitter: boolean;
    /** Jitter factor (0-1, default: 0.1 = 10%) */
    jitterFactor: number;
    /** Custom retry condition */
    shouldRetry?: (error: unknown, attempt: number) => boolean;
    /** Callback on each retry attempt */
    onRetry?: (error: unknown, attempt: number, delay: number) => void;
    /** Alias for shouldRetry for compatibility */
    isRetryable?: (error: unknown) => boolean;
    /** HTTP status codes to retry */
    retryableStatusCodes?: number[];
    /** AbortSignal for cancellation */
    signal?: AbortSignal;
}

export interface RetryState {
    attempt: number;
    totalDelay: number;
    errors: Error[];
    startTime: number;
}

export interface RetryResult<T> {
    success: boolean;
    data?: T;
    error?: Error;
    attempts: number;
    totalTime: number;
    retryState: RetryState;
    totalDelay?: number; // compat
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_RETRY_CONFIG: RetryConfig = {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 30000,
    backoffMultiplier: 2,
    useJitter: true,
    jitterFactor: 0.1,
    retryableStatusCodes: [408, 429, 500, 502, 503, 504],
};

// ============================================================================
// Core Logic
// ============================================================================

/**
 * Calculate delay for a given retry attempt using exponential backoff
 */
export function calculateBackoffDelay(
    attempt: number,
    config: RetryConfig
): number {
    const exponentialDelay = config.baseDelay * Math.pow(config.backoffMultiplier, attempt);
    let delay = Math.min(exponentialDelay, config.maxDelay);

    if (config.useJitter) {
        const jitter = delay * config.jitterFactor * (Math.random() * 2 - 1);
        delay = Math.max(0, delay + jitter);
    }

    return Math.round(delay);
}

/**
 * Promise-based sleep function
 */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
        if (signal?.aborted) {
            reject(new Error('Sleep aborted'));
            return;
        }
        const timeoutId = setTimeout(resolve, ms);
        signal?.addEventListener('abort', () => {
            clearTimeout(timeoutId);
            reject(new Error('Sleep aborted'));
        });
    });
}

/**
 * Execute a function with retry logic
 */
export async function withRetryResult<T>(
    fn: () => Promise<T>,
    config: Partial<RetryConfig> = {}
): Promise<RetryResult<T>> {
    const fullConfig: RetryConfig = {
        ...DEFAULT_RETRY_CONFIG,
        ...config,
    };

    const state: RetryState = {
        attempt: 0,
        totalDelay: 0,
        errors: [],
        startTime: Date.now(),
    };

    while (state.attempt <= fullConfig.maxRetries) {
        if (fullConfig.signal?.aborted) {
            return {
                success: false,
                error: new Error('Retry aborted'),
                attempts: state.attempt + 1,
                totalTime: Date.now() - state.startTime,
                retryState: state,
                totalDelay: state.totalDelay
            };
        }

        try {
            const data = await fn();

            return {
                success: true,
                data,
                attempts: state.attempt + 1,
                totalTime: Date.now() - state.startTime,
                retryState: state,
                totalDelay: state.totalDelay
            };
        } catch (error) {
            const err = error instanceof Error ? error : new Error(String(error));
            state.errors.push(err);

            // Determine if we should retry
            const shouldRetry = fullConfig.shouldRetry
                ? fullConfig.shouldRetry(error, state.attempt)
                : fullConfig.isRetryable
                    ? fullConfig.isRetryable(error) && state.attempt < fullConfig.maxRetries
                    : defaultShouldRetry(error, state.attempt, fullConfig);

            if (!shouldRetry) {
                return {
                    success: false,
                    error: err,
                    attempts: state.attempt + 1,
                    totalTime: Date.now() - state.startTime,
                    retryState: state,
                    totalDelay: state.totalDelay
                };
            }

            // Calculate delay
            let delay = calculateBackoffDelay(state.attempt, fullConfig);

            // Handle Retry-After if available
            if (error instanceof APIError && (error as any).retryAfter) {
                delay = Math.min((error as any).retryAfter * 1000, fullConfig.maxDelay);
            }

            fullConfig.onRetry?.(error, state.attempt, delay);

            try {
                await sleep(delay, fullConfig.signal);
            } catch {
                return {
                    success: false,
                    error: new Error('Retry aborted'),
                    attempts: state.attempt + 1,
                    totalTime: Date.now() - state.startTime,
                    retryState: state,
                    totalDelay: state.totalDelay
                };
            }

            state.totalDelay += delay;
            state.attempt++;
        }
    }

    return {
        success: false,
        error: state.errors[state.errors.length - 1] || new Error('Max retries exceeded'),
        attempts: state.attempt,
        totalTime: Date.now() - state.startTime,
        retryState: state,
        totalDelay: state.totalDelay
    };
}

/**
 * Execute a function with retry logic, throwing on final failure
 */
export async function withRetry<T>(
    fn: () => Promise<T>,
    config: Partial<RetryConfig> = {}
): Promise<T> {
    const result = await withRetryResult(fn, config);
    if (result.success) return result.data as T;
    throw result.error;
}

function defaultShouldRetry(error: unknown, attempt: number, config: RetryConfig): boolean {
    if (attempt >= config.maxRetries) return false;

    // Check status codes if configured
    if (error instanceof APIError && config.retryableStatusCodes?.includes(error.statusCode)) {
        return true;
    }

    // Use isRetryableError from centralized error system
    if (isRetryableError(error)) return true;

    // Check generic network errors
    if (error instanceof Error) {
        const message = error.message.toLowerCase();
        return (
            message.includes('network') ||
            message.includes('timeout') ||
            message.includes('econnrefused') ||
            message.includes('econnreset')
        );
    }

    return false;
}

/** Specialized helpers for compatibility */
export async function withImmediateRetry<T>(fn: () => Promise<T>, config: Partial<RetryConfig> = {}): Promise<T> {
    return withRetry(fn, { ...config, maxRetries: 1, baseDelay: 0 });
}

export async function withAggressiveBackoff<T>(fn: () => Promise<T>, config: Partial<RetryConfig> = {}): Promise<T> {
    return withRetry(fn, { ...config, maxRetries: 5, baseDelay: 2000, multiplier: 3 } as any);
}

export async function withLinearBackoff<T>(fn: () => Promise<T>, config: Partial<RetryConfig> = {}): Promise<T> {
    return withRetry(fn, { ...config, backoffMultiplier: 1 });
}

export function createQueryRetry(maxRetries: number = 3) {
    return (failureCount: number, error: unknown) => failureCount < maxRetries && defaultShouldRetry(error, failureCount, { maxRetries } as any);
}

export function createQueryRetryDelay(baseDelay: number = 1000, maxDelay: number = 30000) {
    return (failureCount: number, error: unknown) => calculateBackoffDelay(failureCount, { baseDelay, maxDelay, backoffMultiplier: 2, useJitter: true, jitterFactor: 0.1 } as any);
}
