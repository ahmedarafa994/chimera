/**
 * Debounce Utility for Project Chimera Frontend
 * Provides request debouncing with cancellation support
 *
 * @module lib/utils/debounce
 */

// ============================================================================
// Types
// ============================================================================

export interface DebouncedFunction<TArgs, TResult> {
    (...args: TArgs[]): Promise<TResult>;
    cancel: () => void;
    flush: () => Promise<TResult | undefined>;
    pending: () => boolean;
}

export interface DebounceOptions {
    /** Delay in milliseconds before executing */
    delay: number;
    /** Execute on leading edge instead of trailing */
    leading?: boolean;
    /** Maximum time to wait before forcing execution */
    maxWait?: number;
}

// ============================================================================
// Request Deduplication Cache
// ============================================================================

/** Cache for in-flight requests to prevent duplicate calls */
const inFlightRequests = new Map<string, Promise<unknown>>();

/**
 * Generate a cache key from function arguments
 */
export function generateCacheKey(args: unknown[]): string {
    return JSON.stringify(args);
}

/**
 * Deduplicate concurrent identical requests
 * If the same request is already in-flight, return the existing promise
 */
export async function deduplicateRequest<T>(
    key: string,
    fn: () => Promise<T>
): Promise<T> {
    // Check if request is already in-flight
    const existing = inFlightRequests.get(key);
    if (existing) {
        return existing as Promise<T>;
    }

    // Create new request and cache it
    const promise = fn().finally(() => {
        inFlightRequests.delete(key);
    });

    inFlightRequests.set(key, promise);
    return promise;
}

// ============================================================================
// Debounce Implementation
// ============================================================================

/**
 * Creates a debounced version of an async function.
 *
 * Features:
 * - Configurable delay per call
 * - Cancellation support
 * - Flush to execute immediately
 * - Leading edge execution option
 * - Maximum wait time option
 *
 * @example
 * ```typescript
 * const debouncedSearch = debounce(
 *   async (query: string) => api.search(query),
 *   { delay: 300, maxWait: 1000 }
 * );
 *
 * // Only the last call within 300ms will execute
 * debouncedSearch("h");
 * debouncedSearch("he");
 * debouncedSearch("hel");
 * debouncedSearch("hello"); // This one executes
 * ```
 */
export function debounce<TArgs, TResult>(
    fn: (...args: TArgs[]) => Promise<TResult>,
    options: DebounceOptions
): DebouncedFunction<TArgs, TResult> {
    const { delay, leading = false, maxWait } = options;

    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let maxWaitTimeoutId: ReturnType<typeof setTimeout> | null = null;
    let lastArgs: TArgs[] | null = null;
    let lastCallTime: number | null = null;
    let pendingPromise: Promise<TResult> | null = null;
    let resolveRef: ((value: TResult) => void) | null = null;
    let rejectRef: ((error: unknown) => void) | null = null;

    const clearTimers = () => {
        if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = null;
        }
        if (maxWaitTimeoutId) {
            clearTimeout(maxWaitTimeoutId);
            maxWaitTimeoutId = null;
        }
    };

    const execute = async (): Promise<TResult> => {
        const args = lastArgs;
        lastArgs = null;
        clearTimers();

        if (!args) {
            throw new Error("No arguments to execute");
        }

        try {
            const result = await fn(...args);
            resolveRef?.(result);
            return result;
        } catch (error) {
            rejectRef?.(error);
            throw error;
        } finally {
            pendingPromise = null;
            resolveRef = null;
            rejectRef = null;
        }
    };

    const debouncedFn = (...args: TArgs[]): Promise<TResult> => {
        lastArgs = args;
        lastCallTime = Date.now();

        // Create promise if not exists
        if (!pendingPromise) {
            pendingPromise = new Promise<TResult>((resolve, reject) => {
                resolveRef = resolve;
                rejectRef = reject;
            });
        }

        // Execute on leading edge if configured
        if (leading && !timeoutId) {
            execute();
            return pendingPromise;
        }

        // Clear existing timeout and set new one
        if (timeoutId) {
            clearTimeout(timeoutId);
        }

        timeoutId = setTimeout(() => {
            execute();
        }, delay);

        // Set up max wait if configured
        if (maxWait && !maxWaitTimeoutId) {
            maxWaitTimeoutId = setTimeout(() => {
                if (lastArgs) {
                    execute();
                }
            }, maxWait);
        }

        return pendingPromise;
    };

    debouncedFn.cancel = () => {
        clearTimers();
        lastArgs = null;
        pendingPromise = null;
        resolveRef = null;
        rejectRef = null;
    };

    debouncedFn.flush = async (): Promise<TResult | undefined> => {
        if (lastArgs) {
            return execute();
        }
        return undefined;
    };

    debouncedFn.pending = (): boolean => {
        return lastArgs !== null;
    };

    return debouncedFn;
}

// ============================================================================
// Throttle Implementation
// ============================================================================

export interface ThrottleOptions {
    /** Minimum time between executions in milliseconds */
    interval: number;
    /** Execute on leading edge */
    leading?: boolean;
    /** Execute on trailing edge */
    trailing?: boolean;
}

/**
 * Creates a throttled version of an async function.
 * Limits execution to at most once per interval.
 */
export function throttle<TArgs, TResult>(
    fn: (...args: TArgs[]) => Promise<TResult>,
    options: ThrottleOptions
): DebouncedFunction<TArgs, TResult> {
    const { interval, leading = true, trailing = true } = options;

    let lastExecuteTime = 0;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let lastArgs: TArgs[] | null = null;
    let pendingPromise: Promise<TResult> | null = null;
    let resolveRef: ((value: TResult) => void) | null = null;
    let rejectRef: ((error: unknown) => void) | null = null;

    const execute = async (): Promise<TResult> => {
        const args = lastArgs;
        lastArgs = null;
        lastExecuteTime = Date.now();

        if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = null;
        }

        if (!args) {
            throw new Error("No arguments to execute");
        }

        try {
            const result = await fn(...args);
            resolveRef?.(result);
            return result;
        } catch (error) {
            rejectRef?.(error);
            throw error;
        } finally {
            pendingPromise = null;
            resolveRef = null;
            rejectRef = null;
        }
    };

    const throttledFn = (...args: TArgs[]): Promise<TResult> => {
        lastArgs = args;
        const now = Date.now();
        const timeSinceLastExecution = now - lastExecuteTime;

        // Create promise if not exists
        if (!pendingPromise) {
            pendingPromise = new Promise<TResult>((resolve, reject) => {
                resolveRef = resolve;
                rejectRef = reject;
            });
        }

        // Execute immediately if enough time has passed and leading is true
        if (timeSinceLastExecution >= interval && leading) {
            execute();
            return pendingPromise;
        }

        // Schedule trailing execution
        if (trailing && !timeoutId) {
            const remainingTime = interval - timeSinceLastExecution;
            timeoutId = setTimeout(() => {
                if (lastArgs) {
                    execute();
                }
            }, Math.max(0, remainingTime));
        }

        return pendingPromise;
    };

    throttledFn.cancel = () => {
        if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = null;
        }
        lastArgs = null;
        pendingPromise = null;
        resolveRef = null;
        rejectRef = null;
    };

    throttledFn.flush = async (): Promise<TResult | undefined> => {
        if (lastArgs) {
            return execute();
        }
        return undefined;
    };

    throttledFn.pending = (): boolean => {
        return lastArgs !== null;
    };

    return throttledFn;
}

// ============================================================================
// Endpoint-Specific Debounce Configurations
// ============================================================================

/** Pre-configured delays for different endpoint types */
export const DEBOUNCE_DELAYS = {
    /** Fast typing inputs (search, autocomplete) */
    TYPING: 150,
    /** Standard form inputs */
    INPUT: 300,
    /** API calls that can be costly */
    API_CALL: 500,
    /** AI generation requests */
    AI_GENERATION: 1000,
    /** Long-running operations */
    HEAVY_OPERATION: 2000,
} as const;
