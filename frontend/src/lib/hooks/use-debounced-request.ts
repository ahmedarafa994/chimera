/**
 * Debounced Request Hook for Project Chimera Frontend
 * React hook for making debounced API calls
 *
 * @module lib/hooks/use-debounced-request
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { debounce, type DebouncedFunction, type DebounceOptions } from "../utils/debounce";

// ============================================================================
// Types
// ============================================================================

export interface UseDebouncedRequestOptions<TArgs, TResult> extends Omit<DebounceOptions, "delay"> {
    /** Delay in milliseconds */
    delay?: number;
    /** Callback when request starts */
    onStart?: () => void;
    /** Callback when request succeeds */
    onSuccess?: (result: TResult) => void;
    /** Callback when request fails */
    onError?: (error: Error) => void;
    /** Callback when request completes (success or error) */
    onSettled?: () => void;
    /** Whether to reset error state on new request */
    resetErrorOnRequest?: boolean;
}

export interface UseDebouncedRequestResult<TArgs, TResult> {
    /** Execute the debounced request */
    execute: (...args: TArgs[]) => Promise<TResult>;
    /** Current loading state */
    isLoading: boolean;
    /** Current error state */
    error: Error | null;
    /** Current result data */
    data: TResult | null;
    /** Cancel pending request */
    cancel: () => void;
    /** Flush pending request immediately */
    flush: () => Promise<TResult | undefined>;
    /** Whether there's a pending request */
    isPending: boolean;
    /** Reset the hook state */
    reset: () => void;
}

// ============================================================================
// Default Options
// ============================================================================

const DEFAULT_OPTIONS: Required<Pick<UseDebouncedRequestOptions<unknown, unknown>, "delay" | "resetErrorOnRequest">> = {
    delay: 300,
    resetErrorOnRequest: true,
};

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * React hook for making debounced API calls with loading and error states
 *
 * @example
 * ```tsx
 * const { execute, isLoading, error, data } = useDebouncedRequest(
 *   async (query: string) => api.search(query),
 *   { delay: 300, onSuccess: (results) => console.log(results) }
 * );
 *
 * // In your component
 * <input onChange={(e) => execute(e.target.value)} />
 * {isLoading && <Spinner />}
 * {error && <Error message={error.message} />}
 * {data && <Results items={data} />}
 * ```
 */
export function useDebouncedRequest<TArgs, TResult>(
    requestFn: (...args: TArgs[]) => Promise<TResult>,
    options: UseDebouncedRequestOptions<TArgs, TResult> = {}
): UseDebouncedRequestResult<TArgs, TResult> {
    const {
        delay = DEFAULT_OPTIONS.delay,
        leading,
        maxWait,
        onStart,
        onSuccess,
        onError,
        onSettled,
        resetErrorOnRequest = DEFAULT_OPTIONS.resetErrorOnRequest,
    } = options;

    // State
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<Error | null>(null);
    const [data, setData] = useState<TResult | null>(null);
    const [isPending, setIsPending] = useState(false);

    // Refs for callbacks to avoid recreating debounced function
    const callbacksRef = useRef({ onStart, onSuccess, onError, onSettled });
    callbacksRef.current = { onStart, onSuccess, onError, onSettled };

    // Create debounced function
    const debouncedFnRef = useRef<DebouncedFunction<TArgs, TResult> | null>(null);

    useEffect(() => {
        debouncedFnRef.current = debounce<TArgs, TResult>(
            async (...args: TArgs[]) => {
                setIsPending(false);
                setIsLoading(true);
                if (resetErrorOnRequest) {
                    setError(null);
                }
                callbacksRef.current.onStart?.();

                try {
                    const result = await requestFn(...args);
                    setData(result);
                    callbacksRef.current.onSuccess?.(result);
                    return result;
                } catch (err) {
                    const error = err instanceof Error ? err : new Error(String(err));
                    setError(error);
                    callbacksRef.current.onError?.(error);
                    throw error;
                } finally {
                    setIsLoading(false);
                    callbacksRef.current.onSettled?.();
                }
            },
            { delay, leading, maxWait }
        );

        return () => {
            debouncedFnRef.current?.cancel();
        };
    }, [requestFn, delay, leading, maxWait, resetErrorOnRequest]);

    // Execute function
    const execute = useCallback((...args: TArgs[]): Promise<TResult> => {
        setIsPending(true);
        if (!debouncedFnRef.current) {
            return Promise.reject(new Error("Debounced function not initialized"));
        }
        return debouncedFnRef.current(...args);
    }, []);

    // Cancel function
    const cancel = useCallback(() => {
        debouncedFnRef.current?.cancel();
        setIsPending(false);
        setIsLoading(false);
    }, []);

    // Flush function
    const flush = useCallback(async (): Promise<TResult | undefined> => {
        return debouncedFnRef.current?.flush();
    }, []);

    // Reset function
    const reset = useCallback(() => {
        cancel();
        setData(null);
        setError(null);
    }, [cancel]);

    return {
        execute,
        isLoading,
        error,
        data,
        cancel,
        flush,
        isPending,
        reset,
    };
}

// ============================================================================
// Additional Hooks
// ============================================================================

/**
 * Hook for debounced input with optimistic updates
 */
export function useDebouncedInput<T>(
    initialValue: T,
    onDebouncedChange: (value: T) => void | Promise<void>,
    delay: number = 300
): [T, (value: T) => void, T] {
    const [immediateValue, setImmediateValue] = useState<T>(initialValue);
    const [debouncedValue, setDebouncedValue] = useState<T>(initialValue);

    const debouncedSetValue = useRef(
        debounce(
            async (value: T) => {
                setDebouncedValue(value);
                await onDebouncedChange(value);
            },
            { delay }
        )
    ).current;

    const setValue = useCallback((value: T) => {
        setImmediateValue(value);
        debouncedSetValue(value);
    }, [debouncedSetValue]);

    useEffect(() => {
        return () => {
            debouncedSetValue.cancel();
        };
    }, [debouncedSetValue]);

    return [immediateValue, setValue, debouncedValue];
}
