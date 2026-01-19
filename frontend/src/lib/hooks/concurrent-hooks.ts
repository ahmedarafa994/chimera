'use client';

import {
  use,
  useTransition,
  useDeferredValue,
  useOptimistic,
  startTransition,
  type TransitionFunction
} from 'react';
import { useCallback, useMemo, useState } from 'react';

// Enhanced hook for transitions with better UX
interface UseEnhancedTransitionOptions {
  timeoutMs?: number;
  onStart?: () => void;
  onComplete?: () => void;
  onError?: (error: Error) => void;
}

export function useEnhancedTransition(options: UseEnhancedTransitionOptions = {}) {
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<Error | null>(null);
  const [isTimedOut, setIsTimedOut] = useState(false);

  const {
    timeoutMs = 5000,
    onStart,
    onComplete,
    onError
  } = options;

  const enhancedStartTransition = useCallback((fn: TransitionFunction) => {
    setError(null);
    setIsTimedOut(false);
    onStart?.();

    // Set timeout for long-running transitions
    const timeoutId = setTimeout(() => {
      setIsTimedOut(true);
    }, timeoutMs);

    startTransition(() => {
      try {
        Promise.resolve(fn()).then(() => {
          clearTimeout(timeoutId);
          onComplete?.();
        }).catch((err) => {
          clearTimeout(timeoutId);
          setError(err);
          onError?.(err);
        });
      } catch (err) {
        clearTimeout(timeoutId);
        const error = err instanceof Error ? err : new Error('Unknown error');
        setError(error);
        onError?.(error);
      }
    });
  }, [timeoutMs, onStart, onComplete, onError]);

  return {
    isPending,
    isTimedOut,
    error,
    startTransition: enhancedStartTransition,
  };
}

// Optimistic UI hook for better perceived performance
interface UseOptimisticStateOptions<T> {
  onOptimisticUpdate?: (optimisticValue: T) => void;
  onRevert?: (originalValue: T) => void;
  onSuccess?: (finalValue: T) => void;
}

export function useOptimisticState<T>(
  initialValue: T,
  options: UseOptimisticStateOptions<T> = {}
) {
  const [optimisticValue, addOptimistic] = useOptimistic(
    initialValue,
    (currentState: T, optimisticUpdate: T) => optimisticUpdate
  );

  const updateOptimistically = useCallback(
    async (
      newValue: T,
      asyncAction: () => Promise<T>
    ) => {
      // Immediately update the UI optimistically
      addOptimistic(newValue);
      options.onOptimisticUpdate?.(newValue);

      try {
        // Perform the actual async operation
        const result = await asyncAction();
        options.onSuccess?.(result);
        return result;
      } catch (error) {
        // Revert to original value on error
        addOptimistic(initialValue);
        options.onRevert?.(initialValue);
        throw error;
      }
    },
    [addOptimistic, initialValue, options]
  );

  return {
    value: optimisticValue,
    updateOptimistically,
  };
}

// Deferred value hook with loading states
export function useDeferredState<T>(value: T, delay = 300) {
  const deferredValue = useDeferredValue(value);
  const isPending = deferredValue !== value;

  return {
    deferredValue,
    isPending,
    isStale: isPending,
  };
}

// Concurrent data fetching with Suspense
export function useSuspenseData<T>(
  promise: Promise<T>,
  key: string
): T {
  // Cache promises to avoid re-creating them
  const cachedPromise = useMemo(() => promise, [key]);
  return use(cachedPromise);
}

// Smart batching for multiple state updates
export function useBatchedUpdates() {
  const batchUpdates = useCallback((fn: () => void) => {
    // React 19 automatically batches updates, but we can be explicit
    startTransition(fn);
  }, []);

  return { batchUpdates };
}

// Priority-based rendering
interface UsePriorityRenderingOptions {
  highPriority?: boolean;
  deferMs?: number;
}

export function usePriorityRendering<T>(
  value: T,
  options: UsePriorityRenderingOptions = {}
) {
  const { highPriority = false, deferMs = 100 } = options;

  // Always call useDeferredValue to avoid conditional hook rules violation
  const deferredValue = useDeferredValue(value);

  // Use immediate value for high priority, deferred for low priority
  const processedValue = highPriority ? value : deferredValue;
  const isPending = processedValue !== value;

  return {
    value: processedValue,
    isPending,
    isHighPriority: highPriority,
  };
}

// Concurrent search with debouncing
interface UseConcurrentSearchOptions {
  debounceMs?: number;
  minQueryLength?: number;
}

export function useConcurrentSearch<T>(
  searchFn: (query: string) => Promise<T[]>,
  options: UseConcurrentSearchOptions = {}
) {
  const { debounceMs = 300, minQueryLength = 2 } = options;

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<T[]>([]);
  const [error, setError] = useState<Error | null>(null);

  const deferredQuery = useDeferredValue(query);
  const { isPending, startTransition } = useEnhancedTransition({
    onError: setError,
  });

  const search = useCallback((searchQuery: string) => {
    setQuery(searchQuery);

    if (searchQuery.length < minQueryLength) {
      setResults([]);
      return;
    }

    startTransition(async () => {
      try {
        const searchResults = await searchFn(deferredQuery);
        setResults(searchResults);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Search failed'));
        setResults([]);
      }
    });
  }, [searchFn, minQueryLength, deferredQuery, startTransition]);

  return {
    query,
    results,
    error,
    isPending,
    search,
    isStale: query !== deferredQuery,
  };
}

// Resource preloading for better performance
export function useResourcePreloader() {
  const preloadedResources = useMemo(() => new Map<string, Promise<any>>(), []);

  const preload = useCallback(<T>(
    key: string,
    loader: () => Promise<T>
  ): Promise<T> => {
    if (!preloadedResources.has(key)) {
      preloadedResources.set(key, loader());
    }
    return preloadedResources.get(key)!;
  }, [preloadedResources]);

  const getPreloaded = useCallback(<T>(key: string): Promise<T> | null => {
    return preloadedResources.get(key) || null;
  }, [preloadedResources]);

  return { preload, getPreloaded };
}

// Concurrent form submission
interface UseConcurrentFormOptions {
  onSuccess?: (result: any) => void;
  onError?: (error: Error) => void;
  enableOptimistic?: boolean;
}

export function useConcurrentForm<T extends Record<string, any>>(
  initialValues: T,
  submitFn: (values: T) => Promise<any>,
  options: UseConcurrentFormOptions = {}
) {
  const { onSuccess, onError, enableOptimistic = true } = options;

  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState<Record<string, string | undefined>>({});

  const { isPending, startTransition } = useEnhancedTransition({
    onError,
  });

  const { value: optimisticValues, updateOptimistically } = useOptimisticState(
    values,
    {
      onSuccess: (finalValues) => {
        setValues(finalValues);
        onSuccess?.(finalValues);
      },
    }
  );

  const updateField = useCallback((field: keyof T, value: any) => {
    if (enableOptimistic) {
      const newValues = { ...optimisticValues, [field]: value };
      updateOptimistically(newValues, async () => newValues);
    } else {
      setValues(prev => ({ ...prev, [field]: value }));
    }

    // Clear field error when user starts typing
    if (errors[field as string]) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  }, [optimisticValues, updateOptimistically, enableOptimistic, errors]);

  const submit = useCallback(() => {
    startTransition(async () => {
      try {
        const result = await submitFn(enableOptimistic ? optimisticValues : values);
        setErrors({});
        onSuccess?.(result);
      } catch (error) {
        if (error instanceof Error) {
          setErrors({ form: error.message });
          onError?.(error);
        }
      }
    });
  }, [submitFn, enableOptimistic, optimisticValues, values, startTransition, onSuccess, onError]);

  return {
    values: enableOptimistic ? optimisticValues : values,
    errors,
    isPending,
    updateField,
    submit,
    setErrors,
  };
}