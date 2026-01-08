/**
 * API Hooks
 * React hooks for API operations with loading states and error handling
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { ApiError, ApiResponse } from '../types';
import { logger } from '../logger';

// ============================================================================
// Types
// ============================================================================

export interface UseApiState<T> {
  data: T | null;
  isLoading: boolean;
  error: ApiError | null;
  isSuccess: boolean;
  isError: boolean;
}

export interface UseApiOptions {
  immediate?: boolean;
  onSuccess?: (data: any) => void;
  onError?: (error: ApiError) => void;
  retryCount?: number;
}

export interface UseApiResult<T, P extends any[]> extends UseApiState<T> {
  execute: (...params: P) => Promise<T | null>;
  reset: () => void;
  refetch: () => Promise<T | null>;
}

// ============================================================================
// Hooks
// ============================================================================

/**
 * Generic API hook for any async operation
 */
export function useApi<T, P extends any[] = []>(
  apiFn: (...params: P) => Promise<ApiResponse<T>>,
  options: UseApiOptions = {}
): UseApiResult<T, P> {
  const { immediate = false, onSuccess, onError, retryCount = 0 } = options;

  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    isLoading: false,
    error: null,
    isSuccess: false,
    isError: false,
  });

  const lastParamsRef = useRef<P | null>(null);
  const retriesRef = useRef(0);

  const execute = useCallback(
    async (...params: P): Promise<T | null> => {
      lastParamsRef.current = params;
      retriesRef.current = 0;

      setState((prev) => ({
        ...prev,
        isLoading: true,
        error: null,
        isError: false,
      }));

      try {
        const response = await apiFn(...params);

        setState({
          data: response.data,
          isLoading: false,
          error: null,
          isSuccess: true,
          isError: false,
        });

        onSuccess?.(response.data);
        return response.data;
      } catch (err) {
        const error = err as ApiError;

        // Retry logic
        if (retriesRef.current < retryCount) {
          retriesRef.current++;
          logger.logWarning('Retrying API call', {
            attempt: retriesRef.current,
            maxRetries: retryCount,
          });
          return execute(...params);
        }

        setState((prev) => ({
          ...prev,
          isLoading: false,
          error,
          isSuccess: false,
          isError: true,
        }));

        onError?.(error);
        logger.logError('API hook error', error);
        return null;
      }
    },
    [apiFn, onSuccess, onError, retryCount]
  );

  const refetch = useCallback(async (): Promise<T | null> => {
    if (lastParamsRef.current) {
      return execute(...lastParamsRef.current);
    }
    return null;
  }, [execute]);

  const reset = useCallback(() => {
    setState({
      data: null,
      isLoading: false,
      error: null,
      isSuccess: false,
      isError: false,
    });
    lastParamsRef.current = null;
  }, []);

  // Execute immediately if requested
  useEffect(() => {
    if (immediate) {
      execute(...([] as unknown as P));
    }
  }, [immediate, execute]);

  return {
    ...state,
    execute,
    reset,
    refetch,
  };
}

/**
 * Hook for mutations (POST, PUT, DELETE)
 */
export function useMutation<T, P extends any[] = []>(
  apiFn: (...params: P) => Promise<ApiResponse<T>>,
  options: UseApiOptions = {}
): UseApiResult<T, P> {
  return useApi(apiFn, { ...options, immediate: false });
}

/**
 * Hook for queries (GET) with automatic fetching
 */
export function useQuery<T>(
  apiFn: () => Promise<ApiResponse<T>>,
  options: UseApiOptions & { enabled?: boolean; refetchInterval?: number } = {}
): UseApiResult<T, []> {
  const { enabled = true, refetchInterval, ...apiOptions } = options;
  const result = useApi(apiFn, { ...apiOptions, immediate: false });
  const intervalRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  useEffect(() => {
    if (enabled) {
      result.execute();
    }
  }, [enabled]);

  // Refetch interval
  useEffect(() => {
    if (refetchInterval && enabled) {
      intervalRef.current = setInterval(() => {
        result.refetch();
      }, refetchInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [refetchInterval, enabled, result.refetch]);

  return result;
}

/**
 * Hook for paginated queries
 */
export function usePaginatedQuery<T>(
  apiFn: (page: number, pageSize: number) => Promise<ApiResponse<{
    items: T[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
  }>>,
  options: UseApiOptions & { pageSize?: number } = {}
) {
  const { pageSize = 20, ...apiOptions } = options;
  const [page, setPage] = useState(1);
  const [allItems, setAllItems] = useState<T[]>([]);

  const result = useApi(
    (p: number) => apiFn(p, pageSize),
    {
      ...apiOptions,
      onSuccess: (data) => {
        if (page === 1) {
          setAllItems(data.items);
        } else {
          setAllItems((prev) => [...prev, ...data.items]);
        }
        apiOptions.onSuccess?.(data);
      },
    }
  );

  const loadMore = useCallback(() => {
    if (result.data && page < result.data.total_pages && !result.isLoading) {
      setPage((p) => p + 1);
      result.execute(page + 1);
    }
  }, [result, page]);

  const refresh = useCallback(() => {
    setPage(1);
    setAllItems([]);
    result.execute(1);
  }, [result]);

  return {
    ...result,
    items: allItems,
    page,
    hasMore: result.data ? page < result.data.total_pages : false,
    loadMore,
    refresh,
  };
}

/**
 * Hook for optimistic updates
 */
export function useOptimisticMutation<T, P extends any[]>(
  apiFn: (...params: P) => Promise<ApiResponse<T>>,
  options: UseApiOptions & {
    optimisticUpdate: (params: P) => T;
    rollback: (error: ApiError, params: P) => void;
  }
) {
  const { optimisticUpdate, rollback, ...apiOptions } = options;
  const [optimisticData, setOptimisticData] = useState<T | null>(null);

  const result = useApi(apiFn, {
    ...apiOptions,
    onError: (error) => {
      // Rollback on error
      if (optimisticData) {
        rollback(error, [] as unknown as P);
        setOptimisticData(null);
      }
      apiOptions.onError?.(error);
    },
  });

  const executeWithOptimism = useCallback(
    async (...params: P): Promise<T | null> => {
      // Apply optimistic update immediately
      const optimistic = optimisticUpdate(params);
      setOptimisticData(optimistic);

      const result_data = await result.execute(...params);

      if (result_data) {
        setOptimisticData(null);
      }

      return result_data;
    },
    [result, optimisticUpdate]
  );

  return {
    ...result,
    data: optimisticData || result.data,
    execute: executeWithOptimism,
    isOptimistic: !!optimisticData,
  };
}