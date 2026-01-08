/**
 * Optimistic Updates
 * 
 * Provides optimistic UI update patterns with rollback:
 * - Mutation tracking
 * - Automatic rollback on failure
 * - Retry with exponential backoff
 * - React Query integration
 * - Zustand integration
 * 
 * @module lib/sync/optimistic-updates
 */

import { useState, useCallback, useRef } from 'react';
import { useMutation, useQueryClient, QueryKey } from '@tanstack/react-query';
import type {
  OptimisticUpdate,
  MutationOptions,
  RetryConfig,
  RetryState,
  CircuitBreakerConfig,
  CircuitBreakerState,
  CircuitState,
} from './types';
import { getEventBus } from './event-bus';

// ============================================================================
// Default Configurations
// ============================================================================

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxAttempts: 3,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  jitter: 0.1,
  retryableErrors: ['NETWORK_ERROR', 'TIMEOUT', '5XX'],
};

const DEFAULT_CIRCUIT_BREAKER_CONFIG: CircuitBreakerConfig = {
  failureThreshold: 5,
  successThreshold: 2,
  resetTimeout: 30000,
  failureWindow: 60000,
};

// ============================================================================
// Optimistic Update Manager
// ============================================================================

export class OptimisticUpdateManager<T = unknown> {
  private updates = new Map<string, OptimisticUpdate<T>>();
  private eventBus = getEventBus();

  /**
   * Create an optimistic update
   */
  create(
    id: string,
    optimisticData: T,
    originalData: T,
    mutation: () => Promise<T>,
    rollback: () => void
  ): OptimisticUpdate<T> {
    const update: OptimisticUpdate<T> = {
      id,
      timestamp: Date.now(),
      optimisticData,
      originalData,
      mutation,
      rollback,
      status: 'pending',
    };

    this.updates.set(id, update);
    this.eventBus.emit('optimistic:created', { id });

    return update;
  }

  /**
   * Execute the mutation
   */
  async execute(id: string): Promise<T> {
    const update = this.updates.get(id);
    if (!update) {
      throw new Error(`Optimistic update ${id} not found`);
    }

    try {
      const result = await update.mutation();
      update.status = 'committed';
      this.updates.set(id, update);
      this.eventBus.emit('optimistic:committed', { id });
      return result;
    } catch (error) {
      this.rollback(id);
      throw error;
    }
  }

  /**
   * Rollback an update
   */
  rollback(id: string): void {
    const update = this.updates.get(id);
    if (!update) return;

    update.rollback();
    update.status = 'rolled_back';
    this.updates.set(id, update);
    this.eventBus.emit('optimistic:rolled_back', { id });
  }

  /**
   * Get update by ID
   */
  get(id: string): OptimisticUpdate<T> | undefined {
    return this.updates.get(id);
  }

  /**
   * Get all pending updates
   */
  getPending(): OptimisticUpdate<T>[] {
    return Array.from(this.updates.values()).filter(
      (u) => u.status === 'pending'
    );
  }

  /**
   * Clear completed updates
   */
  clearCompleted(): void {
    this.updates.forEach((update, id) => {
      if (update.status !== 'pending') {
        this.updates.delete(id);
      }
    });
  }

  /**
   * Clear all updates
   */
  clear(): void {
    this.updates.clear();
  }
}

// ============================================================================
// Retry Logic
// ============================================================================

/**
 * Calculate retry delay with exponential backoff and jitter
 */
export function calculateRetryDelay(
  attempt: number,
  config: RetryConfig = DEFAULT_RETRY_CONFIG
): number {
  const exponentialDelay = config.baseDelay * Math.pow(config.backoffMultiplier, attempt - 1);
  const cappedDelay = Math.min(exponentialDelay, config.maxDelay);
  
  // Add jitter
  const jitterRange = cappedDelay * config.jitter;
  const jitter = Math.random() * jitterRange * 2 - jitterRange;
  
  return Math.max(0, cappedDelay + jitter);
}

/**
 * Check if error is retryable
 */
export function isRetryableError(
  error: Error,
  config: RetryConfig = DEFAULT_RETRY_CONFIG
): boolean {
  if (config.shouldRetry) {
    return config.shouldRetry(error, 0);
  }

  const errorWithCode = error as Error & { code?: string; status?: number; statusCode?: number };
  const errorCode = errorWithCode.code || '';
  const statusCode = errorWithCode.status || errorWithCode.statusCode || 0;

  // Check against retryable error codes
  for (const retryable of config.retryableErrors || []) {
    if (retryable === errorCode) return true;
    if (retryable === 'NETWORK_ERROR' && errorCode === 'ECONNREFUSED') return true;
    if (retryable === 'TIMEOUT' && errorCode === 'ETIMEDOUT') return true;
    if (retryable === '5XX' && statusCode >= 500 && statusCode < 600) return true;
  }

  return false;
}

/**
 * Execute with retry
 */
export async function executeWithRetry<T>(
  fn: () => Promise<T>,
  config: RetryConfig = DEFAULT_RETRY_CONFIG,
  onRetry?: (state: RetryState) => void
): Promise<T> {
  let lastError: Error | undefined;
  let totalDelay = 0;

  for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === config.maxAttempts || !isRetryableError(lastError, config)) {
        throw lastError;
      }

      const delay = calculateRetryDelay(attempt, config);
      totalDelay += delay;

      const state: RetryState = {
        attempt,
        nextDelay: delay,
        totalDelay,
        lastError,
      };

      onRetry?.(state);

      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

// ============================================================================
// Circuit Breaker
// ============================================================================

export class CircuitBreaker {
  private config: CircuitBreakerConfig;
  private state: CircuitBreakerState;
  private failures: Date[] = [];
  private eventBus = getEventBus();

  constructor(config: CircuitBreakerConfig = DEFAULT_CIRCUIT_BREAKER_CONFIG) {
    this.config = config;
    this.state = {
      state: 'closed',
      failures: 0,
      successes: 0,
    };
  }

  /**
   * Get current state
   */
  getState(): CircuitBreakerState {
    return { ...this.state };
  }

  /**
   * Check if circuit allows execution
   */
  canExecute(): boolean {
    this.cleanOldFailures();

    switch (this.state.state) {
      case 'closed':
        return true;
      case 'open':
        if (this.state.nextAttempt && new Date() >= this.state.nextAttempt) {
          this.transitionTo('half-open');
          return true;
        }
        return false;
      case 'half-open':
        return true;
      default:
        return false;
    }
  }

  /**
   * Record success
   */
  recordSuccess(): void {
    this.state.successes++;
    this.state.lastSuccess = new Date();

    if (this.state.state === 'half-open') {
      if (this.state.successes >= this.config.successThreshold) {
        this.transitionTo('closed');
      }
    } else if (this.state.state === 'closed') {
      // Reset failure count on success
      this.state.failures = 0;
      this.failures = [];
    }
  }

  /**
   * Record failure
   */
  recordFailure(): void {
    this.state.failures++;
    this.state.lastFailure = new Date();
    this.failures.push(new Date());

    if (this.state.state === 'half-open') {
      this.transitionTo('open');
    } else if (this.state.state === 'closed') {
      this.cleanOldFailures();
      if (this.failures.length >= this.config.failureThreshold) {
        this.transitionTo('open');
      }
    }
  }

  /**
   * Execute with circuit breaker
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (!this.canExecute()) {
      throw new Error('Circuit breaker is open');
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
   * Reset circuit breaker
   */
  reset(): void {
    this.state = {
      state: 'closed',
      failures: 0,
      successes: 0,
    };
    this.failures = [];
  }

  private transitionTo(newState: CircuitState): void {
    const oldState = this.state.state;
    this.state.state = newState;

    if (newState === 'open') {
      this.state.nextAttempt = new Date(Date.now() + this.config.resetTimeout);
      this.state.successes = 0;
    } else if (newState === 'closed') {
      this.state.failures = 0;
      this.state.successes = 0;
      this.failures = [];
      this.state.nextAttempt = undefined;
    } else if (newState === 'half-open') {
      this.state.successes = 0;
    }

    this.eventBus.emit('circuit-breaker:state-changed', {
      from: oldState,
      to: newState,
    });
  }

  private cleanOldFailures(): void {
    const cutoff = Date.now() - this.config.failureWindow;
    this.failures = this.failures.filter((f) => f.getTime() > cutoff);
    this.state.failures = this.failures.length;
  }
}

// ============================================================================
// React Hooks
// ============================================================================

/**
 * Hook for optimistic mutations
 */
export function useOptimisticUpdate<TData, TVariables>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options: MutationOptions<TData, TVariables> & {
    onOptimisticUpdate?: (variables: TVariables) => TData;
  } = {}
): {
  mutate: (variables: TVariables) => Promise<TData>;
  mutateAsync: (variables: TVariables) => Promise<TData>;
  isLoading: boolean;
  error: Error | null;
  reset: () => void;
} {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [optimisticData, setOptimisticData] = useState<TData | null>(null);
  const originalDataRef = useRef<TData | null>(null);

  const mutate = useCallback(
    async (variables: TVariables): Promise<TData> => {
      setIsLoading(true);
      setError(null);

      // Apply optimistic update
      if (options.onOptimisticUpdate) {
        const optimistic = options.onOptimisticUpdate(variables);
        originalDataRef.current = optimisticData;
        setOptimisticData(optimistic);
      }

      try {
        const result = await mutationFn(variables);
        setOptimisticData(result);
        options.onSuccess?.(result, variables);
        return result;
      } catch (err) {
        const error = err as Error;
        setError(error);

        // Rollback optimistic update
        if (originalDataRef.current !== null) {
          setOptimisticData(originalDataRef.current);
          options.onRollback?.(error, variables);
        }

        options.onError?.(error, variables);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    [mutationFn, options, optimisticData]
  );

  const reset = useCallback(() => {
    setIsLoading(false);
    setError(null);
    setOptimisticData(null);
    originalDataRef.current = null;
  }, []);

  return {
    mutate,
    mutateAsync: mutate,
    isLoading,
    error,
    reset,
  };
}

/**
 * Hook for React Query optimistic mutations
 */
export function useQueryOptimisticMutation<TData, TVariables, TError = Error>(
  queryKey: QueryKey,
  mutationFn: (variables: TVariables) => Promise<TData>,
  options: {
    optimisticUpdate: (variables: TVariables, oldData: TData | undefined) => TData;
    onSuccess?: (data: TData, variables: TVariables) => void;
    onError?: (error: TError, variables: TVariables) => void;
    onSettled?: () => void;
    retry?: number | boolean;
    retryDelay?: number;
  }
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn,
    onMutate: async (variables) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey });

      // Snapshot previous value
      const previousData = queryClient.getQueryData<TData>(queryKey);

      // Optimistically update
      queryClient.setQueryData<TData>(queryKey, (old) =>
        options.optimisticUpdate(variables, old)
      );

      return { previousData };
    },
    onError: (error, variables, context) => {
      // Rollback on error
      if (context?.previousData !== undefined) {
        queryClient.setQueryData(queryKey, context.previousData);
      }
      options.onError?.(error as TError, variables);
    },
    onSuccess: (data, variables) => {
      options.onSuccess?.(data, variables);
    },
    onSettled: () => {
      // Refetch after mutation
      queryClient.invalidateQueries({ queryKey });
      options.onSettled?.();
    },
    retry: options.retry ?? false,
    retryDelay: options.retryDelay,
  });
}

/**
 * Hook for retry logic
 */
export function useRetry<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): {
  execute: () => Promise<T>;
  isRetrying: boolean;
  retryState: RetryState | null;
  reset: () => void;
} {
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryState, setRetryState] = useState<RetryState | null>(null);
  const fullConfig = { ...DEFAULT_RETRY_CONFIG, ...config };

  const execute = useCallback(async () => {
    setIsRetrying(true);
    setRetryState(null);

    try {
      return await executeWithRetry(fn, fullConfig, (state) => {
        setRetryState(state);
      });
    } finally {
      setIsRetrying(false);
    }
  }, [fn, fullConfig]);

  const reset = useCallback(() => {
    setIsRetrying(false);
    setRetryState(null);
  }, []);

  return { execute, isRetrying, retryState, reset };
}

/**
 * Hook for circuit breaker
 */
export function useCircuitBreaker(
  config: Partial<CircuitBreakerConfig> = {}
): {
  execute: <T>(fn: () => Promise<T>) => Promise<T>;
  state: CircuitBreakerState;
  canExecute: boolean;
  reset: () => void;
} {
  // Create circuit breaker instance once using lazy initialization with == null pattern
  const circuitRef = useRef<CircuitBreaker | null>(null);
  if (circuitRef.current == null) {
    circuitRef.current = new CircuitBreaker({ ...DEFAULT_CIRCUIT_BREAKER_CONFIG, ...config });
  }

  // Initialize state with default values, update in callbacks
  const [state, setState] = useState<CircuitBreakerState>({
    state: 'closed',
    failures: 0,
    successes: 0,
  });
  const [canExecuteState, setCanExecuteState] = useState(true);

  const execute = useCallback(async <T>(fn: () => Promise<T>) => {
    const circuit = circuitRef.current!;
    const result = await circuit.execute(fn);
    setState(circuit.getState());
    setCanExecuteState(circuit.canExecute());
    return result;
  }, []);

  const reset = useCallback(() => {
    const circuit = circuitRef.current!;
    circuit.reset();
    setState(circuit.getState());
    setCanExecuteState(circuit.canExecute());
  }, []);

  return {
    execute,
    state,
    canExecute: canExecuteState,
    reset,
  };
}

/**
 * Combined hook for optimistic mutations with retry and circuit breaker
 */
export function useResilientMutation<TData, TVariables>(
  queryKey: QueryKey,
  mutationFn: (variables: TVariables) => Promise<TData>,
  options: {
    optimisticUpdate: (variables: TVariables, oldData: TData | undefined) => TData;
    retry?: Partial<RetryConfig>;
    circuitBreaker?: Partial<CircuitBreakerConfig>;
    onSuccess?: (data: TData, variables: TVariables) => void;
    onError?: (error: Error, variables: TVariables) => void;
  }
) {
  const queryClient = useQueryClient();
  
  // Create circuit breaker instance once using lazy initialization with == null pattern
  const circuitRef = useRef<CircuitBreaker | null>(null);
  if (circuitRef.current == null) {
    circuitRef.current = new CircuitBreaker({ ...DEFAULT_CIRCUIT_BREAKER_CONFIG, ...options.circuitBreaker });
  }
  
  const retryConfigRef = useRef({ ...DEFAULT_RETRY_CONFIG, ...options.retry });

  // Initialize state with default values
  const [circuitState, setCircuitState] = useState<CircuitBreakerState>({
    state: 'closed',
    failures: 0,
    successes: 0,
  });
  const [canExecuteState, setCanExecuteState] = useState(true);

  const wrappedMutationFn = useCallback(
    async (variables: TVariables) => {
      const circuit = circuitRef.current!;
      return circuit.execute(() =>
        executeWithRetry(() => mutationFn(variables), retryConfigRef.current)
      );
    },
    [mutationFn]
  );

  const mutation = useMutation({
    mutationFn: wrappedMutationFn,
    onMutate: async (variables) => {
      await queryClient.cancelQueries({ queryKey });
      const previousData = queryClient.getQueryData<TData>(queryKey);
      queryClient.setQueryData<TData>(queryKey, (old) =>
        options.optimisticUpdate(variables, old)
      );
      return { previousData };
    },
    onError: (error, variables, context) => {
      if (context?.previousData !== undefined) {
        queryClient.setQueryData(queryKey, context.previousData);
      }
      const circuit = circuitRef.current!;
      setCircuitState(circuit.getState());
      setCanExecuteState(circuit.canExecute());
      options.onError?.(error as Error, variables);
    },
    onSuccess: (data, variables) => {
      const circuit = circuitRef.current!;
      setCircuitState(circuit.getState());
      setCanExecuteState(circuit.canExecute());
      options.onSuccess?.(data, variables);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey });
    },
  });

  return {
    ...mutation,
    circuitState,
    canExecute: canExecuteState,
    resetCircuit: () => {
      const circuit = circuitRef.current!;
      circuit.reset();
      setCircuitState(circuit.getState());
      setCanExecuteState(circuit.canExecute());
    },
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Create an optimistic list update helper
 */
export function createListOptimisticUpdate<T extends { id: string | number }>() {
  return {
    add: (item: T) => (oldData: T[] | undefined) => {
      return oldData ? [...oldData, item] : [item];
    },
    update: (id: T['id'], updates: Partial<T>) => (oldData: T[] | undefined) => {
      if (!oldData) return oldData;
      return oldData.map((item) =>
        item.id === id ? { ...item, ...updates } : item
      );
    },
    remove: (id: T['id']) => (oldData: T[] | undefined) => {
      if (!oldData) return oldData;
      return oldData.filter((item) => item.id !== id);
    },
    reorder: (fromIndex: number, toIndex: number) => (oldData: T[] | undefined) => {
      if (!oldData) return oldData;
      const result = [...oldData];
      const [removed] = result.splice(fromIndex, 1);
      result.splice(toIndex, 0, removed);
      return result;
    },
  };
}

/**
 * Create an optimistic object update helper
 */
export function createObjectOptimisticUpdate<T extends Record<string, unknown>>() {
  return {
    set: <K extends keyof T>(key: K, value: T[K]) => (oldData: T | undefined) => {
      if (!oldData) return oldData;
      return { ...oldData, [key]: value };
    },
    merge: (updates: Partial<T>) => (oldData: T | undefined) => {
      if (!oldData) return oldData;
      return { ...oldData, ...updates };
    },
    delete: <K extends keyof T>(key: K) => (oldData: T | undefined) => {
      if (!oldData) return oldData;
      const { [key]: _unused, ...rest } = oldData;
      void _unused; // Suppress unused variable warning
      return rest as T;
    },
  };
}