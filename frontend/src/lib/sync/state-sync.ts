/**
 * State Synchronization
 *
 * Provides state synchronization patterns:
 * - Debounced sync with batching
 * - Conflict resolution
 * - Version tracking
 * - Offline support with queue
 * - React Query integration
 *
 * @module lib/sync/state-sync
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useMutation, useQueryClient, UseMutationOptions } from '@tanstack/react-query';
import type {
  SyncState,
  SyncOperation,
  SyncConfig,
  ConflictInfo,
  ConflictResolver,
} from './types';
import { EventBus, getEventBus } from './event-bus';

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_SYNC_CONFIG: Required<SyncConfig> = {
  debounceDelay: 300,
  maxBatchSize: 10,
  maxRetries: 3,
  retryDelay: 1000,
  optimisticUpdates: true,
  conflictResolution: 'client-wins',
};

// ============================================================================
// State Sync Manager
// ============================================================================

export class StateSyncManager<T = unknown> {
  private config: Required<SyncConfig>;
  private state: SyncState<T>;
  private operations: Map<string, SyncOperation<T>> = new Map();
  private pendingBatch: SyncOperation<T>[] = [];
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private eventBus: EventBus;
  private syncFn: (operations: SyncOperation<T>[]) => Promise<T>;
  private conflictResolver?: ConflictResolver<T>;
  private listeners = new Set<(state: SyncState<T>) => void>();

  constructor(
    initialData: T,
    syncFn: (operations: SyncOperation<T>[]) => Promise<T>,
    config?: SyncConfig,
    conflictResolver?: ConflictResolver<T>
  ) {
    this.config = { ...DEFAULT_SYNC_CONFIG, ...config };
    this.syncFn = syncFn;
    this.conflictResolver = conflictResolver;
    this.eventBus = getEventBus();

    this.state = {
      data: initialData,
      version: 0,
      lastSyncedAt: new Date(),
      isDirty: false,
      syncStatus: 'idle',
    };
  }

  // ============================================================================
  // State Access
  // ============================================================================

  /**
   * Get current state
   */
  getState(): SyncState<T> {
    return { ...this.state };
  }

  /**
   * Get current data
   */
  getData(): T {
    return this.state.data;
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener: (state: SyncState<T>) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  // ============================================================================
  // Operations
  // ============================================================================

  /**
   * Create operation
   */
  create(path: string, data: T): string {
    return this.addOperation('create', path, data);
  }

  /**
   * Update operation
   */
  update(path: string, data: T): string {
    return this.addOperation('update', path, data);
  }

  /**
   * Patch operation (partial update)
   */
  patch(path: string, data: Partial<T>): string {
    return this.addOperation('patch', path, data as T);
  }

  /**
   * Delete operation
   */
  delete(path: string): string {
    return this.addOperation('delete', path);
  }

  /**
   * Add operation to queue
   */
  private addOperation(
    type: SyncOperation['type'],
    path: string,
    data?: T
  ): string {
    const id = `${type}-${path}-${Date.now()}`;

    const operation: SyncOperation<T> = {
      id,
      type,
      path,
      data,
      previousData: this.state.data,
      timestamp: Date.now(),
      status: 'pending',
      retryCount: 0,
    };

    this.operations.set(id, operation);
    this.pendingBatch.push(operation);

    // Apply optimistic update
    if (this.config.optimisticUpdates && data !== undefined) {
      this.applyOptimisticUpdate(operation);
    }

    // Schedule sync
    this.scheduleSync();

    this.eventBus.emit('sync:operation-added', { operationId: id, type, path });

    return id;
  }

  /**
   * Apply optimistic update to state
   */
  private applyOptimisticUpdate(operation: SyncOperation<T>): void {
    if (operation.data === undefined) return;

    // Simple merge for now - can be extended for deep paths
    this.updateState({
      data: operation.data,
      isDirty: true,
    });
  }

  /**
   * Rollback optimistic update
   */
  private rollbackOperation(operation: SyncOperation<T>): void {
    if (operation.previousData !== undefined) {
      this.updateState({
        data: operation.previousData,
      });
    }
    operation.status = 'rolled_back';
    this.operations.set(operation.id, operation);

    this.eventBus.emit('sync:operation-rolled-back', { operationId: operation.id });
  }

  // ============================================================================
  // Synchronization
  // ============================================================================

  /**
   * Schedule sync with debounce
   */
  private scheduleSync(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.sync();
    }, this.config.debounceDelay);
  }

  /**
   * Force immediate sync
   */
  async forceSync(): Promise<void> {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    await this.sync();
  }

  /**
   * Perform sync
   */
  private async sync(): Promise<void> {
    if (this.pendingBatch.length === 0) return;

    // Take batch
    const batch = this.pendingBatch.splice(0, this.config.maxBatchSize);

    // Mark as syncing
    batch.forEach((op) => {
      op.status = 'syncing';
      this.operations.set(op.id, op);
    });

    this.updateState({ syncStatus: 'syncing' });
    this.eventBus.emit('sync:started', { operationIds: batch.map((op) => op.id) });

    try {
      const result = await this.syncFn(batch);

      // Mark as synced
      batch.forEach((op) => {
        op.status = 'synced';
        this.operations.set(op.id, op);
      });

      this.updateState({
        data: result,
        version: this.state.version + 1,
        lastSyncedAt: new Date(),
        isDirty: this.pendingBatch.length > 0,
        syncStatus: 'idle',
        error: undefined,
      });

      this.eventBus.emit('sync:completed', {
        operationIds: batch.map((op) => op.id),
        version: this.state.version,
      });

      // Continue with remaining operations
      if (this.pendingBatch.length > 0) {
        this.scheduleSync();
      }
    } catch (error) {
      await this.handleSyncError(batch, error as Error);
    }
  }

  /**
   * Handle sync error
   */
  private async handleSyncError(
    batch: SyncOperation<T>[],
    error: Error
  ): Promise<void> {
    // Check for conflict
    if (this.isConflictError(error)) {
      await this.handleConflict(batch, error);
      return;
    }

    // Retry logic
    const retriableOps = batch.filter((op) => op.retryCount < this.config.maxRetries);
    const failedOps = batch.filter((op) => op.retryCount >= this.config.maxRetries);

    // Mark failed operations
    failedOps.forEach((op) => {
      op.status = 'failed';
      op.error = error;
      this.operations.set(op.id, op);

      // Rollback if optimistic
      if (this.config.optimisticUpdates) {
        this.rollbackOperation(op);
      }
    });

    // Retry retriable operations
    if (retriableOps.length > 0) {
      retriableOps.forEach((op) => {
        op.retryCount++;
        op.status = 'pending';
        this.operations.set(op.id, op);
        this.pendingBatch.push(op);
      });

      // Schedule retry with delay
      setTimeout(() => {
        this.sync();
      }, this.config.retryDelay * Math.pow(2, retriableOps[0].retryCount - 1));
    }

    this.updateState({
      syncStatus: 'error',
      error,
    });

    this.eventBus.emit('sync:error', {
      operationIds: batch.map((op) => op.id),
      error: error.message,
    });
  }

  /**
   * Check if error is a conflict
   */
  private isConflictError(error: Error): boolean {
    const errorWithStatus = error as Error & { status?: number };
    return error.message.includes('conflict') ||
      error.message.includes('version mismatch') ||
      errorWithStatus.status === 409;
  }

  /**
   * Handle conflict
   */
  private async handleConflict(
    batch: SyncOperation<T>[],
    error: Error
  ): Promise<void> {
    const errorWithData = error as Error & { serverData?: T };
    const serverData = errorWithData.serverData;

    if (!serverData) {
      // Can't resolve without server data
      batch.forEach((op) => {
        op.status = 'failed';
        op.error = error;
        this.operations.set(op.id, op);
      });
      return;
    }

    let resolvedData: T;

    switch (this.config.conflictResolution) {
      case 'client-wins':
        resolvedData = this.state.data;
        break;
      case 'server-wins':
        resolvedData = serverData;
        break;
      case 'merge':
        resolvedData = this.mergeData(this.state.data, serverData);
        break;
      case 'manual':
        if (this.conflictResolver) {
          const conflict: ConflictInfo<T> = {
            operationId: batch[0].id,
            clientData: this.state.data,
            serverData,
            timestamp: Date.now(),
          };
          resolvedData = await this.conflictResolver(conflict);
        } else {
          resolvedData = serverData;
        }
        break;
      default:
        resolvedData = serverData;
    }

    this.updateState({
      data: resolvedData,
      version: this.state.version + 1,
      lastSyncedAt: new Date(),
      isDirty: false,
      syncStatus: 'idle',
    });

    this.eventBus.emit('sync:conflict-resolved', {
      operationIds: batch.map((op) => op.id),
      resolution: this.config.conflictResolution,
    });
  }

  /**
   * Simple merge strategy
   */
  private mergeData(clientData: T, serverData: T): T {
    if (typeof clientData === 'object' && typeof serverData === 'object') {
      return { ...serverData, ...clientData } as T;
    }
    return clientData;
  }

  // ============================================================================
  // State Management
  // ============================================================================

  /**
   * Update state and notify listeners
   */
  private updateState(partial: Partial<SyncState<T>>): void {
    this.state = { ...this.state, ...partial };
    this.listeners.forEach((listener) => listener(this.state));
  }

  /**
   * Get operation by ID
   */
  getOperation(id: string): SyncOperation<T> | undefined {
    return this.operations.get(id);
  }

  /**
   * Get all pending operations
   */
  getPendingOperations(): SyncOperation<T>[] {
    return Array.from(this.operations.values()).filter(
      (op) => op.status === 'pending' || op.status === 'syncing'
    );
  }

  /**
   * Clear completed operations
   */
  clearCompletedOperations(): void {
    this.operations.forEach((op, id) => {
      if (op.status === 'synced' || op.status === 'rolled_back') {
        this.operations.delete(id);
      }
    });
  }

  /**
   * Reset state
   */
  reset(data: T): void {
    this.operations.clear();
    this.pendingBatch = [];
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }

    this.updateState({
      data,
      version: 0,
      lastSyncedAt: new Date(),
      isDirty: false,
      syncStatus: 'idle',
      error: undefined,
    });
  }
}

// ============================================================================
// React Hooks
// ============================================================================

/**
 * Hook for synchronized state
 */
export function useSyncedState<T>(
  initialData: T,
  syncFn: (operations: SyncOperation<T>[]) => Promise<T>,
  config?: SyncConfig
): {
  data: T;
  state: SyncState<T>;
  update: (data: T) => string;
  patch: (data: Partial<T>) => string;
  forceSync: () => Promise<void>;
  reset: (data: T) => void;
  pendingOperations: SyncOperation<T>[];
} {
  const managerRef = useRef<StateSyncManager<T> | null>(null);
  const [state, setState] = useState<SyncState<T>>(() => ({
    data: initialData,
    version: 0,
    lastSyncedAt: new Date(),
    isDirty: false,
    syncStatus: 'idle',
  }));

  // Initialize manager
  useEffect(() => {
    managerRef.current = new StateSyncManager(initialData, syncFn, config);

    const unsubscribe = managerRef.current.subscribe(setState);

    return () => {
      unsubscribe();
    };
  }, [initialData, syncFn, config]);

  const update = useCallback((data: T) => {
    return managerRef.current?.update('/', data) || '';
  }, []);

  const patch = useCallback((data: Partial<T>) => {
    return managerRef.current?.patch('/', data) || '';
  }, []);

  const forceSync = useCallback(async () => {
    await managerRef.current?.forceSync();
  }, []);

  const reset = useCallback((data: T) => {
    managerRef.current?.reset(data);
  }, []);

  // Track pending operations in state to avoid accessing ref during render
  const [pendingOperations, setPendingOperations] = useState<SyncOperation<T>[]>([]);

  // Update pending operations when state changes - schedule for next tick to avoid setState in effect
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      const ops = managerRef.current?.getPendingOperations() || [];
      setPendingOperations(ops);
    }, 0);
    return () => clearTimeout(timeoutId);
  }, [state]);

  return {
    data: state.data,
    state,
    update,
    patch,
    forceSync,
    reset,
    pendingOperations,
  };
}

/**
 * Hook for optimistic mutations with React Query
 */
export function useOptimisticMutation<TData, TVariables, TContext = unknown>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options: UseMutationOptions<TData, Error, TVariables, TContext> & {
    queryKey: unknown[];
    optimisticUpdate?: (variables: TVariables, oldData: TData | undefined) => TData;
  }
) {
  const queryClient = useQueryClient();
  const { queryKey, optimisticUpdate, ...mutationOptions } = options;

  return useMutation({
    mutationFn,
    onMutate: async (variables) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey });

      // Snapshot previous value
      const previousData = queryClient.getQueryData<TData>(queryKey);

      // Optimistically update
      if (optimisticUpdate && previousData !== undefined) {
        queryClient.setQueryData<TData>(queryKey, (old) =>
          optimisticUpdate(variables, old)
        );
      }

      // Return context with snapshot
      return { previousData } as TContext;
    },
    onError: (error, variables, context) => {
      // Rollback on error
      const ctx = context as { previousData?: TData } | undefined;
      if (ctx?.previousData !== undefined) {
        queryClient.setQueryData(queryKey, ctx.previousData);
      }
      // @ts-ignore - Argument count mismatch with current React Query types
      mutationOptions.onError?.(error, variables, context);
    },
    onSettled: (data, error, variables, context) => {
      // Refetch after mutation
      queryClient.invalidateQueries({ queryKey });
      // @ts-ignore - Argument count mismatch with current React Query types
      mutationOptions.onSettled?.(data, error, variables, context);
    },
    ...mutationOptions,
  });
}

/**
 * Hook for debounced sync
 */
export function useDebouncedSync<T>(
  value: T,
  syncFn: (value: T) => Promise<void>,
  delay = 300
): {
  isSyncing: boolean;
  lastSyncedValue: T | null;
  forceSync: () => Promise<void>;
} {
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncedValue, setLastSyncedValue] = useState<T | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestValueRef = useRef(value);

  latestValueRef.current = value;

  const sync = useCallback(async () => {
    setIsSyncing(true);
    try {
      await syncFn(latestValueRef.current);
      setLastSyncedValue(latestValueRef.current);
    } finally {
      setIsSyncing(false);
    }
  }, [syncFn]);

  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(sync, delay);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [value, delay, sync]);

  const forceSync = useCallback(async () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    await sync();
  }, [sync]);

  return { isSyncing, lastSyncedValue, forceSync };
}

/**
 * Hook for offline-first sync
 */
export function useOfflineSync<T>(
  key: string,
  fetchFn: () => Promise<T>,
  syncFn: (data: T) => Promise<T>
): {
  data: T | null;
  isOnline: boolean;
  isSyncing: boolean;
  pendingChanges: number;
  update: (data: T) => void;
  sync: () => Promise<void>;
} {
  const [data, setData] = useState<T | null>(null);
  const [isOnline, setIsOnline] = useState(
    typeof navigator !== 'undefined' ? navigator.onLine : true
  );
  const [isSyncing, setIsSyncing] = useState(false);
  const [pendingChanges, setPendingChanges] = useState(0);
  const pendingRef = useRef<T[]>([]);

  // Monitor online status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Load from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(`offline-sync:${key}`);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setData(parsed.data);
        pendingRef.current = parsed.pending || [];
        setPendingChanges(pendingRef.current.length);
      } catch {
        // Ignore parse errors
      }
    }

    // Fetch fresh data if online
    if (isOnline) {
      fetchFn().then(setData).catch(console.error);
    }
  }, [key, isOnline, fetchFn]);

  // Save to localStorage on change
  useEffect(() => {
    if (data !== null) {
      localStorage.setItem(
        `offline-sync:${key}`,
        JSON.stringify({ data, pending: pendingRef.current })
      );
    }
  }, [key, data, pendingChanges]);

  const sync = useCallback(async () => {
    if (!isOnline || pendingRef.current.length === 0) return;

    setIsSyncing(true);
    try {
      // Sync all pending changes
      while (pendingRef.current.length > 0) {
        const pending = pendingRef.current[0];
        const result = await syncFn(pending);
        pendingRef.current.shift();
        setPendingChanges(pendingRef.current.length);
        setData(result);
      }
    } finally {
      setIsSyncing(false);
    }
  }, [isOnline, syncFn]);

  // Sync when coming online - schedule for next tick to avoid setState in effect
  useEffect(() => {
    if (isOnline && pendingRef.current.length > 0) {
      const timeoutId = setTimeout(() => {
        sync();
      }, 0);
      return () => clearTimeout(timeoutId);
    }
  }, [isOnline, sync]);

  const update = useCallback((newData: T) => {
    setData(newData);

    if (isOnline) {
      // Sync immediately
      syncFn(newData).catch(() => {
        // Queue for later if sync fails
        pendingRef.current.push(newData);
        setPendingChanges(pendingRef.current.length);
      });
    } else {
      // Queue for later
      pendingRef.current.push(newData);
      setPendingChanges(pendingRef.current.length);
    }
  }, [isOnline, syncFn]);



  return {
    data,
    isOnline,
    isSyncing,
    pendingChanges,
    update,
    sync,
  };
}
