/**
 * State Synchronization Utilities for Project Chimera Frontend
 * Provides optimistic updates, state recovery, and event-driven patterns
 *
 * @module lib/state-sync
 */

import { useState, useCallback, useRef, useEffect } from "react";

// ============================================================================
// Types
// ============================================================================

export interface OptimisticState<T> {
    /** Current optimistic value */
    value: T;
    /** Whether an update is pending */
    isPending: boolean;
    /** Error if update failed */
    error: Error | null;
    /** Last confirmed server value */
    confirmedValue: T;
    /** Whether rollback occurred */
    rolledBack: boolean;
}

export interface OptimisticUpdateOptions<T> {
    /** Delay before confirming (for debouncing rapid updates) */
    confirmDelay?: number;
    /** Callback when update succeeds */
    onSuccess?: (value: T) => void;
    /** Callback when update fails and rolls back */
    onRollback?: (error: Error, previousValue: T) => void;
    /** Callback when update is confirmed */
    onConfirm?: (value: T) => void;
}

export interface StateRecoveryOptions {
    /** Storage key for persisting state */
    storageKey: string;
    /** Use sessionStorage instead of localStorage */
    sessionOnly?: boolean;
    /** Maximum age of stored state in ms */
    maxAge?: number;
}

export interface EventSourceOptions {
    /** Event types to listen for */
    eventTypes: string[];
    /** Callback for each event type */
    handlers: Record<string, (data: unknown) => void>;
    /** Reconnection settings */
    reconnect?: boolean;
    reconnectInterval?: number;
}

// ============================================================================
// Optimistic Updates Hook
// ============================================================================

/**
 * Hook for managing optimistic updates with automatic rollback on failure
 *
 * @example
 * ```tsx
 * const [state, updateOptimistically] = useOptimisticUpdate(
 *   initialValue,
 *   async (newValue) => await api.save(newValue)
 * );
 *
 * // Update optimistically - UI updates immediately
 * updateOptimistically(newValue);
 * ```
 */
export function useOptimisticUpdate<T>(
    initialValue: T,
    updateFn: (value: T) => Promise<T>,
    options: OptimisticUpdateOptions<T> = {}
): [OptimisticState<T>, (value: T) => void, () => void] {
    const { confirmDelay = 0, onSuccess, onRollback, onConfirm } = options;

    const [state, setState] = useState<OptimisticState<T>>({
        value: initialValue,
        isPending: false,
        error: null,
        confirmedValue: initialValue,
        rolledBack: false,
    });

    const pendingUpdateRef = useRef<T | null>(null);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const update = useCallback(
        (newValue: T) => {
            // Clear any pending confirmation
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }

            // Store pending value
            pendingUpdateRef.current = newValue;

            // Optimistically update UI immediately
            setState((prev) => ({
                ...prev,
                value: newValue,
                isPending: true,
                error: null,
                rolledBack: false,
            }));

            // Schedule the actual update
            const performUpdate = async () => {
                const valueToUpdate = pendingUpdateRef.current!;
                const previousValue = state.confirmedValue;

                try {
                    const result = await updateFn(valueToUpdate);

                    setState((prev) => ({
                        ...prev,
                        value: result,
                        isPending: false,
                        confirmedValue: result,
                        error: null,
                    }));

                    onSuccess?.(result);
                    onConfirm?.(result);
                } catch (error) {
                    // Rollback to previous confirmed value
                    const err = error instanceof Error ? error : new Error(String(error));

                    setState((prev) => ({
                        ...prev,
                        value: previousValue,
                        isPending: false,
                        error: err,
                        rolledBack: true,
                    }));

                    onRollback?.(err, previousValue);
                }
            };

            if (confirmDelay > 0) {
                timeoutRef.current = setTimeout(performUpdate, confirmDelay);
            } else {
                performUpdate();
            }
        },
        [confirmDelay, onSuccess, onRollback, onConfirm, state.confirmedValue, updateFn]
    );

    const cancel = useCallback(() => {
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }
        pendingUpdateRef.current = null;

        setState((prev) => ({
            ...prev,
            value: prev.confirmedValue,
            isPending: false,
            rolledBack: true,
        }));
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
        };
    }, []);

    return [state, update, cancel];
}

// ============================================================================
// State Recovery Hook
// ============================================================================

/**
 * Hook for state recovery on reconnection
 * Persists state to storage and recovers on page reload or reconnection
 */
export function useStateRecovery<T>(
    key: string,
    initialValue: T,
    options: Omit<StateRecoveryOptions, "storageKey"> = {}
): [T, (value: T) => void, () => void] {
    const { sessionOnly = false, maxAge = 30 * 60 * 1000 } = options;

    const getStorage = useCallback(() => {
        if (typeof window === "undefined") return null;
        return sessionOnly ? window.sessionStorage : window.localStorage;
    }, [sessionOnly]);

    const getStoredValue = useCallback((): T | null => {
        const storage = getStorage();
        if (!storage) return null;

        try {
            const item = storage.getItem(`chimera:state:${key}`);
            if (!item) return null;

            const { value, timestamp } = JSON.parse(item) as {
                value: T;
                timestamp: number;
            };

            // Check if stored value is still valid
            if (Date.now() - timestamp > maxAge) {
                storage.removeItem(`chimera:state:${key}`);
                return null;
            }

            return value;
        } catch {
            return null;
        }
    }, [getStorage, key, maxAge]);

    const [state, setState] = useState<T>(() => {
        const stored = getStoredValue();
        return stored !== null ? stored : initialValue;
    });

    const setValue = useCallback(
        (value: T) => {
            setState(value);

            const storage = getStorage();
            if (storage) {
                try {
                    storage.setItem(
                        `chimera:state:${key}`,
                        JSON.stringify({
                            value,
                            timestamp: Date.now(),
                        })
                    );
                } catch {
                    // Storage full or unavailable
                }
            }
        },
        [getStorage, key]
    );

    const clearRecovery = useCallback(() => {
        const storage = getStorage();
        if (storage) {
            storage.removeItem(`chimera:state:${key}`);
        }
    }, [getStorage, key]);

    return [state, setValue, clearRecovery];
}

// ============================================================================
// Event Bus for Cross-Component State Sync
// ============================================================================

type EventCallback<T = unknown> = (data: T) => void;
type Unsubscribe = () => void;

class EventBus {
    private listeners: Map<string, Set<EventCallback>> = new Map();

    /**
     * Subscribe to an event
     */
    on<T>(event: string, callback: EventCallback<T>): Unsubscribe {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(callback as EventCallback);

        return () => {
            this.listeners.get(event)?.delete(callback as EventCallback);
        };
    }

    /**
     * Emit an event
     */
    emit<T>(event: string, data: T): void {
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            callbacks.forEach((callback) => callback(data));
        }
    }

    /**
     * Subscribe to an event once
     */
    once<T>(event: string, callback: EventCallback<T>): Unsubscribe {
        const unsubscribe = this.on<T>(event, (data) => {
            unsubscribe();
            callback(data);
        });
        return unsubscribe;
    }

    /**
     * Clear all listeners for an event
     */
    off(event: string): void {
        this.listeners.delete(event);
    }

    /**
     * Clear all listeners
     */
    clear(): void {
        this.listeners.clear();
    }
}

// Singleton event bus instance
export const eventBus = new EventBus();

// ============================================================================
// Event-Driven State Update Hook
// ============================================================================

/**
 * Hook that subscribes to events and updates state accordingly
 */
export function useEventDrivenState<T>(
    eventName: string,
    initialValue: T,
    reducer?: (current: T, eventData: unknown) => T
): T {
    const [state, setState] = useState<T>(initialValue);

    useEffect(() => {
        const unsubscribe = eventBus.on(eventName, (data) => {
            if (reducer) {
                setState((current) => reducer(current, data));
            } else {
                setState(data as T);
            }
        });

        return unsubscribe;
    }, [eventName, reducer]);

    return state;
}

// ============================================================================
// State Sync Events
// ============================================================================

/** Pre-defined event types for state synchronization */
export const STATE_SYNC_EVENTS = {
    /** Provider status changed */
    PROVIDER_STATUS_CHANGED: "state:provider:status",
    /** Session updated */
    SESSION_UPDATED: "state:session:updated",
    /** Model changed */
    MODEL_CHANGED: "state:model:changed",
    /** Cache invalidated */
    CACHE_INVALIDATED: "state:cache:invalidated",
    /** Connection status changed */
    CONNECTION_STATUS_CHANGED: "state:connection:status",
    /** Error occurred */
    ERROR_OCCURRED: "state:error",
    /** Generation completed */
    GENERATION_COMPLETED: "state:generation:complete",
} as const;

// ============================================================================
// Sync State Manager
// ============================================================================

/**
 * Central state manager for cross-component synchronization
 */
class StateSyncManager {
    private state: Map<string, unknown> = new Map();
    private subscribers: Map<string, Set<(value: unknown) => void>> = new Map();

    /**
     * Get current value of a state key
     */
    get<T>(key: string): T | undefined {
        return this.state.get(key) as T | undefined;
    }

    /**
     * Set value and notify subscribers
     */
    set<T>(key: string, value: T): void {
        const previous = this.state.get(key);
        this.state.set(key, value);

        // Notify subscribers
        const subs = this.subscribers.get(key);
        if (subs) {
            subs.forEach((callback) => callback(value));
        }

        // Emit global event
        eventBus.emit(`state:${key}`, { key, value, previous });
    }

    /**
     * Subscribe to state changes for a specific key
     */
    subscribe<T>(key: string, callback: (value: T) => void): Unsubscribe {
        if (!this.subscribers.has(key)) {
            this.subscribers.set(key, new Set());
        }
        this.subscribers.get(key)!.add(callback as (value: unknown) => void);

        return () => {
            this.subscribers.get(key)?.delete(callback as (value: unknown) => void);
        };
    }

    /**
     * Reset all state
     */
    reset(): void {
        this.state.clear();
    }
}

export const stateSyncManager = new StateSyncManager();

// ============================================================================
// React Hook for State Sync Manager
// ============================================================================

/**
 * Hook that syncs component state with the global state manager
 */
export function useSyncedState<T>(
    key: string,
    initialValue: T
): [T, (value: T) => void] {
    const [localState, setLocalState] = useState<T>(
        () => stateSyncManager.get<T>(key) ?? initialValue
    );

    useEffect(() => {
        const unsubscribe = stateSyncManager.subscribe<T>(key, setLocalState);
        return unsubscribe;
    }, [key]);

    const setValue = useCallback(
        (value: T) => {
            stateSyncManager.set(key, value);
        },
        [key]
    );

    return [localState, setValue];
}
