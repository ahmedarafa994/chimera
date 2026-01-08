/**
 * Event Bus
 * 
 * Provides a pub/sub event system for cross-component communication:
 * - Type-safe event handling
 * - Event history and replay
 * - Channel-based subscriptions
 * - React hook integration
 * 
 * @module lib/sync/event-bus
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  EventBusConfig,
  BusEvent,
  EventHandler,
  EventFilter,
  Subscription,
  Channel,
  ChannelConfig,
} from './types';

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: Required<EventBusConfig> = {
  maxListeners: 100,
  debug: false,
  historySize: 50,
};

// ============================================================================
// Event Bus Class
// ============================================================================

export class EventBus {
  private config: Required<EventBusConfig>;
  private handlers = new Map<string, Set<EventHandler>>();
  private history: BusEvent[] = [];
  private channels = new Map<string, ChannelImpl>();

  constructor(config: EventBusConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ============================================================================
  // Publishing
  // ============================================================================

  /**
   * Emit an event
   */
  emit<T = unknown>(type: string, payload: T, source?: string): void {
    const event: BusEvent<T> = {
      type,
      payload,
      timestamp: Date.now(),
      source,
    };

    this.log('Emit:', type, payload);

    // Add to history
    this.addToHistory(event);

    // Notify handlers
    this.notifyHandlers(type, event);

    // Notify wildcard handlers
    this.notifyHandlers('*', event);
  }

  /**
   * Emit multiple events
   */
  emitBatch<T = unknown>(events: Array<{ type: string; payload: T; source?: string }>): void {
    events.forEach(({ type, payload, source }) => {
      this.emit(type, payload, source);
    });
  }

  // ============================================================================
  // Subscribing
  // ============================================================================

  /**
   * Subscribe to events of a specific type
   */
  on<T = unknown>(type: string, handler: EventHandler<T>): Subscription {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }

    const handlers = this.handlers.get(type)!;

    if (handlers.size >= this.config.maxListeners) {
      console.warn(`EventBus: Max listeners (${this.config.maxListeners}) reached for "${type}"`);
    }

    handlers.add(handler as EventHandler);
    this.log('Subscribe:', type);

    return {
      unsubscribe: () => {
        handlers.delete(handler as EventHandler);
        this.log('Unsubscribe:', type);
      },
    };
  }

  /**
   * Subscribe to all events
   */
  onAny<T = unknown>(handler: EventHandler<T>): Subscription {
    return this.on('*', handler);
  }

  /**
   * Subscribe once (auto-unsubscribe after first event)
   */
  once<T = unknown>(type: string, handler: EventHandler<T>): Subscription {
    const subscription = this.on<T>(type, (event) => {
      subscription.unsubscribe();
      handler(event);
    });
    return subscription;
  }

  /**
   * Subscribe with filter
   */
  onFiltered<T = unknown>(
    type: string,
    filter: EventFilter<T>,
    handler: EventHandler<T>
  ): Subscription {
    return this.on<T>(type, (event) => {
      if (filter(event)) {
        handler(event);
      }
    });
  }

  /**
   * Wait for an event (Promise-based)
   */
  waitFor<T = unknown>(type: string, timeout?: number): Promise<BusEvent<T>> {
    return new Promise((resolve, reject) => {
      let timeoutId: ReturnType<typeof setTimeout> | undefined;

      const subscription = this.once<T>(type, (event) => {
        if (timeoutId) clearTimeout(timeoutId);
        resolve(event);
      });

      if (timeout) {
        timeoutId = setTimeout(() => {
          subscription.unsubscribe();
          reject(new Error(`Timeout waiting for event "${type}"`));
        }, timeout);
      }
    });
  }

  // ============================================================================
  // Unsubscribing
  // ============================================================================

  /**
   * Remove all handlers for a type
   */
  off(type: string): void {
    this.handlers.delete(type);
    this.log('Off:', type);
  }

  /**
   * Remove all handlers
   */
  offAll(): void {
    this.handlers.clear();
    this.log('Off all');
  }

  // ============================================================================
  // History
  // ============================================================================

  /**
   * Get event history
   */
  getHistory(type?: string): BusEvent[] {
    if (type) {
      return this.history.filter((event) => event.type === type);
    }
    return [...this.history];
  }

  /**
   * Get last event of a type
   */
  getLastEvent<T = unknown>(type: string): BusEvent<T> | undefined {
    for (let i = this.history.length - 1; i >= 0; i--) {
      if (this.history[i].type === type) {
        return this.history[i] as BusEvent<T>;
      }
    }
    return undefined;
  }

  /**
   * Clear history
   */
  clearHistory(): void {
    this.history = [];
  }

  /**
   * Replay history to a handler
   */
  replay<T = unknown>(type: string, handler: EventHandler<T>): void {
    const events = this.getHistory(type);
    events.forEach((event) => handler(event as BusEvent<T>));
  }

  // ============================================================================
  // Channels
  // ============================================================================

  /**
   * Get or create a channel
   */
  channel<T = unknown>(name: string, config?: ChannelConfig): Channel<T> {
    if (!this.channels.has(name)) {
      this.channels.set(name, new ChannelImpl<T>(this, name, config));
    }
    return this.channels.get(name) as Channel<T>;
  }

  /**
   * Delete a channel
   */
  deleteChannel(name: string): void {
    this.channels.delete(name);
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  /**
   * Get listener count for a type
   */
  listenerCount(type: string): number {
    return this.handlers.get(type)?.size || 0;
  }

  /**
   * Get all event types with listeners
   */
  eventTypes(): string[] {
    return Array.from(this.handlers.keys());
  }

  private notifyHandlers<T>(type: string, event: BusEvent<T>): void {
    const handlers = this.handlers.get(type);
    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(event);
        } catch (error) {
          console.error(`EventBus: Handler error for "${type}":`, error);
        }
      });
    }
  }

  private addToHistory(event: BusEvent): void {
    this.history.push(event);
    if (this.history.length > this.config.historySize) {
      this.history.shift();
    }
  }

  private log(...args: unknown[]): void {
    if (this.config.debug) {
      console.log('[EventBus]', ...args);
    }
  }
}

// ============================================================================
// Channel Implementation
// ============================================================================

class ChannelImpl<T = unknown> implements Channel<T> {
  private bus: EventBus;
  private config: Required<ChannelConfig>;
  private lastMessage?: BusEvent<T>;
  private messageHistory: BusEvent<T>[] = [];

  constructor(
    bus: EventBus,
    public readonly name: string,
    config?: ChannelConfig
  ) {
    this.bus = bus;
    this.config = {
      persistLast: true,
      historySize: 10,
      replayOnSubscribe: false,
      ...config,
    };
  }

  subscribe(handler: EventHandler<T>): Subscription {
    // Replay history if configured
    if (this.config.replayOnSubscribe) {
      this.messageHistory.forEach((event) => handler(event));
    }

    return this.bus.on<T>(`channel:${this.name}`, (event) => {
      if (this.config.persistLast) {
        this.lastMessage = event;
      }

      this.messageHistory.push(event);
      if (this.messageHistory.length > this.config.historySize) {
        this.messageHistory.shift();
      }

      handler(event);
    });
  }

  publish(payload: T): void {
    this.bus.emit(`channel:${this.name}`, payload);
  }

  getLastMessage(): BusEvent<T> | undefined {
    return this.lastMessage;
  }

  getHistory(): BusEvent<T>[] {
    return [...this.messageHistory];
  }
}

// ============================================================================
// React Hooks
// ============================================================================

/**
 * Hook to subscribe to events
 */
export function useEventBus<T = unknown>(
  bus: EventBus,
  type: string,
  handler?: EventHandler<T>
): {
  emit: (payload: T) => void;
  lastEvent: BusEvent<T> | undefined;
} {
  const [lastEvent, setLastEvent] = useState<BusEvent<T> | undefined>(
    () => bus.getLastEvent<T>(type)
  );

  useEffect(() => {
    const subscription = bus.on<T>(type, (event) => {
      setLastEvent(event);
      handler?.(event);
    });

    return () => subscription.unsubscribe();
  }, [bus, type, handler]);

  const emit = useCallback(
    (payload: T) => {
      bus.emit(type, payload);
    },
    [bus, type]
  );

  return { emit, lastEvent };
}

/**
 * Hook to subscribe to a channel
 */
export function useChannel<T = unknown>(
  bus: EventBus,
  channelName: string,
  config?: ChannelConfig
): {
  publish: (payload: T) => void;
  lastMessage: BusEvent<T> | undefined;
  messages: BusEvent<T>[];
} {
  const channelRef = useRef<Channel<T> | null>(null);
  const [lastMessage, setLastMessage] = useState<BusEvent<T> | undefined>(undefined);
  const [messages, setMessages] = useState<BusEvent<T>[]>([]);

  // Initialize channel in effect to avoid accessing ref during render
  useEffect(() => {
    channelRef.current = bus.channel<T>(channelName, config);
    
    // Set initial values
    setLastMessage(channelRef.current.getLastMessage());
    setMessages(channelRef.current.getHistory());

    const subscription = channelRef.current.subscribe((event) => {
      setLastMessage(event);
      setMessages(channelRef.current?.getHistory() || []);
    });

    return () => subscription.unsubscribe();
  }, [bus, channelName, config]);

  const publish = useCallback(
    (payload: T) => {
      channelRef.current?.publish(payload);
    },
    []
  );

  return { publish, lastMessage, messages };
}

/**
 * Hook for event-driven state
 */
export function useEventState<T>(
  bus: EventBus,
  type: string,
  initialState: T
): [T, (value: T) => void] {
  const [state, setState] = useState<T>(initialState);

  useEffect(() => {
    const subscription = bus.on<T>(type, (event) => {
      setState(event.payload);
    });

    return () => subscription.unsubscribe();
  }, [bus, type]);

  const setStateAndEmit = useCallback(
    (value: T) => {
      setState(value);
      bus.emit(type, value);
    },
    [bus, type]
  );

  return [state, setStateAndEmit];
}

/**
 * Hook to wait for an event
 */
export function useWaitForEvent<T = unknown>(
  bus: EventBus,
  type: string
): {
  wait: (timeout?: number) => Promise<BusEvent<T>>;
  isWaiting: boolean;
} {
  const [isWaiting, setIsWaiting] = useState(false);

  const wait = useCallback(
    async (timeout?: number) => {
      setIsWaiting(true);
      try {
        return await bus.waitFor<T>(type, timeout);
      } finally {
        setIsWaiting(false);
      }
    },
    [bus, type]
  );

  return { wait, isWaiting };
}

// ============================================================================
// Singleton Instance
// ============================================================================

let defaultBus: EventBus | null = null;

/**
 * Get or create the default event bus
 */
export function getEventBus(config?: EventBusConfig): EventBus {
  if (!defaultBus) {
    defaultBus = new EventBus(config);
  }
  return defaultBus;
}

/**
 * Reset the default event bus
 */
export function resetEventBus(): void {
  defaultBus?.offAll();
  defaultBus = null;
}

// ============================================================================
// Typed Event Helpers
// ============================================================================

/**
 * Create a typed event emitter
 */
export function createTypedEmitter<TEvents extends Record<string, unknown>>(
  bus: EventBus
) {
  return {
    emit<K extends keyof TEvents>(type: K, payload: TEvents[K]): void {
      bus.emit(type as string, payload);
    },
    on<K extends keyof TEvents>(
      type: K,
      handler: EventHandler<TEvents[K]>
    ): Subscription {
      return bus.on(type as string, handler);
    },
    once<K extends keyof TEvents>(
      type: K,
      handler: EventHandler<TEvents[K]>
    ): Subscription {
      return bus.once(type as string, handler);
    },
    waitFor<K extends keyof TEvents>(
      type: K,
      timeout?: number
    ): Promise<BusEvent<TEvents[K]>> {
      return bus.waitFor(type as string, timeout);
    },
  };
}

// ============================================================================
// Common Event Types for Chimera
// ============================================================================

export interface ChimeraEvents extends Record<string, unknown> {
  // Session events
  'session:created': { sessionId: string };
  'session:updated': { sessionId: string; changes: Record<string, unknown> };
  'session:destroyed': { sessionId: string };

  // Model events
  'model:changed': { provider: string; model: string };
  'model:error': { error: string };

  // Generation events
  'generation:started': { requestId: string };
  'generation:progress': { requestId: string; progress: number };
  'generation:completed': { requestId: string; result: string };
  'generation:error': { requestId: string; error: string };

  // Attack events
  'attack:started': { attackId: string };
  'attack:progress': { attackId: string; phase: string; progress: number };
  'attack:completed': { attackId: string; success: boolean };
  'attack:error': { attackId: string; error: string };

  // Strategy events
  'strategy:discovered': { strategyId: string; score: number };
  'strategy:updated': { strategyId: string };
  'strategy:deleted': { strategyId: string };

  // UI events
  'ui:theme-changed': { theme: 'light' | 'dark' | 'system' };
  'ui:sidebar-toggled': { open: boolean };
  'ui:notification': { type: 'info' | 'success' | 'warning' | 'error'; message: string };

  // Error events
  'error:occurred': AppError;
  'error:recovered': AppError;
}

// Import AppError type for event typing
import type { AppError } from './error-handling';

/**
 * Create a typed Chimera event emitter
 */
export function createChimeraEventBus(bus?: EventBus) {
  return createTypedEmitter<ChimeraEvents>(bus || getEventBus());
}