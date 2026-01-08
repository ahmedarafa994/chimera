/**
 * Server-Sent Events (SSE) Manager
 * 
 * Provides SSE connection management for one-way real-time updates:
 * - Auto-reconnection with exponential backoff
 * - Event type filtering
 * - Connection state tracking
 * - React hook integration
 * 
 * @module lib/sync/sse-manager
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  SSEConfig,
  SSEMessage,
  SSEEventHandler,
  ConnectionState,
  ConnectionInfo,
} from './types';

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: Required<SSEConfig> = {
  url: '',
  headers: {},
  autoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectDelay: 1000,
  debug: false,
};

// ============================================================================
// SSE Manager Class
// ============================================================================

export class SSEManager {
  private config: Required<SSEConfig>;
  private eventSource: EventSource | null = null;
  private connectionState: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private connectedAt: Date | null = null;
  private disconnectedAt: Date | null = null;
  private lastError: Error | null = null;

  // Event handling
  private eventHandlers = new Map<string, Set<SSEEventHandler>>();
  private connectionHandlers = new Set<(state: ConnectionState) => void>();

  constructor(config: SSEConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ============================================================================
  // Connection Management
  // ============================================================================

  /**
   * Connect to the SSE endpoint
   */
  connect(): void {
    if (this.eventSource?.readyState === EventSource.OPEN) {
      return;
    }

    this.setConnectionState('connecting');
    this.log('Connecting to', this.config.url);

    try {
      // Note: Standard EventSource doesn't support custom headers
      // For headers support, use a polyfill like event-source-polyfill
      this.eventSource = new EventSource(this.config.url);

      this.eventSource.onopen = () => {
        this.handleOpen();
      };

      this.eventSource.onerror = (event) => {
        this.handleError(event);
      };

      // Default message handler
      this.eventSource.onmessage = (event) => {
        this.handleMessage('message', event);
      };
    } catch (error) {
      this.setConnectionState('error');
      this.lastError = error instanceof Error ? error : new Error(String(error));
    }
  }

  /**
   * Disconnect from the SSE endpoint
   */
  disconnect(): void {
    this.log('Disconnecting');
    this.stopReconnect();

    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }

    this.setConnectionState('disconnected');
    this.disconnectedAt = new Date();
  }

  /**
   * Get current connection info
   */
  getConnectionInfo(): ConnectionInfo {
    return {
      state: this.connectionState,
      url: this.config.url,
      connectedAt: this.connectedAt || undefined,
      disconnectedAt: this.disconnectedAt || undefined,
      reconnectAttempts: this.reconnectAttempts,
      error: this.lastError || undefined,
    };
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connectionState === 'connected' && 
           this.eventSource?.readyState === EventSource.OPEN;
  }

  // ============================================================================
  // Event Handling
  // ============================================================================

  /**
   * Subscribe to events of a specific type
   */
  on<T = unknown>(eventType: string, handler: SSEEventHandler<T>): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set());
      
      // Add event listener to EventSource if connected
      if (this.eventSource) {
        this.addEventSourceListener(eventType);
      }
    }
    
    this.eventHandlers.get(eventType)!.add(handler as SSEEventHandler);

    // Return unsubscribe function
    return () => {
      this.eventHandlers.get(eventType)?.delete(handler as SSEEventHandler);
    };
  }

  /**
   * Subscribe to all events
   */
  onAny<T = unknown>(handler: SSEEventHandler<T>): () => void {
    return this.on('*', handler);
  }

  /**
   * Subscribe to connection state changes
   */
  onConnectionChange(handler: (state: ConnectionState) => void): () => void {
    this.connectionHandlers.add(handler);
    return () => {
      this.connectionHandlers.delete(handler);
    };
  }

  /**
   * Remove all handlers for an event type
   */
  off(eventType: string): void {
    this.eventHandlers.delete(eventType);
  }

  /**
   * Remove all handlers
   */
  offAll(): void {
    this.eventHandlers.clear();
    this.connectionHandlers.clear();
  }

  // ============================================================================
  // Internal Event Handlers
  // ============================================================================

  private handleOpen(): void {
    this.log('Connected');
    this.setConnectionState('connected');
    this.connectedAt = new Date();
    this.reconnectAttempts = 0;
    this.lastError = null;

    // Re-add all event listeners
    this.eventHandlers.forEach((_, eventType) => {
      if (eventType !== '*' && eventType !== 'message') {
        this.addEventSourceListener(eventType);
      }
    });
  }

  private handleError(event: Event): void {
    this.log('Error:', event);
    
    const error = new Error('SSE connection error');
    this.lastError = error;

    // EventSource automatically reconnects on error
    // But we track state for UI purposes
    if (this.eventSource?.readyState === EventSource.CLOSED) {
      this.setConnectionState('disconnected');
      this.disconnectedAt = new Date();

      if (this.config.autoReconnect) {
        this.scheduleReconnect();
      }
    }
  }

  private handleMessage(eventType: string, event: MessageEvent): void {
    try {
      let data: unknown;
      
      try {
        data = JSON.parse(event.data);
      } catch {
        data = event.data;
      }

      const message: SSEMessage = {
        id: event.lastEventId || undefined,
        event: eventType,
        data,
        timestamp: Date.now(),
      };

      this.log('Received:', eventType, data);

      // Emit to specific handlers
      const handlers = this.eventHandlers.get(eventType);
      if (handlers) {
        handlers.forEach((handler) => handler(message));
      }

      // Emit to wildcard handlers
      const wildcardHandlers = this.eventHandlers.get('*');
      if (wildcardHandlers) {
        wildcardHandlers.forEach((handler) => handler(message));
      }
    } catch (error) {
      this.log('Failed to handle message:', error);
    }
  }

  private addEventSourceListener(eventType: string): void {
    if (!this.eventSource || eventType === '*') return;

    this.eventSource.addEventListener(eventType, (event) => {
      this.handleMessage(eventType, event as MessageEvent);
    });
  }

  // ============================================================================
  // Reconnection
  // ============================================================================

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.log('Max reconnection attempts reached');
      this.setConnectionState('error');
      return;
    }

    this.setConnectionState('reconnecting');
    this.reconnectAttempts++;

    // Calculate delay with exponential backoff
    const delay = this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    this.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private stopReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.connectionHandlers.forEach((handler) => handler(state));
  }

  private log(...args: unknown[]): void {
    if (this.config.debug) {
      console.log('[SSE]', ...args);
    }
  }
}

// ============================================================================
// React Hook
// ============================================================================

export interface UseSSEOptions extends SSEConfig {
  /** Connect on mount */
  connectOnMount?: boolean;
  /** Event types to subscribe to */
  eventTypes?: string[];
}

export interface UseSSEReturn<T = unknown> {
  /** Current connection state */
  connectionState: ConnectionState;
  /** Connection info */
  connectionInfo: ConnectionInfo;
  /** Is connected */
  isConnected: boolean;
  /** Connect to server */
  connect: () => void;
  /** Disconnect from server */
  disconnect: () => void;
  /** Last received message */
  lastMessage: SSEMessage<T> | null;
  /** All received messages (limited buffer) */
  messages: SSEMessage<T>[];
  /** Subscribe to events */
  on: <U = T>(eventType: string, handler: SSEEventHandler<U>) => () => void;
  /** Last error */
  error: Error | null;
}

export function useSSE<T = unknown>(options: UseSSEOptions): UseSSEReturn<T> {
  const {
    connectOnMount = true,
    eventTypes = ['message'],
    ...config
  } = options;

  const managerRef = useRef<SSEManager | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [lastMessage, setLastMessage] = useState<SSEMessage<T> | null>(null);
  const [messages, setMessages] = useState<SSEMessage<T>[]>([]);
  const [error, setError] = useState<Error | null>(null);

  // Track connection info in state to avoid accessing ref during render
  const [connectionInfo, setConnectionInfo] = useState<ConnectionInfo>({
    state: 'disconnected',
    url: config.url,
    reconnectAttempts: 0,
  });

  // Initialize manager
  useEffect(() => {
    managerRef.current = new SSEManager(config);

    // Subscribe to connection state changes
    const unsubscribeConnection = managerRef.current.onConnectionChange((state) => {
      setConnectionState(state);
      const info = managerRef.current?.getConnectionInfo();
      if (info) {
        setConnectionInfo(info);
      }
      if (state === 'error' && info?.error) {
        setError(info.error);
      }
    });

    // Subscribe to specified event types
    const unsubscribers = eventTypes.map((eventType) =>
      managerRef.current!.on<T>(eventType, (message) => {
        setLastMessage(message);
        setMessages((prev) => [...prev.slice(-99), message]); // Keep last 100 messages
      })
    );

    // Connect on mount if enabled
    if (connectOnMount) {
      managerRef.current.connect();
    }

    return () => {
      unsubscribeConnection();
      unsubscribers.forEach((unsub) => unsub());
      managerRef.current?.disconnect();
    };
  }, [config.url, connectOnMount, eventTypes]);

  const connect = useCallback(() => {
    managerRef.current?.connect();
  }, []);

  const disconnect = useCallback(() => {
    managerRef.current?.disconnect();
  }, []);

  const on = useCallback(<U = T>(eventType: string, handler: SSEEventHandler<U>) => {
    return managerRef.current?.on(eventType, handler) || (() => {});
  }, []);

  return {
    connectionState,
    connectionInfo,
    isConnected: connectionState === 'connected',
    connect,
    disconnect,
    lastMessage,
    messages,
    on,
    error,
  };
}

// ============================================================================
// Specialized Hooks
// ============================================================================

/**
 * Hook for streaming text responses (like LLM output)
 */
export interface UseStreamingTextOptions extends Omit<UseSSEOptions, 'eventTypes'> {
  /** Event type for text chunks */
  textEvent?: string;
  /** Event type for completion */
  completeEvent?: string;
  /** Event type for errors */
  errorEvent?: string;
}

export interface UseStreamingTextReturn {
  /** Accumulated text */
  text: string;
  /** Is streaming */
  isStreaming: boolean;
  /** Is complete */
  isComplete: boolean;
  /** Start streaming */
  start: () => void;
  /** Stop streaming */
  stop: () => void;
  /** Reset text */
  reset: () => void;
  /** Error if any */
  error: Error | null;
}

export function useStreamingText(options: UseStreamingTextOptions): UseStreamingTextReturn {
  const {
    textEvent = 'text',
    completeEvent = 'complete',
    errorEvent = 'error',
    connectOnMount = false,
    ...config
  } = options;

  const [text, setText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const { connect, disconnect, on, connectionState } = useSSE({
    ...config,
    connectOnMount,
    eventTypes: [textEvent, completeEvent, errorEvent],
  });

  useEffect(() => {
    const unsubText = on<{ text: string }>(textEvent, (message) => {
      setText((prev) => prev + (message.data?.text || ''));
    });

    const unsubComplete = on(completeEvent, () => {
      setIsStreaming(false);
      setIsComplete(true);
    });

    const unsubError = on<{ message: string }>(errorEvent, (message) => {
      setError(new Error(message.data?.message || 'Streaming error'));
      setIsStreaming(false);
    });

    return () => {
      unsubText();
      unsubComplete();
      unsubError();
    };
  }, [on, textEvent, completeEvent, errorEvent]);

  useEffect(() => {
    setIsStreaming(connectionState === 'connected');
  }, [connectionState]);

  const start = useCallback(() => {
    setText('');
    setIsComplete(false);
    setError(null);
    connect();
  }, [connect]);

  const stop = useCallback(() => {
    disconnect();
    setIsStreaming(false);
  }, [disconnect]);

  const reset = useCallback(() => {
    setText('');
    setIsComplete(false);
    setError(null);
  }, []);

  return {
    text,
    isStreaming,
    isComplete,
    start,
    stop,
    reset,
    error,
  };
}

// ============================================================================
// Singleton Instance
// ============================================================================

let defaultManager: SSEManager | null = null;

/**
 * Get or create the default SSE manager
 */
export function getSSEManager(config?: SSEConfig): SSEManager {
  if (!defaultManager && config) {
    defaultManager = new SSEManager(config);
  }
  if (!defaultManager) {
    throw new Error('SSE manager not initialized. Provide config on first call.');
  }
  return defaultManager;
}

/**
 * Reset the default SSE manager
 */
export function resetSSEManager(): void {
  defaultManager?.disconnect();
  defaultManager = null;
}