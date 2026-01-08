/**
 * WebSocket Manager
 * 
 * Provides robust WebSocket connection management:
 * - Auto-reconnection with exponential backoff
 * - Heartbeat/ping-pong for connection health
 * - Message queuing when disconnected
 * - Request/response correlation
 * - Event-based architecture
 * 
 * @module lib/sync/websocket-manager
 */

import { v4 as uuidv4 } from 'uuid';
import type {
  WebSocketConfig,
  WebSocketMessage,
  WebSocketRequest,
  WebSocketResponse,
  WebSocketEvent,
  WebSocketMessageHandler,
  WebSocketEventHandler,
  ConnectionState,
  ConnectionInfo,
} from './types';

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: Required<WebSocketConfig> = {
  url: '',
  protocols: [],
  autoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectDelay: 1000,
  maxReconnectDelay: 30000,
  heartbeatInterval: 30000,
  heartbeatTimeout: 10000,
  queueSize: 100,
  debug: false,
};

// ============================================================================
// WebSocket Manager Class
// ============================================================================

export class WebSocketManager {
  private config: Required<WebSocketConfig>;
  private socket: WebSocket | null = null;
  private connectionState: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private heartbeatTimeoutTimer: ReturnType<typeof setTimeout> | null = null;
  private connectedAt: Date | null = null;
  private disconnectedAt: Date | null = null;
  private lastLatency: number | null = null;
  private lastError: Error | null = null;

  // Message handling
  private messageQueue: WebSocketMessage[] = [];
  private messageHandlers = new Map<string, Set<WebSocketMessageHandler>>();
  private eventHandlers = new Set<WebSocketEventHandler>();
  private pendingRequests = new Map<string, {
    resolve: (response: WebSocketResponse) => void;
    reject: (error: Error) => void;
    timeout: ReturnType<typeof setTimeout>;
  }>();

  constructor(config: WebSocketConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ============================================================================
  // Connection Management
  // ============================================================================

  /**
   * Connect to the WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.setConnectionState('connecting');
      this.log('Connecting to', this.config.url);

      try {
        this.socket = new WebSocket(
          this.config.url,
          this.config.protocols.length > 0 ? this.config.protocols : undefined
        );

        this.socket.onopen = () => {
          this.handleOpen();
          resolve();
        };

        this.socket.onclose = (event) => {
          this.handleClose(event);
        };

        this.socket.onerror = (event) => {
          this.handleError(event);
          if (this.connectionState === 'connecting') {
            reject(new Error('WebSocket connection failed'));
          }
        };

        this.socket.onmessage = (event) => {
          this.handleMessage(event);
        };
      } catch (error) {
        this.setConnectionState('error');
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    this.log('Disconnecting');
    this.stopReconnect();
    this.stopHeartbeat();

    if (this.socket) {
      this.socket.onclose = null; // Prevent reconnection
      this.socket.close(1000, 'Client disconnect');
      this.socket = null;
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
      latency: this.lastLatency || undefined,
      error: this.lastError || undefined,
    };
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connectionState === 'connected' && 
           this.socket?.readyState === WebSocket.OPEN;
  }

  // ============================================================================
  // Message Sending
  // ============================================================================

  /**
   * Send a message (queued if disconnected)
   */
  send<T = unknown>(type: string, payload: T): string {
    const message: WebSocketMessage<T> = {
      id: uuidv4(),
      type,
      payload,
      timestamp: Date.now(),
    };

    if (this.isConnected()) {
      this.sendRaw(message);
    } else {
      this.queueMessage(message);
    }

    return message.id;
  }

  /**
   * Send a request and wait for response
   */
  request<TReq = unknown, TRes = unknown>(
    type: string,
    payload: TReq,
    timeout = 30000
  ): Promise<WebSocketResponse<TRes>> {
    return new Promise((resolve, reject) => {
      const correlationId = uuidv4();
      
      const request: WebSocketRequest<TReq> = {
        id: uuidv4(),
        type,
        payload,
        timestamp: Date.now(),
        correlationId,
        expectResponse: true,
        timeout,
      };

      // Set up timeout
      const timeoutHandle = setTimeout(() => {
        this.pendingRequests.delete(correlationId);
        reject(new Error(`Request timeout after ${timeout}ms`));
      }, timeout);

      // Store pending request
      this.pendingRequests.set(correlationId, {
        resolve: resolve as (response: WebSocketResponse) => void,
        reject,
        timeout: timeoutHandle,
      });

      // Send the request
      if (this.isConnected()) {
        this.sendRaw(request);
      } else {
        // For requests, we reject immediately if not connected
        clearTimeout(timeoutHandle);
        this.pendingRequests.delete(correlationId);
        reject(new Error('Not connected'));
      }
    });
  }

  /**
   * Send raw message to socket
   */
  private sendRaw(message: WebSocketMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
      this.log('Sent:', message.type);
    }
  }

  /**
   * Queue message for later sending
   */
  private queueMessage(message: WebSocketMessage): void {
    if (this.messageQueue.length >= this.config.queueSize) {
      // Remove oldest message
      this.messageQueue.shift();
    }
    this.messageQueue.push(message);
    this.log('Queued message:', message.type, `(${this.messageQueue.length} in queue)`);
  }

  /**
   * Flush queued messages
   */
  private flushQueue(): void {
    if (this.messageQueue.length === 0) return;

    this.log(`Flushing ${this.messageQueue.length} queued messages`);
    
    while (this.messageQueue.length > 0 && this.isConnected()) {
      const message = this.messageQueue.shift();
      if (message) {
        this.sendRaw(message);
      }
    }
  }

  // ============================================================================
  // Event Handling
  // ============================================================================

  /**
   * Subscribe to messages of a specific type
   */
  on<T = unknown>(type: string, handler: WebSocketMessageHandler<T>): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set());
    }
    this.messageHandlers.get(type)!.add(handler as WebSocketMessageHandler);

    // Return unsubscribe function
    return () => {
      this.messageHandlers.get(type)?.delete(handler as WebSocketMessageHandler);
    };
  }

  /**
   * Subscribe to all messages
   */
  onAny<T = unknown>(handler: WebSocketMessageHandler<T>): () => void {
    return this.on('*', handler);
  }

  /**
   * Subscribe to connection events
   */
  onEvent(handler: WebSocketEventHandler): () => void {
    this.eventHandlers.add(handler);
    return () => {
      this.eventHandlers.delete(handler);
    };
  }

  /**
   * Remove all handlers for a message type
   */
  off(type: string): void {
    this.messageHandlers.delete(type);
  }

  /**
   * Remove all handlers
   */
  offAll(): void {
    this.messageHandlers.clear();
    this.eventHandlers.clear();
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

    // Start heartbeat
    this.startHeartbeat();

    // Flush queued messages
    this.flushQueue();

    // Emit event
    this.emitEvent({
      type: 'open',
      timestamp: Date.now(),
    });

    // If this was a reconnection, emit reconnected event
    if (this.disconnectedAt) {
      this.emitEvent({
        type: 'reconnected',
        timestamp: Date.now(),
        data: {
          attempts: this.reconnectAttempts,
          downtime: Date.now() - this.disconnectedAt.getTime(),
        },
      });
    }
  }

  private handleClose(event: CloseEvent): void {
    this.log('Disconnected:', event.code, event.reason);
    this.setConnectionState('disconnected');
    this.disconnectedAt = new Date();
    this.socket = null;

    // Stop heartbeat
    this.stopHeartbeat();

    // Emit event
    this.emitEvent({
      type: 'close',
      timestamp: Date.now(),
      data: { code: event.code, reason: event.reason },
    });

    // Attempt reconnection if enabled and not a clean close
    if (this.config.autoReconnect && event.code !== 1000) {
      this.scheduleReconnect();
    }
  }

  private handleError(event: Event): void {
    this.log('Error:', event);
    const error = new Error('WebSocket error');
    this.lastError = error;

    this.emitEvent({
      type: 'error',
      timestamp: Date.now(),
      error,
    });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data) as WebSocketMessage;
      this.log('Received:', message.type);

      // Handle heartbeat response
      if (message.type === 'pong') {
        this.handlePong(message);
        return;
      }

      // Handle response to pending request
      if (message.correlationId && this.pendingRequests.has(message.correlationId)) {
        const pending = this.pendingRequests.get(message.correlationId)!;
        clearTimeout(pending.timeout);
        this.pendingRequests.delete(message.correlationId);
        pending.resolve(message as WebSocketResponse);
        return;
      }

      // Emit to specific handlers
      const handlers = this.messageHandlers.get(message.type);
      if (handlers) {
        handlers.forEach((handler) => handler(message));
      }

      // Emit to wildcard handlers
      const wildcardHandlers = this.messageHandlers.get('*');
      if (wildcardHandlers) {
        wildcardHandlers.forEach((handler) => handler(message));
      }

      // Emit message event
      this.emitEvent({
        type: 'message',
        timestamp: Date.now(),
        data: message,
      });
    } catch (error) {
      this.log('Failed to parse message:', error);
    }
  }

  // ============================================================================
  // Heartbeat
  // ============================================================================

  private startHeartbeat(): void {
    if (this.config.heartbeatInterval <= 0) return;

    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      this.sendPing();
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }
  }

  private sendPing(): void {
    if (!this.isConnected()) return;

    const pingTime = Date.now();
    
    this.sendRaw({
      id: uuidv4(),
      type: 'ping',
      payload: { timestamp: pingTime },
      timestamp: pingTime,
    });

    // Set timeout for pong response
    this.heartbeatTimeoutTimer = setTimeout(() => {
      this.log('Heartbeat timeout - connection may be dead');
      this.socket?.close(4000, 'Heartbeat timeout');
    }, this.config.heartbeatTimeout);
  }

  private handlePong(message: WebSocketMessage): void {
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }

    const pingTime = (message.payload as { timestamp?: number })?.timestamp;
    if (pingTime) {
      this.lastLatency = Date.now() - pingTime;
      this.log('Latency:', this.lastLatency, 'ms');
    }

    this.emitEvent({
      type: 'heartbeat',
      timestamp: Date.now(),
      data: { latency: this.lastLatency },
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
    const delay = Math.min(
      this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      this.config.maxReconnectDelay
    );

    this.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.emitEvent({
      type: 'reconnecting',
      timestamp: Date.now(),
      data: {
        attempt: this.reconnectAttempts,
        delay,
        maxAttempts: this.config.maxReconnectAttempts,
      },
    });

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch((error) => {
        this.log('Reconnection failed:', error);
      });
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
  }

  private emitEvent(event: WebSocketEvent): void {
    this.eventHandlers.forEach((handler) => handler(event));
  }

  private log(...args: unknown[]): void {
    if (this.config.debug) {
      console.log('[WebSocket]', ...args);
    }
  }
}

// ============================================================================
// React Hook
// ============================================================================

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';

export interface UseWebSocketOptions extends WebSocketConfig {
  /** Connect on mount */
  connectOnMount?: boolean;
  /** Reconnect on URL change */
  reconnectOnUrlChange?: boolean;
}

export interface UseWebSocketReturn {
  /** Current connection state */
  connectionState: ConnectionState;
  /** Connection info */
  connectionInfo: ConnectionInfo;
  /** Is connected */
  isConnected: boolean;
  /** Connect to server */
  connect: () => Promise<void>;
  /** Disconnect from server */
  disconnect: () => void;
  /** Send a message */
  send: <T = unknown>(type: string, payload: T) => string;
  /** Send a request and wait for response */
  request: <TReq = unknown, TRes = unknown>(
    type: string,
    payload: TReq,
    timeout?: number
  ) => Promise<WebSocketResponse<TRes>>;
  /** Subscribe to messages */
  on: <T = unknown>(type: string, handler: WebSocketMessageHandler<T>) => () => void;
  /** Subscribe to events */
  onEvent: (handler: WebSocketEventHandler) => () => void;
  /** Last error */
  error: Error | null;
  /** Latency in ms */
  latency: number | null;
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    connectOnMount = true,
    reconnectOnUrlChange = true,
    ...config
  } = options;

  const managerRef = useRef<WebSocketManager | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [error, setError] = useState<Error | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  // Track connection info in state to avoid accessing ref during render
  const [connectionInfo, setConnectionInfo] = useState<ConnectionInfo>({
    state: 'disconnected',
    url: config.url,
    reconnectAttempts: 0,
  });

  // Memoize the URL for dependency tracking
  const configUrl = useMemo(() => reconnectOnUrlChange ? config.url : null, [reconnectOnUrlChange, config.url]);

  // Initialize manager
  useEffect(() => {
    managerRef.current = new WebSocketManager(config);

    // Subscribe to events
    const unsubscribe = managerRef.current.onEvent((event) => {
      const info = managerRef.current?.getConnectionInfo();
      if (info) {
        setConnectionInfo(info);
      }

      switch (event.type) {
        case 'open':
        case 'reconnected':
          setConnectionState('connected');
          setError(null);
          break;
        case 'close':
          setConnectionState('disconnected');
          break;
        case 'error':
          setError(event.error || null);
          break;
        case 'reconnecting':
          setConnectionState('reconnecting');
          break;
        case 'heartbeat':
          setLatency((event.data as { latency?: number })?.latency || null);
          break;
      }
    });

    // Connect on mount if enabled
    if (connectOnMount) {
      managerRef.current.connect().catch(setError);
    }

    return () => {
      unsubscribe();
      managerRef.current?.disconnect();
    };
  }, [configUrl, config, connectOnMount]);

  const connect = useCallback(async () => {
    await managerRef.current?.connect();
  }, []);

  const disconnect = useCallback(() => {
    managerRef.current?.disconnect();
  }, []);

  const send = useCallback(<T = unknown>(type: string, payload: T) => {
    return managerRef.current?.send(type, payload) || '';
  }, []);

  const request = useCallback(async <TReq = unknown, TRes = unknown>(
    type: string,
    payload: TReq,
    timeout?: number
  ) => {
    if (!managerRef.current) {
      throw new Error('WebSocket manager not initialized');
    }
    return managerRef.current.request<TReq, TRes>(type, payload, timeout);
  }, []);

  const on = useCallback(<T = unknown>(
    type: string,
    handler: WebSocketMessageHandler<T>
  ) => {
    return managerRef.current?.on(type, handler) || (() => {});
  }, []);

  const onEvent = useCallback((handler: WebSocketEventHandler) => {
    return managerRef.current?.onEvent(handler) || (() => {});
  }, []);

  return {
    connectionState,
    connectionInfo,
    isConnected: connectionState === 'connected',
    connect,
    disconnect,
    send,
    request,
    on,
    onEvent,
    error,
    latency,
  };
}

// ============================================================================
// Singleton Instance
// ============================================================================

let defaultManager: WebSocketManager | null = null;

/**
 * Get or create the default WebSocket manager
 */
export function getWebSocketManager(config?: WebSocketConfig): WebSocketManager {
  if (!defaultManager && config) {
    defaultManager = new WebSocketManager(config);
  }
  if (!defaultManager) {
    throw new Error('WebSocket manager not initialized. Provide config on first call.');
  }
  return defaultManager;
}

/**
 * Reset the default WebSocket manager
 */
export function resetWebSocketManager(): void {
  defaultManager?.disconnect();
  defaultManager = null;
}