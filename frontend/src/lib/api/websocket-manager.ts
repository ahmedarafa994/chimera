/**
 * WebSocket Manager
 * Handles real-time bidirectional communication
 */

import { WebSocketMessage, StreamEvent } from './types';
import { authManager } from './auth-manager';
import { logger } from './logger';

// ============================================================================
// Configuration
// ============================================================================

const RECONNECT_INITIAL_DELAY = 1000;
const RECONNECT_MAX_DELAY = 30000;
const RECONNECT_BACKOFF_MULTIPLIER = 2;
const HEARTBEAT_INTERVAL = 30000;
const CONNECTION_TIMEOUT = 10000;

// ============================================================================
// Types
// ============================================================================

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  reconnect?: boolean;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

export type MessageHandler = (message: WebSocketMessage) => void;
export type StreamHandler = (event: StreamEvent) => void;
export type StateHandler = (state: ConnectionState) => void;

// ============================================================================
// WebSocket Manager
// ============================================================================

class WebSocketManager {
  private socket: WebSocket | null = null;
  private config: WebSocketConfig | null = null;
  private connectionState: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private streamHandlers: Map<string, StreamHandler> = new Map();
  private stateHandlers: Set<StateHandler> = new Set();
  private pendingMessages: WebSocketMessage[] = [];

  /**
   * Connect to WebSocket server
   */
  connect(config: WebSocketConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.config = config;
      this.setConnectionState('connecting');

      const url = this.buildUrl(config.url);
      logger.logWebSocket('Connecting', { url });

      try {
        this.socket = new WebSocket(url, config.protocols);
      } catch (error) {
        this.setConnectionState('disconnected');
        reject(error);
        return;
      }

      const connectionTimeout = setTimeout(() => {
        if (this.connectionState === 'connecting') {
          this.socket?.close();
          reject(new Error('Connection timeout'));
        }
      }, CONNECTION_TIMEOUT);

      this.socket.onopen = () => {
        clearTimeout(connectionTimeout);
        this.reconnectAttempts = 0;
        this.setConnectionState('connected');
        this.startHeartbeat();
        this.flushPendingMessages();
        logger.logWebSocket('Connected', { url });
        resolve();
      };

      this.socket.onclose = (event) => {
        clearTimeout(connectionTimeout);
        this.stopHeartbeat();
        logger.logWebSocket('Disconnected', { code: event.code, reason: event.reason });

        if (config.reconnect !== false && !event.wasClean) {
          this.scheduleReconnect();
        } else {
          this.setConnectionState('disconnected');
        }
      };

      this.socket.onerror = (error) => {
        logger.logError('WebSocket error', error);
      };

      this.socket.onmessage = (event) => {
        this.handleMessage(event);
      };
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    this.stopHeartbeat();

    if (this.socket) {
      this.socket.close(1000, 'Client disconnect');
      this.socket = null;
    }

    this.setConnectionState('disconnected');
    logger.logWebSocket('Disconnected by client');
  }

  /**
   * Send message
   */
  send(message: WebSocketMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
      logger.logWebSocket('Sent', { type: message.type });
    } else {
      this.pendingMessages.push(message);
      logger.logWebSocket('Queued message (not connected)', { type: message.type });
    }
  }

  /**
   * Send data message
   */
  sendData(payload: any, requestId?: string): void {
    this.send({
      type: 'data',
      payload,
      timestamp: new Date().toISOString(),
      request_id: requestId,
    });
  }

  /**
   * Subscribe to messages
   */
  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  /**
   * Subscribe to stream events
   */
  onStream(streamId: string, handler: StreamHandler): () => void {
    this.streamHandlers.set(streamId, handler);
    return () => this.streamHandlers.delete(streamId);
  }

  /**
   * Subscribe to state changes
   */
  onStateChange(handler: StateHandler): () => void {
    this.stateHandlers.add(handler);
    handler(this.connectionState);
    return () => this.stateHandlers.delete(handler);
  }

  /**
   * Get current connection state
   */
  getState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connectionState === 'connected' && this.socket?.readyState === WebSocket.OPEN;
  }

  // Private Methods

  private buildUrl(path: string): string {
    const token = authManager.isAuthenticated() ? authManager.getTenantId() : null;
    const url = new URL(path);
    
    if (token) {
      url.searchParams.set('tenant_id', token);
    }

    return url.toString();
  }

  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.stateHandlers.forEach((handler) => handler(state));
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      logger.logWebSocket('Received', { type: message.type });

      // Handle pong
      if (message.type === 'pong') {
        return;
      }

      // Handle stream events
      if (message.type === 'data' && message.request_id) {
        const streamHandler = this.streamHandlers.get(message.request_id);
        if (streamHandler) {
          streamHandler(message.payload as StreamEvent);
        }
      }

      // Notify all message handlers
      this.messageHandlers.forEach((handler) => handler(message));
    } catch (error) {
      logger.logError('Failed to parse WebSocket message', error);
    }
  }

  private scheduleReconnect(): void {
    const maxAttempts = this.config?.maxReconnectAttempts ?? 10;

    if (this.reconnectAttempts >= maxAttempts) {
      logger.logWarning('Max reconnect attempts reached');
      this.setConnectionState('disconnected');
      return;
    }

    this.setConnectionState('reconnecting');
    this.reconnectAttempts++;

    const delay = Math.min(
      RECONNECT_INITIAL_DELAY * Math.pow(RECONNECT_BACKOFF_MULTIPLIER, this.reconnectAttempts - 1),
      RECONNECT_MAX_DELAY
    );

    logger.logWebSocket('Scheduling reconnect', { attempt: this.reconnectAttempts, delay });

    this.reconnectTimeout = setTimeout(() => {
      if (this.config) {
        this.connect(this.config).catch(() => {
          this.scheduleReconnect();
        });
      }
    }, delay);
  }

  private startHeartbeat(): void {
    const interval = this.config?.heartbeatInterval ?? HEARTBEAT_INTERVAL;

    this.heartbeatInterval = setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.send({
          type: 'ping',
          payload: null,
          timestamp: new Date().toISOString(),
        });
      }
    }, interval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private flushPendingMessages(): void {
    while (this.pendingMessages.length > 0) {
      const message = this.pendingMessages.shift();
      if (message) {
        this.send(message);
      }
    }
  }
}

// Export singleton instance
export const wsManager = new WebSocketManager();

// ============================================================================
// Stream Connection Helper
// ============================================================================

export class StreamConnection {
  private streamId: string;
  private unsubscribe?: () => void;

  constructor(streamId: string) {
    this.streamId = streamId;
  }

  onEvent(handler: StreamHandler): this {
    this.unsubscribe = wsManager.onStream(this.streamId, handler);
    return this;
  }

  send(data: any): void {
    wsManager.sendData(data, this.streamId);
  }

  close(): void {
    if (this.unsubscribe) {
      this.unsubscribe();
    }
  }
}

export function createStream(streamId: string): StreamConnection {
  return new StreamConnection(streamId);
}