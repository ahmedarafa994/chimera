/**
 * WebSocket Manager for Project Chimera Frontend
 * 
 * Provides a reusable WebSocket connection manager with:
 * - Auto-reconnection with exponential backoff
 * - Heartbeat/ping-pong support
 * - Event-based message handling
 * - Connection state management
 * 
 * Used by provider-sync.ts, model-updates.ts, and autoadv-service.ts
 */

import { getCurrentApiUrl } from "./api-enhanced";

// =============================================================================
// Types
// =============================================================================

export interface WebSocketManagerOptions {
  /** Enable auto-reconnect (default: true) */
  reconnect?: boolean;
  /** Maximum reconnect attempts (default: 10) */
  maxReconnectAttempts?: number;
  /** Base reconnect interval in ms (default: 1000) */
  reconnectInterval?: number;
  /** Connection timeout in ms (default: 10000) */
  connectionTimeout?: number;
  /** Heartbeat interval in ms (default: 30000, 0 to disable) */
  heartbeatInterval?: number;
}

export type WebSocketState = "connecting" | "connected" | "disconnected" | "reconnecting" | "error";

type MessageHandler<T = unknown> = (data: T) => void;
type OpenHandler = () => void;
type CloseHandler = (event?: CloseEvent) => void;
type ErrorHandler = (error: Error) => void;

// =============================================================================
// Default Options
// =============================================================================

const DEFAULT_OPTIONS: Required<WebSocketManagerOptions> = {
  reconnect: true,
  maxReconnectAttempts: 10,
  reconnectInterval: 1000,
  connectionTimeout: 10000,
  heartbeatInterval: 30000,
};

// =============================================================================
// WebSocket Manager Class
// =============================================================================

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private options: Required<WebSocketManagerOptions>;
  private state: WebSocketState = "disconnected";
  private reconnectAttempts: number = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimer: NodeJS.Timeout | null = null;
  private isIntentionalClose: boolean = false;

  // Event handlers
  private messageHandlers: MessageHandler[] = [];
  private openHandlers: OpenHandler[] = [];
  private closeHandlers: CloseHandler[] = [];
  private errorHandlers: ErrorHandler[] = [];

  constructor(path: string, options: WebSocketManagerOptions = {}) {
    this.url = this.buildWsUrl(path);
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  // ===========================================================================
  // Public Methods
  // ===========================================================================

  /**
   * Connect to the WebSocket server
   */
  connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      console.warn("[WebSocketManager] Already connected or connecting");
      return;
    }

    this.isIntentionalClose = false;
    this.state = "connecting";

    try {
      this.ws = new WebSocket(this.url);
      this.setupEventHandlers();
      this.startConnectionTimeout();
    } catch (error) {
      this.state = "error";
      this.notifyError(error instanceof Error ? error : new Error("Failed to create WebSocket"));
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    this.isIntentionalClose = true;
    this.clearTimers();
    
    if (this.ws) {
      this.ws.close(1000, "Client disconnect");
      this.ws = null;
    }
    
    this.state = "disconnected";
    this.reconnectAttempts = 0;
  }

  /**
   * Send a message through the WebSocket
   */
  send<T>(data: T): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn("[WebSocketManager] Cannot send: not connected");
      return false;
    }

    try {
      const payload = typeof data === "string" ? data : JSON.stringify(data);
      this.ws.send(payload);
      return true;
    } catch (error) {
      console.error("[WebSocketManager] Send error:", error);
      return false;
    }
  }

  /**
   * Get current connection state
   */
  getState(): WebSocketState {
    return this.state;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.state === "connected" && this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get reconnect attempts count
   */
  getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }

  // ===========================================================================
  // Event Registration
  // ===========================================================================

  /**
   * Register a message handler
   */
  onMessage<T = unknown>(handler: MessageHandler<T>): () => void {
    this.messageHandlers.push(handler as MessageHandler);
    return () => {
      this.messageHandlers = this.messageHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register an open handler
   */
  onOpen(handler: OpenHandler): () => void {
    this.openHandlers.push(handler);
    return () => {
      this.openHandlers = this.openHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register a close handler
   */
  onClose(handler: CloseHandler): () => void {
    this.closeHandlers.push(handler);
    return () => {
      this.closeHandlers = this.closeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register an error handler
   */
  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.push(handler);
    return () => {
      this.errorHandlers = this.errorHandlers.filter((h) => h !== handler);
    };
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private buildWsUrl(path: string): string {
    const apiUrl = getCurrentApiUrl();
    const wsProtocol = apiUrl.startsWith("https") ? "wss" : "ws";
    const wsUrl = apiUrl.replace(/^https?/, wsProtocol);
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    return `${wsUrl}${normalizedPath}`;
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      this.clearConnectionTimeout();
      this.state = "connected";
      this.reconnectAttempts = 0;
      this.startHeartbeat();
      this.notifyOpen();
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle pong response
        if (data.type === "pong") {
          return;
        }
        
        this.notifyMessage(data);
      } catch {
        // If not JSON, pass raw data
        this.notifyMessage(event.data);
      }
    };

    this.ws.onclose = (event) => {
      this.clearTimers();
      this.state = "disconnected";
      this.notifyClose(event);

      // Auto-reconnect if not intentional close
      if (!this.isIntentionalClose && !event.wasClean) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.state = "error";
      this.notifyError(new Error("WebSocket error"));
    };
  }

  private startConnectionTimeout(): void {
    this.clearConnectionTimeout();
    
    this.connectionTimer = setTimeout(() => {
      if (this.ws?.readyState === WebSocket.CONNECTING) {
        this.ws.close();
        this.state = "error";
        this.notifyError(new Error("Connection timeout"));
        this.scheduleReconnect();
      }
    }, this.options.connectionTimeout);
  }

  private clearConnectionTimeout(): void {
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }
  }

  private startHeartbeat(): void {
    if (this.options.heartbeatInterval <= 0) return;
    
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: "ping", timestamp: new Date().toISOString() });
      }
    }, this.options.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect(): void {
    if (!this.options.reconnect || this.isIntentionalClose) {
      return;
    }

    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      this.state = "error";
      this.notifyError(new Error("Max reconnection attempts reached"));
      return;
    }

    this.state = "reconnecting";

    // Exponential backoff with jitter
    const delay = this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts);
    const jitter = delay * 0.2 * (Math.random() * 2 - 1);
    const actualDelay = Math.min(delay + jitter, 30000); // Cap at 30 seconds

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, actualDelay);
  }

  private clearTimers(): void {
    this.clearConnectionTimeout();
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  // ===========================================================================
  // Event Notification
  // ===========================================================================

  private notifyMessage<T>(data: T): void {
    this.messageHandlers.forEach((handler) => {
      try {
        handler(data);
      } catch (error) {
        console.error("[WebSocketManager] Message handler error:", error);
      }
    });
  }

  private notifyOpen(): void {
    this.openHandlers.forEach((handler) => {
      try {
        handler();
      } catch (error) {
        console.error("[WebSocketManager] Open handler error:", error);
      }
    });
  }

  private notifyClose(event?: CloseEvent): void {
    this.closeHandlers.forEach((handler) => {
      try {
        handler(event);
      } catch (error) {
        console.error("[WebSocketManager] Close handler error:", error);
      }
    });
  }

  private notifyError(error: Error): void {
    this.errorHandlers.forEach((handler) => {
      try {
        handler(error);
      } catch (err) {
        console.error("[WebSocketManager] Error handler error:", err);
      }
    });
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create a new WebSocket manager instance
 */
export function createWebSocketManager(
  path: string,
  options?: WebSocketManagerOptions
): WebSocketManager {
  return new WebSocketManager(path, options);
}

export default WebSocketManager;