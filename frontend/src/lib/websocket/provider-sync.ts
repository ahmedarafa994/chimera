/**
 * Provider Selection Sync WebSocket for Project Chimera Frontend
 * 
 * Provides real-time synchronization of provider/model selection
 * across multiple browser tabs/windows using WebSocket.
 * 
 * Backend endpoint: /api/v1/providers/ws/selection
 */

import { WebSocketManager } from "../websocket-manager";
import {
  ProviderSelectionSyncMessage,
  getOrCreateSessionId,
} from "../types/provider-management-types";

// =============================================================================
// Configuration
// =============================================================================

const WS_ENDPOINT = "/api/v1/providers/ws/selection";

// =============================================================================
// Types
// =============================================================================

export interface ProviderSyncCallbacks {
  /** Called when selection is updated from another source */
  onSelectionUpdate?: (providerId: string, modelId: string) => void;
  /** Called when selection is cleared */
  onSelectionCleared?: () => void;
  /** Called when sync response is received */
  onSyncResponse?: (data: ProviderSelectionSyncMessage) => void;
  /** Called on connection */
  onConnect?: () => void;
  /** Called on disconnection */
  onDisconnect?: () => void;
  /** Called on error */
  onError?: (error: Error) => void;
}

export interface ProviderSyncOptions {
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Maximum reconnect attempts */
  maxReconnectAttempts?: number;
  /** Reconnect interval in ms */
  reconnectInterval?: number;
  /** Send heartbeat interval in ms */
  heartbeatInterval?: number;
}

const DEFAULT_OPTIONS: ProviderSyncOptions = {
  autoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectInterval: 3000,
  heartbeatInterval: 30000,
};

// =============================================================================
// Provider Sync Manager
// =============================================================================

export class ProviderSyncManager {
  private wsManager: WebSocketManager | null = null;
  private sessionId: string;
  private callbacks: ProviderSyncCallbacks;
  private options: ProviderSyncOptions;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private isConnected: boolean = false;

  constructor(
    callbacks: ProviderSyncCallbacks = {},
    options: ProviderSyncOptions = {}
  ) {
    this.sessionId = getOrCreateSessionId();
    this.callbacks = callbacks;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Connect to the WebSocket
   */
  connect(): void {
    if (this.wsManager) {
      this.disconnect();
    }

    const wsUrl = `${WS_ENDPOINT}?session_id=${encodeURIComponent(this.sessionId)}`;
    
    this.wsManager = new WebSocketManager(wsUrl, {
      reconnect: this.options.autoReconnect,
      maxReconnectAttempts: this.options.maxReconnectAttempts,
      reconnectInterval: this.options.reconnectInterval,
    });

    this.setupHandlers();
    this.wsManager.connect();
  }

  /**
   * Disconnect from the WebSocket
   */
  disconnect(): void {
    this.stopHeartbeat();
    
    if (this.wsManager) {
      this.wsManager.disconnect();
      this.wsManager = null;
    }
    
    this.isConnected = false;
  }

  /**
   * Check if connected
   */
  get connected(): boolean {
    return this.isConnected;
  }

  /**
   * Broadcast a selection update to other clients
   */
  broadcastSelectionUpdate(providerId: string, modelId: string): void {
    if (!this.wsManager || !this.isConnected) {
      console.warn("Cannot broadcast: WebSocket not connected");
      return;
    }

    const message: ProviderSelectionSyncMessage = {
      type: "selection_update",
      session_id: this.sessionId,
      provider_id: providerId,
      model_id: modelId,
      timestamp: new Date().toISOString(),
    };

    this.wsManager.send(message);
  }

  /**
   * Broadcast that selection was cleared
   */
  broadcastSelectionCleared(): void {
    if (!this.wsManager || !this.isConnected) {
      console.warn("Cannot broadcast: WebSocket not connected");
      return;
    }

    const message: ProviderSelectionSyncMessage = {
      type: "selection_cleared",
      session_id: this.sessionId,
      timestamp: new Date().toISOString(),
    };

    this.wsManager.send(message);
  }

  /**
   * Request sync from server
   */
  requestSync(): void {
    if (!this.wsManager || !this.isConnected) {
      console.warn("Cannot request sync: WebSocket not connected");
      return;
    }

    const message: ProviderSelectionSyncMessage = {
      type: "sync_request",
      session_id: this.sessionId,
      timestamp: new Date().toISOString(),
    };

    this.wsManager.send(message);
  }

  /**
   * Update session ID
   */
  updateSessionId(newSessionId: string): void {
    this.sessionId = newSessionId;
    
    // Reconnect with new session ID
    if (this.isConnected) {
      this.disconnect();
      this.connect();
    }
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private setupHandlers(): void {
    if (!this.wsManager) return;

    this.wsManager.onOpen(() => {
      this.isConnected = true;
      this.startHeartbeat();
      this.callbacks.onConnect?.();
      
      // Request initial sync
      this.requestSync();
    });

    this.wsManager.onClose(() => {
      this.isConnected = false;
      this.stopHeartbeat();
      this.callbacks.onDisconnect?.();
    });

    this.wsManager.onError((error) => {
      this.callbacks.onError?.(error);
    });

    this.wsManager.onMessage((data: ProviderSelectionSyncMessage) => {
      this.handleMessage(data);
    });
  }

  private handleMessage(message: ProviderSelectionSyncMessage): void {
    // Ignore messages from our own session
    if (message.session_id === this.sessionId && message.type !== "sync_response") {
      return;
    }

    switch (message.type) {
      case "selection_update":
        if (message.provider_id && message.model_id) {
          this.callbacks.onSelectionUpdate?.(message.provider_id, message.model_id);
        }
        break;

      case "selection_cleared":
        this.callbacks.onSelectionCleared?.();
        break;

      case "sync_response":
        this.callbacks.onSyncResponse?.(message);
        break;
    }
  }

  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      this.stopHeartbeat();
    }

    if (this.options.heartbeatInterval && this.options.heartbeatInterval > 0) {
      this.heartbeatTimer = setInterval(() => {
        if (this.wsManager && this.isConnected) {
          this.wsManager.send({ type: "ping", timestamp: new Date().toISOString() });
        }
      }, this.options.heartbeatInterval);
    }
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let providerSyncInstance: ProviderSyncManager | null = null;

/**
 * Get or create the provider sync manager singleton
 */
export function getProviderSyncManager(
  callbacks?: ProviderSyncCallbacks,
  options?: ProviderSyncOptions
): ProviderSyncManager {
  if (!providerSyncInstance) {
    providerSyncInstance = new ProviderSyncManager(callbacks, options);
  } else if (callbacks) {
    // Update callbacks if provided
    providerSyncInstance = new ProviderSyncManager(callbacks, options);
  }
  return providerSyncInstance;
}

/**
 * Destroy the provider sync manager singleton
 */
export function destroyProviderSyncManager(): void {
  if (providerSyncInstance) {
    providerSyncInstance.disconnect();
    providerSyncInstance = null;
  }
}

// =============================================================================
// React Hook (for convenience)
// =============================================================================

/**
 * Hook configuration for provider sync
 */
export interface UseProviderSyncConfig {
  /** Called when selection is updated */
  onSelectionUpdate?: (providerId: string, modelId: string) => void;
  /** Called when selection is cleared */
  onSelectionCleared?: () => void;
  /** Auto-connect on mount */
  autoConnect?: boolean;
}

/**
 * Create provider sync hook state
 */
export function createProviderSyncHookState() {
  return {
    isConnected: false,
    lastUpdate: null as ProviderSelectionSyncMessage | null,
  };
}