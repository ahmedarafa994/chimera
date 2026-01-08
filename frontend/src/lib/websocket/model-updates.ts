/**
 * Model Updates WebSocket for Project Chimera Frontend
 * 
 * Provides real-time updates when models are added, removed, or updated
 * across providers using WebSocket.
 * 
 * Backend endpoint: /api/v1/models/updates
 */

import { WebSocketManager } from "../websocket-manager";
import {
  ModelUpdateMessage,
  ModelInfo,
  ProviderStatus,
} from "../types/provider-management-types";

// =============================================================================
// Configuration
// =============================================================================

const WS_ENDPOINT = "/api/v1/models/updates";

// =============================================================================
// Types
// =============================================================================

export interface ModelUpdateCallbacks {
  /** Called when a new model is added */
  onModelAdded?: (providerId: string, model: ModelInfo) => void;
  /** Called when a model is removed */
  onModelRemoved?: (providerId: string, modelId: string) => void;
  /** Called when a model is updated */
  onModelUpdated?: (providerId: string, modelId: string, updates: Partial<ModelInfo>) => void;
  /** Called when provider status changes */
  onProviderStatusChanged?: (providerId: string, status: ProviderStatus) => void;
  /** Called on any update (raw message) */
  onUpdate?: (message: ModelUpdateMessage) => void;
  /** Called on connection */
  onConnect?: () => void;
  /** Called on disconnection */
  onDisconnect?: () => void;
  /** Called on error */
  onError?: (error: Error) => void;
}

export interface ModelUpdatesOptions {
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Maximum reconnect attempts */
  maxReconnectAttempts?: number;
  /** Reconnect interval in ms */
  reconnectInterval?: number;
  /** Filter updates by provider IDs (empty = all) */
  providerFilter?: string[];
}

const DEFAULT_OPTIONS: ModelUpdatesOptions = {
  autoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectInterval: 3000,
  providerFilter: [],
};

// =============================================================================
// Model Updates Manager
// =============================================================================

export class ModelUpdatesManager {
  private wsManager: WebSocketManager | null = null;
  private callbacks: ModelUpdateCallbacks;
  private options: ModelUpdatesOptions;
  private isConnected: boolean = false;
  private updateBuffer: ModelUpdateMessage[] = [];
  private bufferFlushTimer: NodeJS.Timeout | null = null;

  constructor(
    callbacks: ModelUpdateCallbacks = {},
    options: ModelUpdatesOptions = {}
  ) {
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

    let wsUrl = WS_ENDPOINT;
    
    // Add provider filter if specified
    if (this.options.providerFilter && this.options.providerFilter.length > 0) {
      const params = new URLSearchParams();
      this.options.providerFilter.forEach((p) => params.append("providers", p));
      wsUrl += `?${params.toString()}`;
    }
    
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
    this.flushBuffer();
    
    if (this.bufferFlushTimer) {
      clearTimeout(this.bufferFlushTimer);
      this.bufferFlushTimer = null;
    }
    
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
   * Update provider filter
   */
  setProviderFilter(providers: string[]): void {
    this.options.providerFilter = providers;
    
    // Reconnect with new filter
    if (this.isConnected) {
      this.disconnect();
      this.connect();
    }
  }

  /**
   * Get buffered updates
   */
  getBufferedUpdates(): ModelUpdateMessage[] {
    return [...this.updateBuffer];
  }

  /**
   * Clear buffered updates
   */
  clearBuffer(): void {
    this.updateBuffer = [];
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private setupHandlers(): void {
    if (!this.wsManager) return;

    this.wsManager.onOpen(() => {
      this.isConnected = true;
      this.callbacks.onConnect?.();
    });

    this.wsManager.onClose(() => {
      this.isConnected = false;
      this.callbacks.onDisconnect?.();
    });

    this.wsManager.onError((error) => {
      this.callbacks.onError?.(error);
    });

    this.wsManager.onMessage((data: ModelUpdateMessage) => {
      this.handleMessage(data);
    });
  }

  private handleMessage(message: ModelUpdateMessage): void {
    // Apply provider filter
    if (
      this.options.providerFilter &&
      this.options.providerFilter.length > 0 &&
      !this.options.providerFilter.includes(message.provider_id)
    ) {
      return;
    }

    // Buffer the update
    this.updateBuffer.push(message);

    // Call raw update callback
    this.callbacks.onUpdate?.(message);

    // Handle specific update types
    switch (message.type) {
      case "model_added":
        if (message.data && typeof message.data === "object") {
          this.callbacks.onModelAdded?.(
            message.provider_id,
            message.data as unknown as ModelInfo
          );
        }
        break;

      case "model_removed":
        if (message.model_id) {
          this.callbacks.onModelRemoved?.(message.provider_id, message.model_id);
        }
        break;

      case "model_updated":
        if (message.model_id && message.data) {
          this.callbacks.onModelUpdated?.(
            message.provider_id,
            message.model_id,
            message.data as Partial<ModelInfo>
          );
        }
        break;

      case "provider_status_changed":
        if (message.data && "status" in message.data) {
          this.callbacks.onProviderStatusChanged?.(
            message.provider_id,
            message.data.status as ProviderStatus
          );
        }
        break;
    }

    // Schedule buffer flush
    this.scheduleBufferFlush();
  }

  private scheduleBufferFlush(): void {
    if (this.bufferFlushTimer) {
      return; // Already scheduled
    }

    // Flush buffer after 5 seconds of no updates
    this.bufferFlushTimer = setTimeout(() => {
      this.flushBuffer();
      this.bufferFlushTimer = null;
    }, 5000);
  }

  private flushBuffer(): void {
    // Keep only last 100 updates
    if (this.updateBuffer.length > 100) {
      this.updateBuffer = this.updateBuffer.slice(-100);
    }
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let modelUpdatesInstance: ModelUpdatesManager | null = null;

/**
 * Get or create the model updates manager singleton
 */
export function getModelUpdatesManager(
  callbacks?: ModelUpdateCallbacks,
  options?: ModelUpdatesOptions
): ModelUpdatesManager {
  if (!modelUpdatesInstance) {
    modelUpdatesInstance = new ModelUpdatesManager(callbacks, options);
  } else if (callbacks) {
    // Create new instance with updated callbacks
    modelUpdatesInstance.disconnect();
    modelUpdatesInstance = new ModelUpdatesManager(callbacks, options);
  }
  return modelUpdatesInstance;
}

/**
 * Destroy the model updates manager singleton
 */
export function destroyModelUpdatesManager(): void {
  if (modelUpdatesInstance) {
    modelUpdatesInstance.disconnect();
    modelUpdatesInstance = null;
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Create a model updates subscription for a specific provider
 */
export function subscribeToProviderUpdates(
  providerId: string,
  callbacks: Omit<ModelUpdateCallbacks, "onUpdate">
): () => void {
  const manager = getModelUpdatesManager(
    {
      ...callbacks,
      onUpdate: (message) => {
        if (message.provider_id === providerId) {
          // Already filtered by the specific callbacks
        }
      },
    },
    { providerFilter: [providerId] }
  );

  manager.connect();

  // Return unsubscribe function
  return () => {
    manager.disconnect();
  };
}

/**
 * Create a model updates subscription for all providers
 */
export function subscribeToAllModelUpdates(
  callbacks: ModelUpdateCallbacks
): () => void {
  const manager = getModelUpdatesManager(callbacks);
  manager.connect();

  // Return unsubscribe function
  return () => {
    manager.disconnect();
  };
}

// =============================================================================
// React Hook State
// =============================================================================

export interface ModelUpdatesHookState {
  /** Whether connected to WebSocket */
  isConnected: boolean;
  /** Recent updates */
  recentUpdates: ModelUpdateMessage[];
  /** Last update timestamp */
  lastUpdateAt: string | null;
}

/**
 * Create initial hook state
 */
export function createModelUpdatesHookState(): ModelUpdatesHookState {
  return {
    isConnected: false,
    recentUpdates: [],
    lastUpdateAt: null,
  };
}