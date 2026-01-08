/**
 * Model Updates WebSocket
 * 
 * Real-time notifications for model availability changes,
 * new model releases, and deprecation warnings.
 */

import type { ModelInfo } from '@/types/provider-management-types';

export interface ModelUpdateMessage {
  type: 'model_available' | 'model_unavailable' | 'model_deprecated' | 'new_model' | 'model_updated' | 'error';
  model_id: string;
  provider_id: string;
  data: unknown;
  timestamp: string;
}

export interface ModelAvailabilityUpdate {
  model_id: string;
  provider_id: string;
  is_available: boolean;
  reason?: string;
}

export interface ModelDeprecationWarning {
  model_id: string;
  provider_id: string;
  deprecation_date: string;
  replacement_model?: string;
  message: string;
}

export interface NewModelNotification {
  model: ModelInfo;
  announcement?: string;
}

export interface ModelUpdatesOptions {
  onModelAvailable?: (modelId: string, providerId: string) => void;
  onModelUnavailable?: (modelId: string, providerId: string, reason: string) => void;
  onModelDeprecated?: (warning: ModelDeprecationWarning) => void;
  onNewModel?: (notification: NewModelNotification) => void;
  onModelUpdated?: (model: ModelInfo) => void;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export interface ModelUpdatesConnection {
  close: () => void;
  send: (message: unknown) => void;
  subscribeToProvider: (providerId: string) => void;
  unsubscribeFromProvider: (providerId: string) => void;
  subscribeToModel: (modelId: string, providerId: string) => void;
  unsubscribeFromModel: (modelId: string, providerId: string) => void;
  isConnected: () => boolean;
}

/**
 * Create a WebSocket connection for model updates
 */
export function createModelUpdatesConnection(
  options: ModelUpdatesOptions = {}
): ModelUpdatesConnection {
  const {
    onModelAvailable,
    onModelUnavailable,
    onModelDeprecated,
    onNewModel,
    onModelUpdated,
    onError,
    onConnect,
    onDisconnect,
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
  } = options;

  const wsUrl =
    process.env.NEXT_PUBLIC_WS_URL ||
    (() => {
      const apiUrl = process.env.NEXT_PUBLIC_CHIMERA_API_URL || process.env.NEXT_PUBLIC_API_URL;
      if (apiUrl) {
        try {
          const url = new URL(apiUrl);
          return url.origin.replace("https://", "wss://").replace("http://", "ws://");
        } catch {
          // ignore
        }
      }
      return "ws://localhost:8003";
    })();
  let ws: WebSocket | null = null;
  let reconnectAttempts = 0;
  let reconnectTimeout: NodeJS.Timeout | null = null;
  let isClosedManually = false;
  const subscribedProviders = new Set<string>();
  const subscribedModels = new Set<string>();

  function connect() {
    try {
      ws = new WebSocket(`${wsUrl}/ws/model-updates`);

      ws.onopen = () => {
        reconnectAttempts = 0;
        onConnect?.();
        
        // Re-subscribe to previously subscribed providers and models
        subscribedProviders.forEach((providerId) => {
          send({ type: 'subscribe_provider', provider_id: providerId });
        });
        subscribedModels.forEach((key) => {
          const [modelId, providerId] = key.split(':');
          send({ type: 'subscribe_model', model_id: modelId, provider_id: providerId });
        });
      };

      ws.onmessage = (event) => {
        try {
          const message: ModelUpdateMessage = JSON.parse(event.data);
          handleMessage(message);
        } catch (error) {
          onError?.(new Error(`Failed to parse WebSocket message: ${error}`));
        }
      };

      ws.onerror = (event) => {
        onError?.(new Error(`WebSocket error: ${event}`));
      };

      ws.onclose = () => {
        onDisconnect?.();
        
        if (!isClosedManually && autoReconnect && reconnectAttempts < maxReconnectAttempts) {
          reconnectTimeout = setTimeout(() => {
            reconnectAttempts++;
            connect();
          }, reconnectInterval);
        }
      };
    } catch (error) {
      onError?.(error instanceof Error ? error : new Error(String(error)));
    }
  }

  function handleMessage(message: ModelUpdateMessage) {
    switch (message.type) {
      case 'model_available':
        onModelAvailable?.(message.model_id, message.provider_id);
        break;
      
      case 'model_unavailable':
        onModelUnavailable?.(
          message.model_id,
          message.provider_id,
          (message.data as { reason: string }).reason
        );
        break;
      
      case 'model_deprecated':
        onModelDeprecated?.(message.data as ModelDeprecationWarning);
        break;
      
      case 'new_model':
        onNewModel?.(message.data as NewModelNotification);
        break;
      
      case 'model_updated':
        onModelUpdated?.(message.data as ModelInfo);
        break;
      
      case 'error':
        onError?.(new Error((message.data as { message: string }).message));
        break;
    }
  }

  function send(message: unknown) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  function subscribeToProvider(providerId: string) {
    subscribedProviders.add(providerId);
    send({ type: 'subscribe_provider', provider_id: providerId });
  }

  function unsubscribeFromProvider(providerId: string) {
    subscribedProviders.delete(providerId);
    send({ type: 'unsubscribe_provider', provider_id: providerId });
  }

  function subscribeToModel(modelId: string, providerId: string) {
    subscribedModels.add(`${modelId}:${providerId}`);
    send({ type: 'subscribe_model', model_id: modelId, provider_id: providerId });
  }

  function unsubscribeFromModel(modelId: string, providerId: string) {
    subscribedModels.delete(`${modelId}:${providerId}`);
    send({ type: 'unsubscribe_model', model_id: modelId, provider_id: providerId });
  }

  function close() {
    isClosedManually = true;
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
    }
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  function isConnected(): boolean {
    return ws !== null && ws.readyState === WebSocket.OPEN;
  }

  // Initialize connection
  connect();

  return {
    close,
    send,
    subscribeToProvider,
    unsubscribeFromProvider,
    subscribeToModel,
    unsubscribeFromModel,
    isConnected,
  };
}

export default createModelUpdatesConnection;
