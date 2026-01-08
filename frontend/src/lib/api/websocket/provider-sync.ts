/**
 * Provider Sync WebSocket
 * 
 * Real-time synchronization of provider status, rate limits,
 * and availability updates.
 */

import type { ProviderStatus, RateLimitInfo } from '@/types/provider-management-types';

export interface ProviderSyncMessage {
  type: 'status_update' | 'rate_limit_alert' | 'provider_available' | 'provider_unavailable' | 'error';
  provider_id: string;
  data: unknown;
  timestamp: string;
}

export interface ProviderStatusUpdate {
  provider_id: string;
  status: ProviderStatus;
}

export interface RateLimitAlert {
  provider_id: string;
  model_id?: string;
  rate_limit: RateLimitInfo;
  severity: 'warning' | 'critical';
  message: string;
}

export interface ProviderSyncOptions {
  onStatusUpdate?: (update: ProviderStatusUpdate) => void;
  onRateLimitAlert?: (alert: RateLimitAlert) => void;
  onProviderAvailable?: (providerId: string) => void;
  onProviderUnavailable?: (providerId: string, reason: string) => void;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export interface ProviderSyncConnection {
  close: () => void;
  send: (message: unknown) => void;
  subscribe: (providerId: string) => void;
  unsubscribe: (providerId: string) => void;
  isConnected: () => boolean;
}

/**
 * Create a WebSocket connection for provider synchronization
 */
export function createProviderSyncConnection(
  options: ProviderSyncOptions = {}
): ProviderSyncConnection {
  const {
    onStatusUpdate,
    onRateLimitAlert,
    onProviderAvailable,
    onProviderUnavailable,
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

  function connect() {
    try {
      ws = new WebSocket(`${wsUrl}/ws/provider-sync`);

      ws.onopen = () => {
        reconnectAttempts = 0;
        onConnect?.();
        
        // Re-subscribe to previously subscribed providers
        subscribedProviders.forEach((providerId) => {
          send({ type: 'subscribe', provider_id: providerId });
        });
      };

      ws.onmessage = (event) => {
        try {
          const message: ProviderSyncMessage = JSON.parse(event.data);
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

  function handleMessage(message: ProviderSyncMessage) {
    switch (message.type) {
      case 'status_update':
        onStatusUpdate?.({
          provider_id: message.provider_id,
          status: message.data as ProviderStatus,
        });
        break;
      
      case 'rate_limit_alert':
        onRateLimitAlert?.(message.data as RateLimitAlert);
        break;
      
      case 'provider_available':
        onProviderAvailable?.(message.provider_id);
        break;
      
      case 'provider_unavailable':
        onProviderUnavailable?.(
          message.provider_id,
          (message.data as { reason: string }).reason
        );
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

  function subscribe(providerId: string) {
    subscribedProviders.add(providerId);
    send({ type: 'subscribe', provider_id: providerId });
  }

  function unsubscribe(providerId: string) {
    subscribedProviders.delete(providerId);
    send({ type: 'unsubscribe', provider_id: providerId });
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
    subscribe,
    unsubscribe,
    isConnected,
  };
}

export default createProviderSyncConnection;
