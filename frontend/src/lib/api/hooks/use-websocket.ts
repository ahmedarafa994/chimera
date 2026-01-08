/**
 * WebSocket Hooks
 * React hooks for WebSocket connections and real-time updates
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { wsManager, ConnectionState, WebSocketConfig, StreamHandler } from '../websocket-manager';
import { WebSocketMessage, StreamEvent } from '../types';
import { logger } from '../logger';

// ============================================================================
// Types
// ============================================================================

/**
 * Model selection update payload
 */
interface ModelSelectionPayload {
  type: 'model_selection_update';
  provider: string;
  model: string;
}

/**
 * Progress event data
 */
interface ProgressEventData {
  progress?: number;
  message?: string;
  error?: string;
}

/**
 * Type guard for model selection payload
 */
function isModelSelectionPayload(payload: unknown): payload is ModelSelectionPayload {
  return (
    typeof payload === 'object' &&
    payload !== null &&
    'type' in payload &&
    (payload as Record<string, unknown>).type === 'model_selection_update' &&
    'provider' in payload &&
    'model' in payload &&
    typeof (payload as Record<string, unknown>).provider === 'string' &&
    typeof (payload as Record<string, unknown>).model === 'string'
  );
}

/**
 * Type guard for progress event data
 */
function isProgressEventData(data: unknown): data is ProgressEventData {
  return typeof data === 'object' && data !== null;
}

export interface UseWebSocketOptions {
  url: string;
  autoConnect?: boolean;
  reconnect?: boolean;
  onMessage?: (message: WebSocketMessage) => void;
  onStateChange?: (state: ConnectionState) => void;
}

export interface UseWebSocketResult {
  connectionState: ConnectionState;
  isConnected: boolean;
  connect: () => Promise<void>;
  disconnect: () => void;
  send: (message: WebSocketMessage) => void;
  sendData: <T = unknown>(payload: T, requestId?: string) => void;
}

export interface UseStreamOptions {
  streamId: string;
  onEvent: StreamHandler;
  autoStart?: boolean;
}

export interface UseStreamResult<T = unknown> {
  isStreaming: boolean;
  start: () => void;
  stop: () => void;
  send: (data: T) => void;
  events: StreamEvent[];
}

// ============================================================================
// Hooks
// ============================================================================

/**
 * Hook for WebSocket connection management
 */
export function useWebSocket(options: UseWebSocketOptions): UseWebSocketResult {
  const { url, autoConnect = true, reconnect = true, onMessage, onStateChange } = options;

  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const messageHandlerRef = useRef(onMessage);
  const stateHandlerRef = useRef(onStateChange);

  // Keep refs updated
  useEffect(() => {
    messageHandlerRef.current = onMessage;
    stateHandlerRef.current = onStateChange;
  }, [onMessage, onStateChange]);

  // Subscribe to connection state changes
  useEffect(() => {
    const unsubscribe = wsManager.onStateChange((state) => {
      setConnectionState(state);
      stateHandlerRef.current?.(state);
    });

    return unsubscribe;
  }, []);

  // Subscribe to messages
  useEffect(() => {
    if (!messageHandlerRef.current) return;

    const unsubscribe = wsManager.onMessage((message) => {
      messageHandlerRef.current?.(message);
    });

    return unsubscribe;
  }, []);

  // Auto-connect
  useEffect(() => {
    if (autoConnect) {
      const config: WebSocketConfig = { url, reconnect };
      wsManager.connect(config).catch((error) => {
        logger.logError('WebSocket auto-connect failed', error);
      });
    }

    return () => {
      if (autoConnect) {
        wsManager.disconnect();
      }
    };
  }, [url, autoConnect, reconnect]);

  const connect = useCallback(async () => {
    const config: WebSocketConfig = { url, reconnect };
    await wsManager.connect(config);
  }, [url, reconnect]);

  const disconnect = useCallback(() => {
    wsManager.disconnect();
  }, []);

  const send = useCallback((message: WebSocketMessage) => {
    wsManager.send(message);
  }, []);

  const sendData = useCallback(<T = unknown>(payload: T, requestId?: string) => {
    wsManager.sendData(payload, requestId);
  }, []);

  return {
    connectionState,
    isConnected: connectionState === 'connected',
    connect,
    disconnect,
    send,
    sendData,
  };
}

/**
 * Hook for streaming data
 */
export function useStream<T = unknown>(options: UseStreamOptions): UseStreamResult<T> {
  const { streamId, onEvent, autoStart = false } = options;

  const [isStreaming, setIsStreaming] = useState(false);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const eventHandlerRef = useRef(onEvent);

  useEffect(() => {
    eventHandlerRef.current = onEvent;
  }, [onEvent]);

  const handleEvent = useCallback((event: StreamEvent) => {
    setEvents((prev) => [...prev, event]);
    eventHandlerRef.current(event);

    if (event.event === 'end' || event.event === 'error') {
      setIsStreaming(false);
    }
  }, []);

  const start = useCallback(() => {
    setIsStreaming(true);
    setEvents([]);
    wsManager.onStream(streamId, handleEvent);
  }, [streamId, handleEvent]);

  const stop = useCallback(() => {
    setIsStreaming(false);
  }, []);

  const send = useCallback(
    (data: T) => {
      wsManager.sendData(data, streamId);
    },
    [streamId]
  );

  useEffect(() => {
    if (autoStart) {
      start();
    }
  }, [autoStart, start]);

  return {
    isStreaming,
    start,
    stop,
    send,
    events,
  };
}

/**
 * Hook for real-time model selection sync
 */
export function useModelSelectionSync(onUpdate: (provider: string, model: string) => void) {
  const [lastUpdate, setLastUpdate] = useState<{ provider: string; model: string } | null>(null);

  useEffect(() => {
    const unsubscribe = wsManager.onMessage((message) => {
      if (message.type === 'data' && isModelSelectionPayload(message.payload)) {
        const { provider, model } = message.payload;
        setLastUpdate({ provider, model });
        onUpdate(provider, model);
      }
    });

    return unsubscribe;
  }, [onUpdate]);

  const broadcastSelection = useCallback((provider: string, model: string) => {
    wsManager.sendData({
      type: 'model_selection_update',
      provider,
      model,
    });
  }, []);

  return {
    lastUpdate,
    broadcastSelection,
  };
}
