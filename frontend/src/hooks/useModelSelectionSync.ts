/**
 * WebSocket hook for real-time model selection synchronization.
 *
 * This hook connects to the backend WebSocket endpoint for real-time updates
 * when provider/model selections change across sessions or users. It handles:
 * - SELECTION_CHANGED: Broadcast when user changes selection
 * - PROVIDER_STATUS: Provider health/availability updates
 * - MODEL_VALIDATION: Validation results for provider/model
 * - PING/PONG: Heartbeat for keep-alive
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - TanStack Query cache synchronization
 * - Connection state management
 * - Event callbacks for selection changes
 *
 * @module useModelSelectionSync
 */

import { useEffect, useRef, useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { queryKeys } from "@/lib/api/query/query-client";
import type { CurrentSelection } from "@/lib/api/query/unified-provider-queries";

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * WebSocket message types from backend
 */
export type WebSocketMessageType =
  | "SELECTION_CHANGED"
  | "PROVIDER_STATUS"
  | "MODEL_VALIDATION"
  | "PING"
  | "PONG"
  | "ERROR";

/**
 * Base WebSocket message structure
 */
export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType;
  data: T;
  timestamp?: string;
}

/**
 * Selection changed event data
 */
export interface SelectionChangedData {
  provider_id: string;
  model_id: string;
  scope: "REQUEST" | "SESSION" | "GLOBAL";
  session_id?: string;
  user_id?: string;
}

/**
 * Provider status update data
 */
export interface ProviderStatusData {
  provider_id: string;
  is_available: boolean;
  health_score?: number;
  message?: string;
}

/**
 * Model validation result data
 */
export interface ModelValidationData {
  provider_id: string;
  model_id: string;
  is_valid: boolean;
  errors?: string[];
}

/**
 * Hook options
 */
export interface UseModelSelectionSyncOptions {
  /**
   * User ID for WebSocket connection
   * @default "default"
   */
  userId?: string;

  /**
   * Session ID for filtering events
   */
  sessionId?: string;

  /**
   * Whether to enable the WebSocket connection
   * @default true
   */
  enabled?: boolean;

  /**
   * Callback when selection changes
   */
  onSelectionChanged?: (selection: SelectionChangedData) => void;

  /**
   * Callback when provider status updates
   */
  onProviderStatus?: (status: ProviderStatusData) => void;

  /**
   * Callback when model validation completes
   */
  onModelValidation?: (validation: ModelValidationData) => void;

  /**
   * Callback when connection opens
   */
  onConnect?: () => void;

  /**
   * Callback when connection closes
   */
  onDisconnect?: () => void;

  /**
   * Callback on error
   */
  onError?: (error: Event) => void;
}

/**
 * Hook return value
 */
export interface UseModelSelectionSyncReturn {
  /**
   * Whether WebSocket is connected
   */
  isConnected: boolean;

  /**
   * Number of reconnection attempts
   */
  reconnectAttempts: number;

  /**
   * Manually trigger reconnection
   */
  reconnect: () => void;

  /**
   * Disconnect WebSocket
   */
  disconnect: () => void;
}

// ============================================================================
// Constants
// ============================================================================

const MAX_RECONNECT_ATTEMPTS = 5;
const INITIAL_RECONNECT_DELAY = 1000; // 1 second
const MAX_RECONNECT_DELAY = 30000; // 30 seconds
const PING_INTERVAL = 30000; // 30 seconds

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook for real-time model selection synchronization via WebSocket.
 *
 * Connects to `/ws/model-selection?user_id={userId}` and listens for
 * selection changes, provider status updates, and model validation results.
 *
 * Automatically updates TanStack Query cache when selection changes are
 * received, keeping the UI synchronized across tabs and devices.
 *
 * @param options - Hook configuration options
 * @returns Connection state and control functions
 *
 * @example
 * ```tsx
 * const { isConnected, reconnect } = useModelSelectionSync({
 *   userId: "user123",
 *   sessionId: "session456",
 *   onSelectionChanged: (selection) => {
 *     console.log("Selection changed:", selection);
 *   },
 * });
 *
 * return (
 *   <div>
 *     <Badge variant={isConnected ? "success" : "secondary"}>
 *       {isConnected ? "Connected" : "Disconnected"}
 *     </Badge>
 *     {!isConnected && <Button onClick={reconnect}>Reconnect</Button>}
 *   </div>
 * );
 * ```
 */
export function useModelSelectionSync(
  options: UseModelSelectionSyncOptions = {}
): UseModelSelectionSyncReturn {
  const {
    userId = "default",
    sessionId,
    enabled = true,
    onSelectionChanged,
    onProviderStatus,
    onModelValidation,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const hasLoggedErrorRef = useRef(false);

  /**
   * Calculate reconnection delay with exponential backoff
   */
  const getReconnectDelay = useCallback((attempts: number): number => {
    return Math.min(
      INITIAL_RECONNECT_DELAY * Math.pow(2, attempts),
      MAX_RECONNECT_DELAY
    );
  }, []);

  /**
   * Handle incoming WebSocket messages
   */
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage;

        switch (message.type) {
          case "SELECTION_CHANGED": {
            const data = message.data as SelectionChangedData;

            // Update TanStack Query cache
            queryClient.setQueryData<CurrentSelection>(
              queryKeys.unifiedProviders.selection(sessionId),
              {
                provider_id: data.provider_id,
                model_id: data.model_id,
                scope: data.scope,
                session_id: data.session_id,
                user_id: data.user_id,
              }
            );

            // Call user callback
            onSelectionChanged?.(data);
            break;
          }

          case "PROVIDER_STATUS": {
            const data = message.data as ProviderStatusData;

            // Invalidate provider queries to trigger refetch
            queryClient.invalidateQueries({
              queryKey: queryKeys.unifiedProviders.list(),
            });

            onProviderStatus?.(data);
            break;
          }

          case "MODEL_VALIDATION": {
            const data = message.data as ModelValidationData;
            onModelValidation?.(data);
            break;
          }

          case "PING": {
            // Respond to server ping with pong
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(
                JSON.stringify({ type: "PONG", timestamp: new Date().toISOString() })
              );
            }
            break;
          }

          case "PONG": {
            // Server acknowledged our ping
            break;
          }

          case "ERROR": {
            console.error("[useModelSelectionSync] Server error:", message.data);
            break;
          }

          default: {
            console.warn("[useModelSelectionSync] Unknown message type:", message.type);
          }
        }
      } catch (error) {
        console.error("[useModelSelectionSync] Failed to parse message:", error);
      }
    },
    [queryClient, sessionId, onSelectionChanged, onProviderStatus, onModelValidation]
  );

  /**
   * Start sending periodic pings to keep connection alive
   */
  const startPingInterval = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }

    pingIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({ type: "PING", timestamp: new Date().toISOString() })
        );
      }
    }, PING_INTERVAL);
  }, []);

  /**
   * Stop ping interval
   */
  const stopPingInterval = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  }, []);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(function connectImpl() {
    // Don't connect if disabled or already connected
    if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Construct WebSocket URL
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = process.env.NEXT_PUBLIC_API_URL?.replace(/^https?:\/\//, "") || "localhost:8001";
    const wsUrl = `${wsProtocol}//${wsHost}/ws/model-selection?user_id=${encodeURIComponent(userId)}`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setReconnectAttempts(0);
        hasLoggedErrorRef.current = false; // Reset error flag on successful connection
        startPingInterval();
        onConnect?.();
      };

      ws.onmessage = handleMessage;

      ws.onerror = (event) => {
        // WebSocket error events don't contain useful details - they're intentionally vague for security
        // Only log once to avoid console spam during development
        if (!hasLoggedErrorRef.current) {
          console.warn(
            "[useModelSelectionSync] WebSocket connection failed. Real-time sync is disabled. " +
            "This is normal if the backend WebSocket endpoint is not running."
          );
          hasLoggedErrorRef.current = true;
        }
        onError?.(event);
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        stopPingInterval();
        onDisconnect?.();

        // Attempt reconnection with exponential backoff (silently)
        if (enabled && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          const delay = getReconnectDelay(reconnectAttempts);
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts((prev) => prev + 1);
            connectImpl();
          }, delay);
        }
      };
    } catch (error) {
      console.error("[useModelSelectionSync] Failed to create WebSocket:", error);
    }
  }, [
    enabled,
    userId,
    reconnectAttempts,
    getReconnectDelay,
    startPingInterval,
    stopPingInterval,
    handleMessage,
    onConnect,
    onDisconnect,
    onError,
  ]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    stopPingInterval();

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      const ws = wsRef.current;
      // Remove event handlers to prevent callbacks after unmount
      ws.onopen = null;
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;

      // Only close if already open to avoid browser warning
      // "WebSocket is closed before the connection is established"
      if (ws.readyState === WebSocket.OPEN) {
        ws.close(1000, "Client disconnect");
      }
      // If CONNECTING, let it fail naturally - we've removed the handlers
      wsRef.current = null;
    }

    setIsConnected(false);
    setReconnectAttempts(0);
  }, [stopPingInterval]);

  /**
   * Manually trigger reconnection
   */
  const reconnect = useCallback(() => {
    disconnect();
    setReconnectAttempts(0);
    connect();
  }, [disconnect, connect]);

  // Connect on mount and when dependencies change
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, userId]); // Only reconnect when these change

  return {
    isConnected,
    reconnectAttempts,
    reconnect,
    disconnect,
  };
}
