"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type WebSocketStatus = "connecting" | "connected" | "disconnected" | "error";

export interface WebSocketMessage {
  status: "processing" | "complete" | "error" | "progress" | "log";
  message?: string;
  enhanced_prompt?: string;
  progress?: number;
  data?: Record<string, unknown>;
}

export interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export function useWebSocket(endpoint: string, options: UseWebSocketOptions = {}) {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    autoConnect = false,
    reconnectAttempts = 3,
    reconnectInterval = 3000,
  } = options;

  const [status, setStatus] = useState<WebSocketStatus>("disconnected");
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const getWebSocketUrl = useCallback(() => {
    // Use window location to construct WebSocket URL
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    return `${protocol}//${host}/api/v1${endpoint}`;
  }, [endpoint]);

  // Use ref to hold connect function to avoid circular dependency with reconnect logic
  const connectRef = useRef<() => void>(() => {});

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setStatus("connecting");
    const url = getWebSocketUrl();

    try {
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        setStatus("connected");
        reconnectCountRef.current = 0;
        onOpen?.();
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch {
          console.error("Failed to parse WebSocket message");
        }
      };

      wsRef.current.onclose = () => {
        setStatus("disconnected");
        onClose?.();

        // Attempt to reconnect using ref to avoid circular dependency
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            connectRef.current();
          }, reconnectInterval);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        setStatus("error");
        onError?.(error);
      };
    } catch {
      console.error("Failed to create WebSocket connection");
      setStatus("error");
    }
  }, [getWebSocketUrl, onMessage, onOpen, onClose, onError, reconnectAttempts, reconnectInterval]);

  // Keep connectRef in sync with connect
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    reconnectCountRef.current = reconnectAttempts; // Prevent reconnection
    wsRef.current?.close();
    setStatus("disconnected");
  }, [reconnectAttempts]);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
      return true;
    }
    return false;
  }, []);

  const sendText = useCallback((text: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(text);
      return true;
    }
    return false;
  }, []);

  // Auto-connect if enabled - schedule for next tick to avoid setState in effect
  useEffect(() => {
    if (autoConnect) {
      const timeoutId = setTimeout(() => {
        connect();
      }, 0);
      return () => {
        clearTimeout(timeoutId);
        disconnect();
      };
    }
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    status,
    lastMessage,
    connect,
    disconnect,
    send,
    sendText,
    isConnected: status === "connected",
    isConnecting: status === "connecting",
  };
}

/**
 * Hook for real-time prompt enhancement via WebSocket
 */
export function useEnhanceWebSocket(options: Omit<UseWebSocketOptions, "onMessage"> = {}) {
  const [enhancedPrompt, setEnhancedPrompt] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.status) {
      case "processing":
        setIsProcessing(true);
        setError(null);
        break;
      case "complete":
        setIsProcessing(false);
        setEnhancedPrompt(message.enhanced_prompt || null);
        break;
      case "error":
        setIsProcessing(false);
        setError(message.message || "An error occurred");
        break;
    }
  }, []);

  const ws = useWebSocket("/ws/enhance", {
    ...options,
    onMessage: handleMessage,
  });

  const enhance = useCallback(
    (prompt: string, type: "standard" | "jailbreak" = "standard", potency?: number) => {
      setEnhancedPrompt(null);
      setError(null);
      setIsProcessing(true);

      const success = ws.send({
        prompt,
        type,
        potency: potency || 7,
      });

      if (!success) {
        setIsProcessing(false);
        setError("WebSocket is not connected");
      }
    },
    [ws]
  );

  return {
    ...ws,
    enhance,
    enhancedPrompt,
    isProcessing,
    error,
  };
}

/**
 * Hook for real-time fuzzing progress via WebSocket
 */
export function useFuzzWebSocket(options: Omit<UseWebSocketOptions, "onMessage"> = {}) {
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<Record<string, unknown>[]>([]);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.status) {
      case "progress":
        if (message.progress !== undefined) {
          setProgress(message.progress);
        }
        break;
      case "log":
        if (message.message) {
          setLogs((prev) => [...prev, message.message!]);
        }
        break;
      case "complete":
        setProgress(100);
        if (message.data) {
          setResults((prev) => [...prev, message.data!]);
        }
        break;
    }
  }, []);

  const ws = useWebSocket("/ws/fuzz", {
    ...options,
    onMessage: handleMessage,
  });

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setProgress(0);
  }, []);

  return {
    ...ws,
    progress,
    logs,
    results,
    clearLogs,
    clearResults,
  };
}

export default useWebSocket;
