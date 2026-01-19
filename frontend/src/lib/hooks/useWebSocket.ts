'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { WSMessage } from '@/types/deepteam';

export interface WebSocketOptions {
  reconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
}

export interface WebSocketHook {
  ws: WebSocket | null;
  isConnected: boolean;
  lastMessage: WSMessage | null;
  sendMessage: (message: WSMessage) => void;
  close: () => void;
  reconnect: () => void;
}

export function useWebSocket(url: string, options: WebSocketOptions = {}): WebSocketHook {
  const {
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    heartbeatInterval = 30000,
  } = options;

  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);

  const reconnectCountRef = useRef(0);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    try {
      const websocket = new WebSocket(url);

      websocket.onopen = () => {
        setIsConnected(true);
        reconnectCountRef.current = 0;

        // Start heartbeat
        heartbeatIntervalRef.current = setInterval(() => {
          if (websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({
              event: 'ping',
              data: {},
              timestamp: new Date().toISOString(),
            }));
          }
        }, heartbeatInterval);
      };

      websocket.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          setLastMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      websocket.onclose = () => {
        setIsConnected(false);
        setWs(null);

        // Clear heartbeat
        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
        }

        // Attempt reconnection
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      setWs(websocket);
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }, [url, reconnectAttempts, reconnectInterval, heartbeatInterval]);

  const sendMessage = useCallback((message: WSMessage) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, [ws]);

  const close = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (ws) {
      ws.close();
    }
  }, [ws]);

  const reconnect = useCallback(() => {
    close();
    reconnectCountRef.current = 0;
    connect();
  }, [close, connect]);

  useEffect(() => {
    connect();

    return () => {
      close();
    };
  }, [connect, close]);

  return {
    ws,
    isConnected,
    lastMessage,
    sendMessage,
    close,
    reconnect,
  };
}