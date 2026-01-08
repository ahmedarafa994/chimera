import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';

/**
 * WebSocket message type
 */
export interface SocketMessage<T = unknown> {
    type: string;
    data?: T;
    [key: string]: unknown;
}

/**
 * WebSocket configuration options
 */
interface WebSocketConfig<T = unknown> {
    url: string;
    onMessage?: (message: SocketMessage<T>) => void;
    onOpen?: () => void;
    onError?: (error: Event) => void;
    reconnectInterval?: number;
    maxRetries?: number;
}

/**
 * WebSocket hook return type
 */
interface UseSocketResult<T = unknown> {
    isConnected: boolean;
    error: Event | null;
    sendMessage: (message: SocketMessage<T>) => void;
}

/**
 * Custom hook for WebSocket connections with automatic reconnection
 *
 * @template T - The type of data expected in messages
 */
export function useSocket<T = unknown>({
    url,
    onMessage,
    onOpen,
    onError,
    reconnectInterval = 3000,
    maxRetries = 5
}: WebSocketConfig<T>): UseSocketResult<T> {
    const ws = useRef<WebSocket | null>(null);
    const reconnectAttempts = useRef(0);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<Event | null>(null);
    const connectRef = useRef<(() => void) | null>(null);

    const connect = useCallback(() => {
        try {
            ws.current = new WebSocket(url);

            ws.current.onopen = () => {
                setIsConnected(true);
                reconnectAttempts.current = 0;
                setError(null);
                onOpen?.();
            };

            ws.current.onmessage = (event: MessageEvent) => {
                try {
                    const data = JSON.parse(event.data) as SocketMessage<T>;

                    // Handle heartbeat
                    if (data.type === 'ping') {
                        ws.current?.send(JSON.stringify({ type: 'pong' }));
                        return;
                    }

                    onMessage?.(data);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            ws.current.onerror = (event: Event) => {
                setError(event);
                onError?.(event);
            };

            ws.current.onclose = () => {
                setIsConnected(false);
                if (reconnectAttempts.current < maxRetries) {
                    setTimeout(() => {
                        reconnectAttempts.current++;
                        // Use the ref to call connect recursively
                        connectRef.current?.();
                    }, reconnectInterval);
                } else {
                    toast.error("Connection lost. Please refresh to reconnect.");
                }
            };

        } catch (e) {
            console.error('WebSocket connection failed:', e);
        }
    }, [url, onMessage, onOpen, onError, reconnectInterval, maxRetries]);

    // Store connect in ref for recursive calls
    connectRef.current = connect;

    useEffect(() => {
        connect();
        return () => {
            ws.current?.close();
        };
    }, []);

    const sendMessage = useCallback((message: SocketMessage<T>) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(message));
        }
    }, []);

    return { isConnected, error, sendMessage };
}
