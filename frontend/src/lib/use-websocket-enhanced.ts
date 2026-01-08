"use client";

/**
 * Enhanced WebSocket Hook for Project Chimera Frontend
 * Adds heartbeat, reconnection, and integration with resilience patterns
 *
 * Mirrors backend's WebSocket pattern from main.py
 *
 * Usage:
 *   const { sendMessage, lastMessage, status } = useWebSocketEnhanced('/ws/enhance');
 */

import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { getCurrentApiUrl } from "./api-enhanced";
import { CircuitBreakerRegistry, type CircuitBreakerStats } from "./resilience";
import { NetworkError, ServiceUnavailableError } from "./errors";

// ============================================================================
// Types
// ============================================================================

export type WebSocketStatus = "connecting" | "connected" | "disconnected" | "reconnecting" | "error";
export type ConnectionQuality = "excellent" | "good" | "fair" | "poor" | "offline";

export interface WebSocketMessage<T = unknown> {
    type: string;
    data: T;
    timestamp: string;
    requestId?: string;
}

export interface UseWebSocketOptions {
    /** Auto-connect on mount (default: true) */
    autoConnect?: boolean;
    /** Auto-reconnect on disconnect (default: true) */
    autoReconnect?: boolean;
    /** Max reconnection attempts (default: 5) */
    maxReconnectAttempts?: number;
    /** Reconnection delay in ms (default: 1000) */
    reconnectDelay?: number;
    /** Heartbeat interval in ms (default: 30000) */
    heartbeatInterval?: number;
    /** Connection timeout in ms (default: 10000) */
    connectionTimeout?: number;
    /** Use circuit breaker (default: true) */
    useCircuitBreaker?: boolean;
    /** Circuit breaker name (default: "websocket") */
    circuitName?: string;
    /** Callback when connected */
    onConnect?: () => void;
    /** Callback when disconnected */
    onDisconnect?: (event: CloseEvent) => void;
    /** Callback when error occurs */
    onError?: (error: Event) => void;
    /** Callback when message received */
    onMessage?: <T>(message: WebSocketMessage<T>) => void;
}

export interface UseWebSocketReturn<TReceive = unknown, TSend = unknown> {
    /** Current connection status */
    status: WebSocketStatus;
    /** Last received message */
    lastMessage: WebSocketMessage<TReceive> | null;
    /** Send a message */
    sendMessage: (message: TSend) => void;
    /** Manually connect */
    connect: () => void;
    /** Manually disconnect */
    disconnect: () => void;
    /** Number of reconnection attempts */
    reconnectAttempts: number;
    /** Last error */
    lastError: Error | null;
    /** Is connected */
    isConnected: boolean;
    /** Circuit breaker stats */
    circuitStats: CircuitBreakerStats | null;
    /** Current latency in ms */
    latency: number | null;
    /** Latency history (last 10 measurements) */
    latencyHistory: number[];
    /** Connection quality assessment */
    quality: ConnectionQuality;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_OPTIONS: Required<Omit<UseWebSocketOptions, "onConnect" | "onDisconnect" | "onError" | "onMessage">> = {
    autoConnect: true,
    autoReconnect: true,
    maxReconnectAttempts: 5,
    reconnectDelay: 1000,
    heartbeatInterval: 30000,
    connectionTimeout: 10000,
    useCircuitBreaker: true,
    circuitName: "websocket",
};

// ============================================================================
// Hook Implementation
// ============================================================================

export function useWebSocketEnhanced<TReceive = unknown, TSend = unknown>(
    path: string,
    options: UseWebSocketOptions = {}
): UseWebSocketReturn<TReceive, TSend> {
    const opts = useMemo(() => ({ ...DEFAULT_OPTIONS, ...options }), [options]);

    // State
    const [status, setStatus] = useState<WebSocketStatus>("disconnected");
    const [lastMessage, setLastMessage] = useState<WebSocketMessage<TReceive> | null>(null);
    const [reconnectAttempts, setReconnectAttempts] = useState(0);
    const [lastError, setLastError] = useState<Error | null>(null);
    const [circuitStats, setCircuitStats] = useState<CircuitBreakerStats | null>(null);
    const [latency, setLatency] = useState<number | null>(null);
    const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
    const [pingStartTime, setPingStartTime] = useState<number | null>(null);

    // Refs
    const wsRef = useRef<WebSocket | null>(null);
    const heartbeatIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const connectionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const isIntentionalDisconnectRef = useRef(false);

    // Derived state
    const isConnected = status === "connected";

    // Get WebSocket URL
    const getWsUrl = useCallback(() => {
        const apiUrl = getCurrentApiUrl();
        const wsProtocol = apiUrl.startsWith("https") ? "wss" : "ws";
        const wsUrl = apiUrl.replace(/^https?/, wsProtocol);
        return `${wsUrl}${path.startsWith("/") ? path : `/${path}`}`;
    }, [path]);

    // Clear timeouts and intervals
    const clearTimers = useCallback(() => {
        if (heartbeatIntervalRef.current) {
            clearInterval(heartbeatIntervalRef.current);
            heartbeatIntervalRef.current = null;
        }
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (connectionTimeoutRef.current) {
            clearTimeout(connectionTimeoutRef.current);
            connectionTimeoutRef.current = null;
        }
    }, []);

    // Start heartbeat
    const startHeartbeat = useCallback(() => {
        if (heartbeatIntervalRef.current) {
            clearInterval(heartbeatIntervalRef.current);
        }

        heartbeatIntervalRef.current = setInterval(() => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                const now = Date.now();
                setPingStartTime(now);
                wsRef.current.send(JSON.stringify({
                    type: "ping",
                    timestamp: new Date(now).toISOString()
                }));
            }
        }, opts.heartbeatInterval);
    }, [opts.heartbeatInterval]);

    // Use ref to hold connect function to avoid circular dependency with reconnect logic
    const connectRef = useRef<() => void>(() => { });

    // Handle reconnection - uses connectRef to avoid circular dependency
    const scheduleReconnect = useCallback(() => {
        if (!opts.autoReconnect || isIntentionalDisconnectRef.current) {
            return;
        }

        if (reconnectAttempts >= opts.maxReconnectAttempts) {
            setStatus("error");
            setLastError(new ServiceUnavailableError("Max reconnection attempts reached"));
            return;
        }

        setStatus("reconnecting");

        const delay = opts.reconnectDelay * Math.pow(2, reconnectAttempts);
        const jitter = delay * 0.2 * (Math.random() * 2 - 1);
        const actualDelay = Math.min(delay + jitter, 30000);

        reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts((prev) => prev + 1);
            connectRef.current();
        }, actualDelay);
    }, [opts.autoReconnect, opts.maxReconnectAttempts, opts.reconnectDelay, reconnectAttempts]);

    // Connect function
    const connect = useCallback(() => {
        // Check circuit breaker
        if (opts.useCircuitBreaker) {
            const breaker = CircuitBreakerRegistry.get(opts.circuitName);
            if (!breaker.canExecute()) {
                setStatus("error");
                setLastError(new NetworkError("Circuit breaker is open"));
                setCircuitStats(breaker.getStats());
                return;
            }
        }

        // Close existing connection
        if (wsRef.current) {
            wsRef.current.close();
        }

        isIntentionalDisconnectRef.current = false;
        setStatus("connecting");
        setLastError(null);

        try {
            const url = getWsUrl();
            wsRef.current = new WebSocket(url);

            // Connection timeout
            connectionTimeoutRef.current = setTimeout(() => {
                if (wsRef.current?.readyState === WebSocket.CONNECTING) {
                    wsRef.current.close();
                    setStatus("error");
                    setLastError(new NetworkError("Connection timeout"));

                    if (opts.useCircuitBreaker) {
                        const breaker = CircuitBreakerRegistry.get(opts.circuitName);
                        breaker.execute(() => Promise.reject(new Error("timeout"))).catch(() => { });
                        setCircuitStats(breaker.getStats());
                    }

                    scheduleReconnect();
                }
            }, opts.connectionTimeout);

            // Event handlers
            wsRef.current.onopen = () => {
                clearTimers();
                setStatus("connected");
                setReconnectAttempts(0);
                setLastError(null);
                startHeartbeat();
                options.onConnect?.();

                // Report success to circuit breaker
                if (opts.useCircuitBreaker) {
                    setCircuitStats(CircuitBreakerRegistry.getStats(opts.circuitName));
                }
            };

            wsRef.current.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data) as WebSocketMessage<TReceive>;

                    // Handle pong response (heartbeat)
                    if (message.type === "pong") {
                        if (pingStartTime) {
                            const rtt = Date.now() - pingStartTime;
                            setLatency(rtt);
                            setLatencyHistory(prev => [rtt, ...prev].slice(0, 10));
                            setPingStartTime(null);
                        }
                        return;
                    }

                    setLastMessage(message);
                    options.onMessage?.(message);
                } catch (e) {
                    console.warn("[WebSocket] Failed to parse message:", e);
                }
            };

            wsRef.current.onerror = (event) => {
                setStatus("error");
                setLastError(new NetworkError("WebSocket error"));
                options.onError?.(event);

                // Report failure to circuit breaker
                if (opts.useCircuitBreaker) {
                    const breaker = CircuitBreakerRegistry.get(opts.circuitName);
                    breaker.execute(() => Promise.reject(new Error("ws_error"))).catch(() => { });
                    setCircuitStats(breaker.getStats());
                }
            };

            wsRef.current.onclose = (event) => {
                clearTimers();
                setStatus("disconnected");
                options.onDisconnect?.(event);

                // Only reconnect if not intentional
                if (!isIntentionalDisconnectRef.current && !event.wasClean) {
                    scheduleReconnect();
                }
            };
        } catch (e) {
            setStatus("error");
            setLastError(e instanceof Error ? e : new NetworkError("Failed to create WebSocket"));
            scheduleReconnect();
        }
    }, [
        getWsUrl,
        opts.useCircuitBreaker,
        opts.circuitName,
        opts.connectionTimeout,
        clearTimers,
        startHeartbeat,
        scheduleReconnect,
        options,
    ]);

    // Keep connectRef in sync with connect
    useEffect(() => {
        connectRef.current = connect;
    }, [connect]);

    // Disconnect function
    const disconnect = useCallback(() => {
        isIntentionalDisconnectRef.current = true;
        clearTimers();

        if (wsRef.current) {
            wsRef.current.close(1000, "User disconnect");
            wsRef.current = null;
        }

        setStatus("disconnected");
        setReconnectAttempts(0);
    }, [clearTimers]);

    // Send message function
    const sendMessage = useCallback((message: TSend) => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
            console.warn("[WebSocket] Cannot send message: not connected");
            return;
        }

        const payload = typeof message === "string" ? message : JSON.stringify(message);
        wsRef.current.send(payload);
    }, []);

    // Auto-connect on mount - schedule for next tick to avoid setState in effect
    useEffect(() => {
        if (opts.autoConnect) {
            const timeoutId = setTimeout(() => {
                connect();
            }, 0);
            return () => {
                clearTimeout(timeoutId);
                isIntentionalDisconnectRef.current = true;
                clearTimers();
                wsRef.current?.close();
            };
        }

        return () => {
            isIntentionalDisconnectRef.current = true;
            clearTimers();
            wsRef.current?.close();
        };
    }, [opts.autoConnect, connect, clearTimers]);

    // Update circuit stats periodically
    useEffect(() => {
        if (!opts.useCircuitBreaker) return;

        const unsubscribe = CircuitBreakerRegistry.addListener((name, stats) => {
            if (name === opts.circuitName) {
                setCircuitStats(stats);
            }
        });

        return unsubscribe;
    }, [opts.useCircuitBreaker, opts.circuitName]);

    // Assess connection quality
    const quality = useMemo((): ConnectionQuality => {
        if (!isConnected) return "offline";
        if (latency === null) return "good"; // Initial state
        if (latency < 100) return "excellent";
        if (latency < 300) return "good";
        if (latency < 600) return "fair";
        return "poor";
    }, [isConnected, latency]);

    return {
        status,
        lastMessage,
        sendMessage,
        connect,
        disconnect,
        reconnectAttempts,
        lastError,
        isConnected,
        circuitStats,
        latency,
        latencyHistory,
        quality,
    };
}

// ============================================================================
// Specialized Hooks
// ============================================================================

/**
 * Hook for the enhance WebSocket endpoint
 */
export interface EnhanceMessage {
    type: "chunk" | "complete" | "error" | "heartbeat";
    content?: string;
    metadata?: Record<string, unknown>;
    error?: string;
}

export interface EnhanceRequest {
    prompt: string;
    technique?: string;
    potency?: number;
}

export function useEnhanceWebSocket(options?: UseWebSocketOptions) {
    return useWebSocketEnhanced<EnhanceMessage, EnhanceRequest>("/ws/enhance", {
        circuitName: "ws-enhance",
        ...options,
    });
}

/**
 * Hook for the fuzz WebSocket endpoint
 */
export interface FuzzMessage {
    type: "progress" | "result" | "complete" | "error";
    query_count?: number;
    jailbreak_count?: number;
    result?: {
        question: string;
        success: boolean;
        response?: string;
    };
    error?: string;
}

export interface FuzzStartRequest {
    target_model: string;
    questions: string[];
    seeds?: string[];
    max_queries?: number;
}

export function useFuzzWebSocket(options?: UseWebSocketOptions) {
    return useWebSocketEnhanced<FuzzMessage, FuzzStartRequest>("/ws/fuzz", {
        circuitName: "ws-fuzz",
        heartbeatInterval: 60000, // Longer for long-running fuzz
        connectionTimeout: 15000,
        ...options,
    });
}
