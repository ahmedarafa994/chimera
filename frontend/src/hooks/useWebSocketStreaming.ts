/**
 * WebSocket Streaming Hook for Real-time Text Generation
 *
 * This hook provides WebSocket-based streaming as an alternative to SSE.
 * It integrates with the unified provider/model selection system and
 * uses the existing WebSocket manager for connection handling.
 *
 * Use cases:
 * - Bidirectional communication (send/receive during stream)
 * - Lower latency requirements
 * - Environments where SSE is not optimal
 *
 * @module hooks/useWebSocketStreaming
 */

import { useCallback, useRef, useState, useEffect } from "react";
import { useUnifiedProviderSelection } from "./useUnifiedProviderSelection";
import {
  wsManager,
  type ConnectionState,
  type WebSocketConfig,
} from "@/lib/api/websocket-manager";
import type { StreamChunk, StreamMetadata } from "./useStreamingGeneration";

// =============================================================================
// Types
// =============================================================================

/**
 * Options for WebSocket streaming
 */
export interface WebSocketStreamingOptions {
  /** WebSocket URL (default: ws://localhost:8000/api/v1/ws/stream) */
  wsUrl?: string;
  /** Session ID for session-scoped selection */
  sessionId?: string;
  /** Explicit provider override */
  provider?: string;
  /** Explicit model override */
  model?: string;
  /** Auto-connect on mount */
  autoConnect?: boolean;
  /** Reconnect on disconnect */
  reconnect?: boolean;
  /** Maximum reconnect attempts */
  maxReconnectAttempts?: number;
  /** Heartbeat interval in ms */
  heartbeatInterval?: number;
}

/**
 * WebSocket stream chunk (extends SSE chunk with WebSocket-specific fields)
 */
export interface WebSocketStreamChunk extends StreamChunk {
  /** Message ID from WebSocket */
  messageId?: string;
  /** Sequence number for ordering */
  sequence?: number;
}

/**
 * WebSocket stream result
 */
export interface WebSocketStreamResult {
  /** Complete generated text */
  fullText: string;
  /** Stream identifier */
  streamId: string;
  /** Provider used */
  provider: string;
  /** Model used */
  model: string;
  /** Total chunks received */
  totalChunks: number;
  /** Total tokens generated */
  totalTokens?: number;
  /** Stream duration in milliseconds */
  durationMs: number;
  /** Reason for completion */
  finishReason?: string;
  /** Final token usage */
  usage?: StreamChunk["usage"];
  /** Stream metadata */
  metadata?: StreamMetadata;
}

/**
 * WebSocket stream state
 */
export type WebSocketStreamState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "streaming"
  | "completed"
  | "error";

/**
 * Return type for useWebSocketStreaming hook
 */
export interface UseWebSocketStreamingReturn {
  // Connection State
  /** Current connection state */
  connectionState: ConnectionState;
  /** Whether WebSocket is connected */
  isConnected: boolean;

  // Stream State
  /** Current stream state */
  state: WebSocketStreamState;
  /** Whether currently streaming */
  isStreaming: boolean;
  /** Accumulated text so far */
  currentText: string;
  /** Current stream ID */
  streamId: string | null;
  /** Last error */
  error: Error | null;
  /** Current chunk index */
  chunkIndex: number;
  /** Stream metadata */
  streamMetadata: StreamMetadata | null;

  // Connection Actions
  /** Connect to WebSocket server */
  connect: () => Promise<void>;
  /** Disconnect from WebSocket server */
  disconnect: () => void;
  /** Reconnect to WebSocket server */
  reconnect: () => void;

  // Streaming Actions
  /** Start streaming text generation */
  streamGenerate: (
    prompt: string,
    options?: {
      systemInstruction?: string;
      temperature?: number;
      maxTokens?: number;
      provider?: string;
      model?: string;
    },
    onChunk?: (chunk: WebSocketStreamChunk) => void,
    onComplete?: (result: WebSocketStreamResult) => void,
    onError?: (error: Error) => void
  ) => Promise<void>;

  /** Start streaming chat completion */
  streamChat: (
    messages: Array<{ role: string; content: string }>,
    options?: {
      temperature?: number;
      maxTokens?: number;
      provider?: string;
      model?: string;
    },
    onChunk?: (chunk: WebSocketStreamChunk) => void,
    onComplete?: (result: WebSocketStreamResult) => void,
    onError?: (error: Error) => void
  ) => Promise<void>;

  /** Abort current stream */
  abortStream: () => void;

  /** Send arbitrary message through WebSocket */
  sendMessage: (message: unknown) => void;

  /** Reset stream state */
  reset: () => void;
}

// =============================================================================
// Constants
// =============================================================================

const DEFAULT_WS_URL = "/api/v1/ws/stream";

// =============================================================================
// Hook Implementation
// =============================================================================

/**
 * Hook for WebSocket-based streaming text generation.
 *
 * Provides real-time bidirectional streaming with the backend,
 * integrated with the Unified Provider/Model Selection System.
 *
 * @param options - WebSocket streaming configuration
 * @returns WebSocket streaming state and actions
 *
 * @example
 * ```tsx
 * function ChatComponent() {
 *   const {
 *     connect,
 *     streamChat,
 *     isConnected,
 *     isStreaming,
 *     currentText,
 *   } = useWebSocketStreaming({ autoConnect: true });
 *
 *   const handleSend = async () => {
 *     await streamChat(
 *       [{ role: "user", content: "Hello!" }],
 *       {},
 *       (chunk) => console.log("Chunk:", chunk.text),
 *       (result) => console.log("Complete:", result.fullText)
 *     );
 *   };
 *
 *   return (
 *     <div>
 *       <div>Status: {isConnected ? "Connected" : "Disconnected"}</div>
 *       <button onClick={handleSend} disabled={!isConnected || isStreaming}>
 *         Send
 *       </button>
 *       <div>{currentText}</div>
 *     </div>
 *   );
 * }
 * ```
 */
export function useWebSocketStreaming(
  options: WebSocketStreamingOptions = {}
): UseWebSocketStreamingReturn {
  const {
    wsUrl = DEFAULT_WS_URL,
    sessionId,
    provider: defaultProvider,
    model: defaultModel,
    autoConnect = false,
    reconnect = true,
    maxReconnectAttempts = 5,
    heartbeatInterval = 30000,
  } = options;

  // Get current selection from unified provider system
  const { selectedProvider, selectedModel } = useUnifiedProviderSelection();

  // Connection state
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("disconnected");

  // Stream state
  const [state, setState] = useState<WebSocketStreamState>("disconnected");
  const [currentText, setCurrentText] = useState<string>("");
  const [streamId, setStreamId] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [chunkIndex, setChunkIndex] = useState<number>(0);
  const [streamMetadata, setStreamMetadata] = useState<StreamMetadata | null>(
    null
  );

  // Refs
  const currentStreamIdRef = useRef<string | null>(null);
  const onChunkRef = useRef<((chunk: WebSocketStreamChunk) => void) | null>(
    null
  );
  const onCompleteRef = useRef<
    ((result: WebSocketStreamResult) => void) | null
  >(null);
  const onErrorRef = useRef<((error: Error) => void) | null>(null);
  const startTimeRef = useRef<number>(0);
  const chunksRef = useRef<WebSocketStreamChunk[]>([]);
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const stateUnsubscribeRef = useRef<(() => void) | null>(null);

  // =============================================================================
  // Message Handler
  // =============================================================================

  const handleMessage = useCallback(
    (message: { type: string; payload?: unknown; request_id?: string }) => {
      // Ignore messages not for our stream
      if (
        currentStreamIdRef.current &&
        message.request_id !== currentStreamIdRef.current
      ) {
        return;
      }

      switch (message.type) {
        case "stream_start": {
          const payload = message.payload as {
            stream_id: string;
            provider?: string;
            model?: string;
            session_id?: string;
          };
          currentStreamIdRef.current = payload.stream_id;
          setStreamId(payload.stream_id);
          setState("streaming");
          setStreamMetadata({
            provider: payload.provider || null,
            model: payload.model || null,
            sessionId: payload.session_id || null,
            streamId: payload.stream_id,
            startedAt: new Date().toISOString(),
            resolutionSource: null,
            resolutionPriority: null,
          });
          break;
        }

        case "stream_chunk":
        case "chunk":
        case "data": {
          const payload = message.payload as WebSocketStreamChunk;
          if (!payload?.text) return;

          setCurrentText((prev) => prev + payload.text);
          setChunkIndex((prev) => prev + 1);
          chunksRef.current.push(payload);
          onChunkRef.current?.(payload);
          break;
        }

        case "stream_end":
        case "done": {
          const payload = message.payload as {
            stream_id?: string;
            finish_reason?: string;
            usage?: StreamChunk["usage"];
          };

          setState("completed");

          const fullText = chunksRef.current.map((c) => c.text).join("");
          const lastChunk =
            chunksRef.current[chunksRef.current.length - 1] || null;

          const result: WebSocketStreamResult = {
            fullText,
            streamId: currentStreamIdRef.current || payload.stream_id || "",
            provider:
              streamMetadata?.provider ||
              lastChunk?.provider ||
              selectedProvider ||
              "",
            model:
              streamMetadata?.model ||
              lastChunk?.model ||
              selectedModel ||
              "",
            totalChunks: chunksRef.current.length,
            totalTokens: payload.usage?.total_tokens,
            durationMs: Date.now() - startTimeRef.current,
            finishReason: payload.finish_reason || lastChunk?.finish_reason,
            usage: payload.usage || lastChunk?.usage,
            metadata: streamMetadata || undefined,
          };

          onCompleteRef.current?.(result);
          currentStreamIdRef.current = null;
          break;
        }

        case "stream_error":
        case "error": {
          const payload = message.payload as {
            error?: string;
            message?: string;
            code?: string;
          };
          const errorMessage =
            payload.error || payload.message || "Stream error";
          const streamError = new Error(errorMessage);
          setState("error");
          setError(streamError);
          onErrorRef.current?.(streamError);
          currentStreamIdRef.current = null;
          break;
        }
      }
    },
    [selectedProvider, selectedModel, streamMetadata]
  );

  // =============================================================================
  // Connection Actions
  // =============================================================================

  const connect = useCallback(async () => {
    // Build WebSocket URL
    const baseUrl = typeof window !== "undefined" ? window.location.origin : "";
    const wsProtocol = baseUrl.startsWith("https") ? "wss" : "ws";
    const wsBase = baseUrl.replace(/^https?/, wsProtocol);

    const url = new URL(wsUrl.startsWith("/") ? `${wsBase}${wsUrl}` : wsUrl);

    if (sessionId) {
      url.searchParams.set("session_id", sessionId);
    }

    const config: WebSocketConfig = {
      url: url.toString(),
      reconnect,
      maxReconnectAttempts,
      heartbeatInterval,
    };

    // Subscribe to state changes
    if (stateUnsubscribeRef.current) {
      stateUnsubscribeRef.current();
    }
    stateUnsubscribeRef.current = wsManager.onStateChange((newState) => {
      setConnectionState(newState);
      if (newState === "connected" && state === "disconnected") {
        setState("connected");
      } else if (newState === "disconnected") {
        setState("disconnected");
      }
    });

    // Subscribe to messages
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
    }
    unsubscribeRef.current = wsManager.onMessage(handleMessage);

    try {
      await wsManager.connect(config);
      setState("connected");
    } catch (err) {
      const connectError =
        err instanceof Error ? err : new Error("Connection failed");
      setError(connectError);
      setState("error");
      throw connectError;
    }
  }, [
    wsUrl,
    sessionId,
    reconnect,
    maxReconnectAttempts,
    heartbeatInterval,
    handleMessage,
    state,
  ]);

  const disconnect = useCallback(() => {
    wsManager.disconnect();
    setState("disconnected");
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
      unsubscribeRef.current = null;
    }
    if (stateUnsubscribeRef.current) {
      stateUnsubscribeRef.current();
      stateUnsubscribeRef.current = null;
    }
  }, []);

  const reconnectAction = useCallback(() => {
    disconnect();
    connect().catch((err) => {
      console.error("Reconnect failed:", err);
    });
  }, [disconnect, connect]);

  // =============================================================================
  // Streaming Actions
  // =============================================================================

  const streamGenerate = useCallback(
    async (
      prompt: string,
      streamOptions?: {
        systemInstruction?: string;
        temperature?: number;
        maxTokens?: number;
        provider?: string;
        model?: string;
      },
      onChunk?: (chunk: WebSocketStreamChunk) => void,
      onComplete?: (result: WebSocketStreamResult) => void,
      onError?: (error: Error) => void
    ) => {
      // Ensure connected
      if (!wsManager.isConnected()) {
        await connect();
      }

      // Reset stream state
      setCurrentText("");
      setStreamId(null);
      setError(null);
      setChunkIndex(0);
      setStreamMetadata(null);
      chunksRef.current = [];
      startTimeRef.current = Date.now();

      // Store callbacks
      onChunkRef.current = onChunk || null;
      onCompleteRef.current = onComplete || null;
      onErrorRef.current = onError || null;

      // Determine provider/model
      const provider =
        streamOptions?.provider ||
        defaultProvider ||
        selectedProvider ||
        undefined;
      const model =
        streamOptions?.model || defaultModel || selectedModel || undefined;

      // Generate a request ID
      const requestId = `gen_${Date.now()}_${Math.random().toString(36).slice(2)}`;
      currentStreamIdRef.current = requestId;

      // Send generate request
      wsManager.sendData(
        {
          action: "generate",
          prompt,
          system_instruction: streamOptions?.systemInstruction,
          temperature: streamOptions?.temperature ?? 0.7,
          max_tokens: streamOptions?.maxTokens,
          provider,
          model,
          session_id: sessionId,
        },
        requestId
      );

      setState("streaming");
    },
    [
      connect,
      defaultProvider,
      defaultModel,
      selectedProvider,
      selectedModel,
      sessionId,
    ]
  );

  const streamChat = useCallback(
    async (
      messages: Array<{ role: string; content: string }>,
      streamOptions?: {
        temperature?: number;
        maxTokens?: number;
        provider?: string;
        model?: string;
      },
      onChunk?: (chunk: WebSocketStreamChunk) => void,
      onComplete?: (result: WebSocketStreamResult) => void,
      onError?: (error: Error) => void
    ) => {
      // Ensure connected
      if (!wsManager.isConnected()) {
        await connect();
      }

      // Reset stream state
      setCurrentText("");
      setStreamId(null);
      setError(null);
      setChunkIndex(0);
      setStreamMetadata(null);
      chunksRef.current = [];
      startTimeRef.current = Date.now();

      // Store callbacks
      onChunkRef.current = onChunk || null;
      onCompleteRef.current = onComplete || null;
      onErrorRef.current = onError || null;

      // Determine provider/model
      const provider =
        streamOptions?.provider ||
        defaultProvider ||
        selectedProvider ||
        undefined;
      const model =
        streamOptions?.model || defaultModel || selectedModel || undefined;

      // Generate a request ID
      const requestId = `chat_${Date.now()}_${Math.random().toString(36).slice(2)}`;
      currentStreamIdRef.current = requestId;

      // Send chat request
      wsManager.sendData(
        {
          action: "chat",
          messages,
          temperature: streamOptions?.temperature ?? 0.7,
          max_tokens: streamOptions?.maxTokens,
          provider,
          model,
          session_id: sessionId,
        },
        requestId
      );

      setState("streaming");
    },
    [
      connect,
      defaultProvider,
      defaultModel,
      selectedProvider,
      selectedModel,
      sessionId,
    ]
  );

  const abortStream = useCallback(() => {
    if (currentStreamIdRef.current && wsManager.isConnected()) {
      wsManager.sendData(
        {
          action: "abort",
          stream_id: currentStreamIdRef.current,
        },
        currentStreamIdRef.current
      );
    }
    currentStreamIdRef.current = null;
    setState("connected");
  }, []);

  const sendMessage = useCallback((message: unknown) => {
    wsManager.send({
      type: "data",
      payload: message,
      timestamp: new Date().toISOString(),
    });
  }, []);

  const reset = useCallback(() => {
    abortStream();
    setCurrentText("");
    setStreamId(null);
    setError(null);
    setChunkIndex(0);
    setStreamMetadata(null);
    chunksRef.current = [];
    setState(wsManager.isConnected() ? "connected" : "disconnected");
  }, [abortStream]);

  // =============================================================================
  // Auto-connect on mount
  // =============================================================================

  useEffect(() => {
    if (autoConnect) {
      connect().catch((err) => {
        console.error("Auto-connect failed:", err);
      });
    }

    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
      if (stateUnsubscribeRef.current) {
        stateUnsubscribeRef.current();
      }
    };
  }, [autoConnect, connect]);

  // =============================================================================
  // Return
  // =============================================================================

  return {
    // Connection State
    connectionState,
    isConnected: connectionState === "connected",

    // Stream State
    state,
    isStreaming: state === "streaming",
    currentText,
    streamId,
    error,
    chunkIndex,
    streamMetadata,

    // Connection Actions
    connect,
    disconnect,
    reconnect: reconnectAction,

    // Streaming Actions
    streamGenerate,
    streamChat,
    abortStream,
    sendMessage,
    reset,
  };
}

// =============================================================================
// Exports
// =============================================================================

export default useWebSocketStreaming;
