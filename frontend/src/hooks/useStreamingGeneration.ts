/**
 * Streaming Generation Hook for Unified Provider/Model Selection System
 *
 * This hook provides streaming text generation that integrates with the
 * unified provider/model selection system. It handles:
 * - Server-Sent Events (SSE) streaming
 * - Provider/model selection consistency
 * - Error handling and cancellation
 * - Stream state management
 *
 * @module hooks/useStreamingGeneration
 */

import { useCallback, useRef, useState } from "react";
import { useUnifiedProviderSelection } from "./useUnifiedProviderSelection";

// =============================================================================
// Types
// =============================================================================

/**
 * Stream chunk received from the server
 */
export interface StreamChunk {
  /** Generated text content */
  text: string;
  /** Index of this chunk in the stream */
  chunk_index: number;
  /** Unique stream identifier */
  stream_id: string;
  /** Provider used for generation */
  provider: string;
  /** Model used for generation */
  model: string;
  /** Whether this is the final chunk */
  is_final: boolean;
  /** Reason for stream completion */
  finish_reason?: string;
  /** Token count for this chunk */
  token_count?: number;
  /** ISO timestamp */
  timestamp: string;
  /** Token usage information */
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Stream result returned after completion
 */
export interface StreamResult {
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
  /** Metadata extracted from response headers */
  metadata?: StreamMetadata;
}

/**
 * Stream metadata extracted from response headers
 */
export interface StreamMetadata {
  /** Resolved provider from X-Stream-Provider header */
  provider: string | null;
  /** Resolved model from X-Stream-Model header */
  model: string | null;
  /** Session ID from X-Stream-Session-Id header */
  sessionId: string | null;
  /** Stream ID from X-Stream-Id header */
  streamId: string | null;
  /** Stream start time from X-Stream-Started-At header */
  startedAt: string | null;
  /** Resolution source from X-Stream-Resolution-Source header */
  resolutionSource: string | null;
  /** Resolution priority from X-Stream-Resolution-Priority header */
  resolutionPriority: number | null;
}

/**
 * Stream error object
 */
export interface StreamError {
  /** Error message */
  message: string;
  /** Error code */
  code?: string;
  /** Stream ID if available */
  streamId?: string;
}

/**
 * Options for streaming generation
 */
export interface StreamingOptions {
  /** Base URL for the API (default: /api/v1) */
  baseUrl?: string;
  /** Session ID for session-scoped selection */
  sessionId?: string;
  /** Explicit provider override */
  provider?: string;
  /** Explicit model override */
  model?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Whether to use JSON Lines instead of SSE */
  useJsonl?: boolean;
}

/**
 * Generation request parameters
 */
export interface GenerateParams {
  /** The prompt text */
  prompt: string;
  /** Optional system instruction */
  systemInstruction?: string;
  /** Temperature for generation (0-2) */
  temperature?: number;
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Top-p sampling */
  topP?: number;
  /** Top-k sampling */
  topK?: number;
  /** Stop sequences */
  stopSequences?: string[];
  /** Explicit provider override for this request */
  provider?: string;
  /** Explicit model override for this request */
  model?: string;
}

/**
 * Chat message format
 */
export interface ChatMessage {
  /** Message role */
  role: "system" | "user" | "assistant";
  /** Message content */
  content: string;
}

/**
 * Chat request parameters
 */
export interface ChatParams {
  /** List of chat messages */
  messages: ChatMessage[];
  /** Temperature for generation (0-2) */
  temperature?: number;
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Top-p sampling */
  topP?: number;
  /** Top-k sampling */
  topK?: number;
  /** Stop sequences */
  stopSequences?: string[];
  /** Explicit provider override for this request */
  provider?: string;
  /** Explicit model override for this request */
  model?: string;
}

/**
 * Stream state
 */
export type StreamState = "idle" | "streaming" | "completed" | "error" | "cancelled";

/**
 * Return type for useStreamingGeneration hook
 */
export interface UseStreamingGenerationReturn {
  // State
  /** Current stream state */
  state: StreamState;
  /** Whether currently streaming */
  isStreaming: boolean;
  /** Accumulated text so far */
  currentText: string;
  /** Current stream ID */
  streamId: string | null;
  /** Last error */
  error: StreamError | null;
  /** Current chunk index */
  chunkIndex: number;
  /** Stream metadata from response headers */
  streamMetadata: StreamMetadata | null;

  // Actions
  /** Start streaming text generation */
  streamGenerate: (
    params: GenerateParams,
    onChunk?: (chunk: StreamChunk) => void,
    onComplete?: (result: StreamResult) => void,
    onError?: (error: StreamError) => void
  ) => Promise<void>;

  /** Start streaming chat completion */
  streamChat: (
    params: ChatParams,
    onChunk?: (chunk: StreamChunk) => void,
    onComplete?: (result: StreamResult) => void,
    onError?: (error: StreamError) => void
  ) => Promise<void>;

  /** Abort the current stream */
  abortStream: () => void;

  /** Reset state to idle */
  reset: () => void;
}

// =============================================================================
// Hook Implementation
// =============================================================================

/**
 * Hook for streaming text generation with unified provider/model selection.
 *
 * Integrates with the Unified Provider/Model Selection System to use
 * the currently selected provider/model while allowing explicit overrides.
 *
 * @param options - Streaming configuration options
 * @returns Streaming state and actions
 *
 * @example
 * ```tsx
 * function GenerationComponent() {
 *   const { streamGenerate, isStreaming, currentText, abortStream } =
 *     useStreamingGeneration();
 *
 *   const handleGenerate = async () => {
 *     await streamGenerate(
 *       { prompt: "Explain quantum computing" },
 *       (chunk) => console.log("Chunk:", chunk.text),
 *       (result) => console.log("Complete:", result.fullText),
 *       (error) => console.error("Error:", error.message)
 *     );
 *   };
 *
 *   return (
 *     <div>
 *       <button onClick={handleGenerate} disabled={isStreaming}>
 *         Generate
 *       </button>
 *       <button onClick={abortStream} disabled={!isStreaming}>
 *         Cancel
 *       </button>
 *       <div>{currentText}</div>
 *     </div>
 *   );
 * }
 * ```
 */
export function useStreamingGeneration(
  options: StreamingOptions = {}
): UseStreamingGenerationReturn {
  const {
    baseUrl = "/api/v1",
    sessionId,
    provider: defaultProvider,
    model: defaultModel,
    timeout = 300000, // 5 minutes
    useJsonl = false,
  } = options;

  // Get current selection from unified provider system
  const { selectedProvider, selectedModel } = useUnifiedProviderSelection();

  // State
  const [state, setState] = useState<StreamState>("idle");
  const [currentText, setCurrentText] = useState<string>("");
  const [streamId, setStreamId] = useState<string | null>(null);
  const [error, setError] = useState<StreamError | null>(null);
  const [chunkIndex, setChunkIndex] = useState<number>(0);
  const [streamMetadata, setStreamMetadata] = useState<StreamMetadata | null>(null);

  // Refs for cleanup
  const abortControllerRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);

  /**
   * Parse SSE event data
   */
  const parseSSEEvent = useCallback((eventText: string): {
    eventType: string;
    data: StreamChunk | { error: string; stream_id?: string } | null;
  } | null => {
    const lines = eventText.split("\n");
    let eventType = "chunk";
    let dataLine = "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLine = line.slice(5).trim();
      }
    }

    if (!dataLine) return null;

    try {
      const data = JSON.parse(dataLine);
      return { eventType, data };
    } catch {
      return null;
    }
  }, []);

  /**
   * Parse JSON Lines data
   */
  const parseJsonlLine = useCallback((line: string): StreamChunk | null => {
    if (!line.trim()) return null;
    try {
      return JSON.parse(line);
    } catch {
      return null;
    }
  }, []);

  /**
   * Extract stream metadata from response headers
   */
  const extractMetadataFromHeaders = useCallback((headers: Headers): StreamMetadata => {
    const priorityStr = headers.get("X-Stream-Resolution-Priority");
    return {
      provider: headers.get("X-Stream-Provider"),
      model: headers.get("X-Stream-Model"),
      sessionId: headers.get("X-Stream-Session-Id"),
      streamId: headers.get("X-Stream-Id"),
      startedAt: headers.get("X-Stream-Started-At"),
      resolutionSource: headers.get("X-Stream-Resolution-Source"),
      resolutionPriority: priorityStr ? parseInt(priorityStr, 10) : null,
    };
  }, []);

  /**
   * Core streaming function
   */
  const doStream = useCallback(
    async (
      endpoint: string,
      body: Record<string, unknown>,
      onChunk?: (chunk: StreamChunk) => void,
      onComplete?: (result: StreamResult) => void,
      onError?: (error: StreamError) => void
    ) => {
      // Reset state
      setState("streaming");
      setCurrentText("");
      setStreamId(null);
      setError(null);
      setChunkIndex(0);
      setStreamMetadata(null);

      // Create abort controller
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      // Set timeout
      const timeoutId = setTimeout(() => {
        abortController.abort();
      }, timeout);

      // Tracking variables
      let fullText = "";
      let currentStreamId = "";
      let totalChunks = 0;
      let lastChunk: StreamChunk | null = null;
      const startTime = Date.now();
      let extractedMetadata: StreamMetadata | null = null;

      try {
        // Build headers
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
          Accept: useJsonl ? "application/x-ndjson" : "text/event-stream",
        };

        if (sessionId) {
          headers["X-Session-Id"] = sessionId;
        }

        // Make request
        const response = await fetch(`${baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
          signal: abortController.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.detail || `HTTP error: ${response.status}`
          );
        }

        if (!response.body) {
          throw new Error("No response body");
        }

        // Extract metadata from response headers
        extractedMetadata = extractMetadataFromHeaders(response.headers);
        setStreamMetadata(extractedMetadata);

        // If we got a stream ID from headers, use it
        if (extractedMetadata.streamId) {
          currentStreamId = extractedMetadata.streamId;
          setStreamId(currentStreamId);
        }

        // Process stream
        const reader = response.body.getReader();
        readerRef.current = reader;
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process based on format
          if (useJsonl) {
            // JSON Lines format - split by newlines
            const lines = buffer.split("\n");
            buffer = lines.pop() || ""; // Keep incomplete line in buffer

            for (const line of lines) {
              const chunk = parseJsonlLine(line);
              if (chunk) {
                currentStreamId = chunk.stream_id;
                setStreamId(currentStreamId);
                fullText += chunk.text;
                setCurrentText(fullText);
                setChunkIndex(chunk.chunk_index);
                totalChunks++;
                lastChunk = chunk;
                onChunk?.(chunk);
              }
            }
          } else {
            // SSE format - split by double newlines
            const events = buffer.split("\n\n");
            buffer = events.pop() || ""; // Keep incomplete event in buffer

            for (const eventText of events) {
              if (!eventText.trim()) continue;

              // Skip comments (keep-alive)
              if (eventText.startsWith(":")) continue;

              const parsed = parseSSEEvent(eventText);
              if (!parsed) continue;

              const { eventType, data } = parsed;

              if (eventType === "error" && data && "error" in data) {
                throw new Error(data.error);
              }

              if (eventType === "done") {
                // Stream complete
                continue;
              }

              if (
                eventType === "chunk" &&
                data &&
                "text" in data &&
                !("error" in data)
              ) {
                const chunk = data as StreamChunk;
                currentStreamId = chunk.stream_id;
                setStreamId(currentStreamId);
                fullText += chunk.text;
                setCurrentText(fullText);
                setChunkIndex(chunk.chunk_index);
                totalChunks++;
                lastChunk = chunk;
                onChunk?.(chunk);
              }
            }
          }
        }

        // Stream completed successfully
        setState("completed");

        const result: StreamResult = {
          fullText,
          streamId: currentStreamId,
          provider: extractedMetadata?.provider || lastChunk?.provider || "",
          model: extractedMetadata?.model || lastChunk?.model || "",
          totalChunks,
          totalTokens: lastChunk?.usage?.total_tokens,
          durationMs: Date.now() - startTime,
          finishReason: lastChunk?.finish_reason,
          usage: lastChunk?.usage,
          metadata: extractedMetadata || undefined,
        };

        onComplete?.(result);
      } catch (err) {
        clearTimeout(timeoutId);

        if (err instanceof Error && err.name === "AbortError") {
          setState("cancelled");
          const cancelError: StreamError = {
            message: "Stream cancelled",
            code: "CANCELLED",
            streamId: currentStreamId || undefined,
          };
          setError(cancelError);
          onError?.(cancelError);
        } else {
          setState("error");
          const streamError: StreamError = {
            message: err instanceof Error ? err.message : "Unknown error",
            code: "STREAM_ERROR",
            streamId: currentStreamId || undefined,
          };
          setError(streamError);
          onError?.(streamError);
        }
      } finally {
        abortControllerRef.current = null;
        readerRef.current = null;
      }
    },
    [baseUrl, sessionId, timeout, useJsonl, parseSSEEvent, parseJsonlLine, extractMetadataFromHeaders]
  );

  /**
   * Stream text generation
   */
  const streamGenerate = useCallback(
    async (
      params: GenerateParams,
      onChunk?: (chunk: StreamChunk) => void,
      onComplete?: (result: StreamResult) => void,
      onError?: (error: StreamError) => void
    ) => {
      const endpoint = useJsonl ? "/stream/generate/jsonl" : "/stream/generate";

      // Use explicit params, then hook options, then current selection
      const provider =
        params.provider || defaultProvider || selectedProvider || undefined;
      const model =
        params.model || defaultModel || selectedModel || undefined;

      const body: Record<string, unknown> = {
        prompt: params.prompt,
        system_instruction: params.systemInstruction,
        temperature: params.temperature ?? 0.7,
        max_tokens: params.maxTokens,
        top_p: params.topP,
        top_k: params.topK,
        stop_sequences: params.stopSequences,
        provider,
        model,
      };

      await doStream(endpoint, body, onChunk, onComplete, onError);
    },
    [
      doStream,
      useJsonl,
      defaultProvider,
      defaultModel,
      selectedProvider,
      selectedModel,
    ]
  );

  /**
   * Stream chat completion
   */
  const streamChat = useCallback(
    async (
      params: ChatParams,
      onChunk?: (chunk: StreamChunk) => void,
      onComplete?: (result: StreamResult) => void,
      onError?: (error: StreamError) => void
    ) => {
      const endpoint = "/stream/chat";

      const provider =
        params.provider || defaultProvider || selectedProvider || undefined;
      const model =
        params.model || defaultModel || selectedModel || undefined;

      const body: Record<string, unknown> = {
        messages: params.messages,
        temperature: params.temperature ?? 0.7,
        max_tokens: params.maxTokens,
        top_p: params.topP,
        top_k: params.topK,
        stop_sequences: params.stopSequences,
        provider,
        model,
      };

      await doStream(endpoint, body, onChunk, onComplete, onError);
    },
    [
      doStream,
      defaultProvider,
      defaultModel,
      selectedProvider,
      selectedModel,
    ]
  );

  /**
   * Abort current stream
   */
  const abortStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (readerRef.current) {
      readerRef.current.cancel().catch(() => {
        // Ignore cancel errors
      });
    }
  }, []);

  /**
   * Reset state
   */
  const reset = useCallback(() => {
    abortStream();
    setState("idle");
    setCurrentText("");
    setStreamId(null);
    setError(null);
    setChunkIndex(0);
    setStreamMetadata(null);
  }, [abortStream]);

  return {
    // State
    state,
    isStreaming: state === "streaming",
    currentText,
    streamId,
    error,
    chunkIndex,
    streamMetadata,

    // Actions
    streamGenerate,
    streamChat,
    abortStream,
    reset,
  };
}

// =============================================================================
// Exports
// =============================================================================

export default useStreamingGeneration;
