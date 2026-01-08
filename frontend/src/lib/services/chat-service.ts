/**
 * Chat Service for Project Chimera Frontend
 * 
 * Provides API methods for chat completions functionality including:
 * - Chat completions (OpenAI-compatible)
 * - Streaming support
 * - Message formatting
 */

import { enhancedApi } from "../api-enhanced";

// =============================================================================
// Configuration
// =============================================================================

const CHAT_BASE_PATH = "/api/v1/chat";

// =============================================================================
// Types
// =============================================================================

export interface ChatMessage {
  /** Role of the message sender */
  role: "system" | "user" | "assistant" | "function";
  /** Message content */
  content: string;
  /** Function name (for function role) */
  name?: string;
  /** Function call (for assistant role) */
  function_call?: {
    name: string;
    arguments: string;
  };
}

export interface ChatCompletionRequest {
  /** Model to use for completion */
  model?: string;
  /** Messages in the conversation */
  messages: ChatMessage[];
  /** Temperature for sampling (0-2) */
  temperature?: number;
  /** Top-p sampling */
  top_p?: number;
  /** Number of completions to generate */
  n?: number;
  /** Whether to stream the response */
  stream?: boolean;
  /** Stop sequences */
  stop?: string | string[];
  /** Maximum tokens to generate */
  max_tokens?: number;
  /** Presence penalty (-2 to 2) */
  presence_penalty?: number;
  /** Frequency penalty (-2 to 2) */
  frequency_penalty?: number;
  /** Logit bias */
  logit_bias?: Record<string, number>;
  /** User identifier */
  user?: string;
  /** Provider to use (optional, uses session default) */
  provider?: string;
}

export interface ChatCompletionChoice {
  /** Index of the choice */
  index: number;
  /** The generated message */
  message: ChatMessage;
  /** Finish reason */
  finish_reason: "stop" | "length" | "function_call" | "content_filter" | null;
}

export interface ChatCompletionUsage {
  /** Prompt tokens used */
  prompt_tokens: number;
  /** Completion tokens generated */
  completion_tokens: number;
  /** Total tokens */
  total_tokens: number;
}

export interface ChatCompletionResponse {
  /** Unique identifier */
  id: string;
  /** Object type */
  object: "chat.completion";
  /** Creation timestamp */
  created: number;
  /** Model used */
  model: string;
  /** Generated choices */
  choices: ChatCompletionChoice[];
  /** Token usage */
  usage?: ChatCompletionUsage;
  /** Provider used */
  provider_used?: string;
}

export interface ChatCompletionChunk {
  /** Unique identifier */
  id: string;
  /** Object type */
  object: "chat.completion.chunk";
  /** Creation timestamp */
  created: number;
  /** Model used */
  model: string;
  /** Delta choices */
  choices: ChatCompletionChunkChoice[];
}

export interface ChatCompletionChunkChoice {
  /** Index of the choice */
  index: number;
  /** Delta content */
  delta: Partial<ChatMessage>;
  /** Finish reason */
  finish_reason: "stop" | "length" | "function_call" | "content_filter" | null;
}

// =============================================================================
// API Methods
// =============================================================================

/**
 * Create a chat completion
 */
export async function createChatCompletion(
  request: ChatCompletionRequest
): Promise<ChatCompletionResponse> {
  const response = await enhancedApi.post<ChatCompletionResponse>(
    `${CHAT_BASE_PATH}/completions`,
    { ...request, stream: false }
  );
  return response;
}

/**
 * Create a streaming chat completion
 */
export async function* createStreamingChatCompletion(
  request: ChatCompletionRequest
): AsyncGenerator<ChatCompletionChunk, void, unknown> {
  const response = await fetch(`${CHAT_BASE_PATH}/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ...request, stream: true }),
  });

  if (!response.ok) {
    throw new Error(`Chat completion failed: ${response.statusText}`);
  }

  if (!response.body) {
    throw new Error("No response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      
      // Process complete SSE messages
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        
        if (!trimmed || trimmed === "data: [DONE]") {
          continue;
        }

        if (trimmed.startsWith("data: ")) {
          try {
            const json = JSON.parse(trimmed.slice(6));
            yield json as ChatCompletionChunk;
          } catch {
            // Skip invalid JSON
            console.warn("Invalid JSON in SSE stream:", trimmed);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Create a system message
 */
export function systemMessage(content: string): ChatMessage {
  return { role: "system", content };
}

/**
 * Create a user message
 */
export function userMessage(content: string): ChatMessage {
  return { role: "user", content };
}

/**
 * Create an assistant message
 */
export function assistantMessage(content: string): ChatMessage {
  return { role: "assistant", content };
}

/**
 * Create a function message
 */
export function functionMessage(name: string, content: string): ChatMessage {
  return { role: "function", name, content };
}

/**
 * Format messages for display
 */
export function formatMessagesForDisplay(messages: ChatMessage[]): string {
  return messages
    .map((msg) => {
      const roleLabel = msg.role.charAt(0).toUpperCase() + msg.role.slice(1);
      return `${roleLabel}: ${msg.content}`;
    })
    .join("\n\n");
}

/**
 * Extract text content from streaming chunks
 */
export function extractChunkContent(chunk: ChatCompletionChunk): string {
  return chunk.choices
    .map((choice) => choice.delta.content || "")
    .join("");
}

/**
 * Accumulate streaming chunks into a complete response
 */
export async function accumulateStreamingResponse(
  stream: AsyncGenerator<ChatCompletionChunk, void, unknown>
): Promise<{ content: string; finishReason: string | null }> {
  let content = "";
  let finishReason: string | null = null;

  for await (const chunk of stream) {
    content += extractChunkContent(chunk);
    
    // Get finish reason from last chunk
    const lastChoice = chunk.choices[chunk.choices.length - 1];
    if (lastChoice?.finish_reason) {
      finishReason = lastChoice.finish_reason;
    }
  }

  return { content, finishReason };
}

// =============================================================================
// State Types
// =============================================================================

export interface ChatState {
  /** Conversation messages */
  messages: ChatMessage[];
  /** Whether a request is in progress */
  isLoading: boolean;
  /** Whether streaming is in progress */
  isStreaming: boolean;
  /** Current streaming content */
  streamingContent: string;
  /** Error message */
  error: string | null;
  /** Last response */
  lastResponse: ChatCompletionResponse | null;
}

/**
 * Create initial chat state
 */
export function createInitialChatState(): ChatState {
  return {
    messages: [],
    isLoading: false,
    isStreaming: false,
    streamingContent: "",
    error: null,
    lastResponse: null,
  };
}

/**
 * Add a message to the chat state
 */
export function addMessage(state: ChatState, message: ChatMessage): ChatState {
  return {
    ...state,
    messages: [...state.messages, message],
  };
}

/**
 * Set loading state
 */
export function setLoading(state: ChatState, isLoading: boolean): ChatState {
  return {
    ...state,
    isLoading,
    error: isLoading ? null : state.error,
  };
}

/**
 * Set streaming state
 */
export function setStreaming(
  state: ChatState,
  isStreaming: boolean,
  content: string = ""
): ChatState {
  return {
    ...state,
    isStreaming,
    streamingContent: content,
  };
}

/**
 * Update streaming content
 */
export function updateStreamingContent(
  state: ChatState,
  content: string
): ChatState {
  return {
    ...state,
    streamingContent: state.streamingContent + content,
  };
}

/**
 * Set error state
 */
export function setError(state: ChatState, error: string): ChatState {
  return {
    ...state,
    isLoading: false,
    isStreaming: false,
    error,
  };
}

/**
 * Clear chat state
 */
export function clearChat(): ChatState {
  return createInitialChatState();
}

// =============================================================================
// Service Object
// =============================================================================

export const chatService = {
  // API methods
  createChatCompletion,
  createStreamingChatCompletion,
  
  // Message helpers
  systemMessage,
  userMessage,
  assistantMessage,
  functionMessage,
  formatMessagesForDisplay,
  
  // Streaming helpers
  extractChunkContent,
  accumulateStreamingResponse,
  
  // State helpers
  createInitialChatState,
  addMessage,
  setLoading,
  setStreaming,
  updateStreamingContent,
  setError,
  clearChat,
};

export default chatService;