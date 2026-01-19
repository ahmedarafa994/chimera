/**
 * Chat Service (Refactored)
 *
 * Provides API methods for chat completions, streaming responses,
 * and conversation management using the new API architecture.
 *
 * This is a refactored version demonstrating migration to the new API client.
 *
 * @module lib/api/services/chat-service
 */

import { apiClient, ENDPOINTS, configManager } from '../core';
import type {
  ChatMessage,
  ChatRequest,
  ChatResponse,
  StreamChunk,
  ProviderName,
} from '../core/types';

// ============================================================================
// Types (Re-exported from core types for backward compatibility)
// ============================================================================

export type { ChatMessage, ChatRequest, ChatResponse, StreamChunk };

// Legacy type aliases for backward compatibility
export type ChatCompletionRequest = ChatRequest;
export type ChatCompletionResponse = ChatResponse;

export interface ChatChoice {
  index: number;
  message: ChatMessage;
  finish_reason: 'stop' | 'length' | 'content_filter' | null;
}

export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface StreamChoice {
  index: number;
  delta: Partial<ChatMessage>;
  finish_reason: 'stop' | 'length' | 'content_filter' | null;
}

export interface ConversationHistory {
  id: string;
  title: string;
  messages: ChatMessage[];
  model: string;
  provider: string;
  created_at: string;
  updated_at: string;
}

// ============================================================================
// Constants
// ============================================================================

const API_BASE = '/chat';

// ============================================================================
// Chat Service
// ============================================================================

/**
 * Chat Service API
 *
 * Refactored to use the new unified API client with:
 * - Automatic retry logic
 * - Circuit breaker protection
 * - Request deduplication
 * - Caching where appropriate
 * - Comprehensive error handling
 */
export const chatService = {
  /**
   * Create a chat completion
   *
   * @example
   * ```typescript
   * const response = await chatService.createCompletion({
   *   messages: [{ role: 'user', content: 'Hello!' }],
   *   model: 'gpt-4',
   * });
   * ```
   */
  async createCompletion(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    return apiClient.post<ChatCompletionResponse>(
      `${API_BASE}/completions`,
      { ...request, stream: false },
      {
        // Don't cache chat completions
        skipCache: true,
        // Use provider-specific circuit breaker
        circuitBreakerKey: request.provider
          ? `provider:${request.provider}`
          : 'chat:completions',
        // Retry on transient failures
        retryConfig: {
          maxRetries: 2,
          baseDelay: 1000,
          retryableStatusCodes: [429, 500, 502, 503, 504],
        },
      }
    );
  },

  /**
   * Create a streaming chat completion
   * Returns an async generator for streaming responses
   *
   * @example
   * ```typescript
   * for await (const chunk of chatService.streamCompletion(request)) {
   *   console.log(chunk.choices[0]?.delta?.content);
   * }
   * ```
   */
  async *streamCompletion(
    request: ChatCompletionRequest
  ): AsyncGenerator<StreamChunk, void, unknown> {
    const baseUrl = configManager.getActiveBaseUrl() || process.env.NEXT_PUBLIC_API_URL || '';

    // Get auth headers from configManager
    const apiHeaders = configManager.getApiHeaders();
    const headers: Record<string, string> = {};

    // Copy headers, ensuring Content-Type is set
    Object.entries(apiHeaders).forEach(([key, value]) => {
      if (value) {
        headers[key] = value;
      }
    });
    headers['Content-Type'] = 'application/json';

    const response = await fetch(`${baseUrl}${API_BASE}/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.message || `HTTP error! status: ${response.status}`
      );
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') return;
            try {
              yield JSON.parse(data) as StreamChunk;
            } catch {
              // Skip invalid JSON
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  /**
   * Get conversation history
   *
   * @example
   * ```typescript
   * const { conversations, total } = await chatService.getConversationHistory({
   *   limit: 10,
   *   offset: 0,
   * });
   * ```
   */
  async getConversationHistory(params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ conversations: ConversationHistory[]; total: number }> {
    return apiClient.get<{ conversations: ConversationHistory[]; total: number }>(
      `${API_BASE}/history`,
      {
        params,
        // Cache conversation list for 30 seconds
        cacheTTL: 30000,
      }
    );
  },

  /**
   * Get a specific conversation
   *
   * @example
   * ```typescript
   * const conversation = await chatService.getConversation('conv-123');
   * ```
   */
  async getConversation(id: string): Promise<ConversationHistory> {
    return apiClient.get<ConversationHistory>(
      `${API_BASE}/conversation/${id}`,
      {
        // Cache individual conversations for 1 minute
        cacheTTL: 60000,
      }
    );
  },

  /**
   * Save a conversation
   *
   * @example
   * ```typescript
   * const saved = await chatService.saveConversation({
   *   title: 'My Chat',
   *   messages: [...],
   *   model: 'gpt-4',
   *   provider: 'openai',
   * });
   * ```
   */
  async saveConversation(
    conversation: Omit<ConversationHistory, 'id' | 'created_at' | 'updated_at'>
  ): Promise<ConversationHistory> {
    const result = await apiClient.post<ConversationHistory>(
      `${API_BASE}/conversation`,
      conversation
    );

    // Invalidate conversation history cache
    apiClient.invalidateCache(/\/api\/v1\/chat\/history/);

    return result;
  },

  /**
   * Update a conversation
   *
   * @example
   * ```typescript
   * const updated = await chatService.updateConversation('conv-123', {
   *   title: 'Updated Title',
   * });
   * ```
   */
  async updateConversation(
    id: string,
    updates: Partial<Omit<ConversationHistory, 'id' | 'created_at' | 'updated_at'>>
  ): Promise<ConversationHistory> {
    const result = await apiClient.patch<ConversationHistory>(
      `${API_BASE}/conversation/${id}`,
      updates
    );

    // Invalidate caches
    apiClient.invalidateCache(/\/api\/v1\/chat\/history/);
    apiClient.invalidateCache(new RegExp(`/api/v1/chat/conversation/${id}`));

    return result;
  },

  /**
   * Delete a conversation
   *
   * @example
   * ```typescript
   * await chatService.deleteConversation('conv-123');
   * ```
   */
  async deleteConversation(id: string): Promise<{ success: boolean }> {
    const result = await apiClient.delete<{ success: boolean }>(
      `${API_BASE}/conversation/${id}`
    );

    // Invalidate caches
    apiClient.invalidateCache(/\/api\/v1\/chat\/history/);
    apiClient.invalidateCache(new RegExp(`/api/v1/chat/conversation/${id}`));

    return result;
  },

  /**
   * Get available models for chat
   *
   * @example
   * ```typescript
   * const { models } = await chatService.getAvailableModels();
   * ```
   */
  async getAvailableModels(): Promise<{
    models: { id: string; name: string; provider: string; context_length: number }[];
  }> {
    return apiClient.get<{
      models: { id: string; name: string; provider: string; context_length: number }[];
    }>(
      `${API_BASE}/models`,
      {
        // Cache models for 5 minutes
        cacheTTL: 300000,
      }
    );
  },

  /**
   * Get available models for a specific provider
   *
   * @example
   * ```typescript
   * const { models } = await chatService.getProviderModels('openai');
   * ```
   */
  async getProviderModels(provider: ProviderName): Promise<{
    models: { id: string; name: string; context_length: number }[];
  }> {
    return apiClient.get<{
      models: { id: string; name: string; context_length: number }[];
    }>(
      `${API_BASE}/models/${provider}`,
      {
        cacheTTL: 300000,
        circuitBreakerKey: `provider:${provider}`,
      }
    );
  },

  /**
   * Estimate token count for messages
   *
   * @example
   * ```typescript
   * const { tokens } = await chatService.estimateTokens(messages, 'gpt-4');
   * ```
   */
  async estimateTokens(
    messages: ChatMessage[],
    model: string
  ): Promise<{ tokens: number; model: string }> {
    return apiClient.post<{ tokens: number; model: string }>(
      `${API_BASE}/tokens/estimate`,
      { messages, model },
      {
        // Cache token estimates for 1 minute
        cacheTTL: 60000,
      }
    );
  },
};

export default chatService;
