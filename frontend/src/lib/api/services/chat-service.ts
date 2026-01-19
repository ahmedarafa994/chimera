/**
 * Chat Service
 *
 * Provides API methods for chat completions, streaming responses,
 * and conversation management.
 */

import { apiClient } from '../client';
import { authManager } from '../auth-manager';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  name?: string;
}

export interface ChatCompletionRequest {
  messages: ChatMessage[];
  model?: string;
  provider?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop?: string[];
  stream?: boolean;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  provider: string;
  choices: ChatChoice[];
  usage: ChatUsage;
}

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

export interface StreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: StreamChoice[];
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

const API_BASE = '/chat';

/**
 * Chat Service API
 */
export const chatService = {
  /**
   * Create a chat completion
   */
  async createCompletion(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    const response = await apiClient.post<ChatCompletionResponse>(
      `${API_BASE}/completions`,
      { ...request, stream: false }
    );
    return response.data;
  },

  /**
   * Create a streaming chat completion
   * Returns an async generator for streaming responses
   */
  async *streamCompletion(
    request: ChatCompletionRequest
  ): AsyncGenerator<StreamChunk, void, unknown> {
    const token = await authManager.getAccessToken();
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1${API_BASE}/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
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
   */
  async getConversationHistory(params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ conversations: ConversationHistory[]; total: number }> {
    const response = await apiClient.get<{ conversations: ConversationHistory[]; total: number }>(
      `${API_BASE}/history`,
      { params }
    );
    return response.data;
  },

  /**
   * Get a specific conversation
   */
  async getConversation(id: string): Promise<ConversationHistory> {
    const response = await apiClient.get<ConversationHistory>(`${API_BASE}/conversation/${id}`);
    return response.data;
  },

  /**
   * Save a conversation
   */
  async saveConversation(conversation: Omit<ConversationHistory, 'id' | 'created_at' | 'updated_at'>): Promise<ConversationHistory> {
    const response = await apiClient.post<ConversationHistory>(
      `${API_BASE}/conversation`,
      conversation
    );
    return response.data;
  },

  /**
   * Delete a conversation
   */
  async deleteConversation(id: string): Promise<{ success: boolean }> {
    const response = await apiClient.delete<{ success: boolean }>(`${API_BASE}/conversation/${id}`);
    return response.data;
  },

  /**
   * Get available models for chat
   */
  async getAvailableModels(): Promise<{
    models: { id: string; name: string; provider: string; context_length: number }[];
  }> {
    const response = await apiClient.get<{
      models: { id: string; name: string; provider: string; context_length: number }[];
    }>(`${API_BASE}/models`);
    return response.data;
  },
};

export default chatService;
