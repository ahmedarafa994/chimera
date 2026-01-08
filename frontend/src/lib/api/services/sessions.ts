/**
 * Sessions API Service
 * Handles session management API calls
 */

import { apiClient } from '../client';
import { Session, Message, ApiResponse, PaginatedResponse } from '../types';

export interface CreateSessionRequest {
  model_id?: string;
  provider?: string;
  system_prompt?: string;
  metadata?: Record<string, any>;
}

export interface SendMessageRequest {
  content: string;
  stream?: boolean;
  metadata?: Record<string, any>;
}

export interface UpdateSessionRequest {
  model_id?: string;
  provider?: string;
  metadata?: Record<string, any>;
}

export const sessionsApi = {
  /**
   * Get current session
   */
  async getCurrentSession(): Promise<ApiResponse<Session>> {
    return apiClient.get<Session>('/session');
  },

  /**
   * Create new session
   */
  async createSession(request: CreateSessionRequest = {}): Promise<ApiResponse<Session>> {
    return apiClient.post<Session>('/session', request);
  },

  /**
   * Get session by ID
   */
  async getSession(sessionId: string): Promise<ApiResponse<Session>> {
    return apiClient.get<Session>(`/session/${sessionId}`);
  },

  /**
   * Update session
   */
  async updateSession(sessionId: string, request: UpdateSessionRequest): Promise<ApiResponse<Session>> {
    return apiClient.patch<Session>(`/session/${sessionId}`, request);
  },

  /**
   * Delete session
   */
  async deleteSession(sessionId: string): Promise<ApiResponse<void>> {
    return apiClient.delete(`/session/${sessionId}`);
  },

  /**
   * Get session messages
   */
  async getMessages(sessionId: string, page = 1, pageSize = 50): Promise<ApiResponse<PaginatedResponse<Message>>> {
    return apiClient.get<PaginatedResponse<Message>>(`/session/${sessionId}/messages`, {
      params: { page, page_size: pageSize },
    });
  },

  /**
   * Send message to session
   */
  async sendMessage(sessionId: string, request: SendMessageRequest): Promise<ApiResponse<Message>> {
    return apiClient.post<Message>(`/session/${sessionId}/message`, request);
  },

  /**
   * Clear session messages
   */
  async clearMessages(sessionId: string): Promise<ApiResponse<void>> {
    return apiClient.delete(`/session/${sessionId}/messages`);
  },

  /**
   * List all sessions
   */
  async listSessions(page = 1, pageSize = 20): Promise<ApiResponse<PaginatedResponse<Session>>> {
    return apiClient.get<PaginatedResponse<Session>>('/sessions', {
      params: { page, page_size: pageSize },
    });
  },

  /**
   * Export session as JSON
   */
  async exportSession(sessionId: string): Promise<ApiResponse<{
    session: Session;
    messages: Message[];
  }>> {
    return apiClient.get(`/session/${sessionId}/export`);
  },
};