'use client';

import { ApiResponse, Session, SessionConfig, Agent, EvaluationResult, RefinementSuggestion } from '@/types/deepteam';

/**
 * DeepTeam API Client
 *
 * Provides methods for interacting with the DeepTeam backend API
 * for AI-powered prompt optimization and jailbreak research.
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

class DeepTeamApiClient {
  private baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl: string = BASE_URL) {
    this.baseUrl = baseUrl;
    this.apiKey = process.env.NEXT_PUBLIC_API_KEY;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // Add any existing headers
    if (options.headers) {
      Object.assign(headers, options.headers);
    }

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Session Management
  async createSession(config: SessionConfig): Promise<ApiResponse<Session>> {
    return this.request<Session>('/api/v1/sessions', {
      method: 'POST',
      body: JSON.stringify({ config }),
    });
  }

  async getSession(sessionId: string): Promise<ApiResponse<Session>> {
    return this.request<Session>(`/api/v1/sessions/${sessionId}`);
  }

  async listSessions(): Promise<ApiResponse<Session[]>> {
    return this.request<Session[]>('/api/v1/sessions');
  }

  async startSession(sessionId: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/api/v1/sessions/${sessionId}/start`, {
      method: 'POST',
    });
  }

  async pauseSession(sessionId: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/api/v1/sessions/${sessionId}/pause`, {
      method: 'POST',
    });
  }

  async stopSession(sessionId: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/api/v1/sessions/${sessionId}/stop`, {
      method: 'POST',
    });
  }

  // Agent Management
  async listAgents(sessionId?: string): Promise<ApiResponse<Agent[]>> {
    const endpoint = sessionId
      ? `/api/v1/sessions/${sessionId}/agents`
      : '/api/v1/agents';
    return this.request<Agent[]>(endpoint);
  }

  async getAgent(agentId: string): Promise<ApiResponse<Agent>> {
    return this.request<Agent>(`/api/v1/agents/${agentId}`);
  }

  // Evaluations
  async listEvaluations(sessionId: string): Promise<ApiResponse<EvaluationResult[]>> {
    return this.request<EvaluationResult[]>(`/api/v1/sessions/${sessionId}/evaluations`);
  }

  async getEvaluation(evaluationId: string): Promise<ApiResponse<EvaluationResult>> {
    return this.request<EvaluationResult>(`/api/v1/evaluations/${evaluationId}`);
  }

  // Refinements
  async listRefinements(sessionId: string): Promise<ApiResponse<RefinementSuggestion[]>> {
    return this.request<RefinementSuggestion[]>(`/api/v1/sessions/${sessionId}/refinements`);
  }

  async applyRefinement(
    sessionId: string,
    refinementId: string
  ): Promise<ApiResponse<void>> {
    return this.request<void>(`/api/v1/sessions/${sessionId}/refinements/${refinementId}/apply`, {
      method: 'POST',
    });
  }

  // WebSocket Connection for Real-time Updates
  createWebSocketConnection(sessionId: string): WebSocket {
    const wsUrl = this.baseUrl.replace('http', 'ws');
    const ws = new WebSocket(`${wsUrl}/ws/sessions/${sessionId}`);

    ws.onopen = () => {
      console.log(`WebSocket connected for session: ${sessionId}`);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log(`WebSocket closed for session: ${sessionId}`);
    };

    return ws;
  }

  // Health Check
  async healthCheck(): Promise<ApiResponse<{ status: string }>> {
    return this.request<{ status: string }>('/health');
  }
}

// Export singleton instance
export const deepTeamClient = new DeepTeamApiClient();
export default deepTeamClient;
