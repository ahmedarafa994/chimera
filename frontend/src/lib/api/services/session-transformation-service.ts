/**
 * Session & Transformation Service - Aligned with Backend API
 *
 * This service is properly aligned with backend endpoints:
 * - POST /api/v1/session (create session)
 * - GET /api/v1/session/{id} (get session)
 * - PUT /api/v1/session/model (update session model)
 * - GET /api/v1/session/stats (session statistics)
 * - POST /api/v1/transform (transform prompt)
 * - POST /api/v1/execute (transform and execute)
 */

import { apiClient } from '../client';
import { apiErrorHandler } from '../../errors/api-error-handler';

// ============================================================================
// Session Types (matching backend Pydantic models)
// ============================================================================

export interface CreateSessionRequest {
  provider?: string;
  model?: string;
}

export interface CreateSessionResponse {
  success: boolean;
  session_id: string;
  provider?: string;
  model?: string;
  message?: string;
}

export interface SessionInfo {
  session_id: string;
  created_at: string;
  last_activity: string;
  provider?: string;
  model?: string;
  request_count: number;
}

export interface UpdateSessionModelRequest {
  provider?: string;
  model: string;
}

export interface UpdateSessionModelResponse {
  success: boolean;
  message?: string;
  reverted_to_default?: boolean;
  provider?: string;
  model?: string;
}

export interface SessionStatsResponse {
  active_sessions: number;
  total_requests: number;
  models_in_use: Record<string, number>;
}

// ============================================================================
// Transformation Types (matching backend Pydantic models)
// ============================================================================

export interface TransformationRequest {
  prompt: string;
  technique_suite?: string;
  potency_level?: number;
  provider?: string;
  model?: string;
  transformation_type?: string;
}

export interface TransformationResponse {
  success: boolean;
  result?: string;
  error?: string;
  transformed_prompt?: string;
  transformation_applied?: string;
  original_prompt?: string;
  technique_suite?: string;
  potency_level?: number;
  metadata?: Record<string, any>;
}

export interface ExecutionRequest {
  prompt?: string;
  core_request?: string;
  transformation?: string;
  technique_suite?: string;
  potency_level?: number;
  provider?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  api_key?: string;
}

export interface ExecutionResponse {
  success: boolean;
  result?: string | {
    provider?: string;
    model?: string;
    latency_ms?: number;
    text?: string;
  };
  error?: string;
  transformation?: string;
  provider?: string;
  model?: string;
  latency_ms?: number;
}

// ============================================================================
// Session & Transformation Service Implementation
// ============================================================================

export class SessionTransformationService {
  private currentSessionId?: string;

  // ============================================================================
  // Session Management Methods
  // ============================================================================

  /**
   * Create a new session
   */
  async createSession(request?: CreateSessionRequest) {
    try {
      const response = await apiClient.post<CreateSessionResponse>('/api/v1/session', request || {});

      // Store session ID for future use
      if (response.data.success && response.data.session_id) {
        this.currentSessionId = response.data.session_id;
      }

      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'CreateSession');
    }
  }

  /**
   * Get session information by ID
   */
  async getSession(sessionId: string) {
    try {
      const response = await apiClient.get<SessionInfo>(`/api/v1/session/${sessionId}`);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetSession');
    }
  }

  /**
   * Get current session (if one exists)
   */
  async getCurrentSession() {
    if (!this.currentSessionId) return null;

    try {
      return await this.getSession(this.currentSessionId);
    } catch (error) {
      // Session might be expired, clear it
      this.currentSessionId = undefined;
      return null;
    }
  }

  /**
   * Update the model for a session
   */
  async updateSessionModel(sessionId: string, request: UpdateSessionModelRequest) {
    try {
      const headers: Record<string, string> = {};
      if (sessionId) {
        headers['X-Session-ID'] = sessionId;
      }

      const response = await apiClient.put<UpdateSessionModelResponse>(
        '/api/v1/session/model',
        request,
        { headers }
      );

      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'UpdateSessionModel');
    }
  }

  /**
   * Update current session model
   */
  async updateCurrentSessionModel(request: UpdateSessionModelRequest) {
    if (!this.currentSessionId) {
      throw new Error('No active session. Create a session first.');
    }

    return this.updateSessionModel(this.currentSessionId, request);
  }

  /**
   * Get session statistics (admin endpoint)
   */
  async getSessionStats() {
    try {
      const response = await apiClient.get<SessionStatsResponse>('/api/v1/session/stats');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetSessionStats');
    }
  }

  /**
   * Clear current session
   */
  clearCurrentSession(): void {
    this.currentSessionId = undefined;
  }

  /**
   * Get current session ID
   */
  getCurrentSessionId(): string | undefined {
    return this.currentSessionId;
  }

  /**
   * Set current session ID
   */
  setCurrentSessionId(sessionId: string): void {
    this.currentSessionId = sessionId;
  }

  // ============================================================================
  // Transformation Methods
  // ============================================================================

  /**
   * Transform a prompt without execution
   */
  async transform(request: TransformationRequest) {
    try {
      const response = await apiClient.post<TransformationResponse>('/api/v1/transform', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'Transform');
    }
  }

  /**
   * Transform and execute a prompt
   */
  async execute(request: ExecutionRequest) {
    try {
      const headers: Record<string, string> = {};
      if (this.currentSessionId) {
        headers['X-Session-ID'] = this.currentSessionId;
      }

      const response = await apiClient.post<ExecutionResponse>(
        '/api/v1/execute',
        request,
        { headers }
      );

      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'Execute');
    }
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Create session with provider and model
   */
  async createSessionWithProvider(provider: string, model: string) {
    return this.createSession({ provider, model });
  }

  /**
   * Transform prompt with specific technique
   */
  async transformWithTechnique(
    prompt: string,
    technique: string,
    potency: number = 5,
    options?: {
      provider?: string;
      model?: string;
    }
  ) {
    return this.transform({
      prompt,
      technique_suite: technique,
      potency_level: potency,
      provider: options?.provider,
      model: options?.model,
    });
  }

  /**
   * Execute prompt with transformation
   */
  async executeWithTransformation(
    prompt: string,
    options?: {
      technique?: string;
      potency?: number;
      provider?: string;
      model?: string;
      temperature?: number;
      maxTokens?: number;
    }
  ) {
    return this.execute({
      core_request: prompt,
      technique_suite: options?.technique,
      potency_level: options?.potency || 5,
      provider: options?.provider,
      model: options?.model,
      temperature: options?.temperature,
      max_tokens: options?.maxTokens,
    });
  }

  /**
   * Get or create session
   */
  async getOrCreateSession(options?: CreateSessionRequest): Promise<string> {
    // Try to get current session
    const current = await this.getCurrentSession();
    if (current) {
      return current.data.session_id;
    }

    // Create new session
    const newSession = await this.createSession(options);
    if (newSession.data.success) {
      return newSession.data.session_id;
    }

    throw new Error('Failed to create session');
  }

  /**
   * Switch session context
   */
  async switchToSession(sessionId: string) {
    // Verify session exists
    await this.getSession(sessionId);

    // Switch context
    this.currentSessionId = sessionId;

    return sessionId;
  }

  /**
   * Create request with session context
   */
  createRequestWithSession<T extends Record<string, any>>(request: T): T & { sessionId?: string } {
    return {
      ...request,
      sessionId: this.currentSessionId,
    };
  }

  /**
   * Parse execution response for common patterns
   */
  parseExecutionResponse(response: ExecutionResponse) {
    let text = '';
    let provider = '';
    let model = '';
    let latency = 0;

    if (typeof response.result === 'string') {
      text = response.result;
    } else if (typeof response.result === 'object' && response.result) {
      text = response.result.text || '';
      provider = response.result.provider || '';
      model = response.result.model || '';
      latency = response.result.latency_ms || 0;
    }

    return {
      success: response.success,
      text,
      provider: provider || response.provider || '',
      model: model || response.model || '',
      latency: latency || response.latency_ms || 0,
      transformation: response.transformation,
      error: response.error,
    };
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const sessionTransformationService = new SessionTransformationService();

// ============================================================================
// Convenience functions for direct usage
// ============================================================================

export const sessionApi = {
  // Session management
  createSession: (request?: CreateSessionRequest) => sessionTransformationService.createSession(request),
  getSession: (sessionId: string) => sessionTransformationService.getSession(sessionId),
  getCurrentSession: () => sessionTransformationService.getCurrentSession(),
  updateSessionModel: (sessionId: string, request: UpdateSessionModelRequest) =>
    sessionTransformationService.updateSessionModel(sessionId, request),
  updateCurrentSessionModel: (request: UpdateSessionModelRequest) =>
    sessionTransformationService.updateCurrentSessionModel(request),
  getSessionStats: () => sessionTransformationService.getSessionStats(),

  // Session utilities
  clearCurrentSession: () => sessionTransformationService.clearCurrentSession(),
  getCurrentSessionId: () => sessionTransformationService.getCurrentSessionId(),
  setCurrentSessionId: (sessionId: string) => sessionTransformationService.setCurrentSessionId(sessionId),
  getOrCreateSession: (options?: CreateSessionRequest) => sessionTransformationService.getOrCreateSession(options),
  switchToSession: (sessionId: string) => sessionTransformationService.switchToSession(sessionId),
};

export const transformationApi = {
  // Transformation methods
  transform: (request: TransformationRequest) => sessionTransformationService.transform(request),
  execute: (request: ExecutionRequest) => sessionTransformationService.execute(request),

  // Convenience methods
  transformWithTechnique: (
    prompt: string,
    technique: string,
    potency?: number,
    options?: { provider?: string; model?: string }
  ) => sessionTransformationService.transformWithTechnique(prompt, technique, potency, options),
  executeWithTransformation: (
    prompt: string,
    options?: {
      technique?: string;
      potency?: number;
      provider?: string;
      model?: string;
      temperature?: number;
      maxTokens?: number;
    }
  ) => sessionTransformationService.executeWithTransformation(prompt, options),

  // Utility methods
  parseExecutionResponse: (response: ExecutionResponse) =>
    sessionTransformationService.parseExecutionResponse(response),
};

export default sessionTransformationService;