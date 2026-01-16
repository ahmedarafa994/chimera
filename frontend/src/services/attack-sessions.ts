/**
 * Attack Session Recording, Replay & Sharing Service
 *
 * Phase 2 feature for competitive differentiation:
 * - Complete session saving with full context
 * - Reproducible test execution
 * - Secure sharing with team members
 * - Import/export capabilities
 */

import { toast } from 'sonner';

export type SessionStatus = 'active' | 'completed' | 'paused' | 'failed' | 'shared';
export type SharePermission = 'view' | 'replay' | 'edit' | 'admin';

export interface AttackStep {
  step_id: string;
  timestamp: string;
  step_type: string; // "prompt", "transform", "execute", "analyze"

  // Step data
  input_data: Record<string, any>;
  output_data: Record<string, any>;
  metadata: Record<string, any>;

  // Results
  success: boolean;
  execution_time: number; // seconds
  error_message?: string;

  // Context
  technique_used?: string;
  target_model?: string;
  target_provider?: string;
}

export interface AttackSession {
  session_id: string;
  name: string;
  description?: string;

  // Ownership and sharing
  owner_id: string;
  created_by: string;
  created_at: string;
  updated_at: string;

  // Session state
  status: SessionStatus;

  // Attack configuration
  target_config: Record<string, any>;
  techniques_config: Record<string, any>;
  original_prompt: string;

  // Session data
  steps: AttackStep[];
  results_summary: Record<string, any>;

  // Metrics
  total_steps: number;
  successful_steps: number;
  total_execution_time: number;

  // Sharing
  is_public: boolean;
  shared_with: Array<Record<string, any>>;

  // Tags and categorization
  tags: string[];
  category?: string;
}

export interface SessionCreate {
  name: string;
  description?: string;

  // Initial configuration
  target_provider: string;
  target_model: string;
  target_config?: Record<string, any>;
  original_prompt: string;

  // Optional configuration
  techniques_config?: Record<string, any>;
  tags?: string[];
  category?: string;
}

export interface SessionUpdate {
  name?: string;
  description?: string;
  status?: SessionStatus;
  tags?: string[];
  category?: string;
}

export interface SessionShare {
  user_ids: string[];
  permission: SharePermission;
  message?: string;
  expires_at?: string;
}

export interface SessionReplay {
  session_id: string;
  replay_name?: string;
  start_from_step?: number;
  stop_at_step?: number;
  modify_target?: Record<string, string>;
  skip_steps?: number[];
}

export interface SessionExport {
  session: AttackSession;
  export_format: string;
  export_metadata: Record<string, any>;
}

export interface SessionListResponse {
  sessions: AttackSession[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface SessionListParams {
  page?: number;
  page_size?: number;
  status?: SessionStatus;
  category?: string;
  search?: string;
  owner_only?: boolean;
  shared_only?: boolean;
}

class AttackSessionService {
  private readonly baseUrl = '/api/v1/sessions';

  /**
   * Create a new attack session
   */
  async createSession(sessionData: SessionCreate): Promise<AttackSession> {
    try {
      const response = await fetch(`${this.baseUrl}/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          name: sessionData.name,
          description: sessionData.description,
          target_provider: sessionData.target_provider,
          target_model: sessionData.target_model,
          target_config: sessionData.target_config || {},
          original_prompt: sessionData.original_prompt,
          techniques_config: sessionData.techniques_config || {},
          tags: sessionData.tags || [],
          category: sessionData.category
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      toast.success('Attack session created successfully');
      return result;
    } catch (error) {
      console.error('Failed to create attack session:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create session');
      throw error;
    }
  }

  /**
   * List attack sessions with filtering and pagination
   */
  async listSessions(params?: SessionListParams): Promise<SessionListResponse> {
    try {
      const searchParams = new URLSearchParams();

      if (params?.page) searchParams.append('page', params.page.toString());
      if (params?.page_size) searchParams.append('page_size', params.page_size.toString());
      if (params?.status) searchParams.append('status', params.status);
      if (params?.category) searchParams.append('category', params.category);
      if (params?.search) searchParams.append('search', params.search);
      if (params?.owner_only) searchParams.append('owner_only', 'true');
      if (params?.shared_only) searchParams.append('shared_only', 'true');

      const response = await fetch(`${this.baseUrl}/?${searchParams}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to list sessions:', error);
      toast.error('Failed to load session history');
      throw error;
    }
  }

  /**
   * Get detailed session information
   */
  async getSession(sessionId: string): Promise<AttackSession> {
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get session:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to load session');
      throw error;
    }
  }

  /**
   * Update session metadata
   */
  async updateSession(sessionId: string, updateData: SessionUpdate): Promise<AttackSession> {
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(updateData),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      toast.success('Session updated successfully');
      return result;
    } catch (error) {
      console.error('Failed to update session:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to update session');
      throw error;
    }
  }

  /**
   * Add a step to an active session
   */
  async addSessionStep(sessionId: string, stepData: Partial<AttackStep>): Promise<AttackStep> {
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}/steps`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(stepData),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to add session step:', error);
      toast.error('Failed to record session step');
      throw error;
    }
  }

  /**
   * Share session with other users
   */
  async shareSession(sessionId: string, shareData: SessionShare): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}/share`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          user_ids: shareData.user_ids,
          permission: shareData.permission,
          message: shareData.message,
          expires_at: shareData.expires_at
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      toast.success(`Session shared with ${shareData.user_ids.length} users`);
    } catch (error) {
      console.error('Failed to share session:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to share session');
      throw error;
    }
  }

  /**
   * Replay an existing session
   */
  async replaySession(sessionId: string, replayData: SessionReplay): Promise<AttackSession> {
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}/replay`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          session_id: replayData.session_id,
          replay_name: replayData.replay_name,
          start_from_step: replayData.start_from_step || 0,
          stop_at_step: replayData.stop_at_step,
          modify_target: replayData.modify_target,
          skip_steps: replayData.skip_steps || []
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      toast.success('Session replayed successfully');
      return result;
    } catch (error) {
      console.error('Failed to replay session:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to replay session');
      throw error;
    }
  }

  /**
   * Export session data
   */
  async exportSession(sessionId: string, format: string = 'json'): Promise<SessionExport> {
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}/export?format=${format}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      toast.success('Session exported successfully');
      return result;
    } catch (error) {
      console.error('Failed to export session:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to export session');
      throw error;
    }
  }

  /**
   * Import session data
   */
  async importSession(importData: SessionExport): Promise<AttackSession> {
    try {
      const response = await fetch(`${this.baseUrl}/import`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(importData),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      toast.success('Session imported successfully');
      return result;
    } catch (error) {
      console.error('Failed to import session:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to import session');
      throw error;
    }
  }

  /**
   * Delete a session
   */
  async deleteSession(sessionId: string): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}`, {
        method: 'DELETE',
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      toast.success('Session deleted successfully');
    } catch (error) {
      console.error('Failed to delete session:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to delete session');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for session status
   */
  getStatusDisplayName(status: SessionStatus): string {
    const displayNames: Record<SessionStatus, string> = {
      active: 'Active',
      completed: 'Completed',
      paused: 'Paused',
      failed: 'Failed',
      shared: 'Shared'
    };
    return displayNames[status];
  }

  /**
   * Get color for session status
   */
  getStatusColor(status: SessionStatus): string {
    const colors: Record<SessionStatus, string> = {
      active: 'green',
      completed: 'blue',
      paused: 'yellow',
      failed: 'red',
      shared: 'purple'
    };
    return colors[status];
  }

  /**
   * Get display name for share permission
   */
  getPermissionDisplayName(permission: SharePermission): string {
    const displayNames: Record<SharePermission, string> = {
      view: 'View Only',
      replay: 'View & Replay',
      edit: 'Edit & Replay',
      admin: 'Full Admin'
    };
    return displayNames[permission];
  }

  /**
   * Format execution time
   */
  formatExecutionTime(seconds: number): string {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  }

  /**
   * Calculate success rate
   */
  calculateSuccessRate(session: AttackSession): number {
    if (session.total_steps === 0) return 0;
    return (session.successful_steps / session.total_steps) * 100;
  }

  /**
   * Get step type display name
   */
  getStepTypeDisplayName(stepType: string): string {
    const displayNames: Record<string, string> = {
      prompt: 'Prompt Input',
      transform: 'Transformation',
      execute: 'Execution',
      analyze: 'Analysis'
    };
    return displayNames[stepType] || stepType;
  }

  /**
   * Create default session request
   */
  createDefaultSession(originalPrompt: string, targetProvider: string, targetModel: string): SessionCreate {
    return {
      name: `Attack Session - ${new Date().toLocaleDateString()}`,
      description: 'Automated attack session',
      target_provider: targetProvider,
      target_model: targetModel,
      original_prompt: originalPrompt,
      target_config: {},
      techniques_config: {},
      tags: ['automated'],
      category: 'research'
    };
  }

  /**
   * Validate session creation data
   */
  validateSessionCreate(data: SessionCreate): string[] {
    const errors: string[] = [];

    if (!data.name || data.name.trim().length === 0) {
      errors.push('Session name is required');
    }

    if (data.name && data.name.length > 255) {
      errors.push('Session name must be less than 255 characters');
    }

    if (!data.original_prompt || data.original_prompt.trim().length === 0) {
      errors.push('Original prompt is required');
    }

    if (!data.target_provider || data.target_provider.trim().length === 0) {
      errors.push('Target provider is required');
    }

    if (!data.target_model || data.target_model.trim().length === 0) {
      errors.push('Target model is required');
    }

    return errors;
  }
}

// Export singleton instance
export const attackSessionService = new AttackSessionService();