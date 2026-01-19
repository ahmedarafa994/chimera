import { apiClient } from '../client';
import { toast } from 'sonner';

// API Response Types
interface SessionApiResponse {
  session_id: string;
  name: string;
  description?: string;
  owner_id?: string;
  ownerId?: string;
  created_by?: string;
  created_at?: string;
  updated_at?: string;
  status?: string;
  target_config?: Record<string, any>;
  techniques_config?: Record<string, any>;
  original_prompt?: string;
  steps?: any[];
  results_summary?: Record<string, any>;
  total_steps?: number;
  successful_steps?: number;
  total_execution_time?: number;
  is_public?: boolean;
  shared_with?: Array<Record<string, any>>;
  tags?: string[];
  category?: string;
}

interface SessionListApiResponse {
  sessions?: SessionApiResponse[];
  total?: number;
  page?: number;
  page_size?: number;
  has_next?: boolean;
  has_prev?: boolean;
}

export type SessionStatus = 'active' | 'completed' | 'paused' | 'failed' | 'shared';
export type SharePermission = 'view' | 'replay' | 'edit' | 'admin';

export interface AttackStep {
  step_id: string;
  timestamp: string;
  step_type: string;
  input_data: Record<string, any>;
  output_data: Record<string, any>;
  metadata: Record<string, any>;
  success: boolean;
  execution_time: number;
  error_message?: string;
  technique_used?: string;
  target_model?: string;
  target_provider?: string;
}

export interface AttackSession {
  session_id: string;
  name: string;
  description?: string;
  owner_id: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  status: SessionStatus;
  target_config: Record<string, any>;
  techniques_config: Record<string, any>;
  original_prompt: string;
  steps: AttackStep[];
  results_summary?: Record<string, any>;
  total_steps: number;
  successful_steps: number;
  total_execution_time: number;
  is_public: boolean;
  shared_with: Array<Record<string, any>>;
  tags: string[];
  category?: string;
}

export interface SessionListResponse {
  sessions: AttackSession[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface SessionCreate {
  name: string;
  description?: string;
  target_provider: string;
  target_model: string;
  original_prompt: string;
  target_config?: Record<string, any>;
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
  modify_target?: { provider?: string; model?: string };
  skip_steps?: number[];
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
  private readonly baseUrl = '/sessions';

  private mapSession = (session: SessionApiResponse): AttackSession => ({
    session_id: session.session_id,
    name: session.name,
    description: session.description ?? '',
    owner_id: session.owner_id ?? session.ownerId ?? '',
    created_by: session.created_by ?? session.owner_id ?? '',
    created_at: session.created_at ? new Date(session.created_at).toISOString() : new Date().toISOString(),
    updated_at: session.updated_at ? new Date(session.updated_at).toISOString() : new Date().toISOString(),
    status: (session.status as SessionStatus) || 'active',
    target_config: session.target_config ?? {},
    techniques_config: session.techniques_config ?? {},
    original_prompt: session.original_prompt ?? '',
    steps: (session.steps ?? []).map((step: Record<string, any>) => ({
      step_id: step.step_id,
      timestamp: step.timestamp ? new Date(step.timestamp).toISOString() : new Date().toISOString(),
      step_type: step.step_type ?? 'execute',
      input_data: step.input_data ?? {},
      output_data: step.output_data ?? {},
      metadata: step.metadata ?? {},
      success: Boolean(step.success),
      execution_time: Number(step.execution_time ?? 0),
      error_message: step.error_message,
      technique_used: step.technique_used,
      target_model: step.target_model,
      target_provider: step.target_provider
    })),
    results_summary: session.results_summary ?? {},
    total_steps: Number(session.total_steps ?? (session.steps?.length || 0)),
    successful_steps: Number(session.successful_steps ?? 0),
    total_execution_time: Number(session.total_execution_time ?? 0),
    is_public: Boolean(session.is_public),
    shared_with: session.shared_with ?? [],
    tags: session.tags ?? [],
    category: session.category
  });

  async createSession(sessionData: SessionCreate): Promise<AttackSession> {
    try {
      const payload = {
        name: sessionData.name,
        description: sessionData.description,
        target_provider: sessionData.target_provider,
        target_model: sessionData.target_model,
        target_config: sessionData.target_config ?? {},
        techniques_config: sessionData.techniques_config ?? {},
        original_prompt: sessionData.original_prompt,
        tags: sessionData.tags ?? [],
        category: sessionData.category
      };

      const response = await apiClient.post(this.baseUrl, payload);
      const mapped = this.mapSession(response.data as SessionApiResponse);
      toast.success('Session saved successfully');
      return mapped;
    } catch (error) {
      console.error('Failed to save session:', error);
      toast.error('Failed to save attack session');
      throw error;
    }
  }

  async getSession(sessionId: string): Promise<AttackSession> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/${sessionId}`);
      return this.mapSession(response.data as SessionApiResponse);
    } catch (error) {
      console.error('Failed to get session:', error);
      toast.error('Failed to load attack session');
      throw error;
    }
  }

  async listSessions(params?: SessionListParams): Promise<SessionListResponse> {
    try {
      const response = await apiClient.get(this.baseUrl, {
        params: {
          page: params?.page,
          page_size: params?.page_size ?? params?.page_size,
          status: params?.status,
          category: params?.category,
          search: params?.search,
          owner_only: params?.owner_only,
          shared_only: params?.shared_only
        }
      });

      const data = response.data as SessionListApiResponse;
      return {
        sessions: (data.sessions ?? []).map(this.mapSession),
        total: data.total ?? 0,
        page: data.page ?? 1,
        page_size: data.page_size ?? params?.page_size ?? 20,
        has_next: Boolean(data.has_next),
        has_prev: Boolean(data.has_prev)
      };
    } catch (error) {
      console.error('Failed to list sessions:', error);
      toast.error('Failed to load attack sessions');
      throw error;
    }
  }

  async updateSession(sessionId: string, updateData: SessionUpdate): Promise<AttackSession> {
    try {
      const response = await apiClient.patch(`${this.baseUrl}/${sessionId}`, updateData);
      return this.mapSession(response.data as SessionApiResponse);
    } catch (error) {
      console.error('Failed to update session:', error);
      toast.error('Failed to update session');
      throw error;
    }
  }

  async deleteSession(sessionId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUrl}/${sessionId}`);
      toast.success('Session deleted successfully');
    } catch (error) {
      console.error('Failed to delete session:', error);
      toast.error('Failed to delete session');
      throw error;
    }
  }

  async addStep(sessionId: string, stepData: Partial<AttackStep>): Promise<AttackStep> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/${sessionId}/steps`, stepData);
      const mapped = this.mapSession({ steps: [response.data as Record<string, any>], session_id: sessionId } as SessionApiResponse).steps[0];
      return mapped;
    } catch (error) {
      console.error('Failed to add step:', error);
      toast.error('Failed to record attack step');
      throw error;
    }
  }

  async shareSession(sessionId: string, shareData: SessionShare): Promise<Record<string, any>> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/${sessionId}/share`, shareData);
      toast.success('Session shared successfully');
      return response.data as Record<string, any>;
    } catch (error) {
      console.error('Failed to share session:', error);
      toast.error('Failed to create share link');
      throw error;
    }
  }

  async replaySession(sessionId: string, replayData: SessionReplay): Promise<AttackSession> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/${sessionId}/replay`, replayData);
      toast.success('Replay session created');
      return this.mapSession(response.data as SessionApiResponse);
    } catch (error) {
      console.error('Failed to replay session:', error);
      toast.error('Failed to replay attack session');
      throw error;
    }
  }

  async importSession(importData: Record<string, any>): Promise<AttackSession> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/import`, importData);
      toast.success('Session imported successfully');
      return this.mapSession(response.data as SessionApiResponse);
    } catch (error) {
      console.error('Failed to import session:', error);
      toast.error('Failed to import attack session');
      throw error;
    }
  }

  async exportSession(sessionId: string, format: string = 'json'): Promise<Record<string, any>> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/${sessionId}/export`, {
        params: { format }
      });
      return response.data as Record<string, any>;
    } catch (error) {
      console.error('Failed to export session:', error);
      toast.error('Failed to export attack session');
      throw error;
    }
  }

  validateSessionCreate(data: SessionCreate): string[] {
    const errors: string[] = [];
    if (!data.name?.trim()) errors.push('Session name is required');
    if (!data.target_provider) errors.push('Target provider is required');
    if (!data.target_model) errors.push('Target model is required');
    if (!data.original_prompt?.trim()) errors.push('Original prompt is required');
    return errors;
  }

  calculateSuccessRate(session: AttackSession): number {
    if (!session.total_steps) return 0;
    return (session.successful_steps / session.total_steps) * 100;
  }

  formatExecutionTime(seconds: number): string {
    const rounded = Math.round(seconds);
    if (rounded < 60) return `${rounded}s`;
    const minutes = Math.floor(rounded / 60);
    const remainingSeconds = rounded % 60;
    return `${minutes}m ${remainingSeconds}s`;
  }

  getStatusColor(status: SessionStatus): string {
    const statusColors: Record<SessionStatus, string> = {
      active: 'blue',
      completed: 'green',
      paused: 'yellow',
      failed: 'red',
      shared: 'purple'
    };
    return statusColors[status] || 'gray';
  }

  getStatusDisplayName(status: SessionStatus): string {
    const displayNames: Record<SessionStatus, string> = {
      active: 'Active',
      completed: 'Completed',
      paused: 'Paused',
      failed: 'Failed',
      shared: 'Shared'
    };
    return displayNames[status] || status;
  }

  getPermissionDisplayName(permission: SharePermission): string {
    const display: Record<SharePermission, string> = {
      view: 'View only',
      replay: 'Replay',
      edit: 'Edit',
      admin: 'Admin'
    };
    return display[permission];
  }

  getStepTypeDisplayName(stepType: string): string {
    const display: Record<string, string> = {
      prompt: 'Prompt',
      transform: 'Transform',
      execute: 'Execute',
      analyze: 'Analyze'
    };
    return display[stepType] ?? stepType;
  }
}

export const attackSessionService = new AttackSessionService();
