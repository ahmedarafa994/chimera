/**
 * Team Workspaces & Collaboration Service
 *
 * Phase 3 enterprise feature for scalability:
 * - Multi-user team environments
 * - Role-based access control (Admin, Researcher, Viewer)
 * - Shared assessment history and API key pools
 * - Activity logging and audit trails
 */

import { apiClient } from '../client';

export type TeamRole = 'owner' | 'admin' | 'researcher' | 'viewer';
export type InvitationStatus = 'pending' | 'accepted' | 'declined' | 'expired';
export type ActivityType =
  | 'user_joined'
  | 'user_left'
  | 'role_changed'
  | 'assessment_created'
  | 'assessment_shared'
  | 'report_generated'
  | 'api_key_added'
  | 'api_key_removed'
  | 'workspace_created'
  | 'workspace_updated';

export interface TeamMember {
  user_id: string;
  username: string;
  email: string;
  role: TeamRole;
  joined_at: string;
  last_active?: string;
  is_active: boolean;
}

export interface TeamWorkspace {
  workspace_id: string;
  name: string;
  description?: string;

  // Ownership
  owner_id: string;
  created_at: string;
  updated_at: string;

  // Settings
  settings: Record<string, any>;

  // Statistics
  member_count: number;
  assessment_count: number;

  // Features
  features_enabled: string[];
}

export interface TeamInvitation {
  invitation_id: string;
  workspace_id: string;
  workspace_name: string;
  invited_email: string;
  invited_by: string;
  invited_by_name: string;
  role: TeamRole;

  status: InvitationStatus;
  created_at: string;
  expires_at: string;
  accepted_at?: string;

  message?: string;
}

export interface ActivityLogEntry {
  activity_id: string;
  workspace_id: string;
  user_id: string;
  username: string;
  activity_type: ActivityType;

  // Activity details
  description: string;
  metadata: Record<string, any>;

  // Timing
  timestamp: string;

  // Context
  ip_address?: string;
  user_agent?: string;
}

export interface WorkspaceCreate {
  name: string;
  description?: string;
  settings?: Record<string, any>;
}

export interface WorkspaceUpdate {
  name?: string;
  description?: string;
  settings?: Record<string, any>;
}

export interface WorkspaceListResponse {
  workspaces: TeamWorkspace[];
  total: number;
}

export interface MembersListResponse {
  members: TeamMember[];
  invitations: TeamInvitation[];
  total_members?: number;
  total_invitations?: number;
}

export interface ActivityLogResponse {
  activities: ActivityLogEntry[];
  total: number;
  page: number;
  page_size: number;
  has_next?: boolean;
  has_prev?: boolean;
}

export interface InviteUserRequest {
  email: string;
  role: TeamRole;
  message?: string;
}

export interface UpdateMemberRole {
  role: TeamRole;
}

export interface ActivityLogParams {
  page?: number;
  page_size?: number;
  activity_type?: ActivityType;
}

const API_BASE = '/workspaces';

/**
 * Workspace Service API
 */
export const workspaceService = {
  mapWorkspace(data: any): TeamWorkspace {
    return {
      workspace_id: data.workspace_id ?? data.id ?? '',
      name: data.name,
      description: data.description,
      owner_id: data.owner_id ?? '',
      created_at: data.created_at ?? new Date().toISOString(),
      updated_at: data.updated_at ?? new Date().toISOString(),
      settings: data.settings ?? {},
      member_count: data.member_count ?? 0,
      assessment_count: data.assessment_count ?? 0,
      features_enabled: data.features_enabled ?? []
    };
  },

  mapMember(data: any): TeamMember {
    return {
      user_id: data.user_id,
      username: data.username,
      email: data.email,
      role: data.role,
      joined_at: data.joined_at ?? new Date().toISOString(),
      last_active: data.last_active,
      is_active: data.is_active ?? true
    };
  },

  mapInvitation(data: any): TeamInvitation {
    return {
      invitation_id: data.invitation_id ?? data.id ?? '',
      workspace_id: data.workspace_id,
      workspace_name: data.workspace_name ?? '',
      invited_email: data.invited_email,
      invited_by: data.invited_by,
      invited_by_name: data.invited_by_name ?? '',
      role: data.role,
      status: data.status,
      created_at: data.created_at ?? new Date().toISOString(),
      expires_at: data.expires_at ?? new Date().toISOString(),
      accepted_at: data.accepted_at,
      message: data.message
    };
  },

  mapActivity(data: any): ActivityLogEntry {
    return {
      activity_id: data.activity_id ?? data.id ?? '',
      workspace_id: data.workspace_id,
      user_id: data.user_id,
      username: data.username,
      activity_type: data.activity_type,
      description: data.description,
      metadata: data.metadata ?? {},
      timestamp: data.timestamp ?? new Date().toISOString(),
      ip_address: data.ip_address,
      user_agent: data.user_agent
    };
  },

  /**
   * Create a new team workspace
   */
  async createWorkspace(workspaceData: WorkspaceCreate): Promise<TeamWorkspace> {
    const response = await apiClient.post<TeamWorkspace>(`${API_BASE}`, {
      name: workspaceData.name,
      description: workspaceData.description,
      settings: workspaceData.settings || {}
    });
    return this.mapWorkspace(response.data);
  },

  /**
   * List user's accessible workspaces
   */
  async listWorkspaces(): Promise<WorkspaceListResponse> {
    const response = await apiClient.get<WorkspaceListResponse>(`${API_BASE}`);
    const data: any = response.data;
    return {
      workspaces: (data.workspaces ?? []).map((ws: any) => this.mapWorkspace(ws)),
      total: data.total ?? data.workspaces?.length ?? 0
    };
  },

  /**
   * Get workspace details
   */
  async getWorkspace(workspaceId: string): Promise<TeamWorkspace> {
    const response = await apiClient.get<TeamWorkspace>(`${API_BASE}/${workspaceId}`);
    return this.mapWorkspace(response.data);
  },

  /**
   * Update workspace settings
   */
  async updateWorkspace(workspaceId: string, updateData: WorkspaceUpdate): Promise<TeamWorkspace> {
    const response = await apiClient.patch<TeamWorkspace>(`${API_BASE}/${workspaceId}`, updateData);
    return this.mapWorkspace(response.data);
  },

  /**
   * List workspace members and invitations
   */
  async listWorkspaceMembers(workspaceId: string): Promise<MembersListResponse> {
    const response = await apiClient.get<MembersListResponse>(`${API_BASE}/${workspaceId}/members`);
    const data: any = response.data;
    return {
      members: (data.members ?? []).map((m: any) => this.mapMember(m)),
      invitations: (data.invitations ?? []).map((inv: any) => this.mapInvitation(inv)),
      total_members: data.total_members,
      total_invitations: data.total_invitations
    };
  },

  /**
   * Invite user to workspace
   */
  async inviteUserToWorkspace(workspaceId: string, inviteRequest: InviteUserRequest): Promise<void> {
    await apiClient.post(`${API_BASE}/${workspaceId}/invite`, {
      email: inviteRequest.email,
      role: inviteRequest.role,
      message: inviteRequest.message
    });
  },

  /**
   * Update member role
   */
  async updateMemberRole(workspaceId: string, userId: string, roleUpdate: UpdateMemberRole): Promise<void> {
    await apiClient.patch(`${API_BASE}/${workspaceId}/members/${userId}/role`, roleUpdate);
  },

  /**
   * Remove member from workspace
   */
  async removeMember(workspaceId: string, userId: string): Promise<void> {
    await apiClient.delete(`${API_BASE}/${workspaceId}/members/${userId}`);
  },

  /**
   * List workspace activity logs
   */
  async listActivityLogs(workspaceId: string, params: ActivityLogParams = {}): Promise<{
    activities: ActivityLogEntry[];
    total: number;
    has_more: boolean;
  }> {
    const response = await apiClient.get(`${API_BASE}/${workspaceId}/activity`, { params });
    const data: any = response.data;
    return {
      activities: (data.activities ?? []).map((a: any) => this.mapActivity(a)),
      total: data.total ?? data.activities?.length ?? 0,
      has_more: Boolean(data.has_next)
    };
  },

  async getWorkspaceActivityLog(
    workspaceId: string,
    params: ActivityLogParams = {}
  ): Promise<ActivityLogResponse> {
    const response = await apiClient.get(`${API_BASE}/${workspaceId}/activity`, { params });
    const data: any = response.data;
    return {
      activities: (data.activities ?? []).map((a: any) => this.mapActivity(a)),
      total: data.total ?? data.activities?.length ?? 0,
      page: data.page ?? params.page ?? 1,
      page_size: data.page_size ?? params.page_size ?? 20,
      has_next: data.has_next,
      has_prev: data.has_prev
    };
  },

  validateWorkspaceCreate(data: WorkspaceCreate): Record<string, string> {
    const errors: Record<string, string> = {};
    if (!data.name?.trim()) errors.name = 'Workspace name is required';
    return errors;
  },

  createDefaultSettings(): Record<string, any> {
    return {
      notifications_enabled: true,
      default_visibility: 'team',
      auto_archive_days: 90
    };
  },

  validateInviteRequest(data: InviteUserRequest): Record<string, string> {
    const errors: Record<string, string> = {};
    if (!data.email?.trim()) errors.email = 'Email is required';
    if (data.email && !/^\S+@\S+\.\S+$/.test(data.email)) errors.email = 'Invalid email format';
    return errors;
  },

  async removeMemberFromWorkspace(workspaceId: string, userId: string): Promise<void> {
    return this.removeMember(workspaceId, userId);
  },

  async deleteWorkspace(workspaceId: string): Promise<void> {
    await apiClient.delete(`${API_BASE}/${workspaceId}`);
  },

  getRolePermissions(role: TeamRole): string[] {
    const permissions: Record<TeamRole, string[]> = {
      owner: ['all'],
      admin: ['manage_members', 'manage_settings', 'create_assessments', 'view_all'],
      researcher: ['create_assessments', 'view_all', 'manage_own_assessments'],
      viewer: ['view_all']
    };
    return permissions[role] || [];
  },

  formatLastActive(date?: string): string {
    if (!date) return 'Never';
    return new Date(date).toLocaleDateString();
  },

  getRoleColor(role: TeamRole): string {
    const colors: Record<TeamRole, string> = {
      owner: 'purple',
      admin: 'red',
      researcher: 'blue',
      viewer: 'gray'
    };
    return colors[role] || 'gray';
  },

  getRoleDisplayName(role: TeamRole): string {
    return role.charAt(0).toUpperCase() + role.slice(1);
  },

  getInvitationStatusColor(status: InvitationStatus): string {
    const colors: Record<InvitationStatus, string> = {
      pending: 'yellow',
      accepted: 'green',
      declined: 'red',
      expired: 'gray'
    };
    return colors[status] || 'gray';
  },

  getInvitationStatusDisplayName(status: InvitationStatus): string {
    return status.charAt(0).toUpperCase() + status.slice(1);
  },

  getActivityTypeIcon(type: ActivityType): string {
    const icons: Record<ActivityType, string> = {
      user_joined: '👋',
      user_left: '🏃',
      role_changed: '🔄',
      assessment_created: '🧪',
      assessment_shared: '🔗',
      report_generated: '📊',
      api_key_added: '🔑',
      api_key_removed: '🚫',
      workspace_created: '🏢',
      workspace_updated: '📝'
    };
    return icons[type] || '⚡';
  },

  getActivityTypeDisplayName(type: ActivityType): string {
    return type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  }
};
