/**
 * Team Workspaces & Collaboration Service
 *
 * Phase 3 enterprise feature for scalability:
 * - Multi-user team environments
 * - Role-based access control (Admin, Researcher, Viewer)
 * - Shared assessment history and API key pools
 * - Activity logging and audit trails
 */

import { toast } from 'sonner';
import { apiClient } from '@/lib/api/client';

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

export interface InviteUserRequest {
  email: string;
  role: TeamRole;
  message?: string;
}

export interface UpdateMemberRole {
  role: TeamRole;
}

export interface WorkspaceListResponse {
  workspaces: TeamWorkspace[];
  total: number;
}

export interface MembersListResponse {
  members: TeamMember[];
  invitations: TeamInvitation[];
  total_members: number;
  total_invitations: number;
}

export interface ActivityLogResponse {
  activities: ActivityLogEntry[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ActivityLogParams {
  page?: number;
  page_size?: number;
  activity_type?: ActivityType;
}

class WorkspaceService {
  private readonly basePath = '/workspaces';

  /**
   * Create a new team workspace
   */
  async createWorkspace(workspaceData: WorkspaceCreate): Promise<TeamWorkspace> {
    try {
      const response = await apiClient.post<TeamWorkspace>(`${this.basePath}/`, {
        name: workspaceData.name,
        description: workspaceData.description,
        settings: workspaceData.settings || {}
      });

      toast.success('Workspace created successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to create workspace:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create workspace');
      throw error;
    }
  }

  /**
   * List user's accessible workspaces
   */
  async listWorkspaces(): Promise<WorkspaceListResponse> {
    try {
      const response = await apiClient.get<WorkspaceListResponse>(`${this.basePath}/`);
      return response.data;
    } catch (error) {
      console.error('Failed to list workspaces:', error);
      toast.error('Failed to load workspaces');
      throw error;
    }
  }

  /**
   * Get workspace details
   */
  async getWorkspace(workspaceId: string): Promise<TeamWorkspace> {
    try {
      const response = await apiClient.get<TeamWorkspace>(`${this.basePath}/${workspaceId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get workspace:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to load workspace');
      throw error;
    }
  }

  /**
   * Update workspace settings
   */
  async updateWorkspace(workspaceId: string, updateData: WorkspaceUpdate): Promise<TeamWorkspace> {
    try {
      const response = await apiClient.patch<TeamWorkspace>(`${this.basePath}/${workspaceId}`, updateData);

      toast.success('Workspace updated successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to update workspace:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to update workspace');
      throw error;
    }
  }

  /**
   * List workspace members and invitations
   */
  async listWorkspaceMembers(workspaceId: string): Promise<MembersListResponse> {
    try {
      const response = await apiClient.get<MembersListResponse>(`${this.basePath}/${workspaceId}/members`);
      return response.data;
    } catch (error) {
      console.error('Failed to list workspace members:', error);
      toast.error('Failed to load workspace members');
      throw error;
    }
  }

  /**
   * Invite user to workspace
   */
  async inviteUserToWorkspace(workspaceId: string, inviteRequest: InviteUserRequest): Promise<void> {
    try {
      await apiClient.post(`${this.basePath}/${workspaceId}/invite`, {
        email: inviteRequest.email,
        role: inviteRequest.role,
        message: inviteRequest.message
      });

      toast.success(`Invitation sent to ${inviteRequest.email}`);
    } catch (error) {
      console.error('Failed to invite user:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to send invitation');
      throw error;
    }
  }

  /**
   * Update member role
   */
  async updateMemberRole(workspaceId: string, userId: string, roleUpdate: UpdateMemberRole): Promise<void> {
    try {
      await apiClient.patch(`${this.basePath}/${workspaceId}/members/${userId}/role`, roleUpdate);

      toast.success('Member role updated successfully');
    } catch (error) {
      console.error('Failed to update member role:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to update member role');
      throw error;
    }
  }

  /**
   * Remove member from workspace
   */
  async removeMemberFromWorkspace(workspaceId: string, userId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.basePath}/${workspaceId}/members/${userId}`);

      toast.success('Member removed from workspace');
    } catch (error) {
      console.error('Failed to remove member:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to remove member');
      throw error;
    }
  }

  /**
   * Get workspace activity log
   */
  async getWorkspaceActivityLog(workspaceId: string, params?: ActivityLogParams): Promise<ActivityLogResponse> {
    try {
      const response = await apiClient.get<ActivityLogResponse>(`${this.basePath}/${workspaceId}/activity`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          activity_type: params?.activity_type,
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to get activity log:', error);
      toast.error('Failed to load activity log');
      throw error;
    }
  }

  /**
   * Delete workspace
   */
  async deleteWorkspace(workspaceId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.basePath}/${workspaceId}`);

      toast.success('Workspace deleted successfully');
    } catch (error) {
      console.error('Failed to delete workspace:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to delete workspace');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for team role
   */
  getRoleDisplayName(role: TeamRole): string {
    const displayNames: Record<TeamRole, string> = {
      owner: 'Owner',
      admin: 'Admin',
      researcher: 'Researcher',
      viewer: 'Viewer'
    };
    return displayNames[role];
  }

  /**
   * Get color for team role
   */
  getRoleColor(role: TeamRole): string {
    const colors: Record<TeamRole, string> = {
      owner: 'purple',
      admin: 'red',
      researcher: 'blue',
      viewer: 'gray'
    };
    return colors[role];
  }

  /**
   * Get role permissions description
   */
  getRolePermissions(role: TeamRole): string[] {
    const permissions: Record<TeamRole, string[]> = {
      owner: [
        'Full workspace control',
        'Delete workspace',
        'Manage all members',
        'All admin permissions'
      ],
      admin: [
        'Invite and remove members',
        'Change member roles',
        'Manage workspace settings',
        'View activity logs',
        'All researcher permissions'
      ],
      researcher: [
        'Create and run assessments',
        'Generate reports',
        'Share sessions',
        'Manage API keys',
        'All viewer permissions'
      ],
      viewer: [
        'View assessments and reports',
        'Access shared sessions',
        'View workspace activity'
      ]
    };
    return permissions[role];
  }

  /**
   * Get display name for invitation status
   */
  getInvitationStatusDisplayName(status: InvitationStatus): string {
    const displayNames: Record<InvitationStatus, string> = {
      pending: 'Pending',
      accepted: 'Accepted',
      declined: 'Declined',
      expired: 'Expired'
    };
    return displayNames[status];
  }

  /**
   * Get color for invitation status
   */
  getInvitationStatusColor(status: InvitationStatus): string {
    const colors: Record<InvitationStatus, string> = {
      pending: 'yellow',
      accepted: 'green',
      declined: 'red',
      expired: 'gray'
    };
    return colors[status];
  }

  /**
   * Get display name for activity type
   */
  getActivityTypeDisplayName(type: ActivityType): string {
    const displayNames: Record<ActivityType, string> = {
      user_joined: 'User Joined',
      user_left: 'User Left',
      role_changed: 'Role Changed',
      assessment_created: 'Assessment Created',
      assessment_shared: 'Assessment Shared',
      report_generated: 'Report Generated',
      api_key_added: 'API Key Added',
      api_key_removed: 'API Key Removed',
      workspace_created: 'Workspace Created',
      workspace_updated: 'Workspace Updated'
    };
    return displayNames[type];
  }

  /**
   * Get icon for activity type
   */
  getActivityTypeIcon(type: ActivityType): string {
    const icons: Record<ActivityType, string> = {
      user_joined: '👋',
      user_left: '👋',
      role_changed: '🔄',
      assessment_created: '🎯',
      assessment_shared: '📤',
      report_generated: '📊',
      api_key_added: '🔑',
      api_key_removed: '🔑',
      workspace_created: '🏗️',
      workspace_updated: '⚙️'
    };
    return icons[type];
  }

  /**
   * Check if user can perform action based on role
   */
  canPerformAction(userRole: TeamRole, requiredRole: TeamRole): boolean {
    const roleHierarchy: Record<TeamRole, number> = {
      viewer: 1,
      researcher: 2,
      admin: 3,
      owner: 4
    };

    return roleHierarchy[userRole] >= roleHierarchy[requiredRole];
  }

  /**
   * Format last active time
   */
  formatLastActive(lastActive?: string): string {
    if (!lastActive) return 'Never';

    const date = new Date(lastActive);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMinutes = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMinutes / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString();
  }

  /**
   * Validate workspace creation data
   */
  validateWorkspaceCreate(data: WorkspaceCreate): string[] {
    const errors: string[] = [];

    if (!data.name || data.name.trim().length === 0) {
      errors.push('Workspace name is required');
    }

    if (data.name && data.name.length > 100) {
      errors.push('Workspace name must be less than 100 characters');
    }

    if (data.description && data.description.length > 500) {
      errors.push('Description must be less than 500 characters');
    }

    return errors;
  }

  /**
   * Validate invitation request
   */
  validateInviteRequest(data: InviteUserRequest): string[] {
    const errors: string[] = [];

    if (!data.email || data.email.trim().length === 0) {
      errors.push('Email address is required');
    }

    if (data.email && !data.email.includes('@')) {
      errors.push('Valid email address is required');
    }

    if (!data.role) {
      errors.push('Role is required');
    }

    return errors;
  }

  /**
   * Create default workspace settings
   */
  createDefaultSettings(): Record<string, any> {
    return {
      auto_share_assessments: false,
      require_approval_for_invites: true,
      allow_guest_access: false,
      default_member_role: 'viewer',
      session_timeout_hours: 8,
      audit_log_retention_days: 90
    };
  }
}

// Export singleton instance
export const workspaceService = new WorkspaceService();