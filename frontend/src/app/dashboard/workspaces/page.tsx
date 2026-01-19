/**
 * Team Workspaces & Collaboration Interface
 *
 * Phase 3 enterprise feature for scalability:
 * - Multi-user team environments
 * - Role-based access control (Admin, Researcher, Viewer)
 * - Shared assessment history and API key pools
 * - Activity logging and audit trails
 */

"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Users,
  UserPlus,
  Settings,
  Shield,
  Activity,
  Crown,
  Mail,
  Clock,
  Plus,
  Trash2,
  Edit,
  Eye,
  RefreshCw,
  Search,
  Filter,
  Calendar,
  ChevronRight,
  AlertTriangle,
  CheckCircle,
  Building2,
  Key,
  FileText,
  BarChart3
} from 'lucide-react';
import { toast } from 'sonner';

// Import services
import {
  workspaceService,
  TeamWorkspace,
  TeamMember,
  TeamInvitation,
  ActivityLogEntry,
  WorkspaceCreate,
  WorkspaceUpdate,
  InviteUserRequest,
  UpdateMemberRole,
  WorkspaceListResponse,
  MembersListResponse,
  ActivityLogResponse,
  TeamRole,
  InvitationStatus,
  ActivityType
} from '@/lib/api/services/workspaces';

export default function WorkspacesPage() {
  // Data state
  const [workspaces, setWorkspaces] = useState<WorkspaceListResponse | null>(null);
  const [selectedWorkspace, setSelectedWorkspace] = useState<TeamWorkspace | null>(null);
  const [workspaceMembers, setWorkspaceMembers] = useState<MembersListResponse | null>(null);
  const [activityLog, setActivityLog] = useState<ActivityLogResponse | null>(null);

  // Form state
  const [newWorkspaceData, setNewWorkspaceData] = useState<WorkspaceCreate>({
    name: '',
    description: ''
  });
  const [editWorkspaceData, setEditWorkspaceData] = useState<WorkspaceUpdate>({});
  const [inviteData, setInviteData] = useState<InviteUserRequest>({
    email: '',
    role: 'viewer'
  });

  // UI state
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showInviteDialog, setShowInviteDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showMemberDialog, setShowMemberDialog] = useState(false);
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [memberToRemove, setMemberToRemove] = useState<TeamMember | null>(null);
  const [workspaceToDelete, setWorkspaceToDelete] = useState<TeamWorkspace | null>(null);

  useEffect(() => {
    loadWorkspaces();
  }, []);

  const loadWorkspaces = useCallback(async () => {
    try {
      setLoading(true);
      const data = await workspaceService.listWorkspaces();
      setWorkspaces(data);

      // Auto-select first workspace if available
      if (data.workspaces.length > 0 && !selectedWorkspace) {
        setSelectedWorkspace(data.workspaces[0]);
        loadWorkspaceData(data.workspaces[0].workspace_id);
      }
    } catch (error) {
      // Error already handled in service
    } finally {
      setLoading(false);
    }
  }, [selectedWorkspace]);

  const loadWorkspaceData = useCallback(async (workspaceId: string) => {
    try {
      const [membersData, activityData] = await Promise.all([
        workspaceService.listWorkspaceMembers(workspaceId),
        workspaceService.getWorkspaceActivityLog(workspaceId, { page: 1, page_size: 20 })
      ]);

      setWorkspaceMembers(membersData);
      setActivityLog(activityData);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const handleCreateWorkspace = useCallback(async () => {
    const errors = workspaceService.validateWorkspaceCreate(newWorkspaceData);
    if (Object.keys(errors).length > 0) {
      toast.error(Object.values(errors).join(', '));
      return;
    }

    try {
      setCreating(true);
      const workspace = await workspaceService.createWorkspace({
        ...newWorkspaceData,
        settings: workspaceService.createDefaultSettings()
      });

      // Refresh workspaces list
      await loadWorkspaces();

      // Select the new workspace
      setSelectedWorkspace(workspace);
      await loadWorkspaceData(workspace.workspace_id);

      // Reset form
      setNewWorkspaceData({ name: '', description: '' });
      setShowCreateDialog(false);

      toast.success(`Workspace "${workspace.name}" created successfully!`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [newWorkspaceData, loadWorkspaces]);

  const handleEditWorkspace = useCallback(async () => {
    if (!selectedWorkspace) return;

    try {
      const updatedWorkspace = await workspaceService.updateWorkspace(
        selectedWorkspace.workspace_id,
        editWorkspaceData
      );

      setSelectedWorkspace(updatedWorkspace);
      await loadWorkspaces();
      setShowEditDialog(false);
      setEditWorkspaceData({});
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedWorkspace, editWorkspaceData, loadWorkspaces]);

  const handleInviteUser = useCallback(async () => {
    if (!selectedWorkspace) return;

    const errors = workspaceService.validateInviteRequest(inviteData);
    if (Object.keys(errors).length > 0) {
      toast.error(Object.values(errors).join(', '));
      return;
    }

    try {
      await workspaceService.inviteUserToWorkspace(selectedWorkspace.workspace_id, inviteData);

      // Refresh members list
      await loadWorkspaceData(selectedWorkspace.workspace_id);

      // Reset form
      setInviteData({ email: '', role: 'viewer' });
      setShowInviteDialog(false);
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedWorkspace, inviteData, loadWorkspaceData]);

  const handleUpdateMemberRole = useCallback(async (member: TeamMember, newRole: TeamRole) => {
    if (!selectedWorkspace) return;

    try {
      await workspaceService.updateMemberRole(selectedWorkspace.workspace_id, member.user_id, { role: newRole });

      // Refresh members list
      await loadWorkspaceData(selectedWorkspace.workspace_id);
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedWorkspace, loadWorkspaceData]);

  const handleRemoveMember = useCallback(async () => {
    if (!selectedWorkspace || !memberToRemove) return;

    try {
      await workspaceService.removeMemberFromWorkspace(
        selectedWorkspace.workspace_id,
        memberToRemove.user_id
      );

      // Refresh members list
      await loadWorkspaceData(selectedWorkspace.workspace_id);

      setMemberToRemove(null);
      setShowMemberDialog(false);
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedWorkspace, memberToRemove, loadWorkspaceData]);

  const handleDeleteWorkspace = useCallback(async () => {
    if (!workspaceToDelete) return;

    try {
      await workspaceService.deleteWorkspace(workspaceToDelete.workspace_id);

      // Refresh workspaces list
      await loadWorkspaces();

      // Clear selection if deleted workspace was selected
      if (selectedWorkspace?.workspace_id === workspaceToDelete.workspace_id) {
        setSelectedWorkspace(null);
        setWorkspaceMembers(null);
        setActivityLog(null);
      }

      setWorkspaceToDelete(null);
      setShowDeleteDialog(false);
    } catch (error) {
      // Error already handled in service
    }
  }, [workspaceToDelete, loadWorkspaces, selectedWorkspace]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Team Workspaces...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load your team collaboration data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Team Workspaces & Collaboration
        </h1>
        <p className="text-muted-foreground text-lg">
          Manage multi-user team environments with role-based access control, shared resources, and activity tracking.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Sidebar: Workspace List */}
        <div className="lg:col-span-1">
          <WorkspaceListView
            workspaces={workspaces}
            selectedWorkspace={selectedWorkspace}
            onSelectWorkspace={(workspace) => {
              setSelectedWorkspace(workspace);
              loadWorkspaceData(workspace.workspace_id);
            }}
            onCreateNew={() => setShowCreateDialog(true)}
            onRefresh={loadWorkspaces}
          />
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          {selectedWorkspace ? (
            <Tabs defaultValue="members" className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div>
                    <h2 className="text-2xl font-bold">{selectedWorkspace.name}</h2>
                    <p className="text-muted-foreground">{selectedWorkspace.description}</p>
                  </div>
                  <Badge variant="outline">
                    {selectedWorkspace.member_count} member{selectedWorkspace.member_count !== 1 ? 's' : ''}
                  </Badge>
                </div>

                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setEditWorkspaceData({
                        name: selectedWorkspace.name,
                        description: selectedWorkspace.description
                      });
                      setShowEditDialog(true);
                    }}
                  >
                    <Settings className="h-4 w-4 mr-2" />
                    Settings
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setWorkspaceToDelete(selectedWorkspace);
                      setShowDeleteDialog(true);
                    }}
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </Button>
                </div>
              </div>

              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="members">Members</TabsTrigger>
                <TabsTrigger value="activity">Activity</TabsTrigger>
                <TabsTrigger value="settings">Settings</TabsTrigger>
              </TabsList>

              <TabsContent value="members" className="space-y-6">
                <WorkspaceMembersView
                  workspaceMembers={workspaceMembers}
                  selectedWorkspace={selectedWorkspace}
                  onInviteUser={() => setShowInviteDialog(true)}
                  onUpdateRole={handleUpdateMemberRole}
                  onRemoveMember={(member) => {
                    setMemberToRemove(member);
                    setShowMemberDialog(true);
                  }}
                  onRefresh={() => loadWorkspaceData(selectedWorkspace.workspace_id)}
                />
              </TabsContent>

              <TabsContent value="activity" className="space-y-6">
                <WorkspaceActivityView
                  activityLog={activityLog}
                  onRefresh={() => loadWorkspaceData(selectedWorkspace.workspace_id)}
                />
              </TabsContent>

              <TabsContent value="settings" className="space-y-6">
                <WorkspaceSettingsView
                  workspace={selectedWorkspace}
                  onUpdate={(updateData) => {
                    setEditWorkspaceData(updateData);
                    handleEditWorkspace();
                  }}
                />
              </TabsContent>
            </Tabs>
          ) : (
            <Card>
              <CardContent className="text-center py-12">
                <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  No Workspace Selected
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  Select a workspace from the sidebar to view team details and manage collaboration.
                </p>
                {(!workspaces || workspaces.total === 0) && (
                  <Button onClick={() => setShowCreateDialog(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Create First Workspace
                  </Button>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Create Workspace Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Create New Workspace</DialogTitle>
            <DialogDescription>
              Set up a new team workspace for collaborative AI security testing.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Workspace Name</Label>
              <Input
                value={newWorkspaceData.name}
                onChange={(e) => setNewWorkspaceData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Security Team Workspace"
              />
            </div>

            <div className="space-y-2">
              <Label>Description (Optional)</Label>
              <Textarea
                value={newWorkspaceData.description || ''}
                onChange={(e) => setNewWorkspaceData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Workspace for our security assessment activities..."
                className="min-h-20"
              />
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateWorkspace} disabled={creating}>
                {creating ? (
                  <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Plus className="h-4 w-4 mr-2" />
                )}
                Create Workspace
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Workspace Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Workspace</DialogTitle>
            <DialogDescription>
              Update workspace information and settings.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Workspace Name</Label>
              <Input
                value={editWorkspaceData.name || ''}
                onChange={(e) => setEditWorkspaceData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Security Team Workspace"
              />
            </div>

            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                value={editWorkspaceData.description || ''}
                onChange={(e) => setEditWorkspaceData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Workspace description..."
                className="min-h-20"
              />
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowEditDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleEditWorkspace}>
                <Settings className="h-4 w-4 mr-2" />
                Update Workspace
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Invite User Dialog */}
      <Dialog open={showInviteDialog} onOpenChange={setShowInviteDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Invite Team Member</DialogTitle>
            <DialogDescription>
              Invite a new member to &quot;{selectedWorkspace?.name}&quot; workspace.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Email Address</Label>
              <Input
                type="email"
                value={inviteData.email}
                onChange={(e) => setInviteData(prev => ({ ...prev, email: e.target.value }))}
                placeholder="colleague@example.com"
              />
            </div>

            <div className="space-y-2">
              <Label>Role</Label>
              <Select
                value={inviteData.role}
                onValueChange={(value: TeamRole) => setInviteData(prev => ({ ...prev, role: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="viewer">Viewer</SelectItem>
                  <SelectItem value="researcher">Researcher</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                </SelectContent>
              </Select>
              <div className="text-xs text-muted-foreground">
                <strong>Permissions:</strong>
                <ul className="list-disc list-inside mt-1">
                  {workspaceService.getRolePermissions(inviteData.role).slice(0, 2).map((permission: string, index: number) => (
                    <li key={index}>{permission}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Message (Optional)</Label>
              <Textarea
                value={inviteData.message || ''}
                onChange={(e) => setInviteData(prev => ({ ...prev, message: e.target.value }))}
                placeholder="Join our security team workspace..."
                className="min-h-16"
              />
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowInviteDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleInviteUser}>
                <UserPlus className="h-4 w-4 mr-2" />
                Send Invitation
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Remove Member Confirmation */}
      <AlertDialog open={showMemberDialog} onOpenChange={setShowMemberDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove Team Member</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to remove {memberToRemove?.username} from this workspace?
              They will lose access to all shared resources and assessments.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleRemoveMember} className="bg-red-600 hover:bg-red-700">
              Remove Member
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete Workspace Confirmation */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Workspace</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{workspaceToDelete?.name}&quot;? This action cannot be undone
              and will remove all shared data, assessments, and team access.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteWorkspace} className="bg-red-600 hover:bg-red-700">
              Delete Workspace
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// Workspace List View Component
function WorkspaceListView({
  workspaces,
  selectedWorkspace,
  onSelectWorkspace,
  onCreateNew,
  onRefresh
}: {
  workspaces: WorkspaceListResponse | null;
  selectedWorkspace: TeamWorkspace | null;
  onSelectWorkspace: (workspace: TeamWorkspace) => void;
  onCreateNew: () => void;
  onRefresh: () => void;
}) {
  if (!workspaces) {
    return (
      <Card>
        <CardContent className="p-4">
          <div className="animate-pulse space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="h-12 bg-gray-200 rounded"></div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5" />
            Workspaces
          </CardTitle>
          <Button size="sm" onClick={onRefresh} variant="outline">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>
          {workspaces.total} workspace{workspaces.total !== 1 ? 's' : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-96">
          {workspaces.total === 0 ? (
            <div className="text-center py-8 px-4">
              <Users className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <h4 className="font-semibold mb-2">No Workspaces</h4>
              <p className="text-sm text-muted-foreground mb-3">
                Create your first workspace to start collaborating with your team.
              </p>
              <Button size="sm" onClick={onCreateNew}>
                <Plus className="h-4 w-4 mr-1" />
                Create Workspace
              </Button>
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {workspaces.workspaces.map((workspace) => (
                <button
                  key={workspace.workspace_id}
                  onClick={() => onSelectWorkspace(workspace)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    selectedWorkspace?.workspace_id === workspace.workspace_id
                      ? 'bg-primary/10 border border-primary/20'
                      : 'hover:bg-muted'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="font-medium truncate">{workspace.name}</h4>
                    <Badge variant="secondary" className="text-xs">
                      {workspace.member_count}
                    </Badge>
                  </div>
                  {workspace.description && (
                    <p className="text-sm text-muted-foreground truncate">
                      {workspace.description}
                    </p>
                  )}
                  <div className="text-xs text-muted-foreground mt-1">
                    Updated {new Date(workspace.updated_at).toLocaleDateString()}
                  </div>
                </button>
              ))}

              <Button
                variant="outline"
                className="w-full mt-2"
                onClick={onCreateNew}
              >
                <Plus className="h-4 w-4 mr-2" />
                New Workspace
              </Button>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

// Workspace Members View Component
function WorkspaceMembersView({
  workspaceMembers,
  selectedWorkspace,
  onInviteUser,
  onUpdateRole,
  onRemoveMember,
  onRefresh
}: {
  workspaceMembers: MembersListResponse | null;
  selectedWorkspace: TeamWorkspace;
  onInviteUser: () => void;
  onUpdateRole: (member: TeamMember, role: TeamRole) => void;
  onRemoveMember: (member: TeamMember) => void;
  onRefresh: () => void;
}) {
  if (!workspaceMembers) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="flex items-center space-x-4">
                <div className="w-10 h-10 bg-gray-200 rounded-full"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                  <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Team Members</h3>
          <p className="text-sm text-muted-foreground">
            {workspaceMembers.total_members} active member{workspaceMembers.total_members !== 1 ? 's' : ''}
            {(workspaceMembers.total_invitations ?? 0) > 0 && (
              <span> • {workspaceMembers.total_invitations} pending invitation{workspaceMembers.total_invitations !== 1 ? 's' : ''}</span>
            )}
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={onRefresh} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={onInviteUser} size="sm">
            <UserPlus className="h-4 w-4 mr-2" />
            Invite Member
          </Button>
        </div>
      </div>

      {/* Active Members */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Active Members</CardTitle>
        </CardHeader>
        <CardContent>
          {workspaceMembers.members.length === 0 ? (
            <div className="text-center py-6">
              <Users className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <p className="text-muted-foreground">No active members</p>
            </div>
          ) : (
            <div className="space-y-3">
              {workspaceMembers.members.map((member) => (
                <div key={member.user_id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                      {member.role === 'owner' ? (
                        <Crown className="h-5 w-5 text-yellow-600" />
                      ) : (
                        <span className="font-semibold text-sm">{member.username.charAt(0).toUpperCase()}</span>
                      )}
                    </div>
                    <div>
                      <h4 className="font-medium flex items-center gap-2">
                        {member.username}
                        {member.role === 'owner' && (
                          <Badge variant="secondary" className="text-xs bg-yellow-100 text-yellow-800">
                            Owner
                          </Badge>
                        )}
                      </h4>
                      <p className="text-sm text-muted-foreground">{member.email}</p>
                      <p className="text-xs text-muted-foreground">
                        Last active: {workspaceService.formatLastActive(member.last_active)}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className={`bg-${workspaceService.getRoleColor(member.role)}-50`}>
                      {workspaceService.getRoleDisplayName(member.role)}
                    </Badge>

                    {member.role !== 'owner' && (
                      <div className="flex gap-1">
                        <Select
                          value={member.role}
                          onValueChange={(value: TeamRole) => onUpdateRole(member, value)}
                        >
                          <SelectTrigger className="w-32 h-8">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="viewer">Viewer</SelectItem>
                            <SelectItem value="researcher">Researcher</SelectItem>
                            <SelectItem value="admin">Admin</SelectItem>
                          </SelectContent>
                        </Select>

                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onRemoveMember(member)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pending Invitations */}
      {workspaceMembers.invitations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Pending Invitations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {workspaceMembers.invitations.map((invitation) => (
                <div key={invitation.invitation_id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center">
                      <Mail className="h-5 w-5 text-gray-500" />
                    </div>
                    <div>
                      <h4 className="font-medium">{invitation.invited_email}</h4>
                      <p className="text-sm text-muted-foreground">
                        Invited by {invitation.invited_by_name} • {new Date(invitation.created_at).toLocaleDateString()}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Expires: {new Date(invitation.expires_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className={`bg-${workspaceService.getRoleColor(invitation.role)}-50`}>
                      {workspaceService.getRoleDisplayName(invitation.role)}
                    </Badge>
                    <Badge variant="outline" className={`bg-${workspaceService.getInvitationStatusColor(invitation.status)}-50`}>
                      {workspaceService.getInvitationStatusDisplayName(invitation.status)}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Workspace Activity View Component
function WorkspaceActivityView({
  activityLog,
  onRefresh
}: {
  activityLog: ActivityLogResponse | null;
  onRefresh: () => void;
}) {
  if (!activityLog) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-center space-x-4">
                <div className="w-8 h-8 bg-gray-200 rounded"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Activity Log</h3>
          <p className="text-sm text-muted-foreground">
            Recent workspace activity and audit trail
          </p>
        </div>
        <Button onClick={onRefresh} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <Card>
        <CardContent className="p-0">
          <ScrollArea className="h-96">
            {activityLog.activities.length === 0 ? (
              <div className="text-center py-8">
                <Activity className="h-8 w-8 text-gray-400 mx-auto mb-3" />
                <p className="text-muted-foreground">No activity recorded yet</p>
              </div>
            ) : (
              <div className="divide-y">
                {activityLog.activities.map((activity) => (
                  <div key={activity.activity_id} className="p-4 hover:bg-muted/50">
                    <div className="flex items-start gap-3">
                      <div className="text-lg mt-0.5">
                        {workspaceService.getActivityTypeIcon(activity.activity_type)}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium">{activity.description}</p>
                        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                          <span>{activity.username}</span>
                          <span>•</span>
                          <span>{new Date(activity.timestamp).toLocaleString()}</span>
                          <span>•</span>
                          <Badge variant="secondary" className="text-xs">
                            {workspaceService.getActivityTypeDisplayName(activity.activity_type)}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

// Workspace Settings View Component
function WorkspaceSettingsView({
  workspace,
  onUpdate
}: {
  workspace: TeamWorkspace;
  onUpdate: (updateData: WorkspaceUpdate) => void;
}) {
  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Workspace Settings</h3>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">General Settings</CardTitle>
          <CardDescription>
            Configure workspace behavior and default permissions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <div>
                <Label className="font-medium">Auto-share Assessments</Label>
                <p className="text-xs text-muted-foreground">Automatically share new assessments with all members</p>
              </div>
              {/* Settings toggles would be implemented here */}
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label className="font-medium">Require Invite Approval</Label>
                <p className="text-xs text-muted-foreground">Admin approval required for new invitations</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Security & Compliance</CardTitle>
          <CardDescription>
            Audit and security configuration options
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <Label className="font-medium">Session Timeout</Label>
              <p className="text-muted-foreground">8 hours</p>
            </div>
            <div>
              <Label className="font-medium">Audit Retention</Label>
              <p className="text-muted-foreground">90 days</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
