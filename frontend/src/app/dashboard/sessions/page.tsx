/**
 * Attack Session Recording, Replay & Sharing Interface
 *
 * Phase 2 feature for competitive differentiation:
 * - Complete session saving with full context
 * - Reproducible test execution
 * - Secure sharing with team members
 * - Import/export capabilities
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
import {
  Play,
  Pause,
  Square,
  Share2,
  Download,
  Upload,
  RefreshCw,
  Search,
  Filter,
  Plus,
  Settings,
  Trash2,
  Copy,
  Edit,
  Eye,
  Users,
  Calendar,
  Clock,
  Target,
  Zap,
  CheckCircle,
  AlertTriangle,
  FileText,
  BarChart3
} from 'lucide-react';
import { toast } from 'sonner';

// Import services
import {
  attackSessionService,
  AttackSession,
  SessionCreate,
  SessionUpdate,
  SessionShare,
  SessionReplay,
  SessionListResponse,
  SessionListParams,
  SessionStatus,
  SharePermission
} from '@/lib/api/services/attack-sessions';

export default function AttackSessionsPage() {
  // Data state
  const [sessions, setSessions] = useState<SessionListResponse | null>(null);
  const [selectedSession, setSelectedSession] = useState<AttackSession | null>(null);

  // Form state
  const [newSessionData, setNewSessionData] = useState<SessionCreate>({
    name: '',
    description: '',
    target_provider: 'openai',
    target_model: 'gpt-4',
    original_prompt: ''
  });
  const [shareData, setShareData] = useState<SessionShare>({
    user_ids: [],
    permission: 'view'
  });
  const [replayData, setReplayData] = useState<SessionReplay>({
    session_id: '',
    replay_name: ''
  });

  // Filter and pagination state
  const [filters, setFilters] = useState<SessionListParams>({
    page: 1,
    page_size: 20
  });

  // UI state
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [showReplayDialog, setShowReplayDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showDetailsDialog, setShowDetailsDialog] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, [filters]);

  const loadSessions = useCallback(async () => {
    try {
      setLoading(true);
      const data = await attackSessionService.listSessions(filters);
      setSessions(data);
    } catch (error) {
      // Error already handled in service
    } finally {
      setLoading(false);
    }
  }, [filters]);

  const handleCreateSession = useCallback(async () => {
    const errors = attackSessionService.validateSessionCreate(newSessionData);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setCreating(true);
      const session = await attackSessionService.createSession(newSessionData);

      // Refresh sessions list
      await loadSessions();

      // Reset form
      setNewSessionData({
        name: '',
        description: '',
        target_provider: 'openai',
        target_model: 'gpt-4',
        original_prompt: ''
      });
      setShowCreateDialog(false);

      toast.success(`Session "${session.name}" created successfully!`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [newSessionData, loadSessions]);

  const handleShareSession = useCallback(async () => {
    if (!selectedSession) return;

    if (shareData.user_ids.length === 0) {
      toast.error('Please enter at least one user ID');
      return;
    }

    try {
      await attackSessionService.shareSession(selectedSession.session_id, shareData);

      // Refresh sessions list
      await loadSessions();

      // Reset form
      setShareData({
        user_ids: [],
        permission: 'view'
      });
      setShowShareDialog(false);
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedSession, shareData, loadSessions]);

  const handleReplaySession = useCallback(async () => {
    if (!selectedSession) return;

    try {
      const replaySession = await attackSessionService.replaySession(
        selectedSession.session_id,
        {
          ...replayData,
          session_id: selectedSession.session_id
        }
      );

      // Refresh sessions list
      await loadSessions();

      // Reset form
      setReplayData({
        session_id: '',
        replay_name: ''
      });
      setShowReplayDialog(false);

      toast.success(`Replay session "${replaySession.name}" created!`);
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedSession, replayData, loadSessions]);

  const handleDeleteSession = useCallback(async () => {
    if (!sessionToDelete) return;

    try {
      await attackSessionService.deleteSession(sessionToDelete);

      // Refresh sessions list
      await loadSessions();

      setSessionToDelete(null);
      setShowDeleteDialog(false);
    } catch (error) {
      // Error already handled in service
    }
  }, [sessionToDelete, loadSessions]);

  const handleExportSession = useCallback(async (sessionId: string, format: string = 'json') => {
    try {
      const exportData = await attackSessionService.exportSession(sessionId, format);

      // Create download link
      const dataStr = JSON.stringify(exportData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);

      const link = document.createElement('a');
      link.href = url;
      link.download = `session_${sessionId}_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      URL.revokeObjectURL(url);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const handleFilterChange = useCallback((key: keyof SessionListParams, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value, page: 1 })); // Reset to page 1 when filtering
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Attack Sessions...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load your session data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Attack Session Management
        </h1>
        <p className="text-muted-foreground text-lg">
          Record, replay, and share complete attack sessions with full reproducibility and team collaboration features.
        </p>
      </div>

      <Tabs defaultValue="sessions" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="sessions">Session History</TabsTrigger>
          <TabsTrigger value="shared">Shared Sessions</TabsTrigger>
        </TabsList>

        <TabsContent value="sessions" className="space-y-6">
          <SessionManagementView
            sessions={sessions}
            filters={filters}
            onFilterChange={handleFilterChange}
            onRefresh={loadSessions}
            onCreateNew={() => setShowCreateDialog(true)}
            onViewDetails={(session) => {
              setSelectedSession(session);
              setShowDetailsDialog(true);
            }}
            onShare={(session) => {
              setSelectedSession(session);
              setShowShareDialog(true);
            }}
            onReplay={(session) => {
              setSelectedSession(session);
              setReplayData(prev => ({ ...prev, session_id: session.session_id }));
              setShowReplayDialog(true);
            }}
            onExport={handleExportSession}
            onDelete={(sessionId) => {
              setSessionToDelete(sessionId);
              setShowDeleteDialog(true);
            }}
          />
        </TabsContent>

        <TabsContent value="shared" className="space-y-6">
          <SharedSessionsView
            filters={{ ...filters, shared_only: true }}
            onFilterChange={handleFilterChange}
            onRefresh={loadSessions}
            onViewDetails={(session) => {
              setSelectedSession(session);
              setShowDetailsDialog(true);
            }}
            onReplay={(session) => {
              setSelectedSession(session);
              setReplayData(prev => ({ ...prev, session_id: session.session_id }));
              setShowReplayDialog(true);
            }}
            onExport={handleExportSession}
          />
        </TabsContent>
      </Tabs>

      {/* Create Session Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Create New Attack Session</DialogTitle>
            <DialogDescription>
              Set up a new session to record your attack attempts for later replay and analysis.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Session Name</Label>
              <Input
                value={newSessionData.name}
                onChange={(e) => setNewSessionData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Attack Session Name"
              />
            </div>

            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                value={newSessionData.description || ''}
                onChange={(e) => setNewSessionData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Session description..."
                className="min-h-20"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Target Provider</Label>
                <Select
                  value={newSessionData.target_provider}
                  onValueChange={(value) => setNewSessionData(prev => ({ ...prev, target_provider: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                    <SelectItem value="google">Google</SelectItem>
                    <SelectItem value="deepseek">DeepSeek</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Target Model</Label>
                <Input
                  value={newSessionData.target_model}
                  onChange={(e) => setNewSessionData(prev => ({ ...prev, target_model: e.target.value }))}
                  placeholder="gpt-4"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Original Prompt</Label>
              <Textarea
                value={newSessionData.original_prompt}
                onChange={(e) => setNewSessionData(prev => ({ ...prev, original_prompt: e.target.value }))}
                placeholder="Enter the original prompt to test..."
                className="min-h-24"
              />
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateSession} disabled={creating}>
                {creating ? (
                  <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Plus className="h-4 w-4 mr-2" />
                )}
                Create Session
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Share Session Dialog */}
      <Dialog open={showShareDialog} onOpenChange={setShowShareDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Share Session</DialogTitle>
            <DialogDescription>
              Share &quot;{selectedSession?.name}&quot; with team members with specific permissions.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>User IDs (comma-separated)</Label>
              <Input
                value={shareData.user_ids.join(', ')}
                onChange={(e) => setShareData(prev => ({
                  ...prev,
                  user_ids: e.target.value.split(',').map(id => id.trim()).filter(Boolean)
                }))}
                placeholder="user1@example.com, user2@example.com"
              />
            </div>

            <div className="space-y-2">
              <Label>Permission Level</Label>
              <Select
                value={shareData.permission}
                onValueChange={(value: SharePermission) => setShareData(prev => ({ ...prev, permission: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="view">View Only</SelectItem>
                  <SelectItem value="replay">View & Replay</SelectItem>
                  <SelectItem value="edit">Edit & Replay</SelectItem>
                  <SelectItem value="admin">Full Admin</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {attackSessionService.getPermissionDisplayName(shareData.permission)}
              </p>
            </div>

            <div className="space-y-2">
              <Label>Share Message (Optional)</Label>
              <Textarea
                value={shareData.message || ''}
                onChange={(e) => setShareData(prev => ({ ...prev, message: e.target.value }))}
                placeholder="Message to include with the share..."
                className="min-h-20"
              />
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowShareDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleShareSession}>
                <Share2 className="h-4 w-4 mr-2" />
                Share Session
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Replay Session Dialog */}
      <Dialog open={showReplayDialog} onOpenChange={setShowReplayDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Replay Session</DialogTitle>
            <DialogDescription>
              Create a replay of &quot;{selectedSession?.name}&quot; with optional modifications.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Replay Name</Label>
              <Input
                value={replayData.replay_name || ''}
                onChange={(e) => setReplayData(prev => ({ ...prev, replay_name: e.target.value }))}
                placeholder={`Replay of ${selectedSession?.name || 'session'}`}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Start From Step</Label>
                <Input
                  type="number"
                  value={replayData.start_from_step || 0}
                  onChange={(e) => setReplayData(prev => ({ ...prev, start_from_step: parseInt(e.target.value) || 0 }))}
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <Label>Stop At Step (Optional)</Label>
                <Input
                  type="number"
                  value={replayData.stop_at_step || ''}
                  onChange={(e) => setReplayData(prev => ({ ...prev, stop_at_step: e.target.value ? parseInt(e.target.value) : undefined }))}
                  min="1"
                />
              </div>
            </div>

            <div className="bg-muted rounded-lg p-3 text-sm">
              <div className="font-medium mb-1">Original Session Info</div>
              <div className="text-muted-foreground space-y-1">
                <div>Total Steps: {selectedSession?.total_steps || 0}</div>
                <div>Success Rate: {selectedSession ? attackSessionService.calculateSuccessRate(selectedSession).toFixed(1) : 0}%</div>
                <div>Duration: {selectedSession ? attackSessionService.formatExecutionTime(selectedSession.total_execution_time) : 'N/A'}</div>
              </div>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowReplayDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleReplaySession}>
                <Play className="h-4 w-4 mr-2" />
                Start Replay
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Session Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{selectedSession?.name}</DialogTitle>
            <DialogDescription>
              Complete session details and step-by-step execution log
            </DialogDescription>
          </DialogHeader>

          {selectedSession && <SessionDetailsView session={selectedSession} />}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Session</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this session? This action cannot be undone and will remove all session data, steps, and sharing information.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteSession} className="bg-red-600 hover:bg-red-700">
              Delete Session
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// Session Management View Component
function SessionManagementView({
  sessions,
  filters,
  onFilterChange,
  onRefresh,
  onCreateNew,
  onViewDetails,
  onShare,
  onReplay,
  onExport,
  onDelete
}: {
  sessions: SessionListResponse | null;
  filters: SessionListParams;
  onFilterChange: (key: keyof SessionListParams, value: any) => void;
  onRefresh: () => void;
  onCreateNew: () => void;
  onViewDetails: (session: AttackSession) => void;
  onShare: (session: AttackSession) => void;
  onReplay: (session: AttackSession) => void;
  onExport: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
}) {
  if (!sessions) {
    return (
      <Card>
        <CardContent className="text-center py-12">
          <RefreshCw className="h-12 w-12 text-gray-400 mx-auto mb-4 animate-spin" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Loading Sessions...
          </h3>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Filters and Actions */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="flex flex-wrap gap-2">
          <Input
            placeholder="Search sessions..."
            value={filters.search || ''}
            onChange={(e) => onFilterChange('search', e.target.value)}
            className="w-64"
          />

          <Select
            value={filters.status || 'all'}
            onValueChange={(value) => onFilterChange('status', value === 'all' ? undefined : value)}
          >
            <SelectTrigger className="w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="active">Active</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="paused">Paused</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={filters.category || 'all'}
            onValueChange={(value) => onFilterChange('category', value === 'all' ? undefined : value)}
          >
            <SelectTrigger className="w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              <SelectItem value="research">Research</SelectItem>
              <SelectItem value="testing">Testing</SelectItem>
              <SelectItem value="production">Production</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex gap-2">
          <Button onClick={onRefresh} variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={onCreateNew}>
            <Plus className="h-4 w-4 mr-2" />
            New Session
          </Button>
        </div>
      </div>

      {sessions.total === 0 ? (
        <Card>
          <CardContent className="text-center py-12">
            <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              No Attack Sessions Found
            </h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Create your first attack session to start recording and replaying your security tests.
            </p>
            <Button onClick={onCreateNew}>
              <Plus className="h-4 w-4 mr-2" />
              Create First Session
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {/* Sessions Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {sessions.sessions.map((session) => (
              <SessionCard
                key={session.session_id}
                session={session}
                onViewDetails={onViewDetails}
                onShare={onShare}
                onReplay={onReplay}
                onExport={onExport}
                onDelete={onDelete}
              />
            ))}
          </div>

          {/* Pagination */}
          {sessions.total > sessions.page_size && (
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Showing {((sessions.page - 1) * sessions.page_size) + 1} to {Math.min(sessions.page * sessions.page_size, sessions.total)} of {sessions.total} sessions
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!sessions.has_prev}
                  onClick={() => onFilterChange('page', sessions.page - 1)}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!sessions.has_next}
                  onClick={() => onFilterChange('page', sessions.page + 1)}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Session Card Component
function SessionCard({
  session,
  onViewDetails,
  onShare,
  onReplay,
  onExport,
  onDelete
}: {
  session: AttackSession;
  onViewDetails: (session: AttackSession) => void;
  onShare: (session: AttackSession) => void;
  onReplay: (session: AttackSession) => void;
  onExport: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
}) {
  const successRate = attackSessionService.calculateSuccessRate(session);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{session.name}</CardTitle>
          <Badge variant="outline" className={`bg-${attackSessionService.getStatusColor(session.status)}-50`}>
            {attackSessionService.getStatusDisplayName(session.status)}
          </Badge>
        </div>
        <CardDescription>
          {session.description || 'No description provided'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-muted-foreground">Target</div>
            <div className="font-medium">{session.target_config.provider} / {session.target_config.model}</div>
          </div>
          <div>
            <div className="text-muted-foreground">Created</div>
            <div className="font-medium">{new Date(session.created_at).toLocaleDateString()}</div>
          </div>
          <div>
            <div className="text-muted-foreground">Steps</div>
            <div className="font-medium">{session.total_steps}</div>
          </div>
          <div>
            <div className="text-muted-foreground">Success Rate</div>
            <div className="font-medium">{successRate.toFixed(1)}%</div>
          </div>
        </div>

        {session.tags && session.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {session.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            ))}
            {session.tags.length > 3 && (
              <Badge variant="secondary" className="text-xs">
                +{session.tags.length - 3} more
              </Badge>
            )}
          </div>
        )}

        <Separator />

        <div className="flex justify-between">
          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={() => onViewDetails(session)}>
              <Eye className="h-4 w-4 mr-1" />
              Details
            </Button>
            <Button size="sm" variant="outline" onClick={() => onReplay(session)}>
              <Play className="h-4 w-4 mr-1" />
              Replay
            </Button>
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={() => onShare(session)}>
              <Share2 className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="outline" onClick={() => onExport(session.session_id)}>
              <Download className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="outline" onClick={() => onDelete(session.session_id)}>
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Shared Sessions View Component (placeholder)
function SharedSessionsView({
  filters,
  onFilterChange,
  onRefresh,
  onViewDetails,
  onReplay,
  onExport
}: {
  filters: SessionListParams;
  onFilterChange: (key: keyof SessionListParams, value: any) => void;
  onRefresh: () => void;
  onViewDetails: (session: AttackSession) => void;
  onReplay: (session: AttackSession) => void;
  onExport: (sessionId: string) => void;
}) {
  return (
    <Card>
      <CardContent className="text-center py-12">
        <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          Shared Sessions
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Sessions shared with you by team members will appear here.
        </p>
      </CardContent>
    </Card>
  );
}

// Session Details View Component
function SessionDetailsView({ session }: { session: AttackSession }) {
  return (
    <div className="space-y-6">
      {/* Session Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold">{session.total_steps}</div>
          <div className="text-sm text-muted-foreground">Total Steps</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold">{attackSessionService.calculateSuccessRate(session).toFixed(1)}%</div>
          <div className="text-sm text-muted-foreground">Success Rate</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold">{attackSessionService.formatExecutionTime(session.total_execution_time)}</div>
          <div className="text-sm text-muted-foreground">Duration</div>
        </div>
        <div className="text-center">
          <Badge variant="outline" className={`bg-${attackSessionService.getStatusColor(session.status)}-50`}>
            {attackSessionService.getStatusDisplayName(session.status)}
          </Badge>
          <div className="text-sm text-muted-foreground mt-1">Status</div>
        </div>
      </div>

      <Separator />

      {/* Configuration */}
      <div>
        <h4 className="font-semibold mb-3">Configuration</h4>
        <div className="bg-muted rounded-lg p-4 space-y-2 text-sm">
          <div><span className="font-medium">Target:</span> {session.target_config.provider} / {session.target_config.model}</div>
          <div><span className="font-medium">Original Prompt:</span> {session.original_prompt}</div>
          {session.category && <div><span className="font-medium">Category:</span> {session.category}</div>}
        </div>
      </div>

      {/* Steps */}
      <div>
        <h4 className="font-semibold mb-3">Execution Steps ({session.steps.length})</h4>
        <div className="space-y-3 max-h-64 overflow-y-auto">
          {session.steps.map((step, index) => (
            <div key={step.step_id} className="border rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">Step {index + 1}</Badge>
                  <span className="text-sm font-medium">{attackSessionService.getStepTypeDisplayName(step.step_type)}</span>
                  {step.success ? (
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                  )}
                </div>
                <div className="text-xs text-muted-foreground">
                  {attackSessionService.formatExecutionTime(step.execution_time)}
                </div>
              </div>
              {step.technique_used && (
                <div className="text-sm text-muted-foreground">
                  Technique: {step.technique_used}
                </div>
              )}
              {step.error_message && (
                <div className="text-sm text-red-600 mt-1">
                  Error: {step.error_message}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
