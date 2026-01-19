/**
 * Scheduled Testing & Monitoring Interface
 *
 * Phase 3 enterprise feature for automation:
 * - Recurring adversarial test scheduling
 * - Alert system for behavior changes
 * - Defense regression monitoring
 * - Compliance documentation support
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
  Clock,
  Calendar,
  Play,
  Pause,
  Plus,
  Trash2,
  Settings,
  Bell,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Search,
  Filter,
  BarChart3,
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  Eye,
  Edit,
  Copy,
  Mail,
  Webhook,
  MessageSquare,
  Users,
  Target,
  Shield,
  AlertCircle,
  Info,
  XCircle
} from 'lucide-react';
import { toast } from 'sonner';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

// Import services
import {
  scheduledTestingService,
  ScheduledTest,
  ScheduleExecution,
  AlertEvent,
  MonitoringDashboard,
  ScheduleCreate,
  ScheduleUpdate,
  AlertRule,
  ScheduleListResponse,
  ExecutionListResponse,
  AlertListResponse,
  ScheduleFrequency,
  ScheduleStatus,
  AlertSeverity,
  AlertType,
  MonitoringMetric
} from '@/lib/api/services/scheduled-testing';

export default function ScheduledTestingPage() {
  // Data state
  const [dashboard, setDashboard] = useState<MonitoringDashboard | null>(null);
  const [schedules, setSchedules] = useState<ScheduleListResponse | null>(null);
  const [selectedSchedule, setSelectedSchedule] = useState<ScheduledTest | null>(null);
  const [executions, setExecutions] = useState<ExecutionListResponse | null>(null);
  const [alerts, setAlerts] = useState<AlertListResponse | null>(null);

  // Form state
  const [newScheduleData, setNewScheduleData] = useState<ScheduleCreate>({
    name: '',
    description: '',
    test_config: {},
    frequency: 'daily'
  });
  const [editScheduleData, setEditScheduleData] = useState<ScheduleUpdate>({});
  const [newAlertRule, setNewAlertRule] = useState<AlertRule>(scheduledTestingService.createDefaultAlertRule());

  // UI state
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showAlertRuleDialog, setShowAlertRuleDialog] = useState(false);
  const [scheduleToDelete, setScheduleToDelete] = useState<ScheduledTest | null>(null);
  const [activeTab, setActiveTab] = useState('dashboard');

  useEffect(() => {
    loadDashboard();
    loadSchedules();
    loadAlerts();
  }, []);

  const loadDashboard = useCallback(async () => {
    try {
      const data = await scheduledTestingService.getMonitoringDashboard();
      setDashboard(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const loadSchedules = useCallback(async () => {
    try {
      setLoading(true);
      const data = await scheduledTestingService.listSchedules({ page: 1, page_size: 50 });
      setSchedules(data);
    } catch (error) {
      // Error already handled in service
    } finally {
      setLoading(false);
    }
  }, []);

  const loadExecutions = useCallback(async (scheduleId: string) => {
    try {
      const data = await scheduledTestingService.listScheduleExecutions(scheduleId, { page: 1, page_size: 20 });
      setExecutions(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const loadAlerts = useCallback(async () => {
    try {
      const data = await scheduledTestingService.listAlerts({ page: 1, page_size: 20 });
      setAlerts(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const handleCreateSchedule = useCallback(async () => {
    const errors = scheduledTestingService.validateScheduleCreate(newScheduleData);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setCreating(true);
      const schedule = await scheduledTestingService.createSchedule(newScheduleData);

      // Refresh data
      await Promise.all([loadSchedules(), loadDashboard()]);

      // Select the new schedule
      setSelectedSchedule(schedule);

      // Reset form
      setNewScheduleData({
        name: '',
        description: '',
        test_config: {},
        frequency: 'daily'
      });
      setShowCreateDialog(false);

      toast.success(`Schedule "${schedule.name}" created successfully!`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [newScheduleData, loadSchedules, loadDashboard]);

  const handleUpdateSchedule = useCallback(async () => {
    if (!selectedSchedule) return;

    try {
      const updatedSchedule = await scheduledTestingService.updateSchedule(
        selectedSchedule.schedule_id,
        editScheduleData
      );

      setSelectedSchedule(updatedSchedule);
      await Promise.all([loadSchedules(), loadDashboard()]);
      setShowEditDialog(false);
      setEditScheduleData({});
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedSchedule, editScheduleData, loadSchedules, loadDashboard]);

  const handleTriggerExecution = useCallback(async (schedule: ScheduledTest) => {
    try {
      await scheduledTestingService.triggerExecution(schedule.schedule_id);

      // Refresh executions if this schedule is selected
      if (selectedSchedule?.schedule_id === schedule.schedule_id) {
        await loadExecutions(schedule.schedule_id);
      }

      await loadDashboard();
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedSchedule, loadExecutions, loadDashboard]);

  const handleDeleteSchedule = useCallback(async () => {
    if (!scheduleToDelete) return;

    try {
      await scheduledTestingService.deleteSchedule(scheduleToDelete.schedule_id);

      // Refresh schedules
      await Promise.all([loadSchedules(), loadDashboard()]);

      // Clear selection if deleted schedule was selected
      if (selectedSchedule?.schedule_id === scheduleToDelete.schedule_id) {
        setSelectedSchedule(null);
      }

      setScheduleToDelete(null);
      setShowDeleteDialog(false);
    } catch (error) {
      // Error already handled in service
    }
  }, [scheduleToDelete, loadSchedules, loadDashboard, selectedSchedule]);

  const handleAcknowledgeAlert = useCallback(async (alert: AlertEvent) => {
    try {
      await scheduledTestingService.acknowledgeAlert(alert.alert_id);
      await loadAlerts();
    } catch (error) {
      // Error already handled in service
    }
  }, [loadAlerts]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Scheduled Testing...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load your monitoring and scheduled test data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Scheduled Testing & Monitoring
        </h1>
        <p className="text-muted-foreground text-lg">
          Automate recurring security assessments with intelligent monitoring, alerting, and regression detection.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          <TabsTrigger value="schedules">Schedules</TabsTrigger>
          <TabsTrigger value="executions">Executions</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="space-y-6">
          <MonitoringDashboardView
            dashboard={dashboard}
            onRefresh={loadDashboard}
          />
        </TabsContent>

        <TabsContent value="schedules" className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="text-2xl font-bold">Scheduled Tests</h2>
              <Badge variant="outline">
                {schedules?.total || 0} schedule{(schedules?.total || 0) !== 1 ? 's' : ''}
              </Badge>
            </div>
            <Button onClick={() => setShowCreateDialog(true)}>
              <Plus className="h-4 w-4 mr-2" />
              New Schedule
            </Button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Schedule List */}
            <div className="lg:col-span-1">
              <ScheduleListView
                schedules={schedules}
                selectedSchedule={selectedSchedule}
                onSelectSchedule={(schedule: ScheduledTest) => {
                  setSelectedSchedule(schedule);
                  loadExecutions(schedule.schedule_id);
                }}
                onTriggerExecution={handleTriggerExecution}
                onEditSchedule={(schedule: ScheduledTest) => {
                  setEditScheduleData({
                    name: schedule.name,
                    description: schedule.description,
                    frequency: schedule.frequency,
                    status: schedule.status
                  });
                  setShowEditDialog(true);
                }}
                onDeleteSchedule={(schedule: ScheduledTest) => {
                  setScheduleToDelete(schedule);
                  setShowDeleteDialog(true);
                }}
                onRefresh={loadSchedules}
              />
            </div>

            {/* Schedule Details */}
            <div className="lg:col-span-2">
              {selectedSchedule ? (
                <ScheduleDetailsView
                  schedule={selectedSchedule}
                  executions={executions}
                  onRefresh={() => loadExecutions(selectedSchedule.schedule_id)}
                />
              ) : (
                <Card>
                  <CardContent className="text-center py-12">
                    <Calendar className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      No Schedule Selected
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300">
                      Select a scheduled test from the list to view details and execution history.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="executions" className="space-y-6">
          <ExecutionHistoryView
            executions={executions}
            selectedSchedule={selectedSchedule}
            onRefresh={() => selectedSchedule && loadExecutions(selectedSchedule.schedule_id)}
          />
        </TabsContent>

        <TabsContent value="alerts" className="space-y-6">
          <AlertsView
            alerts={alerts}
            onAcknowledgeAlert={handleAcknowledgeAlert}
            onRefresh={loadAlerts}
          />
        </TabsContent>
      </Tabs>

      {/* Create Schedule Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Create Scheduled Test</DialogTitle>
            <DialogDescription>
              Set up a recurring security assessment with automated monitoring and alerting.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Schedule Name</Label>
                <Input
                  value={newScheduleData.name}
                  onChange={(e) => setNewScheduleData(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Daily Security Assessment"
                />
              </div>

              <div className="space-y-2">
                <Label>Frequency</Label>
                <Select
                  value={newScheduleData.frequency}
                  onValueChange={(value: ScheduleFrequency) => setNewScheduleData(prev => ({ ...prev, frequency: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {scheduledTestingService.getAvailableFrequencies().map(freq => (
                      <SelectItem key={freq.id} value={freq.id}>
                        {freq.name} - {freq.description}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                value={newScheduleData.description}
                onChange={(e) => setNewScheduleData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Automated security testing for production models..."
                className="min-h-20"
              />
            </div>

            {newScheduleData.frequency === 'custom_cron' && (
              <div className="space-y-2">
                <Label>Cron Expression</Label>
                <Input
                  value={newScheduleData.cron_expression || ''}
                  onChange={(e) => setNewScheduleData(prev => ({ ...prev, cron_expression: e.target.value }))}
                  placeholder="0 2 * * *"
                />
                <p className="text-xs text-muted-foreground">
                  Use standard cron format (minute hour day month weekday)
                </p>
              </div>
            )}

            <div className="space-y-2">
              <Label>Test Configuration</Label>
              <Textarea
                value={JSON.stringify(newScheduleData.test_config, null, 2)}
                onChange={(e) => {
                  try {
                    const config = JSON.parse(e.target.value);
                    setNewScheduleData(prev => ({ ...prev, test_config: config }));
                  } catch (error) {
                    // Invalid JSON, ignore for now
                  }
                }}
                placeholder='{"test_suite_name": "Security Assessment", "target_models": ["gpt-4"], "test_techniques": ["prompt_injection_basic"]}'
                className="min-h-32 font-mono text-sm"
              />
              <p className="text-xs text-muted-foreground">
                JSON configuration for CI/CD test suite (same format as CI/CD integration)
              </p>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateSchedule} disabled={creating}>
                {creating ? (
                  <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Plus className="h-4 w-4 mr-2" />
                )}
                Create Schedule
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Schedule Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Schedule</DialogTitle>
            <DialogDescription>
              Update schedule settings and configuration.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Schedule Name</Label>
              <Input
                value={editScheduleData.name || ''}
                onChange={(e) => setEditScheduleData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Schedule name"
              />
            </div>

            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                value={editScheduleData.description || ''}
                onChange={(e) => setEditScheduleData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Schedule description"
                className="min-h-20"
              />
            </div>

            <div className="space-y-2">
              <Label>Status</Label>
              <Select
                value={editScheduleData.status || 'active'}
                onValueChange={(value: ScheduleStatus) => setEditScheduleData(prev => ({ ...prev, status: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="paused">Paused</SelectItem>
                  <SelectItem value="disabled">Disabled</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowEditDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleUpdateSchedule}>
                <Settings className="h-4 w-4 mr-2" />
                Update Schedule
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Schedule</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{scheduleToDelete?.name}&quot;? This action cannot be undone
              and will remove all execution history and alert configurations.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteSchedule} className="bg-red-600 hover:bg-red-700">
              Delete Schedule
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// Dashboard View Component
function MonitoringDashboardView({
  dashboard,
  onRefresh
}: {
  dashboard: MonitoringDashboard | null;
  onRefresh: () => void;
}) {
  if (!dashboard) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            <div className="grid grid-cols-4 gap-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-24 bg-gray-200 rounded"></div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Monitoring Dashboard</h2>
        <Button onClick={onRefresh} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <Calendar className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <div className="text-2xl font-bold">{dashboard.total_schedules}</div>
            <div className="text-sm text-muted-foreground">Total Schedules</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Activity className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <div className="text-2xl font-bold">{dashboard.active_schedules}</div>
            <div className="text-sm text-muted-foreground">Active Schedules</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Play className="h-8 w-8 mx-auto mb-2 text-orange-600" />
            <div className="text-2xl font-bold">{dashboard.recent_executions}</div>
            <div className="text-sm text-muted-foreground">Recent Executions</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <CheckCircle className="h-8 w-8 mx-auto mb-2 text-purple-600" />
            <div className="text-2xl font-bold">{dashboard.success_rate.toFixed(1)}%</div>
            <div className="text-sm text-muted-foreground">Success Rate</div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Success Rate Trend</CardTitle>
            <CardDescription>Last 7 days success rate percentage</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={dashboard.success_rate_trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Execution Count</CardTitle>
            <CardDescription>Daily execution volume over last 7 days</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={dashboard.execution_count_trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Recent Executions</CardTitle>
            <CardDescription>Latest test executions</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              {dashboard.recent_executions_list.length === 0 ? (
                <div className="text-center py-8">
                  <Play className="h-8 w-8 text-gray-400 mx-auto mb-3" />
                  <p className="text-muted-foreground">No recent executions</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {dashboard.recent_executions_list.map((execution) => (
                    <div key={execution.execution_id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="text-lg">
                          {execution.status === 'success' ? (
                            <CheckCircle className="h-5 w-5 text-green-600" />
                          ) : execution.status === 'failed' ? (
                            <XCircle className="h-5 w-5 text-red-600" />
                          ) : (
                            <Clock className="h-5 w-5 text-yellow-600" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium">{execution.schedule_id}</p>
                          <p className="text-sm text-muted-foreground">
                            {execution.success_rate.toFixed(1)}% success • {execution.total_tests} tests
                          </p>
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {scheduledTestingService.formatExecutionTime(execution.duration_seconds || 0)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Alerts</CardTitle>
            <CardDescription>Latest monitoring alerts</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              {dashboard.recent_alerts.length === 0 ? (
                <div className="text-center py-8">
                  <Bell className="h-8 w-8 text-gray-400 mx-auto mb-3" />
                  <p className="text-muted-foreground">No recent alerts</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {dashboard.recent_alerts.map((alert) => (
                    <div key={alert.alert_id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="text-lg">
                          {alert.severity === 'critical' ? (
                            <AlertTriangle className="h-5 w-5 text-red-600" />
                          ) : alert.severity === 'warning' ? (
                            <AlertCircle className="h-5 w-5 text-yellow-600" />
                          ) : (
                            <Info className="h-5 w-5 text-blue-600" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium">{alert.title}</p>
                          <p className="text-sm text-muted-foreground">{alert.message}</p>
                        </div>
                      </div>
                      <Badge variant="outline" className={`bg-${scheduledTestingService.getAlertSeverityColor(alert.severity)}-50`}>
                        {scheduledTestingService.getAlertSeverityDisplayName(alert.severity)}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Unhealthy Schedules */}
      {dashboard.unhealthy_schedules.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-600" />
              Health Issues
            </CardTitle>
            <CardDescription>
              Schedules requiring attention
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {dashboard.unhealthy_schedules.map((schedule) => (
                <div key={schedule.schedule_id} className="flex items-center justify-between p-3 border rounded-lg bg-red-50 border-red-200">
                  <div>
                    <h4 className="font-medium">{schedule.name}</h4>
                    <p className="text-sm text-muted-foreground">
                      {schedule.failure_count} failures • Status: {scheduledTestingService.getStatusDisplayName(schedule.status)}
                    </p>
                  </div>
                  <Badge variant="outline" className={`bg-${scheduledTestingService.getStatusColor(schedule.status)}-50`}>
                    {scheduledTestingService.getStatusDisplayName(schedule.status)}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Placeholder components for the different views - in a real implementation these would be fully featured

function ScheduleListView({ schedules, selectedSchedule, onSelectSchedule, onTriggerExecution, onEditSchedule, onDeleteSchedule, onRefresh }: any) {
  if (!schedules) return <div>Loading...</div>;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Schedules
          </CardTitle>
          <Button size="sm" onClick={onRefresh} variant="outline">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>
          {schedules.total} scheduled test{schedules.total !== 1 ? 's' : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-96">
          {schedules.total === 0 ? (
            <div className="text-center py-8 px-4">
              <Calendar className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <h4 className="font-semibold mb-2">No Scheduled Tests</h4>
              <p className="text-sm text-muted-foreground mb-3">
                Create your first scheduled test to start automated monitoring.
              </p>
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {schedules.schedules.map((schedule: ScheduledTest) => (
                <div
                  key={schedule.schedule_id}
                  className={`p-3 rounded-lg border transition-colors cursor-pointer ${
                    selectedSchedule?.schedule_id === schedule.schedule_id
                      ? 'bg-primary/10 border-primary/20'
                      : 'hover:bg-muted'
                  }`}
                  onClick={() => onSelectSchedule(schedule)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium truncate">{schedule.name}</h4>
                    <div className="flex gap-1">
                      <Badge variant="outline" className={`bg-${scheduledTestingService.getStatusColor(schedule.status)}-50 text-xs`}>
                        {scheduledTestingService.getStatusDisplayName(schedule.status)}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground truncate mb-2">
                    {schedule.description}
                  </p>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{scheduledTestingService.getFrequencyDisplayName(schedule.frequency)}</span>
                    <span>{scheduledTestingService.formatNextExecution(schedule.next_execution)}</span>
                  </div>
                  <div className="flex items-center gap-1 mt-2">
                    <Button size="sm" onClick={(e) => { e.stopPropagation(); onTriggerExecution(schedule); }}>
                      <Play className="h-3 w-3 mr-1" />
                      Run
                    </Button>
                    <Button size="sm" variant="outline" onClick={(e) => { e.stopPropagation(); onEditSchedule(schedule); }}>
                      <Edit className="h-3 w-3" />
                    </Button>
                    <Button size="sm" variant="outline" onClick={(e) => { e.stopPropagation(); onDeleteSchedule(schedule); }}>
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function ScheduleDetailsView({ schedule, executions, onRefresh }: any) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>{schedule.name}</CardTitle>
          <CardDescription>{schedule.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <Label className="font-medium">Frequency</Label>
              <p>{scheduledTestingService.getFrequencyDisplayName(schedule.frequency)}</p>
            </div>
            <div>
              <Label className="font-medium">Status</Label>
              <Badge variant="outline" className={`bg-${scheduledTestingService.getStatusColor(schedule.status)}-50`}>
                {scheduledTestingService.getStatusDisplayName(schedule.status)}
              </Badge>
            </div>
            <div>
              <Label className="font-medium">Next Execution</Label>
              <p>{scheduledTestingService.formatNextExecution(schedule.next_execution)}</p>
            </div>
            <div>
              <Label className="font-medium">Last Execution</Label>
              <p>{scheduledTestingService.formatLastExecution(schedule.last_execution)}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent Executions</CardTitle>
          <CardDescription>Latest execution history for this schedule</CardDescription>
        </CardHeader>
        <CardContent>
          {!executions || executions.executions.length === 0 ? (
            <div className="text-center py-8">
              <Play className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <p className="text-muted-foreground">No executions yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {executions.executions.map((execution: ScheduleExecution) => (
                <div key={execution.execution_id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    {execution.status === 'success' ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : execution.status === 'failed' ? (
                      <XCircle className="h-5 w-5 text-red-600" />
                    ) : (
                      <Clock className="h-5 w-5 text-yellow-600" />
                    )}
                    <div>
                      <p className="font-medium">{execution.status.toUpperCase()}</p>
                      <p className="text-sm text-muted-foreground">
                        {execution.success_rate.toFixed(1)}% success • {execution.total_tests} tests
                      </p>
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {new Date(execution.started_at).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ExecutionHistoryView({ executions, selectedSchedule, onRefresh }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution History</CardTitle>
        <CardDescription>
          {selectedSchedule ? `Executions for "${selectedSchedule.name}"` : 'Select a schedule to view executions'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <Play className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Execution history will appear here</p>
        </div>
      </CardContent>
    </Card>
  );
}

function AlertsView({ alerts, onAcknowledgeAlert, onRefresh }: any) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Alerts & Notifications</h2>
        <Button onClick={onRefresh} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <Card>
        <CardContent>
          {!alerts || alerts.alerts.length === 0 ? (
            <div className="text-center py-8">
              <Bell className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <p className="text-muted-foreground">No alerts to display</p>
            </div>
          ) : (
            <div className="space-y-3">
              {alerts.alerts.map((alert: AlertEvent) => (
                <div key={alert.alert_id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-3">
                    {alert.severity === 'critical' ? (
                      <AlertTriangle className="h-6 w-6 text-red-600" />
                    ) : alert.severity === 'warning' ? (
                      <AlertCircle className="h-6 w-6 text-yellow-600" />
                    ) : (
                      <Info className="h-6 w-6 text-blue-600" />
                    )}
                    <div>
                      <h4 className="font-medium">{alert.title}</h4>
                      <p className="text-sm text-muted-foreground">{alert.message}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(alert.triggered_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className={`bg-${scheduledTestingService.getAlertSeverityColor(alert.severity)}-50`}>
                      {scheduledTestingService.getAlertSeverityDisplayName(alert.severity)}
                    </Badge>
                    {!alert.acknowledged_at && (
                      <Button size="sm" onClick={() => onAcknowledgeAlert(alert)}>
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Acknowledge
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
