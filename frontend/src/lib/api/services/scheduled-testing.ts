import { apiClient } from '../client';
import { toast } from 'sonner';

// API Response Types
interface ScheduleApiResponse {
  schedule_id?: string;
  id?: string;
  name: string;
  description?: string;
  workspace_id?: string;
  frequency: string;
  cron_expression?: string;
  timezone?: string;
  next_execution?: string;
  last_execution?: string;
  alert_rules?: any[];
  status?: string;
  created_at?: string;
  updated_at?: string;
  failure_count?: number;
  execution_count?: number;
  test_config?: {
    target_model_ids?: string[];
    target_models?: string[];
    technique_ids?: string[];
    techniques?: string[];
    dataset_id?: string;
    test_sample_size?: number;
    sample_size?: number;
    notification_channels?: string[];
  };
}

interface ExecutionApiResponse {
  execution_id?: string;
  id?: string;
  schedule_id: string;
  started_at?: string;
  timestamp?: string;
  completed_at?: string;
  status?: string;
  success_rate: number;
  total_tests: number;
  failed_tests?: number;
  new_vulnerabilities?: number;
  alerts_triggered?: any[];
  duration_seconds?: number;
}

interface DashboardApiResponse {
  active_schedules?: number;
  total_schedules?: number;
  total_executions?: number;
  recent_executions?: number;
  pending_alerts?: number;
  unresolved_alerts?: number;
  success_rate?: number;
  performance_score?: number;
  success_rate_trend?: Array<{ date: string; value: number }>;
  execution_count_trend?: Array<{ date: string; value: number }>;
  failure_rate_by_provider?: Record<string, number>;
  recent_executions_list?: ExecutionApiResponse[];
  recent_alerts?: any[];
  unhealthy_schedules?: ScheduleApiResponse[];
}

interface ScheduleListApiResponse {
  schedules?: ScheduleApiResponse[];
  total?: number;
  page?: number;
  page_size?: number;
  has_next?: boolean;
  has_prev?: boolean;
}

interface ExecutionListApiResponse {
  executions?: ExecutionApiResponse[];
  total?: number;
  page?: number;
  page_size?: number;
  has_next?: boolean;
  has_prev?: boolean;
}

interface AlertListApiResponse {
  alerts?: any[];
  total?: number;
  page?: number;
  page_size?: number;
  has_next?: boolean;
  has_prev?: boolean;
}

interface TriggerExecutionApiResponse {
  execution_id: string;
}

export type ScheduleFrequency = 'hourly' | 'daily' | 'weekly' | 'monthly' | 'custom_cron';
export type AlertType = 'email' | 'webhook' | 'slack' | 'teams';
export type MonitoringMetric =
  | 'success_rate'
  | 'failure_count'
  | 'response_time'
  | 'new_vulnerabilities'
  | 'regression_detection';
export type AlertSeverity = 'info' | 'warning' | 'critical';
export type ScheduleStatus = 'active' | 'paused' | 'disabled' | 'error';

export interface AlertRule {
  rule_id: string;
  name: string;
  description?: string;
  metric: MonitoringMetric;
  comparison_operator?: '>' | '<' | '>=' | '<=' | '==' | '!=';
  threshold?: number;
  severity: AlertSeverity;
  alert_type?: AlertType;
  alert_target?: string;
  cooldown_minutes?: number;
  enabled: boolean;
}

export interface ScheduledTest {
  id: string;
  schedule_id: string;
  name: string;
  description?: string;
  workspace_id?: string;
  frequency: ScheduleFrequency;
  cron_expression?: string;
  timezone: string;
  next_execution?: string;
  last_execution?: string;
  target_model_ids: string[];
  technique_ids: string[];
  dataset_id?: string;
  test_sample_size?: number;
  alert_rules: AlertRule[];
  notification_channels: string[];
  status: ScheduleStatus;
  created_at: string;
  updated_at: string;
  failure_count?: number;
  execution_count?: number;
  test_config?: Record<string, any>;
}

export interface TestExecutionResult {
  execution_id: string;
  schedule_id: string;
  started_at: string;
  completed_at?: string;
  status: 'success' | 'failed' | 'error' | 'timeout';
  success_rate: number;
  total_tests: number;
  failed_tests?: number;
  new_vulnerabilities?: number;
  alerts_triggered?: any[];
  duration_seconds?: number;
}

export interface MonitoringDashboard {
  active_schedules: number;
  total_schedules: number;
  total_executions: number;
  recent_executions: number;
  unresolved_alerts: number;
  performance_score: number;
  success_rate: number;
  success_rate_trend: { date: string; value: number }[];
  execution_count_trend: { date: string; value: number }[];
  failure_rate_by_provider: Record<string, number>;
  recent_executions_list: TestExecutionResult[];
  recent_alerts: any[];
  unhealthy_schedules: ScheduledTest[];
}

export interface ScheduleListResponse {
  schedules: ScheduledTest[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ExecutionListResponse {
  executions: TestExecutionResult[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface AlertListResponse {
  alerts: any[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export type ScheduleCreate = Partial<ScheduledTest>;
export type ScheduleUpdate = Partial<ScheduledTest>;
export type ScheduleExecution = TestExecutionResult;
export type AlertEvent = any;

export class ScheduledTestingService {
  private readonly baseUrl = '/scheduled-testing';
  private readonly schedulePath = `${this.baseUrl}/schedules`;

  private mapSchedule(apiSchedule: any): ScheduledTest {
    const testConfig = apiSchedule.test_config ?? {};
    const id = apiSchedule.schedule_id ?? apiSchedule.id ?? '';
    return {
      id,
      schedule_id: id,
      name: apiSchedule.name,
      description: apiSchedule.description,
      workspace_id: apiSchedule.workspace_id,
      frequency: apiSchedule.frequency,
      cron_expression: apiSchedule.cron_expression,
      timezone: apiSchedule.timezone ?? 'UTC',
      next_execution: apiSchedule.next_execution,
      last_execution: apiSchedule.last_execution,
      target_model_ids: testConfig.target_model_ids ?? testConfig.target_models ?? [],
      technique_ids: testConfig.technique_ids ?? testConfig.techniques ?? [],
      dataset_id: testConfig.dataset_id,
      test_sample_size: testConfig.test_sample_size ?? testConfig.sample_size,
      alert_rules: apiSchedule.alert_rules ?? [],
      notification_channels: testConfig.notification_channels ?? [],
      status: apiSchedule.status ?? 'active',
      created_at: apiSchedule.created_at ?? new Date().toISOString(),
      updated_at: apiSchedule.updated_at ?? new Date().toISOString(),
      failure_count: apiSchedule.failure_count ?? 0,
      execution_count: apiSchedule.execution_count ?? 0,
      test_config: testConfig
    };
  }

  private mapExecution(apiExecution: any): TestExecutionResult {
    return {
      execution_id: apiExecution.execution_id ?? apiExecution.id ?? '',
      schedule_id: apiExecution.schedule_id,
      started_at: apiExecution.started_at ?? apiExecution.timestamp ?? new Date().toISOString(),
      completed_at: apiExecution.completed_at,
      status: apiExecution.status ?? 'success',
      success_rate: apiExecution.success_rate,
      total_tests: apiExecution.total_tests,
      failed_tests: apiExecution.failed_tests,
      new_vulnerabilities: apiExecution.new_vulnerabilities,
      alerts_triggered: apiExecution.alerts_triggered,
      duration_seconds: apiExecution.duration_seconds
    };
  }

  createDefaultAlertRule(): AlertRule {
    return {
      rule_id: '',
      name: 'New Alert Rule',
      metric: 'success_rate',
      comparison_operator: '>',
      threshold: 0.1,
      severity: 'warning',
      enabled: true
    };
  }

  async getDashboard(workspaceId?: string): Promise<MonitoringDashboard> {
    const response = await apiClient.get(`${this.baseUrl}/dashboard`, {
      params: { workspace_id: workspaceId }
    });
    const data = response.data as DashboardApiResponse;
    return {
      active_schedules: data.active_schedules ?? 0,
      total_schedules: data.total_schedules ?? 0,
      total_executions: data.total_executions ?? 0,
      recent_executions: data.recent_executions ?? 0,
      unresolved_alerts: data.pending_alerts ?? data.unresolved_alerts ?? 0,
      performance_score: data.success_rate ?? data.performance_score ?? 0,
      success_rate: data.success_rate ?? 0,
      success_rate_trend: data.success_rate_trend ?? [],
      execution_count_trend: data.execution_count_trend ?? [],
      failure_rate_by_provider: data.failure_rate_by_provider ?? {},
      recent_executions_list: (data.recent_executions_list ?? []).map((e: any) => this.mapExecution(e)),
      recent_alerts: data.recent_alerts ?? [],
      unhealthy_schedules: (data.unhealthy_schedules ?? []).map((s: any) => this.mapSchedule(s))
    };
  }

  getMonitoringDashboard(workspaceId?: string) {
    return this.getDashboard(workspaceId);
  }

  async createSchedule(data: ScheduleCreate): Promise<ScheduledTest> {
    try {
      const payload = {
        name: data.name,
        description: data.description,
        frequency: data.frequency,
        cron_expression: data.cron_expression,
        timezone: data.timezone,
        workspace_id: data.workspace_id,
        alert_rules: data.alert_rules ?? [],
        test_config: {
          target_model_ids: data.target_model_ids ?? [],
          technique_ids: data.technique_ids ?? [],
          dataset_id: data.dataset_id,
          test_sample_size: data.test_sample_size,
          notification_channels: data.notification_channels ?? []
        }
      };
      const response = await apiClient.post(this.schedulePath, payload);
      toast.success('Test schedule created');
      return this.mapSchedule(response.data as ScheduleApiResponse);
    } catch (error) {
      console.error('Failed to create schedule:', error);
      toast.error('Failed to create test schedule');
      throw error;
    }
  }

  async getSchedule(id: string): Promise<ScheduledTest> {
    const response = await apiClient.get(`${this.schedulePath}/${id}`);
    return this.mapSchedule(response.data as ScheduleApiResponse);
  }

  async listSchedules(params?: {
    workspace_id?: string;
    status?: ScheduleStatus;
    page?: number;
    page_size?: number;
  }): Promise<ScheduleListResponse> {
    const response = await apiClient.get(this.schedulePath, {
      params: {
        workspace_id: params?.workspace_id,
        status_filter: params?.status,
        page: params?.page,
        page_size: params?.page_size
      }
    });
    const data = response.data as ScheduleListApiResponse;
    return {
      schedules: (data.schedules ?? []).map((s: any) => this.mapSchedule(s)),
      total: data.total ?? data.schedules?.length ?? 0,
      page: data.page ?? 1,
      page_size: data.page_size ?? 20,
      has_next: Boolean(data.has_next),
      has_prev: Boolean(data.has_prev)
    };
  }

  async updateSchedule(id: string, data: ScheduleUpdate): Promise<ScheduledTest> {
    const payload: any = {
      name: data.name,
      description: data.description,
      frequency: data.frequency,
      cron_expression: data.cron_expression,
      timezone: data.timezone,
      status: data.status,
      alert_rules: data.alert_rules,
      test_config: {
        target_model_ids: data.target_model_ids,
        technique_ids: data.technique_ids,
        dataset_id: data.dataset_id,
        test_sample_size: data.test_sample_size,
        notification_channels: data.notification_channels
      }
    };
    const response = await apiClient.patch(`${this.schedulePath}/${id}`, payload);
    toast.success('Schedule updated');
    return this.mapSchedule(response.data as ScheduleApiResponse);
  }

  async deleteSchedule(id: string): Promise<void> {
    await apiClient.delete(`${this.schedulePath}/${id}`);
    toast.success('Schedule deleted');
  }

  async triggerNow(id: string): Promise<{ execution_id: string }> {
    const response = await apiClient.post(
      `${this.schedulePath}/${id}/execute`
    );
    toast.success('Test triggered successfully');
    return response.data as TriggerExecutionApiResponse;
  }

  triggerExecution(id: string) {
    return this.triggerNow(id);
  }

  async getExecutionHistory(
    id: string,
    params: { page?: number; page_size?: number } = {}
  ): Promise<ExecutionListResponse> {
    const response = await apiClient.get(`${this.schedulePath}/${id}/executions`, {
      params
    });
    const data = response.data as ExecutionListApiResponse;
    return {
      executions: (data.executions ?? []).map((e: any) => this.mapExecution(e)),
      total: data.total ?? data.executions?.length ?? 0,
      page: data.page ?? 1,
      page_size: data.page_size ?? params.page_size ?? 20,
      has_next: Boolean(data.has_next),
      has_prev: Boolean(data.has_prev)
    };
  }

  listScheduleExecutions(id: string, params: { page?: number; page_size?: number } = {}) {
    return this.getExecutionHistory(id, params);
  }

  async listAlerts(params: { page?: number; page_size?: number } = {}): Promise<AlertListResponse> {
    const response = await apiClient.get(`${this.baseUrl}/alerts`, { params });
    const data = response.data as AlertListApiResponse;
    return {
      alerts: data.alerts ?? [],
      total: data.total ?? data.alerts?.length ?? 0,
      page: data.page ?? 1,
      page_size: data.page_size ?? params.page_size ?? 20,
      has_next: Boolean(data.has_next),
      has_prev: Boolean(data.has_prev)
    };
  }

  async acknowledgeAlert(alertId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/alerts/${alertId}/acknowledge`);
  }

  getStatusColor(status: ScheduleStatus): string {
    const colors: Record<ScheduleStatus, string> = {
      active: 'green',
      paused: 'yellow',
      disabled: 'gray',
      error: 'red'
    };
    return colors[status];
  }

  getFrequencyDisplayName(freq: ScheduleFrequency): string {
    const names: Record<ScheduleFrequency, string> = {
      hourly: 'Hourly',
      daily: 'Daily',
      weekly: 'Weekly',
      monthly: 'Monthly',
      custom_cron: 'Custom Cron'
    };
    return names[freq];
  }

  getAvailableFrequencies(): Array<{id: ScheduleFrequency, name: string, description: string}> {
    return [
      { id: 'hourly', name: 'Hourly', description: 'Run every hour' },
      { id: 'daily', name: 'Daily', description: 'Run once a day' },
      { id: 'weekly', name: 'Weekly', description: 'Run once a week' },
      { id: 'monthly', name: 'Monthly', description: 'Run once a month' },
      { id: 'custom_cron', name: 'Custom Cron', description: 'Run on custom schedule' }
    ];
  }

  getStatusDisplayName(status: ScheduleStatus): string {
    const names: Record<ScheduleStatus, string> = {
      active: 'Active',
      paused: 'Paused',
      disabled: 'Disabled',
      error: 'Error'
    };
    return names[status] || status;
  }

  getAlertSeverityDisplayName(severity: AlertSeverity): string {
    const names: Record<AlertSeverity, string> = {
      info: 'Info',
      warning: 'Warning',
      critical: 'Critical'
    };
    return names[severity] || severity;
  }

  getAlertSeverityColor(severity: AlertSeverity): string {
    const colors: Record<AlertSeverity, string> = {
      info: 'blue',
      warning: 'yellow',
      critical: 'red'
    };
    return colors[severity] || 'gray';
  }

  formatExecutionTime(seconds: number): string {
    const rounded = Math.round(seconds ?? 0);
    if (rounded < 60) return `${rounded}s`;
    const minutes = Math.floor(rounded / 60);
    const rem = rounded % 60;
    return `${minutes}m ${rem}s`;
  }

  formatNextExecution(next?: string): string {
    if (!next) return 'Not scheduled';
    return new Date(next).toLocaleString();
  }

  formatLastExecution(last?: string): string {
    if (!last) return 'N/A';
    return new Date(last).toLocaleString();
  }

  validateScheduleCreate(data: ScheduleCreate): string[] {
    const errors: string[] = [];
    if (!data.name?.trim()) errors.push('Schedule name is required');
    if (!data.frequency) errors.push('Frequency is required');
    if (data.frequency === 'custom_cron' && !data.cron_expression) {
      errors.push('Cron expression is required for custom schedules');
    }
    return errors;
  }
}

export const scheduledTestingService = new ScheduledTestingService();
