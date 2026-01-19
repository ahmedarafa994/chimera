/**
 * Scheduled Testing & Monitoring Service
 *
 * Phase 3 enterprise feature for automation:
 * - Recurring adversarial test scheduling
 * - Alert system for behavior changes
 * - Defense regression monitoring
 * - Compliance documentation support
 */

import { toast } from 'sonner';
import { apiClient } from '@/lib/api/client';

export type ScheduleFrequency = 'hourly' | 'daily' | 'weekly' | 'monthly' | 'custom_cron';
export type AlertType = 'email' | 'webhook' | 'slack' | 'teams';
export type MonitoringMetric = 'success_rate' | 'failure_count' | 'response_time' | 'new_vulnerabilities' | 'regression_detection';
export type AlertSeverity = 'info' | 'warning' | 'critical';
export type ScheduleStatus = 'active' | 'paused' | 'disabled' | 'error';

export interface AlertRule {
  rule_id: string;
  name: string;
  description: string;
  metric: MonitoringMetric;

  // Threshold configuration
  threshold_value: number;
  comparison_operator: '>' | '<' | '>=' | '<=' | '==' | '!=';

  // Alert settings
  alert_type: AlertType;
  alert_target: string;
  severity: AlertSeverity;

  // Timing
  cooldown_minutes: number;
  enabled: boolean;
}

export interface ScheduledTest {
  schedule_id: string;
  name: string;
  description: string;

  // Test configuration
  test_config: Record<string, any>;

  // Schedule configuration
  frequency: ScheduleFrequency;
  cron_expression?: string;
  timezone: string;

  // Execution settings
  max_execution_time: number;
  retry_on_failure: boolean;
  max_retries: number;

  // Monitoring and alerts
  alert_rules: AlertRule[];
  enable_regression_detection: boolean;
  baseline_window_days: number;

  // Metadata
  created_by: string;
  workspace_id?: string;
  created_at: string;
  updated_at: string;

  // Status
  status: ScheduleStatus;
  next_execution?: string;
  last_execution?: string;
  execution_count: number;
  failure_count: number;
}

export interface ScheduleExecution {
  execution_id: string;
  schedule_id: string;
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;

  // Status and results
  status: string;
  test_execution_id?: string;

  // Metrics
  success_rate: number;
  total_tests: number;
  failed_tests: number;
  new_vulnerabilities: number;

  // Alerts triggered
  alerts_triggered: Array<Record<string, any>>;

  // Error details
  error_message?: string;
  retry_count: number;
}

export interface AlertEvent {
  alert_id: string;
  schedule_id: string;
  execution_id?: string;
  rule_id: string;

  // Alert details
  metric: MonitoringMetric;
  threshold_value: number;
  actual_value: number;
  severity: AlertSeverity;

  // Message
  title: string;
  message: string;

  // Timing
  triggered_at: string;
  acknowledged_at?: string;
  resolved_at?: string;

  // Delivery
  alert_type: AlertType;
  delivery_status: string;
  delivery_attempts: number;
}

export interface ScheduleCreate {
  name: string;
  description: string;
  test_config: Record<string, any>;
  frequency: ScheduleFrequency;
  cron_expression?: string;
  timezone?: string;
  workspace_id?: string;
  alert_rules?: AlertRule[];
}

export interface ScheduleUpdate {
  name?: string;
  description?: string;
  test_config?: Record<string, any>;
  frequency?: ScheduleFrequency;
  cron_expression?: string;
  timezone?: string;
  status?: ScheduleStatus;
  alert_rules?: AlertRule[];
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
  executions: ScheduleExecution[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface AlertListResponse {
  alerts: AlertEvent[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface MonitoringDashboard {
  total_schedules: number;
  active_schedules: number;
  recent_executions: number;
  success_rate: number;

  // Recent activity
  recent_executions_list: ScheduleExecution[];
  recent_alerts: AlertEvent[];

  // Trends
  success_rate_trend: Array<{ date: string; value: number }>;
  execution_count_trend: Array<{ date: string; value: number }>;

  // Health status
  unhealthy_schedules: ScheduledTest[];
  pending_alerts: number;
}

export interface ScheduleListParams {
  page?: number;
  page_size?: number;
  workspace_id?: string;
  status?: ScheduleStatus;
}

export interface ExecutionListParams {
  page?: number;
  page_size?: number;
}

export interface AlertListParams {
  page?: number;
  page_size?: number;
  severity?: AlertSeverity;
  unresolved_only?: boolean;
}

class ScheduledTestingService {
  private readonly baseUrl = '/scheduled-testing';

  /**
   * Create a new scheduled test
   */
  async createSchedule(scheduleData: ScheduleCreate): Promise<ScheduledTest> {
    try {
      const response = await apiClient.post<ScheduledTest>(`${this.baseUrl}/schedules`, scheduleData);
      toast.success('Scheduled test created successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to create scheduled test:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create scheduled test');
      throw error;
    }
  }

  /**
   * List scheduled tests with filtering and pagination
   */
  async listSchedules(params?: ScheduleListParams): Promise<ScheduleListResponse> {
    try {
      const response = await apiClient.get<ScheduleListResponse>(`${this.baseUrl}/schedules`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          workspace_id: params?.workspace_id,
          status_filter: params?.status
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to list schedules:', error);
      toast.error('Failed to load scheduled tests');
      throw error;
    }
  }

  /**
   * Get schedule details
   */
  async getSchedule(scheduleId: string): Promise<ScheduledTest> {
    try {
      const response = await apiClient.get<ScheduledTest>(`${this.baseUrl}/schedules/${scheduleId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get schedule:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to load schedule');
      throw error;
    }
  }

  /**
   * Update scheduled test
   */
  async updateSchedule(scheduleId: string, updateData: ScheduleUpdate): Promise<ScheduledTest> {
    try {
      const response = await apiClient.patch<ScheduledTest>(`${this.baseUrl}/schedules/${scheduleId}`, updateData);
      toast.success('Schedule updated successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to update schedule:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to update schedule');
      throw error;
    }
  }

  /**
   * Trigger manual execution of scheduled test
   */
  async triggerExecution(scheduleId: string): Promise<{ execution_id: string; message: string }> {
    try {
      const response = await apiClient.post<{ execution_id: string; message: string }>(`${this.baseUrl}/schedules/${scheduleId}/execute`);
      toast.success('Test execution triggered successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to trigger execution:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to trigger execution');
      throw error;
    }
  }

  /**
   * List executions for a schedule
   */
  async listScheduleExecutions(scheduleId: string, params?: ExecutionListParams): Promise<ExecutionListResponse> {
    try {
      const response = await apiClient.get<ExecutionListResponse>(`${this.baseUrl}/schedules/${scheduleId}/executions`, {
        params: {
          page: params?.page,
          page_size: params?.page_size
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to list executions:', error);
      toast.error('Failed to load executions');
      throw error;
    }
  }

  /**
   * List alerts
   */
  async listAlerts(params?: AlertListParams): Promise<AlertListResponse> {
    try {
      const response = await apiClient.get<AlertListResponse>(`${this.baseUrl}/alerts`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          severity: params?.severity,
          unresolved_only: params?.unresolved_only ? 'true' : undefined
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to list alerts:', error);
      toast.error('Failed to load alerts');
      throw error;
    }
  }

  /**
   * Acknowledge an alert
   */
  async acknowledgeAlert(alertId: string): Promise<{ message: string }> {
    try {
      const response = await apiClient.post<{ message: string }>(`${this.baseUrl}/alerts/${alertId}/acknowledge`);
      toast.success('Alert acknowledged successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to acknowledge alert');
      throw error;
    }
  }

  /**
   * Get monitoring dashboard data
   */
  async getMonitoringDashboard(workspaceId?: string): Promise<MonitoringDashboard> {
    try {
      const response = await apiClient.get<MonitoringDashboard>(`${this.baseUrl}/dashboard`, {
        params: { workspace_id: workspaceId }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get dashboard data:', error);
      toast.error('Failed to load dashboard data');
      throw error;
    }
  }

  /**
   * Delete scheduled test
   */
  async deleteSchedule(scheduleId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUrl}/schedules/${scheduleId}`);
      toast.success('Schedule deleted successfully');
    } catch (error) {
      console.error('Failed to delete schedule:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to delete schedule');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for schedule frequency
   */
  getFrequencyDisplayName(frequency: ScheduleFrequency): string {
    const displayNames: Record<ScheduleFrequency, string> = {
      hourly: 'Hourly',
      daily: 'Daily',
      weekly: 'Weekly',
      monthly: 'Monthly',
      custom_cron: 'Custom (Cron)'
    };
    return displayNames[frequency];
  }

  /**
   * Get color for schedule status
   */
  getStatusColor(status: ScheduleStatus): string {
    const colors: Record<ScheduleStatus, string> = {
      active: 'green',
      paused: 'yellow',
      disabled: 'gray',
      error: 'red'
    };
    return colors[status];
  }

  /**
   * Get display name for schedule status
   */
  getStatusDisplayName(status: ScheduleStatus): string {
    const displayNames: Record<ScheduleStatus, string> = {
      active: 'Active',
      paused: 'Paused',
      disabled: 'Disabled',
      error: 'Error'
    };
    return displayNames[status];
  }

  /**
   * Get color for alert severity
   */
  getAlertSeverityColor(severity: AlertSeverity): string {
    const colors: Record<AlertSeverity, string> = {
      info: 'blue',
      warning: 'yellow',
      critical: 'red'
    };
    return colors[severity];
  }

  /**
   * Get display name for alert severity
   */
  getAlertSeverityDisplayName(severity: AlertSeverity): string {
    const displayNames: Record<AlertSeverity, string> = {
      info: 'Info',
      warning: 'Warning',
      critical: 'Critical'
    };
    return displayNames[severity];
  }

  /**
   * Get display name for monitoring metric
   */
  getMetricDisplayName(metric: MonitoringMetric): string {
    const displayNames: Record<MonitoringMetric, string> = {
      success_rate: 'Success Rate',
      failure_count: 'Failure Count',
      response_time: 'Response Time',
      new_vulnerabilities: 'New Vulnerabilities',
      regression_detection: 'Regression Detection'
    };
    return displayNames[metric];
  }

  /**
   * Get display name for alert type
   */
  getAlertTypeDisplayName(type: AlertType): string {
    const displayNames: Record<AlertType, string> = {
      email: 'Email',
      webhook: 'Webhook',
      slack: 'Slack',
      teams: 'Microsoft Teams'
    };
    return displayNames[type];
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
   * Format next execution time
   */
  formatNextExecution(nextExecution?: string): string {
    if (!nextExecution) return 'Not scheduled';

    const now = new Date();
    const next = new Date(nextExecution);
    const diff = next.getTime() - now.getTime();

    if (diff < 0) {
      return 'Overdue';
    } else if (diff < 60000) {
      return 'In less than 1 minute';
    } else if (diff < 3600000) {
      const minutes = Math.floor(diff / 60000);
      return `In ${minutes} minute${minutes !== 1 ? 's' : ''}`;
    } else if (diff < 86400000) {
      const hours = Math.floor(diff / 3600000);
      return `In ${hours} hour${hours !== 1 ? 's' : ''}`;
    } else {
      const days = Math.floor(diff / 86400000);
      return `In ${days} day${days !== 1 ? 's' : ''}`;
    }
  }

  /**
   * Format last execution time
   */
  formatLastExecution(lastExecution?: string): string {
    if (!lastExecution) return 'Never executed';

    const now = new Date();
    const last = new Date(lastExecution);
    const diff = now.getTime() - last.getTime();

    if (diff < 60000) {
      return 'Just now';
    } else if (diff < 3600000) {
      const minutes = Math.floor(diff / 60000);
      return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    } else if (diff < 86400000) {
      const hours = Math.floor(diff / 3600000);
      return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    } else {
      const days = Math.floor(diff / 86400000);
      return `${days} day${days !== 1 ? 's' : ''} ago`;
    }
  }

  /**
   * Validate schedule create data
   */
  validateScheduleCreate(data: ScheduleCreate): string[] {
    const errors: string[] = [];

    if (!data.name || data.name.trim().length === 0) {
      errors.push('Schedule name is required');
    }

    if (data.name && data.name.length > 100) {
      errors.push('Schedule name must be less than 100 characters');
    }

    if (!data.description || data.description.trim().length === 0) {
      errors.push('Description is required');
    }

    if (data.description && data.description.length > 1000) {
      errors.push('Description must be less than 1000 characters');
    }

    if (!data.test_config || Object.keys(data.test_config).length === 0) {
      errors.push('Test configuration is required');
    }

    if (data.frequency === 'custom_cron' && !data.cron_expression) {
      errors.push('Cron expression is required for custom frequency');
    }

    return errors;
  }

  /**
   * Create default alert rule
   */
  createDefaultAlertRule(): AlertRule {
    return {
      rule_id: crypto.randomUUID(),
      name: 'Low Success Rate Alert',
      description: 'Alert when success rate drops below threshold',
      metric: 'success_rate',
      threshold_value: 90,
      comparison_operator: '<',
      alert_type: 'email',
      alert_target: '',
      severity: 'warning',
      cooldown_minutes: 60,
      enabled: true
    };
  }

  /**
   * Get available frequencies
   */
  getAvailableFrequencies(): Array<{id: ScheduleFrequency, name: string, description: string}> {
    return [
      {
        id: 'hourly',
        name: 'Hourly',
        description: 'Run every hour'
      },
      {
        id: 'daily',
        name: 'Daily',
        description: 'Run once per day'
      },
      {
        id: 'weekly',
        name: 'Weekly',
        description: 'Run once per week'
      },
      {
        id: 'monthly',
        name: 'Monthly',
        description: 'Run once per month'
      },
      {
        id: 'custom_cron',
        name: 'Custom',
        description: 'Use custom cron expression'
      }
    ];
  }

  /**
   * Get available metrics for alerting
   */
  getAvailableMetrics(): Array<{id: MonitoringMetric, name: string, description: string}> {
    return [
      {
        id: 'success_rate',
        name: 'Success Rate',
        description: 'Percentage of tests that passed'
      },
      {
        id: 'failure_count',
        name: 'Failure Count',
        description: 'Number of failed tests'
      },
      {
        id: 'new_vulnerabilities',
        name: 'New Vulnerabilities',
        description: 'Number of newly discovered vulnerabilities'
      },
      {
        id: 'response_time',
        name: 'Response Time',
        description: 'Average response time for tests'
      },
      {
        id: 'regression_detection',
        name: 'Regression Detection',
        description: 'Detection of security regressions'
      }
    ];
  }
}

// Export singleton instance
export const scheduledTestingService = new ScheduledTestingService();
