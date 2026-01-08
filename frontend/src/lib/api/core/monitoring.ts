/**
 * API Monitoring and Observability
 *
 * Provides request/response timing metrics, health check monitoring,
 * and alerting thresholds for API failures.
 *
 * @module lib/api/core/monitoring
 */

import { configManager } from './config';
import { APIError } from '../../errors';

// ============================================================================
// Types
// ============================================================================

export interface RequestMetrics {
  requestId: string;
  method: string;
  url: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  status?: number;
  success: boolean;
  error?: string;
  errorCode?: string;
  retries: number;
  cached: boolean;
  provider?: string;
  endpoint?: string;
}

export interface AggregatedMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  successRate: number;
  averageLatency: number;
  p50Latency: number;
  p95Latency: number;
  p99Latency: number;
  errorsByCode: Record<string, number>;
  requestsByEndpoint: Record<string, number>;
  requestsByProvider: Record<string, number>;
}

export interface HealthCheckResult {
  endpoint: string;
  healthy: boolean;
  latencyMs: number;
  lastCheck: number;
  consecutiveFailures: number;
  error?: string;
}

export interface AlertThreshold {
  metric: 'errorRate' | 'latency' | 'consecutiveFailures';
  threshold: number;
  window?: number; // Time window in ms for rate calculations
  callback: (value: number, threshold: number) => void;
}

export interface MonitoringConfig {
  enabled: boolean;
  maxStoredMetrics: number;
  metricsRetentionMs: number;
  healthCheckIntervalMs: number;
  alertThresholds: AlertThreshold[];
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: MonitoringConfig = {
  enabled: true,
  maxStoredMetrics: 1000,
  metricsRetentionMs: 60 * 60 * 1000, // 1 hour
  healthCheckIntervalMs: 30000, // 30 seconds
  alertThresholds: [],
};

// ============================================================================
// Request ID Generation
// ============================================================================

let requestCounter = 0;

export function generateRequestId(): string {
  const timestamp = Date.now().toString(36);
  const counter = (requestCounter++).toString(36).padStart(4, '0');
  const random = Math.random().toString(36).substring(2, 6);
  return `req_${timestamp}_${counter}_${random}`;
}

// ============================================================================
// Metrics Collector
// ============================================================================

class MetricsCollector {
  private metrics: RequestMetrics[] = [];
  private config: MonitoringConfig;
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor(config: Partial<MonitoringConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    if (typeof window !== 'undefined' && this.config.enabled) {
      this.cleanupInterval = setInterval(
        () => this.cleanup(),
        this.config.metricsRetentionMs / 4
      );
    }
  }

  /**
   * Record a request metric
   */
  record(metric: RequestMetrics): void {
    if (!this.config.enabled) return;

    this.metrics.push(metric);

    // Enforce max stored metrics
    if (this.metrics.length > this.config.maxStoredMetrics) {
      this.metrics = this.metrics.slice(-this.config.maxStoredMetrics);
    }

    // Check alert thresholds
    this.checkAlerts();

    // Log in development
    if (configManager.getConfig().logging.enabled) {
      this.logMetric(metric);
    }
  }

  /**
   * Start tracking a request
   */
  startRequest(method: string, url: string): RequestMetrics {
    return {
      requestId: generateRequestId(),
      method,
      url,
      startTime: Date.now(),
      success: false,
      retries: 0,
      cached: false,
    };
  }

  /**
   * Complete a request metric
   */
  completeRequest(
    metric: RequestMetrics,
    options: {
      status?: number;
      success: boolean;
      error?: Error;
      retries?: number;
      cached?: boolean;
      provider?: string;
      endpoint?: string;
    }
  ): RequestMetrics {
    const completed: RequestMetrics = {
      ...metric,
      endTime: Date.now(),
      duration: Date.now() - metric.startTime,
      status: options.status,
      success: options.success,
      retries: options.retries ?? metric.retries,
      cached: options.cached ?? metric.cached,
      provider: options.provider,
      endpoint: options.endpoint,
    };

    if (options.error) {
      completed.error = options.error.message;
      if (options.error instanceof APIError) {
        completed.errorCode = options.error.errorCode;
      } else if ((options.error as any).errorCode) {
        completed.errorCode = (options.error as any).errorCode;
      }
    }

    this.record(completed);
    return completed;
  }

  /**
   * Get aggregated metrics
   */
  getAggregatedMetrics(windowMs?: number): AggregatedMetrics {
    const cutoff = windowMs ? Date.now() - windowMs : 0;
    const relevantMetrics = this.metrics.filter(m => m.startTime >= cutoff);

    const totalRequests = relevantMetrics.length;
    const successfulRequests = relevantMetrics.filter(m => m.success).length;
    const failedRequests = totalRequests - successfulRequests;

    // Calculate latencies
    const latencies = relevantMetrics
      .filter(m => m.duration !== undefined)
      .map(m => m.duration!)
      .sort((a, b) => a - b);

    const averageLatency = latencies.length > 0
      ? latencies.reduce((a, b) => a + b, 0) / latencies.length
      : 0;

    const p50Latency = this.percentile(latencies, 50);
    const p95Latency = this.percentile(latencies, 95);
    const p99Latency = this.percentile(latencies, 99);

    // Count errors by code
    const errorsByCode: Record<string, number> = {};
    relevantMetrics
      .filter(m => m.errorCode)
      .forEach(m => {
        errorsByCode[m.errorCode!] = (errorsByCode[m.errorCode!] || 0) + 1;
      });

    // Count requests by endpoint
    const requestsByEndpoint: Record<string, number> = {};
    relevantMetrics
      .filter(m => m.endpoint)
      .forEach(m => {
        requestsByEndpoint[m.endpoint!] = (requestsByEndpoint[m.endpoint!] || 0) + 1;
      });

    // Count requests by provider
    const requestsByProvider: Record<string, number> = {};
    relevantMetrics
      .filter(m => m.provider)
      .forEach(m => {
        requestsByProvider[m.provider!] = (requestsByProvider[m.provider!] || 0) + 1;
      });

    return {
      totalRequests,
      successfulRequests,
      failedRequests,
      successRate: totalRequests > 0 ? successfulRequests / totalRequests : 1,
      averageLatency,
      p50Latency,
      p95Latency,
      p99Latency,
      errorsByCode,
      requestsByEndpoint,
      requestsByProvider,
    };
  }

  /**
   * Get recent metrics
   */
  getRecentMetrics(count: number = 100): RequestMetrics[] {
    return this.metrics.slice(-count);
  }

  /**
   * Get metrics for a specific endpoint
   */
  getEndpointMetrics(endpoint: string, windowMs?: number): RequestMetrics[] {
    const cutoff = windowMs ? Date.now() - windowMs : 0;
    return this.metrics.filter(
      m => m.endpoint === endpoint && m.startTime >= cutoff
    );
  }

  /**
   * Clear all metrics
   */
  clear(): void {
    this.metrics = [];
  }

  /**
   * Set alert thresholds
   */
  setAlertThresholds(thresholds: AlertThreshold[]): void {
    this.config.alertThresholds = thresholds;
  }

  /**
   * Destroy the collector
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.metrics = [];
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private percentile(sortedValues: number[], p: number): number {
    if (sortedValues.length === 0) return 0;
    const index = Math.ceil((p / 100) * sortedValues.length) - 1;
    return sortedValues[Math.max(0, index)];
  }

  private cleanup(): void {
    const cutoff = Date.now() - this.config.metricsRetentionMs;
    this.metrics = this.metrics.filter(m => m.startTime >= cutoff);
  }

  private checkAlerts(): void {
    for (const threshold of this.config.alertThresholds) {
      const windowMs = threshold.window || 60000; // Default 1 minute window
      const metrics = this.getAggregatedMetrics(windowMs);

      let value: number;
      switch (threshold.metric) {
        case 'errorRate':
          value = 1 - metrics.successRate;
          break;
        case 'latency':
          value = metrics.p95Latency;
          break;
        case 'consecutiveFailures':
          value = this.getConsecutiveFailures();
          break;
        default:
          continue;
      }

      if (value >= threshold.threshold) {
        threshold.callback(value, threshold.threshold);
      }
    }
  }

  private getConsecutiveFailures(): number {
    let count = 0;
    for (let i = this.metrics.length - 1; i >= 0; i--) {
      if (!this.metrics[i].success) {
        count++;
      } else {
        break;
      }
    }
    return count;
  }

  private logMetric(metric: RequestMetrics): void {
    const config = configManager.getConfig();
    if (!config.logging.enabled) return;

    const logData: Record<string, unknown> = {
      requestId: metric.requestId,
      method: metric.method,
      url: metric.url,
      success: metric.success,
    };

    if (config.logging.includeTimings && metric.duration !== undefined) {
      logData.duration = `${metric.duration}ms`;
    }

    if (metric.status) {
      logData.status = metric.status;
    }

    if (metric.error) {
      logData.error = metric.error;
    }

    if (metric.retries > 0) {
      logData.retries = metric.retries;
    }

    if (metric.cached) {
      logData.cached = true;
    }

    const level = metric.success ? 'info' : 'error';
    const message = `[API] ${metric.method} ${metric.url}`;

    if (config.logging.level === 'debug' || level === 'error') {
      console[level](message, logData);
    }
  }
}

// ============================================================================
// Health Check Monitor
// ============================================================================

class HealthCheckMonitor {
  private results: Map<string, HealthCheckResult> = new Map();
  private intervals: Map<string, ReturnType<typeof setInterval>> = new Map();
  private config: MonitoringConfig;

  constructor(config: Partial<MonitoringConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Register a health check endpoint
   */
  register(
    endpoint: string,
    checkFn: () => Promise<boolean>,
    intervalMs?: number
  ): void {
    // Initialize result
    this.results.set(endpoint, {
      endpoint,
      healthy: true,
      latencyMs: 0,
      lastCheck: 0,
      consecutiveFailures: 0,
    });

    // Start periodic check
    const interval = intervalMs || this.config.healthCheckIntervalMs;
    const intervalId = setInterval(
      () => this.performCheck(endpoint, checkFn),
      interval
    );
    this.intervals.set(endpoint, intervalId);

    // Perform initial check
    this.performCheck(endpoint, checkFn);
  }

  /**
   * Unregister a health check endpoint
   */
  unregister(endpoint: string): void {
    const intervalId = this.intervals.get(endpoint);
    if (intervalId) {
      clearInterval(intervalId);
      this.intervals.delete(endpoint);
    }
    this.results.delete(endpoint);
  }

  /**
   * Get health check result for an endpoint
   */
  getResult(endpoint: string): HealthCheckResult | undefined {
    return this.results.get(endpoint);
  }

  /**
   * Get all health check results
   */
  getAllResults(): HealthCheckResult[] {
    return Array.from(this.results.values());
  }

  /**
   * Check if all endpoints are healthy
   */
  isHealthy(): boolean {
    return Array.from(this.results.values()).every(r => r.healthy);
  }

  /**
   * Get unhealthy endpoints
   */
  getUnhealthyEndpoints(): HealthCheckResult[] {
    return Array.from(this.results.values()).filter(r => !r.healthy);
  }

  /**
   * Manually trigger a health check
   */
  async check(endpoint: string, checkFn: () => Promise<boolean>): Promise<HealthCheckResult> {
    return this.performCheck(endpoint, checkFn);
  }

  /**
   * Destroy the monitor
   */
  destroy(): void {
    this.intervals.forEach(intervalId => clearInterval(intervalId));
    this.intervals.clear();
    this.results.clear();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private async performCheck(
    endpoint: string,
    checkFn: () => Promise<boolean>
  ): Promise<HealthCheckResult> {
    const startTime = Date.now();
    let healthy = false;
    let error: string | undefined;

    try {
      healthy = await checkFn();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }

    const latencyMs = Date.now() - startTime;
    const existing = this.results.get(endpoint);
    const consecutiveFailures = healthy
      ? 0
      : (existing?.consecutiveFailures || 0) + 1;

    const result: HealthCheckResult = {
      endpoint,
      healthy,
      latencyMs,
      lastCheck: Date.now(),
      consecutiveFailures,
      error,
    };

    this.results.set(endpoint, result);
    return result;
  }
}

// ============================================================================
// Logger
// ============================================================================

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  data?: unknown;
  timestamp: string;
  requestId?: string;
}

class APILogger {
  private logs: LogEntry[] = [];
  private maxLogs: number = 1000;

  log(level: LogLevel, message: string, data?: unknown, requestId?: string): void {
    const config = configManager.getConfig();

    if (!config.logging.enabled) return;

    const levelPriority: Record<LogLevel, number> = {
      debug: 0,
      info: 1,
      warn: 2,
      error: 3,
    };

    if (levelPriority[level] < levelPriority[config.logging.level]) {
      return;
    }

    const entry: LogEntry = {
      level,
      message,
      data,
      timestamp: new Date().toISOString(),
      requestId,
    };

    this.logs.push(entry);

    // Enforce max logs
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs);
    }

    // Console output
    const prefix = requestId ? `[${requestId}]` : '';
    const fullMessage = `${prefix} ${message}`;

    switch (level) {
      case 'debug':
        console.debug(fullMessage, data);
        break;
      case 'info':
        console.info(fullMessage, data);
        break;
      case 'warn':
        console.warn(fullMessage, data);
        break;
      case 'error':
        console.error(fullMessage, data);
        break;
    }
  }

  debug(message: string, data?: unknown, requestId?: string): void {
    this.log('debug', message, data, requestId);
  }

  info(message: string, data?: unknown, requestId?: string): void {
    this.log('info', message, data, requestId);
  }

  warn(message: string, data?: unknown, requestId?: string): void {
    this.log('warn', message, data, requestId);
  }

  error(message: string, data?: unknown, requestId?: string): void {
    this.log('error', message, data, requestId);
  }

  getLogs(level?: LogLevel, count?: number): LogEntry[] {
    let filtered = level
      ? this.logs.filter(l => l.level === level)
      : this.logs;

    if (count) {
      filtered = filtered.slice(-count);
    }

    return filtered;
  }

  clear(): void {
    this.logs = [];
  }
}

// ============================================================================
// Singleton Instances
// ============================================================================

export const metricsCollector = new MetricsCollector();
export const healthCheckMonitor = new HealthCheckMonitor();
export const apiLogger = new APILogger();

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Create a request tracker for monitoring
 */
export function trackRequest(method: string, url: string): {
  metric: RequestMetrics;
  complete: (options: {
    status?: number;
    success: boolean;
    error?: Error;
    retries?: number;
    cached?: boolean;
    provider?: string;
    endpoint?: string;
  }) => RequestMetrics;
} {
  const metric = metricsCollector.startRequest(method, url);

  return {
    metric,
    complete: (options) => metricsCollector.completeRequest(metric, options),
  };
}

/**
 * Get current API health status
 */
export function getAPIHealth(): {
  healthy: boolean;
  metrics: AggregatedMetrics;
  healthChecks: HealthCheckResult[];
} {
  const metrics = metricsCollector.getAggregatedMetrics(60000); // Last minute
  const healthChecks = healthCheckMonitor.getAllResults();
  const healthy = metrics.successRate >= 0.95 && healthCheckMonitor.isHealthy();

  return {
    healthy,
    metrics,
    healthChecks,
  };
}

/**
 * Export metrics for external monitoring systems
 */
export function exportMetrics(): {
  timestamp: string;
  aggregated: AggregatedMetrics;
  recent: RequestMetrics[];
  healthChecks: HealthCheckResult[];
} {
  return {
    timestamp: new Date().toISOString(),
    aggregated: metricsCollector.getAggregatedMetrics(),
    recent: metricsCollector.getRecentMetrics(50),
    healthChecks: healthCheckMonitor.getAllResults(),
  };
}
