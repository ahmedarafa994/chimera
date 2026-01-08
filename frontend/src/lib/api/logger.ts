/**
 * API Logger
 * Comprehensive logging for API requests, responses, and errors
 */

// ============================================================================
// Types
// ============================================================================

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  data?: Record<string, any>;
  requestId?: string;
}

interface PerformanceLog {
  requestId: string;
  method: string;
  url: string;
  status: number;
  duration: number;
  timestamp: string;
}

// ============================================================================
// Configuration
// ============================================================================

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const CURRENT_LOG_LEVEL: LogLevel = 
  (process.env.NEXT_PUBLIC_LOG_LEVEL as LogLevel) || 
  (process.env.NODE_ENV === 'production' ? 'warn' : 'debug');

// ============================================================================
// Logger Implementation
// ============================================================================

class ApiLogger {
  private logs: LogEntry[] = [];
  private performanceLogs: PerformanceLog[] = [];
  private maxLogSize: number = 1000;
  private onLogCallback?: (entry: LogEntry) => void;

  private isEnabled(level: LogLevel): boolean {
    return LOG_LEVELS[level] >= LOG_LEVELS[CURRENT_LOG_LEVEL];
  }

  private formatEntry(level: LogLevel, message: string, data?: Record<string, any>): LogEntry {
    return {
      timestamp: new Date().toISOString(),
      level,
      message,
      data,
    };
  }

  private storeLog(entry: LogEntry): void {
    this.logs.push(entry);
    if (this.logs.length > this.maxLogSize) {
      this.logs = this.logs.slice(-this.maxLogSize);
    }
    if (this.onLogCallback) {
      this.onLogCallback(entry);
    }
  }

  private consoleOutput(entry: LogEntry): void {
    if (typeof window === 'undefined') return;

    const styles: Record<LogLevel, string> = {
      debug: 'color: #6b7280',
      info: 'color: #3b82f6',
      warn: 'color: #f59e0b',
      error: 'color: #ef4444',
    };

    const prefix = `[API ${entry.level.toUpperCase()}]`;
    const style = styles[entry.level];

    if (entry.data) {
      console.groupCollapsed(`%c${prefix} ${entry.message}`, style);
      console.log('Timestamp:', entry.timestamp);
      console.log('Data:', entry.data);
      console.groupEnd();
    } else {
      console.log(`%c${prefix} ${entry.message}`, style);
    }
  }

  logDebug(message: string, data?: Record<string, any>): void {
    if (!this.isEnabled('debug')) return;
    const entry = this.formatEntry('debug', message, data);
    this.storeLog(entry);
    this.consoleOutput(entry);
  }

  logInfo(message: string, data?: Record<string, any>): void {
    if (!this.isEnabled('info')) return;
    const entry = this.formatEntry('info', message, data);
    this.storeLog(entry);
    this.consoleOutput(entry);
  }

  logWarning(message: string, data?: Record<string, any>): void {
    if (!this.isEnabled('warn')) return;
    const entry = this.formatEntry('warn', message, data);
    this.storeLog(entry);
    this.consoleOutput(entry);
  }

  logError(message: string, error: Error | unknown, data?: Record<string, any>): void {
    if (!this.isEnabled('error')) return;
    const errorData = {
      ...data,
      error: error instanceof Error ? {
        name: error.name,
        message: error.message,
        stack: error.stack,
      } : error,
    };
    const entry = this.formatEntry('error', message, errorData);
    this.storeLog(entry);
    this.consoleOutput(entry);
  }

  logRequest(requestId: string, method: string, url: string, data?: Record<string, any>): void {
    this.logDebug(`${method} ${url}`, { requestId, ...data });
  }

  logResponse(requestId: string | undefined, status: number, duration: number, data?: any): void {
    const message = `Response ${status} in ${duration}ms`;
    if (requestId) {
      this.performanceLogs.push({
        requestId,
        method: '',
        url: '',
        status,
        duration,
        timestamp: new Date().toISOString(),
      });
    }
    if (status >= 400) {
      this.logWarning(message, { requestId, status, duration, data });
    } else {
      this.logDebug(message, { requestId, status, duration });
    }
  }

  logCacheHit(key: string): void {
    this.logDebug('Cache hit', { key });
  }

  logWebSocket(event: string, data?: Record<string, any>): void {
    this.logDebug(`WebSocket: ${event}`, data);
  }

  getLogs(): LogEntry[] {
    return [...this.logs];
  }

  getPerformanceMetrics(): {
    totalRequests: number;
    avgDuration: number;
    errorRate: number;
    p50Duration: number;
    p95Duration: number;
    p99Duration: number;
  } {
    if (this.performanceLogs.length === 0) {
      return { totalRequests: 0, avgDuration: 0, errorRate: 0, p50Duration: 0, p95Duration: 0, p99Duration: 0 };
    }
    const durations = this.performanceLogs.map(log => log.duration).sort((a, b) => a - b);
    const errors = this.performanceLogs.filter(log => log.status >= 400);
    const totalDuration = durations.reduce((sum, d) => sum + d, 0);
    const percentile = (arr: number[], p: number): number => {
      const index = Math.ceil((p / 100) * arr.length) - 1;
      return arr[Math.max(0, index)];
    };
    return {
      totalRequests: this.performanceLogs.length,
      avgDuration: totalDuration / this.performanceLogs.length,
      errorRate: errors.length / this.performanceLogs.length,
      p50Duration: percentile(durations, 50),
      p95Duration: percentile(durations, 95),
      p99Duration: percentile(durations, 99),
    };
  }

  clearLogs(): void {
    this.logs = [];
    this.performanceLogs = [];
  }

  setLogCallback(callback: (entry: LogEntry) => void): void {
    this.onLogCallback = callback;
  }

  exportLogs(): string {
    return JSON.stringify({
      logs: this.logs,
      performance: this.performanceLogs,
      metrics: this.getPerformanceMetrics(),
      exportedAt: new Date().toISOString(),
    }, null, 2);
  }
}

export const logger = new ApiLogger();