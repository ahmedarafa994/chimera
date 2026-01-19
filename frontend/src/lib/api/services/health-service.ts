/**
 * Health Monitoring Service - Aligned with Backend API
 *
 * This service is properly aligned with backend endpoints:
 * - GET /api/v1/health (basic health check)
 * - GET /api/v1/health/ready (readiness probe)
 * - GET /api/v1/health/full (comprehensive health)
 * - GET /api/v1/health/integration (service dependencies)
 * - GET /api/v1/metrics (system metrics)
 * - GET /api/v1/integration/stats (integration statistics)
 */

import { apiClient } from '../client';
import { apiErrorHandler } from '../../errors/api-error-handler';

// ============================================================================
// Types (matching backend response structures)
// ============================================================================

export interface BasicHealthResponse {
  status: string;
  timestamp: string;
}

export interface ReadinessResponse {
  status: 'ready' | 'not_ready';
  timestamp: string;
  checks: Array<{
    name: string;
    status: 'pass' | 'fail';
    latency_ms?: number;
    error?: string;
  }>;
}

export interface FullHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version?: string;
  uptime_seconds?: number;
  services: Array<{
    name: string;
    status: 'healthy' | 'degraded' | 'unhealthy';
    latency_ms?: number;
    last_check?: string;
    error?: string;
    details?: Record<string, any>;
  }>;
  providers: Array<{
    provider: string;
    status: string;
    latency_ms?: number;
    error_message?: string;
  }>;
}

export interface IntegrationHealthResponse {
  status: string;
  dependencies: Array<{
    name: string;
    type: string;
    status: 'connected' | 'disconnected' | 'degraded';
    latency_ms?: number;
    version?: string;
    error?: string;
  }>;
  graph: {
    nodes: Array<{
      id: string;
      name: string;
      type: string;
      status: string;
    }>;
    edges: Array<{
      from: string;
      to: string;
      relationship: string;
    }>;
  };
}

export interface MetricsResponse {
  timestamp: string;
  metrics: {
    status: string;
    cache: {
      enabled: boolean;
      entries: number;
      hit_rate?: number;
      memory_usage?: number;
    };
    providers: Record<string, {
      requests: number;
      success_rate: number;
      average_latency_ms: number;
      error_count: number;
    }>;
    system?: {
      cpu_percent?: number;
      memory_percent?: number;
      disk_usage?: number;
      active_connections?: number;
    };
    requests?: {
      total: number;
      per_second: number;
      average_latency_ms: number;
      error_rate: number;
    };
  };
}

export interface IntegrationStatsResponse {
  total_services: number;
  active_services: number;
  failed_services: number;
  average_latency_ms: number;
  services: Array<{
    name: string;
    status: string;
    requests_count: number;
    success_rate: number;
    average_latency_ms: number;
  }>;
  timestamp: string;
}

// ============================================================================
// Health Monitoring Service Implementation
// ============================================================================

export class HealthMonitoringService {
  /**
   * Basic health check (liveness probe)
   */
  async getBasicHealth() {
    try {
      const response = await apiClient.get<BasicHealthResponse>('/api/v1/health');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetBasicHealth');
    }
  }

  /**
   * Readiness probe with dependency checks
   */
  async getReadinessCheck() {
    try {
      const response = await apiClient.get<ReadinessResponse>('/api/v1/health/ready');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetReadinessCheck');
    }
  }

  /**
   * Comprehensive health check with all services
   */
  async getFullHealth() {
    try {
      const response = await apiClient.get<FullHealthResponse>('/api/v1/health/full');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetFullHealth');
    }
  }

  /**
   * Integration health with service dependency graph
   */
  async getIntegrationHealth() {
    try {
      const response = await apiClient.get<IntegrationHealthResponse>('/api/v1/health/integration');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetIntegrationHealth');
    }
  }

  /**
   * System metrics and performance data
   */
  async getMetrics() {
    try {
      const response = await apiClient.get<MetricsResponse>('/api/v1/metrics');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetMetrics');
    }
  }

  /**
   * Integration service statistics
   */
  async getIntegrationStats() {
    try {
      const response = await apiClient.get<IntegrationStatsResponse>('/api/v1/integration/stats');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetIntegrationStats');
    }
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Check if the system is healthy
   */
  async isHealthy(): Promise<boolean> {
    try {
      const health = await this.getBasicHealth();
      return health.data.status === 'healthy' || health.data.status === 'ok';
    } catch {
      return false;
    }
  }

  /**
   * Check if the system is ready to serve traffic
   */
  async isReady(): Promise<boolean> {
    try {
      const readiness = await this.getReadinessCheck();
      return readiness.data.status === 'ready';
    } catch {
      return false;
    }
  }

  /**
   * Get overall system status
   */
  async getSystemStatus(): Promise<'healthy' | 'degraded' | 'unhealthy'> {
    try {
      const health = await this.getFullHealth();
      return health.data.status;
    } catch {
      return 'unhealthy';
    }
  }

  /**
   * Get provider health summary
   */
  async getProviderHealthSummary() {
    try {
      const health = await this.getFullHealth();
      return health.data.providers.map(provider => ({
        name: provider.provider,
        status: provider.status,
        latency: provider.latency_ms,
        hasError: !!provider.error_message,
      }));
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetProviderHealthSummary');
    }
  }

  /**
   * Get service health summary
   */
  async getServiceHealthSummary() {
    try {
      const health = await this.getFullHealth();
      return health.data.services.map(service => ({
        name: service.name,
        status: service.status,
        latency: service.latency_ms,
        lastCheck: service.last_check,
        hasError: !!service.error,
      }));
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetServiceHealthSummary');
    }
  }

  /**
   * Calculate overall health score (0-100)
   */
  async calculateHealthScore(): Promise<number> {
    try {
      const health = await this.getFullHealth();
      let score = 0;
      let total = 0;

      // Check main status
      if (health.data.status === 'healthy') score += 40;
      else if (health.data.status === 'degraded') score += 20;
      total += 40;

      // Check services
      const healthyServices = health.data.services.filter(s => s.status === 'healthy').length;
      const serviceScore = (healthyServices / health.data.services.length) * 30;
      score += serviceScore;
      total += 30;

      // Check providers
      const healthyProviders = health.data.providers.filter(p => p.status === 'healthy').length;
      const providerScore = (healthyProviders / health.data.providers.length) * 30;
      score += providerScore;
      total += 30;

      return Math.round((score / total) * 100);
    } catch {
      return 0;
    }
  }

  /**
   * Get performance metrics summary
   */
  async getPerformanceSummary() {
    try {
      const metrics = await this.getMetrics();
      const data = metrics.data.metrics;

      return {
        cacheHitRate: data.cache?.hit_rate || 0,
        averageLatency: data.requests?.average_latency_ms || 0,
        errorRate: data.requests?.error_rate || 0,
        requestsPerSecond: data.requests?.per_second || 0,
        systemCpu: data.system?.cpu_percent || 0,
        systemMemory: data.system?.memory_percent || 0,
        activeConnections: data.system?.active_connections || 0,
      };
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetPerformanceSummary');
    }
  }

  /**
   * Monitor system health continuously
   */
  startHealthMonitoring(
    callback: (status: { healthy: boolean; status: string; score: number }) => void,
    intervalMs: number = 30000
  ): () => void {
    const checkHealth = async () => {
      try {
        const [isHealthy, status, score] = await Promise.all([
          this.isHealthy(),
          this.getSystemStatus(),
          this.calculateHealthScore(),
        ]);

        callback({ healthy: isHealthy, status, score });
      } catch (error) {
        callback({ healthy: false, status: 'error', score: 0 });
      }
    };

    // Initial check
    checkHealth();

    // Set up interval
    const interval = setInterval(checkHealth, intervalMs);

    // Return cleanup function
    return () => clearInterval(interval);
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const healthMonitoringService = new HealthMonitoringService();

// ============================================================================
// Convenience functions for direct usage
// ============================================================================

export const healthApi = {
  getBasicHealth: () => healthMonitoringService.getBasicHealth(),
  getReadinessCheck: () => healthMonitoringService.getReadinessCheck(),
  getFullHealth: () => healthMonitoringService.getFullHealth(),
  getIntegrationHealth: () => healthMonitoringService.getIntegrationHealth(),
  getMetrics: () => healthMonitoringService.getMetrics(),
  getIntegrationStats: () => healthMonitoringService.getIntegrationStats(),

  // Convenience methods
  isHealthy: () => healthMonitoringService.isHealthy(),
  isReady: () => healthMonitoringService.isReady(),
  getSystemStatus: () => healthMonitoringService.getSystemStatus(),
  getProviderHealthSummary: () => healthMonitoringService.getProviderHealthSummary(),
  getServiceHealthSummary: () => healthMonitoringService.getServiceHealthSummary(),
  calculateHealthScore: () => healthMonitoringService.calculateHealthScore(),
  getPerformanceSummary: () => healthMonitoringService.getPerformanceSummary(),
  startHealthMonitoring: (
    callback: (status: { healthy: boolean; status: string; score: number }) => void,
    intervalMs?: number
  ) => healthMonitoringService.startHealthMonitoring(callback, intervalMs),
};

export default healthMonitoringService;