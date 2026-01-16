"use client";

import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import type {
  ProviderHealthMetrics,
  ProviderStatus,
  HealthHistoryEntry,
  TimeRange,
} from "@/components/health-dashboard";

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const HEALTH_ENDPOINT = `${API_BASE_URL}/api/v1/providers/health`;

// =============================================================================
// Types
// =============================================================================

interface HealthDashboardResponse {
  status: string;
  providers: Record<string, ProviderHealthData>;
  summary: HealthSummary;
  alerts: HealthAlert[];
  monitoring: MonitoringConfig;
  updated_at: string;
}

interface ProviderHealthData {
  provider_id: string;
  provider_name: string;
  status: ProviderStatus;
  latency?: {
    current_ms: number;
    avg_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    min_ms: number;
    max_ms: number;
  };
  requests?: {
    total: number;
    successful: number;
    failed: number;
    rate_limited: number;
    error_rate_percent: number;
    success_rate_percent: number;
  };
  availability?: {
    uptime_percent: number;
  };
  circuit_breaker?: {
    state: "closed" | "open" | "half_open";
    failure_count: number;
  };
  timestamps?: {
    last_check: string | null;
    last_success: string | null;
    last_failure: string | null;
  };
}

interface HealthSummary {
  total_providers: number;
  operational: number;
  degraded: number;
  down: number;
  unknown: number;
  overall_status: ProviderStatus;
}

interface HealthAlert {
  id: string;
  provider_id: string;
  severity: "info" | "warning" | "critical";
  message: string;
  created_at: string;
  acknowledged: boolean;
  resolved: boolean;
}

interface MonitoringConfig {
  running: boolean;
  check_interval_seconds: number;
}

interface HealthMetricsResponse {
  providers: Record<
    string,
    {
      status: ProviderStatus;
      latency_ms: number;
      error_rate: number;
      uptime_percent: number;
      circuit_breaker: string;
      last_check: string | null;
    }
  >;
  summary: HealthSummary;
  updated_at: string;
}

interface HealthHistoryResponse {
  provider_id: string | null;
  entries: Array<{
    timestamp: string;
    latency_ms: number;
    error_rate: number;
    success_rate: number;
    uptime_percent?: number;
    total_requests?: number;
    failed_requests?: number;
  }>;
  total_count: number;
}

interface QuotaDashboardResponse {
  providers: Record<string, QuotaStatus>;
  summary: Record<string, unknown>;
  alerts: QuotaAlert[];
  updated_at: string;
}

interface QuotaStatus {
  provider_id: string;
  provider_name: string;
  requests_used: number;
  requests_limit: number;
  tokens_used: number;
  tokens_limit: number;
  cost_usd: number;
  usage_percent: number;
  period: "daily" | "monthly";
  reset_at: string;
}

interface QuotaAlert {
  id: string;
  provider_id: string;
  severity: "warning" | "critical";
  message: string;
  threshold_percent: number;
  current_percent: number;
}

interface RateLimitDashboardResponse {
  providers: Record<
    string,
    {
      provider_id: string;
      provider_name: string;
      is_rate_limited: boolean;
      rate_limit_hits_last_hour: number;
      requests_total: number;
      status: string;
    }
  >;
  summary: {
    total_providers: number;
    currently_rate_limited: number;
    approaching_limit: number;
  };
  updated_at: string;
}

export interface UseProviderHealthOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  providerId?: string;
}

export interface UseProviderHealthResult {
  // Dashboard data
  providers: ProviderHealthMetrics[];
  summary: HealthSummary | null;
  alerts: HealthAlert[];
  quotas: QuotaStatus[];
  rateLimits: Record<string, unknown>;

  // State
  isLoading: boolean;
  isRefreshing: boolean;
  error: Error | null;
  lastUpdated: Date | null;

  // Actions
  refresh: () => Promise<void>;
  refreshProvider: (providerId: string) => Promise<void>;
  acknowledgeAlert: (alertId: string) => Promise<void>;
  resolveAlert: (alertId: string) => Promise<void>;
  triggerHealthCheck: (providerId?: string) => Promise<void>;
}

export interface UseHealthHistoryOptions {
  providerId?: string;
  timeRange?: TimeRange;
  limit?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export interface UseHealthHistoryResult {
  data: HealthHistoryEntry[];
  isLoading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

// =============================================================================
// API Helpers
// =============================================================================

async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  // Add API key header if available
  const apiKey =
    typeof window !== "undefined"
      ? localStorage.getItem("chimera_api_key") || ""
      : "";

  if (apiKey) {
    (headers as Record<string, string>)["X-API-Key"] = apiKey;
  }

  const response = await fetch(`${HEALTH_ENDPOINT}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `API request failed: ${response.status}`);
  }

  return response.json();
}

// =============================================================================
// Transform Functions
// =============================================================================

function transformProviderData(data: ProviderHealthData): ProviderHealthMetrics {
  return {
    provider_id: data.provider_id,
    provider_name: data.provider_name || data.provider_id,
    status: data.status || "unknown",
    latency_ms: data.latency?.current_ms || data.latency?.avg_ms || 0,
    latency_trend: "stable", // TODO: Calculate from history
    error_rate: data.requests?.error_rate_percent || 0,
    error_rate_trend: "stable", // TODO: Calculate from history
    uptime_percent: data.availability?.uptime_percent || 100,
    circuit_breaker_state: data.circuit_breaker?.state || "closed",
    last_check: data.timestamps?.last_check || null,
    total_requests: data.requests?.total || 0,
    successful_requests: data.requests?.successful || 0,
    failed_requests: data.requests?.failed || 0,
    rate_limited_requests: data.requests?.rate_limited || 0,
  };
}

// =============================================================================
// useProviderHealth Hook
// =============================================================================

export function useProviderHealthDashboard(
  options: UseProviderHealthOptions = {}
): UseProviderHealthResult {
  const { autoRefresh = true, refreshInterval = 30000 } = options;

  const [providers, setProviders] = useState<ProviderHealthMetrics[]>([]);
  const [summary, setSummary] = useState<HealthSummary | null>(null);
  const [alerts, setAlerts] = useState<HealthAlert[]>([]);
  const [quotas, setQuotas] = useState<QuotaStatus[]>([]);
  const [rateLimits, setRateLimits] = useState<Record<string, unknown>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch all dashboard data
  const refresh = useCallback(async () => {
    const isInitialLoad = providers.length === 0;
    if (isInitialLoad) {
      setIsLoading(true);
    } else {
      setIsRefreshing(true);
    }

    try {
      setError(null);

      // Fetch all data in parallel
      const [dashboardResponse, quotaResponse, rateLimitResponse] = await Promise.all([
        apiRequest<HealthDashboardResponse>("/dashboard"),
        apiRequest<QuotaDashboardResponse>("/quota").catch(() => null),
        apiRequest<RateLimitDashboardResponse>("/rate-limits").catch(() => null),
      ]);

      // Transform provider data
      const transformedProviders = Object.values(dashboardResponse.providers || {}).map(
        transformProviderData
      );

      setProviders(transformedProviders);
      setSummary(dashboardResponse.summary);
      setAlerts(dashboardResponse.alerts || []);
      setLastUpdated(new Date());

      // Set quota data
      if (quotaResponse) {
        setQuotas(Object.values(quotaResponse.providers || {}));
      }

      // Set rate limit data
      if (rateLimitResponse) {
        setRateLimits(rateLimitResponse.providers || {});
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Failed to fetch health data"));
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [providers.length]);

  // Refresh a specific provider
  const refreshProvider = useCallback(async (providerId: string) => {
    try {
      await apiRequest<{ success: boolean }>(`/check?provider_id=${providerId}`, {
        method: "POST",
      });
      // Refresh all data after triggering a check
      await refresh();
    } catch (err) {
      throw err instanceof Error ? err : new Error("Failed to refresh provider");
    }
  }, [refresh]);

  // Acknowledge an alert
  const acknowledgeAlert = useCallback(async (alertId: string) => {
    await apiRequest<{ success: boolean }>(`/alerts/${alertId}/acknowledge`, {
      method: "POST",
    });
    // Update local state
    setAlerts((prev) =>
      prev.map((a) => (a.id === alertId ? { ...a, acknowledged: true } : a))
    );
  }, []);

  // Resolve an alert
  const resolveAlert = useCallback(async (alertId: string) => {
    await apiRequest<{ success: boolean }>(`/alerts/${alertId}/resolve`, {
      method: "POST",
    });
    // Update local state
    setAlerts((prev) =>
      prev.map((a) => (a.id === alertId ? { ...a, resolved: true } : a))
    );
  }, []);

  // Trigger a health check
  const triggerHealthCheck = useCallback(
    async (providerId?: string) => {
      const endpoint = providerId
        ? `/check?provider_id=${providerId}`
        : "/check";
      await apiRequest<{ success: boolean }>(endpoint, {
        method: "POST",
      });
      // Refresh after triggering check
      await refresh();
    },
    [refresh]
  );

  // Initial fetch
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(refresh, refreshInterval);
      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, refresh]);

  return {
    providers,
    summary,
    alerts,
    quotas,
    rateLimits,
    isLoading,
    isRefreshing,
    error,
    lastUpdated,
    refresh,
    refreshProvider,
    acknowledgeAlert,
    resolveAlert,
    triggerHealthCheck,
  };
}

// =============================================================================
// useHealthHistory Hook
// =============================================================================

export function useHealthHistory(
  options: UseHealthHistoryOptions = {}
): UseHealthHistoryResult {
  const {
    providerId,
    timeRange = "24h",
    limit = 100,
    autoRefresh = false,
    refreshInterval = 60000,
  } = options;

  const [data, setData] = useState<HealthHistoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Calculate start time based on time range
  const startTime = useMemo(() => {
    const now = new Date();
    switch (timeRange) {
      case "1h":
        return new Date(now.getTime() - 60 * 60 * 1000);
      case "6h":
        return new Date(now.getTime() - 6 * 60 * 60 * 1000);
      case "24h":
        return new Date(now.getTime() - 24 * 60 * 60 * 1000);
      case "7d":
        return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      case "30d":
        return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      default:
        return new Date(now.getTime() - 24 * 60 * 60 * 1000);
    }
  }, [timeRange]);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      setError(null);

      const params = new URLSearchParams({
        limit: limit.toString(),
        start_time: startTime.toISOString(),
      });

      if (providerId) {
        params.set("provider_id", providerId);
      }

      const response = await apiRequest<HealthHistoryResponse>(
        `/history?${params.toString()}`
      );

      setData(response.entries);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Failed to fetch history"));
    } finally {
      setIsLoading(false);
    }
  }, [providerId, limit, startTime]);

  // Initial fetch and refetch on dependency change
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(refresh, refreshInterval);
      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, refresh]);

  return {
    data,
    isLoading,
    error,
    refresh,
  };
}

// =============================================================================
// useProviderHealthDetail Hook (for single provider)
// =============================================================================

export interface ProviderHealthDetail extends ProviderHealthMetrics {
  uptime_history?: {
    last_hour?: number;
    last_24_hours?: number;
    last_7_days?: number;
    last_30_days?: number;
  };
  quota?: QuotaStatus;
}

export interface UseProviderHealthDetailResult {
  provider: ProviderHealthDetail | null;
  history: HealthHistoryEntry[];
  isLoading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

export function useProviderHealthDetail(
  providerId: string,
  options: { autoRefresh?: boolean; refreshInterval?: number } = {}
): UseProviderHealthDetailResult {
  const { autoRefresh = true, refreshInterval = 30000 } = options;

  const [provider, setProvider] = useState<ProviderHealthDetail | null>(null);
  const [history, setHistory] = useState<HealthHistoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      setError(null);

      // Fetch provider details and history in parallel
      const [detailResponse, historyResponse] = await Promise.all([
        apiRequest<{
          provider_id: string;
          health: ProviderHealthData;
          uptime: Record<string, unknown>;
          quota: QuotaStatus | null;
        }>(`/${providerId}`),
        apiRequest<HealthHistoryResponse>(`/history?provider_id=${providerId}&limit=100`),
      ]);

      const transformedProvider = transformProviderData(detailResponse.health);

      setProvider({
        ...transformedProvider,
        uptime_history: {
          last_hour: (detailResponse.uptime?.last_hour as { uptime_percent?: number })?.uptime_percent,
          last_24_hours: (detailResponse.uptime?.last_24_hours as { uptime_percent?: number })?.uptime_percent,
          last_7_days: (detailResponse.uptime?.last_7_days as { uptime_percent?: number })?.uptime_percent,
          last_30_days: (detailResponse.uptime?.last_30_days as { uptime_percent?: number })?.uptime_percent,
        },
        quota: detailResponse.quota || undefined,
      });

      setHistory(historyResponse.entries);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Failed to fetch provider details"));
    } finally {
      setIsLoading(false);
    }
  }, [providerId]);

  // Initial fetch
  useEffect(() => {
    if (providerId) {
      refresh();
    }
  }, [providerId, refresh]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0 && providerId) {
      refreshTimerRef.current = setInterval(refresh, refreshInterval);
      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, refresh, providerId]);

  return {
    provider,
    history,
    isLoading,
    error,
    refresh,
  };
}

export default useProviderHealthDashboard;
