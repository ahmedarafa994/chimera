"use client";

import Link from "next/link";
import { useState } from "react";
import { HealthDashboard } from "@/components/health/HealthDashboard";
import {
  ProviderHealthDashboard,
  QuotaDashboard,
  RateLimitDashboard,
  type QuotaStatus,
  type RateLimitMetrics,
} from "@/components/health-dashboard";
import {
  useProviderHealthDashboard,
  useHealthHistory,
} from "@/hooks";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Activity,
  Heart,
  Gauge,
  BarChart3,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Key,
  Settings,
  Clock,
  TrendingUp,
  Zap,
  Server,
} from "lucide-react";
import type { TimeRange } from "@/components/health-dashboard";

/**
 * Enhanced Health Dashboard Page
 *
 * Main dashboard combining:
 * - System health monitoring (liveness, services)
 * - Provider health monitoring (status, latency, error rates)
 * - Quota usage tracking
 * - Rate limit visualization
 */
export default function HealthPage() {
  const [activeTab, setActiveTab] = useState<"providers" | "system" | "quota" | "rate-limits">("providers");
  const [timeRange, setTimeRange] = useState<TimeRange>("24h");

  // Fetch provider health data
  const {
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
  } = useProviderHealthDashboard({
    autoRefresh: true,
    refreshInterval: 30000,
  });

  // Fetch health history for charts
  const {
    data: historyData,
  } = useHealthHistory({
    timeRange,
    autoRefresh: true,
    refreshInterval: 60000,
  });

  // Handle time range change
  const handleTimeRangeChange = (range: TimeRange) => {
    setTimeRange(range);
  };

  // Transform quotas to QuotaStatus format for QuotaDashboard
  const quotaStatuses: QuotaStatus[] = quotas.map((q) => ({
    provider_id: q.provider_id,
    provider_name: q.provider_name,
    usage: q.usage_percent,
    limit: 100,
    usage_percent: q.usage_percent,
    tokens_used: q.tokens_used,
    tokens_limit: q.tokens_limit,
    tokens_percent: q.tokens_limit ? (q.tokens_used / q.tokens_limit) * 100 : null,
    requests_used: q.requests_used,
    requests_limit: q.requests_limit,
    requests_percent: q.requests_limit ? (q.requests_used / q.requests_limit) * 100 : null,
    period: q.period === "daily" ? "daily" : "monthly",
    period_start_at: null,
    reset_at: q.reset_at,
    cost_used: q.cost_usd || 0,
    cost_limit: null,
    cost_currency: "USD",
    warning_threshold_percent: 80,
    critical_threshold_percent: 95,
    is_warning: q.usage_percent >= 80 && q.usage_percent < 95,
    is_critical: q.usage_percent >= 95 && q.usage_percent < 100,
    is_exceeded: q.usage_percent >= 100,
    updated_at: null,
  }));

  // Transform rate limits to RateLimitMetrics format
  const rateLimitMetrics: RateLimitMetrics[] = Object.entries(rateLimits).map(([providerId, data]) => {
    const d = data as Record<string, unknown>;
    return {
      provider_id: providerId,
      provider_name: (d.provider_name as string) || providerId,
      requests_per_minute: 0,
      tokens_per_minute: 0,
      provider_rpm_cap: 60,
      provider_tpm_cap: 100000,
      custom_rpm_limit: null,
      custom_tpm_limit: null,
      effective_rpm_limit: 60,
      effective_tpm_limit: 100000,
      rpm_usage_percent: 0,
      tpm_usage_percent: 0,
      is_rate_limited: (d.is_rate_limited as boolean) || false,
      rate_limit_reset_at: null,
      rate_limit_retry_after_seconds: null,
      rate_limit_hits_last_hour: (d.rate_limit_hits_last_hour as number) || 0,
      rate_limit_hits_last_24h: 0,
      burst_capacity: 10,
      burst_remaining: 10,
      window_start_at: null,
      updated_at: null,
    };
  });

  // Calculate overall stats
  const operationalCount = summary?.operational || 0;
  const degradedCount = summary?.degraded || 0;
  const downCount = summary?.down || 0;
  const totalProviders = summary?.total_providers || providers.length;
  const activeAlerts = alerts.filter((a) => !a.resolved).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Heart className="h-8 w-8 text-rose-500" />
            Health Dashboard
          </h1>
          <p className="text-muted-foreground">
            Monitor real-time health status, performance metrics, and resource usage for all services.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/dashboard/api-keys">
            <Button variant="outline" size="sm">
              <Key className="h-4 w-4 mr-2" />
              API Keys
            </Button>
          </Link>
          <Link href="/dashboard/providers">
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Providers
            </Button>
          </Link>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refresh()}
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      <Separator />

      {/* Overview Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {/* Operational Card */}
        <Card className={operationalCount === totalProviders ? "border-emerald-500/30" : ""}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Operational</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold text-emerald-500">
                {operationalCount}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              of {totalProviders} providers
            </p>
          </CardContent>
        </Card>

        {/* Degraded Card */}
        <Card className={degradedCount > 0 ? "border-yellow-500/30" : ""}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Degraded</CardTitle>
            <AlertTriangle className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className={`text-2xl font-bold ${degradedCount > 0 ? "text-yellow-500" : "text-muted-foreground"}`}>
                {degradedCount}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              experiencing issues
            </p>
          </CardContent>
        </Card>

        {/* Down Card */}
        <Card className={downCount > 0 ? "border-red-500/30" : ""}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Down</CardTitle>
            <XCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className={`text-2xl font-bold ${downCount > 0 ? "text-red-500" : "text-muted-foreground"}`}>
                {downCount}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              not responding
            </p>
          </CardContent>
        </Card>

        {/* Active Alerts Card */}
        <Card className={activeAlerts > 0 ? "border-orange-500/30" : ""}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <Activity className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className={`text-2xl font-bold ${activeAlerts > 0 ? "text-orange-500" : "text-muted-foreground"}`}>
                {activeAlerts}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              require attention
            </p>
          </CardContent>
        </Card>

        {/* Last Updated Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Updated</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <div className="text-lg font-medium">
                {lastUpdated ? lastUpdated.toLocaleTimeString() : "Never"}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              auto-refresh: 30s
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
        <TabsList className="grid w-full grid-cols-4 max-w-2xl">
          <TabsTrigger value="providers" className="gap-2">
            <Zap className="h-4 w-4" />
            <span className="hidden sm:inline">Providers</span>
          </TabsTrigger>
          <TabsTrigger value="system" className="gap-2">
            <Server className="h-4 w-4" />
            <span className="hidden sm:inline">System</span>
          </TabsTrigger>
          <TabsTrigger value="quota" className="gap-2">
            <BarChart3 className="h-4 w-4" />
            <span className="hidden sm:inline">Quotas</span>
          </TabsTrigger>
          <TabsTrigger value="rate-limits" className="gap-2">
            <Gauge className="h-4 w-4" />
            <span className="hidden sm:inline">Rate Limits</span>
          </TabsTrigger>
        </TabsList>

        {/* Provider Health Tab */}
        <TabsContent value="providers" className="mt-6">
          <ProviderHealthDashboard
            providers={providers}
            summary={summary}
            alerts={alerts}
            history={historyData}
            isLoading={isLoading}
            isRefreshing={isRefreshing}
            error={error}
            lastUpdated={lastUpdated}
            onRefresh={refresh}
            onRefreshProvider={refreshProvider}
            onAcknowledgeAlert={acknowledgeAlert}
            onResolveAlert={resolveAlert}
            onTimeRangeChange={handleTimeRangeChange}
            showCharts={true}
            showAlerts={true}
          />
        </TabsContent>

        {/* System Health Tab */}
        <TabsContent value="system" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                System Services
              </CardTitle>
              <CardDescription>
                Monitor the health status of core platform services and dependencies.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <HealthDashboard autoRefresh={true} refreshInterval={30000} />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Quota Usage Tab */}
        <TabsContent value="quota" className="mt-6">
          {isLoading ? (
            <Card>
              <CardHeader>
                <Skeleton className="h-6 w-48" />
                <Skeleton className="h-4 w-64" />
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-48" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : quotaStatuses.length > 0 ? (
            <QuotaDashboard
              quotas={quotaStatuses}
              isLoading={isLoading}
              isRefreshing={isRefreshing}
              error={error}
              onRefresh={refresh}
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Quota Usage
                </CardTitle>
                <CardDescription>
                  No quota data available. Configure API keys to track usage.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <BarChart3 className="h-16 w-16 text-muted-foreground/50 mb-4" />
                <p className="text-muted-foreground text-center max-w-md">
                  Quota tracking will be available once you configure API keys for your providers.
                </p>
                <Link href="/dashboard/api-keys" className="mt-4">
                  <Button>
                    <Key className="h-4 w-4 mr-2" />
                    Configure API Keys
                  </Button>
                </Link>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Rate Limits Tab */}
        <TabsContent value="rate-limits" className="mt-6">
          {isLoading ? (
            <Card>
              <CardHeader>
                <Skeleton className="h-6 w-48" />
                <Skeleton className="h-4 w-64" />
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-48" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : rateLimitMetrics.length > 0 ? (
            <RateLimitDashboard
              providers={rateLimitMetrics}
              isLoading={isLoading}
              isRefreshing={isRefreshing}
              error={error}
              onRefresh={refresh}
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Gauge className="h-5 w-5" />
                  Rate Limits
                </CardTitle>
                <CardDescription>
                  No rate limit data available. Configure API keys to track rate limits.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Gauge className="h-16 w-16 text-muted-foreground/50 mb-4" />
                <p className="text-muted-foreground text-center max-w-md">
                  Rate limit monitoring will be available once you configure API keys and start making requests.
                </p>
                <Link href="/dashboard/api-keys" className="mt-4">
                  <Button>
                    <Key className="h-4 w-4 mr-2" />
                    Configure API Keys
                  </Button>
                </Link>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Quick Stats Footer */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card className="hover:bg-muted/50 transition-colors">
          <Link href="/dashboard/api-keys">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Key className="h-5 w-5 text-primary" />
                API Key Management
              </CardTitle>
              <CardDescription>
                Add, edit, and test API keys. Configure failover and view usage statistics.
              </CardDescription>
            </CardHeader>
          </Link>
        </Card>

        <Card className="hover:bg-muted/50 transition-colors">
          <Link href="/dashboard/providers">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Settings className="h-5 w-5 text-blue-500" />
                Provider Configuration
              </CardTitle>
              <CardDescription>
                Configure provider settings, default models, and rate limit policies.
              </CardDescription>
            </CardHeader>
          </Link>
        </Card>

        <Card className="hover:bg-muted/50 transition-colors">
          <Link href="/dashboard/metrics">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <TrendingUp className="h-5 w-5 text-emerald-500" />
                Performance Metrics
              </CardTitle>
              <CardDescription>
                View detailed performance metrics, historical trends, and cost analytics.
              </CardDescription>
            </CardHeader>
          </Link>
        </Card>
      </div>
    </div>
  );
}
