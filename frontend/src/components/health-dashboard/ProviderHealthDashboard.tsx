"use client";

import * as React from "react";
import { useState, useCallback, useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Activity,
  AlertTriangle,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Clock,
  TrendingUp,
  Bell,
  BellOff,
  Gauge,
} from "lucide-react";
import { toast } from "sonner";

import { ProviderHealthCard, type ProviderHealthMetrics, type ProviderStatus } from "./ProviderHealthCard";
import { HealthMetricsChart, type HealthHistoryEntry, type TimeRange } from "./HealthMetricsChart";
import { UptimeBadges, CircularUptime } from "./UptimeIndicator";

// =============================================================================
// Types
// =============================================================================

export interface HealthSummary {
  total_providers: number;
  operational: number;
  degraded: number;
  down: number;
  unknown: number;
  overall_status: ProviderStatus;
}

export interface HealthAlert {
  id: string;
  provider_id: string;
  severity: "info" | "warning" | "critical";
  message: string;
  created_at: string;
  acknowledged: boolean;
  resolved: boolean;
}

export interface ProviderHealthDashboardProps {
  // Data
  providers: ProviderHealthMetrics[];
  summary: HealthSummary | null;
  alerts: HealthAlert[];
  history?: HealthHistoryEntry[];

  // State
  isLoading?: boolean;
  isRefreshing?: boolean;
  error?: Error | null;
  lastUpdated?: Date | null;

  // Callbacks
  onRefresh?: () => Promise<void>;
  onRefreshProvider?: (providerId: string) => Promise<void>;
  onAcknowledgeAlert?: (alertId: string) => Promise<void>;
  onResolveAlert?: (alertId: string) => Promise<void>;
  onViewProviderDetails?: (providerId: string) => void;
  onTimeRangeChange?: (range: TimeRange) => void;

  // Options
  autoRefresh?: boolean;
  refreshInterval?: number;
  showCharts?: boolean;
  showAlerts?: boolean;
  className?: string;
}

// =============================================================================
// Status Configuration
// =============================================================================

const overallStatusConfig: Record<
  ProviderStatus,
  {
    icon: React.ElementType;
    color: string;
    bgColor: string;
    label: string;
  }
> = {
  operational: {
    icon: CheckCircle2,
    color: "text-emerald-500",
    bgColor: "bg-emerald-500/10",
    label: "All Systems Operational",
  },
  degraded: {
    icon: AlertTriangle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-500/10",
    label: "Some Systems Degraded",
  },
  down: {
    icon: XCircle,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    label: "Critical Issues Detected",
  },
  unknown: {
    icon: AlertTriangle,
    color: "text-gray-500",
    bgColor: "bg-gray-500/10",
    label: "Status Unknown",
  },
};

const severityConfig = {
  info: { color: "text-blue-500", bgColor: "bg-blue-500/10", borderColor: "border-blue-500/30" },
  warning: { color: "text-yellow-500", bgColor: "bg-yellow-500/10", borderColor: "border-yellow-500/30" },
  critical: { color: "text-red-500", bgColor: "bg-red-500/10", borderColor: "border-red-500/30" },
};

// =============================================================================
// Sub-Components
// =============================================================================

interface SummaryCardProps {
  summary: HealthSummary;
  isRefreshing: boolean;
  onRefresh: () => void;
}

function SummaryCard({ summary, isRefreshing, onRefresh }: SummaryCardProps) {
  const config = overallStatusConfig[summary.overall_status];
  const StatusIcon = config.icon;
  const avgUptime = summary.operational > 0
    ? ((summary.operational / summary.total_providers) * 100)
    : 0;

  return (
    <Card className={cn("border-2", `border-${summary.overall_status === "operational" ? "emerald" : summary.overall_status === "degraded" ? "yellow" : "red"}-500/30`)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn("p-3 rounded-full", config.bgColor)}>
              <StatusIcon className={cn("h-8 w-8", config.color)} />
            </div>
            <div>
              <CardTitle className="text-2xl">Provider Health Dashboard</CardTitle>
              <CardDescription className="flex items-center gap-2">
                <Badge variant="outline" className={config.color}>
                  {config.label}
                </Badge>
              </CardDescription>
            </div>
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={onRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <div className="text-2xl font-bold">{summary.total_providers}</div>
            <div className="text-xs text-muted-foreground">Total Providers</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-emerald-500/10">
            <div className="text-2xl font-bold text-emerald-500">{summary.operational}</div>
            <div className="text-xs text-muted-foreground">Operational</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-yellow-500/10">
            <div className="text-2xl font-bold text-yellow-500">{summary.degraded}</div>
            <div className="text-xs text-muted-foreground">Degraded</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-red-500/10">
            <div className="text-2xl font-bold text-red-500">{summary.down}</div>
            <div className="text-xs text-muted-foreground">Down</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <CircularUptime uptimePercent={avgUptime} size={60} strokeWidth={5} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface AlertsListProps {
  alerts: HealthAlert[];
  onAcknowledge: (alertId: string) => void;
  onResolve: (alertId: string) => void;
}

function AlertsList({ alerts, onAcknowledge, onResolve }: AlertsListProps) {
  const activeAlerts = alerts.filter((a) => !a.resolved);
  const resolvedAlerts = alerts.filter((a) => a.resolved);

  if (alerts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <BellOff className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <p className="text-muted-foreground">No alerts at this time</p>
        <p className="text-xs text-muted-foreground mt-1">
          All providers are operating normally
        </p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-[300px]">
      <div className="space-y-2">
        {activeAlerts.length > 0 && (
          <>
            <h4 className="text-sm font-medium text-muted-foreground mb-2">
              Active Alerts ({activeAlerts.length})
            </h4>
            {activeAlerts.map((alert) => {
              const config = severityConfig[alert.severity];
              return (
                <div
                  key={alert.id}
                  className={cn(
                    "p-3 rounded-lg border",
                    config.bgColor,
                    config.borderColor
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant="outline" className={config.color}>
                          {alert.severity.toUpperCase()}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {alert.provider_id}
                        </span>
                      </div>
                      <p className="text-sm">{alert.message}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {new Date(alert.created_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="flex gap-1 ml-2">
                      {!alert.acknowledged && (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7"
                                onClick={() => onAcknowledge(alert.id)}
                              >
                                <Bell className="h-3.5 w-3.5" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>Acknowledge</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7"
                              onClick={() => onResolve(alert.id)}
                            >
                              <CheckCircle2 className="h-3.5 w-3.5" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Resolve</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </div>
                </div>
              );
            })}
          </>
        )}

        {resolvedAlerts.length > 0 && (
          <>
            <h4 className="text-sm font-medium text-muted-foreground mt-4 mb-2">
              Resolved ({resolvedAlerts.length})
            </h4>
            {resolvedAlerts.slice(0, 5).map((alert) => (
              <div
                key={alert.id}
                className="p-3 rounded-lg border border-muted/50 bg-muted/30 opacity-60"
              >
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  <span className="text-xs text-muted-foreground">{alert.provider_id}</span>
                </div>
                <p className="text-sm text-muted-foreground">{alert.message}</p>
              </div>
            ))}
          </>
        )}
      </div>
    </ScrollArea>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function ProviderHealthDashboard({
  providers,
  summary,
  alerts,
  history = [],
  isLoading = false,
  isRefreshing = false,
  error = null,
  lastUpdated = null,
  onRefresh,
  onRefreshProvider,
  onAcknowledgeAlert,
  onResolveAlert,
  onViewProviderDetails,
  onTimeRangeChange,
  showCharts = true,
  showAlerts = true,
  className,
}: ProviderHealthDashboardProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "providers" | "charts" | "alerts">(
    "overview"
  );
  const [selectedProviderId, setSelectedProviderId] = useState<string | null>(null);
  const [refreshingProviderId, setRefreshingProviderId] = useState<string | null>(null);

  // Handle refresh
  const handleRefresh = useCallback(async () => {
    if (onRefresh) {
      await onRefresh();
      toast.success("Dashboard refreshed");
    }
  }, [onRefresh]);

  // Handle provider refresh
  const handleRefreshProvider = useCallback(
    async (providerId: string) => {
      if (onRefreshProvider) {
        setRefreshingProviderId(providerId);
        try {
          await onRefreshProvider(providerId);
          toast.success(`Provider ${providerId} refreshed`);
        } finally {
          setRefreshingProviderId(null);
        }
      }
    },
    [onRefreshProvider]
  );

  // Handle alert acknowledgment
  const handleAcknowledgeAlert = useCallback(
    async (alertId: string) => {
      if (onAcknowledgeAlert) {
        await onAcknowledgeAlert(alertId);
        toast.success("Alert acknowledged");
      }
    },
    [onAcknowledgeAlert]
  );

  // Handle alert resolution
  const handleResolveAlert = useCallback(
    async (alertId: string) => {
      if (onResolveAlert) {
        await onResolveAlert(alertId);
        toast.success("Alert resolved");
      }
    },
    [onResolveAlert]
  );

  // Get provider history for charts
  const providerHistory = useMemo(() => {
    if (selectedProviderId) {
      return history.filter((h) => true); // Filter by provider if needed
    }
    return history;
  }, [history, selectedProviderId]);

  // Sorted providers (unhealthy first)
  const sortedProviders = useMemo(() => {
    const statusOrder: Record<ProviderStatus, number> = {
      down: 0,
      degraded: 1,
      unknown: 2,
      operational: 3,
    };
    return [...providers].sort(
      (a, b) => statusOrder[a.status] - statusOrder[b.status]
    );
  }, [providers]);

  // Active alerts count
  const activeAlertsCount = useMemo(
    () => alerts.filter((a) => !a.resolved).length,
    [alerts]
  );

  // Loading state
  if (isLoading) {
    return (
      <div className={cn("space-y-6", className)}>
        <Card>
          <CardHeader>
            <Skeleton className="h-8 w-64" />
            <Skeleton className="h-4 w-96" />
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-5 gap-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-20" />
              ))}
            </div>
          </CardContent>
        </Card>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-64" />
          ))}
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <Card className={cn("border-destructive", className)}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <XCircle className="h-5 w-5" />
            Error Loading Dashboard
          </CardTitle>
          <CardDescription>{error.message}</CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Empty state
  if (!summary || providers.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Provider Health Dashboard
          </CardTitle>
          <CardDescription>No providers configured</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center py-8">
          <Gauge className="h-16 w-16 text-muted-foreground/50 mb-4" />
          <p className="text-muted-foreground">
            Configure API keys to start monitoring provider health
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Summary Card */}
      <SummaryCard
        summary={summary}
        isRefreshing={isRefreshing}
        onRefresh={handleRefresh}
      />

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" className="gap-2">
            <Activity className="h-4 w-4" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="providers" className="gap-2">
            <Gauge className="h-4 w-4" />
            Providers ({providers.length})
          </TabsTrigger>
          {showCharts && (
            <TabsTrigger value="charts" className="gap-2">
              <TrendingUp className="h-4 w-4" />
              Metrics
            </TabsTrigger>
          )}
          {showAlerts && (
            <TabsTrigger value="alerts" className="gap-2 relative">
              <Bell className="h-4 w-4" />
              Alerts
              {activeAlertsCount > 0 && (
                <Badge
                  variant="destructive"
                  className="absolute -top-1 -right-1 h-5 w-5 p-0 flex items-center justify-center text-[10px]"
                >
                  {activeAlertsCount}
                </Badge>
              )}
            </TabsTrigger>
          )}
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {sortedProviders.slice(0, 6).map((provider) => (
              <ProviderHealthCard
                key={provider.provider_id}
                metrics={provider}
                isRefreshing={refreshingProviderId === provider.provider_id}
                onRefresh={handleRefreshProvider}
                onViewDetails={onViewProviderDetails}
                showActions
              />
            ))}
          </div>
          {sortedProviders.length > 6 && (
            <div className="text-center">
              <Button variant="outline" onClick={() => setActiveTab("providers")}>
                View All {sortedProviders.length} Providers
              </Button>
            </div>
          )}
        </TabsContent>

        {/* Providers Tab */}
        <TabsContent value="providers" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {sortedProviders.map((provider) => (
              <ProviderHealthCard
                key={provider.provider_id}
                metrics={provider}
                isRefreshing={refreshingProviderId === provider.provider_id}
                onRefresh={handleRefreshProvider}
                onViewDetails={onViewProviderDetails}
                showActions
              />
            ))}
          </div>
        </TabsContent>

        {/* Charts Tab */}
        {showCharts && (
          <TabsContent value="charts" className="mt-4">
            <HealthMetricsChart
              data={providerHistory}
              onTimeRangeChange={onTimeRangeChange}
            />
          </TabsContent>
        )}

        {/* Alerts Tab */}
        {showAlerts && (
          <TabsContent value="alerts" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="h-5 w-5" />
                  Health Alerts
                </CardTitle>
                <CardDescription>
                  Monitor and manage provider health alerts
                </CardDescription>
              </CardHeader>
              <CardContent>
                <AlertsList
                  alerts={alerts}
                  onAcknowledge={handleAcknowledgeAlert}
                  onResolve={handleResolveAlert}
                />
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>

      {/* Last Updated */}
      {lastUpdated && (
        <div className="text-center text-sm text-muted-foreground">
          <Clock className="h-3.5 w-3.5 inline mr-1" />
          Last updated: {lastUpdated.toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

export default ProviderHealthDashboard;
