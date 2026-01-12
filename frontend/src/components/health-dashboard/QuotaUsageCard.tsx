"use client";

import * as React from "react";
import { useMemo, useState, useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Coins,
  RefreshCw,
  Clock,
  TrendingUp,
  BarChart3,
  Zap,
  DollarSign,
  Calendar,
  Timer,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";

// =============================================================================
// Types
// =============================================================================

export type QuotaPeriod = "hourly" | "daily" | "monthly";
export type QuotaAlertLevel = "normal" | "warning" | "critical" | "exceeded";

export interface QuotaStatus {
  provider_id: string;
  provider_name: string;
  usage: number;
  limit: number;
  usage_percent: number;
  tokens_used: number;
  tokens_limit: number | null;
  tokens_percent: number | null;
  requests_used: number;
  requests_limit: number | null;
  requests_percent: number | null;
  period: QuotaPeriod;
  period_start_at: string | null;
  reset_at: string | null;
  cost_used: number;
  cost_limit: number | null;
  cost_currency: string;
  warning_threshold_percent: number;
  critical_threshold_percent: number;
  is_warning: boolean;
  is_critical: boolean;
  is_exceeded: boolean;
  updated_at: string | null;
}

export interface QuotaHistoryEntry {
  timestamp: string;
  usage_percent: number;
  tokens_used: number;
  requests_used: number;
  cost_used: number;
}

export interface QuotaUsageCardProps {
  quota: QuotaStatus;
  history?: QuotaHistoryEntry[];
  isLoading?: boolean;
  isRefreshing?: boolean;
  onRefresh?: () => Promise<void>;
  onViewDetails?: () => void;
  showHistory?: boolean;
  showActions?: boolean;
  compact?: boolean;
  className?: string;
}

export interface QuotaDashboardProps {
  quotas: QuotaStatus[];
  history?: Record<string, QuotaHistoryEntry[]>;
  isLoading?: boolean;
  isRefreshing?: boolean;
  error?: Error | null;
  onRefresh?: () => Promise<void>;
  onRefreshProvider?: (providerId: string) => Promise<void>;
  onViewDetails?: (providerId: string) => void;
  className?: string;
}

// =============================================================================
// Configuration
// =============================================================================

const alertLevelConfig: Record<
  QuotaAlertLevel,
  {
    icon: React.ElementType;
    color: string;
    bgColor: string;
    borderColor: string;
    progressColor: string;
    label: string;
  }
> = {
  normal: {
    icon: CheckCircle2,
    color: "text-emerald-500",
    bgColor: "bg-emerald-500/10",
    borderColor: "border-emerald-500/30",
    progressColor: "bg-emerald-500",
    label: "Normal",
  },
  warning: {
    icon: AlertTriangle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500/30",
    progressColor: "bg-yellow-500",
    label: "Warning",
  },
  critical: {
    icon: AlertCircle,
    color: "text-orange-500",
    bgColor: "bg-orange-500/10",
    borderColor: "border-orange-500/30",
    progressColor: "bg-orange-500",
    label: "Critical",
  },
  exceeded: {
    icon: AlertCircle,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/30",
    progressColor: "bg-red-500",
    label: "Exceeded",
  },
};

const periodLabels: Record<QuotaPeriod, string> = {
  hourly: "Hourly",
  daily: "Daily",
  monthly: "Monthly",
};

const providerColors: Record<string, { color: string; bgColor: string }> = {
  openai: { color: "text-green-500", bgColor: "bg-green-500/10" },
  anthropic: { color: "text-orange-500", bgColor: "bg-orange-500/10" },
  google: { color: "text-blue-500", bgColor: "bg-blue-500/10" },
  deepseek: { color: "text-purple-500", bgColor: "bg-purple-500/10" },
  qwen: { color: "text-cyan-500", bgColor: "bg-cyan-500/10" },
  bigmodel: { color: "text-indigo-500", bgColor: "bg-indigo-500/10" },
};

// Chart configuration
const historyChartConfig: ChartConfig = {
  usage_percent: {
    label: "Usage %",
    color: "hsl(var(--chart-1))",
  },
};

// =============================================================================
// Helper Functions
// =============================================================================

function getAlertLevel(quota: QuotaStatus): QuotaAlertLevel {
  if (quota.is_exceeded) return "exceeded";
  if (quota.is_critical) return "critical";
  if (quota.is_warning) return "warning";
  return "normal";
}

function formatNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return num.toLocaleString();
}

function formatCurrency(amount: number, currency: string): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  }).format(amount);
}

function formatTimeRemaining(resetAt: string | null): string {
  if (!resetAt) return "Unknown";
  const now = new Date();
  const reset = new Date(resetAt);
  const diffMs = reset.getTime() - now.getTime();

  if (diffMs <= 0) return "Now";

  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

  if (hours > 24) {
    const days = Math.floor(hours / 24);
    return `${days}d ${hours % 24}h`;
  }
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// =============================================================================
// Sub-Components
// =============================================================================

interface UsageProgressProps {
  label: string;
  used: number;
  limit: number | null;
  percent: number | null;
  icon: React.ElementType;
  alertLevel: QuotaAlertLevel;
  formatValue?: (val: number) => string;
}

function UsageProgress({
  label,
  used,
  limit,
  percent,
  icon: Icon,
  alertLevel,
  formatValue = formatNumber,
}: UsageProgressProps) {
  const config = alertLevelConfig[alertLevel];
  const displayPercent = percent ?? (limit ? (used / limit) * 100 : 0);
  const clampedPercent = Math.min(100, Math.max(0, displayPercent));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Icon className="h-4 w-4" />
          <span>{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-medium">{formatValue(used)}</span>
          {limit && (
            <>
              <span className="text-muted-foreground">/</span>
              <span className="text-muted-foreground">{formatValue(limit)}</span>
            </>
          )}
        </div>
      </div>
      <div className="relative h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={cn(
            "absolute inset-y-0 left-0 rounded-full transition-all duration-500",
            config.progressColor
          )}
          style={{ width: `${clampedPercent}%` }}
        />
        {/* Threshold markers */}
        <div
          className="absolute top-0 h-full w-px bg-yellow-500/50"
          style={{ left: "80%" }}
        />
        <div
          className="absolute top-0 h-full w-px bg-red-500/50"
          style={{ left: "95%" }}
        />
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{displayPercent.toFixed(1)}% used</span>
        {limit && <span>{formatNumber(limit - used)} remaining</span>}
      </div>
    </div>
  );
}

interface QuotaHistoryChartProps {
  history: QuotaHistoryEntry[];
  height?: number;
  className?: string;
}

function QuotaHistoryChart({
  history,
  height = 150,
  className,
}: QuotaHistoryChartProps) {
  const formattedData = useMemo(
    () =>
      history.map((entry) => ({
        ...entry,
        time: formatTimestamp(entry.timestamp),
      })),
    [history]
  );

  if (history.length === 0) {
    return (
      <div
        className={cn(
          "flex items-center justify-center text-muted-foreground text-sm",
          className
        )}
        style={{ height }}
      >
        No history data available
      </div>
    );
  }

  return (
    <ChartContainer
      config={historyChartConfig}
      className={cn("w-full", className)}
      style={{ height }}
    >
      <AreaChart
        data={formattedData}
        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
      >
        <defs>
          <linearGradient id="quotaGradient" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="5%"
              stopColor="var(--color-usage_percent)"
              stopOpacity={0.3}
            />
            <stop
              offset="95%"
              stopColor="var(--color-usage_percent)"
              stopOpacity={0}
            />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
        />
        <YAxis
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
          tickFormatter={(value) => `${value}%`}
          domain={[0, 100]}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              labelFormatter={(value) => `Time: ${value}`}
              formatter={(value) => [`${Number(value).toFixed(1)}%`, "Usage"]}
            />
          }
        />
        {/* Warning threshold line */}
        <Area
          type="monotone"
          dataKey="usage_percent"
          stroke="var(--color-usage_percent)"
          fill="url(#quotaGradient)"
          strokeWidth={2}
        />
      </AreaChart>
    </ChartContainer>
  );
}

// =============================================================================
// Main QuotaUsageCard Component
// =============================================================================

export function QuotaUsageCard({
  quota,
  history = [],
  isLoading = false,
  isRefreshing = false,
  onRefresh,
  onViewDetails,
  showHistory = true,
  showActions = true,
  compact = false,
  className,
}: QuotaUsageCardProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "details" | "history">(
    "overview"
  );

  const alertLevel = useMemo(() => getAlertLevel(quota), [quota]);
  const config = alertLevelConfig[alertLevel];
  const AlertIcon = config.icon;
  const providerStyle = providerColors[quota.provider_id] || {
    color: "text-gray-500",
    bgColor: "bg-gray-500/10",
  };

  const handleRefresh = useCallback(async () => {
    if (onRefresh) {
      await onRefresh();
    }
  }, [onRefresh]);

  // Loading state
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-32" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-2 w-full" />
            <Skeleton className="h-2 w-full" />
            <Skeleton className="h-2 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  // Compact view
  if (compact) {
    return (
      <div
        className={cn(
          "flex items-center gap-3 p-3 rounded-lg border transition-all hover:bg-muted/50",
          config.bgColor,
          config.borderColor,
          className
        )}
      >
        <div className={cn("p-2 rounded-lg", providerStyle.bgColor)}>
          <Coins className={cn("h-4 w-4", providerStyle.color)} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate">{quota.provider_name}</p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{quota.usage_percent.toFixed(1)}% used</span>
            <span>•</span>
            <span>Resets in {formatTimeRemaining(quota.reset_at)}</span>
          </div>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className={cn("p-1.5 rounded-full", config.bgColor)}>
                <AlertIcon className={cn("h-4 w-4", config.color)} />
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>
                {config.label}: {quota.usage_percent.toFixed(1)}% of quota used
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    );
  }

  return (
    <Card className={cn("transition-all hover:shadow-md", config.borderColor, className)}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={cn("p-2.5 rounded-lg", providerStyle.bgColor)}>
              <Coins className={cn("h-5 w-5", providerStyle.color)} />
            </div>
            <div>
              <CardTitle className="text-lg">{quota.provider_name}</CardTitle>
              <CardDescription className="flex items-center gap-1.5">
                <Calendar className="h-3 w-3" />
                {periodLabels[quota.period]} Quota
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge
                    variant="outline"
                    className={cn(config.color, config.bgColor, "gap-1.5")}
                  >
                    <AlertIcon className="h-3 w-3" />
                    {config.label}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{quota.usage_percent.toFixed(1)}% of quota used</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            {showActions && onRefresh && (
              <Button
                variant="ghost"
                size="icon"
                onClick={handleRefresh}
                disabled={isRefreshing}
                className="h-8 w-8"
              >
                <RefreshCw
                  className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Main Usage Progress */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-lg font-semibold">
            <span className={config.color}>{quota.usage_percent.toFixed(1)}%</span>
            <div className="flex items-center gap-1 text-sm font-normal text-muted-foreground">
              <Timer className="h-3.5 w-3.5" />
              Resets in {formatTimeRemaining(quota.reset_at)}
            </div>
          </div>
          <div className="relative h-3 w-full rounded-full bg-muted overflow-hidden">
            <div
              className={cn(
                "absolute inset-y-0 left-0 rounded-full transition-all duration-500",
                config.progressColor
              )}
              style={{ width: `${Math.min(100, quota.usage_percent)}%` }}
            />
            {/* Warning threshold marker */}
            <div
              className="absolute top-0 h-full w-0.5 bg-yellow-500"
              style={{ left: `${quota.warning_threshold_percent}%` }}
            />
            {/* Critical threshold marker */}
            <div
              className="absolute top-0 h-full w-0.5 bg-red-500"
              style={{ left: `${quota.critical_threshold_percent}%` }}
            />
          </div>
        </div>

        <Tabs
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as typeof activeTab)}
        >
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="details">Details</TabsTrigger>
            {showHistory && <TabsTrigger value="history">History</TabsTrigger>}
          </TabsList>

          <TabsContent value="overview" className="mt-4 space-y-4">
            {/* Quick Stats */}
            <div className="grid grid-cols-2 gap-3">
              {/* Tokens Used */}
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <Zap className="h-3 w-3" />
                  Tokens Used
                </div>
                <div className="text-lg font-semibold">
                  {formatNumber(quota.tokens_used)}
                </div>
                {quota.tokens_limit && (
                  <div className="text-xs text-muted-foreground">
                    / {formatNumber(quota.tokens_limit)}
                  </div>
                )}
              </div>

              {/* Requests Made */}
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <BarChart3 className="h-3 w-3" />
                  Requests Made
                </div>
                <div className="text-lg font-semibold">
                  {formatNumber(quota.requests_used)}
                </div>
                {quota.requests_limit && (
                  <div className="text-xs text-muted-foreground">
                    / {formatNumber(quota.requests_limit)}
                  </div>
                )}
              </div>

              {/* Cost Used */}
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <DollarSign className="h-3 w-3" />
                  Cost
                </div>
                <div className="text-lg font-semibold">
                  {formatCurrency(quota.cost_used, quota.cost_currency)}
                </div>
                {quota.cost_limit && (
                  <div className="text-xs text-muted-foreground">
                    / {formatCurrency(quota.cost_limit, quota.cost_currency)}
                  </div>
                )}
              </div>

              {/* Time Remaining */}
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <Clock className="h-3 w-3" />
                  Time Left
                </div>
                <div className="text-lg font-semibold">
                  {formatTimeRemaining(quota.reset_at)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Until reset
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="details" className="mt-4 space-y-4">
            {/* Detailed Usage Bars */}
            {quota.tokens_limit && (
              <UsageProgress
                label="Token Usage"
                used={quota.tokens_used}
                limit={quota.tokens_limit}
                percent={quota.tokens_percent}
                icon={Zap}
                alertLevel={
                  (quota.tokens_percent ?? 0) > quota.critical_threshold_percent
                    ? "critical"
                    : (quota.tokens_percent ?? 0) > quota.warning_threshold_percent
                    ? "warning"
                    : "normal"
                }
              />
            )}

            {quota.requests_limit && (
              <UsageProgress
                label="Request Usage"
                used={quota.requests_used}
                limit={quota.requests_limit}
                percent={quota.requests_percent}
                icon={BarChart3}
                alertLevel={
                  (quota.requests_percent ?? 0) > quota.critical_threshold_percent
                    ? "critical"
                    : (quota.requests_percent ?? 0) > quota.warning_threshold_percent
                    ? "warning"
                    : "normal"
                }
              />
            )}

            {quota.cost_limit && (
              <UsageProgress
                label="Cost Usage"
                used={quota.cost_used}
                limit={quota.cost_limit}
                percent={(quota.cost_used / quota.cost_limit) * 100}
                icon={DollarSign}
                alertLevel={
                  (quota.cost_used / quota.cost_limit) * 100 >
                  quota.critical_threshold_percent
                    ? "critical"
                    : (quota.cost_used / quota.cost_limit) * 100 >
                      quota.warning_threshold_percent
                    ? "warning"
                    : "normal"
                }
                formatValue={(val) => formatCurrency(val, quota.cost_currency)}
              />
            )}

            {/* Thresholds Info */}
            <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2 border-t">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-yellow-500" />
                Warning: {quota.warning_threshold_percent}%
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-red-500" />
                Critical: {quota.critical_threshold_percent}%
              </div>
            </div>
          </TabsContent>

          {showHistory && (
            <TabsContent value="history" className="mt-4">
              <QuotaHistoryChart history={history} height={180} />
            </TabsContent>
          )}
        </Tabs>

        {/* Alert Banner */}
        {(quota.is_warning || quota.is_critical || quota.is_exceeded) && (
          <div
            className={cn(
              "flex items-center gap-2 p-3 rounded-lg border",
              config.bgColor,
              config.borderColor
            )}
          >
            <AlertIcon className={cn("h-4 w-4 shrink-0", config.color)} />
            <div className="flex-1 text-sm">
              {quota.is_exceeded && (
                <span className="font-medium text-red-600">
                  Quota exceeded! Requests may be blocked.
                </span>
              )}
              {quota.is_critical && !quota.is_exceeded && (
                <span className="font-medium text-orange-600">
                  Approaching quota limit. Consider upgrading or reducing usage.
                </span>
              )}
              {quota.is_warning && !quota.is_critical && !quota.is_exceeded && (
                <span className="font-medium text-yellow-600">
                  Usage above warning threshold. Monitor closely.
                </span>
              )}
            </div>
          </div>
        )}

        {/* View Details Button */}
        {showActions && onViewDetails && (
          <Button variant="outline" className="w-full" onClick={onViewDetails}>
            View Details
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Quota Dashboard Component
// =============================================================================

export function QuotaDashboard({
  quotas,
  history = {},
  isLoading = false,
  isRefreshing = false,
  error = null,
  onRefresh,
  onRefreshProvider,
  onViewDetails,
  className,
}: QuotaDashboardProps) {
  // Summary stats
  const summary = useMemo(() => {
    const warnings = quotas.filter((q) => q.is_warning && !q.is_critical).length;
    const critical = quotas.filter((q) => q.is_critical && !q.is_exceeded).length;
    const exceeded = quotas.filter((q) => q.is_exceeded).length;
    const totalCost = quotas.reduce((sum, q) => sum + q.cost_used, 0);
    const avgUsage =
      quotas.length > 0
        ? quotas.reduce((sum, q) => sum + q.usage_percent, 0) / quotas.length
        : 0;

    return { warnings, critical, exceeded, totalCost, avgUsage };
  }, [quotas]);

  // Sorted quotas (most critical first)
  const sortedQuotas = useMemo(() => {
    return [...quotas].sort((a, b) => {
      if (a.is_exceeded !== b.is_exceeded) return a.is_exceeded ? -1 : 1;
      if (a.is_critical !== b.is_critical) return a.is_critical ? -1 : 1;
      if (a.is_warning !== b.is_warning) return a.is_warning ? -1 : 1;
      return b.usage_percent - a.usage_percent;
    });
  }, [quotas]);

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
            <AlertCircle className="h-5 w-5" />
            Error Loading Quota Data
          </CardTitle>
          <CardDescription>{error.message}</CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={onRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Empty state
  if (quotas.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Coins className="h-5 w-5" />
            Quota Usage Dashboard
          </CardTitle>
          <CardDescription>No quota data available</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center py-8">
          <TrendingUp className="h-16 w-16 text-muted-foreground/50 mb-4" />
          <p className="text-muted-foreground">
            Configure API keys to start tracking quota usage
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Summary Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-full bg-primary/10">
                <Coins className="h-8 w-8 text-primary" />
              </div>
              <div>
                <CardTitle className="text-2xl">Quota Usage Dashboard</CardTitle>
                <CardDescription>
                  Monitor API usage across all providers
                </CardDescription>
              </div>
            </div>
            {onRefresh && (
              <Button
                variant="outline"
                size="icon"
                onClick={onRefresh}
                disabled={isRefreshing}
              >
                <RefreshCw
                  className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                />
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center p-3 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold">{quotas.length}</div>
              <div className="text-xs text-muted-foreground">Providers</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-emerald-500/10">
              <div className="text-2xl font-bold text-emerald-500">
                {summary.avgUsage.toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">Avg Usage</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-yellow-500/10">
              <div className="text-2xl font-bold text-yellow-500">
                {summary.warnings}
              </div>
              <div className="text-xs text-muted-foreground">Warnings</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-orange-500/10">
              <div className="text-2xl font-bold text-orange-500">
                {summary.critical}
              </div>
              <div className="text-xs text-muted-foreground">Critical</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-blue-500/10">
              <div className="text-2xl font-bold text-blue-500">
                ${summary.totalCost.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">Total Cost</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quota Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sortedQuotas.map((quota) => (
          <QuotaUsageCard
            key={quota.provider_id}
            quota={quota}
            history={history[quota.provider_id] || []}
            isRefreshing={isRefreshing}
            onRefresh={
              onRefreshProvider
                ? () => onRefreshProvider(quota.provider_id)
                : undefined
            }
            onViewDetails={
              onViewDetails
                ? () => onViewDetails(quota.provider_id)
                : undefined
            }
            showHistory
            showActions
          />
        ))}
      </div>
    </div>
  );
}

export default QuotaUsageCard;
