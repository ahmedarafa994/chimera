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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Gauge,
  RefreshCw,
  Clock,
  Zap,
  Activity,
  TrendingUp,
  Ban,
  Timer,
} from "lucide-react";

// =============================================================================
// Types
// =============================================================================

export interface RateLimitMetrics {
  provider_id: string;
  provider_name: string;
  requests_per_minute: number;
  tokens_per_minute: number;
  provider_rpm_cap: number;
  provider_tpm_cap: number;
  custom_rpm_limit: number | null;
  custom_tpm_limit: number | null;
  effective_rpm_limit: number;
  effective_tpm_limit: number;
  rpm_usage_percent: number;
  tpm_usage_percent: number;
  is_rate_limited: boolean;
  rate_limit_reset_at: string | null;
  rate_limit_retry_after_seconds: number | null;
  rate_limit_hits_last_hour: number;
  rate_limit_hits_last_24h: number;
  burst_capacity: number;
  burst_remaining: number;
  window_start_at: string | null;
  updated_at: string | null;
}

export interface RateLimitGaugeProps {
  metrics: RateLimitMetrics;
  isLoading?: boolean;
  isRefreshing?: boolean;
  onRefresh?: () => Promise<void>;
  onViewDetails?: () => void;
  showActions?: boolean;
  compact?: boolean;
  className?: string;
}

export interface RateLimitDashboardProps {
  providers: RateLimitMetrics[];
  isLoading?: boolean;
  isRefreshing?: boolean;
  error?: Error | null;
  onRefresh?: () => Promise<void>;
  onRefreshProvider?: (providerId: string) => Promise<void>;
  onViewDetails?: (providerId: string) => void;
  className?: string;
}

export type RateLimitLevel = "low" | "moderate" | "high" | "critical" | "limited";

// =============================================================================
// Configuration
// =============================================================================

const rateLimitLevelConfig: Record<
  RateLimitLevel,
  {
    icon: React.ElementType;
    color: string;
    bgColor: string;
    borderColor: string;
    gaugeColor: string;
    label: string;
    description: string;
  }
> = {
  low: {
    icon: CheckCircle2,
    color: "text-emerald-500",
    bgColor: "bg-emerald-500/10",
    borderColor: "border-emerald-500/30",
    gaugeColor: "#10b981",
    label: "Low",
    description: "Rate limit usage is low, plenty of capacity available",
  },
  moderate: {
    icon: Activity,
    color: "text-blue-500",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/30",
    gaugeColor: "#3b82f6",
    label: "Moderate",
    description: "Moderate rate limit usage, normal operation",
  },
  high: {
    icon: AlertTriangle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500/30",
    gaugeColor: "#eab308",
    label: "High",
    description: "Approaching rate limit, consider throttling requests",
  },
  critical: {
    icon: AlertCircle,
    color: "text-orange-500",
    bgColor: "bg-orange-500/10",
    borderColor: "border-orange-500/30",
    gaugeColor: "#f97316",
    label: "Critical",
    description: "Near rate limit cap, requests may be throttled soon",
  },
  limited: {
    icon: Ban,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/30",
    gaugeColor: "#ef4444",
    label: "Rate Limited",
    description: "Currently rate limited, requests are being blocked",
  },
};

const providerColors: Record<string, { color: string; bgColor: string }> = {
  openai: { color: "text-green-500", bgColor: "bg-green-500/10" },
  anthropic: { color: "text-orange-500", bgColor: "bg-orange-500/10" },
  google: { color: "text-blue-500", bgColor: "bg-blue-500/10" },
  deepseek: { color: "text-purple-500", bgColor: "bg-purple-500/10" },
  qwen: { color: "text-cyan-500", bgColor: "bg-cyan-500/10" },
  bigmodel: { color: "text-indigo-500", bgColor: "bg-indigo-500/10" },
};

// =============================================================================
// Helper Functions
// =============================================================================

function getRateLimitLevel(
  metrics: RateLimitMetrics
): RateLimitLevel {
  if (metrics.is_rate_limited) return "limited";

  const maxUsage = Math.max(metrics.rpm_usage_percent, metrics.tpm_usage_percent);

  if (maxUsage >= 90) return "critical";
  if (maxUsage >= 75) return "high";
  if (maxUsage >= 50) return "moderate";
  return "low";
}

function formatNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return num.toLocaleString();
}

function formatTimeRemaining(seconds: number | null): string {
  if (seconds === null || seconds <= 0) return "N/A";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

// =============================================================================
// Gauge Component
// =============================================================================

interface GaugeDisplayProps {
  value: number;
  maxValue: number;
  current: number;
  label: string;
  unit: string;
  size?: number;
  strokeWidth?: number;
  level: RateLimitLevel;
  className?: string;
}

function GaugeDisplay({
  value,
  maxValue,
  current,
  label,
  unit,
  size = 120,
  strokeWidth = 12,
  level,
  className,
}: GaugeDisplayProps) {
  const config = rateLimitLevelConfig[level];
  const radius = (size - strokeWidth) / 2;
  const circumference = Math.PI * radius; // Half circle
  const clampedValue = Math.min(100, Math.max(0, value));
  const offset = circumference - (clampedValue / 100) * circumference;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn("relative inline-flex flex-col items-center justify-center", className)}
            style={{ width: size, height: size / 2 + 40 }}
          >
            {/* Gauge Arc */}
            <svg
              width={size}
              height={size / 2 + strokeWidth}
              className="overflow-visible"
            >
              {/* Background arc */}
              <path
                d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
                fill="none"
                stroke="currentColor"
                strokeWidth={strokeWidth}
                className="text-muted/30"
                strokeLinecap="round"
              />
              {/* Value arc */}
              <path
                d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
                fill="none"
                stroke={config.gaugeColor}
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={offset}
                style={{
                  transition: "stroke-dashoffset 0.5s ease-in-out",
                }}
              />
              {/* Tick marks */}
              {[0, 25, 50, 75, 100].map((tick) => {
                const angle = Math.PI - (tick / 100) * Math.PI;
                const x1 = size / 2 + (radius - strokeWidth) * Math.cos(angle);
                const y1 = size / 2 - (radius - strokeWidth) * Math.sin(angle);
                const x2 = size / 2 + (radius - strokeWidth - 6) * Math.cos(angle);
                const y2 = size / 2 - (radius - strokeWidth - 6) * Math.sin(angle);
                return (
                  <line
                    key={tick}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="currentColor"
                    strokeWidth={1}
                    className="text-muted-foreground/50"
                  />
                );
              })}
              {/* Needle */}
              <g
                style={{
                  transformOrigin: `${size / 2}px ${size / 2}px`,
                  transform: `rotate(${-90 + clampedValue * 1.8}deg)`,
                  transition: "transform 0.5s ease-in-out",
                }}
              >
                <line
                  x1={size / 2}
                  y1={size / 2}
                  x2={size / 2}
                  y2={strokeWidth + 10}
                  stroke={config.gaugeColor}
                  strokeWidth={2}
                />
                <circle
                  cx={size / 2}
                  cy={size / 2}
                  r={4}
                  fill={config.gaugeColor}
                />
              </g>
            </svg>

            {/* Value Display */}
            <div className="text-center -mt-2">
              <div className={cn("text-xl font-bold", config.color)}>
                {formatNumber(current)}
              </div>
              <div className="text-xs text-muted-foreground">
                {label} ({value.toFixed(0)}%)
              </div>
              <div className="text-[10px] text-muted-foreground">
                / {formatNumber(maxValue)} {unit}
              </div>
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <div className="text-center">
            <p className="font-medium">{label}</p>
            <p className="text-sm">
              {formatNumber(current)} / {formatNumber(maxValue)} {unit}
            </p>
            <p className="text-xs text-muted-foreground">
              {config.description}
            </p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// Linear Rate Meter Component
// =============================================================================

interface LinearRateMeterProps {
  label: string;
  current: number;
  limit: number;
  percent: number;
  icon: React.ElementType;
  level: RateLimitLevel;
  unit?: string;
}

function LinearRateMeter({
  label,
  current,
  limit,
  percent,
  icon: Icon,
  level,
  unit = "rpm",
}: LinearRateMeterProps) {
  const config = rateLimitLevelConfig[level];
  const clampedPercent = Math.min(100, Math.max(0, percent));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Icon className="h-4 w-4" />
          <span>{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn("font-medium", config.color)}>
            {formatNumber(current)}
          </span>
          <span className="text-muted-foreground">
            / {formatNumber(limit)} {unit}
          </span>
        </div>
      </div>
      <div className="relative h-2.5 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
          style={{
            width: `${clampedPercent}%`,
            backgroundColor: config.gaugeColor,
          }}
        />
        {/* Warning zone marker at 75% */}
        <div
          className="absolute top-0 h-full w-px bg-yellow-500/70"
          style={{ left: "75%" }}
        />
        {/* Critical zone marker at 90% */}
        <div
          className="absolute top-0 h-full w-px bg-red-500/70"
          style={{ left: "90%" }}
        />
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{percent.toFixed(1)}% of limit</span>
        <span>{formatNumber(limit - current)} remaining</span>
      </div>
    </div>
  );
}

// =============================================================================
// Main RateLimitGauge Component
// =============================================================================

export function RateLimitGauge({
  metrics,
  isLoading = false,
  isRefreshing = false,
  onRefresh,
  onViewDetails,
  showActions = true,
  compact = false,
  className,
}: RateLimitGaugeProps) {
  const level = useMemo(() => getRateLimitLevel(metrics), [metrics]);
  const config = rateLimitLevelConfig[level];
  const LevelIcon = config.icon;
  const providerStyle = providerColors[metrics.provider_id] || {
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
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-2 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  // Compact view
  if (compact) {
    const maxUsage = Math.max(metrics.rpm_usage_percent, metrics.tpm_usage_percent);
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
          <Gauge className={cn("h-4 w-4", providerStyle.color)} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate">{metrics.provider_name}</p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{metrics.requests_per_minute.toFixed(0)} RPM</span>
            <span>•</span>
            <span>{maxUsage.toFixed(1)}% used</span>
          </div>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className={cn("p-1.5 rounded-full", config.bgColor)}>
                <LevelIcon className={cn("h-4 w-4", config.color)} />
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>
                {config.label}: {config.description}
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    );
  }

  return (
    <Card
      className={cn(
        "transition-all hover:shadow-md",
        config.borderColor,
        className
      )}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={cn("p-2.5 rounded-lg", providerStyle.bgColor)}>
              <Gauge className={cn("h-5 w-5", providerStyle.color)} />
            </div>
            <div>
              <CardTitle className="text-lg">{metrics.provider_name}</CardTitle>
              <CardDescription className="flex items-center gap-1.5">
                <Activity className="h-3 w-3" />
                Rate Limit Status
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
                    <LevelIcon className="h-3 w-3" />
                    {config.label}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{config.description}</p>
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

      <CardContent className="space-y-6">
        {/* Gauge Displays */}
        <div className="flex justify-center gap-4">
          <GaugeDisplay
            value={metrics.rpm_usage_percent}
            maxValue={metrics.effective_rpm_limit}
            current={metrics.requests_per_minute}
            label="Requests"
            unit="RPM"
            level={
              metrics.is_rate_limited
                ? "limited"
                : metrics.rpm_usage_percent >= 90
                ? "critical"
                : metrics.rpm_usage_percent >= 75
                ? "high"
                : metrics.rpm_usage_percent >= 50
                ? "moderate"
                : "low"
            }
          />
          <GaugeDisplay
            value={metrics.tpm_usage_percent}
            maxValue={metrics.effective_tpm_limit}
            current={metrics.tokens_per_minute}
            label="Tokens"
            unit="TPM"
            level={
              metrics.is_rate_limited
                ? "limited"
                : metrics.tpm_usage_percent >= 90
                ? "critical"
                : metrics.tpm_usage_percent >= 75
                ? "high"
                : metrics.tpm_usage_percent >= 50
                ? "moderate"
                : "low"
            }
          />
        </div>

        {/* Linear Meters */}
        <div className="space-y-4">
          <LinearRateMeter
            label="Requests per Minute"
            current={metrics.requests_per_minute}
            limit={metrics.effective_rpm_limit}
            percent={metrics.rpm_usage_percent}
            icon={Zap}
            level={level}
            unit="RPM"
          />
          <LinearRateMeter
            label="Tokens per Minute"
            current={metrics.tokens_per_minute}
            limit={metrics.effective_tpm_limit}
            percent={metrics.tpm_usage_percent}
            icon={Activity}
            level={level}
            unit="TPM"
          />
        </div>

        {/* Rate Limit Stats */}
        <div className="grid grid-cols-3 gap-3">
          <div className="p-3 rounded-lg bg-muted/50 text-center">
            <div className="text-xs text-muted-foreground mb-1">
              Hits (1h)
            </div>
            <div
              className={cn(
                "text-lg font-semibold",
                metrics.rate_limit_hits_last_hour > 0
                  ? "text-orange-500"
                  : "text-muted-foreground"
              )}
            >
              {metrics.rate_limit_hits_last_hour}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-muted/50 text-center">
            <div className="text-xs text-muted-foreground mb-1">
              Hits (24h)
            </div>
            <div
              className={cn(
                "text-lg font-semibold",
                metrics.rate_limit_hits_last_24h > 0
                  ? "text-orange-500"
                  : "text-muted-foreground"
              )}
            >
              {metrics.rate_limit_hits_last_24h}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-muted/50 text-center">
            <div className="text-xs text-muted-foreground mb-1">Burst</div>
            <div className="text-lg font-semibold">
              {metrics.burst_remaining}/{metrics.burst_capacity}
            </div>
          </div>
        </div>

        {/* Rate Limited Alert */}
        {metrics.is_rate_limited && (
          <div
            className={cn(
              "flex items-center gap-3 p-3 rounded-lg border",
              config.bgColor,
              config.borderColor
            )}
          >
            <Ban className={cn("h-5 w-5 shrink-0", config.color)} />
            <div className="flex-1">
              <p className="font-medium text-red-600">
                Currently Rate Limited
              </p>
              <p className="text-sm text-muted-foreground">
                Retry in{" "}
                {formatTimeRemaining(metrics.rate_limit_retry_after_seconds)}
              </p>
            </div>
            {metrics.rate_limit_reset_at && (
              <div className="text-right text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Timer className="h-3 w-3" />
                  Resets at
                </div>
                <div className="font-medium">
                  {new Date(metrics.rate_limit_reset_at).toLocaleTimeString()}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Custom Limits Info */}
        {(metrics.custom_rpm_limit || metrics.custom_tpm_limit) && (
          <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2 border-t">
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              Custom limits applied
            </div>
            {metrics.custom_rpm_limit && (
              <span>RPM: {formatNumber(metrics.custom_rpm_limit)}</span>
            )}
            {metrics.custom_tpm_limit && (
              <span>TPM: {formatNumber(metrics.custom_tpm_limit)}</span>
            )}
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
// Rate Limit Dashboard Component
// =============================================================================

export function RateLimitDashboard({
  providers,
  isLoading = false,
  isRefreshing = false,
  error = null,
  onRefresh,
  onRefreshProvider,
  onViewDetails,
  className,
}: RateLimitDashboardProps) {
  // Summary stats
  const summary = useMemo(() => {
    const rateLimited = providers.filter((p) => p.is_rate_limited).length;
    const critical = providers.filter(
      (p) =>
        !p.is_rate_limited &&
        Math.max(p.rpm_usage_percent, p.tpm_usage_percent) >= 90
    ).length;
    const high = providers.filter(
      (p) =>
        !p.is_rate_limited &&
        Math.max(p.rpm_usage_percent, p.tpm_usage_percent) >= 75 &&
        Math.max(p.rpm_usage_percent, p.tpm_usage_percent) < 90
    ).length;
    const totalHitsLast24h = providers.reduce(
      (sum, p) => sum + p.rate_limit_hits_last_24h,
      0
    );
    const avgRpm =
      providers.length > 0
        ? providers.reduce((sum, p) => sum + p.requests_per_minute, 0) /
          providers.length
        : 0;

    return { rateLimited, critical, high, totalHitsLast24h, avgRpm };
  }, [providers]);

  // Sorted providers (most critical first)
  const sortedProviders = useMemo(() => {
    return [...providers].sort((a, b) => {
      if (a.is_rate_limited !== b.is_rate_limited)
        return a.is_rate_limited ? -1 : 1;
      const aMax = Math.max(a.rpm_usage_percent, a.tpm_usage_percent);
      const bMax = Math.max(b.rpm_usage_percent, b.tpm_usage_percent);
      return bMax - aMax;
    });
  }, [providers]);

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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-80" />
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
            Error Loading Rate Limit Data
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
  if (providers.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Gauge className="h-5 w-5" />
            Rate Limit Dashboard
          </CardTitle>
          <CardDescription>No rate limit data available</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center py-8">
          <Activity className="h-16 w-16 text-muted-foreground/50 mb-4" />
          <p className="text-muted-foreground">
            Configure API keys to start monitoring rate limits
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
                <Gauge className="h-8 w-8 text-primary" />
              </div>
              <div>
                <CardTitle className="text-2xl">Rate Limit Dashboard</CardTitle>
                <CardDescription>
                  Monitor API rate limits across all providers
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
              <div className="text-2xl font-bold">{providers.length}</div>
              <div className="text-xs text-muted-foreground">Providers</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-red-500/10">
              <div className="text-2xl font-bold text-red-500">
                {summary.rateLimited}
              </div>
              <div className="text-xs text-muted-foreground">Rate Limited</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-orange-500/10">
              <div className="text-2xl font-bold text-orange-500">
                {summary.critical}
              </div>
              <div className="text-xs text-muted-foreground">Critical</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-yellow-500/10">
              <div className="text-2xl font-bold text-yellow-500">
                {summary.high}
              </div>
              <div className="text-xs text-muted-foreground">High Usage</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-blue-500/10">
              <div className="text-2xl font-bold text-blue-500">
                {summary.avgRpm.toFixed(0)}
              </div>
              <div className="text-xs text-muted-foreground">Avg RPM</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Provider Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sortedProviders.map((provider) => (
          <RateLimitGauge
            key={provider.provider_id}
            metrics={provider}
            isRefreshing={isRefreshing}
            onRefresh={
              onRefreshProvider
                ? () => onRefreshProvider(provider.provider_id)
                : undefined
            }
            onViewDetails={
              onViewDetails
                ? () => onViewDetails(provider.provider_id)
                : undefined
            }
            showActions
          />
        ))}
      </div>
    </div>
  );
}

export default RateLimitGauge;
