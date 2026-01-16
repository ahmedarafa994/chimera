"use client";

import * as React from "react";
import { useCallback, useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Activity,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Shield,
  Timer,
} from "lucide-react";
import { UptimeIndicator } from "./UptimeIndicator";

// =============================================================================
// Types
// =============================================================================

export type ProviderStatus = "operational" | "degraded" | "down" | "unknown";

export interface ProviderHealthMetrics {
  provider_id: string;
  provider_name: string;
  status: ProviderStatus;
  latency_ms: number;
  latency_trend: "up" | "down" | "stable";
  error_rate: number;
  error_rate_trend: "up" | "down" | "stable";
  uptime_percent: number;
  circuit_breaker_state: "closed" | "open" | "half_open";
  last_check: string | null;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  rate_limited_requests: number;
}

export interface ProviderHealthCardProps {
  metrics: ProviderHealthMetrics;
  isRefreshing?: boolean;
  onRefresh?: (providerId: string) => Promise<void>;
  onViewDetails?: (providerId: string) => void;
  showActions?: boolean;
  compact?: boolean;
  className?: string;
}

// =============================================================================
// Status Configuration
// =============================================================================

const statusConfig: Record<
  ProviderStatus,
  {
    icon: React.ElementType;
    color: string;
    bgColor: string;
    borderColor: string;
    label: string;
    description: string;
  }
> = {
  operational: {
    icon: CheckCircle2,
    color: "text-emerald-500",
    bgColor: "bg-emerald-500/10",
    borderColor: "border-emerald-500/30",
    label: "Operational",
    description: "Provider is healthy and responding normally",
  },
  degraded: {
    icon: AlertTriangle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500/30",
    label: "Degraded",
    description: "Provider is experiencing higher latency or error rates",
  },
  down: {
    icon: XCircle,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/30",
    label: "Down",
    description: "Provider is not responding or has critical errors",
  },
  unknown: {
    icon: AlertTriangle,
    color: "text-gray-500",
    bgColor: "bg-gray-500/10",
    borderColor: "border-gray-500/30",
    label: "Unknown",
    description: "Unable to determine provider status",
  },
};

const circuitBreakerConfig: Record<
  string,
  { color: string; label: string; description: string }
> = {
  closed: {
    color: "text-emerald-500",
    label: "Closed",
    description: "Circuit breaker is closed, requests flow normally",
  },
  open: {
    color: "text-red-500",
    label: "Open",
    description: "Circuit breaker is open, requests are blocked",
  },
  half_open: {
    color: "text-yellow-500",
    label: "Half Open",
    description: "Circuit breaker is testing if provider has recovered",
  },
};

// =============================================================================
// Provider Icons
// =============================================================================

const providerIcons: Record<string, { color: string; bgColor: string }> = {
  openai: { color: "text-green-500", bgColor: "bg-green-500/10" },
  anthropic: { color: "text-orange-500", bgColor: "bg-orange-500/10" },
  google: { color: "text-blue-500", bgColor: "bg-blue-500/10" },
  deepseek: { color: "text-purple-500", bgColor: "bg-purple-500/10" },
  qwen: { color: "text-cyan-500", bgColor: "bg-cyan-500/10" },
  bigmodel: { color: "text-indigo-500", bgColor: "bg-indigo-500/10" },
  routeway: { color: "text-pink-500", bgColor: "bg-pink-500/10" },
  cursor: { color: "text-teal-500", bgColor: "bg-teal-500/10" },
};

// =============================================================================
// Helper Components
// =============================================================================

interface TrendIndicatorProps {
  trend: "up" | "down" | "stable";
  isGoodWhenUp?: boolean;
}

function TrendIndicator({ trend, isGoodWhenUp = true }: TrendIndicatorProps) {
  const isPositive = isGoodWhenUp ? trend === "up" : trend === "down";
  const isNegative = isGoodWhenUp ? trend === "down" : trend === "up";

  if (trend === "stable") {
    return <Minus className="h-3 w-3 text-muted-foreground" />;
  }

  const Icon = trend === "up" ? TrendingUp : TrendingDown;
  const colorClass = isPositive
    ? "text-emerald-500"
    : isNegative
    ? "text-red-500"
    : "text-muted-foreground";

  return <Icon className={cn("h-3 w-3", colorClass)} />;
}

interface MetricItemProps {
  label: string;
  value: string | number;
  icon: React.ElementType;
  trend?: "up" | "down" | "stable";
  isGoodWhenUp?: boolean;
  tooltip?: string;
}

function MetricItem({
  label,
  value,
  icon: Icon,
  trend,
  isGoodWhenUp,
  tooltip,
}: MetricItemProps) {
  const content = (
    <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
      <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-xs text-muted-foreground truncate">{label}</p>
        <div className="flex items-center gap-1">
          <p className="text-sm font-medium">{value}</p>
          {trend && <TrendIndicator trend={trend} isGoodWhenUp={isGoodWhenUp} />}
        </div>
      </div>
    </div>
  );

  if (tooltip) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{content}</TooltipTrigger>
          <TooltipContent>
            <p>{tooltip}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return content;
}

// =============================================================================
// Main Component
// =============================================================================

export function ProviderHealthCard({
  metrics,
  isRefreshing = false,
  onRefresh,
  onViewDetails,
  showActions = true,
  compact = false,
  className,
}: ProviderHealthCardProps) {
  const config = statusConfig[metrics.status];
  const StatusIcon = config.icon;
  const providerStyle = providerIcons[metrics.provider_id] || {
    color: "text-gray-500",
    bgColor: "bg-gray-500/10",
  };
  const cbConfig = circuitBreakerConfig[metrics.circuit_breaker_state] || circuitBreakerConfig.closed;

  // Format last check time
  const lastCheckFormatted = useMemo(() => {
    if (!metrics.last_check) return "Never";
    const date = new Date(metrics.last_check);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);

    if (diffSecs < 60) return `${diffSecs}s ago`;
    const diffMins = Math.floor(diffSecs / 60);
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  }, [metrics.last_check]);

  // Format success rate
  const successRate = useMemo(() => {
    if (metrics.total_requests === 0) return 100;
    return ((metrics.successful_requests / metrics.total_requests) * 100).toFixed(1);
  }, [metrics.total_requests, metrics.successful_requests]);

  // Handle refresh
  const handleRefresh = useCallback(async () => {
    if (onRefresh) {
      await onRefresh(metrics.provider_id);
    }
  }, [onRefresh, metrics.provider_id]);

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
          <Zap className={cn("h-4 w-4", providerStyle.color)} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate">{metrics.provider_name}</p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{metrics.latency_ms.toFixed(0)}ms</span>
            <span>•</span>
            <span>{metrics.error_rate.toFixed(1)}% errors</span>
          </div>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className={cn("p-1.5 rounded-full", config.bgColor)}>
                <StatusIcon className={cn("h-4 w-4", config.color)} />
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>{config.label}: {config.description}</p>
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
              <Zap className={cn("h-5 w-5", providerStyle.color)} />
            </div>
            <div>
              <CardTitle className="text-lg">{metrics.provider_name}</CardTitle>
              <CardDescription className="flex items-center gap-1.5">
                <Clock className="h-3 w-3" />
                Last check: {lastCheckFormatted}
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
                    <StatusIcon className="h-3 w-3" />
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

      <CardContent className="space-y-4">
        {/* Uptime Indicator */}
        <UptimeIndicator
          uptimePercent={metrics.uptime_percent}
          size="sm"
          showLabel
        />

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-2">
          <MetricItem
            label="Latency"
            value={`${metrics.latency_ms.toFixed(0)}ms`}
            icon={Timer}
            trend={metrics.latency_trend}
            isGoodWhenUp={false}
            tooltip="Average response latency"
          />
          <MetricItem
            label="Error Rate"
            value={`${metrics.error_rate.toFixed(2)}%`}
            icon={AlertTriangle}
            trend={metrics.error_rate_trend}
            isGoodWhenUp={false}
            tooltip="Percentage of failed requests"
          />
          <MetricItem
            label="Success Rate"
            value={`${successRate}%`}
            icon={CheckCircle2}
            tooltip="Percentage of successful requests"
          />
          <MetricItem
            label="Total Requests"
            value={metrics.total_requests.toLocaleString()}
            icon={Activity}
            tooltip="Total number of requests made"
          />
        </div>

        {/* Circuit Breaker Status */}
        <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">Circuit Breaker</span>
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge variant="outline" className={cbConfig.color}>
                  {cbConfig.label}
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <p>{cbConfig.description}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        {/* Rate Limited Requests Warning */}
        {metrics.rate_limited_requests > 0 && (
          <div className="flex items-center gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
            <AlertTriangle className="h-4 w-4 text-yellow-500 shrink-0" />
            <span className="text-sm text-yellow-700 dark:text-yellow-400">
              {metrics.rate_limited_requests.toLocaleString()} rate limited requests
            </span>
          </div>
        )}

        {/* View Details Button */}
        {showActions && onViewDetails && (
          <Button
            variant="outline"
            className="w-full"
            onClick={() => onViewDetails(metrics.provider_id)}
          >
            View Details
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

export default ProviderHealthCard;
