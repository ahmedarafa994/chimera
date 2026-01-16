/**
 * LatencyCard Component
 *
 * Displays real-time API and processing latency metrics for Aegis Campaign Dashboard:
 * - Average, P50, P95, P99 latency display
 * - Latency distribution mini histogram
 * - Separate metrics for API vs processing time
 * - Color coding: green <1s, amber <3s, red >3s
 *
 * Follows glass morphism styling pattern from existing components.
 */

"use client";

import { memo, useMemo } from "react";
import {
  Timer,
  Gauge,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Server,
  Cpu,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { LatencyMetrics, LatencyTimeSeries } from "@/types/aegis-telemetry";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  Cell,
  Area,
  AreaChart,
} from "recharts";

// ============================================================================
// Types
// ============================================================================

export interface LatencyCardProps {
  /** Current latency metrics */
  latencyMetrics: LatencyMetrics;
  /** Time series data for trend chart */
  latencyHistory: LatencyTimeSeries[];
  /** Trend direction for latency */
  latencyTrend?: "up" | "down" | "stable";
  /** Percentage change in latency trend */
  latencyTrendChange?: number;
  /** Whether the component is in loading state */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show in compact mode */
  compact?: boolean;
  /** Whether to show the trend chart (default: true) */
  showChart?: boolean;
}

// ============================================================================
// Configuration
// ============================================================================

/**
 * Latency threshold configuration for color coding
 * Values in milliseconds - green <1000ms, amber <3000ms, red >3000ms
 */
interface LatencyColorConfig {
  label: string;
  textClass: string;
  bgClass: string;
  borderClass: string;
  chartColor: string;
}

/**
 * Get latency color configuration based on latency value (in ms)
 * Green: <1000ms, Amber: <3000ms, Red: >=3000ms
 */
function getLatencyColorConfig(latencyMs: number): LatencyColorConfig {
  if (latencyMs < 1000) {
    return {
      label: "Fast",
      textClass: "text-emerald-400",
      bgClass: "bg-emerald-500/20",
      borderClass: "border-emerald-500/30",
      chartColor: "#10b981",
    };
  } else if (latencyMs < 3000) {
    return {
      label: "Moderate",
      textClass: "text-amber-400",
      bgClass: "bg-amber-500/20",
      borderClass: "border-amber-500/30",
      chartColor: "#f59e0b",
    };
  } else {
    return {
      label: "Slow",
      textClass: "text-red-400",
      bgClass: "bg-red-500/20",
      borderClass: "border-red-500/30",
      chartColor: "#ef4444",
    };
  }
}

const TREND_CONFIG = {
  up: {
    icon: TrendingUp,
    label: "Increasing",
    textClass: "text-red-400", // Latency going up is bad
    bgClass: "bg-red-500/10",
    borderClass: "border-red-500/20",
  },
  down: {
    icon: TrendingDown,
    label: "Decreasing",
    textClass: "text-emerald-400", // Latency going down is good
    bgClass: "bg-emerald-500/10",
    borderClass: "border-emerald-500/20",
  },
  stable: {
    icon: Minus,
    label: "Stable",
    textClass: "text-gray-400",
    bgClass: "bg-gray-500/10",
    borderClass: "border-gray-500/20",
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format latency value for display
 */
function formatLatency(ms: number): string {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${Math.round(ms)}ms`;
}

/**
 * Format latency with compact display
 */
function formatLatencyCompact(ms: number): string {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  return `${Math.round(ms)}ms`;
}

/**
 * Format percentage for display
 */
function formatPercentage(value: number, decimals: number = 1): string {
  return value.toFixed(decimals);
}

/**
 * Calculate trend from history
 */
function calculateLatencyTrend(
  history: LatencyTimeSeries[],
  currentLatency: number
): { trend: "up" | "down" | "stable"; change: number } {
  if (history.length < 2) {
    return { trend: "stable", change: 0 };
  }

  // Get last 10 data points for trend calculation
  const recentHistory = history.slice(-10);
  const firstLatency = recentHistory[0].total_latency_ms;
  const change = currentLatency - firstLatency;
  const percentChange = firstLatency > 0 ? (change / firstLatency) * 100 : 0;

  let trend: "up" | "down" | "stable";
  if (percentChange > 10) {
    trend = "up";
  } else if (percentChange < -10) {
    trend = "down";
  } else {
    trend = "stable";
  }

  return { trend, change: percentChange };
}

/**
 * Generate histogram data from percentiles
 */
function generateHistogramData(metrics: LatencyMetrics) {
  // Create buckets based on percentiles for a distribution view
  const buckets = [
    { range: "Min", value: metrics.min_latency_ms, label: "Minimum" },
    { range: "P50", value: metrics.p50_latency_ms, label: "50th Percentile" },
    { range: "Avg", value: metrics.avg_latency_ms, label: "Average" },
    { range: "P95", value: metrics.p95_latency_ms, label: "95th Percentile" },
    { range: "P99", value: metrics.p99_latency_ms, label: "99th Percentile" },
    { range: "Max", value: metrics.max_latency_ms, label: "Maximum" },
  ];
  return buckets;
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Main latency display with large value
 */
const LatencyDisplay = memo(function LatencyDisplay({
  latencyMs,
  colorConfig,
  compact,
}: {
  latencyMs: number;
  colorConfig: LatencyColorConfig;
  compact?: boolean;
}) {
  const formattedValue = useMemo(() => {
    if (latencyMs >= 1000) {
      return { value: (latencyMs / 1000).toFixed(2), unit: "s" };
    }
    return { value: Math.round(latencyMs).toString(), unit: "ms" };
  }, [latencyMs]);

  return (
    <div className="flex items-baseline gap-2">
      <div className={cn("flex items-center gap-2", compact && "gap-1")}>
        <Timer
          className={cn(
            colorConfig.textClass,
            compact ? "h-5 w-5" : "h-6 w-6"
          )}
          aria-hidden="true"
        />
        <span
          className={cn(
            "font-bold tabular-nums tracking-tight",
            colorConfig.textClass,
            compact ? "text-3xl" : "text-4xl"
          )}
        >
          {formattedValue.value}
        </span>
        <span
          className={cn(
            "font-medium text-muted-foreground",
            compact ? "text-sm" : "text-base"
          )}
        >
          {formattedValue.unit}
        </span>
      </div>
    </div>
  );
});

/**
 * Trend indicator badge for latency
 */
const LatencyTrendIndicator = memo(function LatencyTrendIndicator({
  trend,
  change,
  compact,
}: {
  trend: "up" | "down" | "stable";
  change: number;
  compact?: boolean;
}) {
  const config = TREND_CONFIG[trend];
  const Icon = config.icon;
  const absChange = Math.abs(change);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className={cn(
              "gap-1 font-medium transition-all duration-300",
              config.bgClass,
              config.borderClass,
              config.textClass,
              compact ? "px-1.5 py-0.5 text-xs" : "px-2 py-1 text-sm"
            )}
          >
            <Icon
              className={cn(
                trend !== "stable" && "animate-pulse",
                compact ? "h-3 w-3" : "h-3.5 w-3.5"
              )}
              aria-hidden="true"
            />
            <span>
              {absChange > 0
                ? `${trend === "up" ? "+" : "-"}${formatPercentage(absChange)}%`
                : "0%"}
            </span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {config.label} - Latency{" "}
            {trend === "stable"
              ? "is stable"
              : trend === "up"
              ? "increased"
              : "decreased"}{" "}
            by {formatPercentage(absChange)}% over recent operations
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * API vs Processing latency breakdown bar
 */
const LatencyBreakdown = memo(function LatencyBreakdown({
  latencyMetrics,
  compact,
}: {
  latencyMetrics: LatencyMetrics;
  compact?: boolean;
}) {
  const totalLatency = latencyMetrics.total_latency_ms || 1;
  const apiPercentage = (latencyMetrics.api_latency_ms / totalLatency) * 100;
  const processingPercentage = (latencyMetrics.processing_latency_ms / totalLatency) * 100;

  return (
    <div className={cn("space-y-2", compact && "space-y-1.5")}>
      {/* Latency header */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Latency Breakdown</span>
        <span className="text-xs font-medium text-foreground tabular-nums">
          {formatLatency(latencyMetrics.total_latency_ms)} total
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-2 bg-white/5 rounded-full overflow-hidden flex">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className="h-full bg-sky-500 transition-all duration-500"
                style={{ width: `${apiPercentage}%` }}
              />
            </TooltipTrigger>
            <TooltipContent>
              <p>
                API latency: {formatLatency(latencyMetrics.api_latency_ms)} (
                {formatPercentage(apiPercentage)}%)
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className="h-full bg-purple-500 transition-all duration-500"
                style={{ width: `${processingPercentage}%` }}
              />
            </TooltipTrigger>
            <TooltipContent>
              <p>
                Processing latency:{" "}
                {formatLatency(latencyMetrics.processing_latency_ms)} (
                {formatPercentage(processingPercentage)}%)
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between text-[10px]">
        <div className="flex items-center gap-1.5">
          <Server className="h-3 w-3 text-sky-400" aria-hidden="true" />
          <span className="text-sky-400">API</span>
          <span className="text-muted-foreground tabular-nums">
            {formatLatency(latencyMetrics.api_latency_ms)}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <Cpu className="h-3 w-3 text-purple-400" aria-hidden="true" />
          <span className="text-purple-400">Processing</span>
          <span className="text-muted-foreground tabular-nums">
            {formatLatency(latencyMetrics.processing_latency_ms)}
          </span>
        </div>
      </div>
    </div>
  );
});

/**
 * Percentile metrics grid
 */
const PercentileMetrics = memo(function PercentileMetrics({
  latencyMetrics,
  compact,
}: {
  latencyMetrics: LatencyMetrics;
  compact?: boolean;
}) {
  const percentiles = [
    { label: "Avg", value: latencyMetrics.avg_latency_ms, description: "Average latency" },
    { label: "P50", value: latencyMetrics.p50_latency_ms, description: "50th percentile (median)" },
    { label: "P95", value: latencyMetrics.p95_latency_ms, description: "95th percentile" },
    { label: "P99", value: latencyMetrics.p99_latency_ms, description: "99th percentile" },
  ];

  return (
    <div className={cn("grid gap-2", compact ? "grid-cols-2" : "grid-cols-4")}>
      {percentiles.map((percentile) => {
        const colorConfig = getLatencyColorConfig(percentile.value);
        return (
          <TooltipProvider key={percentile.label}>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "flex flex-col items-center justify-center rounded-lg p-2",
                    colorConfig.bgClass,
                    `border ${colorConfig.borderClass}`
                  )}
                >
                  <span className={cn("text-xs", colorConfig.textClass.replace("text-", "text-").replace("-400", "-400/80"))}>
                    {percentile.label}
                  </span>
                  <span className={cn("text-sm font-semibold tabular-nums", colorConfig.textClass)}>
                    {formatLatencyCompact(percentile.value)}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>{percentile.description}: {formatLatency(percentile.value)}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      })}
    </div>
  );
});

/**
 * Custom tooltip for the histogram chart
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const HistogramTooltip = ({ active, payload }: any) => {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload;

  return (
    <div className="rounded-lg border border-white/10 bg-black/80 p-2 shadow-xl backdrop-blur-sm">
      <p className="text-xs font-medium text-foreground">{data.label}</p>
      <p className="text-sm font-semibold text-cyan-400 tabular-nums mt-1">
        {formatLatency(data.value)}
      </p>
    </div>
  );
};

/**
 * Latency distribution histogram
 */
const LatencyHistogram = memo(function LatencyHistogram({
  data,
  height = 80,
}: {
  data: ReturnType<typeof generateHistogramData>;
  height?: number;
}) {
  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-muted-foreground text-xs"
        style={{ height }}
      >
        <span>Awaiting data...</span>
      </div>
    );
  }

  // Find max value for normalization
  const maxValue = Math.max(...data.map((d) => d.value));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
        >
          <XAxis
            dataKey="range"
            tick={{ fontSize: 10, fill: "#9ca3af" }}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
          />
          <YAxis hide domain={[0, maxValue * 1.1]} />
          <RechartsTooltip content={<HistogramTooltip />} />
          <Bar
            dataKey="value"
            radius={[4, 4, 0, 0]}
            isAnimationActive={true}
            animationDuration={500}
          >
            {data.map((entry, index) => {
              const colorConfig = getLatencyColorConfig(entry.value);
              return (
                <Cell
                  key={`cell-${index}`}
                  fill={colorConfig.chartColor}
                  fillOpacity={0.8}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
});

/**
 * Custom tooltip for the trend chart
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const TrendChartTooltip = ({ active, payload }: any) => {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload as LatencyTimeSeries;
  const time = new Date(data.timestamp).toLocaleTimeString();

  return (
    <div className="rounded-lg border border-white/10 bg-black/80 p-2 shadow-xl backdrop-blur-sm">
      <p className="text-xs font-medium text-foreground">{time}</p>
      <div className="mt-1 space-y-0.5 text-xs">
        <p className="text-sky-400">
          API: {formatLatency(data.api_latency_ms)}
        </p>
        <p className="text-purple-400">
          Processing: {formatLatency(data.processing_latency_ms)}
        </p>
        <p className="text-cyan-400">
          Total: {formatLatency(data.total_latency_ms)}
        </p>
      </div>
    </div>
  );
};

/**
 * Latency trend area chart
 */
const LatencyTrendChart = memo(function LatencyTrendChart({
  data,
  height = 80,
}: {
  data: LatencyTimeSeries[];
  height?: number;
}) {
  if (data.length < 2) {
    return (
      <div
        className="flex items-center justify-center text-muted-foreground text-xs"
        style={{ height }}
      >
        <span>Awaiting data...</span>
      </div>
    );
  }

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
        >
          <defs>
            <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="timestamp"
            hide
            tickFormatter={(value: string) =>
              new Date(value).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })
            }
          />
          <YAxis hide domain={["dataMin", "dataMax"]} />
          <RechartsTooltip content={<TrendChartTooltip />} />
          <Area
            type="monotone"
            dataKey="total_latency_ms"
            stroke="#06b6d4"
            strokeWidth={2}
            fill="url(#latencyGradient)"
            isAnimationActive={true}
            animationDuration={500}
            animationEasing="ease-out"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
});

/**
 * Min/Max latency display
 */
const MinMaxLatency = memo(function MinMaxLatency({
  latencyMetrics,
}: {
  latencyMetrics: LatencyMetrics;
}) {
  return (
    <div className="flex items-center justify-between text-xs">
      <div className="flex items-center gap-1.5">
        <Zap className="h-3 w-3 text-emerald-400" aria-hidden="true" />
        <span className="text-muted-foreground">Min:</span>
        <span className="text-emerald-400 font-medium tabular-nums">
          {formatLatency(latencyMetrics.min_latency_ms)}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <Activity className="h-3 w-3 text-red-400" aria-hidden="true" />
        <span className="text-muted-foreground">Max:</span>
        <span className="text-red-400 font-medium tabular-nums">
          {formatLatency(latencyMetrics.max_latency_ms)}
        </span>
      </div>
    </div>
  );
});

/**
 * Loading skeleton
 */
const LoadingSkeleton = memo(function LoadingSkeleton({
  compact,
}: {
  compact?: boolean;
}) {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-4 w-32 bg-white/10 rounded" />
      <div className="h-10 w-40 bg-white/10 rounded" />
      <div className="h-8 w-full bg-white/10 rounded" />
      <div className={cn("grid gap-2", compact ? "grid-cols-2" : "grid-cols-4")}>
        {[...Array(compact ? 2 : 4)].map((_, i) => (
          <div key={i} className="h-14 bg-white/10 rounded" />
        ))}
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * LatencyCard displays real-time API and processing latency metrics.
 *
 * Features:
 * - Large latency display with color coding
 * - API/processing latency breakdown with visual bar
 * - Percentile metrics (Avg, P50, P95, P99)
 * - Latency distribution histogram
 * - Latency trend line chart
 * - Min/Max latency values
 * - Latency trend indicator
 */
export const LatencyCard = memo(function LatencyCard({
  latencyMetrics,
  latencyHistory,
  latencyTrend: externalLatencyTrend,
  latencyTrendChange: externalLatencyTrendChange,
  isLoading = false,
  className,
  compact = false,
  showChart = true,
}: LatencyCardProps) {
  // Get color configuration based on average latency
  const colorConfig = useMemo(
    () => getLatencyColorConfig(latencyMetrics.avg_latency_ms),
    [latencyMetrics.avg_latency_ms]
  );

  // Calculate trend if not provided externally
  const { trend, change } = useMemo(() => {
    if (
      externalLatencyTrend !== undefined &&
      externalLatencyTrendChange !== undefined
    ) {
      return { trend: externalLatencyTrend, change: externalLatencyTrendChange };
    }
    return calculateLatencyTrend(latencyHistory, latencyMetrics.avg_latency_ms);
  }, [
    externalLatencyTrend,
    externalLatencyTrendChange,
    latencyHistory,
    latencyMetrics.avg_latency_ms,
  ]);

  // Generate histogram data
  const histogramData = useMemo(
    () => generateHistogramData(latencyMetrics),
    [latencyMetrics]
  );

  // Loading state
  if (isLoading) {
    return (
      <GlassCard
        variant="default"
        intensity="medium"
        className={cn("p-4", className)}
      >
        <LoadingSkeleton compact={compact} />
      </GlassCard>
    );
  }

  return (
    <GlassCard
      variant="default"
      intensity="medium"
      className={cn("p-4 overflow-hidden", className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Gauge className="h-4 w-4 text-cyan-400" aria-hidden="true" />
          <h3 className="text-sm font-medium text-muted-foreground">
            Latency Metrics
          </h3>
          <Badge
            variant="outline"
            className={cn(
              "text-xs",
              colorConfig.bgClass,
              colorConfig.borderClass,
              colorConfig.textClass
            )}
          >
            {colorConfig.label}
          </Badge>
        </div>
        <LatencyTrendIndicator trend={trend} change={change} compact={compact} />
      </div>

      {/* Main Latency Display */}
      <div className="flex items-start justify-between mb-4">
        <LatencyDisplay
          latencyMs={latencyMetrics.avg_latency_ms}
          colorConfig={colorConfig}
          compact={compact}
        />
      </div>

      {/* Latency Breakdown (API vs Processing) */}
      <div className={cn("mb-4", compact && "mb-3")}>
        <LatencyBreakdown latencyMetrics={latencyMetrics} compact={compact} />
      </div>

      {/* Histogram Chart */}
      {showChart && !compact && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground">Latency Distribution</span>
          </div>
          <LatencyHistogram data={histogramData} height={80} />
        </div>
      )}

      {/* Trend Chart */}
      {showChart && !compact && latencyHistory.length >= 2 && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground">Latency Trend</span>
          </div>
          <LatencyTrendChart data={latencyHistory} height={60} />
        </div>
      )}

      {/* Percentile Metrics */}
      <PercentileMetrics latencyMetrics={latencyMetrics} compact={compact} />

      {/* Footer with Min/Max (non-compact only) */}
      {!compact && (
        <div className="mt-3 pt-3 border-t border-white/10">
          <MinMaxLatency latencyMetrics={latencyMetrics} />
        </div>
      )}
    </GlassCard>
  );
});

// Named export for index
export default LatencyCard;
