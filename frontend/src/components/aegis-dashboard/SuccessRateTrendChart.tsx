/**
 * SuccessRateTrendChart Component
 *
 * Displays attack success rate over time for Aegis Campaign Dashboard with:
 * - Responsive area chart with gradient fill
 * - X-axis: time (last 30 minutes or campaign duration)
 * - Y-axis: success rate percentage (0-100%)
 * - Interactive tooltip showing exact values
 * - Smooth animation for new data points
 *
 * Follows glass morphism styling pattern from existing components.
 */

"use client";

import { memo, useMemo, useCallback } from "react";
import { Activity, TrendingUp, Clock } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import { cn } from "@/lib/utils";
import { SuccessRateTimeSeries } from "@/types/aegis-telemetry";
import {
  Area,
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

// ============================================================================
// Types
// ============================================================================

export interface SuccessRateTrendChartProps {
  /** Time series data for the chart */
  data: SuccessRateTimeSeries[];
  /** Current success rate percentage */
  currentRate?: number;
  /** Campaign start time (ISO string) */
  campaignStartTime?: string | null;
  /** Whether the component is in loading state */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Height of the chart (default: 300) */
  height?: number;
  /** Whether to show the header */
  showHeader?: boolean;
  /** Whether to show grid lines */
  showGrid?: boolean;
  /** Whether to show reference line at 50% */
  showReferenceLine?: boolean;
  /** Time window to display in minutes (default: 30) */
  timeWindowMinutes?: number;
}

// ============================================================================
// Configuration
// ============================================================================

/** Chart configuration for ChartContainer */
const chartConfig: ChartConfig = {
  successRate: {
    label: "Success Rate",
    color: "hsl(var(--chart-1))",
  },
};

/** Color configuration based on current success rate */
const getChartColorConfig = (rate: number) => {
  if (rate >= 70) {
    return {
      stroke: "#10b981",
      fill: "#10b981",
      label: "Excellent",
      bgClass: "bg-emerald-500/10",
      borderClass: "border-emerald-500/20",
      textClass: "text-emerald-400",
    };
  } else if (rate >= 40) {
    return {
      stroke: "#f59e0b",
      fill: "#f59e0b",
      label: "Moderate",
      bgClass: "bg-amber-500/10",
      borderClass: "border-amber-500/20",
      textClass: "text-amber-400",
    };
  } else {
    return {
      stroke: "#ef4444",
      fill: "#ef4444",
      label: "Low",
      bgClass: "bg-red-500/10",
      borderClass: "border-red-500/20",
      textClass: "text-red-400",
    };
  }
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format timestamp for X-axis display
 */
function formatTimeLabel(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

/**
 * Format timestamp for tooltip display
 */
function formatTooltipTime(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "";
  }
}

/**
 * Calculate time range display text
 */
function getTimeRangeText(
  data: SuccessRateTimeSeries[],
  campaignStartTime?: string | null
): string {
  if (data.length === 0) return "No data";

  const firstTimestamp = data[0].timestamp;
  const lastTimestamp = data[data.length - 1].timestamp;

  try {
    const start = new Date(firstTimestamp);
    const end = new Date(lastTimestamp);
    const durationMs = end.getTime() - start.getTime();
    const durationMinutes = Math.round(durationMs / 60000);

    if (durationMinutes < 1) {
      return "< 1 min";
    } else if (durationMinutes < 60) {
      return `${durationMinutes} min`;
    } else {
      const hours = Math.floor(durationMinutes / 60);
      const mins = durationMinutes % 60;
      return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
    }
  } catch {
    return "Unknown";
  }
}

/**
 * Filter data to time window
 */
function filterToTimeWindow(
  data: SuccessRateTimeSeries[],
  timeWindowMinutes: number
): SuccessRateTimeSeries[] {
  if (data.length === 0) return [];

  const now = new Date();
  const cutoffTime = new Date(now.getTime() - timeWindowMinutes * 60 * 1000);

  return data.filter((point) => {
    try {
      const pointTime = new Date(point.timestamp);
      return pointTime >= cutoffTime;
    } catch {
      return false;
    }
  });
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Chart header with title and status
 */
const ChartHeader = memo(function ChartHeader({
  currentRate,
  timeRangeText,
  dataPointCount,
}: {
  currentRate: number;
  timeRangeText: string;
  dataPointCount: number;
}) {
  const colorConfig = getChartColorConfig(currentRate);

  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <Activity
          className={cn("h-4 w-4", colorConfig.textClass)}
          aria-hidden="true"
        />
        <h3 className="text-sm font-medium text-foreground">
          Success Rate Trend
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
      <div className="flex items-center gap-3">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" aria-hidden="true" />
                <span>{timeRangeText}</span>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>Time range of displayed data</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="secondary" className="text-xs">
                {dataPointCount} points
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <p>Number of data points in the chart</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
});

/**
 * Custom tooltip content for the chart
 */
const CustomTooltip = memo(function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{
    value: number;
    payload: {
      timestamp: string;
      success_rate: number;
      total_attempts: number;
      successful_attacks: number;
    };
  }>;
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload;
  const colorConfig = getChartColorConfig(data.success_rate);

  return (
    <div className="bg-background/95 backdrop-blur-sm border border-white/10 rounded-lg p-3 shadow-xl">
      <div className="flex items-center gap-2 mb-2 pb-2 border-b border-white/10">
        <div
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: colorConfig.stroke }}
        />
        <span className="text-xs font-medium text-foreground">
          {formatTooltipTime(data.timestamp)}
        </span>
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-muted-foreground">Success Rate</span>
          <span
            className={cn("text-sm font-bold tabular-nums", colorConfig.textClass)}
          >
            {data.success_rate.toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-muted-foreground">Total Attempts</span>
          <span className="text-xs font-medium text-foreground tabular-nums">
            {data.total_attempts.toLocaleString()}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-muted-foreground">Successful</span>
          <span className="text-xs font-medium text-emerald-400 tabular-nums">
            {data.successful_attacks.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
});

/**
 * Empty state when no data is available
 */
const EmptyState = memo(function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 py-12">
      <div className="p-3 rounded-full bg-white/5">
        <TrendingUp className="h-6 w-6 text-muted-foreground" aria-hidden="true" />
      </div>
      <div className="text-center">
        <p className="text-sm text-muted-foreground">No data available</p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          Success rate history will appear here as attacks are executed
        </p>
      </div>
    </div>
  );
});

/**
 * Loading skeleton for the chart
 */
const LoadingSkeleton = memo(function LoadingSkeleton({
  height,
}: {
  height: number;
}) {
  return (
    <div className="animate-pulse space-y-4">
      <div className="flex items-center justify-between">
        <div className="h-4 w-32 bg-white/10 rounded" />
        <div className="h-4 w-20 bg-white/10 rounded" />
      </div>
      <div
        className="bg-white/5 rounded-lg flex items-end justify-around px-4 pb-4"
        style={{ height }}
      >
        {[...Array(12)].map((_, i) => (
          <div
            key={i}
            className="w-4 bg-white/10 rounded-t"
            style={{ height: `${((i * 13) % 60) + 20}%` }}
          />
        ))}
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * SuccessRateTrendChart displays attack success rate over time for the Aegis Dashboard.
 *
 * Features:
 * - Responsive area chart with gradient fill
 * - X-axis showing time (last 30 minutes by default or campaign duration)
 * - Y-axis showing success rate percentage (0-100%)
 * - Interactive tooltip with detailed metrics
 * - Smooth animation for new data points
 * - Optional reference line at 50% threshold
 */
export const SuccessRateTrendChart = memo(function SuccessRateTrendChart({
  data,
  currentRate,
  campaignStartTime,
  isLoading = false,
  className,
  height = 300,
  showHeader = true,
  showGrid = true,
  showReferenceLine = true,
  timeWindowMinutes = 30,
}: SuccessRateTrendChartProps) {
  // Filter data to time window
  const filteredData = useMemo(
    () => filterToTimeWindow(data, timeWindowMinutes),
    [data, timeWindowMinutes]
  );

  // Calculate current rate from data if not provided
  const effectiveCurrentRate = useMemo(() => {
    if (currentRate !== undefined) return currentRate;
    if (filteredData.length > 0) {
      return filteredData[filteredData.length - 1].success_rate;
    }
    return 0;
  }, [currentRate, filteredData]);

  // Get color config based on current rate
  const colorConfig = useMemo(
    () => getChartColorConfig(effectiveCurrentRate),
    [effectiveCurrentRate]
  );

  // Transform data for chart
  const chartData = useMemo(() => {
    return filteredData.map((point, index) => ({
      ...point,
      index,
      timeLabel: formatTimeLabel(point.timestamp),
    }));
  }, [filteredData]);

  // Calculate time range text
  const timeRangeText = useMemo(
    () => getTimeRangeText(filteredData, campaignStartTime),
    [filteredData, campaignStartTime]
  );

  // Custom X-axis tick formatter
  const formatXAxisTick = useCallback((value: string) => {
    return formatTimeLabel(value);
  }, []);

  // Loading state
  if (isLoading) {
    return (
      <GlassCard
        variant="default"
        intensity="medium"
        className={cn("p-4", className)}
      >
        <LoadingSkeleton height={height} />
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
      {showHeader && (
        <ChartHeader
          currentRate={effectiveCurrentRate}
          timeRangeText={timeRangeText}
          dataPointCount={chartData.length}
        />
      )}

      {/* Chart Area */}
      {chartData.length < 2 ? (
        <div style={{ height }}>
          <EmptyState />
        </div>
      ) : (
        <ChartContainer config={chartConfig} className="w-full" style={{ height }}>
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="successRateGradient" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={colorConfig.fill}
                  stopOpacity={0.4}
                />
                <stop
                  offset="50%"
                  stopColor={colorConfig.fill}
                  stopOpacity={0.15}
                />
                <stop
                  offset="95%"
                  stopColor={colorConfig.fill}
                  stopOpacity={0.02}
                />
              </linearGradient>
            </defs>

            {/* Grid lines */}
            {showGrid && (
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.05)"
                vertical={false}
              />
            )}

            {/* X-axis: Time */}
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatXAxisTick}
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
              tickLine={{ stroke: "rgba(255,255,255,0.1)" }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              minTickGap={50}
              interval="preserveStartEnd"
            />

            {/* Y-axis: Success Rate (0-100%) */}
            <YAxis
              domain={[0, 100]}
              tickFormatter={(value: number) => `${value}%`}
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
              tickLine={{ stroke: "rgba(255,255,255,0.1)" }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              width={45}
              ticks={[0, 25, 50, 75, 100]}
            />

            {/* Reference line at 50% threshold */}
            {showReferenceLine && (
              <ReferenceLine
                y={50}
                stroke="rgba(255,255,255,0.2)"
                strokeDasharray="5 5"
                label={{
                  value: "50%",
                  position: "right",
                  fill: "rgba(255,255,255,0.4)",
                  fontSize: 10,
                }}
              />
            )}

            {/* Interactive tooltip */}
            <ChartTooltip
              content={<CustomTooltip />}
              cursor={{
                stroke: "rgba(255,255,255,0.2)",
                strokeWidth: 1,
              }}
            />

            {/* Area chart */}
            <Area
              type="monotone"
              dataKey="success_rate"
              stroke={colorConfig.stroke}
              strokeWidth={2}
              fill="url(#successRateGradient)"
              isAnimationActive={true}
              animationDuration={500}
              animationEasing="ease-out"
              dot={false}
              activeDot={{
                r: 4,
                fill: colorConfig.fill,
                stroke: "rgba(255,255,255,0.8)",
                strokeWidth: 2,
              }}
            />
          </AreaChart>
        </ChartContainer>
      )}

      {/* Footer with current stats */}
      {chartData.length >= 2 && (
        <div className="mt-3 pt-3 border-t border-white/10">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <div
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: colorConfig.stroke }}
                />
                <span className="text-muted-foreground">Current</span>
                <span
                  className={cn("font-medium tabular-nums", colorConfig.textClass)}
                >
                  {effectiveCurrentRate.toFixed(1)}%
                </span>
              </div>
              {chartData.length > 0 && (
                <div className="flex items-center gap-1.5">
                  <span className="text-muted-foreground">Total Attacks</span>
                  <span className="font-medium text-foreground tabular-nums">
                    {chartData[chartData.length - 1].total_attempts.toLocaleString()}
                  </span>
                </div>
              )}
            </div>
            <span className="text-muted-foreground/70">
              Last {timeWindowMinutes} min window
            </span>
          </div>
        </div>
      )}
    </GlassCard>
  );
});

// Named export for index
export default SuccessRateTrendChart;
