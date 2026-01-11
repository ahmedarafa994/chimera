/**
 * SuccessRateTimeSeriesChart Component
 *
 * Line chart showing success rate evolution over campaign duration.
 * Supports zoom, tooltips, multiple series for comparison.
 * Uses Recharts LineChart with responsive container.
 */

"use client";

import * as React from "react";
import { useMemo, useState, useCallback, useRef, Suspense } from "react";
import {
  ZoomIn,
  ZoomOut,
  RefreshCw,
  Download,
  Maximize2,
  TrendingUp,
  TrendingDown,
  Minus,
  Info,
  AlertTriangle,
} from "lucide-react";
import { RechartsComponents } from "@/lib/components/lazy-components";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  TelemetryTimeSeries,
  TimeSeriesDataPoint,
  TimeGranularity,
  CampaignSummary,
} from "@/types/campaign-analytics";

// Destructure Recharts components from lazy loader
const {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip: RechartsTooltip,
  Legend,
  ResponsiveContainer,
} = RechartsComponents;

// =============================================================================
// Types
// =============================================================================

/**
 * Data point for chart rendering with normalized timestamp.
 */
export interface ChartDataPoint {
  timestamp: string;
  formattedTime: string;
  value: number;
  count?: number | null;
  [key: string]: string | number | null | undefined; // For multiple series
}

/**
 * Series configuration for comparison mode.
 */
export interface ChartSeries {
  id: string;
  name: string;
  color: string;
  data: TimeSeriesDataPoint[];
  campaign?: CampaignSummary;
}

/**
 * Zoom state for chart interaction.
 */
export interface ZoomState {
  left: number | string;
  right: number | string;
  refAreaLeft: string | null;
  refAreaRight: string | null;
  isZooming: boolean;
}

/**
 * Props for the SuccessRateTimeSeriesChart component.
 */
export interface SuccessRateTimeSeriesChartProps {
  /** Primary time series data */
  timeSeries?: TelemetryTimeSeries | null;
  /** Additional series for comparison */
  comparisonSeries?: ChartSeries[];
  /** Chart title */
  title?: string;
  /** Chart description */
  description?: string;
  /** Height of the chart in pixels */
  height?: number;
  /** Show zoom controls */
  showZoomControls?: boolean;
  /** Show granularity selector */
  showGranularitySelector?: boolean;
  /** Show download button */
  showDownloadButton?: boolean;
  /** Show legend */
  showLegend?: boolean;
  /** Show trend indicator */
  showTrendIndicator?: boolean;
  /** Show data point counts */
  showPointCounts?: boolean;
  /** Granularity options available */
  granularityOptions?: TimeGranularity[];
  /** Current granularity */
  granularity?: TimeGranularity;
  /** Callback when granularity changes */
  onGranularityChange?: (granularity: TimeGranularity) => void;
  /** Callback when zoom changes */
  onZoomChange?: (startTime: string | null, endTime: string | null) => void;
  /** Callback when chart is exported */
  onExport?: (format: "png" | "svg") => void;
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: string | null;
  /** Custom CSS class */
  className?: string;
  /** Reference to chart container for export */
  chartRef?: React.RefObject<HTMLDivElement>;
}

// =============================================================================
// Constants
// =============================================================================

/** Default series colors for comparison */
const SERIES_COLORS = [
  "#22c55e", // green-500 (primary)
  "#3b82f6", // blue-500
  "#f59e0b", // amber-500
  "#ef4444", // red-500
  "#8b5cf6", // violet-500
  "#06b6d4", // cyan-500
];

/** Granularity labels for display */
const GRANULARITY_LABELS: Record<TimeGranularity, string> = {
  minute: "Per Minute",
  hour: "Per Hour",
  day: "Per Day",
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Format timestamp based on granularity for display.
 */
function formatTimestamp(timestamp: string, granularity: TimeGranularity): string {
  const date = new Date(timestamp);

  switch (granularity) {
    case "minute":
      return date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false
      });
    case "hour":
      return date.toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        hour12: false,
      });
    case "day":
      return date.toLocaleDateString([], {
        month: "short",
        day: "numeric",
      });
    default:
      return timestamp;
  }
}

/**
 * Format value as percentage for display.
 */
function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Calculate trend from data points.
 */
function calculateTrend(dataPoints: TimeSeriesDataPoint[]): {
  direction: "up" | "down" | "neutral";
  change: number;
} {
  if (dataPoints.length < 2) {
    return { direction: "neutral", change: 0 };
  }

  const first = dataPoints[0].value;
  const last = dataPoints[dataPoints.length - 1].value;
  const change = last - first;

  if (Math.abs(change) < 0.01) {
    return { direction: "neutral", change };
  }

  return {
    direction: change > 0 ? "up" : "down",
    change,
  };
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Loading skeleton for the chart.
 */
export function SuccessRateTimeSeriesChartSkeleton({
  height = 400,
  className,
}: {
  height?: number;
  className?: string;
}) {
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <Skeleton className="h-5 w-48" />
          <div className="flex gap-2">
            <Skeleton className="h-8 w-24" />
            <Skeleton className="h-8 w-8" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Skeleton className="w-full" style={{ height: `${height}px` }} />
        <div className="mt-4 flex items-center justify-center gap-4">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-24" />
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Error state for the chart.
 */
export function SuccessRateTimeSeriesChartError({
  error,
  onRetry,
  className,
}: {
  error: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <Card className={cn("border-destructive/50", className)}>
      <CardContent className="flex flex-col items-center justify-center py-12">
        <AlertTriangle className="h-12 w-12 text-destructive mb-4" />
        <p className="text-lg font-medium text-destructive mb-2">
          Failed to Load Chart
        </p>
        <p className="text-sm text-muted-foreground mb-4 text-center max-w-md">
          {error}
        </p>
        {onRetry && (
          <Button variant="outline" onClick={onRetry}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Empty state for the chart.
 */
export function SuccessRateTimeSeriesChartEmpty({
  message = "No data available for the selected time range",
  className,
}: {
  message?: string;
  className?: string;
}) {
  return (
    <Card className={className}>
      <CardContent className="flex flex-col items-center justify-center py-12">
        <TrendingUp className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <p className="text-lg font-medium text-muted-foreground mb-2">
          No Time Series Data
        </p>
        <p className="text-sm text-muted-foreground text-center max-w-md">
          {message}
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Custom tooltip component for the chart.
 */
interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    value: number;
    name: string;
    color: string;
    dataKey: string;
    payload: ChartDataPoint;
  }>;
  label?: string;
  showCounts?: boolean;
}

function CustomChartTooltip({
  active,
  payload,
  label,
  showCounts = true,
}: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  return (
    <Card className="p-3 shadow-lg border bg-popover">
      <p className="font-semibold text-sm mb-2 text-popover-foreground">
        {label}
      </p>
      <div className="space-y-1.5">
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-sm text-muted-foreground">
                {entry.name}:
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-medium text-sm">
                {formatPercentage(entry.value)}
              </span>
              {showCounts && entry.payload?.count != null && (
                <Badge variant="secondary" className="text-xs py-0 px-1">
                  n={entry.payload.count}
                </Badge>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/**
 * Trend indicator component.
 */
function TrendIndicator({
  direction,
  change,
}: {
  direction: "up" | "down" | "neutral";
  change: number;
}) {
  const Icon = direction === "up"
    ? TrendingUp
    : direction === "down"
    ? TrendingDown
    : Minus;

  const colorClass = direction === "up"
    ? "text-green-500"
    : direction === "down"
    ? "text-red-500"
    : "text-muted-foreground";

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={cn("flex items-center gap-1", colorClass)}>
            <Icon className="h-4 w-4" />
            <span className="text-sm font-medium">
              {direction === "neutral"
                ? "Stable"
                : `${change > 0 ? "+" : ""}${formatPercentage(change)}`}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {direction === "up"
              ? "Success rate improved"
              : direction === "down"
              ? "Success rate declined"
              : "Success rate remained stable"}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * Success Rate Time Series Chart Component
 *
 * Displays success rate evolution over time with support for:
 * - Multiple series comparison
 * - Zoom controls
 * - Granularity selection
 * - Export functionality
 * - Interactive tooltips
 */
export function SuccessRateTimeSeriesChart({
  timeSeries,
  comparisonSeries = [],
  title = "Success Rate Over Time",
  description,
  height = 400,
  showZoomControls = true,
  showGranularitySelector = true,
  showDownloadButton = true,
  showLegend = true,
  showTrendIndicator = true,
  showPointCounts = true,
  granularityOptions = ["minute", "hour", "day"],
  granularity = "hour",
  onGranularityChange,
  onZoomChange,
  onExport,
  isLoading = false,
  error = null,
  className,
  chartRef: externalChartRef,
}: SuccessRateTimeSeriesChartProps) {
  // Internal chart ref if not provided
  const internalChartRef = useRef<HTMLDivElement>(null);
  const chartContainerRef = externalChartRef || internalChartRef;

  // Zoom state
  const [zoomState, setZoomState] = useState<ZoomState>({
    left: "dataMin",
    right: "dataMax",
    refAreaLeft: null,
    refAreaRight: null,
    isZooming: false,
  });

  // Prepare chart data
  const chartData = useMemo((): ChartDataPoint[] => {
    if (!timeSeries?.data_points?.length && comparisonSeries.length === 0) {
      return [];
    }

    // If we have primary time series data
    if (timeSeries?.data_points?.length) {
      const dataMap = new Map<string, ChartDataPoint>();

      // Process primary series
      timeSeries.data_points.forEach((point) => {
        const formattedTime = formatTimestamp(point.timestamp, granularity);
        dataMap.set(point.timestamp, {
          timestamp: point.timestamp,
          formattedTime,
          value: point.value,
          count: point.count,
        });
      });

      // Process comparison series
      comparisonSeries.forEach((series) => {
        series.data.forEach((point) => {
          const existing = dataMap.get(point.timestamp);
          if (existing) {
            existing[`${series.id}_value`] = point.value;
            existing[`${series.id}_count`] = point.count;
          } else {
            const formattedTime = formatTimestamp(point.timestamp, granularity);
            dataMap.set(point.timestamp, {
              timestamp: point.timestamp,
              formattedTime,
              value: 0, // No primary data for this timestamp
              count: null,
              [`${series.id}_value`]: point.value,
              [`${series.id}_count`]: point.count,
            });
          }
        });
      });

      // Sort by timestamp and return
      return Array.from(dataMap.values()).sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
    }

    // Only comparison series data
    if (comparisonSeries.length > 0) {
      const dataMap = new Map<string, ChartDataPoint>();

      comparisonSeries.forEach((series) => {
        series.data.forEach((point) => {
          const formattedTime = formatTimestamp(point.timestamp, granularity);
          const existing = dataMap.get(point.timestamp);

          if (existing) {
            existing[`${series.id}_value`] = point.value;
            existing[`${series.id}_count`] = point.count;
          } else {
            dataMap.set(point.timestamp, {
              timestamp: point.timestamp,
              formattedTime,
              value: 0,
              count: null,
              [`${series.id}_value`]: point.value,
              [`${series.id}_count`]: point.count,
            });
          }
        });
      });

      return Array.from(dataMap.values()).sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
    }

    return [];
  }, [timeSeries, comparisonSeries, granularity]);

  // Calculate trend for primary series
  const trend = useMemo(() => {
    if (!timeSeries?.data_points?.length) {
      return { direction: "neutral" as const, change: 0 };
    }
    return calculateTrend(timeSeries.data_points);
  }, [timeSeries]);

  // Zoom handlers
  const handleZoomIn = useCallback(() => {
    if (chartData.length < 3) return;

    const middleIndex = Math.floor(chartData.length / 2);
    const quarter = Math.floor(chartData.length / 4);

    const leftIndex = Math.max(0, middleIndex - quarter);
    const rightIndex = Math.min(chartData.length - 1, middleIndex + quarter);

    setZoomState((prev) => ({
      ...prev,
      left: chartData[leftIndex].formattedTime,
      right: chartData[rightIndex].formattedTime,
    }));

    onZoomChange?.(chartData[leftIndex].timestamp, chartData[rightIndex].timestamp);
  }, [chartData, onZoomChange]);

  const handleZoomOut = useCallback(() => {
    setZoomState({
      left: "dataMin",
      right: "dataMax",
      refAreaLeft: null,
      refAreaRight: null,
      isZooming: false,
    });
    onZoomChange?.(null, null);
  }, [onZoomChange]);

  // Handle granularity change
  const handleGranularityChange = useCallback(
    (value: string) => {
      onGranularityChange?.(value as TimeGranularity);
    },
    [onGranularityChange]
  );

  // Handle export
  const handleExport = useCallback(
    (format: "png" | "svg") => {
      onExport?.(format);
    },
    [onExport]
  );

  // Render loading state
  if (isLoading) {
    return <SuccessRateTimeSeriesChartSkeleton height={height} className={className} />;
  }

  // Render error state
  if (error) {
    return <SuccessRateTimeSeriesChartError error={error} className={className} />;
  }

  // Render empty state
  if (chartData.length === 0) {
    return <SuccessRateTimeSeriesChartEmpty className={className} />;
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <CardTitle className="text-base font-medium">{title}</CardTitle>
            {showTrendIndicator && (
              <TrendIndicator direction={trend.direction} change={trend.change} />
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Granularity selector */}
            {showGranularitySelector && (
              <Select value={granularity} onValueChange={handleGranularityChange}>
                <SelectTrigger className="w-32 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {granularityOptions.map((option) => (
                    <SelectItem key={option} value={option}>
                      {GRANULARITY_LABELS[option]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            {/* Zoom controls */}
            {showZoomControls && (
              <div className="flex items-center gap-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={handleZoomIn}
                        disabled={chartData.length < 3}
                      >
                        <ZoomIn className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Zoom In</TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={handleZoomOut}
                      >
                        <ZoomOut className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Zoom Out / Reset</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            )}

            {/* Download button */}
            {showDownloadButton && onExport && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => handleExport("png")}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Export Chart</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
        </div>

        {description && (
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
        )}
      </CardHeader>

      <CardContent className="pt-0">
        <div ref={chartContainerRef}>
          <Suspense
            fallback={
              <Skeleton className="w-full" style={{ height: `${height}px` }} />
            }
          >
            <ResponsiveContainer width="100%" height={height}>
              <LineChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <defs>
                  {/* Gradient for primary series */}
                  <linearGradient id="successRateGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={SERIES_COLORS[0]} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={SERIES_COLORS[0]} stopOpacity={0} />
                  </linearGradient>
                  {/* Gradients for comparison series */}
                  {comparisonSeries.map((series, index) => (
                    <linearGradient
                      key={series.id}
                      id={`gradient-${series.id}`}
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="5%"
                        stopColor={series.color || SERIES_COLORS[(index + 1) % SERIES_COLORS.length]}
                        stopOpacity={0.3}
                      />
                      <stop
                        offset="95%"
                        stopColor={series.color || SERIES_COLORS[(index + 1) % SERIES_COLORS.length]}
                        stopOpacity={0}
                      />
                    </linearGradient>
                  ))}
                </defs>

                <CartesianGrid
                  strokeDasharray="3 3"
                  className="stroke-muted"
                  opacity={0.5}
                />

                <XAxis
                  dataKey="formattedTime"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  interval="preserveStartEnd"
                  minTickGap={50}
                  domain={[zoomState.left, zoomState.right]}
                />

                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  domain={[0, 1]}
                  width={45}
                />

                <RechartsTooltip
                  content={<CustomChartTooltip showCounts={showPointCounts} />}
                  cursor={{ stroke: "hsl(var(--muted-foreground))", strokeWidth: 1 }}
                />

                {showLegend && (
                  <Legend
                    verticalAlign="bottom"
                    height={36}
                    iconType="circle"
                    formatter={(value) => (
                      <span className="text-sm text-foreground">{value}</span>
                    )}
                  />
                )}

                {/* Primary series */}
                {timeSeries?.data_points?.length && (
                  <Line
                    type="monotone"
                    dataKey="value"
                    name={timeSeries.campaign_id ? "Primary Campaign" : "Success Rate"}
                    stroke={SERIES_COLORS[0]}
                    strokeWidth={2}
                    dot={{ fill: SERIES_COLORS[0], r: 3, strokeWidth: 0 }}
                    activeDot={{ r: 6, strokeWidth: 2, stroke: "#fff" }}
                    connectNulls
                  />
                )}

                {/* Comparison series */}
                {comparisonSeries.map((series, index) => (
                  <Line
                    key={series.id}
                    type="monotone"
                    dataKey={`${series.id}_value`}
                    name={series.name}
                    stroke={series.color || SERIES_COLORS[(index + 1) % SERIES_COLORS.length]}
                    strokeWidth={2}
                    dot={{
                      fill: series.color || SERIES_COLORS[(index + 1) % SERIES_COLORS.length],
                      r: 3,
                      strokeWidth: 0,
                    }}
                    activeDot={{ r: 6, strokeWidth: 2, stroke: "#fff" }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Suspense>
        </div>

        {/* Data summary */}
        <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-4">
            <span>
              {timeSeries?.total_points ?? chartData.length} data points
            </span>
            {timeSeries?.start_time && timeSeries?.end_time && (
              <span>
                {new Date(timeSeries.start_time).toLocaleDateString()} -{" "}
                {new Date(timeSeries.end_time).toLocaleDateString()}
              </span>
            )}
          </div>
          {comparisonSeries.length > 0 && (
            <Badge variant="secondary">
              {comparisonSeries.length + 1} series
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * Simple success rate chart without comparison features.
 */
export function SimpleSuccessRateChart({
  timeSeries,
  height = 300,
  className,
}: {
  timeSeries?: TelemetryTimeSeries | null;
  height?: number;
  className?: string;
}) {
  return (
    <SuccessRateTimeSeriesChart
      timeSeries={timeSeries}
      height={height}
      showZoomControls={false}
      showGranularitySelector={false}
      showDownloadButton={false}
      showLegend={false}
      showTrendIndicator={true}
      showPointCounts={false}
      className={className}
    />
  );
}

/**
 * Comparison chart with multiple campaigns.
 */
export function ComparisonSuccessRateChart({
  comparisonSeries,
  title = "Campaign Comparison",
  height = 400,
  className,
}: {
  comparisonSeries: ChartSeries[];
  title?: string;
  height?: number;
  className?: string;
}) {
  return (
    <SuccessRateTimeSeriesChart
      comparisonSeries={comparisonSeries}
      title={title}
      height={height}
      showZoomControls={true}
      showGranularitySelector={false}
      showDownloadButton={true}
      showLegend={true}
      showTrendIndicator={false}
      className={className}
    />
  );
}

// =============================================================================
// Export Types
// =============================================================================

export type {
  ChartDataPoint,
  ChartSeries,
  ZoomState,
  SuccessRateTimeSeriesChartProps,
};
