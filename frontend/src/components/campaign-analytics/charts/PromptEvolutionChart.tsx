/**
 * PromptEvolutionChart Component
 *
 * Combined chart showing prompt iteration count vs success rate correlation.
 * Displays a scatter plot with trend line and tooltips showing prompt snippets.
 * Uses Recharts ScatterChart and ComposedChart with responsive container.
 */

"use client";

import * as React from "react";
import { useMemo, useState, useCallback, useRef, Suspense } from "react";
import {
  Download,
  RefreshCw,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Filter,
  Eye,
  EyeOff,
  Info,
  X,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
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
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import type {
  TelemetryEventSummary,
  ExecutionStatusEnum,
} from "@/types/campaign-analytics";

// Destructure Recharts components from lazy loader
const {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip: RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ZAxis,
  ReferenceLine,
  Cell,
  ComposedChart,
  Line,
} = RechartsComponents;

// =============================================================================
// Types
// =============================================================================

/**
 * Data point for the scatter plot.
 */
export interface PromptEvolutionDataPoint {
  /** Iteration/sequence number (X-axis) */
  iteration: number;
  /** Success rate at this iteration (Y-axis) */
  successRate: number;
  /** Original prompt snippet for tooltip */
  promptSnippet: string;
  /** Full prompt (optional, for drill-down) */
  fullPrompt?: string;
  /** Transformed prompt snippet */
  transformedSnippet?: string;
  /** Technique used */
  technique: string;
  /** Provider used */
  provider: string;
  /** Model used */
  model: string;
  /** Latency in ms */
  latencyMs: number;
  /** Whether this attempt was successful */
  isSuccess: boolean;
  /** Execution status */
  status: ExecutionStatusEnum;
  /** Potency level */
  potencyLevel: number;
  /** Token count */
  totalTokens: number;
  /** Timestamp */
  timestamp: string;
  /** Event ID for reference */
  eventId: string;
}

/**
 * Trend line data point.
 */
export interface TrendLinePoint {
  iteration: number;
  trendValue: number;
}

/**
 * Props for the PromptEvolutionChart component.
 */
export interface PromptEvolutionChartProps {
  /** Telemetry event data */
  events?: TelemetryEventSummary[] | null;
  /** Pre-computed data points (alternative to events) */
  dataPoints?: PromptEvolutionDataPoint[] | null;
  /** Chart title */
  title?: string;
  /** Chart description */
  description?: string;
  /** Height of the chart in pixels */
  height?: number;
  /** Show trend line */
  showTrendLine?: boolean;
  /** Show correlation coefficient */
  showCorrelation?: boolean;
  /** Show download button */
  showDownloadButton?: boolean;
  /** Show legend */
  showLegend?: boolean;
  /** Show filter controls */
  showFilterControls?: boolean;
  /** Show point size based on tokens */
  sizeByTokens?: boolean;
  /** Maximum prompt snippet length in tooltip */
  maxSnippetLength?: number;
  /** Color successful points */
  successColor?: string;
  /** Color failed points */
  failureColor?: string;
  /** Color partial success points */
  partialSuccessColor?: string;
  /** Callback when a data point is clicked */
  onPointClick?: (point: PromptEvolutionDataPoint) => void;
  /** Callback when chart is exported */
  onExport?: (format: "png" | "svg") => void;
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: string | null;
  /** Callback when retry is clicked */
  onRetry?: () => void;
  /** Custom CSS class */
  className?: string;
  /** Reference to chart container for export */
  chartRef?: React.RefObject<HTMLDivElement>;
}

/**
 * Filter configuration for the chart.
 */
export interface PromptEvolutionFilter {
  techniques: string[];
  providers: string[];
  successOnly: boolean;
  minIteration?: number;
  maxIteration?: number;
}

/**
 * Correlation result.
 */
export interface CorrelationResult {
  coefficient: number;
  strength: "strong" | "moderate" | "weak" | "none";
  direction: "positive" | "negative" | "none";
}

// =============================================================================
// Constants
// =============================================================================

/** Default colors */
const DEFAULT_SUCCESS_COLOR = "#22c55e"; // green-500
const DEFAULT_FAILURE_COLOR = "#ef4444"; // red-500
const DEFAULT_PARTIAL_SUCCESS_COLOR = "#f59e0b"; // amber-500
const DEFAULT_TREND_COLOR = "#3b82f6"; // blue-500

/** Trend line styles */
const TREND_LINE_STYLES = {
  stroke: DEFAULT_TREND_COLOR,
  strokeWidth: 2,
  strokeDasharray: "5 5",
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Truncate text to specified length with ellipsis.
 */
function truncateText(text: string, maxLength: number): string {
  if (!text || text.length <= maxLength) return text || "";
  return text.slice(0, maxLength - 3) + "...";
}

/**
 * Convert telemetry events to chart data points.
 */
function eventsToDataPoints(
  events: TelemetryEventSummary[]
): PromptEvolutionDataPoint[] {
  return events
    .sort((a, b) => a.sequence_number - b.sequence_number)
    .map((event, index) => ({
      iteration: event.sequence_number,
      successRate: event.success_indicator ? 1 : 0,
      promptSnippet: event.original_prompt_preview || "",
      technique: event.technique_suite,
      provider: event.provider,
      model: event.model,
      latencyMs: event.total_latency_ms,
      isSuccess: event.success_indicator,
      status: event.status,
      potencyLevel: event.potency_level,
      totalTokens: event.total_tokens,
      timestamp: event.created_at,
      eventId: event.id,
    }));
}

/**
 * Calculate linear regression trend line.
 */
function calculateTrendLine(
  points: PromptEvolutionDataPoint[]
): TrendLinePoint[] {
  if (points.length < 2) return [];

  const n = points.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;

  points.forEach((point) => {
    sumX += point.iteration;
    sumY += point.successRate;
    sumXY += point.iteration * point.successRate;
    sumX2 += point.iteration * point.iteration;
  });

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Generate trend line points
  const minX = Math.min(...points.map((p) => p.iteration));
  const maxX = Math.max(...points.map((p) => p.iteration));

  return [
    { iteration: minX, trendValue: slope * minX + intercept },
    { iteration: maxX, trendValue: slope * maxX + intercept },
  ];
}

/**
 * Calculate cumulative success rate at each iteration.
 */
function calculateCumulativeSuccessRate(
  points: PromptEvolutionDataPoint[]
): PromptEvolutionDataPoint[] {
  let successCount = 0;

  return points.map((point, index) => {
    if (point.isSuccess) {
      successCount++;
    }
    return {
      ...point,
      successRate: successCount / (index + 1),
    };
  });
}

/**
 * Calculate Pearson correlation coefficient.
 */
function calculateCorrelation(
  points: PromptEvolutionDataPoint[]
): CorrelationResult {
  if (points.length < 3) {
    return { coefficient: 0, strength: "none", direction: "none" };
  }

  const n = points.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  let sumY2 = 0;

  points.forEach((point) => {
    sumX += point.iteration;
    sumY += point.successRate;
    sumXY += point.iteration * point.successRate;
    sumX2 += point.iteration * point.iteration;
    sumY2 += point.successRate * point.successRate;
  });

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt(
    (n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY)
  );

  if (denominator === 0) {
    return { coefficient: 0, strength: "none", direction: "none" };
  }

  const r = numerator / denominator;
  const absR = Math.abs(r);

  let strength: "strong" | "moderate" | "weak" | "none";
  if (absR >= 0.7) strength = "strong";
  else if (absR >= 0.4) strength = "moderate";
  else if (absR >= 0.2) strength = "weak";
  else strength = "none";

  let direction: "positive" | "negative" | "none";
  if (r > 0.1) direction = "positive";
  else if (r < -0.1) direction = "negative";
  else direction = "none";

  return { coefficient: r, strength, direction };
}

/**
 * Get color for a data point based on success status.
 */
function getPointColor(
  point: PromptEvolutionDataPoint,
  successColor: string,
  failureColor: string,
  partialSuccessColor: string
): string {
  if (point.status === "partial_success") return partialSuccessColor;
  return point.isSuccess ? successColor : failureColor;
}

/**
 * Format timestamp for display.
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Loading skeleton for the chart.
 */
export function PromptEvolutionChartSkeleton({
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
          <Skeleton className="h-5 w-56" />
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
          <Skeleton className="h-4 w-24" />
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Error state for the chart.
 */
export function PromptEvolutionChartError({
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
export function PromptEvolutionChartEmpty({
  message = "No prompt evolution data available for this campaign",
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
          No Evolution Data
        </p>
        <p className="text-sm text-muted-foreground text-center max-w-md">
          {message}
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Custom tooltip for scatter plot points.
 */
interface ScatterTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: PromptEvolutionDataPoint;
  }>;
  maxSnippetLength: number;
}

function CustomScatterTooltip({
  active,
  payload,
  maxSnippetLength,
}: ScatterTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <Card className="p-3 shadow-lg border bg-popover min-w-[280px] max-w-[400px]">
      <div className="space-y-3">
        {/* Header with iteration and status */}
        <div className="flex items-center justify-between gap-2">
          <span className="font-semibold text-sm text-popover-foreground">
            Iteration #{data.iteration}
          </span>
          <Badge
            variant={data.isSuccess ? "default" : "destructive"}
            className="text-xs"
          >
            {data.isSuccess ? "Success" : "Failed"}
          </Badge>
        </div>

        {/* Prompt snippet */}
        {data.promptSnippet && (
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground font-medium">
              Prompt:
            </span>
            <p className="text-xs bg-muted/50 p-2 rounded border text-foreground break-words">
              {truncateText(data.promptSnippet, maxSnippetLength)}
            </p>
          </div>
        )}

        {/* Transformed prompt snippet */}
        {data.transformedSnippet && (
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground font-medium">
              Transformed:
            </span>
            <p className="text-xs bg-muted/50 p-2 rounded border text-foreground break-words">
              {truncateText(data.transformedSnippet, maxSnippetLength)}
            </p>
          </div>
        )}

        {/* Metrics grid */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Success Rate:</span>
            <span className="font-medium">
              {(data.successRate * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Technique:</span>
            <span className="font-medium truncate max-w-[100px]">
              {data.technique}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Provider:</span>
            <span className="font-medium">{data.provider}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Model:</span>
            <span className="font-medium truncate max-w-[100px]">
              {data.model}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Latency:</span>
            <span className="font-medium">{data.latencyMs.toFixed(0)}ms</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Tokens:</span>
            <span className="font-medium">
              {data.totalTokens.toLocaleString()}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Potency:</span>
            <span className="font-medium">Level {data.potencyLevel}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Time:</span>
            <span className="font-medium">
              {formatTimestamp(data.timestamp)}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
}

/**
 * Correlation indicator component.
 */
function CorrelationIndicator({ correlation }: { correlation: CorrelationResult }) {
  const Icon =
    correlation.direction === "positive"
      ? TrendingUp
      : correlation.direction === "negative"
      ? TrendingDown
      : Minus;

  const colorClass =
    correlation.direction === "positive"
      ? "text-green-500"
      : correlation.direction === "negative"
      ? "text-red-500"
      : "text-muted-foreground";

  const strengthLabel = {
    strong: "Strong",
    moderate: "Moderate",
    weak: "Weak",
    none: "No",
  }[correlation.strength];

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={cn("flex items-center gap-1", colorClass)}>
            <Icon className="h-4 w-4" />
            <span className="text-sm font-medium">
              r = {correlation.coefficient.toFixed(3)}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {strengthLabel}{" "}
            {correlation.direction !== "none" ? correlation.direction : ""}{" "}
            correlation between iteration count and success rate
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Legend component.
 */
function EvolutionChartLegend({
  successColor,
  failureColor,
  partialSuccessColor,
  showTrendLine,
  className,
}: {
  successColor: string;
  failureColor: string;
  partialSuccessColor: string;
  showTrendLine: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center justify-center gap-4 text-xs",
        className
      )}
    >
      <div className="flex items-center gap-1.5">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: successColor }}
        />
        <span className="text-muted-foreground">Successful</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: failureColor }}
        />
        <span className="text-muted-foreground">Failed</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: partialSuccessColor }}
        />
        <span className="text-muted-foreground">Partial Success</span>
      </div>
      {showTrendLine && (
        <div className="flex items-center gap-1.5">
          <div
            className="w-6 h-0.5"
            style={{
              backgroundColor: TREND_LINE_STYLES.stroke,
              backgroundImage: `repeating-linear-gradient(90deg, transparent, transparent 2px, ${TREND_LINE_STYLES.stroke} 2px, ${TREND_LINE_STYLES.stroke} 4px)`,
            }}
          />
          <span className="text-muted-foreground">Trend Line</span>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * Prompt Evolution Chart Component
 *
 * Displays a scatter plot showing the correlation between prompt iteration
 * count and success rate. Features include:
 * - Interactive scatter plot with tooltips
 * - Linear regression trend line
 * - Correlation coefficient display
 * - Color-coded success/failure points
 * - Point sizing by token count (optional)
 * - Filtering by technique and provider
 */
export function PromptEvolutionChart({
  events,
  dataPoints: providedDataPoints,
  title = "Prompt Evolution",
  description,
  height = 400,
  showTrendLine = true,
  showCorrelation = true,
  showDownloadButton = true,
  showLegend = true,
  showFilterControls = false,
  sizeByTokens = false,
  maxSnippetLength = 150,
  successColor = DEFAULT_SUCCESS_COLOR,
  failureColor = DEFAULT_FAILURE_COLOR,
  partialSuccessColor = DEFAULT_PARTIAL_SUCCESS_COLOR,
  onPointClick,
  onExport,
  isLoading = false,
  error = null,
  onRetry,
  className,
  chartRef: externalChartRef,
}: PromptEvolutionChartProps) {
  // Internal chart ref if not provided
  const internalChartRef = useRef<HTMLDivElement>(null);
  const chartContainerRef = externalChartRef || internalChartRef;

  // State
  const [showCumulativeRate, setShowCumulativeRate] = useState(true);
  const [trendLineVisible, setTrendLineVisible] = useState(showTrendLine);

  // Convert events to data points if needed
  const rawDataPoints = useMemo((): PromptEvolutionDataPoint[] => {
    if (providedDataPoints && providedDataPoints.length > 0) {
      return providedDataPoints;
    }
    if (events && events.length > 0) {
      return eventsToDataPoints(events);
    }
    return [];
  }, [events, providedDataPoints]);

  // Calculate cumulative success rate if enabled
  const chartData = useMemo(() => {
    if (showCumulativeRate) {
      return calculateCumulativeSuccessRate(rawDataPoints);
    }
    return rawDataPoints;
  }, [rawDataPoints, showCumulativeRate]);

  // Calculate trend line
  const trendLineData = useMemo(() => {
    if (!trendLineVisible || chartData.length < 2) return [];
    return calculateTrendLine(chartData);
  }, [chartData, trendLineVisible]);

  // Calculate correlation
  const correlation = useMemo(() => {
    return calculateCorrelation(chartData);
  }, [chartData]);

  // Get unique techniques and providers for filtering
  const uniqueTechniques = useMemo(() => {
    return [...new Set(chartData.map((p) => p.technique))];
  }, [chartData]);

  const uniqueProviders = useMemo(() => {
    return [...new Set(chartData.map((p) => p.provider))];
  }, [chartData]);

  // Calculate size domain for ZAxis
  const tokenRange = useMemo(() => {
    if (!sizeByTokens || chartData.length === 0) return [64, 64];
    const tokens = chartData.map((p) => p.totalTokens);
    return [Math.min(...tokens), Math.max(...tokens)];
  }, [chartData, sizeByTokens]);

  // Handlers
  const handlePointClick = useCallback(
    (data: any) => {
      if (data && data.payload && onPointClick) {
        onPointClick(data.payload as PromptEvolutionDataPoint);
      }
    },
    [onPointClick]
  );

  const handleExport = useCallback(
    (format: "png" | "svg") => {
      onExport?.(format);
    },
    [onExport]
  );

  const toggleTrendLine = useCallback(() => {
    setTrendLineVisible((prev) => !prev);
  }, []);

  const toggleCumulativeRate = useCallback(() => {
    setShowCumulativeRate((prev) => !prev);
  }, []);

  // Render loading state
  if (isLoading) {
    return <PromptEvolutionChartSkeleton height={height} className={className} />;
  }

  // Render error state
  if (error) {
    return (
      <PromptEvolutionChartError
        error={error}
        onRetry={onRetry}
        className={className}
      />
    );
  }

  // Render empty state
  if (chartData.length === 0) {
    return <PromptEvolutionChartEmpty className={className} />;
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <CardTitle className="text-base font-medium">{title}</CardTitle>
            {showCorrelation && <CorrelationIndicator correlation={correlation} />}
          </div>

          <div className="flex items-center gap-2">
            {/* Cumulative rate toggle */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="cumulative-rate"
                      checked={showCumulativeRate}
                      onCheckedChange={toggleCumulativeRate}
                      className="h-5"
                    />
                    <Label
                      htmlFor="cumulative-rate"
                      className="text-xs text-muted-foreground cursor-pointer"
                    >
                      Cumulative
                    </Label>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Show cumulative success rate over iterations</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Trend line toggle */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={trendLineVisible ? "secondary" : "ghost"}
                    size="icon"
                    className="h-8 w-8"
                    onClick={toggleTrendLine}
                    aria-label={
                      trendLineVisible ? "Hide trend line" : "Show trend line"
                    }
                  >
                    <TrendingUp className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {trendLineVisible ? "Hide Trend Line" : "Show Trend Line"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

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
                      aria-label="Export chart"
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
              <ScatterChart
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  className="stroke-muted"
                  opacity={0.5}
                />

                <XAxis
                  type="number"
                  dataKey="iteration"
                  name="Iteration"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  domain={["dataMin", "dataMax"]}
                  label={{
                    value: "Prompt Iteration",
                    position: "bottom",
                    offset: 0,
                    style: { fontSize: 12, fill: "hsl(var(--muted-foreground))" },
                  }}
                />

                <YAxis
                  type="number"
                  dataKey="successRate"
                  name="Success Rate"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  domain={[0, 1]}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  width={50}
                  label={{
                    value: "Success Rate",
                    angle: -90,
                    position: "insideLeft",
                    style: { fontSize: 12, fill: "hsl(var(--muted-foreground))" },
                  }}
                />

                {sizeByTokens && (
                  <ZAxis
                    type="number"
                    dataKey="totalTokens"
                    domain={tokenRange as [number, number]}
                    range={[40, 200]}
                    name="Tokens"
                  />
                )}

                <RechartsTooltip
                  content={
                    <CustomScatterTooltip maxSnippetLength={maxSnippetLength} />
                  }
                  cursor={{ strokeDasharray: "3 3" }}
                />

                {/* Trend line reference lines */}
                {trendLineVisible && trendLineData.length === 2 && (
                  <ReferenceLine
                    segment={[
                      {
                        x: trendLineData[0].iteration,
                        y: trendLineData[0].trendValue,
                      },
                      {
                        x: trendLineData[1].iteration,
                        y: trendLineData[1].trendValue,
                      },
                    ]}
                    stroke={TREND_LINE_STYLES.stroke}
                    strokeWidth={TREND_LINE_STYLES.strokeWidth}
                    strokeDasharray={TREND_LINE_STYLES.strokeDasharray}
                    ifOverflow="extendDomain"
                  />
                )}

                {/* Reference lines for success thresholds */}
                <ReferenceLine
                  y={0.5}
                  stroke="hsl(var(--muted-foreground))"
                  strokeDasharray="3 3"
                  strokeOpacity={0.3}
                />

                <Scatter
                  name="Prompt Iterations"
                  data={chartData}
                  onClick={handlePointClick}
                  cursor="pointer"
                >
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={getPointColor(
                        entry,
                        successColor,
                        failureColor,
                        partialSuccessColor
                      )}
                      className="transition-opacity hover:opacity-80"
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </Suspense>
        </div>

        {/* Legend and summary */}
        <div className="mt-4 space-y-3">
          {showLegend && (
            <EvolutionChartLegend
              successColor={successColor}
              failureColor={failureColor}
              partialSuccessColor={partialSuccessColor}
              showTrendLine={trendLineVisible}
            />
          )}

          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center gap-4">
              <span>{chartData.length} iterations</span>
              <span>
                {chartData.filter((p) => p.isSuccess).length} successful (
                {(
                  (chartData.filter((p) => p.isSuccess).length /
                    chartData.length) *
                  100
                ).toFixed(1)}
                %)
              </span>
            </div>
            {uniqueTechniques.length > 1 && (
              <span className="text-xs">
                {uniqueTechniques.length} techniques | {uniqueProviders.length}{" "}
                providers
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * Simple prompt evolution chart without controls.
 */
export function SimplePromptEvolutionChart({
  events,
  dataPoints,
  height = 300,
  className,
  onPointClick,
}: {
  events?: TelemetryEventSummary[] | null;
  dataPoints?: PromptEvolutionDataPoint[] | null;
  height?: number;
  className?: string;
  onPointClick?: (point: PromptEvolutionDataPoint) => void;
}) {
  return (
    <PromptEvolutionChart
      events={events}
      dataPoints={dataPoints}
      height={height}
      showTrendLine={true}
      showCorrelation={false}
      showDownloadButton={false}
      showLegend={false}
      showFilterControls={false}
      className={className}
      onPointClick={onPointClick}
    />
  );
}

/**
 * Compact prompt evolution chart for dashboard widgets.
 */
export function CompactPromptEvolutionChart({
  events,
  dataPoints,
  className,
  onPointClick,
}: {
  events?: TelemetryEventSummary[] | null;
  dataPoints?: PromptEvolutionDataPoint[] | null;
  className?: string;
  onPointClick?: (point: PromptEvolutionDataPoint) => void;
}) {
  return (
    <PromptEvolutionChart
      events={events}
      dataPoints={dataPoints}
      title="Prompt Evolution"
      height={250}
      showTrendLine={true}
      showCorrelation={true}
      showDownloadButton={false}
      showLegend={false}
      showFilterControls={false}
      className={className}
      onPointClick={onPointClick}
    />
  );
}

/**
 * Full-featured prompt evolution chart with all controls.
 */
export function DetailedPromptEvolutionChart({
  events,
  dataPoints,
  title = "Prompt Evolution Analysis",
  height = 500,
  className,
  onPointClick,
  onExport,
}: {
  events?: TelemetryEventSummary[] | null;
  dataPoints?: PromptEvolutionDataPoint[] | null;
  title?: string;
  height?: number;
  className?: string;
  onPointClick?: (point: PromptEvolutionDataPoint) => void;
  onExport?: (format: "png" | "svg") => void;
}) {
  return (
    <PromptEvolutionChart
      events={events}
      dataPoints={dataPoints}
      title={title}
      description="Scatter plot showing the correlation between prompt iteration count and cumulative success rate. Click on points to view prompt details."
      height={height}
      showTrendLine={true}
      showCorrelation={true}
      showDownloadButton={true}
      showLegend={true}
      showFilterControls={true}
      sizeByTokens={true}
      maxSnippetLength={200}
      className={className}
      onPointClick={onPointClick}
      onExport={onExport}
    />
  );
}

// =============================================================================
// Export Types (Note: These types are already exported where they're defined)
// =============================================================================
