/**
 * LatencyDistributionChart Component
 *
 * Histogram/box plot showing latency distribution.
 * Displays p50, p90, p95, p99 markers with filtering by technique/provider.
 * Uses Recharts BarChart with responsive container and lazy loading.
 */

"use client";

import * as React from "react";
import { useMemo, useState, useCallback, useRef, Suspense } from "react";
import {
  ArrowUp,
  ArrowDown,
  Download,
  RefreshCw,
  AlertTriangle,
  Filter,
  X,
  Clock,
  BarChart3,
  Info,
  BoxSelect,
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import type {
  TelemetryEventSummary,
  CampaignStatistics,
  DistributionStats,
} from "@/types/campaign-analytics";

// Destructure Recharts components from lazy loader
const {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip: RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  ComposedChart,
  Line,
} = RechartsComponents;

// =============================================================================
// Types
// =============================================================================

/**
 * Display mode for the chart.
 */
export type ChartViewMode = "histogram" | "boxplot" | "combined";

/**
 * Single bin in the histogram.
 */
export interface HistogramBin {
  /** Start of bin range in ms */
  rangeStart: number;
  /** End of bin range in ms */
  rangeEnd: number;
  /** Label for display */
  label: string;
  /** Count of events in this bin */
  count: number;
  /** Percentage of total events */
  percentage: number;
  /** Color for the bin */
  color: string;
}

/**
 * Box plot statistics.
 */
export interface BoxPlotStats {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  mean: number;
  iqr: number;
  lowerWhisker: number;
  upperWhisker: number;
  outliers: number[];
}

/**
 * Latency data point for chart.
 */
export interface LatencyDataPoint {
  eventId: string;
  latencyMs: number;
  technique: string;
  provider: string;
  model: string;
  isSuccess: boolean;
  timestamp: string;
}

/**
 * Props for the LatencyDistributionChart component.
 */
export interface LatencyDistributionChartProps {
  /** Telemetry events to analyze */
  events?: TelemetryEventSummary[] | null;
  /** Pre-computed latency statistics from API */
  statistics?: CampaignStatistics | null;
  /** Pre-computed data points (alternative to events) */
  dataPoints?: LatencyDataPoint[] | null;
  /** Chart title */
  title?: string;
  /** Chart description */
  description?: string;
  /** Height of the chart in pixels */
  height?: number;
  /** Initial view mode */
  initialViewMode?: ChartViewMode;
  /** Number of histogram bins */
  binCount?: number;
  /** Show view mode toggle */
  showViewModeToggle?: boolean;
  /** Show filter controls */
  showFilterControls?: boolean;
  /** Show download button */
  showDownloadButton?: boolean;
  /** Show percentile markers */
  showPercentileMarkers?: boolean;
  /** Show statistics summary */
  showStatsSummary?: boolean;
  /** Percentile colors */
  percentileColors?: {
    p50: string;
    p90: string;
    p95: string;
    p99: string;
  };
  /** Callback when chart is exported */
  onExport?: (format: "png" | "svg") => void;
  /** Callback when a bin is clicked */
  onBinClick?: (bin: HistogramBin) => void;
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
 * Filter configuration.
 */
export interface LatencyFilter {
  techniques: string[];
  providers: string[];
  successOnly: boolean;
}

// =============================================================================
// Constants
// =============================================================================

/** Default percentile colors */
const DEFAULT_PERCENTILE_COLORS = {
  p50: "#22c55e", // green-500
  p90: "#eab308", // yellow-500
  p95: "#f97316", // orange-500
  p99: "#ef4444", // red-500
};

/** Histogram bar color gradient by range */
const LATENCY_COLORS = {
  fast: "#22c55e", // green for fast responses
  normal: "#3b82f6", // blue for normal
  slow: "#f97316", // orange for slow
  verySlow: "#ef4444", // red for very slow
};

/** View mode labels */
const VIEW_MODE_LABELS: Record<ChartViewMode, string> = {
  histogram: "Histogram",
  boxplot: "Box Plot",
  combined: "Combined",
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Convert telemetry events to latency data points.
 */
function eventsToDataPoints(
  events: TelemetryEventSummary[]
): LatencyDataPoint[] {
  return events.map((event) => ({
    eventId: event.id,
    latencyMs: event.total_latency_ms,
    technique: event.technique_suite,
    provider: event.provider,
    model: event.model,
    isSuccess: event.success_indicator,
    timestamp: event.created_at,
  }));
}

/**
 * Calculate percentile from sorted array.
 */
function calculatePercentile(sortedArray: number[], percentile: number): number {
  if (sortedArray.length === 0) return 0;
  const index = (percentile / 100) * (sortedArray.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) {
    return sortedArray[lower];
  }

  const fraction = index - lower;
  return sortedArray[lower] * (1 - fraction) + sortedArray[upper] * fraction;
}

/**
 * Calculate box plot statistics from latency values.
 */
function calculateBoxPlotStats(latencies: number[]): BoxPlotStats {
  if (latencies.length === 0) {
    return {
      min: 0,
      q1: 0,
      median: 0,
      q3: 0,
      max: 0,
      p50: 0,
      p90: 0,
      p95: 0,
      p99: 0,
      mean: 0,
      iqr: 0,
      lowerWhisker: 0,
      upperWhisker: 0,
      outliers: [],
    };
  }

  const sorted = [...latencies].sort((a, b) => a - b);
  const n = sorted.length;

  const min = sorted[0];
  const max = sorted[n - 1];
  const mean = sorted.reduce((sum, val) => sum + val, 0) / n;

  const q1 = calculatePercentile(sorted, 25);
  const median = calculatePercentile(sorted, 50);
  const q3 = calculatePercentile(sorted, 75);
  const p50 = median;
  const p90 = calculatePercentile(sorted, 90);
  const p95 = calculatePercentile(sorted, 95);
  const p99 = calculatePercentile(sorted, 99);

  const iqr = q3 - q1;
  const lowerWhisker = Math.max(min, q1 - 1.5 * iqr);
  const upperWhisker = Math.min(max, q3 + 1.5 * iqr);

  // Find outliers
  const outliers = sorted.filter(
    (val) => val < lowerWhisker || val > upperWhisker
  );

  return {
    min,
    q1,
    median,
    q3,
    max,
    p50,
    p90,
    p95,
    p99,
    mean,
    iqr,
    lowerWhisker,
    upperWhisker,
    outliers,
  };
}

/**
 * Create histogram bins from latency data.
 */
function createHistogramBins(
  latencies: number[],
  binCount: number,
  stats: BoxPlotStats
): HistogramBin[] {
  if (latencies.length === 0) return [];

  const min = stats.min;
  const max = stats.max;
  const range = max - min;
  const binWidth = range / binCount;

  // Create bins
  const bins: HistogramBin[] = [];
  for (let i = 0; i < binCount; i++) {
    const rangeStart = min + i * binWidth;
    const rangeEnd = i === binCount - 1 ? max + 0.001 : min + (i + 1) * binWidth;

    // Count values in this bin
    const count = latencies.filter(
      (val) => val >= rangeStart && val < rangeEnd
    ).length;

    // Determine color based on position relative to percentiles
    let color = LATENCY_COLORS.normal;
    const midpoint = (rangeStart + rangeEnd) / 2;
    if (midpoint <= stats.p50) {
      color = LATENCY_COLORS.fast;
    } else if (midpoint <= stats.p90) {
      color = LATENCY_COLORS.normal;
    } else if (midpoint <= stats.p95) {
      color = LATENCY_COLORS.slow;
    } else {
      color = LATENCY_COLORS.verySlow;
    }

    bins.push({
      rangeStart,
      rangeEnd,
      label: formatLatencyRange(rangeStart, rangeEnd),
      count,
      percentage: (count / latencies.length) * 100,
      color,
    });
  }

  return bins;
}

/**
 * Format latency value for display.
 */
function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Format latency range for bin label.
 */
function formatLatencyRange(start: number, end: number): string {
  return `${formatLatency(start)}-${formatLatency(end)}`;
}

/**
 * Get latency tier label.
 */
function getLatencyTier(ms: number, p50: number, p90: number, p95: number): string {
  if (ms <= p50) return "Fast";
  if (ms <= p90) return "Normal";
  if (ms <= p95) return "Slow";
  return "Very Slow";
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Loading skeleton for the chart.
 */
export function LatencyDistributionChartSkeleton({
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
          <Skeleton className="h-4 w-24" />
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Error state for the chart.
 */
export function LatencyDistributionChartError({
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
export function LatencyDistributionChartEmpty({
  message = "No latency data available for this campaign",
  className,
}: {
  message?: string;
  className?: string;
}) {
  return (
    <Card className={className}>
      <CardContent className="flex flex-col items-center justify-center py-12">
        <Clock className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <p className="text-lg font-medium text-muted-foreground mb-2">
          No Latency Data
        </p>
        <p className="text-sm text-muted-foreground text-center max-w-md">
          {message}
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Custom tooltip for histogram bars.
 */
interface HistogramTooltipProps {
  active?: boolean;
  payload?: Array<{
    value: number;
    name: string;
    payload: HistogramBin;
  }>;
  percentileColors: typeof DEFAULT_PERCENTILE_COLORS;
  stats: BoxPlotStats;
}

function CustomHistogramTooltip({
  active,
  payload,
  percentileColors,
  stats,
}: HistogramTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const bin = payload[0]?.payload;
  if (!bin) return null;

  const midpoint = (bin.rangeStart + bin.rangeEnd) / 2;
  const tier = getLatencyTier(midpoint, stats.p50, stats.p90, stats.p95);

  return (
    <Card className="p-3 shadow-lg border bg-popover min-w-[200px]">
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-4">
          <span className="font-semibold text-sm text-popover-foreground">
            {bin.label}
          </span>
          <Badge
            variant="secondary"
            className="text-xs"
            style={{
              backgroundColor: `${bin.color}20`,
              color: bin.color,
            }}
          >
            {tier}
          </Badge>
        </div>
        <div className="text-sm space-y-1">
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Count:</span>
            <span className="font-medium">{bin.count.toLocaleString()}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Percentage:</span>
            <span className="font-medium">{bin.percentage.toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </Card>
  );
}

/**
 * Percentile markers legend.
 */
function PercentileLegend({
  stats,
  colors,
  className,
}: {
  stats: BoxPlotStats;
  colors: typeof DEFAULT_PERCENTILE_COLORS;
  className?: string;
}) {
  const markers = [
    { label: "P50 (Median)", value: stats.p50, color: colors.p50 },
    { label: "P90", value: stats.p90, color: colors.p90 },
    { label: "P95", value: stats.p95, color: colors.p95 },
    { label: "P99", value: stats.p99, color: colors.p99 },
  ];

  return (
    <div className={cn("flex flex-wrap items-center justify-center gap-4", className)}>
      {markers.map((marker) => (
        <div key={marker.label} className="flex items-center gap-1.5">
          <div
            className="w-0.5 h-4 rounded"
            style={{ backgroundColor: marker.color }}
          />
          <span className="text-xs text-muted-foreground">
            {marker.label}: {formatLatency(marker.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * Statistics summary panel.
 */
function StatsSummary({
  stats,
  totalCount,
  className,
}: {
  stats: BoxPlotStats;
  totalCount: number;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm",
        className
      )}
    >
      <div className="space-y-1">
        <span className="text-muted-foreground">Mean</span>
        <p className="font-medium">{formatLatency(stats.mean)}</p>
      </div>
      <div className="space-y-1">
        <span className="text-muted-foreground">Median (P50)</span>
        <p className="font-medium">{formatLatency(stats.median)}</p>
      </div>
      <div className="space-y-1">
        <span className="text-muted-foreground">Min / Max</span>
        <p className="font-medium">
          {formatLatency(stats.min)} / {formatLatency(stats.max)}
        </p>
      </div>
      <div className="space-y-1">
        <span className="text-muted-foreground">Total Events</span>
        <p className="font-medium">{totalCount.toLocaleString()}</p>
      </div>
    </div>
  );
}

/**
 * Filter chips display.
 */
function FilterChips({
  filter,
  onRemoveTechnique,
  onRemoveProvider,
  onClearAll,
}: {
  filter: LatencyFilter;
  onRemoveTechnique: (technique: string) => void;
  onRemoveProvider: (provider: string) => void;
  onClearAll: () => void;
}) {
  const hasFilters =
    filter.techniques.length > 0 ||
    filter.providers.length > 0 ||
    filter.successOnly;

  if (!hasFilters) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 mt-2">
      <span className="text-xs text-muted-foreground">Filters:</span>
      {filter.techniques.map((technique) => (
        <Badge key={technique} variant="secondary" className="text-xs gap-1 pr-1">
          {technique}
          <button
            onClick={() => onRemoveTechnique(technique)}
            className="ml-1 hover:bg-muted rounded-full p-0.5"
            aria-label={`Remove ${technique} filter`}
          >
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
      {filter.providers.map((provider) => (
        <Badge key={provider} variant="outline" className="text-xs gap-1 pr-1">
          {provider}
          <button
            onClick={() => onRemoveProvider(provider)}
            className="ml-1 hover:bg-muted rounded-full p-0.5"
            aria-label={`Remove ${provider} filter`}
          >
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
      {filter.successOnly && (
        <Badge variant="secondary" className="text-xs">
          Success Only
        </Badge>
      )}
      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-xs"
        onClick={onClearAll}
      >
        Clear all
      </Button>
    </div>
  );
}

/**
 * Box plot visualization (simplified as reference lines in the histogram).
 */
function BoxPlotOverlay({
  stats,
  height,
  percentileColors,
}: {
  stats: BoxPlotStats;
  height: number;
  percentileColors: typeof DEFAULT_PERCENTILE_COLORS;
}) {
  // This component renders reference lines for percentiles
  // The actual box plot shape is rendered via the histogram background
  return (
    <>
      <ReferenceLine
        x={stats.p50}
        stroke={percentileColors.p50}
        strokeWidth={2}
        strokeDasharray="none"
        label={{
          value: "P50",
          position: "top",
          fill: percentileColors.p50,
          fontSize: 10,
        }}
      />
      <ReferenceLine
        x={stats.p90}
        stroke={percentileColors.p90}
        strokeWidth={2}
        strokeDasharray="4 4"
        label={{
          value: "P90",
          position: "top",
          fill: percentileColors.p90,
          fontSize: 10,
        }}
      />
      <ReferenceLine
        x={stats.p95}
        stroke={percentileColors.p95}
        strokeWidth={2}
        strokeDasharray="4 4"
        label={{
          value: "P95",
          position: "top",
          fill: percentileColors.p95,
          fontSize: 10,
        }}
      />
      <ReferenceLine
        x={stats.p99}
        stroke={percentileColors.p99}
        strokeWidth={2}
        strokeDasharray="4 4"
        label={{
          value: "P99",
          position: "top",
          fill: percentileColors.p99,
          fontSize: 10,
        }}
      />
    </>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * Latency Distribution Chart Component
 *
 * Displays a histogram/box plot showing latency distribution.
 * Features:
 * - Histogram view with configurable bins
 * - Box plot statistics with percentile markers
 * - P50, P90, P95, P99 reference lines
 * - Filtering by technique and provider
 * - Export support
 */
export function LatencyDistributionChart({
  events,
  statistics,
  dataPoints: providedDataPoints,
  title = "Latency Distribution",
  description,
  height = 400,
  initialViewMode = "histogram",
  binCount = 20,
  showViewModeToggle = true,
  showFilterControls = true,
  showDownloadButton = true,
  showPercentileMarkers = true,
  showStatsSummary = true,
  percentileColors = DEFAULT_PERCENTILE_COLORS,
  onExport,
  onBinClick,
  isLoading = false,
  error = null,
  onRetry,
  className,
  chartRef: externalChartRef,
}: LatencyDistributionChartProps) {
  // Internal chart ref if not provided
  const internalChartRef = useRef<HTMLDivElement>(null);
  const chartContainerRef = externalChartRef || internalChartRef;

  // State
  const [viewMode, setViewMode] = useState<ChartViewMode>(initialViewMode);
  const [filter, setFilter] = useState<LatencyFilter>({
    techniques: [],
    providers: [],
    successOnly: false,
  });

  // Convert events to data points if needed
  const rawDataPoints = useMemo((): LatencyDataPoint[] => {
    if (providedDataPoints && providedDataPoints.length > 0) {
      return providedDataPoints;
    }
    if (events && events.length > 0) {
      return eventsToDataPoints(events);
    }
    return [];
  }, [events, providedDataPoints]);

  // Get unique techniques and providers for filtering
  const uniqueTechniques = useMemo(() => {
    return [...new Set(rawDataPoints.map((p) => p.technique))];
  }, [rawDataPoints]);

  const uniqueProviders = useMemo(() => {
    return [...new Set(rawDataPoints.map((p) => p.provider))];
  }, [rawDataPoints]);

  // Apply filters to data
  const filteredDataPoints = useMemo(() => {
    let filtered = rawDataPoints;

    if (filter.techniques.length > 0) {
      filtered = filtered.filter((p) => filter.techniques.includes(p.technique));
    }
    if (filter.providers.length > 0) {
      filtered = filtered.filter((p) => filter.providers.includes(p.provider));
    }
    if (filter.successOnly) {
      filtered = filtered.filter((p) => p.isSuccess);
    }

    return filtered;
  }, [rawDataPoints, filter]);

  // Extract latency values
  const latencies = useMemo(() => {
    return filteredDataPoints.map((p) => p.latencyMs);
  }, [filteredDataPoints]);

  // Calculate box plot statistics
  const stats = useMemo(() => {
    // If we have API statistics and no custom filters, use them
    if (
      statistics?.latency_ms &&
      filter.techniques.length === 0 &&
      filter.providers.length === 0 &&
      !filter.successOnly
    ) {
      const apiStats = statistics.latency_ms;
      return {
        min: apiStats.min_value ?? 0,
        max: apiStats.max_value ?? 0,
        mean: apiStats.mean ?? 0,
        median: apiStats.median ?? 0,
        q1: apiStats.percentiles?.p50 ? (apiStats.percentiles.p50 * 0.5) : 0, // Approximate
        q3: apiStats.percentiles?.p90 ? ((apiStats.percentiles.p90 + (apiStats.median ?? 0)) / 2) : 0, // Approximate
        p50: apiStats.percentiles?.p50 ?? apiStats.median ?? 0,
        p90: apiStats.percentiles?.p90 ?? 0,
        p95: apiStats.percentiles?.p95 ?? 0,
        p99: apiStats.percentiles?.p99 ?? 0,
        iqr: 0,
        lowerWhisker: apiStats.min_value ?? 0,
        upperWhisker: apiStats.max_value ?? 0,
        outliers: [],
      };
    }
    // Otherwise, calculate from raw data
    return calculateBoxPlotStats(latencies);
  }, [latencies, statistics, filter]);

  // Create histogram bins
  const histogramBins = useMemo(() => {
    return createHistogramBins(latencies, binCount, stats);
  }, [latencies, binCount, stats]);

  // Handlers
  const handleViewModeChange = useCallback((value: string) => {
    setViewMode(value as ChartViewMode);
  }, []);

  const handleTechniqueToggle = useCallback((technique: string) => {
    setFilter((prev) => ({
      ...prev,
      techniques: prev.techniques.includes(technique)
        ? prev.techniques.filter((t) => t !== technique)
        : [...prev.techniques, technique],
    }));
  }, []);

  const handleProviderToggle = useCallback((provider: string) => {
    setFilter((prev) => ({
      ...prev,
      providers: prev.providers.includes(provider)
        ? prev.providers.filter((p) => p !== provider)
        : [...prev.providers, provider],
    }));
  }, []);

  const handleRemoveTechnique = useCallback((technique: string) => {
    setFilter((prev) => ({
      ...prev,
      techniques: prev.techniques.filter((t) => t !== technique),
    }));
  }, []);

  const handleRemoveProvider = useCallback((provider: string) => {
    setFilter((prev) => ({
      ...prev,
      providers: prev.providers.filter((p) => p !== provider),
    }));
  }, []);

  const handleClearFilters = useCallback(() => {
    setFilter({
      techniques: [],
      providers: [],
      successOnly: false,
    });
  }, []);

  const toggleSuccessOnly = useCallback(() => {
    setFilter((prev) => ({
      ...prev,
      successOnly: !prev.successOnly,
    }));
  }, []);

  const handleBarClick = useCallback(
    (data: any) => {
      if (data?.payload && onBinClick) {
        onBinClick(data.payload as HistogramBin);
      }
    },
    [onBinClick]
  );

  const handleExport = useCallback(
    (format: "png" | "svg") => {
      onExport?.(format);
    },
    [onExport]
  );

  // Render loading state
  if (isLoading) {
    return <LatencyDistributionChartSkeleton height={height} className={className} />;
  }

  // Render error state
  if (error) {
    return (
      <LatencyDistributionChartError
        error={error}
        onRetry={onRetry}
        className={className}
      />
    );
  }

  // Render empty state
  if (latencies.length === 0 && !statistics) {
    return <LatencyDistributionChartEmpty className={className} />;
  }

  const hasActiveFilters =
    filter.techniques.length > 0 ||
    filter.providers.length > 0 ||
    filter.successOnly;

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <CardTitle className="text-base font-medium">{title}</CardTitle>
            <Badge variant="secondary" className="gap-1">
              <Clock className="h-3 w-3" />
              {filteredDataPoints.length.toLocaleString()} events
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            {/* View mode toggle */}
            {showViewModeToggle && (
              <Select value={viewMode} onValueChange={handleViewModeChange}>
                <SelectTrigger className="w-28 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(VIEW_MODE_LABELS).map(([mode, label]) => (
                    <SelectItem key={mode} value={mode}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            {/* Success only toggle */}
            {showFilterControls && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="success-only"
                        checked={filter.successOnly}
                        onCheckedChange={toggleSuccessOnly}
                        className="h-5"
                      />
                      <Label
                        htmlFor="success-only"
                        className="text-xs text-muted-foreground cursor-pointer hidden sm:inline"
                      >
                        Success
                      </Label>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Show only successful requests</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}

            {/* Technique filter */}
            {showFilterControls && uniqueTechniques.length > 1 && (
              <DropdownMenu>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant={
                            filter.techniques.length > 0 ? "secondary" : "ghost"
                          }
                          size="icon"
                          className="h-8 w-8 relative"
                          aria-label="Filter techniques"
                        >
                          <Filter className="h-4 w-4" />
                          {filter.techniques.length > 0 && (
                            <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-primary text-primary-foreground text-[10px] flex items-center justify-center">
                              {filter.techniques.length}
                            </span>
                          )}
                        </Button>
                      </DropdownMenuTrigger>
                    </TooltipTrigger>
                    <TooltipContent>Filter Techniques</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <DropdownMenuContent align="end" className="w-56 max-h-80 overflow-y-auto">
                  <DropdownMenuLabel>Filter by Technique</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {uniqueTechniques.map((technique) => (
                    <DropdownMenuCheckboxItem
                      key={technique}
                      checked={filter.techniques.includes(technique)}
                      onCheckedChange={() => handleTechniqueToggle(technique)}
                    >
                      {technique}
                    </DropdownMenuCheckboxItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            )}

            {/* Provider filter */}
            {showFilterControls && uniqueProviders.length > 1 && (
              <DropdownMenu>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant={
                            filter.providers.length > 0 ? "secondary" : "ghost"
                          }
                          size="icon"
                          className="h-8 w-8 relative"
                          aria-label="Filter providers"
                        >
                          <BarChart3 className="h-4 w-4" />
                          {filter.providers.length > 0 && (
                            <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-primary text-primary-foreground text-[10px] flex items-center justify-center">
                              {filter.providers.length}
                            </span>
                          )}
                        </Button>
                      </DropdownMenuTrigger>
                    </TooltipTrigger>
                    <TooltipContent>Filter Providers</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <DropdownMenuContent align="end" className="w-56 max-h-80 overflow-y-auto">
                  <DropdownMenuLabel>Filter by Provider</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {uniqueProviders.map((provider) => (
                    <DropdownMenuCheckboxItem
                      key={provider}
                      checked={filter.providers.includes(provider)}
                      onCheckedChange={() => handleProviderToggle(provider)}
                    >
                      {provider}
                    </DropdownMenuCheckboxItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
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

        {/* Filter chips */}
        <FilterChips
          filter={filter}
          onRemoveTechnique={handleRemoveTechnique}
          onRemoveProvider={handleRemoveProvider}
          onClearAll={handleClearFilters}
        />
      </CardHeader>

      <CardContent className="pt-0">
        <div ref={chartContainerRef}>
          <Suspense
            fallback={
              <Skeleton className="w-full" style={{ height: `${height}px` }} />
            }
          >
            <ResponsiveContainer width="100%" height={height}>
              <BarChart
                data={histogramBins}
                margin={{ top: 30, right: 30, left: 20, bottom: 20 }}
                onClick={handleBarClick}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  className="stroke-muted"
                  opacity={0.5}
                  vertical={false}
                />
                <XAxis
                  dataKey="label"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={(value) => value.toLocaleString()}
                  width={50}
                  label={{
                    value: "Count",
                    angle: -90,
                    position: "insideLeft",
                    style: { fontSize: 12, fill: "hsl(var(--muted-foreground))" },
                  }}
                />
                <RechartsTooltip
                  content={
                    <CustomHistogramTooltip
                      percentileColors={percentileColors}
                      stats={stats}
                    />
                  }
                  cursor={{ fill: "hsl(var(--muted))", opacity: 0.3 }}
                />

                {/* Percentile reference lines */}
                {showPercentileMarkers && viewMode !== "histogram" && (
                  <>
                    <ReferenceLine
                      x={stats.p50}
                      stroke={percentileColors.p50}
                      strokeWidth={2}
                      label={{ value: "P50", position: "top", fill: percentileColors.p50, fontSize: 10 }}
                    />
                    <ReferenceLine
                      x={stats.p90}
                      stroke={percentileColors.p90}
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      label={{ value: "P90", position: "top", fill: percentileColors.p90, fontSize: 10 }}
                    />
                    <ReferenceLine
                      x={stats.p95}
                      stroke={percentileColors.p95}
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      label={{ value: "P95", position: "top", fill: percentileColors.p95, fontSize: 10 }}
                    />
                    <ReferenceLine
                      x={stats.p99}
                      stroke={percentileColors.p99}
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      label={{ value: "P99", position: "top", fill: percentileColors.p99, fontSize: 10 }}
                    />
                  </>
                )}

                <Bar
                  dataKey="count"
                  name="Count"
                  radius={[4, 4, 0, 0]}
                  cursor="pointer"
                  maxBarSize={60}
                >
                  {histogramBins.map((bin, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={bin.color}
                      className="transition-opacity hover:opacity-80"
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Suspense>
        </div>

        {/* Percentile legend */}
        {showPercentileMarkers && (
          <PercentileLegend
            stats={stats}
            colors={percentileColors}
            className="mt-4"
          />
        )}

        {/* Stats summary */}
        {showStatsSummary && (
          <StatsSummary
            stats={stats}
            totalCount={filteredDataPoints.length}
            className="mt-4 pt-4 border-t"
          />
        )}

        {/* Footer info */}
        <div className="flex items-center justify-between text-sm text-muted-foreground mt-4">
          <div className="flex items-center gap-4">
            <span>{histogramBins.length} bins</span>
            {hasActiveFilters && (
              <span className="text-xs">
                Filtered from {rawDataPoints.length.toLocaleString()} total
              </span>
            )}
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1 cursor-help">
                  <Info className="h-3 w-3" />
                  <span className="text-xs">Latency Analysis</span>
                </div>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                <p>
                  Distribution of response latencies across all telemetry events.
                  Percentile markers show P50 (median), P90, P95, and P99 values.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * Simple latency chart without controls.
 */
export function SimpleLatencyChart({
  events,
  statistics,
  dataPoints,
  height = 300,
  className,
  onBinClick,
}: {
  events?: TelemetryEventSummary[] | null;
  statistics?: CampaignStatistics | null;
  dataPoints?: LatencyDataPoint[] | null;
  height?: number;
  className?: string;
  onBinClick?: (bin: HistogramBin) => void;
}) {
  return (
    <LatencyDistributionChart
      events={events}
      statistics={statistics}
      dataPoints={dataPoints}
      height={height}
      showViewModeToggle={false}
      showFilterControls={false}
      showDownloadButton={false}
      showStatsSummary={false}
      binCount={15}
      className={className}
      onBinClick={onBinClick}
    />
  );
}

/**
 * Compact latency chart for dashboard widgets.
 */
export function CompactLatencyChart({
  events,
  statistics,
  dataPoints,
  className,
  onBinClick,
}: {
  events?: TelemetryEventSummary[] | null;
  statistics?: CampaignStatistics | null;
  dataPoints?: LatencyDataPoint[] | null;
  className?: string;
  onBinClick?: (bin: HistogramBin) => void;
}) {
  return (
    <LatencyDistributionChart
      events={events}
      statistics={statistics}
      dataPoints={dataPoints}
      title="Latency"
      height={250}
      binCount={12}
      showViewModeToggle={false}
      showFilterControls={false}
      showDownloadButton={false}
      showStatsSummary={false}
      className={className}
      onBinClick={onBinClick}
    />
  );
}

/**
 * Detailed latency chart with all controls.
 */
export function DetailedLatencyChart({
  events,
  statistics,
  dataPoints,
  title = "Latency Distribution Analysis",
  height = 500,
  className,
  onBinClick,
  onExport,
}: {
  events?: TelemetryEventSummary[] | null;
  statistics?: CampaignStatistics | null;
  dataPoints?: LatencyDataPoint[] | null;
  title?: string;
  height?: number;
  className?: string;
  onBinClick?: (bin: HistogramBin) => void;
  onExport?: (format: "png" | "svg") => void;
}) {
  return (
    <LatencyDistributionChart
      events={events}
      statistics={statistics}
      dataPoints={dataPoints}
      title={title}
      description="Histogram showing the distribution of response latencies. Use filters to analyze by technique or provider."
      height={height}
      binCount={25}
      showViewModeToggle={true}
      showFilterControls={true}
      showDownloadButton={true}
      showPercentileMarkers={true}
      showStatsSummary={true}
      className={className}
      onBinClick={onBinClick}
      onExport={onExport}
    />
  );
}

/**
 * Latency chart focused on percentiles only.
 */
export function PercentileLatencyChart({
  events,
  statistics,
  dataPoints,
  height = 350,
  className,
}: {
  events?: TelemetryEventSummary[] | null;
  statistics?: CampaignStatistics | null;
  dataPoints?: LatencyDataPoint[] | null;
  height?: number;
  className?: string;
}) {
  return (
    <LatencyDistributionChart
      events={events}
      statistics={statistics}
      dataPoints={dataPoints}
      title="Latency Percentiles"
      height={height}
      initialViewMode="boxplot"
      binCount={15}
      showViewModeToggle={false}
      showFilterControls={false}
      showDownloadButton={false}
      showPercentileMarkers={true}
      showStatsSummary={true}
      className={className}
    />
  );
}

// =============================================================================
// Export Types
// =============================================================================

export type {
  HistogramBin,
  BoxPlotStats,
  LatencyDataPoint,
  LatencyDistributionChartProps,
  LatencyFilter,
};
