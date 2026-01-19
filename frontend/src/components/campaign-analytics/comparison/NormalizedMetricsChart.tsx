/**
 * NormalizedMetricsChart Component
 *
 * Radar/spider chart showing normalized (0-1) metrics for each campaign.
 * Overlays multiple campaigns for visual comparison with color-coded series.
 * Uses Recharts RadarChart with responsive container and lazy loading.
 */

"use client";

import * as React from "react";
import { useMemo, useState, useCallback, useRef, Suspense } from "react";
import {
  Download,
  RefreshCw,
  AlertTriangle,
  Info,
  Eye,
  EyeOff,
  Radar as RadarIcon,
} from "lucide-react";
import { RechartsComponents } from "@/lib/components/lazy-components";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type {
  CampaignComparisonItem,
  CampaignComparison,
} from "@/types/campaign-analytics";

// Destructure Recharts components from lazy loader
const {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip: RechartsTooltip,
} = RechartsComponents;

// =============================================================================
// Types
// =============================================================================

/**
 * Normalized metric definition for radar chart.
 */
export interface NormalizedMetric {
  /** Unique identifier */
  id: string;
  /** Display label on the chart */
  label: string;
  /** Short label for compact display */
  shortLabel?: string;
  /** Description shown in tooltip */
  description?: string;
  /** Whether higher values are better (for color coding) */
  higherIsBetter: boolean;
  /** Get normalized value (0-1) from campaign item */
  getValue: (item: CampaignComparisonItem) => number | null;
}

/**
 * Data point structure for radar chart.
 */
export interface RadarDataPoint {
  /** Metric label for the axis */
  metric: string;
  /** Metric ID */
  metricId: string;
  /** Full metric label */
  fullLabel: string;
  /** Description */
  description?: string;
  /** Value for each campaign (keyed by campaign ID) */
  [campaignId: string]: number | string | undefined;
}

/**
 * Campaign series configuration.
 */
export interface CampaignSeries {
  id: string;
  name: string;
  color: string;
  visible: boolean;
}

/**
 * Props for the NormalizedMetricsChart component.
 */
export interface NormalizedMetricsChartProps {
  /** Comparison data from API */
  comparison: CampaignComparison | null | undefined;
  /** Chart title */
  title?: string;
  /** Chart description */
  description?: string;
  /** Height of the chart in pixels */
  height?: number;
  /** Show legend */
  showLegend?: boolean;
  /** Show metric visibility controls */
  showMetricControls?: boolean;
  /** Show download button */
  showDownloadButton?: boolean;
  /** Show tooltips on hover */
  showTooltips?: boolean;
  /** Custom metric configurations (overrides defaults) */
  metricConfigs?: NormalizedMetric[];
  /** Custom campaign colors */
  campaignColors?: Record<string, string>;
  /** Fill opacity for radar areas (0-1) */
  fillOpacity?: number;
  /** Stroke width for radar lines */
  strokeWidth?: number;
  /** Animation duration in ms (0 to disable) */
  animationDuration?: number;
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
  chartRef?: React.RefObject<HTMLDivElement | null>;
}

// =============================================================================
// Constants
// =============================================================================

/**
 * Default color palette for campaigns.
 */
const CAMPAIGN_COLORS = [
  "#3b82f6", // Blue
  "#22c55e", // Green
  "#f59e0b", // Amber
  "#ef4444", // Red
  "#8b5cf6", // Purple
  "#06b6d4", // Cyan
  "#ec4899", // Pink
  "#84cc16", // Lime
];

/**
 * Default normalized metrics for comparison.
 */
const DEFAULT_NORMALIZED_METRICS: NormalizedMetric[] = [
  {
    id: "success_rate",
    label: "Success Rate",
    shortLabel: "Success",
    description: "Overall success rate normalized to 0-1 scale",
    higherIsBetter: true,
    getValue: (item) => item.normalized_success_rate ?? item.success_rate ?? null,
  },
  {
    id: "latency",
    label: "Latency Score",
    shortLabel: "Latency",
    description: "Latency efficiency (1 = fastest, 0 = slowest)",
    higherIsBetter: true,
    getValue: (item) => {
      // Invert latency so higher is better (lower latency = higher score)
      const normalized = item.normalized_latency;
      if (normalized === null || normalized === undefined) return null;
      return 1 - normalized;
    },
  },
  {
    id: "cost_efficiency",
    label: "Cost Efficiency",
    shortLabel: "Cost",
    description: "Cost efficiency (1 = most efficient, 0 = least efficient)",
    higherIsBetter: true,
    getValue: (item) => {
      // Invert cost so higher is better (lower cost = higher score)
      const normalized = item.normalized_cost;
      if (normalized === null || normalized === undefined) return null;
      return 1 - normalized;
    },
  },
  {
    id: "effectiveness",
    label: "Effectiveness",
    shortLabel: "Effect.",
    description: "Overall effectiveness score",
    higherIsBetter: true,
    getValue: (item) => item.normalized_effectiveness ?? null,
  },
  {
    id: "semantic_success",
    label: "Semantic Success",
    shortLabel: "Semantic",
    description: "Semantic success rate (how well the output matches intent)",
    higherIsBetter: true,
    getValue: (item) => item.semantic_success_mean ?? null,
  },
  {
    id: "throughput",
    label: "Throughput",
    shortLabel: "Thruput",
    description: "Normalized attempts per time unit",
    higherIsBetter: true,
    getValue: (item) => {
      // Calculate throughput as attempts/duration, then normalize
      if (!item.duration_seconds || item.duration_seconds <= 0) return null;
      const rate = item.total_attempts / item.duration_seconds;
      // Cap at reasonable maximum and normalize
      return Math.min(rate / 10, 1); // Assuming 10/s is a good baseline
    },
  },
];

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get campaign color by index.
 */
function getCampaignColor(index: number, customColors?: Record<string, string>, campaignId?: string): string {
  if (customColors && campaignId && customColors[campaignId]) {
    return customColors[campaignId];
  }
  return CAMPAIGN_COLORS[index % CAMPAIGN_COLORS.length];
}

/**
 * Format percentage value for display.
 */
function formatPercentage(value: number | null | undefined): string {
  if (value === null || value === undefined) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Truncate label for chart display.
 */
function truncateLabel(label: string, maxLength: number = 12): string {
  if (label.length <= maxLength) return label;
  return label.slice(0, maxLength - 2) + "…";
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Loading skeleton for the chart.
 */
export function NormalizedMetricsChartSkeleton({
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
            <Skeleton className="h-8 w-8" />
            <Skeleton className="h-8 w-8" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center" style={{ height: `${height}px` }}>
          <div className="relative">
            {/* Hexagon skeleton shape */}
            <Skeleton className="h-64 w-64 rounded-full" />
            <div className="absolute inset-0 flex items-center justify-center">
              <Skeleton className="h-48 w-48 rounded-full opacity-50" />
              <div className="absolute">
                <Skeleton className="h-32 w-32 rounded-full opacity-30" />
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4 flex items-center justify-center gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="flex items-center gap-2">
              <Skeleton className="h-3 w-3 rounded-full" />
              <Skeleton className="h-4 w-20" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Error state for the chart.
 */
export function NormalizedMetricsChartError({
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
export function NormalizedMetricsChartEmpty({
  message = "Select at least 2 campaigns to compare normalized metrics",
  className,
}: {
  message?: string;
  className?: string;
}) {
  return (
    <Card className={className}>
      <CardContent className="flex flex-col items-center justify-center py-12">
        <RadarIcon className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <p className="text-lg font-medium text-muted-foreground mb-2">
          No Comparison Data
        </p>
        <p className="text-sm text-muted-foreground text-center max-w-md">
          {message}
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Custom tooltip component for radar chart.
 */
interface RadarTooltipProps {
  active?: boolean;
  payload?: Array<{
    value: number;
    name: string;
    dataKey: string;
    color: string;
    payload: RadarDataPoint;
  }>;
  label?: string;
  metrics: NormalizedMetric[];
}

function CustomRadarTooltip({ active, payload, metrics }: RadarTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const data = payload[0]?.payload;
  if (!data) return null;

  // Find metric config
  const metricConfig = metrics.find((m) => m.id === data.metricId);

  return (
    <Card className="p-3 shadow-lg border bg-popover min-w-[220px]">
      <p className="font-semibold text-sm mb-2 text-popover-foreground">
        {data.fullLabel}
      </p>
      {metricConfig?.description && (
        <p className="text-xs text-muted-foreground mb-3">
          {metricConfig.description}
        </p>
      )}
      <div className="space-y-1.5">
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-sm text-muted-foreground truncate max-w-[120px]">
                {entry.name}
              </span>
            </div>
            <span className="font-medium text-sm tabular-nums">
              {formatPercentage(entry.value)}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

/**
 * Campaign visibility toggle in legend.
 */
function CampaignLegendItem({
  series,
  onToggle,
}: {
  series: CampaignSeries;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className={cn(
        "flex items-center gap-2 px-2 py-1 rounded-md transition-colors",
        "hover:bg-muted/50 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        !series.visible && "opacity-50"
      )}
      aria-pressed={series.visible}
      aria-label={`${series.visible ? "Hide" : "Show"} ${series.name}`}
    >
      <div
        className={cn(
          "w-3 h-3 rounded-full border-2 transition-colors",
          series.visible ? "bg-current" : "bg-transparent"
        )}
        style={{
          borderColor: series.color,
          backgroundColor: series.visible ? series.color : "transparent",
        }}
      />
      <span className="text-sm truncate max-w-[100px]">{series.name}</span>
      {series.visible ? (
        <Eye className="h-3 w-3 text-muted-foreground" />
      ) : (
        <EyeOff className="h-3 w-3 text-muted-foreground" />
      )}
    </button>
  );
}

/**
 * Metric visibility controls popover.
 */
function MetricControlsPopover({
  metrics,
  visibleMetrics,
  onToggleMetric,
}: {
  metrics: NormalizedMetric[];
  visibleMetrics: Set<string>;
  onToggleMetric: (metricId: string) => void;
}) {
  return (
    <Popover>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <PopoverTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                aria-label="Configure visible metrics"
              >
                <Info className="h-4 w-4" />
              </Button>
            </PopoverTrigger>
          </TooltipTrigger>
          <TooltipContent>Configure Metrics</TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <PopoverContent align="end" className="w-56">
        <div className="space-y-2">
          <p className="text-sm font-medium mb-2">Visible Metrics</p>
          {metrics.map((metric) => (
            <div key={metric.id} className="flex items-center gap-2">
              <Checkbox
                id={`metric-${metric.id}`}
                checked={visibleMetrics.has(metric.id)}
                onCheckedChange={() => onToggleMetric(metric.id)}
              />
              <label
                htmlFor={`metric-${metric.id}`}
                className="text-sm cursor-pointer flex-1"
              >
                {metric.label}
              </label>
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * NormalizedMetricsChart Component
 *
 * Radar/spider chart for comparing normalized metrics across multiple campaigns.
 * Features:
 * - Overlay multiple campaigns (up to 4)
 * - Toggle campaign visibility
 * - Configure which metrics to display
 * - Interactive tooltips
 * - Export functionality
 *
 * @example
 * ```tsx
 * <NormalizedMetricsChart
 *   comparison={comparisonData}
 *   showLegend
 *   showMetricControls
 * />
 * ```
 */
export function NormalizedMetricsChart({
  comparison,
  title = "Normalized Campaign Metrics",
  description,
  height = 400,
  showLegend = true,
  showMetricControls = true,
  showDownloadButton = true,
  showTooltips = true,
  metricConfigs = DEFAULT_NORMALIZED_METRICS,
  campaignColors,
  fillOpacity = 0.25,
  strokeWidth = 2,
  animationDuration = 300,
  onExport,
  isLoading = false,
  error = null,
  onRetry,
  className,
  chartRef: externalChartRef,
}: NormalizedMetricsChartProps) {
  // Internal chart ref if not provided
  const internalChartRef = useRef<HTMLDivElement>(null);
  const chartContainerRef = externalChartRef || internalChartRef;

  // Campaign visibility state
  const [campaignSeriesState, setCampaignSeriesState] = useState<CampaignSeries[]>([]);

  // Metric visibility state
  const [visibleMetrics, setVisibleMetrics] = useState<Set<string>>(
    () => new Set(metricConfigs.map((m) => m.id))
  );

  // Initialize campaign series when comparison changes
  React.useEffect(() => {
    if (comparison?.campaigns) {
      setCampaignSeriesState(
        comparison.campaigns.map((campaign, index) => ({
          id: campaign.campaign_id,
          name: campaign.campaign_name,
          color: getCampaignColor(index, campaignColors, campaign.campaign_id),
          visible: true,
        }))
      );
    }
  }, [comparison, campaignColors]);

  // Prepare radar chart data
  const chartData = useMemo((): RadarDataPoint[] => {
    if (!comparison?.campaigns?.length) return [];

    const campaigns = comparison.campaigns;
    const visibleCampaigns = campaignSeriesState.filter((s) => s.visible);

    return metricConfigs
      .filter((metric) => visibleMetrics.has(metric.id))
      .map((metric) => {
        const dataPoint: RadarDataPoint = {
          metric: metric.shortLabel || truncateLabel(metric.label),
          metricId: metric.id,
          fullLabel: metric.label,
          description: metric.description,
        };

        campaigns.forEach((campaign) => {
          // Only include visible campaigns
          const isVisible = visibleCampaigns.some((s) => s.id === campaign.campaign_id);
          if (isVisible) {
            const value = metric.getValue(campaign);
            // Ensure value is between 0 and 1
            dataPoint[campaign.campaign_id] = value !== null
              ? Math.max(0, Math.min(1, value))
              : 0;
          }
        });

        return dataPoint;
      });
  }, [comparison, campaignSeriesState, metricConfigs, visibleMetrics]);

  // Visible campaign series
  const visibleSeries = useMemo(
    () => campaignSeriesState.filter((s) => s.visible),
    [campaignSeriesState]
  );

  // Handlers
  const handleToggleCampaign = useCallback((campaignId: string) => {
    setCampaignSeriesState((prev) =>
      prev.map((s) =>
        s.id === campaignId ? { ...s, visible: !s.visible } : s
      )
    );
  }, []);

  const handleToggleMetric = useCallback((metricId: string) => {
    setVisibleMetrics((prev) => {
      const next = new Set(prev);
      if (next.has(metricId)) {
        // Don't allow hiding all metrics
        if (next.size > 3) {
          next.delete(metricId);
        }
      } else {
        next.add(metricId);
      }
      return next;
    });
  }, []);

  const handleExport = useCallback(
    (format: "png" | "svg") => {
      onExport?.(format);
    },
    [onExport]
  );

  // Render loading state
  if (isLoading) {
    return <NormalizedMetricsChartSkeleton height={height} className={className} />;
  }

  // Render error state
  if (error) {
    return (
      <NormalizedMetricsChartError
        error={error}
        onRetry={onRetry}
        className={className}
      />
    );
  }

  // Render empty state
  if (!comparison || comparison.campaigns.length < 2 || chartData.length === 0) {
    return <NormalizedMetricsChartEmpty className={className} />;
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <CardTitle className="text-base font-medium">{title}</CardTitle>
            <Badge variant="secondary" className="text-xs">
              {visibleSeries.length} campaigns
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            {/* Metric controls */}
            {showMetricControls && (
              <MetricControlsPopover
                metrics={metricConfigs}
                visibleMetrics={visibleMetrics}
                onToggleMetric={handleToggleMetric}
              />
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
      </CardHeader>

      <CardContent className="pt-0">
        <div ref={chartContainerRef}>
          <Suspense
            fallback={
              <Skeleton className="w-full" style={{ height: `${height}px` }} />
            }
          >
            <ResponsiveContainer width="100%" height={height}>
              <RadarChart
                cx="50%"
                cy="50%"
                outerRadius="75%"
                data={chartData}
              >
                <PolarGrid
                  className="stroke-muted"
                  strokeDasharray="3 3"
                />
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{
                    fontSize: 11,
                    fill: "hsl(var(--muted-foreground))",
                  }}
                  tickLine={false}
                />
                <PolarRadiusAxis
                  angle={30}
                  domain={[0, 1]}
                  tick={{
                    fontSize: 10,
                    fill: "hsl(var(--muted-foreground))",
                  }}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  tickCount={5}
                />

                {/* Render a Radar for each visible campaign */}
                {visibleSeries.map((series) => (
                  <Radar
                    key={series.id}
                    name={series.name}
                    dataKey={series.id}
                    stroke={series.color}
                    fill={series.color}
                    fillOpacity={fillOpacity}
                    strokeWidth={strokeWidth}
                    animationDuration={animationDuration}
                    dot={{
                      r: 4,
                      fill: series.color,
                      stroke: "hsl(var(--background))",
                      strokeWidth: 2,
                    }}
                    activeDot={{
                      r: 6,
                      fill: series.color,
                      stroke: "hsl(var(--background))",
                      strokeWidth: 2,
                    }}
                  />
                ))}

                {/* Tooltip */}
                {showTooltips && (
                  <RechartsTooltip
                    content={<CustomRadarTooltip metrics={metricConfigs} />}
                  />
                )}

                {/* Built-in legend (if not using custom) */}
                {showLegend && !campaignSeriesState.length && (
                  <Legend
                    wrapperStyle={{
                      paddingTop: "20px",
                    }}
                  />
                )}
              </RadarChart>
            </ResponsiveContainer>
          </Suspense>
        </div>

        {/* Custom interactive legend */}
        {showLegend && campaignSeriesState.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
            {campaignSeriesState.map((series) => (
              <CampaignLegendItem
                key={series.id}
                series={series}
                onToggle={() => handleToggleCampaign(series.id)}
              />
            ))}
          </div>
        )}

        {/* Summary stats */}
        <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground border-t pt-4">
          <span>{visibleMetrics.size} metrics displayed</span>
          <div className="flex items-center gap-2">
            <span className="text-xs">
              Click legend items to show/hide campaigns
            </span>
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
 * Simple normalized metrics chart without controls.
 */
export function SimpleNormalizedMetricsChart({
  comparison,
  height = 350,
  className,
}: {
  comparison: CampaignComparison | null | undefined;
  height?: number;
  className?: string;
}) {
  return (
    <NormalizedMetricsChart
      comparison={comparison}
      height={height}
      showMetricControls={false}
      showDownloadButton={false}
      showLegend={true}
      showTooltips={true}
      className={className}
    />
  );
}

/**
 * Compact normalized metrics chart for dashboard widgets.
 */
export function CompactNormalizedMetricsChart({
  comparison,
  className,
}: {
  comparison: CampaignComparison | null | undefined;
  className?: string;
}) {
  // Show only 4 key metrics for compact view
  const compactMetrics = DEFAULT_NORMALIZED_METRICS.slice(0, 4);

  return (
    <NormalizedMetricsChart
      comparison={comparison}
      title="Campaign Comparison"
      height={280}
      showMetricControls={false}
      showDownloadButton={false}
      showLegend={true}
      showTooltips={true}
      metricConfigs={compactMetrics}
      fillOpacity={0.3}
      className={className}
    />
  );
}

/**
 * Detailed normalized metrics chart with all controls.
 */
export function DetailedNormalizedMetricsChart({
  comparison,
  onExport,
  className,
}: {
  comparison: CampaignComparison | null | undefined;
  onExport?: (format: "png" | "svg") => void;
  className?: string;
}) {
  return (
    <NormalizedMetricsChart
      comparison={comparison}
      title="Detailed Campaign Metrics Comparison"
      description="Compare normalized performance metrics across campaigns. Higher values indicate better performance for each metric."
      height={450}
      showMetricControls={true}
      showDownloadButton={true}
      showLegend={true}
      showTooltips={true}
      onExport={onExport}
      fillOpacity={0.2}
      strokeWidth={2.5}
      className={className}
    />
  );
}

// =============================================================================
// Export Types
// =============================================================================

// Note: Types are already exported where they're defined above
export default NormalizedMetricsChart;
