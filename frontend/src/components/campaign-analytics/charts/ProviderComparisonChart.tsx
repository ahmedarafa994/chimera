/**
 * ProviderComparisonChart Component
 *
 * Grouped bar chart comparing success rates across providers.
 * Includes latency overlay, cost metrics if available, and drill-down support.
 * Uses Recharts with responsive container and lazy loading.
 */

"use client";

import * as React from "react";
import { useMemo, useState, useCallback, useRef, Suspense } from "react";
import {
  ArrowUp,
  ArrowDown,
  Download,
  BarChart3,
  RefreshCw,
  AlertTriangle,
  Filter,
  X,
  Clock,
  DollarSign,
  Info,
  Layers,
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
import type {
  ProviderBreakdown,
  BreakdownItem,
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
  Line,
} = RechartsComponents;

// =============================================================================
// Types
// =============================================================================

/**
 * Display mode for the chart.
 */
export type DisplayMode = "success_rate" | "grouped" | "stacked";

/**
 * Sort field options for providers.
 */
export type ProviderSortField =
  | "name"
  | "success_rate"
  | "attempts"
  | "avg_latency_ms"
  | "total_cost_cents";

/**
 * Sort direction.
 */
export type SortDirection = "asc" | "desc";

/**
 * Success tier for color coding.
 */
export type SuccessTier = "excellent" | "good" | "moderate" | "poor" | "critical";

/**
 * Data point for chart rendering.
 */
export interface ProviderChartDataPoint extends BreakdownItem {
  tier: SuccessTier;
  color: string;
  normalizedSuccessRate: number;
  normalizedLatency?: number | null;
  formattedCost?: string | null;
}

/**
 * Props for the ProviderComparisonChart component.
 */
export interface ProviderComparisonChartProps {
  /** Provider breakdown data from API */
  breakdown?: ProviderBreakdown | null;
  /** Chart title */
  title?: string;
  /** Chart description */
  description?: string;
  /** Height of the chart in pixels */
  height?: number;
  /** Initial display mode */
  initialDisplayMode?: DisplayMode;
  /** Show display mode toggle */
  showDisplayModeToggle?: boolean;
  /** Show sorting controls */
  showSortControls?: boolean;
  /** Show filter controls */
  showFilterControls?: boolean;
  /** Show download button */
  showDownloadButton?: boolean;
  /** Show legend */
  showLegend?: boolean;
  /** Show latency overlay */
  showLatencyOverlay?: boolean;
  /** Show cost metrics */
  showCostMetrics?: boolean;
  /** Minimum attempts threshold for filtering */
  minAttemptsThreshold?: number;
  /** Callback when a provider is clicked for drill-down */
  onProviderDrillDown?: (provider: string) => void;
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

// =============================================================================
// Constants
// =============================================================================

/** Success tier thresholds and colors */
const SUCCESS_TIERS: Record<
  SuccessTier,
  { min: number; max: number; color: string; label: string }
> = {
  excellent: { min: 0.8, max: 1.0, color: "#22c55e", label: "Excellent (80-100%)" },
  good: { min: 0.6, max: 0.8, color: "#84cc16", label: "Good (60-80%)" },
  moderate: { min: 0.4, max: 0.6, color: "#eab308", label: "Moderate (40-60%)" },
  poor: { min: 0.2, max: 0.4, color: "#f97316", label: "Poor (20-40%)" },
  critical: { min: 0.0, max: 0.2, color: "#ef4444", label: "Critical (0-20%)" },
};

/** Provider-specific colors */
const PROVIDER_COLORS: Record<string, string> = {
  openai: "#00A67E",
  anthropic: "#D4A574",
  google: "#4285F4",
  gemini: "#8E75B2",
  azure: "#0078D4",
  deepseek: "#00CED1",
  qwen: "#FF6600",
  cursor: "#7C3AED",
  mock: "#6B7280",
};

/** Fallback colors for unknown providers */
const FALLBACK_COLORS = [
  "#3b82f6", // blue
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
  "#6366f1", // indigo
];

/** Sort field labels */
const SORT_FIELD_LABELS: Record<ProviderSortField, string> = {
  name: "Provider Name",
  success_rate: "Success Rate",
  attempts: "Attempts",
  avg_latency_ms: "Avg Latency",
  total_cost_cents: "Total Cost",
};

/** Display mode labels */
const DISPLAY_MODE_LABELS: Record<DisplayMode, string> = {
  success_rate: "Success Rate",
  grouped: "Grouped Metrics",
  stacked: "Success/Failure",
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Determine success tier based on success rate.
 */
function getSuccessTier(successRate: number): SuccessTier {
  if (successRate >= 0.8) return "excellent";
  if (successRate >= 0.6) return "good";
  if (successRate >= 0.4) return "moderate";
  if (successRate >= 0.2) return "poor";
  return "critical";
}

/**
 * Get color for success rate.
 */
function getSuccessRateColor(successRate: number): string {
  const tier = getSuccessTier(successRate);
  return SUCCESS_TIERS[tier].color;
}

/**
 * Get color for provider.
 */
function getProviderColor(provider: string, index: number): string {
  const lowerProvider = provider.toLowerCase();
  if (PROVIDER_COLORS[lowerProvider]) {
    return PROVIDER_COLORS[lowerProvider];
  }
  return FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

/**
 * Format value as percentage.
 */
function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format latency in milliseconds.
 */
function formatLatency(ms: number | null | undefined): string {
  if (ms == null) return "N/A";
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Format cost in cents to dollars.
 */
function formatCost(cents: number | null | undefined): string {
  if (cents == null) return "N/A";
  if (cents < 100) return `${cents.toFixed(1)}¢`;
  return `$${(cents / 100).toFixed(2)}`;
}

/**
 * Format provider name for display.
 */
function formatProviderName(name: string): string {
  // Capitalize first letter and common provider names
  const knownProviders: Record<string, string> = {
    openai: "OpenAI",
    anthropic: "Anthropic",
    google: "Google",
    gemini: "Gemini",
    azure: "Azure",
    deepseek: "DeepSeek",
    qwen: "Qwen",
    cursor: "Cursor",
    mock: "Mock",
  };

  const lower = name.toLowerCase();
  if (knownProviders[lower]) {
    return knownProviders[lower];
  }

  // Capitalize first letter of each word
  return name
    .split(/[_-]/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Sort breakdown items based on field and direction.
 */
function sortItems(
  items: ProviderChartDataPoint[],
  field: ProviderSortField,
  direction: SortDirection
): ProviderChartDataPoint[] {
  return [...items].sort((a, b) => {
    let comparison = 0;

    switch (field) {
      case "name":
        comparison = a.name.localeCompare(b.name);
        break;
      case "success_rate":
        comparison = a.success_rate - b.success_rate;
        break;
      case "attempts":
        comparison = a.attempts - b.attempts;
        break;
      case "avg_latency_ms":
        const latA = a.avg_latency_ms ?? Infinity;
        const latB = b.avg_latency_ms ?? Infinity;
        comparison = latA - latB;
        break;
      case "total_cost_cents":
        const costA = a.total_cost_cents ?? 0;
        const costB = b.total_cost_cents ?? 0;
        comparison = costA - costB;
        break;
    }

    return direction === "desc" ? -comparison : comparison;
  });
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Loading skeleton for the chart.
 */
export function ProviderComparisonChartSkeleton({
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
            <Skeleton className="h-8 w-8" />
            <Skeleton className="h-8 w-8" />
            <Skeleton className="h-8 w-24" />
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
export function ProviderComparisonChartError({
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
export function ProviderComparisonChartEmpty({
  message = "No provider data available for this campaign",
  className,
}: {
  message?: string;
  className?: string;
}) {
  return (
    <Card className={className}>
      <CardContent className="flex flex-col items-center justify-center py-12">
        <Layers className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <p className="text-lg font-medium text-muted-foreground mb-2">
          No Provider Data
        </p>
        <p className="text-sm text-muted-foreground text-center max-w-md">
          {message}
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Custom tooltip component for the bar chart.
 */
interface BarTooltipProps {
  active?: boolean;
  payload?: Array<{
    value: number;
    name: string;
    dataKey: string;
    color?: string;
    payload: ProviderChartDataPoint;
  }>;
  label?: string;
  showLatency?: boolean;
  showCost?: boolean;
}

function CustomBarTooltip({
  active,
  payload,
  label,
  showLatency = true,
  showCost = true,
}: BarTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <Card className="p-3 shadow-lg border bg-popover min-w-[220px]">
      <div className="flex items-center gap-2 mb-3">
        <div
          className="w-4 h-4 rounded-full"
          style={{ backgroundColor: data.color }}
        />
        <p className="font-semibold text-sm text-popover-foreground">
          {formatProviderName(data.name)}
        </p>
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">Success Rate:</span>
          <div className="flex items-center gap-2">
            <span className="font-medium" style={{ color: getSuccessRateColor(data.success_rate) }}>
              {formatPercentage(data.success_rate)}
            </span>
            <Badge
              variant="secondary"
              className="text-xs py-0 px-1"
              style={{ backgroundColor: `${getSuccessRateColor(data.success_rate)}20`, color: getSuccessRateColor(data.success_rate) }}
            >
              {data.tier}
            </Badge>
          </div>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">Attempts:</span>
          <span className="font-medium">{data.attempts.toLocaleString()}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">Successes:</span>
          <span className="font-medium">{data.successes.toLocaleString()}</span>
        </div>
        {showLatency && data.avg_latency_ms != null && (
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground flex items-center gap-1">
              <Clock className="h-3 w-3" />
              Avg Latency:
            </span>
            <span className="font-medium">{formatLatency(data.avg_latency_ms)}</span>
          </div>
        )}
        {showCost && data.total_cost_cents != null && (
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground flex items-center gap-1">
              <DollarSign className="h-3 w-3" />
              Total Cost:
            </span>
            <span className="font-medium">{formatCost(data.total_cost_cents)}</span>
          </div>
        )}
        {data.avg_tokens != null && (
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">Avg Tokens:</span>
            <span className="font-medium">{data.avg_tokens.toLocaleString()}</span>
          </div>
        )}
      </div>
    </Card>
  );
}

/**
 * Legend component showing success tiers.
 */
function SuccessTierLegend({ className }: { className?: string }) {
  return (
    <div className={cn("flex flex-wrap items-center justify-center gap-4", className)}>
      {Object.entries(SUCCESS_TIERS).map(([tier, config]) => (
        <div key={tier} className="flex items-center gap-1.5">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: config.color }}
          />
          <span className="text-xs text-muted-foreground">{config.label}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * Metrics summary component.
 */
function MetricsSummary({
  chartData,
  showLatency,
  showCost,
}: {
  chartData: ProviderChartDataPoint[];
  showLatency: boolean;
  showCost: boolean;
}) {
  const avgSuccessRate = useMemo(() => {
    if (chartData.length === 0) return 0;
    const totalAttempts = chartData.reduce((sum, d) => sum + d.attempts, 0);
    const totalSuccesses = chartData.reduce((sum, d) => sum + d.successes, 0);
    return totalAttempts > 0 ? totalSuccesses / totalAttempts : 0;
  }, [chartData]);

  const avgLatency = useMemo(() => {
    const withLatency = chartData.filter((d) => d.avg_latency_ms != null);
    if (withLatency.length === 0) return null;
    const sum = withLatency.reduce((s, d) => s + (d.avg_latency_ms ?? 0), 0);
    return sum / withLatency.length;
  }, [chartData]);

  const totalCost = useMemo(() => {
    const withCost = chartData.filter((d) => d.total_cost_cents != null);
    if (withCost.length === 0) return null;
    return withCost.reduce((s, d) => s + (d.total_cost_cents ?? 0), 0);
  }, [chartData]);

  return (
    <div className="flex flex-wrap items-center justify-center gap-6 text-sm">
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground">Avg Success:</span>
        <span
          className="font-medium"
          style={{ color: getSuccessRateColor(avgSuccessRate) }}
        >
          {formatPercentage(avgSuccessRate)}
        </span>
      </div>
      {showLatency && avgLatency != null && (
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">Avg Latency:</span>
          <span className="font-medium">{formatLatency(avgLatency)}</span>
        </div>
      )}
      {showCost && totalCost != null && (
        <div className="flex items-center gap-2">
          <DollarSign className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">Total Cost:</span>
          <span className="font-medium">{formatCost(totalCost)}</span>
        </div>
      )}
    </div>
  );
}

/**
 * Filter chips display.
 */
function FilterChips({
  selectedProviders,
  onRemove,
  onClearAll,
}: {
  selectedProviders: string[];
  onRemove: (provider: string) => void;
  onClearAll: () => void;
}) {
  if (selectedProviders.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 mt-2">
      <span className="text-xs text-muted-foreground">Showing:</span>
      {selectedProviders.map((provider) => (
        <Badge
          key={provider}
          variant="secondary"
          className="text-xs gap-1 pr-1"
        >
          {formatProviderName(provider)}
          <button
            onClick={() => onRemove(provider)}
            className="ml-1 hover:bg-muted rounded-full p-0.5"
            aria-label={`Remove ${provider} filter`}
          >
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
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

// =============================================================================
// Main Component
// =============================================================================

/**
 * Provider Comparison Chart Component
 *
 * Displays a grouped bar chart comparing success rates across providers.
 * Features:
 * - Color-coded by success tier or provider
 * - Latency overlay line chart
 * - Cost metrics display
 * - Sort by different metrics
 * - Filter providers
 * - Drill-down on click
 */
export function ProviderComparisonChart({
  breakdown,
  title = "Provider Comparison",
  description,
  height = 400,
  initialDisplayMode = "success_rate",
  showDisplayModeToggle = true,
  showSortControls = true,
  showFilterControls = true,
  showDownloadButton = true,
  showLegend = true,
  showLatencyOverlay = true,
  showCostMetrics = true,
  minAttemptsThreshold = 0,
  onProviderDrillDown,
  onExport,
  isLoading = false,
  error = null,
  onRetry,
  className,
  chartRef: externalChartRef,
}: ProviderComparisonChartProps) {
  // Internal chart ref if not provided
  const internalChartRef = useRef<HTMLDivElement>(null);
  const chartContainerRef = externalChartRef || internalChartRef;

  // State
  const [displayMode, setDisplayMode] = useState<DisplayMode>(initialDisplayMode);
  const [sortField, setSortField] = useState<ProviderSortField>("success_rate");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [selectedProviders, setSelectedProviders] = useState<string[]>([]);
  const [activeLatencyOverlay, setActiveLatencyOverlay] = useState(showLatencyOverlay);

  // Process and transform data
  const chartData = useMemo((): ProviderChartDataPoint[] => {
    if (!breakdown?.items?.length) return [];

    // Filter by minimum attempts if specified
    let filtered = breakdown.items.filter(
      (item) => item.attempts >= minAttemptsThreshold
    );

    // Filter by selected providers if any
    if (selectedProviders.length > 0) {
      filtered = filtered.filter((item) =>
        selectedProviders.includes(item.name)
      );
    }

    // Find max latency for normalization
    const maxLatency = Math.max(
      ...filtered.map((item) => item.avg_latency_ms ?? 0)
    );

    // Transform to chart data points with tier and color
    const transformed: ProviderChartDataPoint[] = filtered.map((item, index) => ({
      ...item,
      tier: getSuccessTier(item.success_rate),
      color: displayMode === "success_rate"
        ? getSuccessRateColor(item.success_rate)
        : getProviderColor(item.name, index),
      normalizedSuccessRate: item.success_rate * 100, // For display
      normalizedLatency: item.avg_latency_ms != null && maxLatency > 0
        ? (item.avg_latency_ms / maxLatency) * 100
        : null,
      formattedCost: formatCost(item.total_cost_cents),
    }));

    // Sort the data
    return sortItems(transformed, sortField, sortDirection);
  }, [breakdown, minAttemptsThreshold, selectedProviders, sortField, sortDirection, displayMode]);

  // Check if any providers have cost data
  const hasCostData = useMemo(() => {
    return chartData.some((item) => item.total_cost_cents != null);
  }, [chartData]);

  // Check if any providers have latency data
  const hasLatencyData = useMemo(() => {
    return chartData.some((item) => item.avg_latency_ms != null);
  }, [chartData]);

  // All available providers for filter
  const allProviders = useMemo(() => {
    if (!breakdown?.items?.length) return [];
    return breakdown.items.map((item) => item.name);
  }, [breakdown]);

  // Handlers
  const handleSortFieldChange = useCallback((value: string) => {
    setSortField(value as ProviderSortField);
  }, []);

  const toggleSortDirection = useCallback(() => {
    setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
  }, []);

  const handleDisplayModeChange = useCallback((value: string) => {
    setDisplayMode(value as DisplayMode);
  }, []);

  const handleProviderToggle = useCallback((provider: string) => {
    setSelectedProviders((prev) =>
      prev.includes(provider)
        ? prev.filter((p) => p !== provider)
        : [...prev, provider]
    );
  }, []);

  const handleRemoveFilter = useCallback((provider: string) => {
    setSelectedProviders((prev) => prev.filter((p) => p !== provider));
  }, []);

  const handleClearFilters = useCallback(() => {
    setSelectedProviders([]);
  }, []);

  const handleBarClick = useCallback(
    (data: ProviderChartDataPoint) => {
      onProviderDrillDown?.(data.name);
    },
    [onProviderDrillDown]
  );

  const handleExport = useCallback(
    (format: "png" | "svg") => {
      onExport?.(format);
    },
    [onExport]
  );

  const toggleLatencyOverlay = useCallback(() => {
    setActiveLatencyOverlay((prev) => !prev);
  }, []);

  // Render loading state
  if (isLoading) {
    return <ProviderComparisonChartSkeleton height={height} className={className} />;
  }

  // Render error state
  if (error) {
    return (
      <ProviderComparisonChartError
        error={error}
        onRetry={onRetry}
        className={className}
      />
    );
  }

  // Render empty state
  if (chartData.length === 0) {
    return <ProviderComparisonChartEmpty className={className} />;
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <CardTitle className="text-base font-medium">{title}</CardTitle>
            {breakdown?.best_provider && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="gap-1">
                      <span className="text-xs">Best:</span>
                      <span className="font-medium">
                        {formatProviderName(breakdown.best_provider)}
                      </span>
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Highest success rate provider</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Display mode toggle */}
            {showDisplayModeToggle && (
              <Select value={displayMode} onValueChange={handleDisplayModeChange}>
                <SelectTrigger className="w-36 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(DISPLAY_MODE_LABELS).map(([mode, label]) => (
                    <SelectItem key={mode} value={mode}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            {/* Sort controls */}
            {showSortControls && (
              <div className="flex items-center gap-1">
                <Select value={sortField} onValueChange={handleSortFieldChange}>
                  <SelectTrigger className="w-32 h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(SORT_FIELD_LABELS).map(([field, label]) => (
                      <SelectItem key={field} value={field}>
                        {label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={toggleSortDirection}
                        aria-label={`Sort ${sortDirection === "asc" ? "descending" : "ascending"}`}
                      >
                        {sortDirection === "desc" ? (
                          <ArrowDown className="h-4 w-4" />
                        ) : (
                          <ArrowUp className="h-4 w-4" />
                        )}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      {sortDirection === "asc" ? "Ascending" : "Descending"}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            )}

            {/* Latency overlay toggle */}
            {hasLatencyData && showLatencyOverlay && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={activeLatencyOverlay ? "secondary" : "ghost"}
                      size="icon"
                      className="h-8 w-8"
                      onClick={toggleLatencyOverlay}
                      aria-label="Toggle latency overlay"
                    >
                      <Clock className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {activeLatencyOverlay ? "Hide Latency" : "Show Latency"}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}

            {/* Filter controls */}
            {showFilterControls && allProviders.length > 0 && (
              <DropdownMenu>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant={selectedProviders.length > 0 ? "secondary" : "ghost"}
                          size="icon"
                          className="h-8 w-8 relative"
                          aria-label="Filter providers"
                        >
                          <Filter className="h-4 w-4" />
                          {selectedProviders.length > 0 && (
                            <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-primary text-primary-foreground text-[10px] flex items-center justify-center">
                              {selectedProviders.length}
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
                  {allProviders.map((provider) => (
                    <DropdownMenuCheckboxItem
                      key={provider}
                      checked={selectedProviders.includes(provider)}
                      onCheckedChange={() => handleProviderToggle(provider)}
                    >
                      {formatProviderName(provider)}
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
          selectedProviders={selectedProviders}
          onRemove={handleRemoveFilter}
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
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                onClick={(data) => {
                  if (data?.activePayload?.[0]?.payload) {
                    handleBarClick(data.activePayload[0].payload);
                  }
                }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  className="stroke-muted"
                  opacity={0.5}
                  vertical={false}
                />
                <XAxis
                  dataKey="name"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={formatProviderName}
                />
                <YAxis
                  yAxisId="left"
                  domain={[0, 100]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={(value) => `${value}%`}
                  width={45}
                />
                {activeLatencyOverlay && hasLatencyData && (
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[0, "auto"]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                    tickFormatter={(value) => formatLatency(value)}
                    width={60}
                  />
                )}
                <RechartsTooltip
                  content={
                    <CustomBarTooltip
                      showLatency={hasLatencyData}
                      showCost={hasCostData && showCostMetrics}
                    />
                  }
                  cursor={{ fill: "hsl(var(--muted))", opacity: 0.3 }}
                />
                {showLegend && (
                  <Legend
                    verticalAlign="bottom"
                    height={36}
                    iconType="rect"
                    formatter={(value) => (
                      <span className="text-sm text-foreground">{value}</span>
                    )}
                  />
                )}

                {/* Main success rate bar */}
                <Bar
                  yAxisId="left"
                  dataKey="normalizedSuccessRate"
                  name="Success Rate"
                  radius={[4, 4, 0, 0]}
                  cursor="pointer"
                  maxBarSize={80}
                >
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.color}
                      className="transition-opacity hover:opacity-80"
                    />
                  ))}
                </Bar>

                {/* Latency overlay line */}
                {activeLatencyOverlay && hasLatencyData && (
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="avg_latency_ms"
                    name="Avg Latency"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={{ fill: "#8b5cf6", r: 4, strokeWidth: 0 }}
                    activeDot={{ r: 6, strokeWidth: 2, stroke: "#fff" }}
                  />
                )}
              </BarChart>
            </ResponsiveContainer>
          </Suspense>
        </div>

        {/* Legend and summary */}
        <div className="mt-4 space-y-3">
          {showLegend && displayMode === "success_rate" && <SuccessTierLegend />}

          <MetricsSummary
            chartData={chartData}
            showLatency={hasLatencyData}
            showCost={hasCostData && showCostMetrics}
          />

          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center gap-4">
              <span>{chartData.length} providers</span>
              {breakdown?.best_provider && (
                <span className="hidden sm:inline">
                  Best: {formatProviderName(breakdown.best_provider)}
                </span>
              )}
            </div>
            {minAttemptsThreshold > 0 && (
              <span className="text-xs">
                Showing providers with ≥{minAttemptsThreshold} attempts
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
 * Simple provider chart without controls.
 */
export function SimpleProviderChart({
  breakdown,
  height = 300,
  className,
  onProviderDrillDown,
}: {
  breakdown?: ProviderBreakdown | null;
  height?: number;
  className?: string;
  onProviderDrillDown?: (provider: string) => void;
}) {
  return (
    <ProviderComparisonChart
      breakdown={breakdown}
      height={height}
      showDisplayModeToggle={false}
      showSortControls={false}
      showFilterControls={false}
      showDownloadButton={false}
      showLegend={false}
      showLatencyOverlay={false}
      showCostMetrics={false}
      className={className}
      onProviderDrillDown={onProviderDrillDown}
    />
  );
}

/**
 * Provider chart with latency focus.
 */
export function ProviderLatencyChart({
  breakdown,
  height = 400,
  className,
  title = "Provider Latency Comparison",
}: {
  breakdown?: ProviderBreakdown | null;
  height?: number;
  className?: string;
  title?: string;
}) {
  return (
    <ProviderComparisonChart
      breakdown={breakdown}
      title={title}
      height={height}
      showDisplayModeToggle={false}
      showLatencyOverlay={true}
      showCostMetrics={false}
      showLegend={true}
      className={className}
    />
  );
}

/**
 * Provider chart with cost focus.
 */
export function ProviderCostChart({
  breakdown,
  height = 400,
  className,
  title = "Provider Cost Comparison",
}: {
  breakdown?: ProviderBreakdown | null;
  height?: number;
  className?: string;
  title?: string;
}) {
  return (
    <ProviderComparisonChart
      breakdown={breakdown}
      title={title}
      height={height}
      showDisplayModeToggle={false}
      showLatencyOverlay={false}
      showCostMetrics={true}
      showLegend={true}
      className={className}
    />
  );
}

/**
 * Compact provider chart for dashboards.
 */
export function CompactProviderChart({
  breakdown,
  onProviderDrillDown,
  className,
}: {
  breakdown?: ProviderBreakdown | null;
  onProviderDrillDown?: (provider: string) => void;
  className?: string;
}) {
  // Show top 5 providers only
  const limitedBreakdown = useMemo(() => {
    if (!breakdown) return null;
    return {
      ...breakdown,
      items: [...breakdown.items]
        .sort((a, b) => b.success_rate - a.success_rate)
        .slice(0, 5),
    };
  }, [breakdown]);

  return (
    <ProviderComparisonChart
      breakdown={limitedBreakdown}
      title="Top 5 Providers"
      height={250}
      showDisplayModeToggle={false}
      showSortControls={false}
      showFilterControls={false}
      showDownloadButton={false}
      showLegend={false}
      showLatencyOverlay={false}
      showCostMetrics={false}
      className={className}
      onProviderDrillDown={onProviderDrillDown}
    />
  );
}

// =============================================================================
// Export Types
// =============================================================================

export type {
  ProviderChartDataPoint,
  ProviderComparisonChartProps,
};
