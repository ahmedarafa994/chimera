"use client";

import * as React from "react";
import {
  BarChart3,
  Target,
  TrendingUp,
  Timer,
  Clock,
  Percent,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  StatisticsCard,
  StatisticsCardSkeleton,
  DistributionStatCard,
  type ValueFormat,
  type ColorVariant,
} from "./StatisticsCard";
import { useCampaignStatistics, useCampaignSummary } from "@/lib/api/query/campaign-queries";
import type { CampaignStatistics, CampaignSummary, DistributionStats } from "@/types/campaign-analytics";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

// =============================================================================
// Types
// =============================================================================

/**
 * Props for the StatisticsSummaryPanel component.
 */
export interface StatisticsSummaryPanelProps {
  /** Campaign ID to display statistics for */
  campaignId: string | null;
  /** Optional pre-loaded statistics (skips API call) */
  statistics?: CampaignStatistics | null;
  /** Optional pre-loaded summary (skips API call for summary data) */
  summary?: CampaignSummary | null;
  /** Show compact view (smaller cards) */
  compact?: boolean;
  /** Show loading skeleton */
  isLoading?: boolean;
  /** Custom error message */
  error?: string | null;
  /** Called when a statistic card is clicked */
  onStatisticClick?: (metricType: string) => void;
  /** Additional CSS classes */
  className?: string;
  /** Show refresh button */
  showRefresh?: boolean;
  /** Called when refresh is requested */
  onRefresh?: () => void;
}

/**
 * Configuration for a single statistic metric display.
 */
interface StatisticMetricConfig {
  key: string;
  label: string;
  icon: LucideIcon;
  format: ValueFormat;
  variant?: ColorVariant;
  precision?: number;
  unit?: string;
  description?: string;
  /** Whether a higher value is better (for coloring) */
  higherIsBetter?: boolean;
  /** Extract value from statistics object */
  getValue: (stats: CampaignStatistics, summary?: CampaignSummary | null) => number | null | undefined;
  /** Extract distribution stats for tooltip */
  getDistribution?: (stats: CampaignStatistics) => DistributionStats | null | undefined;
  /** Dynamically calculate variant based on value */
  getVariant?: (value: number | null | undefined) => ColorVariant;
}

// =============================================================================
// Metric Configurations
// =============================================================================

/**
 * Configuration for the 6 main statistics to display in the summary panel.
 */
const SUMMARY_METRICS: StatisticMetricConfig[] = [
  {
    key: "total_attempts",
    label: "Total Attempts",
    icon: BarChart3,
    format: "compact",
    precision: 0,
    description: "Total prompts executed",
    higherIsBetter: true,
    getValue: (stats) => stats.attempts.total,
    getVariant: (value) => {
      if (value === null || value === undefined) return "default";
      return value > 0 ? "info" : "muted";
    },
  },
  {
    key: "success_rate_mean",
    label: "Success Rate (Mean)",
    icon: Target,
    format: "percentage",
    precision: 1,
    description: "Average success rate",
    higherIsBetter: true,
    getValue: (stats) => stats.success_rate.mean,
    getDistribution: (stats) => stats.success_rate,
    getVariant: (value) => {
      if (value === null || value === undefined) return "default";
      if (value >= 0.7) return "success";
      if (value >= 0.4) return "warning";
      return "error";
    },
  },
  {
    key: "success_rate_median",
    label: "Median Success",
    icon: Percent,
    format: "percentage",
    precision: 1,
    description: "Median success rate",
    higherIsBetter: true,
    getValue: (stats) => stats.success_rate.median,
    getDistribution: (stats) => stats.success_rate,
    getVariant: (value) => {
      if (value === null || value === undefined) return "default";
      if (value >= 0.7) return "success";
      if (value >= 0.4) return "warning";
      return "error";
    },
  },
  {
    key: "success_rate_p95",
    label: "P95 Success",
    icon: TrendingUp,
    format: "percentage",
    precision: 1,
    description: "95th percentile success",
    higherIsBetter: true,
    getValue: (stats) => stats.success_rate.percentiles?.p95 ?? null,
    getDistribution: (stats) => stats.success_rate,
    getVariant: (value) => {
      if (value === null || value === undefined) return "default";
      if (value >= 0.9) return "success";
      if (value >= 0.6) return "warning";
      return "error";
    },
  },
  {
    key: "avg_latency",
    label: "Avg Latency",
    icon: Timer,
    format: "duration",
    precision: 0,
    description: "Average response time",
    higherIsBetter: false,
    getValue: (stats) => stats.latency_ms.mean,
    getDistribution: (stats) => stats.latency_ms,
    getVariant: (value) => {
      if (value === null || value === undefined) return "default";
      // Lower latency is better
      if (value <= 1000) return "success"; // Under 1 second
      if (value <= 3000) return "warning"; // Under 3 seconds
      return "error";
    },
  },
  {
    key: "total_duration",
    label: "Total Duration",
    icon: Clock,
    format: "custom",
    precision: 0,
    description: "Campaign run time",
    higherIsBetter: false,
    getValue: (stats, summary) => {
      // Use duration from stats if available, otherwise from summary
      if (stats.total_duration_seconds !== null && stats.total_duration_seconds !== undefined) {
        return stats.total_duration_seconds * 1000; // Convert to ms for consistent formatting
      }
      if (summary?.duration_seconds !== null && summary?.duration_seconds !== undefined) {
        return summary.duration_seconds * 1000;
      }
      return null;
    },
    getVariant: () => "muted",
  },
];

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format duration in seconds to human-readable string.
 */
function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";

  const seconds = ms / 1000;

  if (seconds < 60) {
    return `${seconds.toFixed(0)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  if (minutes < 60) {
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  if (hours < 24) {
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  }

  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;

  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Loading skeleton for the entire panel.
 */
function StatisticsSummaryPanelSkeleton({
  compact = false,
  className,
}: {
  compact?: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid gap-4",
        compact
          ? "grid-cols-2 md:grid-cols-3 lg:grid-cols-6"
          : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
        className
      )}
    >
      {Array.from({ length: 6 }).map((_, index) => (
        <StatisticsCardSkeleton key={index} />
      ))}
    </div>
  );
}

/**
 * Error state for the panel.
 */
function StatisticsSummaryPanelError({
  error,
  onRetry,
  className,
}: {
  error: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <Alert variant="destructive" className={className}>
      <AlertTriangle className="h-4 w-4" />
      <AlertDescription className="flex items-center justify-between">
        <span>{error}</span>
        {onRetry && (
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="h-3 w-3 mr-1" />
            Retry
          </Button>
        )}
      </AlertDescription>
    </Alert>
  );
}

/**
 * Empty state when no campaign is selected.
 */
function StatisticsSummaryPanelEmpty({ className }: { className?: string }) {
  return (
    <Card className={cn("border-dashed", className)}>
      <CardContent className="flex flex-col items-center justify-center py-10 text-center">
        <BarChart3 className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <h3 className="text-lg font-medium text-muted-foreground">
          No Campaign Selected
        </h3>
        <p className="text-sm text-muted-foreground/70 max-w-sm mt-1">
          Select a campaign to view its statistical summary including success rates,
          latency, and duration metrics.
        </p>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * StatisticsSummaryPanel Component
 *
 * Displays a responsive grid of StatisticsCards showing key campaign metrics:
 * - Total Attempts
 * - Success Rate (Mean)
 * - Median Success
 * - P95 Success
 * - Average Latency
 * - Total Duration
 *
 * Features:
 * - Responsive 2x3 grid layout (adapts to screen size)
 * - Loading skeletons during data fetch
 * - Error handling with retry option
 * - Optional compact mode for smaller displays
 * - Distribution tooltips with full stats on hover
 * - Automatic color coding based on metric values
 * - Click handlers for drill-down navigation
 *
 * @example
 * ```tsx
 * // Basic usage with campaign ID
 * <StatisticsSummaryPanel campaignId="campaign-123" />
 *
 * // With pre-loaded data (skips API call)
 * <StatisticsSummaryPanel
 *   campaignId="campaign-123"
 *   statistics={preloadedStats}
 *   summary={preloadedSummary}
 * />
 *
 * // Compact mode for sidebars
 * <StatisticsSummaryPanel
 *   campaignId="campaign-123"
 *   compact
 *   onStatisticClick={(metric) => console.log('Clicked:', metric)}
 * />
 * ```
 *
 * @accessibility
 * - Semantic grid structure for screen readers
 * - ARIA labels on interactive cards
 * - Keyboard navigation support
 * - Color is not the only indicator (includes icons and labels)
 */
export function StatisticsSummaryPanel({
  campaignId,
  statistics: propStatistics,
  summary: propSummary,
  compact = false,
  isLoading: propIsLoading,
  error: propError,
  onStatisticClick,
  className,
  showRefresh = false,
  onRefresh,
}: StatisticsSummaryPanelProps) {
  // Fetch statistics if not provided
  const {
    data: fetchedStatistics,
    isLoading: statsLoading,
    error: statsError,
    refetch: refetchStats,
  } = useCampaignStatistics(
    campaignId,
    // Only fetch if campaignId exists and statistics not provided
    !!campaignId && !propStatistics
  );

  // Fetch summary if not provided (for duration data)
  const {
    data: fetchedSummary,
    isLoading: summaryLoading,
    error: summaryError,
  } = useCampaignSummary(
    campaignId,
    // Only fetch if campaignId exists and summary not provided
    !!campaignId && !propSummary && !propStatistics
  );

  // Use provided data or fetched data
  const statistics = propStatistics ?? fetchedStatistics ?? null;
  const summary = propSummary ?? fetchedSummary ?? null;
  const isLoading = propIsLoading ?? (statsLoading || summaryLoading);
  const error = propError ?? (statsError ? "Failed to load campaign statistics" : null);

  // Handle refresh
  const handleRefresh = React.useCallback(() => {
    if (onRefresh) {
      onRefresh();
    } else {
      refetchStats();
    }
  }, [onRefresh, refetchStats]);

  // Handle card click
  const handleCardClick = React.useCallback(
    (metricKey: string) => {
      if (onStatisticClick) {
        onStatisticClick(metricKey);
      }
    },
    [onStatisticClick]
  );

  // No campaign selected
  if (!campaignId) {
    return <StatisticsSummaryPanelEmpty className={className} />;
  }

  // Loading state
  if (isLoading) {
    return <StatisticsSummaryPanelSkeleton compact={compact} className={className} />;
  }

  // Error state
  if (error) {
    return (
      <StatisticsSummaryPanelError
        error={error}
        onRetry={handleRefresh}
        className={className}
      />
    );
  }

  // No data available
  if (!statistics) {
    return (
      <StatisticsSummaryPanelError
        error="No statistics data available for this campaign"
        onRetry={handleRefresh}
        className={className}
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* Optional header with refresh */}
      {showRefresh && (
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">
            Campaign Statistics
          </h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            className="h-8 px-2"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}

      {/* Statistics Grid */}
      <div
        className={cn(
          "grid gap-4",
          compact
            ? "grid-cols-2 md:grid-cols-3 lg:grid-cols-6"
            : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
          className
        )}
        role="list"
        aria-label="Campaign statistics summary"
      >
        {SUMMARY_METRICS.map((metric) => {
          const value = metric.getValue(statistics, summary);
          const distribution = metric.getDistribution?.(statistics);
          const variant = metric.getVariant?.(value) ?? metric.variant ?? "default";

          // Special handling for duration formatting
          const formatValue =
            metric.format === "custom" && metric.key === "total_duration"
              ? (v: number | string) => formatDuration(typeof v === "number" ? v : null)
              : undefined;

          return (
            <StatisticsCard
              key={metric.key}
              label={metric.label}
              value={value}
              format={metric.format === "custom" ? "number" : metric.format}
              formatValue={formatValue}
              precision={metric.precision}
              unit={metric.unit}
              icon={metric.icon}
              variant={variant}
              description={metric.description}
              distributionStats={distribution}
              onClick={
                onStatisticClick
                  ? () => handleCardClick(metric.key)
                  : undefined
              }
              ariaLabel={`${metric.label}: ${
                value !== null && value !== undefined
                  ? metric.format === "custom" && metric.key === "total_duration"
                    ? formatDuration(value)
                    : metric.format === "percentage"
                    ? `${(value * 100).toFixed(metric.precision ?? 1)}%`
                    : String(value)
                  : "No data"
              }`}
            />
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * Compact version of the statistics summary panel.
 */
export function CompactStatisticsSummary({
  campaignId,
  statistics,
  className,
  onStatisticClick,
}: Omit<StatisticsSummaryPanelProps, "compact">) {
  return (
    <StatisticsSummaryPanel
      campaignId={campaignId}
      statistics={statistics}
      compact
      className={className}
      onStatisticClick={onStatisticClick}
    />
  );
}

/**
 * Inline summary showing only key metrics (for headers/toolbars).
 */
export function InlineStatisticsSummary({
  statistics,
  className,
}: {
  statistics: CampaignStatistics | null | undefined;
  className?: string;
}) {
  if (!statistics) {
    return null;
  }

  const successRate = statistics.success_rate.mean;
  const avgLatency = statistics.latency_ms.mean;
  const totalAttempts = statistics.attempts.total;

  return (
    <div
      className={cn(
        "flex items-center gap-4 text-sm text-muted-foreground",
        className
      )}
    >
      <span className="flex items-center gap-1.5">
        <Target className="h-3.5 w-3.5" />
        <span className="font-medium">
          {successRate !== null && successRate !== undefined
            ? `${(successRate * 100).toFixed(1)}%`
            : "—"}
        </span>
        <span className="text-xs">success</span>
      </span>
      <span className="text-border">|</span>
      <span className="flex items-center gap-1.5">
        <Timer className="h-3.5 w-3.5" />
        <span className="font-medium">
          {avgLatency !== null && avgLatency !== undefined
            ? avgLatency >= 1000
              ? `${(avgLatency / 1000).toFixed(1)}s`
              : `${avgLatency.toFixed(0)}ms`
            : "—"}
        </span>
        <span className="text-xs">avg</span>
      </span>
      <span className="text-border">|</span>
      <span className="flex items-center gap-1.5">
        <BarChart3 className="h-3.5 w-3.5" />
        <span className="font-medium">{totalAttempts}</span>
        <span className="text-xs">attempts</span>
      </span>
    </div>
  );
}

// =============================================================================
// Exports
// =============================================================================

export default StatisticsSummaryPanel;

// Export sub-components for flexibility
export {
  StatisticsSummaryPanelSkeleton,
  StatisticsSummaryPanelError,
  StatisticsSummaryPanelEmpty,
};
