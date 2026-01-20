"use client";

import * as React from "react";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { DistributionStats, PercentileStats } from "@/types/campaign-analytics";

// =============================================================================
// Types
// =============================================================================

/**
 * Trend direction for delta values.
 */
export type TrendDirection = "up" | "down" | "neutral";

/**
 * Metric type for different statistical measures.
 */
export type StatisticMetricType =
  | "mean"
  | "median"
  | "p50"
  | "p90"
  | "p95"
  | "p99"
  | "std_dev"
  | "min"
  | "max"
  | "count"
  | "total"
  | "rate"
  | "custom";

/**
 * Format for displaying values.
 */
export type ValueFormat =
  | "number"
  | "percentage"
  | "duration"
  | "currency"
  | "compact"
  | "custom";

/**
 * Color scheme variants for the card.
 */
export type ColorVariant =
  | "default"
  | "success"
  | "warning"
  | "error"
  | "info"
  | "muted";

/**
 * Props for the StatisticsCard component.
 */
export interface StatisticsCardProps {
  /** Main label displayed at the top of the card */
  label: string;
  /** Primary value to display */
  value: number | string | null | undefined;
  /** Optional description or sublabel */
  description?: string;
  /** Type of metric being displayed */
  metricType?: StatisticMetricType;
  /** Format for the value display */
  format?: ValueFormat;
  /** Number of decimal places for number values */
  precision?: number;
  /** Custom format function */
  formatValue?: (value: number | string) => string;
  /** Unit to display after the value (e.g., "ms", "%", "tokens") */
  unit?: string;
  /** Icon to display in the header */
  icon?: LucideIcon;
  /** Delta/change from previous period */
  delta?: number | null;
  /** Label for the delta value */
  deltaLabel?: string;
  /** Whether higher delta is better (affects color) */
  deltaPositiveIsGood?: boolean;
  /** Trend direction (overrides automatic calculation from delta) */
  trend?: TrendDirection;
  /** Color variant for the card */
  variant?: ColorVariant;
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: string | null;
  /** Additional CSS classes */
  className?: string;
  /** Show full distribution stats on hover */
  distributionStats?: DistributionStats | null;
  /** Click handler */
  onClick?: () => void;
  /** Accessible name for screen readers */
  ariaLabel?: string;
}

/**
 * Props for creating a StatisticsCard from DistributionStats.
 */
export interface DistributionStatCardProps {
  /** Label for the card */
  label: string;
  /** Distribution statistics object */
  stats: DistributionStats | null | undefined;
  /** Which metric from the distribution to display */
  primaryMetric?: "mean" | "median" | "p95" | "p99";
  /** Format for the value */
  format?: ValueFormat;
  /** Unit for the value */
  unit?: string;
  /** Precision for number display */
  precision?: number;
  /** Icon for the card */
  icon?: LucideIcon;
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: string | null;
  /** Additional CSS classes */
  className?: string;
  /** Click handler */
  onClick?: () => void;
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format a number based on the specified format type.
 */
function formatNumber(
  value: number | string | null | undefined,
  format: ValueFormat = "number",
  precision: number = 2,
  customFormatter?: (value: number | string) => string
): string {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "string" && customFormatter) {
    return customFormatter(value);
  }

  const numValue = typeof value === "string" ? parseFloat(value) : value;
  if (isNaN(numValue)) {
    return typeof value === "string" ? value : "—";
  }

  if (customFormatter) {
    return customFormatter(numValue);
  }

  switch (format) {
    case "percentage":
      return `${(numValue * 100).toFixed(precision)}%`;
    case "duration":
      if (numValue >= 1000) {
        return `${(numValue / 1000).toFixed(precision)}s`;
      }
      return `${numValue.toFixed(precision)}ms`;
    case "currency":
      if (numValue >= 100) {
        return `$${(numValue / 100).toFixed(2)}`;
      }
      return `${numValue.toFixed(0)}¢`;
    case "compact":
      if (Math.abs(numValue) >= 1000000) {
        return `${(numValue / 1000000).toFixed(1)}M`;
      }
      if (Math.abs(numValue) >= 1000) {
        return `${(numValue / 1000).toFixed(1)}K`;
      }
      return numValue.toFixed(precision);
    case "custom":
      return numValue.toFixed(precision);
    default:
      return numValue.toFixed(precision);
  }
}

/**
 * Calculate trend direction from delta value.
 */
function calculateTrend(delta: number | null | undefined): TrendDirection {
  if (delta === null || delta === undefined || delta === 0) {
    return "neutral";
  }
  return delta > 0 ? "up" : "down";
}

/**
 * Get color classes for a variant.
 */
function getVariantClasses(variant: ColorVariant): {
  border: string;
  icon: string;
  value: string;
} {
  switch (variant) {
    case "success":
      return {
        border: "border-green-500/30",
        icon: "text-green-500",
        value: "text-green-600 dark:text-green-400",
      };
    case "warning":
      return {
        border: "border-yellow-500/30",
        icon: "text-yellow-500",
        value: "text-yellow-600 dark:text-yellow-400",
      };
    case "error":
      return {
        border: "border-red-500/30",
        icon: "text-red-500",
        value: "text-red-600 dark:text-red-400",
      };
    case "info":
      return {
        border: "border-blue-500/30",
        icon: "text-blue-500",
        value: "text-blue-600 dark:text-blue-400",
      };
    case "muted":
      return {
        border: "border-muted",
        icon: "text-muted-foreground",
        value: "text-muted-foreground",
      };
    default:
      return {
        border: "",
        icon: "text-muted-foreground",
        value: "text-foreground",
      };
  }
}

/**
 * Get trend color based on direction and whether positive is good.
 */
function getTrendColor(
  trend: TrendDirection,
  positiveIsGood: boolean = true
): string {
  if (trend === "neutral") {
    return "text-muted-foreground";
  }
  const isGood = (trend === "up") === positiveIsGood;
  return isGood
    ? "text-green-600 dark:text-green-400"
    : "text-red-600 dark:text-red-400";
}

// =============================================================================
// Components
// =============================================================================

/**
 * Loading skeleton for the StatisticsCard.
 */
export function StatisticsCardSkeleton({
  className,
}: {
  className?: string;
}) {
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-4" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-8 w-20 mb-2" />
        <Skeleton className="h-3 w-32" />
      </CardContent>
    </Card>
  );
}

/**
 * Error state for the StatisticsCard.
 */
function StatisticsCardError({
  error,
  label,
  className,
}: {
  error: string;
  label: string;
  className?: string;
}) {
  return (
    <Card className={cn("overflow-hidden border-red-500/30", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
        <AlertTriangle className="h-4 w-4 text-red-500" />
      </CardHeader>
      <CardContent>
        <div className="text-sm text-red-500">{error}</div>
      </CardContent>
    </Card>
  );
}

/**
 * Tooltip content showing full distribution statistics.
 */
function DistributionTooltip({
  stats,
  format,
  precision,
  unit,
}: {
  stats: DistributionStats;
  format?: ValueFormat;
  precision?: number;
  unit?: string;
}) {
  const formatVal = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "—";
    const formatted = formatNumber(val, format, precision);
    return unit ? `${formatted} ${unit}` : formatted;
  };

  return (
    <div className="space-y-2 text-xs">
      <div className="font-medium text-foreground">Distribution Statistics</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <div className="text-muted-foreground">Mean:</div>
        <div className="font-mono">{formatVal(stats.mean)}</div>
        <div className="text-muted-foreground">Median:</div>
        <div className="font-mono">{formatVal(stats.median)}</div>
        <div className="text-muted-foreground">Std Dev:</div>
        <div className="font-mono">{formatVal(stats.std_dev)}</div>
        <div className="text-muted-foreground">Min:</div>
        <div className="font-mono">{formatVal(stats.min_value)}</div>
        <div className="text-muted-foreground">Max:</div>
        <div className="font-mono">{formatVal(stats.max_value)}</div>
      </div>
      {stats.percentiles && (
        <>
          <div className="border-t border-border pt-2 mt-2">
            <div className="font-medium text-foreground mb-1">Percentiles</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <div className="text-muted-foreground">P50:</div>
              <div className="font-mono">{formatVal(stats.percentiles.p50)}</div>
              <div className="text-muted-foreground">P90:</div>
              <div className="font-mono">{formatVal(stats.percentiles.p90)}</div>
              <div className="text-muted-foreground">P95:</div>
              <div className="font-mono">{formatVal(stats.percentiles.p95)}</div>
              <div className="text-muted-foreground">P99:</div>
              <div className="font-mono">{formatVal(stats.percentiles.p99)}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * StatisticsCard Component
 *
 * A reusable card component for displaying statistical metrics with:
 * - Primary value with optional formatting
 * - Delta/change indicator with trend icon
 * - Support for mean, median, p95, std_dev metrics
 * - Loading skeleton and error states
 * - Distribution statistics tooltip
 *
 * @example
 * ```tsx
 * <StatisticsCard
 *   label="Success Rate"
 *   value={0.85}
 *   format="percentage"
 *   delta={0.05}
 *   deltaLabel="vs last period"
 *   icon={TrendingUp}
 *   variant="success"
 * />
 * ```
 *
 * @accessibility
 * - Semantic structure with proper heading hierarchy
 * - ARIA labels for interactive elements
 * - Keyboard navigable when clickable
 * - Screen reader friendly statistics
 */
export function StatisticsCard({
  label,
  value,
  description,
  metricType = "custom",
  format = "number",
  precision = 2,
  formatValue,
  unit,
  icon: Icon,
  delta,
  deltaLabel = "vs previous",
  deltaPositiveIsGood = true,
  trend: trendOverride,
  variant = "default",
  isLoading = false,
  error,
  className,
  distributionStats,
  onClick,
  ariaLabel,
}: StatisticsCardProps) {
  // Loading state
  if (isLoading) {
    return <StatisticsCardSkeleton className={className} />;
  }

  // Error state
  if (error) {
    return (
      <StatisticsCardError
        error={error}
        label={label}
        className={className}
      />
    );
  }

  // Calculate trend
  const trend = trendOverride ?? calculateTrend(delta);
  const trendColor = getTrendColor(trend, deltaPositiveIsGood);
  const variantClasses = getVariantClasses(variant);

  // Format the value
  const formattedValue = formatNumber(value, format, precision, formatValue);
  const displayValue = unit ? `${formattedValue} ${unit}` : formattedValue;

  // Format the delta
  const formattedDelta =
    delta !== null && delta !== undefined
      ? `${delta > 0 ? "+" : ""}${formatNumber(Math.abs(delta), format === "percentage" ? "percentage" : "number", precision)}`
      : null;

  // Render trend icon based on direction
  const renderTrendIcon = () => {
    const iconClassName = "h-3 w-3";
    switch (trend) {
      case "up":
        return <TrendingUp className={iconClassName} aria-hidden="true" />;
      case "down":
        return <TrendingDown className={iconClassName} aria-hidden="true" />;
      default:
        return <Minus className={iconClassName} aria-hidden="true" />;
    }
  };

  // Build the card content
  const cardContent = (
    <Card
      className={cn(
        "overflow-hidden transition-all duration-200",
        variantClasses.border,
        onClick && "cursor-pointer hover:shadow-md hover:border-primary/50",
        className
      )}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      aria-label={ariaLabel ?? `${label}: ${displayValue}`}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
        {Icon && (
          <Icon
            className={cn("h-4 w-4", variantClasses.icon)}
            aria-hidden="true"
          />
        )}
      </CardHeader>
      <CardContent>
        {/* Main Value */}
        <div
          className={cn("text-2xl font-bold tracking-tight", variantClasses.value)}
          aria-live="polite"
        >
          {displayValue}
        </div>

        {/* Delta and Description */}
        <div className="flex items-center gap-2 mt-1">
          {formattedDelta !== null && (
            <div
              className={cn("flex items-center gap-1 text-xs font-medium", trendColor)}
              aria-label={`Change: ${formattedDelta} ${deltaLabel}`}
            >
              {renderTrendIcon()}
              <span>{formattedDelta}</span>
            </div>
          )}
          {(description || deltaLabel) && delta !== undefined && delta !== null && (
            <span className="text-xs text-muted-foreground">
              {description || deltaLabel}
            </span>
          )}
          {description && (delta === undefined || delta === null) && (
            <span className="text-xs text-muted-foreground">{description}</span>
          )}
        </div>

        {/* Metric Type Badge */}
        {metricType !== "custom" && metricType !== "total" && metricType !== "count" && (
          <div className="mt-2">
            <span className="inline-flex items-center px-2 py-0.5 text-[10px] font-medium rounded-full bg-muted text-muted-foreground">
              {metricType.toUpperCase()}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );

  // Wrap with tooltip if distribution stats are provided
  if (distributionStats) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{cardContent}</TooltipTrigger>
          <TooltipContent
            side="bottom"
            className="p-3 max-w-xs"
            aria-label="Distribution statistics details"
          >
            <DistributionTooltip
              stats={distributionStats}
              format={format}
              precision={precision}
              unit={unit}
            />
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return cardContent;
}

/**
 * DistributionStatCard Component
 *
 * A convenience wrapper for StatisticsCard that extracts values from
 * a DistributionStats object.
 *
 * @example
 * ```tsx
 * <DistributionStatCard
 *   label="Success Rate"
 *   stats={campaignStats.success_rate}
 *   primaryMetric="mean"
 *   format="percentage"
 * />
 * ```
 */
export function DistributionStatCard({
  label,
  stats,
  primaryMetric = "mean",
  format = "number",
  unit,
  precision = 2,
  icon,
  isLoading = false,
  error,
  className,
  onClick,
}: DistributionStatCardProps) {
  // Extract the primary value from stats
  const value = React.useMemo(() => {
    if (!stats) return null;

    switch (primaryMetric) {
      case "mean":
        return stats.mean;
      case "median":
        return stats.median;
      case "p95":
        return stats.percentiles?.p95 ?? null;
      case "p99":
        return stats.percentiles?.p99 ?? null;
      default:
        return stats.mean;
    }
  }, [stats, primaryMetric]);

  // Calculate delta as difference between mean and median if available
  const delta = React.useMemo(() => {
    if (!stats || stats.mean === null || stats.median === null) return null;
    if (primaryMetric === "mean" && stats.median !== undefined && stats.mean !== undefined) {
      return stats.mean - stats.median;
    }
    return null;
  }, [stats, primaryMetric]);

  return (
    <StatisticsCard
      label={label}
      value={value}
      metricType={primaryMetric}
      format={format}
      unit={unit}
      precision={precision}
      icon={icon}
      distributionStats={stats}
      isLoading={isLoading}
      error={error}
      className={className}
      onClick={onClick}
    />
  );
}

// =============================================================================
// Preset Cards for Common Metrics
// =============================================================================

/**
 * Pre-configured StatisticsCard for success rate metrics.
 */
export function SuccessRateCard({
  stats,
  delta,
  isLoading,
  error,
  className,
  onClick,
}: {
  stats: DistributionStats | null | undefined;
  delta?: number | null;
  isLoading?: boolean;
  error?: string | null;
  className?: string;
  onClick?: () => void;
}) {
  const value = stats?.mean;
  const variant: ColorVariant =
    value !== null && value !== undefined
      ? value >= 0.7
        ? "success"
        : value >= 0.4
          ? "warning"
          : "error"
      : "default";

  return (
    <StatisticsCard
      label="Success Rate"
      value={value}
      format="percentage"
      precision={1}
      delta={delta}
      deltaPositiveIsGood={true}
      variant={variant}
      distributionStats={stats}
      isLoading={isLoading}
      error={error}
      className={className}
      onClick={onClick}
    />
  );
}

/**
 * Pre-configured StatisticsCard for latency metrics.
 */
export function LatencyCard({
  stats,
  delta,
  isLoading,
  error,
  className,
  onClick,
}: {
  stats: DistributionStats | null | undefined;
  delta?: number | null;
  isLoading?: boolean;
  error?: string | null;
  className?: string;
  onClick?: () => void;
}) {
  const value = stats?.median ?? stats?.mean;
  const p95 = stats?.percentiles?.p95;

  return (
    <StatisticsCard
      label="Latency (p50)"
      value={value}
      description={p95 !== null && p95 !== undefined ? `P95: ${formatNumber(p95, "duration", 0)}` : undefined}
      format="duration"
      precision={0}
      delta={delta}
      deltaPositiveIsGood={false}
      distributionStats={stats}
      isLoading={isLoading}
      error={error}
      className={className}
      onClick={onClick}
    />
  );
}

/**
 * Pre-configured StatisticsCard for token usage metrics.
 */
export function TokenUsageCard({
  stats,
  total,
  isLoading,
  error,
  className,
  onClick,
}: {
  stats: DistributionStats | null | undefined;
  total?: number;
  isLoading?: boolean;
  error?: string | null;
  className?: string;
  onClick?: () => void;
}) {
  return (
    <StatisticsCard
      label="Avg Tokens"
      value={stats?.mean}
      description={total !== undefined ? `Total: ${formatNumber(total, "compact", 0)}` : undefined}
      format="compact"
      precision={0}
      unit="tokens"
      distributionStats={stats}
      isLoading={isLoading}
      error={error}
      className={className}
      onClick={onClick}
    />
  );
}

/**
 * Pre-configured StatisticsCard for cost metrics.
 */
export function CostCard({
  stats,
  totalCents,
  isLoading,
  error,
  className,
  onClick,
}: {
  stats: DistributionStats | null | undefined;
  totalCents?: number;
  isLoading?: boolean;
  error?: string | null;
  className?: string;
  onClick?: () => void;
}) {
  return (
    <StatisticsCard
      label="Avg Cost"
      value={stats?.mean}
      description={
        totalCents !== undefined
          ? `Total: ${formatNumber(totalCents, "currency", 2)}`
          : undefined
      }
      format="currency"
      precision={2}
      distributionStats={stats}
      isLoading={isLoading}
      error={error}
      className={className}
      onClick={onClick}
    />
  );
}

export default StatisticsCard;
