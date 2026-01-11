"use client";

import * as React from "react";
import {
  ChevronDown,
  ChevronRight,
  Crown,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type {
  CampaignComparisonItem,
  CampaignComparison,
} from "@/types/campaign-analytics";

// =============================================================================
// Types
// =============================================================================

/**
 * Metric value type enum for determining best/worst highlighting behavior.
 */
export type MetricDirection = "higher_is_better" | "lower_is_better" | "neutral";

/**
 * Display format for metric values.
 */
export type MetricFormat =
  | "number"
  | "percentage"
  | "duration"
  | "currency"
  | "compact"
  | "string";

/**
 * Single metric row configuration.
 */
export interface MetricRowConfig {
  /** Unique identifier for the metric */
  id: string;
  /** Display label for the metric */
  label: string;
  /** Description shown in tooltip */
  description?: string;
  /** Which direction is better */
  direction: MetricDirection;
  /** Display format */
  format: MetricFormat;
  /** Number of decimal places */
  precision?: number;
  /** Unit suffix */
  unit?: string;
  /** Key path in CampaignComparisonItem (or getter function) */
  getValue: (item: CampaignComparisonItem) => number | string | null | undefined;
  /** Optional icon to display */
  icon?: LucideIcon;
  /** Group this metric belongs to (for expandable sections) */
  group?: string;
  /** Is this a summary metric (shown at top level) */
  isSummary?: boolean;
}

/**
 * Metric group configuration for expandable sections.
 */
export interface MetricGroupConfig {
  /** Unique identifier */
  id: string;
  /** Display label */
  label: string;
  /** Description */
  description?: string;
  /** Icon for the group header */
  icon?: LucideIcon;
  /** Default expanded state */
  defaultExpanded?: boolean;
}

/**
 * Props for the ComparisonTable component.
 */
export interface ComparisonTableProps {
  /** Comparison data from API */
  comparison: CampaignComparison | null | undefined;
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: string | null;
  /** Retry callback for error state */
  onRetry?: () => void;
  /** Custom metric configurations (overrides defaults) */
  metricConfigs?: MetricRowConfig[];
  /** Custom group configurations */
  groupConfigs?: MetricGroupConfig[];
  /** Show rank badges (1st, 2nd, etc.) */
  showRankBadges?: boolean;
  /** Highlight best values */
  highlightBest?: boolean;
  /** Highlight worst values */
  highlightWorst?: boolean;
  /** Show delta values relative to first campaign */
  showDeltas?: boolean;
  /** Enable expandable row sections */
  enableExpanding?: boolean;
  /** Default expanded groups */
  defaultExpandedGroups?: string[];
  /** Compact mode (less padding) */
  compact?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Click handler for campaign column header */
  onCampaignClick?: (campaignId: string) => void;
}

// =============================================================================
// Constants
// =============================================================================

/**
 * Default metric configurations for comparison table.
 */
const DEFAULT_METRIC_CONFIGS: MetricRowConfig[] = [
  // Summary metrics
  {
    id: "success_rate",
    label: "Success Rate",
    description: "Overall success rate of the campaign",
    direction: "higher_is_better",
    format: "percentage",
    precision: 1,
    getValue: (item) => item.success_rate,
    isSummary: true,
  },
  {
    id: "total_attempts",
    label: "Total Attempts",
    description: "Total number of prompt attempts",
    direction: "neutral",
    format: "compact",
    getValue: (item) => item.total_attempts,
    isSummary: true,
  },
  {
    id: "latency_mean",
    label: "Avg Latency",
    description: "Mean latency across all attempts",
    direction: "lower_is_better",
    format: "duration",
    precision: 0,
    getValue: (item) => item.latency_mean,
    isSummary: true,
  },
  {
    id: "total_cost_cents",
    label: "Total Cost",
    description: "Total cost in cents",
    direction: "lower_is_better",
    format: "currency",
    precision: 2,
    getValue: (item) => item.total_cost_cents,
    isSummary: true,
  },

  // Performance metrics (expandable group)
  {
    id: "semantic_success_mean",
    label: "Semantic Success",
    description: "Mean semantic success score",
    direction: "higher_is_better",
    format: "percentage",
    precision: 1,
    getValue: (item) => item.semantic_success_mean,
    group: "performance",
  },
  {
    id: "latency_p95",
    label: "P95 Latency",
    description: "95th percentile latency",
    direction: "lower_is_better",
    format: "duration",
    precision: 0,
    getValue: (item) => item.latency_p95,
    group: "performance",
  },
  {
    id: "duration_seconds",
    label: "Total Duration",
    description: "Total campaign duration",
    direction: "neutral",
    format: "duration",
    precision: 0,
    unit: "s",
    getValue: (item) =>
      item.duration_seconds ? item.duration_seconds * 1000 : null,
    group: "performance",
  },

  // Token usage metrics (expandable group)
  {
    id: "avg_tokens",
    label: "Avg Tokens/Request",
    description: "Average tokens per request",
    direction: "lower_is_better",
    format: "compact",
    getValue: (item) => item.avg_tokens,
    group: "tokens",
  },
  {
    id: "total_tokens",
    label: "Total Tokens",
    description: "Total tokens used in campaign",
    direction: "neutral",
    format: "compact",
    getValue: (item) => item.total_tokens,
    group: "tokens",
  },

  // Cost metrics (expandable group)
  {
    id: "avg_cost_per_attempt",
    label: "Cost per Attempt",
    description: "Average cost per attempt in cents",
    direction: "lower_is_better",
    format: "currency",
    precision: 3,
    getValue: (item) => item.avg_cost_per_attempt,
    group: "cost",
  },

  // Strategy metrics (expandable group)
  {
    id: "best_technique",
    label: "Best Technique",
    description: "Most effective transformation technique",
    direction: "neutral",
    format: "string",
    getValue: (item) => item.best_technique ?? "—",
    group: "strategy",
  },
  {
    id: "best_provider",
    label: "Best Provider",
    description: "Most effective LLM provider",
    direction: "neutral",
    format: "string",
    getValue: (item) => item.best_provider ?? "—",
    group: "strategy",
  },

  // Normalized metrics (expandable group)
  {
    id: "normalized_success_rate",
    label: "Normalized Success",
    description: "Normalized success rate (0-1 scale)",
    direction: "higher_is_better",
    format: "percentage",
    precision: 0,
    getValue: (item) => item.normalized_success_rate,
    group: "normalized",
  },
  {
    id: "normalized_latency",
    label: "Normalized Latency",
    description: "Normalized latency score (0-1 scale, lower is better)",
    direction: "lower_is_better",
    format: "percentage",
    precision: 0,
    getValue: (item) => item.normalized_latency,
    group: "normalized",
  },
  {
    id: "normalized_cost",
    label: "Normalized Cost",
    description: "Normalized cost score (0-1 scale, lower is better)",
    direction: "lower_is_better",
    format: "percentage",
    precision: 0,
    getValue: (item) => item.normalized_cost,
    group: "normalized",
  },
  {
    id: "normalized_effectiveness",
    label: "Normalized Effectiveness",
    description: "Overall normalized effectiveness score",
    direction: "higher_is_better",
    format: "percentage",
    precision: 0,
    getValue: (item) => item.normalized_effectiveness,
    group: "normalized",
  },
];

/**
 * Default metric group configurations.
 */
const DEFAULT_GROUP_CONFIGS: MetricGroupConfig[] = [
  {
    id: "performance",
    label: "Performance Metrics",
    description: "Latency and timing statistics",
    defaultExpanded: true,
  },
  {
    id: "tokens",
    label: "Token Usage",
    description: "Token consumption statistics",
    defaultExpanded: false,
  },
  {
    id: "cost",
    label: "Cost Analysis",
    description: "Cost breakdown and efficiency",
    defaultExpanded: false,
  },
  {
    id: "strategy",
    label: "Strategy Insights",
    description: "Best performing techniques and providers",
    defaultExpanded: true,
  },
  {
    id: "normalized",
    label: "Normalized Scores",
    description: "Normalized metrics for radar chart visualization",
    defaultExpanded: false,
  },
];

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format a value based on the specified format type.
 */
function formatValue(
  value: number | string | null | undefined,
  format: MetricFormat,
  precision: number = 2,
  unit?: string
): string {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "string") {
    return value;
  }

  let formatted: string;
  switch (format) {
    case "percentage":
      // Handle both 0-1 and 0-100 ranges
      const percentValue = value <= 1 ? value * 100 : value;
      formatted = `${percentValue.toFixed(precision)}%`;
      break;
    case "duration":
      if (value >= 1000) {
        formatted = `${(value / 1000).toFixed(precision)}s`;
      } else {
        formatted = `${value.toFixed(precision)}ms`;
      }
      break;
    case "currency":
      if (value >= 100) {
        formatted = `$${(value / 100).toFixed(2)}`;
      } else {
        formatted = `${value.toFixed(precision)}¢`;
      }
      break;
    case "compact":
      if (Math.abs(value) >= 1000000) {
        formatted = `${(value / 1000000).toFixed(1)}M`;
      } else if (Math.abs(value) >= 1000) {
        formatted = `${(value / 1000).toFixed(1)}K`;
      } else {
        formatted = value.toFixed(0);
      }
      break;
    default:
      formatted = value.toFixed(precision);
  }

  return unit ? `${formatted} ${unit}` : formatted;
}

/**
 * Calculate delta between two values.
 */
function calculateDelta(
  current: number | null | undefined,
  baseline: number | null | undefined
): number | null {
  if (
    current === null ||
    current === undefined ||
    baseline === null ||
    baseline === undefined ||
    baseline === 0
  ) {
    return null;
  }
  return ((current - baseline) / Math.abs(baseline)) * 100;
}

/**
 * Get rank of a value among a list of values.
 */
function getRank(
  value: number | null | undefined,
  allValues: (number | null | undefined)[],
  direction: MetricDirection
): number {
  if (value === null || value === undefined) {
    return allValues.length;
  }

  const sortedValues = allValues
    .filter((v): v is number => v !== null && v !== undefined)
    .sort((a, b) =>
      direction === "higher_is_better" ? b - a : a - b
    );

  return sortedValues.indexOf(value) + 1;
}

/**
 * Determine if a value is the best among all values.
 */
function isBestValue(
  value: number | null | undefined,
  allValues: (number | null | undefined)[],
  direction: MetricDirection
): boolean {
  if (direction === "neutral" || value === null || value === undefined) {
    return false;
  }

  const validValues = allValues.filter(
    (v): v is number => v !== null && v !== undefined
  );

  if (validValues.length === 0) return false;

  if (direction === "higher_is_better") {
    return value === Math.max(...validValues);
  } else {
    return value === Math.min(...validValues);
  }
}

/**
 * Determine if a value is the worst among all values.
 */
function isWorstValue(
  value: number | null | undefined,
  allValues: (number | null | undefined)[],
  direction: MetricDirection
): boolean {
  if (direction === "neutral" || value === null || value === undefined) {
    return false;
  }

  const validValues = allValues.filter(
    (v): v is number => v !== null && v !== undefined
  );

  if (validValues.length <= 1) return false;

  if (direction === "higher_is_better") {
    return value === Math.min(...validValues);
  } else {
    return value === Math.max(...validValues);
  }
}

/**
 * Get color classes for delta value.
 */
function getDeltaColor(
  delta: number,
  direction: MetricDirection
): string {
  if (direction === "neutral") {
    return "text-muted-foreground";
  }

  const isPositive = delta > 0;
  const isGood =
    (direction === "higher_is_better" && isPositive) ||
    (direction === "lower_is_better" && !isPositive);

  return isGood
    ? "text-green-600 dark:text-green-400"
    : "text-red-600 dark:text-red-400";
}

/**
 * Get trend icon for delta value.
 */
function getTrendIcon(delta: number): LucideIcon {
  if (Math.abs(delta) < 0.5) {
    return Minus;
  }
  return delta > 0 ? TrendingUp : TrendingDown;
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Rank badge component.
 */
function RankBadge({ rank, total }: { rank: number; total: number }) {
  if (rank > total) return null;

  const getRankStyle = () => {
    switch (rank) {
      case 1:
        return "bg-yellow-500/20 text-yellow-700 dark:text-yellow-400 border-yellow-500/30";
      case 2:
        return "bg-slate-400/20 text-slate-600 dark:text-slate-400 border-slate-400/30";
      case 3:
        return "bg-amber-600/20 text-amber-700 dark:text-amber-500 border-amber-600/30";
      default:
        return "bg-muted text-muted-foreground border-border";
    }
  };

  return (
    <Badge
      variant="outline"
      className={cn("ml-2 text-[10px] px-1.5 py-0", getRankStyle())}
    >
      {rank === 1 ? (
        <Crown className="h-3 w-3 mr-0.5" />
      ) : null}
      {rank}
    </Badge>
  );
}

/**
 * Delta indicator component.
 */
function DeltaIndicator({
  delta,
  direction,
  showIcon = true,
}: {
  delta: number | null;
  direction: MetricDirection;
  showIcon?: boolean;
}) {
  if (delta === null) return null;

  const TrendIcon = getTrendIcon(delta);
  const colorClass = getDeltaColor(delta, direction);
  const formattedDelta = `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%`;

  return (
    <span className={cn("flex items-center gap-0.5 text-xs", colorClass)}>
      {showIcon && <TrendIcon className="h-3 w-3" />}
      {formattedDelta}
    </span>
  );
}

/**
 * Value cell with highlighting and optional delta.
 */
function ValueCell({
  value,
  formattedValue,
  isBest,
  isWorst,
  highlightBest,
  highlightWorst,
  rank,
  totalCampaigns,
  showRankBadges,
  delta,
  direction,
  showDelta,
}: {
  value: number | string | null | undefined;
  formattedValue: string;
  isBest: boolean;
  isWorst: boolean;
  highlightBest: boolean;
  highlightWorst: boolean;
  rank: number;
  totalCampaigns: number;
  showRankBadges: boolean;
  delta: number | null;
  direction: MetricDirection;
  showDelta: boolean;
}) {
  const getCellClasses = () => {
    const classes = ["font-mono", "tabular-nums"];

    if (highlightBest && isBest) {
      classes.push(
        "text-green-600 dark:text-green-400",
        "font-semibold",
        "bg-green-500/10",
        "rounded-md",
        "px-2",
        "py-1"
      );
    } else if (highlightWorst && isWorst) {
      classes.push(
        "text-red-600 dark:text-red-400",
        "bg-red-500/10",
        "rounded-md",
        "px-2",
        "py-1"
      );
    }

    return classes.join(" ");
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center">
        <span className={getCellClasses()}>{formattedValue}</span>
        {showRankBadges && typeof value === "number" && direction !== "neutral" && (
          <RankBadge rank={rank} total={totalCampaigns} />
        )}
      </div>
      {showDelta && delta !== null && (
        <DeltaIndicator delta={delta} direction={direction} />
      )}
    </div>
  );
}

/**
 * Metric row component (single row in the table).
 */
function MetricRow({
  config,
  campaigns,
  highlightBest,
  highlightWorst,
  showRankBadges,
  showDeltas,
  compact,
}: {
  config: MetricRowConfig;
  campaigns: CampaignComparisonItem[];
  highlightBest: boolean;
  highlightWorst: boolean;
  showRankBadges: boolean;
  showDeltas: boolean;
  compact: boolean;
}) {
  // Extract values for all campaigns
  const values = campaigns.map((campaign) => config.getValue(campaign));
  const numericValues = values.filter(
    (v): v is number => typeof v === "number"
  );
  const baselineValue = typeof values[0] === "number" ? values[0] : null;

  const Icon = config.icon;

  return (
    <TableRow
      className={cn(
        "transition-colors hover:bg-muted/50",
        compact && "h-10"
      )}
    >
      <TableCell
        className={cn(
          "font-medium",
          compact ? "py-2" : "py-3"
        )}
      >
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger className="flex items-center gap-2 cursor-help">
              {Icon && (
                <Icon className="h-4 w-4 text-muted-foreground" />
              )}
              <span>{config.label}</span>
            </TooltipTrigger>
            {config.description && (
              <TooltipContent side="right" className="max-w-xs">
                <p>{config.description}</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      </TableCell>
      {campaigns.map((campaign, idx) => {
        const value = values[idx];
        const numericValue = typeof value === "number" ? value : null;
        const formattedValue = formatValue(
          value,
          config.format,
          config.precision,
          config.unit
        );
        const isBest = isBestValue(numericValue, numericValues, config.direction);
        const isWorst = isWorstValue(numericValue, numericValues, config.direction);
        const rank = getRank(numericValue, numericValues, config.direction);
        const delta =
          idx > 0 && showDeltas
            ? calculateDelta(numericValue, baselineValue)
            : null;

        return (
          <TableCell
            key={campaign.campaign_id}
            className={cn(
              "text-center",
              compact ? "py-2" : "py-3"
            )}
          >
            <ValueCell
              value={value}
              formattedValue={formattedValue}
              isBest={isBest}
              isWorst={isWorst}
              highlightBest={highlightBest}
              highlightWorst={highlightWorst}
              rank={rank}
              totalCampaigns={campaigns.length}
              showRankBadges={showRankBadges}
              delta={delta}
              direction={config.direction}
              showDelta={showDeltas}
            />
          </TableCell>
        );
      })}
    </TableRow>
  );
}

/**
 * Expandable metric group component.
 */
function MetricGroup({
  group,
  metrics,
  campaigns,
  highlightBest,
  highlightWorst,
  showRankBadges,
  showDeltas,
  compact,
  defaultExpanded,
}: {
  group: MetricGroupConfig;
  metrics: MetricRowConfig[];
  campaigns: CampaignComparisonItem[];
  highlightBest: boolean;
  highlightWorst: boolean;
  showRankBadges: boolean;
  showDeltas: boolean;
  compact: boolean;
  defaultExpanded: boolean;
}) {
  const [isOpen, setIsOpen] = React.useState(defaultExpanded);
  const Icon = group.icon;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <TableRow className="bg-muted/30 hover:bg-muted/50">
        <TableCell colSpan={campaigns.length + 1} className="py-0">
          <CollapsibleTrigger className="flex items-center gap-2 w-full py-2.5 cursor-pointer select-none">
            {isOpen ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
            {Icon && <Icon className="h-4 w-4 text-muted-foreground" />}
            <span className="font-medium text-sm">{group.label}</span>
            {group.description && (
              <span className="text-xs text-muted-foreground ml-2">
                — {group.description}
              </span>
            )}
            <Badge variant="secondary" className="ml-auto text-xs">
              {metrics.length} metrics
            </Badge>
          </CollapsibleTrigger>
        </TableCell>
      </TableRow>
      <CollapsibleContent asChild>
        <tbody>
          {metrics.map((metric) => (
            <MetricRow
              key={metric.id}
              config={metric}
              campaigns={campaigns}
              highlightBest={highlightBest}
              highlightWorst={highlightWorst}
              showRankBadges={showRankBadges}
              showDeltas={showDeltas}
              compact={compact}
            />
          ))}
        </tbody>
      </CollapsibleContent>
    </Collapsible>
  );
}

/**
 * Campaign column header with status badge.
 */
function CampaignHeader({
  campaign,
  isBaseline,
  onClick,
}: {
  campaign: CampaignComparisonItem;
  isBaseline: boolean;
  onClick?: () => void;
}) {
  const getStatusColor = () => {
    switch (campaign.status) {
      case "completed":
        return "bg-green-500/20 text-green-700 dark:text-green-400";
      case "running":
        return "bg-blue-500/20 text-blue-700 dark:text-blue-400";
      case "failed":
        return "bg-red-500/20 text-red-700 dark:text-red-400";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  return (
    <div
      className={cn(
        "flex flex-col items-center gap-1",
        onClick && "cursor-pointer hover:opacity-80"
      )}
      onClick={onClick}
    >
      <span className="font-semibold text-sm truncate max-w-[150px]">
        {campaign.campaign_name}
      </span>
      <div className="flex items-center gap-1">
        <Badge variant="outline" className={cn("text-[10px]", getStatusColor())}>
          {campaign.status}
        </Badge>
        {isBaseline && (
          <Badge variant="secondary" className="text-[10px]">
            Baseline
          </Badge>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Loading Skeleton
// =============================================================================

/**
 * Loading skeleton for ComparisonTable.
 */
export function ComparisonTableSkeleton({
  campaignCount = 3,
  rowCount = 6,
  className,
}: {
  campaignCount?: number;
  rowCount?: number;
  className?: string;
}) {
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="flex flex-row items-center justify-between pb-4">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-8 w-24" />
      </CardHeader>
      <CardContent className="pt-0">
        <div className="border rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>
                  <Skeleton className="h-4 w-20" />
                </TableHead>
                {Array.from({ length: campaignCount }).map((_, i) => (
                  <TableHead key={i} className="text-center">
                    <div className="flex flex-col items-center gap-1">
                      <Skeleton className="h-4 w-24" />
                      <Skeleton className="h-4 w-16" />
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {Array.from({ length: rowCount }).map((_, rowIdx) => (
                <TableRow key={rowIdx}>
                  <TableCell>
                    <Skeleton className="h-4 w-28" />
                  </TableCell>
                  {Array.from({ length: campaignCount }).map((_, colIdx) => (
                    <TableCell key={colIdx} className="text-center">
                      <div className="flex flex-col items-center gap-1">
                        <Skeleton className="h-5 w-16" />
                        <Skeleton className="h-3 w-12" />
                      </div>
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Error State
// =============================================================================

/**
 * Error state for ComparisonTable.
 */
export function ComparisonTableError({
  error,
  onRetry,
  className,
}: {
  error: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <Card className={cn("overflow-hidden border-red-500/30", className)}>
      <CardContent className="flex flex-col items-center justify-center py-12 gap-4">
        <AlertTriangle className="h-12 w-12 text-red-500" />
        <div className="text-center">
          <h3 className="font-semibold text-lg">Comparison Error</h3>
          <p className="text-sm text-muted-foreground mt-1">{error}</p>
        </div>
        {onRetry && (
          <Button
            variant="outline"
            onClick={onRetry}
            className="gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Try Again
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Empty State
// =============================================================================

/**
 * Empty state for ComparisonTable.
 */
export function ComparisonTableEmpty({
  className,
}: {
  className?: string;
}) {
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardContent className="flex flex-col items-center justify-center py-12 gap-4">
        <div className="p-4 rounded-full bg-muted">
          <Table className="h-8 w-8 text-muted-foreground" />
        </div>
        <div className="text-center">
          <h3 className="font-semibold text-lg">No Campaigns to Compare</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Select at least 2 campaigns to see the comparison table.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * ComparisonTable Component
 *
 * Table component showing metrics rows with campaign columns for comparing
 * multiple campaigns side-by-side. Highlights best/worst values and supports
 * expanding rows for detailed breakdown by metric category.
 *
 * @example
 * ```tsx
 * <ComparisonTable
 *   comparison={comparisonData}
 *   showRankBadges
 *   highlightBest
 *   highlightWorst
 *   showDeltas
 *   enableExpanding
 * />
 * ```
 *
 * @accessibility
 * - Uses semantic table structure
 * - Keyboard navigable expandable sections
 * - Screen reader friendly with proper ARIA attributes
 * - Color-blind safe highlighting with additional visual cues
 */
export function ComparisonTable({
  comparison,
  isLoading = false,
  error,
  onRetry,
  metricConfigs = DEFAULT_METRIC_CONFIGS,
  groupConfigs = DEFAULT_GROUP_CONFIGS,
  showRankBadges = true,
  highlightBest = true,
  highlightWorst = true,
  showDeltas = true,
  enableExpanding = true,
  defaultExpandedGroups = ["performance", "strategy"],
  compact = false,
  className,
  onCampaignClick,
}: ComparisonTableProps) {
  // Handle loading state
  if (isLoading) {
    return (
      <ComparisonTableSkeleton
        campaignCount={3}
        rowCount={6}
        className={className}
      />
    );
  }

  // Handle error state
  if (error) {
    return (
      <ComparisonTableError
        error={error}
        onRetry={onRetry}
        className={className}
      />
    );
  }

  // Handle empty state
  if (!comparison || comparison.campaigns.length < 2) {
    return <ComparisonTableEmpty className={className} />;
  }

  const { campaigns } = comparison;

  // Separate summary metrics from grouped metrics
  const summaryMetrics = metricConfigs.filter((m) => m.isSummary);
  const groupedMetrics = metricConfigs.filter((m) => m.group && !m.isSummary);

  // Group metrics by their group ID
  const metricsByGroup = groupConfigs.reduce((acc, group) => {
    acc[group.id] = groupedMetrics.filter((m) => m.group === group.id);
    return acc;
  }, {} as Record<string, MetricRowConfig[]>);

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="flex flex-row items-center justify-between pb-4">
        <CardTitle className="text-lg font-semibold">
          Campaign Comparison
        </CardTitle>
        {comparison.compared_at && (
          <span className="text-xs text-muted-foreground">
            Compared at{" "}
            {new Date(comparison.compared_at).toLocaleString()}
          </span>
        )}
      </CardHeader>
      <CardContent className="pt-0">
        <div className="border rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead className="w-[200px] font-semibold">
                  Metric
                </TableHead>
                {campaigns.map((campaign, idx) => (
                  <TableHead
                    key={campaign.campaign_id}
                    className="text-center min-w-[150px]"
                  >
                    <CampaignHeader
                      campaign={campaign}
                      isBaseline={idx === 0}
                      onClick={
                        onCampaignClick
                          ? () => onCampaignClick(campaign.campaign_id)
                          : undefined
                      }
                    />
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {/* Summary Metrics (always visible) */}
              {summaryMetrics.map((metric) => (
                <MetricRow
                  key={metric.id}
                  config={metric}
                  campaigns={campaigns}
                  highlightBest={highlightBest}
                  highlightWorst={highlightWorst}
                  showRankBadges={showRankBadges}
                  showDeltas={showDeltas}
                  compact={compact}
                />
              ))}

              {/* Grouped Metrics (expandable sections) */}
              {enableExpanding
                ? groupConfigs
                    .filter((group) => metricsByGroup[group.id]?.length > 0)
                    .map((group) => (
                      <MetricGroup
                        key={group.id}
                        group={group}
                        metrics={metricsByGroup[group.id]}
                        campaigns={campaigns}
                        highlightBest={highlightBest}
                        highlightWorst={highlightWorst}
                        showRankBadges={showRankBadges}
                        showDeltas={showDeltas}
                        compact={compact}
                        defaultExpanded={
                          group.defaultExpanded ??
                          defaultExpandedGroups.includes(group.id)
                        }
                      />
                    ))
                : // Flat rendering when expanding is disabled
                  groupedMetrics.map((metric) => (
                    <MetricRow
                      key={metric.id}
                      config={metric}
                      campaigns={campaigns}
                      highlightBest={highlightBest}
                      highlightWorst={highlightWorst}
                      showRankBadges={showRankBadges}
                      showDeltas={showDeltas}
                      compact={compact}
                    />
                  ))}
            </TableBody>
          </Table>
        </div>

        {/* Winner Summary */}
        {(comparison.best_success_rate_campaign ||
          comparison.best_latency_campaign ||
          comparison.best_cost_efficiency_campaign) && (
          <div className="mt-4 flex flex-wrap gap-4">
            {comparison.best_success_rate_campaign && (
              <WinnerBadge
                label="Highest Success Rate"
                campaignId={comparison.best_success_rate_campaign}
                campaigns={campaigns}
              />
            )}
            {comparison.best_latency_campaign && (
              <WinnerBadge
                label="Lowest Latency"
                campaignId={comparison.best_latency_campaign}
                campaigns={campaigns}
              />
            )}
            {comparison.best_cost_efficiency_campaign && (
              <WinnerBadge
                label="Best Cost Efficiency"
                campaignId={comparison.best_cost_efficiency_campaign}
                campaigns={campaigns}
              />
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Winner badge component for summary section.
 */
function WinnerBadge({
  label,
  campaignId,
  campaigns,
}: {
  label: string;
  campaignId: string;
  campaigns: CampaignComparisonItem[];
}) {
  const campaign = campaigns.find((c) => c.campaign_id === campaignId);
  if (!campaign) return null;

  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
      <Crown className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
      <div className="flex flex-col">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-sm font-medium">{campaign.campaign_name}</span>
      </div>
    </div>
  );
}

// =============================================================================
// Convenience Variants
// =============================================================================

/**
 * Simple comparison table without expandable sections.
 */
export function SimpleComparisonTable(
  props: Omit<ComparisonTableProps, "enableExpanding">
) {
  return <ComparisonTable {...props} enableExpanding={false} />;
}

/**
 * Compact comparison table with reduced padding.
 */
export function CompactComparisonTable(
  props: Omit<ComparisonTableProps, "compact">
) {
  return <ComparisonTable {...props} compact />;
}

/**
 * Minimal comparison table showing only summary metrics.
 */
export function SummaryComparisonTable(
  props: Omit<ComparisonTableProps, "enableExpanding" | "metricConfigs">
) {
  const summaryOnlyConfigs = DEFAULT_METRIC_CONFIGS.filter((m) => m.isSummary);
  return (
    <ComparisonTable
      {...props}
      metricConfigs={summaryOnlyConfigs}
      enableExpanding={false}
    />
  );
}

export default ComparisonTable;
