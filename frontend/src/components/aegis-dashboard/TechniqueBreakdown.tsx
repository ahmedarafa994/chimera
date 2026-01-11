/**
 * TechniqueBreakdown Component
 *
 * Displays performance metrics for each transformation technique in the Aegis Campaign Dashboard:
 * - Bar chart (horizontal) showing success rate per technique
 * - Metrics: success rate, average score, attempt count
 * - Sortable by success rate or usage count
 * - Color-coded technique categories (AutoDAN, GPTFuzz, Chimera framing, etc.)
 *
 * Follows glass morphism styling pattern from existing components.
 */

"use client";

import { memo, useMemo, useState, useCallback } from "react";
import {
  Layers,
  Target,
  Zap,
  TrendingUp,
  Clock,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import {
  TechniquePerformance,
  TechniqueCategory,
  TECHNIQUE_CATEGORY_LABELS,
  TECHNIQUE_CATEGORY_COLORS,
} from "@/types/aegis-telemetry";
import {
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";

// ============================================================================
// Types
// ============================================================================

export interface TechniqueBreakdownProps {
  /** Array of technique performance data */
  techniques: TechniquePerformance[];
  /** Whether the component is in loading state */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Maximum number of techniques to display (default: 10) */
  maxDisplay?: number;
  /** Whether to show the chart (default: true) */
  showChart?: boolean;
  /** Whether to show in compact mode */
  compact?: boolean;
  /** Initial sort field */
  defaultSortBy?: SortField;
  /** Initial sort direction */
  defaultSortDirection?: SortDirection;
}

type SortField = "success_rate" | "total_applications" | "avg_score" | "best_score";
type SortDirection = "asc" | "desc";

interface SortOption {
  field: SortField;
  label: string;
  icon: typeof Target;
}

// ============================================================================
// Configuration
// ============================================================================

const SORT_OPTIONS: SortOption[] = [
  { field: "success_rate", label: "Success Rate", icon: Target },
  { field: "total_applications", label: "Usage Count", icon: Layers },
  { field: "avg_score", label: "Avg Score", icon: TrendingUp },
  { field: "best_score", label: "Best Score", icon: Zap },
];

/**
 * Get technique category color
 */
function getTechniqueColor(category: TechniqueCategory): string {
  return TECHNIQUE_CATEGORY_COLORS[category] || TECHNIQUE_CATEGORY_COLORS[TechniqueCategory.OTHER];
}

/**
 * Get technique category label
 */
function getTechniqueLabel(category: TechniqueCategory): string {
  return TECHNIQUE_CATEGORY_LABELS[category] || TECHNIQUE_CATEGORY_LABELS[TechniqueCategory.OTHER];
}

/**
 * Get success rate color class based on threshold
 */
function getSuccessRateColorClass(rate: number): string {
  if (rate >= 70) return "text-emerald-400";
  if (rate >= 40) return "text-amber-400";
  return "text-red-400";
}

/**
 * Get success rate background class based on threshold
 */
function getSuccessRateBgClass(rate: number): string {
  if (rate >= 70) return "bg-emerald-500";
  if (rate >= 40) return "bg-amber-500";
  return "bg-red-500";
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Sort techniques by specified field and direction
 */
function sortTechniques(
  techniques: TechniquePerformance[],
  field: SortField,
  direction: SortDirection
): TechniquePerformance[] {
  return [...techniques].sort((a, b) => {
    const aVal = a[field];
    const bVal = b[field];
    const comparison = aVal - bVal;
    return direction === "desc" ? -comparison : comparison;
  });
}

/**
 * Format percentage for display
 */
function formatPercentage(value: number, decimals: number = 1): string {
  return value.toFixed(decimals);
}

/**
 * Format number with thousands separator
 */
function formatNumber(value: number): string {
  return value.toLocaleString();
}

/**
 * Format execution time
 */
function formatExecutionTime(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Header with title and sort controls
 */
const ChartHeader = memo(function ChartHeader({
  totalTechniques,
  sortBy,
  sortDirection,
  onSortChange,
  onDirectionToggle,
  compact,
}: {
  totalTechniques: number;
  sortBy: SortField;
  sortDirection: SortDirection;
  onSortChange: (field: SortField) => void;
  onDirectionToggle: () => void;
  compact?: boolean;
}) {
  const currentSort = SORT_OPTIONS.find((opt) => opt.field === sortBy) || SORT_OPTIONS[0];
  const SortIcon = currentSort.icon;

  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <Layers
          className="h-4 w-4 text-violet-400"
          aria-hidden="true"
        />
        <h3 className="text-sm font-medium text-foreground">
          Technique Performance
        </h3>
        <Badge variant="secondary" className="text-xs">
          {totalTechniques} techniques
        </Badge>
      </div>
      {!compact && (
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 text-xs bg-white/5 border-white/10 hover:bg-white/10"
              >
                <SortIcon className="h-3 w-3" aria-hidden="true" />
                <span>{currentSort.label}</span>
                <ChevronDown className="h-3 w-3 opacity-50" aria-hidden="true" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-40">
              {SORT_OPTIONS.map((option) => {
                const Icon = option.icon;
                return (
                  <DropdownMenuItem
                    key={option.field}
                    onClick={() => onSortChange(option.field)}
                    className={cn(
                      "gap-2",
                      sortBy === option.field && "bg-white/10"
                    )}
                  >
                    <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>{option.label}</span>
                  </DropdownMenuItem>
                );
              })}
            </DropdownMenuContent>
          </DropdownMenu>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0 bg-white/5 border-white/10 hover:bg-white/10"
                  onClick={onDirectionToggle}
                >
                  {sortDirection === "desc" ? (
                    <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
                  ) : (
                    <ChevronUp className="h-3.5 w-3.5" aria-hidden="true" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{sortDirection === "desc" ? "Highest first" : "Lowest first"}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}
    </div>
  );
});

/**
 * Individual technique row with metrics
 */
const TechniqueRow = memo(function TechniqueRow({
  technique,
  maxApplications,
  compact,
}: {
  technique: TechniquePerformance;
  maxApplications: number;
  compact?: boolean;
}) {
  const categoryColor = getTechniqueColor(technique.technique_category);
  const categoryLabel = getTechniqueLabel(technique.technique_category);
  const successRateColorClass = getSuccessRateColorClass(technique.success_rate);
  const progressValue = (technique.total_applications / maxApplications) * 100;

  return (
    <div
      className={cn(
        "group rounded-lg p-3 transition-all duration-200",
        "bg-white/[0.02] hover:bg-white/[0.05]",
        "border border-white/[0.03] hover:border-white/[0.08]"
      )}
    >
      {/* Technique name and category */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="h-2 w-2 rounded-full flex-shrink-0"
            style={{ backgroundColor: categoryColor }}
            aria-hidden="true"
          />
          <span className="text-sm font-medium text-foreground truncate">
            {technique.technique_name}
          </span>
          <Badge
            variant="outline"
            className="text-[10px] px-1.5 py-0 flex-shrink-0"
            style={{
              backgroundColor: `${categoryColor}15`,
              borderColor: `${categoryColor}30`,
              color: categoryColor,
            }}
          >
            {categoryLabel}
          </Badge>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                className={cn(
                  "text-lg font-bold tabular-nums flex-shrink-0",
                  successRateColorClass
                )}
              >
                {formatPercentage(technique.success_rate)}%
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p>Success rate: {formatPercentage(technique.success_rate)}%</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Progress bar for usage */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
          <span>Applications</span>
          <span className="tabular-nums">{formatNumber(technique.total_applications)}</span>
        </div>
        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500 ease-out"
            style={{
              width: `${progressValue}%`,
              backgroundColor: categoryColor,
            }}
          />
        </div>
      </div>

      {/* Metrics grid */}
      {!compact && (
        <div className="grid grid-cols-4 gap-2 mt-3">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex flex-col items-center p-1.5 rounded bg-white/[0.03]">
                  <span className="text-[10px] text-muted-foreground">Success</span>
                  <span className="text-xs font-medium text-emerald-400 tabular-nums">
                    {formatNumber(technique.success_count)}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Successful applications: {formatNumber(technique.success_count)}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex flex-col items-center p-1.5 rounded bg-white/[0.03]">
                  <span className="text-[10px] text-muted-foreground">Failed</span>
                  <span className="text-xs font-medium text-red-400 tabular-nums">
                    {formatNumber(technique.failure_count)}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Failed applications: {formatNumber(technique.failure_count)}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex flex-col items-center p-1.5 rounded bg-white/[0.03]">
                  <span className="text-[10px] text-muted-foreground">Avg</span>
                  <span className="text-xs font-medium text-foreground tabular-nums">
                    {formatPercentage(technique.avg_score * 100, 0)}%
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Average score: {formatPercentage(technique.avg_score * 100, 1)}%</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex flex-col items-center p-1.5 rounded bg-white/[0.03]">
                  <span className="text-[10px] text-muted-foreground">Best</span>
                  <span className="text-xs font-medium text-violet-400 tabular-nums">
                    {formatPercentage(technique.best_score * 100, 0)}%
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Best score: {formatPercentage(technique.best_score * 100, 1)}%</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}

      {/* Execution time indicator */}
      {technique.avg_execution_time_ms > 0 && !compact && (
        <div className="flex items-center gap-1 mt-2 text-[10px] text-muted-foreground">
          <Clock className="h-2.5 w-2.5" aria-hidden="true" />
          <span>Avg: {formatExecutionTime(technique.avg_execution_time_ms)}</span>
        </div>
      )}
    </div>
  );
});

/**
 * Horizontal bar chart for technique comparison
 */
const TechniqueBarChart = memo(function TechniqueBarChart({
  data,
  height = 200,
}: {
  data: TechniquePerformance[];
  height?: number;
}) {
  // Transform data for chart
  const chartData = useMemo(() => {
    return data.map((technique) => ({
      name: technique.technique_name.length > 15
        ? technique.technique_name.substring(0, 12) + "..."
        : technique.technique_name,
      fullName: technique.technique_name,
      success_rate: technique.success_rate,
      category: technique.technique_category,
      color: getTechniqueColor(technique.technique_category),
    }));
  }, [data]);

  if (chartData.length === 0) return null;

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 40, left: 0, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.05)"
            horizontal={false}
          />
          <XAxis
            type="number"
            domain={[0, 100]}
            tickFormatter={(value: number) => `${value}%`}
            stroke="rgba(255,255,255,0.3)"
            tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
            tickLine={{ stroke: "rgba(255,255,255,0.1)" }}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="rgba(255,255,255,0.3)"
            tick={{ fill: "rgba(255,255,255,0.7)", fontSize: 11 }}
            tickLine={{ stroke: "rgba(255,255,255,0.1)" }}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
            width={100}
          />
          <Bar
            dataKey="success_rate"
            radius={[0, 4, 4, 0]}
            animationDuration={500}
            animationEasing="ease-out"
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.color}
                fillOpacity={0.8}
              />
            ))}
            <LabelList
              dataKey="success_rate"
              position="right"
              formatter={(value: number) => `${value.toFixed(1)}%`}
              fill="rgba(255,255,255,0.7)"
              fontSize={10}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
});

/**
 * Category summary badges
 */
const CategorySummary = memo(function CategorySummary({
  techniques,
}: {
  techniques: TechniquePerformance[];
}) {
  // Group by category
  const categoryStats = useMemo(() => {
    const stats = new Map<TechniqueCategory, { count: number; avgRate: number }>();

    techniques.forEach((tech) => {
      const existing = stats.get(tech.technique_category);
      if (existing) {
        existing.count += 1;
        existing.avgRate = (existing.avgRate * (existing.count - 1) + tech.success_rate) / existing.count;
      } else {
        stats.set(tech.technique_category, { count: 1, avgRate: tech.success_rate });
      }
    });

    return Array.from(stats.entries())
      .map(([category, data]) => ({
        category,
        ...data,
        label: getTechniqueLabel(category),
        color: getTechniqueColor(category),
      }))
      .sort((a, b) => b.avgRate - a.avgRate);
  }, [techniques]);

  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {categoryStats.map(({ category, count, avgRate, label, color }) => (
        <TooltipProvider key={category}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="outline"
                className="gap-1.5 px-2 py-1"
                style={{
                  backgroundColor: `${color}10`,
                  borderColor: `${color}25`,
                }}
              >
                <div
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span style={{ color }}>{label}</span>
                <span className="text-muted-foreground">({count})</span>
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <div className="text-xs">
                <p className="font-medium">{label}</p>
                <p className="text-muted-foreground">{count} techniques</p>
                <p className="text-muted-foreground">
                  Avg success: {formatPercentage(avgRate)}%
                </p>
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ))}
    </div>
  );
});

/**
 * Empty state when no techniques are available
 */
const EmptyState = memo(function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-3">
      <div className="p-3 rounded-full bg-white/5">
        <Layers className="h-6 w-6 text-muted-foreground" aria-hidden="true" />
      </div>
      <div className="text-center">
        <p className="text-sm text-muted-foreground">No techniques applied yet</p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          Technique performance will appear here as transformations are applied
        </p>
      </div>
    </div>
  );
});

/**
 * Loading skeleton for the component
 */
const LoadingSkeleton = memo(function LoadingSkeleton({
  compact,
}: {
  compact?: boolean;
}) {
  return (
    <div className="animate-pulse space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="h-4 w-40 bg-white/10 rounded" />
        <div className="h-7 w-32 bg-white/10 rounded" />
      </div>
      {/* Category badges */}
      <div className="flex gap-2">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="h-6 w-20 bg-white/10 rounded-full" />
        ))}
      </div>
      {/* Chart area */}
      {!compact && <div className="h-40 bg-white/5 rounded-lg" />}
      {/* Technique rows */}
      <div className="space-y-2">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="h-20 bg-white/5 rounded-lg" />
        ))}
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * TechniqueBreakdown displays performance metrics for each transformation technique.
 *
 * Features:
 * - Horizontal bar chart showing success rate per technique
 * - Color-coded by technique category (AutoDAN, GPTFuzz, Chimera framing, etc.)
 * - Sortable by success rate, usage count, avg score, or best score
 * - Detailed metrics for each technique (success/failure counts, scores)
 * - Category summary badges
 * - Execution time indicators
 */
export const TechniqueBreakdown = memo(function TechniqueBreakdown({
  techniques,
  isLoading = false,
  className,
  maxDisplay = 10,
  showChart = true,
  compact = false,
  defaultSortBy = "success_rate",
  defaultSortDirection = "desc",
}: TechniqueBreakdownProps) {
  // Sort state
  const [sortBy, setSortBy] = useState<SortField>(defaultSortBy);
  const [sortDirection, setSortDirection] = useState<SortDirection>(defaultSortDirection);

  // Sort and limit techniques
  const displayedTechniques = useMemo(() => {
    const sorted = sortTechniques(techniques, sortBy, sortDirection);
    return sorted.slice(0, maxDisplay);
  }, [techniques, sortBy, sortDirection, maxDisplay]);

  // Maximum applications for progress bar scaling
  const maxApplications = useMemo(() => {
    if (displayedTechniques.length === 0) return 1;
    return Math.max(...displayedTechniques.map((t) => t.total_applications), 1);
  }, [displayedTechniques]);

  // Handlers
  const handleSortChange = useCallback((field: SortField) => {
    if (field === sortBy) {
      // Toggle direction if same field
      setSortDirection((prev) => (prev === "desc" ? "asc" : "desc"));
    } else {
      setSortBy(field);
      setSortDirection("desc");
    }
  }, [sortBy]);

  const handleDirectionToggle = useCallback(() => {
    setSortDirection((prev) => (prev === "desc" ? "asc" : "desc"));
  }, []);

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

  // Empty state
  if (techniques.length === 0) {
    return (
      <GlassCard
        variant="default"
        intensity="medium"
        className={cn("p-4", className)}
      >
        <ChartHeader
          totalTechniques={0}
          sortBy={sortBy}
          sortDirection={sortDirection}
          onSortChange={handleSortChange}
          onDirectionToggle={handleDirectionToggle}
          compact={compact}
        />
        <EmptyState />
      </GlassCard>
    );
  }

  return (
    <GlassCard
      variant="default"
      intensity="medium"
      className={cn("p-4 overflow-hidden", className)}
    >
      {/* Header with sort controls */}
      <ChartHeader
        totalTechniques={techniques.length}
        sortBy={sortBy}
        sortDirection={sortDirection}
        onSortChange={handleSortChange}
        onDirectionToggle={handleDirectionToggle}
        compact={compact}
      />

      {/* Category summary */}
      {!compact && <CategorySummary techniques={techniques} />}

      {/* Horizontal bar chart */}
      {showChart && !compact && displayedTechniques.length > 0 && (
        <div className="mb-4">
          <TechniqueBarChart
            data={displayedTechniques}
            height={Math.min(displayedTechniques.length * 35 + 30, 250)}
          />
        </div>
      )}

      {/* Technique rows */}
      <div className={cn("space-y-2", showChart && !compact && "mt-4")}>
        {displayedTechniques.map((technique) => (
          <TechniqueRow
            key={`${technique.technique_name}-${technique.technique_category}`}
            technique={technique}
            maxApplications={maxApplications}
            compact={compact}
          />
        ))}
      </div>

      {/* Show more indicator */}
      {techniques.length > maxDisplay && (
        <div className="mt-3 pt-3 border-t border-white/10">
          <p className="text-xs text-muted-foreground text-center">
            Showing {maxDisplay} of {techniques.length} techniques
          </p>
        </div>
      )}
    </GlassCard>
  );
});

// Named export for index
export default TechniqueBreakdown;
