/**
 * SuccessRateCard Component
 *
 * Displays real-time attack success rate for Aegis Campaign Dashboard with:
 * - Large percentage display with color coding (green >70%, amber >40%, red <40%)
 * - Trend arrow (up/down/stable) based on last 10 attempts
 * - Sparkline mini-chart of recent success rates
 * - Total attempts counter
 *
 * Follows glass morphism styling pattern from existing components.
 */

"use client";

import { memo, useMemo } from "react";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Zap,
  Trophy,
  AlertTriangle,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  AttackMetrics,
  SuccessRateTimeSeries,
} from "@/types/aegis-telemetry";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
} from "recharts";

// ============================================================================
// Types
// ============================================================================

export interface SuccessRateCardProps {
  /** Current attack metrics */
  metrics: AttackMetrics;
  /** Time series data for sparkline */
  successRateHistory: SuccessRateTimeSeries[];
  /** Trend direction based on recent attempts */
  trend?: "up" | "down" | "stable";
  /** Percentage change for trend */
  trendChange?: number;
  /** Whether the component is in loading state */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show in compact mode */
  compact?: boolean;
}

// ============================================================================
// Configuration
// ============================================================================

/**
 * Color configuration based on success rate thresholds
 */
interface SuccessRateColorConfig {
  label: string;
  textClass: string;
  bgClass: string;
  borderClass: string;
  gradientFrom: string;
  gradientTo: string;
  chartColor: string;
  icon: typeof Trophy;
}

const getSuccessRateColorConfig = (rate: number): SuccessRateColorConfig => {
  if (rate >= 70) {
    return {
      label: "Excellent",
      textClass: "text-emerald-400",
      bgClass: "bg-emerald-500/20",
      borderClass: "border-emerald-500/30",
      gradientFrom: "from-emerald-500/20",
      gradientTo: "to-emerald-500/5",
      chartColor: "#10b981",
      icon: Trophy,
    };
  } else if (rate >= 40) {
    return {
      label: "Moderate",
      textClass: "text-amber-400",
      bgClass: "bg-amber-500/20",
      borderClass: "border-amber-500/30",
      gradientFrom: "from-amber-500/20",
      gradientTo: "to-amber-500/5",
      chartColor: "#f59e0b",
      icon: Target,
    };
  } else {
    return {
      label: "Low",
      textClass: "text-red-400",
      bgClass: "bg-red-500/20",
      borderClass: "border-red-500/30",
      gradientFrom: "from-red-500/20",
      gradientTo: "to-red-500/5",
      chartColor: "#ef4444",
      icon: AlertTriangle,
    };
  }
};

const TREND_CONFIG = {
  up: {
    icon: TrendingUp,
    label: "Improving",
    textClass: "text-emerald-400",
    bgClass: "bg-emerald-500/10",
    borderClass: "border-emerald-500/20",
  },
  down: {
    icon: TrendingDown,
    label: "Declining",
    textClass: "text-red-400",
    bgClass: "bg-red-500/10",
    borderClass: "border-red-500/20",
  },
  stable: {
    icon: Minus,
    label: "Stable",
    textClass: "text-gray-400",
    bgClass: "bg-gray-500/10",
    borderClass: "border-gray-500/20",
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

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
 * Calculate trend from history
 */
function calculateTrend(
  history: SuccessRateTimeSeries[],
  currentRate: number
): { trend: "up" | "down" | "stable"; change: number } {
  if (history.length < 2) {
    return { trend: "stable", change: 0 };
  }

  // Get last 10 data points for trend calculation
  const recentHistory = history.slice(-10);
  const firstRate = recentHistory[0].success_rate;
  const change = currentRate - firstRate;

  let trend: "up" | "down" | "stable";
  if (change > 5) {
    trend = "up";
  } else if (change < -5) {
    trend = "down";
  } else {
    trend = "stable";
  }

  return { trend, change };
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Animated percentage display
 */
const PercentageDisplay = memo(function PercentageDisplay({
  value,
  colorConfig,
  compact,
}: {
  value: number;
  colorConfig: SuccessRateColorConfig;
  compact?: boolean;
}) {
  const Icon = colorConfig.icon;

  return (
    <div className="flex items-baseline gap-2">
      <div className={cn("flex items-center gap-2", compact && "gap-1")}>
        <Icon
          className={cn(
            colorConfig.textClass,
            compact ? "h-5 w-5" : "h-6 w-6"
          )}
          aria-hidden="true"
        />
        <span
          className={cn(
            "font-bold tabular-nums tracking-tight",
            colorConfig.textClass,
            compact ? "text-3xl" : "text-5xl"
          )}
        >
          {formatPercentage(value)}
        </span>
        <span
          className={cn(
            "font-semibold",
            colorConfig.textClass,
            compact ? "text-lg" : "text-2xl"
          )}
        >
          %
        </span>
      </div>
    </div>
  );
});

/**
 * Trend indicator badge
 */
const TrendIndicator = memo(function TrendIndicator({
  trend,
  change,
  compact,
}: {
  trend: "up" | "down" | "stable";
  change: number;
  compact?: boolean;
}) {
  const config = TREND_CONFIG[trend];
  const Icon = config.icon;
  const absChange = Math.abs(change);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className={cn(
              "gap-1 font-medium transition-all duration-300",
              config.bgClass,
              config.borderClass,
              config.textClass,
              compact ? "px-1.5 py-0.5 text-xs" : "px-2 py-1 text-sm"
            )}
          >
            <Icon
              className={cn(
                trend !== "stable" && "animate-pulse",
                compact ? "h-3 w-3" : "h-3.5 w-3.5"
              )}
              aria-hidden="true"
            />
            <span>{absChange > 0 ? `${absChange > 0 ? (trend === "up" ? "+" : "-") : ""}${formatPercentage(absChange)}%` : "0%"}</span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {config.label} - {trend === "stable" ? "No significant change" : `${formatPercentage(absChange)}% ${trend === "up" ? "increase" : "decrease"}`} in last 10 attempts
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Sparkline chart for success rate history
 */
const SparklineChart = memo(function SparklineChart({
  data,
  color,
  compact,
}: {
  data: SuccessRateTimeSeries[];
  color: string;
  compact?: boolean;
}) {
  // Transform data for chart
  const chartData = useMemo(() => {
    return data.map((point, index) => ({
      index,
      value: point.success_rate,
    }));
  }, [data]);

  if (chartData.length < 2) {
    return (
      <div
        className={cn(
          "flex items-center justify-center text-muted-foreground text-xs",
          compact ? "h-8" : "h-12"
        )}
      >
        <span>Awaiting data...</span>
      </div>
    );
  }

  return (
    <div className={cn("w-full", compact ? "h-8" : "h-12")}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          margin={{ top: 0, right: 0, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id={`sparklineGradient-${color}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            fill={`url(#sparklineGradient-${color})`}
            isAnimationActive={true}
            animationDuration={300}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
});

/**
 * Attack statistics badges
 */
const AttackStats = memo(function AttackStats({
  metrics,
  compact,
}: {
  metrics: AttackMetrics;
  compact?: boolean;
}) {
  return (
    <div className={cn("grid gap-2", compact ? "grid-cols-2" : "grid-cols-4")}>
      {/* Total Attempts */}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div
              className={cn(
                "flex flex-col items-center justify-center rounded-lg p-2",
                "bg-white/5 border border-white/10"
              )}
            >
              <span className="text-xs text-muted-foreground">Attempts</span>
              <span className="text-sm font-semibold text-foreground tabular-nums">
                {formatNumber(metrics.total_attempts)}
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <p>Total attack attempts made</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* Successful */}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div
              className={cn(
                "flex flex-col items-center justify-center rounded-lg p-2",
                "bg-emerald-500/10 border border-emerald-500/20"
              )}
            >
              <span className="text-xs text-emerald-400/80">Success</span>
              <span className="text-sm font-semibold text-emerald-400 tabular-nums">
                {formatNumber(metrics.successful_attacks)}
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <p>Successful attack attempts</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {!compact && (
        <>
          {/* Failed */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "flex flex-col items-center justify-center rounded-lg p-2",
                    "bg-red-500/10 border border-red-500/20"
                  )}
                >
                  <span className="text-xs text-red-400/80">Failed</span>
                  <span className="text-sm font-semibold text-red-400 tabular-nums">
                    {formatNumber(metrics.failed_attacks)}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Failed attack attempts</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          {/* Best Score */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "flex flex-col items-center justify-center rounded-lg p-2",
                    "bg-violet-500/10 border border-violet-500/20"
                  )}
                >
                  <span className="text-xs text-violet-400/80">Best</span>
                  <span className="text-sm font-semibold text-violet-400 tabular-nums">
                    {formatPercentage(metrics.best_score * 100, 0)}%
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Best attack score achieved</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </>
      )}
    </div>
  );
});

/**
 * Current streak indicator
 */
const StreakIndicator = memo(function StreakIndicator({
  streak,
}: {
  streak: number;
}) {
  if (streak === 0) return null;

  const isPositive = streak > 0;
  const absStreak = Math.abs(streak);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn(
              "flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium",
              isPositive
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                : "bg-red-500/10 text-red-400 border border-red-500/20"
            )}
          >
            <Zap className="h-3 w-3" aria-hidden="true" />
            <span>
              {absStreak} {isPositive ? "Win" : "Loss"} Streak
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {isPositive
              ? `${absStreak} successful attacks in a row`
              : `${absStreak} failed attacks in a row`}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * SuccessRateCard displays real-time attack success rate for the Aegis Dashboard.
 *
 * Features:
 * - Large percentage display with color-coded thresholds
 * - Trend indicator based on recent attack history
 * - Sparkline mini-chart showing rate over time
 * - Attack statistics (total, successful, failed, best score)
 * - Current win/loss streak indicator
 */
export const SuccessRateCard = memo(function SuccessRateCard({
  metrics,
  successRateHistory,
  trend: externalTrend,
  trendChange: externalTrendChange,
  isLoading = false,
  className,
  compact = false,
}: SuccessRateCardProps) {
  // Calculate color config based on current success rate
  const colorConfig = useMemo(
    () => getSuccessRateColorConfig(metrics.success_rate),
    [metrics.success_rate]
  );

  // Calculate trend if not provided externally
  const { trend, change } = useMemo(() => {
    if (externalTrend !== undefined && externalTrendChange !== undefined) {
      return { trend: externalTrend, change: externalTrendChange };
    }
    return calculateTrend(successRateHistory, metrics.success_rate);
  }, [externalTrend, externalTrendChange, successRateHistory, metrics.success_rate]);

  // Loading state
  if (isLoading) {
    return (
      <GlassCard
        variant="default"
        intensity="medium"
        className={cn("p-4", className)}
      >
        <div className="animate-pulse space-y-4">
          <div className="h-4 w-24 bg-white/10 rounded" />
          <div className="h-12 w-32 bg-white/10 rounded" />
          <div className="h-8 w-full bg-white/10 rounded" />
          <div className="grid grid-cols-4 gap-2">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-12 bg-white/10 rounded" />
            ))}
          </div>
        </div>
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
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-muted-foreground">
            Attack Success Rate
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
        {metrics.current_streak !== 0 && (
          <StreakIndicator streak={metrics.current_streak} />
        )}
      </div>

      {/* Main Percentage Display */}
      <div className="flex items-start justify-between mb-4">
        <PercentageDisplay
          value={metrics.success_rate}
          colorConfig={colorConfig}
          compact={compact}
        />
        <TrendIndicator trend={trend} change={change} compact={compact} />
      </div>

      {/* Sparkline Chart */}
      <div className={cn("mb-4", compact && "mb-2")}>
        <SparklineChart
          data={successRateHistory}
          color={colorConfig.chartColor}
          compact={compact}
        />
      </div>

      {/* Attack Statistics */}
      <AttackStats metrics={metrics} compact={compact} />

      {/* Average Score Footer (only in non-compact mode) */}
      {!compact && metrics.average_score > 0 && (
        <div className="mt-3 pt-3 border-t border-white/10">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Average Score</span>
            <span className="font-medium text-foreground tabular-nums">
              {formatPercentage(metrics.average_score * 100, 1)}%
            </span>
          </div>
        </div>
      )}
    </GlassCard>
  );
});

// Named export for index
export default SuccessRateCard;
