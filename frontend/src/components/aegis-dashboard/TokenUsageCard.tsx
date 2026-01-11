/**
 * TokenUsageCard Component
 *
 * Displays real-time token consumption and cost estimates for Aegis Campaign Dashboard:
 * - Prompt tokens / completion tokens breakdown
 * - Running cost estimate in USD
 * - Cost per successful attack metric
 * - Token usage trend line
 *
 * Follows glass morphism styling pattern from existing components.
 */

"use client";

import { memo, useMemo } from "react";
import {
  Coins,
  TrendingUp,
  TrendingDown,
  Minus,
  Database,
  ArrowDownToLine,
  ArrowUpFromLine,
  DollarSign,
  Target,
  Cpu,
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
import { TokenUsage, TokenUsageTimeSeries } from "@/types/aegis-telemetry";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
} from "recharts";

// ============================================================================
// Types
// ============================================================================

export interface TokenUsageCardProps {
  /** Current token usage data */
  tokenUsage: TokenUsage;
  /** Time series data for trend chart */
  tokenUsageHistory: TokenUsageTimeSeries[];
  /** Number of successful attacks (for cost per attack calculation) */
  successfulAttacks: number;
  /** Total number of attacks (for efficiency metrics) */
  totalAttacks: number;
  /** Trend direction for cost */
  costTrend?: "up" | "down" | "stable";
  /** Percentage change in cost trend */
  costTrendChange?: number;
  /** Whether the component is in loading state */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show in compact mode */
  compact?: boolean;
  /** Whether to show the trend chart (default: true) */
  showChart?: boolean;
}

// ============================================================================
// Configuration
// ============================================================================

/**
 * Cost threshold configuration for color coding
 */
interface CostColorConfig {
  label: string;
  textClass: string;
  bgClass: string;
  borderClass: string;
  chartColor: string;
}

/**
 * Get cost color configuration based on cost per attack
 * Lower is better, so colors are inverted from success rate
 */
function getCostColorConfig(costPerAttack: number): CostColorConfig {
  if (costPerAttack <= 0.01) {
    return {
      label: "Efficient",
      textClass: "text-emerald-400",
      bgClass: "bg-emerald-500/20",
      borderClass: "border-emerald-500/30",
      chartColor: "#10b981",
    };
  } else if (costPerAttack <= 0.05) {
    return {
      label: "Moderate",
      textClass: "text-amber-400",
      bgClass: "bg-amber-500/20",
      borderClass: "border-amber-500/30",
      chartColor: "#f59e0b",
    };
  } else {
    return {
      label: "High",
      textClass: "text-red-400",
      bgClass: "bg-red-500/20",
      borderClass: "border-red-500/30",
      chartColor: "#ef4444",
    };
  }
}

const TREND_CONFIG = {
  up: {
    icon: TrendingUp,
    label: "Increasing",
    textClass: "text-red-400", // Cost going up is bad
    bgClass: "bg-red-500/10",
    borderClass: "border-red-500/20",
  },
  down: {
    icon: TrendingDown,
    label: "Decreasing",
    textClass: "text-emerald-400", // Cost going down is good
    bgClass: "bg-emerald-500/10",
    borderClass: "border-emerald-500/20",
  },
  stable: {
    icon: Minus,
    label: "Stable",
    textClass: "text-gray-400",
    bgClass: "bg-gray-500/10",
    borderClass: "border-gray-500/20",
  },
};

// Provider pricing (per 1M tokens) - approximations
const PROVIDER_PRICING: Record<string, { input: number; output: number; label: string }> = {
  google: { input: 0.075, output: 0.30, label: "Gemini" },
  openai: { input: 2.50, output: 10.00, label: "GPT-4" },
  anthropic: { input: 3.00, output: 15.00, label: "Claude" },
  deepseek: { input: 0.14, output: 0.28, label: "DeepSeek" },
  default: { input: 1.00, output: 2.00, label: "Unknown" },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format USD currency for display
 */
function formatCurrency(value: number, decimals: number = 4): string {
  if (value < 0.0001 && value > 0) {
    return "<$0.0001";
  }
  if (value >= 1) {
    return `$${value.toFixed(2)}`;
  }
  return `$${value.toFixed(decimals)}`;
}

/**
 * Format large numbers with K/M/B suffixes
 */
function formatTokenCount(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toLocaleString();
}

/**
 * Format percentage for display
 */
function formatPercentage(value: number, decimals: number = 1): string {
  return value.toFixed(decimals);
}

/**
 * Calculate cost per successful attack
 */
function calculateCostPerSuccess(cost: number, successfulAttacks: number): number | null {
  if (successfulAttacks === 0) return null;
  return cost / successfulAttacks;
}

/**
 * Calculate token efficiency (successful attacks per 1K tokens)
 */
function calculateTokenEfficiency(successfulAttacks: number, totalTokens: number): number | null {
  if (totalTokens === 0) return null;
  return (successfulAttacks / totalTokens) * 1000;
}

/**
 * Calculate trend from history
 */
function calculateCostTrend(
  history: TokenUsageTimeSeries[],
  currentCost: number
): { trend: "up" | "down" | "stable"; change: number } {
  if (history.length < 2) {
    return { trend: "stable", change: 0 };
  }

  // Get last 10 data points for trend calculation
  const recentHistory = history.slice(-10);
  const firstCost = recentHistory[0].cost_usd;
  const change = currentCost - firstCost;
  const percentChange = firstCost > 0 ? (change / firstCost) * 100 : 0;

  let trend: "up" | "down" | "stable";
  if (percentChange > 10) {
    trend = "up";
  } else if (percentChange < -10) {
    trend = "down";
  } else {
    trend = "stable";
  }

  return { trend, change: percentChange };
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Main cost display with large USD value
 */
const CostDisplay = memo(function CostDisplay({
  cost,
  colorConfig,
  compact,
}: {
  cost: number;
  colorConfig: CostColorConfig;
  compact?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-2">
      <div className={cn("flex items-center gap-2", compact && "gap-1")}>
        <DollarSign
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
            compact ? "text-3xl" : "text-4xl"
          )}
        >
          {cost < 0.01 && cost > 0
            ? cost.toFixed(4)
            : cost.toFixed(cost >= 1 ? 2 : 3)}
        </span>
        <span
          className={cn(
            "font-medium text-muted-foreground",
            compact ? "text-sm" : "text-base"
          )}
        >
          USD
        </span>
      </div>
    </div>
  );
});

/**
 * Trend indicator badge for cost
 */
const CostTrendIndicator = memo(function CostTrendIndicator({
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
            <span>
              {absChange > 0
                ? `${trend === "up" ? "+" : "-"}${formatPercentage(absChange)}%`
                : "0%"}
            </span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {config.label} - Cost{" "}
            {trend === "stable"
              ? "is stable"
              : trend === "up"
              ? "increased"
              : "decreased"}{" "}
            by {formatPercentage(absChange)}% over recent operations
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Token breakdown showing prompt vs completion tokens
 */
const TokenBreakdown = memo(function TokenBreakdown({
  tokenUsage,
  compact,
}: {
  tokenUsage: TokenUsage;
  compact?: boolean;
}) {
  const promptPercentage =
    tokenUsage.total_tokens > 0
      ? (tokenUsage.prompt_tokens / tokenUsage.total_tokens) * 100
      : 0;
  const completionPercentage = 100 - promptPercentage;

  return (
    <div className={cn("space-y-2", compact && "space-y-1.5")}>
      {/* Token count header */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Token Usage</span>
        <span className="text-xs font-medium text-foreground tabular-nums">
          {formatTokenCount(tokenUsage.total_tokens)} total
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-2 bg-white/5 rounded-full overflow-hidden flex">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className="h-full bg-cyan-500 transition-all duration-500"
                style={{ width: `${promptPercentage}%` }}
              />
            </TooltipTrigger>
            <TooltipContent>
              <p>
                Prompt tokens: {formatTokenCount(tokenUsage.prompt_tokens)} (
                {formatPercentage(promptPercentage)}%)
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className="h-full bg-violet-500 transition-all duration-500"
                style={{ width: `${completionPercentage}%` }}
              />
            </TooltipTrigger>
            <TooltipContent>
              <p>
                Completion tokens:{" "}
                {formatTokenCount(tokenUsage.completion_tokens)} (
                {formatPercentage(completionPercentage)}%)
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Token legend */}
      <div className="flex items-center justify-between text-[10px]">
        <div className="flex items-center gap-1.5">
          <ArrowUpFromLine className="h-3 w-3 text-cyan-400" aria-hidden="true" />
          <span className="text-cyan-400">Prompt</span>
          <span className="text-muted-foreground tabular-nums">
            {formatTokenCount(tokenUsage.prompt_tokens)}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <ArrowDownToLine className="h-3 w-3 text-violet-400" aria-hidden="true" />
          <span className="text-violet-400">Completion</span>
          <span className="text-muted-foreground tabular-nums">
            {formatTokenCount(tokenUsage.completion_tokens)}
          </span>
        </div>
      </div>
    </div>
  );
});

/**
 * Cost metrics grid
 */
const CostMetrics = memo(function CostMetrics({
  tokenUsage,
  successfulAttacks,
  totalAttacks,
  compact,
}: {
  tokenUsage: TokenUsage;
  successfulAttacks: number;
  totalAttacks: number;
  compact?: boolean;
}) {
  const costPerSuccess = calculateCostPerSuccess(
    tokenUsage.cost_estimate_usd,
    successfulAttacks
  );
  const tokenEfficiency = calculateTokenEfficiency(
    successfulAttacks,
    tokenUsage.total_tokens
  );
  const costPerAttack = totalAttacks > 0
    ? tokenUsage.cost_estimate_usd / totalAttacks
    : null;

  return (
    <div className={cn("grid gap-2", compact ? "grid-cols-2" : "grid-cols-4")}>
      {/* Total Cost */}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div
              className={cn(
                "flex flex-col items-center justify-center rounded-lg p-2",
                "bg-white/5 border border-white/10"
              )}
            >
              <span className="text-xs text-muted-foreground">Total Cost</span>
              <span className="text-sm font-semibold text-foreground tabular-nums">
                {formatCurrency(tokenUsage.cost_estimate_usd, 4)}
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <p>Cumulative cost for all LLM API calls</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* Cost Per Attack */}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div
              className={cn(
                "flex flex-col items-center justify-center rounded-lg p-2",
                "bg-amber-500/10 border border-amber-500/20"
              )}
            >
              <span className="text-xs text-amber-400/80">$/Attack</span>
              <span className="text-sm font-semibold text-amber-400 tabular-nums">
                {costPerAttack !== null
                  ? formatCurrency(costPerAttack, 4)
                  : "—"}
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <p>Average cost per attack attempt</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {!compact && (
        <>
          {/* Cost Per Success */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "flex flex-col items-center justify-center rounded-lg p-2",
                    "bg-emerald-500/10 border border-emerald-500/20"
                  )}
                >
                  <span className="text-xs text-emerald-400/80">$/Success</span>
                  <span className="text-sm font-semibold text-emerald-400 tabular-nums">
                    {costPerSuccess !== null
                      ? formatCurrency(costPerSuccess, 4)
                      : "—"}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Average cost per successful attack</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          {/* Token Efficiency */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "flex flex-col items-center justify-center rounded-lg p-2",
                    "bg-violet-500/10 border border-violet-500/20"
                  )}
                >
                  <span className="text-xs text-violet-400/80">Efficiency</span>
                  <span className="text-sm font-semibold text-violet-400 tabular-nums">
                    {tokenEfficiency !== null
                      ? tokenEfficiency.toFixed(2)
                      : "—"}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Successful attacks per 1,000 tokens used</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </>
      )}
    </div>
  );
});

/**
 * Custom tooltip for the trend chart
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const CustomChartTooltip = ({ active, payload }: any) => {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload as TokenUsageTimeSeries;
  const time = new Date(data.timestamp).toLocaleTimeString();

  return (
    <div className="rounded-lg border border-white/10 bg-black/80 p-2 shadow-xl backdrop-blur-sm">
      <p className="text-xs font-medium text-foreground">{time}</p>
      <div className="mt-1 space-y-0.5 text-xs">
        <p className="text-cyan-400">
          Prompt: {formatTokenCount(data.prompt_tokens)}
        </p>
        <p className="text-violet-400">
          Completion: {formatTokenCount(data.completion_tokens)}
        </p>
        <p className="text-emerald-400">
          Cost: {formatCurrency(data.cost_usd)}
        </p>
      </div>
    </div>
  );
};

/**
 * Token usage trend line chart
 */
const TokenTrendChart = memo(function TokenTrendChart({
  data,
  height = 80,
}: {
  data: TokenUsageTimeSeries[];
  height?: number;
}) {
  // Transform data for chart - show cumulative token usage
  const chartData = useMemo(() => {
    let cumulativeTotal = 0;
    return data.map((point) => {
      cumulativeTotal = point.total_tokens;
      return {
        ...point,
        cumulative: cumulativeTotal,
      };
    });
  }, [data]);

  if (chartData.length < 2) {
    return (
      <div
        className="flex items-center justify-center text-muted-foreground text-xs"
        style={{ height }}
      >
        <span>Awaiting data...</span>
      </div>
    );
  }

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
        >
          <defs>
            <linearGradient id="tokenGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="timestamp"
            hide
            tickFormatter={(value: string) =>
              new Date(value).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })
            }
          />
          <YAxis hide domain={["dataMin", "dataMax"]} />
          <RechartsTooltip content={<CustomChartTooltip />} />
          <Area
            type="monotone"
            dataKey="total_tokens"
            stroke="#8b5cf6"
            strokeWidth={2}
            fill="url(#tokenGradient)"
            isAnimationActive={true}
            animationDuration={500}
            animationEasing="ease-out"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
});

/**
 * Provider info badge
 */
const ProviderBadge = memo(function ProviderBadge({
  provider,
  model,
}: {
  provider: string | null;
  model: string | null;
}) {
  const providerKey = (provider?.toLowerCase() || "default") as keyof typeof PROVIDER_PRICING;
  const pricing = PROVIDER_PRICING[providerKey] || PROVIDER_PRICING.default;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="gap-1.5 text-xs bg-white/5 border-white/10"
          >
            <Cpu className="h-3 w-3" aria-hidden="true" />
            <span>{model || pricing.label}</span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <div className="text-xs">
            <p className="font-medium">{provider || "Unknown Provider"}</p>
            <p className="text-muted-foreground">
              Input: ${pricing.input}/1M tokens
            </p>
            <p className="text-muted-foreground">
              Output: ${pricing.output}/1M tokens
            </p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Loading skeleton
 */
const LoadingSkeleton = memo(function LoadingSkeleton({
  compact,
}: {
  compact?: boolean;
}) {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-4 w-32 bg-white/10 rounded" />
      <div className="h-10 w-40 bg-white/10 rounded" />
      <div className="h-8 w-full bg-white/10 rounded" />
      <div className={cn("grid gap-2", compact ? "grid-cols-2" : "grid-cols-4")}>
        {[...Array(compact ? 2 : 4)].map((_, i) => (
          <div key={i} className="h-14 bg-white/10 rounded" />
        ))}
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * TokenUsageCard displays real-time token consumption and cost estimates.
 *
 * Features:
 * - Large cost display with USD value
 * - Prompt/completion token breakdown with visual bar
 * - Cost per attack and cost per successful attack metrics
 * - Token efficiency metric
 * - Token usage trend line chart
 * - Provider information badge
 * - Cost trend indicator
 */
export const TokenUsageCard = memo(function TokenUsageCard({
  tokenUsage,
  tokenUsageHistory,
  successfulAttacks,
  totalAttacks,
  costTrend: externalCostTrend,
  costTrendChange: externalCostTrendChange,
  isLoading = false,
  className,
  compact = false,
  showChart = true,
}: TokenUsageCardProps) {
  // Calculate cost per attack for color config
  const costPerAttack = useMemo(() => {
    if (successfulAttacks === 0) return 0;
    return tokenUsage.cost_estimate_usd / successfulAttacks;
  }, [tokenUsage.cost_estimate_usd, successfulAttacks]);

  // Get color configuration
  const colorConfig = useMemo(
    () => getCostColorConfig(costPerAttack),
    [costPerAttack]
  );

  // Calculate trend if not provided externally
  const { trend, change } = useMemo(() => {
    if (
      externalCostTrend !== undefined &&
      externalCostTrendChange !== undefined
    ) {
      return { trend: externalCostTrend, change: externalCostTrendChange };
    }
    return calculateCostTrend(tokenUsageHistory, tokenUsage.cost_estimate_usd);
  }, [
    externalCostTrend,
    externalCostTrendChange,
    tokenUsageHistory,
    tokenUsage.cost_estimate_usd,
  ]);

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

  return (
    <GlassCard
      variant="default"
      intensity="medium"
      className={cn("p-4 overflow-hidden", className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Coins className="h-4 w-4 text-amber-400" aria-hidden="true" />
          <h3 className="text-sm font-medium text-muted-foreground">
            Token Usage & Cost
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
        <ProviderBadge provider={tokenUsage.provider} model={tokenUsage.model} />
      </div>

      {/* Main Cost Display */}
      <div className="flex items-start justify-between mb-4">
        <CostDisplay
          cost={tokenUsage.cost_estimate_usd}
          colorConfig={colorConfig}
          compact={compact}
        />
        <CostTrendIndicator trend={trend} change={change} compact={compact} />
      </div>

      {/* Token Breakdown */}
      <div className={cn("mb-4", compact && "mb-3")}>
        <TokenBreakdown tokenUsage={tokenUsage} compact={compact} />
      </div>

      {/* Trend Chart */}
      {showChart && !compact && (
        <div className="mb-4">
          <TokenTrendChart data={tokenUsageHistory} height={60} />
        </div>
      )}

      {/* Cost Metrics */}
      <CostMetrics
        tokenUsage={tokenUsage}
        successfulAttacks={successfulAttacks}
        totalAttacks={totalAttacks}
        compact={compact}
      />

      {/* Footer with attack counts (non-compact only) */}
      {!compact && (
        <div className="mt-3 pt-3 border-t border-white/10">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1.5">
              <Target className="h-3 w-3 text-muted-foreground" aria-hidden="true" />
              <span className="text-muted-foreground">
                {totalAttacks} attacks
              </span>
              <span className="text-emerald-400">
                ({successfulAttacks} successful)
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <Database className="h-3 w-3 text-muted-foreground" aria-hidden="true" />
              <span className="text-muted-foreground">
                {formatTokenCount(tokenUsage.total_tokens)} tokens used
              </span>
            </div>
          </div>
        </div>
      )}
    </GlassCard>
  );
});

// Named export for index
export default TokenUsageCard;
