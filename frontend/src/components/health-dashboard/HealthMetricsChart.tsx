"use client";

import * as React from "react";
import { useMemo, useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Area,
  AreaChart,
  ComposedChart,
  Bar,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { Activity, AlertTriangle, Clock, TrendingDown, TrendingUp } from "lucide-react";

// =============================================================================
// Types
// =============================================================================

export interface HealthHistoryEntry {
  timestamp: string;
  latency_ms: number;
  error_rate: number;
  success_rate: number;
  uptime_percent?: number;
  total_requests?: number;
  failed_requests?: number;
}

export interface HealthMetricsChartProps {
  data: HealthHistoryEntry[];
  providerId?: string;
  providerName?: string;
  isLoading?: boolean;
  error?: Error | null;
  height?: number;
  showLatency?: boolean;
  showErrorRate?: boolean;
  showCombined?: boolean;
  onTimeRangeChange?: (range: TimeRange) => void;
  className?: string;
}

export type TimeRange = "1h" | "6h" | "24h" | "7d" | "30d";

// =============================================================================
// Chart Configuration
// =============================================================================

const latencyChartConfig: ChartConfig = {
  latency_ms: {
    label: "Latency (ms)",
    color: "hsl(var(--chart-1))",
  },
};

const errorRateChartConfig: ChartConfig = {
  error_rate: {
    label: "Error Rate (%)",
    color: "hsl(var(--destructive))",
  },
  success_rate: {
    label: "Success Rate (%)",
    color: "hsl(var(--chart-2))",
  },
};

const combinedChartConfig: ChartConfig = {
  latency_ms: {
    label: "Latency (ms)",
    color: "hsl(var(--chart-1))",
  },
  error_rate: {
    label: "Error Rate (%)",
    color: "hsl(var(--destructive))",
  },
};

// =============================================================================
// Helper Functions
// =============================================================================

function formatTimestamp(timestamp: string, range: TimeRange): string {
  const date = new Date(timestamp);

  switch (range) {
    case "1h":
    case "6h":
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    case "24h":
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    case "7d":
      return date.toLocaleDateString([], { weekday: "short", hour: "2-digit" });
    case "30d":
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    default:
      return date.toLocaleString();
  }
}

function calculateStats(data: HealthHistoryEntry[]) {
  if (data.length === 0) {
    return {
      avgLatency: 0,
      maxLatency: 0,
      minLatency: 0,
      avgErrorRate: 0,
      latencyTrend: "stable" as const,
      errorTrend: "stable" as const,
    };
  }

  const latencies = data.map((d) => d.latency_ms);
  const errorRates = data.map((d) => d.error_rate);

  const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const maxLatency = Math.max(...latencies);
  const minLatency = Math.min(...latencies);
  const avgErrorRate = errorRates.reduce((a, b) => a + b, 0) / errorRates.length;

  // Calculate trends (compare first half to second half)
  const halfPoint = Math.floor(data.length / 2);
  const firstHalfLatency = latencies.slice(0, halfPoint);
  const secondHalfLatency = latencies.slice(halfPoint);
  const firstHalfErrors = errorRates.slice(0, halfPoint);
  const secondHalfErrors = errorRates.slice(halfPoint);

  const firstLatencyAvg =
    firstHalfLatency.reduce((a, b) => a + b, 0) / (firstHalfLatency.length || 1);
  const secondLatencyAvg =
    secondHalfLatency.reduce((a, b) => a + b, 0) / (secondHalfLatency.length || 1);
  const firstErrorAvg =
    firstHalfErrors.reduce((a, b) => a + b, 0) / (firstHalfErrors.length || 1);
  const secondErrorAvg =
    secondHalfErrors.reduce((a, b) => a + b, 0) / (secondHalfErrors.length || 1);

  const latencyDiff = ((secondLatencyAvg - firstLatencyAvg) / (firstLatencyAvg || 1)) * 100;
  const errorDiff = ((secondErrorAvg - firstErrorAvg) / (firstErrorAvg || 1)) * 100;

  const latencyTrend: "up" | "down" | "stable" =
    latencyDiff > 10 ? "up" : latencyDiff < -10 ? "down" : "stable";
  const errorTrend: "up" | "down" | "stable" =
    errorDiff > 10 ? "up" : errorDiff < -10 ? "down" : "stable";

  return {
    avgLatency,
    maxLatency,
    minLatency,
    avgErrorRate,
    latencyTrend,
    errorTrend,
  };
}

// =============================================================================
// Sub-Components
// =============================================================================

interface StatsOverviewProps {
  stats: ReturnType<typeof calculateStats>;
}

function StatsOverview({ stats }: StatsOverviewProps) {
  const TrendIcon = stats.latencyTrend === "up" ? TrendingUp : TrendingDown;
  const trendColor =
    stats.latencyTrend === "up"
      ? "text-red-500"
      : stats.latencyTrend === "down"
      ? "text-emerald-500"
      : "text-muted-foreground";

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="p-3 rounded-lg bg-muted/50">
        <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
          <Clock className="h-3 w-3" />
          Avg Latency
        </div>
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold">{stats.avgLatency.toFixed(0)}ms</span>
          {stats.latencyTrend !== "stable" && (
            <TrendIcon className={cn("h-4 w-4", trendColor)} />
          )}
        </div>
      </div>

      <div className="p-3 rounded-lg bg-muted/50">
        <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
          <Activity className="h-3 w-3" />
          Max Latency
        </div>
        <span className="text-lg font-semibold">{stats.maxLatency.toFixed(0)}ms</span>
      </div>

      <div className="p-3 rounded-lg bg-muted/50">
        <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
          <Clock className="h-3 w-3" />
          Min Latency
        </div>
        <span className="text-lg font-semibold">{stats.minLatency.toFixed(0)}ms</span>
      </div>

      <div className="p-3 rounded-lg bg-muted/50">
        <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
          <AlertTriangle className="h-3 w-3" />
          Avg Error Rate
        </div>
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold">{stats.avgErrorRate.toFixed(2)}%</span>
          {stats.errorTrend === "up" && (
            <Badge variant="destructive" className="text-[10px] h-4">
              Increasing
            </Badge>
          )}
        </div>
      </div>
    </div>
  );
}

interface TimeRangeSelectorProps {
  selected: TimeRange;
  onChange: (range: TimeRange) => void;
}

function TimeRangeSelector({ selected, onChange }: TimeRangeSelectorProps) {
  const ranges: { value: TimeRange; label: string }[] = [
    { value: "1h", label: "1H" },
    { value: "6h", label: "6H" },
    { value: "24h", label: "24H" },
    { value: "7d", label: "7D" },
    { value: "30d", label: "30D" },
  ];

  return (
    <div className="flex gap-1">
      {ranges.map((range) => (
        <Button
          key={range.value}
          variant={selected === range.value ? "default" : "outline"}
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={() => onChange(range.value)}
        >
          {range.label}
        </Button>
      ))}
    </div>
  );
}

// =============================================================================
// Chart Components
// =============================================================================

interface LatencyChartProps {
  data: HealthHistoryEntry[];
  timeRange: TimeRange;
  height: number;
}

function LatencyChart({ data, timeRange, height }: LatencyChartProps) {
  const formattedData = useMemo(
    () =>
      data.map((entry) => ({
        ...entry,
        time: formatTimestamp(entry.timestamp, timeRange),
      })),
    [data, timeRange]
  );

  return (
    <ChartContainer config={latencyChartConfig} className="w-full" style={{ height }}>
      <AreaChart data={formattedData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="var(--color-latency_ms)" stopOpacity={0.3} />
            <stop offset="95%" stopColor="var(--color-latency_ms)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
        />
        <YAxis
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
          tickFormatter={(value) => `${value}ms`}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              labelFormatter={(value) => `Time: ${value}`}
              formatter={(value, name) => [
                <span key="value" className="font-mono font-medium">
                  {Number(value).toFixed(0)}ms
                </span>,
                "Latency",
              ]}
            />
          }
        />
        <Area
          type="monotone"
          dataKey="latency_ms"
          stroke="var(--color-latency_ms)"
          fill="url(#latencyGradient)"
          strokeWidth={2}
        />
      </AreaChart>
    </ChartContainer>
  );
}

interface ErrorRateChartProps {
  data: HealthHistoryEntry[];
  timeRange: TimeRange;
  height: number;
}

function ErrorRateChart({ data, timeRange, height }: ErrorRateChartProps) {
  const formattedData = useMemo(
    () =>
      data.map((entry) => ({
        ...entry,
        time: formatTimestamp(entry.timestamp, timeRange),
      })),
    [data, timeRange]
  );

  return (
    <ChartContainer config={errorRateChartConfig} className="w-full" style={{ height }}>
      <ComposedChart data={formattedData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
        />
        <YAxis
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
          tickFormatter={(value) => `${value}%`}
          domain={[0, "auto"]}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              labelFormatter={(value) => `Time: ${value}`}
            />
          }
        />
        <ChartLegend content={<ChartLegendContent />} />
        <Bar
          dataKey="error_rate"
          fill="var(--color-error_rate)"
          opacity={0.8}
          radius={[2, 2, 0, 0]}
        />
        <Line
          type="monotone"
          dataKey="success_rate"
          stroke="var(--color-success_rate)"
          strokeWidth={2}
          dot={false}
        />
      </ComposedChart>
    </ChartContainer>
  );
}

interface CombinedChartProps {
  data: HealthHistoryEntry[];
  timeRange: TimeRange;
  height: number;
}

function CombinedChart({ data, timeRange, height }: CombinedChartProps) {
  const formattedData = useMemo(
    () =>
      data.map((entry) => ({
        ...entry,
        time: formatTimestamp(entry.timestamp, timeRange),
      })),
    [data, timeRange]
  );

  return (
    <ChartContainer config={combinedChartConfig} className="w-full" style={{ height }}>
      <LineChart data={formattedData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
        />
        <YAxis
          yAxisId="left"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
          tickFormatter={(value) => `${value}ms`}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          className="text-muted-foreground"
          tickFormatter={(value) => `${value}%`}
        />
        <ChartTooltip content={<ChartTooltipContent />} />
        <ChartLegend content={<ChartLegendContent />} />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="latency_ms"
          stroke="var(--color-latency_ms)"
          strokeWidth={2}
          dot={false}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="error_rate"
          stroke="var(--color-error_rate)"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ChartContainer>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function HealthMetricsChart({
  data,
  providerId,
  providerName,
  isLoading = false,
  error = null,
  height = 250,
  showLatency = true,
  showErrorRate = true,
  showCombined = true,
  onTimeRangeChange,
  className,
}: HealthMetricsChartProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>("24h");
  const [activeTab, setActiveTab] = useState<"latency" | "errors" | "combined">("latency");

  const stats = useMemo(() => calculateStats(data), [data]);

  const handleTimeRangeChange = useCallback(
    (range: TimeRange) => {
      setTimeRange(range);
      onTimeRangeChange?.(range);
    },
    [onTimeRangeChange]
  );

  // Loading state
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-64" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[250px] w-full" />
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (error) {
    return (
      <Card className={cn("border-destructive", className)}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
            Error Loading Metrics
          </CardTitle>
          <CardDescription>{error.message}</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  // Empty state
  if (data.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Health Metrics</CardTitle>
          <CardDescription>No historical data available</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[200px]">
          <p className="text-muted-foreground">
            Metrics will appear here once the provider starts receiving requests
          </p>
        </CardContent>
      </Card>
    );
  }

  // Determine which tabs to show
  const tabs: Array<{ value: string; label: string; show: boolean }> = [
    { value: "latency", label: "Latency", show: showLatency },
    { value: "errors", label: "Error Rate", show: showErrorRate },
    { value: "combined", label: "Combined", show: showCombined },
  ].filter((t) => t.show);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              {providerName ? `${providerName} Health Metrics` : "Health Metrics"}
            </CardTitle>
            <CardDescription>
              Historical performance data and trends
            </CardDescription>
          </div>
          <TimeRangeSelector selected={timeRange} onChange={handleTimeRangeChange} />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Stats Overview */}
        <StatsOverview stats={stats} />

        {/* Charts */}
        {tabs.length > 1 ? (
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as typeof activeTab)}
          >
            <TabsList className="grid w-full" style={{ gridTemplateColumns: `repeat(${tabs.length}, 1fr)` }}>
              {tabs.map((tab) => (
                <TabsTrigger key={tab.value} value={tab.value}>
                  {tab.label}
                </TabsTrigger>
              ))}
            </TabsList>

            {showLatency && (
              <TabsContent value="latency" className="mt-4">
                <LatencyChart data={data} timeRange={timeRange} height={height} />
              </TabsContent>
            )}

            {showErrorRate && (
              <TabsContent value="errors" className="mt-4">
                <ErrorRateChart data={data} timeRange={timeRange} height={height} />
              </TabsContent>
            )}

            {showCombined && (
              <TabsContent value="combined" className="mt-4">
                <CombinedChart data={data} timeRange={timeRange} height={height} />
              </TabsContent>
            )}
          </Tabs>
        ) : (
          <div className="mt-4">
            {showLatency && <LatencyChart data={data} timeRange={timeRange} height={height} />}
            {!showLatency && showErrorRate && (
              <ErrorRateChart data={data} timeRange={timeRange} height={height} />
            )}
            {!showLatency && !showErrorRate && showCombined && (
              <CombinedChart data={data} timeRange={timeRange} height={height} />
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default HealthMetricsChart;
