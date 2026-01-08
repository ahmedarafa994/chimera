"use client";

import * as React from "react";
import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { getApiConfig, getApiHeaders } from "@/lib/api-config";
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Layers,
} from "lucide-react";

// Types for model comparison data
interface ModelPerformance {
  model_name: string;
  total_attempts: number;
  successful_attempts: number;
  success_rate: number;
  avg_response_time_ms: number;
  avg_confidence_score: number;
  technique_specific_stats?: {
    bypass_count: number;
    detection_rate: number;
    avg_turns_to_success: number;
  };
}

interface ComparisonResult {
  technique_id: string;
  technique_name: string;
  time_window_hours: number;
  models: ModelPerformance[];
  comparison_summary: {
    best_performer: string;
    worst_performer: string;
    avg_success_rate_delta: number;
    recommendation: string;
  };
  generated_at: string;
}

interface ModelComparisonViewProps {
  className?: string;
  defaultTechniqueId?: string;
  availableTechniques?: Array<{ id: string; name: string }>;
}

// Performance indicator component
function PerformanceIndicator({
  value,
  threshold = 50,
  inverted = false,
}: {
  value: number;
  threshold?: number;
  inverted?: boolean;
}) {
  const isGood = inverted ? value < threshold : value >= threshold;
  const Icon = isGood ? TrendingUp : value === threshold ? Minus : TrendingDown;
  const color = isGood
    ? "text-emerald-500"
    : value === threshold
      ? "text-amber-500"
      : "text-rose-500";

  return <Icon className={cn("size-4", color)} />;
}

// Model card component for side-by-side display
function ModelCard({
  model,
  isBest,
  isWorst,
}: {
  model: ModelPerformance;
  isBest: boolean;
  isWorst: boolean;
}) {
  return (
    <Card
      className={cn(
        "relative overflow-hidden transition-all duration-200",
        isBest && "ring-2 ring-emerald-500/50",
        isWorst && "ring-2 ring-rose-500/50"
      )}
    >
      {isBest && (
        <div className="absolute top-2 right-2">
          <Badge className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20">
            <CheckCircle2 className="size-3 mr-1" />
            Best
          </Badge>
        </div>
      )}
      {isWorst && (
        <div className="absolute top-2 right-2">
          <Badge variant="destructive" className="bg-rose-500/10 text-rose-500 border-rose-500/20">
            <XCircle className="size-3 mr-1" />
            Lowest
          </Badge>
        </div>
      )}

      <CardHeader className="pb-3">
        <CardTitle className="text-lg font-semibold truncate pr-16">
          {model.model_name}
        </CardTitle>
        <CardDescription className="text-xs text-muted-foreground">
          {model.total_attempts} total attempts
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Success Rate - Primary Metric */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Success Rate</span>
            <div className="flex items-center gap-2">
              <PerformanceIndicator value={model.success_rate} threshold={50} />
              <span className="text-2xl font-bold tabular-nums">
                {model.success_rate.toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-500",
                model.success_rate >= 70
                  ? "bg-emerald-500"
                  : model.success_rate >= 40
                    ? "bg-amber-500"
                    : "bg-rose-500"
              )}
              style={{ width: `${Math.min(100, model.success_rate)}%` }}
            />
          </div>
        </div>

        {/* Secondary Metrics */}
        <div className="grid grid-cols-2 gap-3 pt-2">
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Avg Response</span>
            <p className="text-sm font-medium tabular-nums">
              {model.avg_response_time_ms.toFixed(0)}ms
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Confidence</span>
            <p className="text-sm font-medium tabular-nums">
              {(model.avg_confidence_score * 100).toFixed(1)}%
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Successful</span>
            <p className="text-sm font-medium tabular-nums text-emerald-500">
              {model.successful_attempts}
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Failed</span>
            <p className="text-sm font-medium tabular-nums text-rose-500">
              {model.total_attempts - model.successful_attempts}
            </p>
          </div>
        </div>

        {/* Technique-specific stats */}
        {model.technique_specific_stats && (
          <div className="pt-3 border-t border-border">
            <p className="text-xs text-muted-foreground mb-2">Technique Stats</p>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Bypasses</span>
                <p className="font-medium">{model.technique_specific_stats.bypass_count}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Detection</span>
                <p className="font-medium">
                  {(model.technique_specific_stats.detection_rate * 100).toFixed(0)}%
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Avg Turns</span>
                <p className="font-medium">
                  {model.technique_specific_stats.avg_turns_to_success.toFixed(1)}
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Loading skeleton
function ComparisonSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-10 w-32" />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-6 w-32" />
              <Skeleton className="h-4 w-24" />
            </CardHeader>
            <CardContent className="space-y-4">
              <Skeleton className="h-8 w-full" />
              <div className="grid grid-cols-2 gap-3">
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

// Main component
export function ModelComparisonView({
  className,
  defaultTechniqueId = "autodan",
  availableTechniques = [
    { id: "autodan", name: "AutoDAN" },
    { id: "pair", name: "PAIR" },
    { id: "gcg", name: "GCG" },
    { id: "tap", name: "TAP" },
    { id: "gptfuzz", name: "GPTFuzz" },
  ],
}: ModelComparisonViewProps) {
  const [selectedTechnique, setSelectedTechnique] = useState(defaultTechniqueId);
  const [timeWindow, setTimeWindow] = useState<number>(168); // 7 days
  const [comparisonData, setComparisonData] = useState<ComparisonResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchComparison = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const config = getApiConfig();
      const headers = getApiHeaders();
      const baseUrl = config.backendApiUrl;

      const response = await fetch(
        `${baseUrl}/jailbreak-quality/analytics/model-comparison?technique_id=${selectedTechnique}&time_window_hours=${timeWindow}`,
        { headers }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch comparison: ${response.statusText}`);
      }

      const data = await response.json();
      setComparisonData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load comparison data");
    } finally {
      setLoading(false);
    }
  }, [selectedTechnique, timeWindow]);

  useEffect(() => {
    fetchComparison();
  }, [fetchComparison]);

  const timeWindowOptions = [
    { value: 24, label: "Last 24 hours" },
    { value: 72, label: "Last 3 days" },
    { value: 168, label: "Last 7 days" },
    { value: 720, label: "Last 30 days" },
  ];

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Layers className="size-5 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-semibold">Cross-Model Comparison</h2>
            <p className="text-sm text-muted-foreground">
              Compare jailbreak performance across models
            </p>
          </div>
        </div>

        <Button variant="outline" size="sm" onClick={fetchComparison} disabled={loading}>
          <RefreshCw className={cn("size-4 mr-2", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <Select value={selectedTechnique} onValueChange={setSelectedTechnique}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select technique" />
          </SelectTrigger>
          <SelectContent>
            {availableTechniques.map((tech) => (
              <SelectItem key={tech.id} value={tech.id}>
                {tech.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={timeWindow.toString()} onValueChange={(v) => setTimeWindow(parseInt(v))}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Time window" />
          </SelectTrigger>
          <SelectContent>
            {timeWindowOptions.map((opt) => (
              <SelectItem key={opt.value} value={opt.value.toString()}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Content */}
      {loading ? (
        <ComparisonSkeleton />
      ) : error ? (
        <Card className="border-destructive/50">
          <CardContent className="flex items-center gap-3 py-6">
            <AlertTriangle className="size-5 text-destructive" />
            <div>
              <p className="font-medium">Failed to load comparison</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </div>
            <Button variant="outline" size="sm" onClick={fetchComparison} className="ml-auto">
              Retry
            </Button>
          </CardContent>
        </Card>
      ) : comparisonData ? (
        <Tabs defaultValue="cards" className="space-y-4">
          <TabsList>
            <TabsTrigger value="cards">
              <Layers className="size-4 mr-2" />
              Cards View
            </TabsTrigger>
            <TabsTrigger value="table">
              <BarChart3 className="size-4 mr-2" />
              Table View
            </TabsTrigger>
          </TabsList>

          {/* Summary Card */}
          {comparisonData.comparison_summary && (
            <Card className="bg-muted/30">
              <CardContent className="py-4">
                <div className="flex flex-wrap items-center gap-6 text-sm">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="size-4 text-emerald-500" />
                    <span className="text-muted-foreground">Best:</span>
                    <span className="font-medium">
                      {comparisonData.comparison_summary.best_performer}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <XCircle className="size-4 text-rose-500" />
                    <span className="text-muted-foreground">Lowest:</span>
                    <span className="font-medium">
                      {comparisonData.comparison_summary.worst_performer}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <TrendingUp className="size-4 text-primary" />
                    <span className="text-muted-foreground">Delta:</span>
                    <span className="font-medium">
                      {comparisonData.comparison_summary.avg_success_rate_delta.toFixed(1)}%
                    </span>
                  </div>
                </div>
                {comparisonData.comparison_summary.recommendation && (
                  <p className="mt-3 text-sm text-muted-foreground border-t border-border pt-3">
                    <strong>Recommendation:</strong>{" "}
                    {comparisonData.comparison_summary.recommendation}
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          <TabsContent value="cards" className="mt-0">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {comparisonData.models.map((model) => (
                <ModelCard
                  key={model.model_name}
                  model={model}
                  isBest={model.model_name === comparisonData.comparison_summary?.best_performer}
                  isWorst={model.model_name === comparisonData.comparison_summary?.worst_performer}
                />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="table" className="mt-0">
            <Card>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">
                        Model
                      </th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">
                        Success Rate
                      </th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">
                        Attempts
                      </th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">
                        Successful
                      </th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">
                        Avg Response
                      </th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">
                        Confidence
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonData.models
                      .sort((a, b) => b.success_rate - a.success_rate)
                      .map((model, idx) => (
                        <tr
                          key={model.model_name}
                          className={cn(
                            "border-b border-border last:border-0",
                            idx === 0 && "bg-emerald-500/5",
                            idx === comparisonData.models.length - 1 && "bg-rose-500/5"
                          )}
                        >
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{model.model_name}</span>
                              {idx === 0 && (
                                <Badge className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20 text-xs">
                                  Best
                                </Badge>
                              )}
                            </div>
                          </td>
                          <td className="py-3 px-4 text-right">
                            <div className="flex items-center justify-end gap-2">
                              <PerformanceIndicator value={model.success_rate} />
                              <span className="font-medium tabular-nums">
                                {model.success_rate.toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4 text-right tabular-nums">
                            {model.total_attempts}
                          </td>
                          <td className="py-3 px-4 text-right tabular-nums text-emerald-500">
                            {model.successful_attempts}
                          </td>
                          <td className="py-3 px-4 text-right tabular-nums">
                            {model.avg_response_time_ms.toFixed(0)}ms
                          </td>
                          <td className="py-3 px-4 text-right tabular-nums">
                            {(model.avg_confidence_score * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12 text-center">
            <BarChart3 className="size-12 text-muted-foreground/50 mb-4" />
            <p className="text-muted-foreground">No comparison data available</p>
            <p className="text-sm text-muted-foreground/70">
              Select a technique and time window to view model comparisons
            </p>
          </CardContent>
        </Card>
      )}

      {/* Timestamp */}
      {comparisonData?.generated_at && (
        <p className="text-xs text-muted-foreground text-right">
          Generated at: {new Date(comparisonData.generated_at).toLocaleString()}
        </p>
      )}
    </div>
  );
}

export default ModelComparisonView;