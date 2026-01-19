/**
 * Technique Effectiveness Chart Component
 *
 * Visualizes the effectiveness of different attack techniques using bar charts
 * or radar charts, showing success rates, attempt counts, and comparative metrics.
 */

"use client";

import React, { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart3, Radar as RadarIcon, TrendingUp, Zap, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

// Types
export interface TechniqueEffectivenessData {
  technique: string;
  displayName?: string;
  attempts: number;
  successes: number;
  failures: number;
  successRate: number;
  avgLatencyMs: number;
  avgTokens: number;
  avgCost: number;
  category?: string;
  difficulty?: 'easy' | 'medium' | 'hard' | 'expert';
  potencyLevel?: number;
}

export interface TechniqueEffectivenessChartProps {
  data: TechniqueEffectivenessData[];
  campaignId?: string;
  viewMode?: 'bar' | 'radar';
  sortBy?: 'technique' | 'successRate' | 'attempts' | 'latency';
  sortOrder?: 'asc' | 'desc';
  showControls?: boolean;
  showLegend?: boolean;
  height?: number;
  className?: string;
  onTechniqueClick?: (technique: string) => void;
  onDrillDown?: (technique: string) => void;
}

// Utility functions
const getTechniqueColor = (successRate: number): string => {
  if (successRate >= 0.8) return "#22c55e"; // Excellent (Green)
  if (successRate >= 0.6) return "#3b82f6"; // Good (Blue)
  if (successRate >= 0.4) return "#f59e0b"; // Moderate (Yellow)
  if (successRate >= 0.2) return "#f97316"; // Poor (Orange)
  return "#ef4444"; // Critical (Red)
};

const getTechniqueEffectivenessTier = (successRate: number): { label: string; color: string } => {
  if (successRate >= 0.8) return { label: "Excellent", color: "success" };
  if (successRate >= 0.6) return { label: "Good", color: "blue" };
  if (successRate >= 0.4) return { label: "Moderate", color: "yellow" };
  if (successRate >= 0.2) return { label: "Poor", color: "orange" };
  return { label: "Critical", color: "destructive" };
};

const formatTechniqueName = (technique: string): string => {
  return technique
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
};

// Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const tier = getTechniqueEffectivenessTier(data.successRate);

    return (
      <div className="bg-background border rounded-lg shadow-lg p-4 min-w-[200px]">
        <h4 className="font-semibold text-sm mb-2">{formatTechniqueName(label)}</h4>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span>Success Rate:</span>
            <div className="flex items-center gap-2">
              <span className="font-mono">{(data.successRate * 100).toFixed(1)}%</span>
              <Badge variant={tier.color as any} className="text-xs">
                {tier.label}
              </Badge>
            </div>
          </div>
          <div className="flex justify-between">
            <span>Attempts:</span>
            <span className="font-mono">{data.attempts.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span>Successes:</span>
            <span className="font-mono">{data.successes.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span>Avg Latency:</span>
            <span className="font-mono">{data.avgLatencyMs.toFixed(0)}ms</span>
          </div>
          {data.category && (
            <div className="flex justify-between">
              <span>Category:</span>
              <span className="text-muted-foreground">{data.category}</span>
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

// Main Chart Component
export function TechniqueEffectivenessChart({
  data,
  campaignId,
  viewMode = 'bar',
  sortBy = 'successRate',
  sortOrder = 'desc',
  showControls = true,
  showLegend = true,
  height = 400,
  className,
  onTechniqueClick,
  onDrillDown,
}: TechniqueEffectivenessChartProps) {
  const [currentViewMode, setCurrentViewMode] = useState(viewMode);
  const [currentSortBy, setCurrentSortBy] = useState(sortBy);
  const [currentSortOrder, setCurrentSortOrder] = useState(sortOrder);

  // Sort and prepare data
  const sortedData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const sorted = [...data].sort((a, b) => {
      let aValue: any, bValue: any;

      switch (currentSortBy) {
        case 'technique':
          aValue = a.displayName || a.technique;
          bValue = b.displayName || b.technique;
          break;
        case 'successRate':
          aValue = a.successRate;
          bValue = b.successRate;
          break;
        case 'attempts':
          aValue = a.attempts;
          bValue = b.attempts;
          break;
        case 'latency':
          aValue = a.avgLatencyMs;
          bValue = b.avgLatencyMs;
          break;
        default:
          aValue = a.successRate;
          bValue = b.successRate;
      }

      if (currentSortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    // Enhance data with colors and formatted names
    return sorted.map(item => ({
      ...item,
      displayName: item.displayName || formatTechniqueName(item.technique),
      fill: getTechniqueColor(item.successRate),
      successRatePercent: item.successRate * 100,
    }));
  }, [data, currentSortBy, currentSortOrder]);

  // Handle technique click
  const handleTechniqueClick = (techniqueData: any) => {
    if (onTechniqueClick) {
      onTechniqueClick(techniqueData.technique);
    } else if (onDrillDown) {
      onDrillDown(techniqueData.technique);
    }
  };

  // Bar Chart Component
  const BarChartView = () => (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={sortedData}
        margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
      >
        <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
        <XAxis
          dataKey="displayName"
          angle={-45}
          textAnchor="end"
          height={80}
          interval={0}
          fontSize={11}
        />
        <YAxis
          domain={[0, 100]}
          tickFormatter={(value) => `${value}%`}
        />
        <Tooltip content={<CustomTooltip />} />
        {showLegend && (
          <Legend
            content={() => (
              <div className="flex justify-center gap-4 mt-4 text-xs">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-green-500 rounded" />
                  <span>Excellent (≥80%)</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-blue-500 rounded" />
                  <span>Good (60-79%)</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-yellow-500 rounded" />
                  <span>Moderate (40-59%)</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-orange-500 rounded" />
                  <span>Poor (20-39%)</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-red-500 rounded" />
                  <span>Critical (&lt;20%)</span>
                </div>
              </div>
            )}
          />
        )}
        <Bar
          dataKey="successRatePercent"
          radius={[4, 4, 0, 0]}
          cursor="pointer"
          onClick={handleTechniqueClick}
        />
      </BarChart>
    </ResponsiveContainer>
  );

  // Radar Chart Component
  const RadarChartView = () => {
    const radarData = sortedData.slice(0, 8).map(item => ({
      technique: item.displayName.length > 15
        ? item.displayName.substring(0, 15) + "..."
        : item.displayName,
      fullTechnique: item.displayName,
      successRate: item.successRatePercent,
      efficiency: Math.max(0, 100 - (item.avgLatencyMs / 50)), // Normalized efficiency
      reliability: (item.successes / Math.max(item.attempts, 1)) * 100,
    }));

    return (
      <ResponsiveContainer width="100%" height={height}>
        <RadarChart data={radarData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <PolarGrid />
          <PolarAngleAxis dataKey="technique" />
          <PolarRadiusAxis
            angle={30}
            domain={[0, 100]}
            tickFormatter={(value) => `${value}%`}
          />
          <Radar
            name="Success Rate"
            dataKey="successRate"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.3}
            strokeWidth={2}
          />
          <Radar
            name="Efficiency"
            dataKey="efficiency"
            stroke="#10b981"
            fill="#10b981"
            fillOpacity={0.2}
            strokeWidth={2}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    );
  };

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">
            Technique Effectiveness
          </CardTitle>
          {showControls && (
            <div className="flex items-center gap-2">
              <Select value={currentSortBy} onValueChange={(value: "successRate" | "technique" | "latency" | "attempts") => setCurrentSortBy(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="successRate">Success Rate</SelectItem>
                  <SelectItem value="attempts">Attempts</SelectItem>
                  <SelectItem value="technique">Name</SelectItem>
                  <SelectItem value="latency">Latency</SelectItem>
                </SelectContent>
              </Select>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
              >
                {currentSortOrder === 'desc' ? '↓' : '↑'}
              </Button>

              <Tabs value={currentViewMode} onValueChange={(v) => setCurrentViewMode(v as 'bar' | 'radar')}>
                <TabsList>
                  <TabsTrigger value="bar" className="flex items-center gap-1">
                    <BarChart3 className="h-3 w-3" />
                    Bar
                  </TabsTrigger>
                  <TabsTrigger value="radar" className="flex items-center gap-1">
                    <RadarIcon className="h-3 w-3" />
                    Radar
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          )}
        </div>

        {/* Summary Stats */}
        <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Zap className="h-3 w-3" />
            <span>{sortedData.length} techniques analyzed</span>
          </div>
          {sortedData.length > 0 && (
            <>
              <div className="flex items-center gap-1">
                <TrendingUp className="h-3 w-3" />
                <span>
                  Avg: {(sortedData.reduce((acc, t) => acc + t.successRate, 0) / sortedData.length * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" />
                <span>
                  Top: {formatTechniqueName(sortedData[0]?.technique || '')}
                  ({(sortedData[0]?.successRate * 100).toFixed(1)}%)
                </span>
              </div>
            </>
          )}
        </div>
      </CardHeader>

      <CardContent>
        {currentViewMode === 'bar' ? <BarChartView /> : <RadarChartView />}
      </CardContent>
    </Card>
  );
}

// Skeleton Loading Component
export function TechniqueEffectivenessChartSkeleton() {
  return (
    <Card className="w-full">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-48" />
          <div className="flex gap-2">
            <Skeleton className="h-8 w-32" />
            <Skeleton className="h-8 w-12" />
            <Skeleton className="h-8 w-24" />
          </div>
        </div>
        <div className="flex gap-4">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-36" />
        </div>
      </CardHeader>
      <CardContent>
        <Skeleton className="w-full h-[400px]" />
      </CardContent>
    </Card>
  );
}

// Empty State Component
export function TechniqueEffectivenessChartEmpty() {
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg font-semibold">
          Technique Effectiveness
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center justify-center h-[400px] text-muted-foreground">
          <BarChart3 className="h-16 w-16 mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No technique data available</h3>
          <p className="text-sm text-center max-w-md">
            Run campaigns with different techniques to see effectiveness analysis and comparison charts.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// Variant Components
export function SimpleTechniqueChart(props: Omit<TechniqueEffectivenessChartProps, 'showControls'>) {
  return <TechniqueEffectivenessChart {...props} showControls={false} />;
}

export function TechniqueRadarChart(props: Omit<TechniqueEffectivenessChartProps, 'viewMode'>) {
  return <TechniqueEffectivenessChart {...props} viewMode="radar" />;
}

export function CompactTechniqueChart(props: Omit<TechniqueEffectivenessChartProps, 'height' | 'showControls'>) {
  return <TechniqueEffectivenessChart {...props} height={300} showControls={false} />;
}

// Default export
export default TechniqueEffectivenessChart;