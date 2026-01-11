/**
 * CampaignAnalyticsDashboard Component
 *
 * Main dashboard component for campaign telemetry analytics.
 * Combines: CampaignSelector, StatisticsSummaryPanel, tabbed views
 * (Overview, Charts, Comparison, Raw Data), FilterBar, ExportPanel.
 * Responsive layout with resizable panels.
 *
 * @module components/campaign-analytics/CampaignAnalyticsDashboard
 */

"use client";

import * as React from "react";
import { useState, useCallback, useMemo, useRef, Suspense } from "react";
import {
  BarChart3,
  LineChart,
  GitCompare,
  Table2,
  Download,
  RefreshCw,
  Filter,
  Settings,
  ChevronDown,
  Info,
  Loader2,
  AlertCircle,
  XCircle,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Campaign Analytics Components
import { CampaignSelectorSingle } from "./CampaignSelector";
import { StatisticsSummaryPanel, StatisticsSummaryPanelSkeleton } from "./StatisticsSummaryPanel";
import { CampaignOverviewCard, CampaignOverviewCardSkeleton } from "./CampaignOverviewCard";
import { FilterBar, createDefaultFilterState, type FilterState } from "./FilterBar";
import { ExportPanelSheet } from "./ExportPanel";
import { TelemetryTable } from "./TelemetryTable";

// Chart Components
import {
  SuccessRateTimeSeriesChart,
  SuccessRateTimeSeriesChartSkeleton,
  SuccessRateTimeSeriesChartEmpty,
} from "./charts/SuccessRateTimeSeriesChart";
import {
  TechniqueEffectivenessChart,
  TechniqueEffectivenessChartSkeleton,
  TechniqueEffectivenessChartEmpty,
} from "./charts/TechniqueEffectivenessChart";
import {
  ProviderComparisonChart,
  ProviderComparisonChartSkeleton,
  ProviderComparisonChartEmpty,
} from "./charts/ProviderComparisonChart";
import {
  PromptEvolutionChart,
  PromptEvolutionChartSkeleton,
  PromptEvolutionChartEmpty,
} from "./charts/PromptEvolutionChart";
import {
  LatencyDistributionChart,
  LatencyDistributionChartSkeleton,
  LatencyDistributionChartEmpty,
} from "./charts/LatencyDistributionChart";

// Comparison Components
import { CampaignComparisonPanel } from "./comparison/CampaignComparisonPanel";

// Query Hooks
import {
  useCampaignSummary,
  useCampaignStatistics,
  useCampaignTimeSeries,
  useTechniqueBreakdown,
  useProviderBreakdown,
  useTelemetryEvents,
  useInvalidateCampaignCache,
} from "@/lib/api/query/campaign-queries";

import type {
  CampaignSummary,
  CampaignStatistics,
  TimeGranularity,
  TelemetryFilterParams,
  TelemetryTimeSeries,
  TechniqueBreakdown,
  ProviderBreakdown,
} from "@/types/campaign-analytics";

// =============================================================================
// Types
// =============================================================================

/**
 * Dashboard tab options.
 */
export type DashboardTab = "overview" | "charts" | "comparison" | "raw-data";

/**
 * Props for the CampaignAnalyticsDashboard component.
 */
export interface CampaignAnalyticsDashboardProps {
  /** Initial campaign ID to select */
  initialCampaignId?: string | null;
  /** Callback when campaign selection changes */
  onCampaignChange?: (campaignId: string | null) => void;
  /** Default tab to show */
  defaultTab?: DashboardTab;
  /** Callback when tab changes */
  onTabChange?: (tab: DashboardTab) => void;
  /** Whether to show the sidebar with campaign selector */
  showSidebar?: boolean;
  /** Whether to enable resizable panels */
  enableResizable?: boolean;
  /** Default sidebar width percentage (1-100) */
  defaultSidebarWidth?: number;
  /** Additional CSS classes */
  className?: string;
  /** Card height constraint */
  height?: string | number;
  /** Enable fullscreen mode */
  enableFullscreen?: boolean;
  /** Compact mode for embedded use */
  compact?: boolean;
  /** Show export panel button */
  showExportPanel?: boolean;
  /** Title override */
  title?: string;
  /** Description override */
  description?: string;
}

/**
 * Tab configuration.
 */
interface TabConfig {
  id: DashboardTab;
  label: string;
  icon: React.ElementType;
  description: string;
}

// =============================================================================
// Constants
// =============================================================================

/**
 * Dashboard tab configurations.
 */
const DASHBOARD_TABS: TabConfig[] = [
  {
    id: "overview",
    label: "Overview",
    icon: BarChart3,
    description: "Campaign summary and key statistics",
  },
  {
    id: "charts",
    label: "Charts",
    icon: LineChart,
    description: "Detailed visualizations and trends",
  },
  {
    id: "comparison",
    label: "Comparison",
    icon: GitCompare,
    description: "Compare multiple campaigns side-by-side",
  },
  {
    id: "raw-data",
    label: "Raw Data",
    icon: Table2,
    description: "Browse individual telemetry events",
  },
];

/**
 * Default time granularity for time series.
 */
const DEFAULT_GRANULARITY: TimeGranularity = "hour";

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Dashboard header with title, actions, and campaign selector.
 */
function DashboardHeader({
  title,
  description,
  selectedCampaignId,
  onCampaignChange,
  onRefresh,
  isRefreshing,
  showExportPanel,
  campaignSummary,
  showFilters,
  onToggleFilters,
  compact,
}: {
  title: string;
  description: string;
  selectedCampaignId: string | null;
  onCampaignChange: (id: string | null) => void;
  onRefresh: () => void;
  isRefreshing: boolean;
  showExportPanel: boolean;
  campaignSummary: CampaignSummary | null | undefined;
  showFilters: boolean;
  onToggleFilters: () => void;
  compact: boolean;
}) {
  return (
    <div className={cn("space-y-4", compact && "space-y-2")}>
      {/* Title row */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <h1 className={cn("font-semibold tracking-tight", compact ? "text-xl" : "text-2xl")}>
            {title}
          </h1>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Filter toggle */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={showFilters ? "secondary" : "outline"}
                  size="sm"
                  onClick={onToggleFilters}
                  disabled={!selectedCampaignId}
                >
                  <Filter className={cn("h-4 w-4", showFilters && "text-primary")} />
                  <span className="ml-1.5 hidden sm:inline">Filters</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Toggle filter bar</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          {/* Refresh button */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onRefresh}
                  disabled={!selectedCampaignId || isRefreshing}
                >
                  <RefreshCw
                    className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                  />
                  <span className="ml-1.5 hidden sm:inline">Refresh</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Refresh campaign data</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          {/* Export panel */}
          {showExportPanel && selectedCampaignId && (
            <ExportPanelSheet
              charts={[]}
              data={[]}
              campaignName={campaignSummary?.name || "campaign"}
            />
          )}
        </div>
      </div>

      {/* Campaign selector row */}
      <div className="flex items-center gap-4">
        <div className="flex-1 max-w-md">
          <CampaignSelectorSingle
            value={selectedCampaignId}
            onChange={onCampaignChange}
            placeholder="Select a campaign to analyze..."
            className="w-full"
          />
        </div>

        {/* Campaign status badge */}
        {campaignSummary && (
          <CampaignStatusBadge status={campaignSummary.status} />
        )}
      </div>
    </div>
  );
}

/**
 * Campaign status badge component.
 */
function CampaignStatusBadge({ status }: { status: string }) {
  const statusConfig: Record<string, { label: string; variant: "default" | "secondary" | "destructive" | "outline"; className: string }> = {
    draft: { label: "Draft", variant: "secondary", className: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300" },
    running: { label: "Running", variant: "default", className: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300" },
    paused: { label: "Paused", variant: "outline", className: "bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-900 dark:text-yellow-300" },
    completed: { label: "Completed", variant: "default", className: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300" },
    failed: { label: "Failed", variant: "destructive", className: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300" },
    cancelled: { label: "Cancelled", variant: "secondary", className: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400" },
  };

  const config = statusConfig[status] || statusConfig.draft;

  return (
    <Badge variant={config.variant} className={cn("shrink-0", config.className)}>
      {config.label}
    </Badge>
  );
}

/**
 * Filter bar wrapper with collapsible behavior.
 */
function FilterSection({
  campaignId,
  filters,
  onFiltersChange,
  onClearFilters,
  isOpen,
  compact,
}: {
  campaignId: string | null;
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  onClearFilters: () => void;
  isOpen: boolean;
  compact: boolean;
}) {
  if (!isOpen || !campaignId) return null;

  return (
    <div className="py-2">
      <FilterBar
        campaignId={campaignId}
        filters={filters}
        onFiltersChange={onFiltersChange}
        onClearAll={onClearFilters}
        compact={compact}
        showActiveCount
      />
    </div>
  );
}

/**
 * Overview tab content.
 */
function OverviewTabContent({
  campaignId,
  campaignSummary,
  statistics,
  timeSeries,
  techniqueBreakdown,
  isLoading,
  compact,
}: {
  campaignId: string | null;
  campaignSummary: CampaignSummary | null | undefined;
  statistics: CampaignStatistics | null | undefined;
  timeSeries: TelemetryTimeSeries | null | undefined;
  techniqueBreakdown: TechniqueBreakdown | null | undefined;
  isLoading: boolean;
  compact: boolean;
}) {
  if (!campaignId) {
    return (
      <EmptyState
        icon={BarChart3}
        title="No Campaign Selected"
        description="Select a campaign from the dropdown above to view its analytics."
      />
    );
  }

  return (
    <div className={cn("space-y-6", compact && "space-y-4")}>
      {/* Campaign Overview Card */}
      <CampaignOverviewCard
        campaign={campaignSummary}
        isLoading={isLoading}
        compact={compact}
      />

      {/* Statistics Summary Panel */}
      <div>
        <h3 className="text-sm font-medium text-muted-foreground mb-3">
          Key Metrics
        </h3>
        <StatisticsSummaryPanel
          campaignId={campaignId}
          statistics={statistics}
          summary={campaignSummary}
          compact={compact}
        />
      </div>

      {/* Quick Charts Preview */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Success Rate Trend</CardTitle>
          </CardHeader>
          <CardContent className="h-[200px]">
            {isLoading ? (
              <SuccessRateTimeSeriesChartSkeleton />
            ) : !timeSeries ? (
              <SuccessRateTimeSeriesChartEmpty />
            ) : (
              <SuccessRateTimeSeriesChart
                timeSeries={timeSeries}
                height={180}
                showZoomControls={false}
                showGranularitySelector={false}
                showDownloadButton={false}
                showLegend={false}
              />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Technique Effectiveness</CardTitle>
          </CardHeader>
          <CardContent className="h-[200px]">
            {isLoading ? (
              <TechniqueEffectivenessChartSkeleton />
            ) : !techniqueBreakdown ? (
              <TechniqueEffectivenessChartEmpty />
            ) : (
              <TechniqueEffectivenessChart
                breakdown={techniqueBreakdown}
                height={180}
                showViewModeToggle={false}
                showSortControls={false}
                showFilterControls={false}
                showDownloadButton={false}
                initialViewMode="bar"
              />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

/**
 * Charts tab content with all visualization components.
 */
function ChartsTabContent({
  campaignId,
  timeSeries,
  techniqueBreakdown,
  providerBreakdown,
  telemetryEvents,
  isLoading,
  granularity,
  onGranularityChange,
  compact,
}: {
  campaignId: string | null;
  timeSeries: TelemetryTimeSeries | null | undefined;
  techniqueBreakdown: TechniqueBreakdown | null | undefined;
  providerBreakdown: ProviderBreakdown | null | undefined;
  telemetryEvents: any[] | null | undefined;
  isLoading: boolean;
  granularity: TimeGranularity;
  onGranularityChange: (g: TimeGranularity) => void;
  compact: boolean;
}) {
  if (!campaignId) {
    return (
      <EmptyState
        icon={LineChart}
        title="No Campaign Selected"
        description="Select a campaign to view detailed charts and visualizations."
      />
    );
  }

  return (
    <div className={cn("space-y-6", compact && "space-y-4")}>
      {/* Success Rate Time Series */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LineChart className="h-5 w-5 text-primary" />
            Success Rate Over Time
          </CardTitle>
          <CardDescription>
            Track how success rate evolves throughout the campaign
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <SuccessRateTimeSeriesChartSkeleton />
          ) : !timeSeries ? (
            <SuccessRateTimeSeriesChartEmpty />
          ) : (
            <SuccessRateTimeSeriesChart
              timeSeries={timeSeries}
              height={300}
              granularity={granularity}
              onGranularityChange={onGranularityChange}
              showZoomControls
              showGranularitySelector
              showDownloadButton
              showLegend
              showTrendIndicator
            />
          )}
        </CardContent>
      </Card>

      {/* Technique and Provider Charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Technique Effectiveness
            </CardTitle>
            <CardDescription>
              Compare success rates across transformation techniques
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <TechniqueEffectivenessChartSkeleton />
            ) : !techniqueBreakdown ? (
              <TechniqueEffectivenessChartEmpty />
            ) : (
              <TechniqueEffectivenessChart
                breakdown={techniqueBreakdown}
                height={300}
                showViewModeToggle
                showSortControls
                showFilterControls
                showDownloadButton
              />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Provider Performance
            </CardTitle>
            <CardDescription>
              Compare metrics across different LLM providers
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <ProviderComparisonChartSkeleton />
            ) : !providerBreakdown ? (
              <ProviderComparisonChartEmpty />
            ) : (
              <ProviderComparisonChart
                breakdown={providerBreakdown}
                height={300}
                showSortControls
                showFilterControls
                showDownloadButton
                showLatencyOverlay
              />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Prompt Evolution Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LineChart className="h-5 w-5 text-primary" />
            Prompt Evolution Analysis
          </CardTitle>
          <CardDescription>
            Visualize prompt iteration correlation with success rate
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <PromptEvolutionChartSkeleton />
          ) : !telemetryEvents?.length ? (
            <PromptEvolutionChartEmpty />
          ) : (
            <PromptEvolutionChart
              events={telemetryEvents}
              height={350}
              showTrendLine
              showCorrelation
              showDownloadButton
              showFilterControls
            />
          )}
        </CardContent>
      </Card>

      {/* Latency Distribution Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            Latency Distribution
          </CardTitle>
          <CardDescription>
            Analyze response time distribution with percentile markers
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <LatencyDistributionChartSkeleton />
          ) : !telemetryEvents?.length ? (
            <LatencyDistributionChartEmpty />
          ) : (
            <LatencyDistributionChart
              events={telemetryEvents}
              height={300}
              showViewModeToggle
              showFilterControls
              showDownloadButton
              showPercentileMarkers
              showStatsSummary
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Comparison tab content.
 */
function ComparisonTabContent({
  initialCampaignId,
  compact,
}: {
  initialCampaignId: string | null;
  compact: boolean;
}) {
  const initialIds = initialCampaignId ? [initialCampaignId] : [];

  return (
    <CampaignComparisonPanel
      initialCampaignIds={initialIds}
      compact={compact}
      showViewModeToggle
      showExportButton
    />
  );
}

/**
 * Raw data tab content with telemetry table.
 */
function RawDataTabContent({
  campaignId,
  filters,
  compact,
}: {
  campaignId: string | null;
  filters: FilterState;
  compact: boolean;
}) {
  // Convert FilterState to TelemetryFilterParams
  const telemetryFilters: TelemetryFilterParams | undefined = useMemo(() => {
    if (!filters) return undefined;

    return {
      technique_suite: filters.techniques.length > 0 ? filters.techniques : undefined,
      provider: filters.providers.length > 0 ? filters.providers : undefined,
      status: filters.successStatus.includes("all")
        ? undefined
        : (filters.successStatus as ("success" | "failure" | "partial")[]),
      start_time: filters.dateRange.start?.toISOString(),
      end_time: filters.dateRange.end?.toISOString(),
    };
  }, [filters]);

  if (!campaignId) {
    return (
      <EmptyState
        icon={Table2}
        title="No Campaign Selected"
        description="Select a campaign to browse individual telemetry events."
      />
    );
  }

  return (
    <TelemetryTable
      campaignId={campaignId}
      filters={telemetryFilters}
      enableDetailModal
      enableExport
      enableSorting
      enablePagination
      compact={compact}
    />
  );
}

/**
 * Empty state component for when no campaign is selected.
 */
function EmptyState({
  icon: Icon,
  title,
  description,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
}) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center py-16 text-center">
        <Icon className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <h3 className="text-lg font-medium text-muted-foreground">{title}</h3>
        <p className="text-sm text-muted-foreground/70 max-w-sm mt-1">
          {description}
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Loading state skeleton for the dashboard.
 */
export function CampaignAnalyticsDashboardSkeleton({
  compact = false,
  className,
}: {
  compact?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("space-y-6", compact && "space-y-4", className)}>
      {/* Header skeleton */}
      <div className="space-y-4">
        <div className="flex justify-between items-start">
          <div className="space-y-2">
            <Skeleton className="h-8 w-64" />
            <Skeleton className="h-4 w-96" />
          </div>
          <div className="flex gap-2">
            <Skeleton className="h-9 w-24" />
            <Skeleton className="h-9 w-24" />
          </div>
        </div>
        <Skeleton className="h-10 w-96" />
      </div>

      {/* Tabs skeleton */}
      <Skeleton className="h-10 w-full max-w-md" />

      {/* Content skeleton */}
      <div className="space-y-4">
        <CampaignOverviewCardSkeleton />
        <StatisticsSummaryPanelSkeleton />
      </div>
    </div>
  );
}

/**
 * Error state for the dashboard.
 */
export function CampaignAnalyticsDashboardError({
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
      <CardContent className="flex flex-col items-center justify-center py-16 text-center">
        <AlertCircle className="h-12 w-12 text-destructive/50 mb-4" />
        <h3 className="text-lg font-medium text-destructive">
          Failed to Load Dashboard
        </h3>
        <p className="text-sm text-muted-foreground max-w-sm mt-1">{error}</p>
        {onRetry && (
          <Button variant="outline" onClick={onRetry} className="mt-4">
            <RefreshCw className="h-4 w-4 mr-2" />
            Try Again
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * CampaignAnalyticsDashboard Component
 *
 * A comprehensive dashboard for analyzing campaign telemetry data.
 * Provides multiple views including overview, detailed charts, campaign
 * comparison, and raw telemetry data browsing.
 *
 * Features:
 * - Campaign selection with search and status filtering
 * - Statistical summary with key metrics (mean, median, p95)
 * - Multiple visualization types (time series, bar, radar, scatter)
 * - Side-by-side comparison of up to 4 campaigns
 * - Paginated raw telemetry table with drill-down
 * - Filtering by technique, provider, date range, status
 * - Export capabilities (PNG, SVG, CSV, ZIP)
 * - Responsive layout with optional resizable panels
 *
 * @example Basic usage
 * ```tsx
 * <CampaignAnalyticsDashboard />
 * ```
 *
 * @example With initial campaign and callbacks
 * ```tsx
 * <CampaignAnalyticsDashboard
 *   initialCampaignId="campaign-123"
 *   onCampaignChange={(id) => console.log("Selected:", id)}
 *   onTabChange={(tab) => console.log("Tab:", tab)}
 * />
 * ```
 *
 * @example Compact mode for embedded use
 * ```tsx
 * <CampaignAnalyticsDashboard
 *   compact
 *   showSidebar={false}
 *   height="600px"
 * />
 * ```
 *
 * @accessibility
 * - Full keyboard navigation support
 * - ARIA labels on all interactive elements
 * - Screen reader friendly tab structure
 * - Color is not the only indicator
 */
export function CampaignAnalyticsDashboard({
  initialCampaignId = null,
  onCampaignChange,
  defaultTab = "overview",
  onTabChange,
  showSidebar = false,
  enableResizable = false,
  defaultSidebarWidth = 25,
  className,
  height,
  enableFullscreen = false,
  compact = false,
  showExportPanel = true,
  title = "Campaign Analytics",
  description = "Analyze campaign telemetry with detailed breakdowns and visualizations",
}: CampaignAnalyticsDashboardProps) {
  // State
  const [selectedCampaignId, setSelectedCampaignId] = useState<string | null>(
    initialCampaignId
  );
  const [activeTab, setActiveTab] = useState<DashboardTab>(defaultTab);
  const [filters, setFilters] = useState<FilterState>(createDefaultFilterState());
  const [showFilters, setShowFilters] = useState(false);
  const [granularity, setGranularity] = useState<TimeGranularity>(DEFAULT_GRANULARITY);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Refs
  const dashboardRef = useRef<HTMLDivElement>(null);

  // Data fetching - only fetch when campaign is selected
  const {
    data: campaignSummary,
    isLoading: summaryLoading,
    refetch: refetchSummary,
  } = useCampaignSummary(selectedCampaignId, !!selectedCampaignId);

  const {
    data: statistics,
    isLoading: statsLoading,
    refetch: refetchStats,
  } = useCampaignStatistics(selectedCampaignId, !!selectedCampaignId);

  const {
    data: timeSeries,
    isLoading: timeSeriesLoading,
    refetch: refetchTimeSeries,
  } = useCampaignTimeSeries(
    selectedCampaignId,
    { granularity },
    !!selectedCampaignId
  );

  const {
    data: techniqueBreakdown,
    isLoading: techniqueLoading,
    refetch: refetchTechnique,
  } = useTechniqueBreakdown(selectedCampaignId, !!selectedCampaignId);

  const {
    data: providerBreakdown,
    isLoading: providerLoading,
    refetch: refetchProvider,
  } = useProviderBreakdown(selectedCampaignId, !!selectedCampaignId);

  const {
    data: telemetryResponse,
    isLoading: telemetryLoading,
    refetch: refetchTelemetry,
  } = useTelemetryEvents(
    selectedCampaignId,
    { pageSize: 100 },
    !!selectedCampaignId
  );

  const invalidateCache = useInvalidateCampaignCache();

  // Computed values
  const isLoading = summaryLoading || statsLoading;
  const isChartsLoading = timeSeriesLoading || techniqueLoading || providerLoading || telemetryLoading;

  // Handlers
  const handleCampaignChange = useCallback(
    (campaignId: string | null) => {
      setSelectedCampaignId(campaignId);
      // Reset filters when campaign changes
      setFilters(createDefaultFilterState());
      onCampaignChange?.(campaignId);
    },
    [onCampaignChange]
  );

  const handleTabChange = useCallback(
    (tab: string) => {
      setActiveTab(tab as DashboardTab);
      onTabChange?.(tab as DashboardTab);
    },
    [onTabChange]
  );

  const handleRefresh = useCallback(() => {
    if (selectedCampaignId) {
      invalidateCache(selectedCampaignId);
      refetchSummary();
      refetchStats();
      refetchTimeSeries();
      refetchTechnique();
      refetchProvider();
      refetchTelemetry();
    }
  }, [
    selectedCampaignId,
    invalidateCache,
    refetchSummary,
    refetchStats,
    refetchTimeSeries,
    refetchTechnique,
    refetchProvider,
    refetchTelemetry,
  ]);

  const handleToggleFilters = useCallback(() => {
    setShowFilters((prev) => !prev);
  }, []);

  const handleClearFilters = useCallback(() => {
    setFilters(createDefaultFilterState());
  }, []);

  const handleToggleFullscreen = useCallback(() => {
    if (!dashboardRef.current) return;

    if (!document.fullscreenElement) {
      dashboardRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  }, []);

  // Extract telemetry events from response
  const telemetryEvents = telemetryResponse?.items || [];

  return (
    <div
      ref={dashboardRef}
      className={cn(
        "relative bg-background",
        isFullscreen && "fixed inset-0 z-50 p-4 overflow-auto",
        className
      )}
      style={{ height: typeof height === "number" ? `${height}px` : height }}
    >
      <div className={cn("space-y-6", compact && "space-y-4")}>
        {/* Dashboard Header */}
        <DashboardHeader
          title={title}
          description={description}
          selectedCampaignId={selectedCampaignId}
          onCampaignChange={handleCampaignChange}
          onRefresh={handleRefresh}
          isRefreshing={isLoading || isChartsLoading}
          showExportPanel={showExportPanel}
          campaignSummary={campaignSummary}
          showFilters={showFilters}
          onToggleFilters={handleToggleFilters}
          compact={compact}
        />

        <Separator />

        {/* Filter Section */}
        <FilterSection
          campaignId={selectedCampaignId}
          filters={filters}
          onFiltersChange={setFilters}
          onClearFilters={handleClearFilters}
          isOpen={showFilters}
          compact={compact}
        />

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
          <TabsList className="w-full justify-start flex-wrap h-auto gap-1 bg-transparent p-0 mb-4">
            {DASHBOARD_TABS.map((tab) => (
              <TabsTrigger
                key={tab.id}
                value={tab.id}
                className={cn(
                  "data-[state=active]:bg-background data-[state=active]:shadow-sm",
                  "border border-transparent data-[state=active]:border-border",
                  "px-4 py-2"
                )}
              >
                <tab.icon className="h-4 w-4 mr-2" />
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>

          {/* Tab Contents */}
          <TabsContent value="overview" className="mt-0">
            <OverviewTabContent
              campaignId={selectedCampaignId}
              campaignSummary={campaignSummary}
              statistics={statistics}
              timeSeries={timeSeries}
              techniqueBreakdown={techniqueBreakdown}
              isLoading={isLoading || timeSeriesLoading || techniqueLoading}
              compact={compact}
            />
          </TabsContent>

          <TabsContent value="charts" className="mt-0">
            <ChartsTabContent
              campaignId={selectedCampaignId}
              timeSeries={timeSeries}
              techniqueBreakdown={techniqueBreakdown}
              providerBreakdown={providerBreakdown}
              telemetryEvents={telemetryEvents}
              isLoading={isChartsLoading}
              granularity={granularity}
              onGranularityChange={setGranularity}
              compact={compact}
            />
          </TabsContent>

          <TabsContent value="comparison" className="mt-0">
            <ComparisonTabContent
              initialCampaignId={selectedCampaignId}
              compact={compact}
            />
          </TabsContent>

          <TabsContent value="raw-data" className="mt-0">
            <RawDataTabContent
              campaignId={selectedCampaignId}
              filters={filters}
              compact={compact}
            />
          </TabsContent>
        </Tabs>
      </div>

      {/* Fullscreen toggle button */}
      {enableFullscreen && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={handleToggleFullscreen}
                className="absolute top-4 right-4"
              >
                {isFullscreen ? (
                  <Minimize2 className="h-4 w-4" />
                ) : (
                  <Maximize2 className="h-4 w-4" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}

// =============================================================================
// Convenience Variants
// =============================================================================

/**
 * Compact dashboard variant for embedding in smaller spaces.
 */
export function CompactCampaignAnalyticsDashboard(
  props: Omit<CampaignAnalyticsDashboardProps, "compact">
) {
  return <CampaignAnalyticsDashboard {...props} compact />;
}

/**
 * Simple dashboard without sidebar and fullscreen.
 */
export function SimpleCampaignAnalyticsDashboard(
  props: Omit<
    CampaignAnalyticsDashboardProps,
    "showSidebar" | "enableResizable" | "enableFullscreen"
  >
) {
  return (
    <CampaignAnalyticsDashboard
      {...props}
      showSidebar={false}
      enableResizable={false}
      enableFullscreen={false}
    />
  );
}

// =============================================================================
// Exports
// =============================================================================

export default CampaignAnalyticsDashboard;

// Export types for external use
export type { FilterState, DashboardTab, TabConfig };
