/**
 * CampaignComparisonPanel Component
 *
 * Main comparison view combining:
 * - Multi-select campaign selector (up to 4 campaigns)
 * - Comparison table with metrics and rankings
 * - Normalized metrics radar chart
 * - Clear comparison button
 *
 * @example
 * ```tsx
 * <CampaignComparisonPanel
 *   initialCampaignIds={["id1", "id2"]}
 *   onCampaignChange={(ids) => console.log("Selected:", ids)}
 * />
 * ```
 */

"use client";

import * as React from "react";
import { useMemo, useState, useCallback, useRef } from "react";
import {
  X,
  RefreshCw,
  AlertTriangle,
  GitCompare,
  Download,
  Maximize2,
  Minimize2,
  BarChart3,
  Table as TableIcon,
  Radar as RadarIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useCampaignComparison } from "@/lib/api/query/campaign-queries";
import { CampaignComparisonSelector } from "../CampaignSelector";
import { ComparisonTable, ComparisonTableSkeleton } from "./ComparisonTable";
import {
  NormalizedMetricsChart,
  NormalizedMetricsChartSkeleton,
} from "./NormalizedMetricsChart";

// =============================================================================
// Types
// =============================================================================

/**
 * View mode for the comparison panel
 */
export type ComparisonViewMode = "combined" | "table" | "chart";

/**
 * Props for the CampaignComparisonPanel component.
 */
export interface CampaignComparisonPanelProps {
  /** Initial campaign IDs to compare */
  initialCampaignIds?: string[];
  /** Callback when selection changes */
  onCampaignChange?: (campaignIds: string[]) => void;
  /** Maximum number of campaigns to compare (2-4) */
  maxCampaigns?: number;
  /** Title for the panel */
  title?: string;
  /** Description for the panel */
  description?: string;
  /** Default view mode */
  defaultViewMode?: ComparisonViewMode;
  /** Show view mode toggle */
  showViewModeToggle?: boolean;
  /** Show export button */
  showExportButton?: boolean;
  /** Callback for export action */
  onExport?: (format: "png" | "svg" | "csv") => void;
  /** Allow fullscreen mode */
  allowFullscreen?: boolean;
  /** Compact mode */
  compact?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Card height constraint */
  height?: string | number;
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Header section with campaign selector and actions
 */
function ComparisonHeader({
  selectedIds,
  onSelectionChange,
  maxCampaigns,
  onClear,
  onRefresh,
  isRefreshing,
  viewMode,
  onViewModeChange,
  showViewModeToggle,
  showExportButton,
  onExport,
  allowFullscreen,
  isFullscreen,
  onToggleFullscreen,
  compact,
}: {
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
  maxCampaigns: number;
  onClear: () => void;
  onRefresh: () => void;
  isRefreshing: boolean;
  viewMode: ComparisonViewMode;
  onViewModeChange: (mode: ComparisonViewMode) => void;
  showViewModeToggle: boolean;
  showExportButton: boolean;
  onExport?: (format: "png" | "svg" | "csv") => void;
  allowFullscreen: boolean;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
  compact: boolean;
}) {
  return (
    <div className={cn("flex flex-col gap-4", compact && "gap-2")}>
      {/* Campaign selector row */}
      <div className="flex items-start gap-3 flex-wrap">
        <div className="flex-1 min-w-[300px]">
          <CampaignComparisonSelector
            values={selectedIds}
            onMultiChange={onSelectionChange}
            className="w-full"
            ariaLabel="Select campaigns to compare"
          />
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Clear button */}
          {selectedIds.length > 0 && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={onClear}
                    className="h-9 w-9"
                    aria-label="Clear comparison"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Clear Comparison</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}

          {/* Refresh button */}
          {selectedIds.length >= 2 && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={onRefresh}
                    disabled={isRefreshing}
                    className="h-9 w-9"
                    aria-label="Refresh comparison data"
                  >
                    <RefreshCw
                      className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                    />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Refresh Data</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      </div>

      {/* View mode and export controls */}
      {selectedIds.length >= 2 && (showViewModeToggle || showExportButton || allowFullscreen) && (
        <div className="flex items-center justify-between">
          {/* View mode toggle */}
          {showViewModeToggle && (
            <div className="flex items-center gap-1 p-1 bg-muted rounded-lg">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={viewMode === "combined" ? "secondary" : "ghost"}
                      size="sm"
                      onClick={() => onViewModeChange("combined")}
                      className="h-7 px-2"
                      aria-pressed={viewMode === "combined"}
                    >
                      <BarChart3 className="h-3.5 w-3.5 mr-1" />
                      <span className="text-xs">Combined</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Show table and chart together</TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={viewMode === "table" ? "secondary" : "ghost"}
                      size="sm"
                      onClick={() => onViewModeChange("table")}
                      className="h-7 px-2"
                      aria-pressed={viewMode === "table"}
                    >
                      <TableIcon className="h-3.5 w-3.5 mr-1" />
                      <span className="text-xs">Table</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Show comparison table only</TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={viewMode === "chart" ? "secondary" : "ghost"}
                      size="sm"
                      onClick={() => onViewModeChange("chart")}
                      className="h-7 px-2"
                      aria-pressed={viewMode === "chart"}
                    >
                      <RadarIcon className="h-3.5 w-3.5 mr-1" />
                      <span className="text-xs">Chart</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Show radar chart only</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          )}

          {/* Right side actions */}
          <div className="flex items-center gap-2">
            {/* Export button */}
            {showExportButton && onExport && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onExport("csv")}
                      className="h-8"
                    >
                      <Download className="h-3.5 w-3.5 mr-1.5" />
                      Export
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Export comparison data</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}

            {/* Fullscreen toggle */}
            {allowFullscreen && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={onToggleFullscreen}
                      className="h-8 w-8"
                      aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                    >
                      {isFullscreen ? (
                        <Minimize2 className="h-4 w-4" />
                      ) : (
                        <Maximize2 className="h-4 w-4" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Empty state when no campaigns are selected
 */
function ComparisonEmptyState({
  message = "Select 2-4 campaigns to compare their performance metrics side-by-side.",
}: {
  message?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      <div className="p-4 rounded-full bg-muted/50 mb-4">
        <GitCompare className="h-10 w-10 text-muted-foreground/60" />
      </div>
      <h3 className="font-semibold text-lg mb-2">No Campaigns Selected</h3>
      <p className="text-sm text-muted-foreground max-w-md mb-6">
        {message}
      </p>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline" className="font-normal">
          Minimum: 2 campaigns
        </Badge>
        <Badge variant="outline" className="font-normal">
          Maximum: 4 campaigns
        </Badge>
      </div>
    </div>
  );
}

/**
 * Waiting state when only one campaign is selected
 */
function ComparisonWaitingState({
  selectedCount,
}: {
  selectedCount: number;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      <div className="p-4 rounded-full bg-amber-500/10 mb-4">
        <GitCompare className="h-10 w-10 text-amber-600 dark:text-amber-400" />
      </div>
      <h3 className="font-semibold text-lg mb-2">Select More Campaigns</h3>
      <p className="text-sm text-muted-foreground max-w-md mb-4">
        You've selected {selectedCount} campaign. Select at least one more to see the comparison.
      </p>
      <Badge variant="secondary" className="text-xs">
        {selectedCount} of 2-4 campaigns selected
      </Badge>
    </div>
  );
}

/**
 * Error state for comparison panel
 */
function ComparisonErrorState({
  error,
  onRetry,
}: {
  error: string;
  onRetry?: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      <div className="p-4 rounded-full bg-destructive/10 mb-4">
        <AlertTriangle className="h-10 w-10 text-destructive" />
      </div>
      <h3 className="font-semibold text-lg mb-2">Comparison Failed</h3>
      <p className="text-sm text-muted-foreground max-w-md mb-4">
        {error}
      </p>
      {onRetry && (
        <Button variant="outline" onClick={onRetry}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Try Again
        </Button>
      )}
    </div>
  );
}

/**
 * Loading skeleton for the comparison content
 */
function ComparisonLoadingSkeleton({
  viewMode,
}: {
  viewMode: ComparisonViewMode;
}) {
  if (viewMode === "table") {
    return <ComparisonTableSkeleton campaignCount={3} rowCount={6} />;
  }

  if (viewMode === "chart") {
    return <NormalizedMetricsChartSkeleton height={400} />;
  }

  // Combined view
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <ComparisonTableSkeleton campaignCount={3} rowCount={5} />
      <NormalizedMetricsChartSkeleton height={380} />
    </div>
  );
}

/**
 * Combined view showing both table and chart
 */
function CombinedView({
  comparison,
  isLoading,
  error,
  onRetry,
  onCampaignClick,
  chartRef,
}: {
  comparison: ReturnType<typeof useCampaignComparison>["data"];
  isLoading: boolean;
  error: string | null;
  onRetry: () => void;
  onCampaignClick?: (campaignId: string) => void;
  chartRef?: React.RefObject<HTMLDivElement>;
}) {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Comparison Table */}
      <div className="lg:col-span-1">
        <ComparisonTable
          comparison={comparison}
          isLoading={isLoading}
          error={error}
          onRetry={onRetry}
          showRankBadges
          highlightBest
          highlightWorst
          showDeltas
          enableExpanding
          onCampaignClick={onCampaignClick}
        />
      </div>

      {/* Normalized Metrics Chart */}
      <div className="lg:col-span-1">
        <NormalizedMetricsChart
          comparison={comparison}
          isLoading={isLoading}
          error={error}
          onRetry={onRetry}
          title="Normalized Comparison"
          description="Performance metrics normalized to 0-100% scale"
          height={380}
          showLegend
          showMetricControls
          showDownloadButton={false}
          chartRef={chartRef}
        />
      </div>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * CampaignComparisonPanel Component
 *
 * A comprehensive panel for comparing multiple campaigns with:
 * - Multi-select campaign selector (2-4 campaigns)
 * - Comparison table with metrics, rankings, and deltas
 * - Normalized metrics radar chart for visual comparison
 * - Clear comparison button
 * - View mode toggle (combined/table/chart)
 * - Export functionality
 * - Fullscreen support
 *
 * @accessibility
 * - Full keyboard navigation
 * - ARIA labels for all interactive elements
 * - Screen reader friendly status updates
 * - Focus management for mode transitions
 */
export function CampaignComparisonPanel({
  initialCampaignIds = [],
  onCampaignChange,
  maxCampaigns = 4,
  title = "Campaign Comparison",
  description,
  defaultViewMode = "combined",
  showViewModeToggle = true,
  showExportButton = true,
  onExport,
  allowFullscreen = true,
  compact = false,
  className,
  height,
}: CampaignComparisonPanelProps) {
  // State
  const [selectedCampaignIds, setSelectedCampaignIds] = useState<string[]>(initialCampaignIds);
  const [viewMode, setViewMode] = useState<ComparisonViewMode>(defaultViewMode);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Refs
  const chartRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // Fetch comparison data
  const {
    data: comparison,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useCampaignComparison(selectedCampaignIds, {
    normalize_metrics: true,
    include_time_series: false,
  });

  // Handlers
  const handleSelectionChange = useCallback(
    (ids: string[]) => {
      setSelectedCampaignIds(ids);
      onCampaignChange?.(ids);
    },
    [onCampaignChange]
  );

  const handleClear = useCallback(() => {
    setSelectedCampaignIds([]);
    onCampaignChange?.([]);
  }, [onCampaignChange]);

  const handleRefresh = useCallback(() => {
    refetch();
  }, [refetch]);

  const handleViewModeChange = useCallback((mode: ComparisonViewMode) => {
    setViewMode(mode);
  }, []);

  const handleToggleFullscreen = useCallback(() => {
    if (!panelRef.current) return;

    if (!isFullscreen) {
      panelRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  }, [isFullscreen]);

  const handleCampaignClick = useCallback((campaignId: string) => {
    // Could navigate to campaign detail or open a modal
  }, []);

  // Listen for fullscreen change events
  React.useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  // Compute error message
  const errorMessage = useMemo(() => {
    if (!isError) return null;
    if (error instanceof Error) return error.message;
    return "Failed to load comparison data. Please try again.";
  }, [isError, error]);

  // Determine what to render
  const hasNoSelection = selectedCampaignIds.length === 0;
  const hasInsufficientSelection = selectedCampaignIds.length === 1;
  const hasValidSelection = selectedCampaignIds.length >= 2;

  return (
    <Card
      ref={panelRef}
      className={cn(
        "overflow-hidden",
        isFullscreen && "fixed inset-0 z-50 rounded-none",
        className
      )}
      style={height && !isFullscreen ? { height } : undefined}
    >
      <CardHeader className={cn(compact ? "pb-2 space-y-0" : "pb-4")}>
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <GitCompare className="h-5 w-5 text-muted-foreground" />
            <CardTitle className={cn(compact ? "text-base" : "text-lg")}>
              {title}
            </CardTitle>
            {hasValidSelection && (
              <Badge variant="secondary" className="text-xs font-normal">
                {selectedCampaignIds.length} campaigns
              </Badge>
            )}
          </div>
        </div>
        {description && (
          <CardDescription className="mt-1">{description}</CardDescription>
        )}
      </CardHeader>

      <CardContent className={cn(compact ? "pt-0" : "pt-2")}>
        {/* Header with selector and controls */}
        <ComparisonHeader
          selectedIds={selectedCampaignIds}
          onSelectionChange={handleSelectionChange}
          maxCampaigns={maxCampaigns}
          onClear={handleClear}
          onRefresh={handleRefresh}
          isRefreshing={isFetching}
          viewMode={viewMode}
          onViewModeChange={handleViewModeChange}
          showViewModeToggle={showViewModeToggle && hasValidSelection}
          showExportButton={showExportButton && hasValidSelection}
          onExport={onExport}
          allowFullscreen={allowFullscreen}
          isFullscreen={isFullscreen}
          onToggleFullscreen={handleToggleFullscreen}
          compact={compact}
        />

        {/* Main content area */}
        <div className={cn("mt-6", compact && "mt-4")}>
          {/* Empty state */}
          {hasNoSelection && (
            <ComparisonEmptyState />
          )}

          {/* Waiting for more selections */}
          {hasInsufficientSelection && (
            <ComparisonWaitingState selectedCount={selectedCampaignIds.length} />
          )}

          {/* Loading state */}
          {hasValidSelection && isLoading && !comparison && (
            <ComparisonLoadingSkeleton viewMode={viewMode} />
          )}

          {/* Error state */}
          {hasValidSelection && isError && !comparison && (
            <ComparisonErrorState
              error={errorMessage || "Unknown error"}
              onRetry={handleRefresh}
            />
          )}

          {/* Comparison content */}
          {hasValidSelection && comparison && (
            <>
              {viewMode === "combined" && (
                <CombinedView
                  comparison={comparison}
                  isLoading={isLoading && isFetching}
                  error={errorMessage}
                  onRetry={handleRefresh}
                  onCampaignClick={handleCampaignClick}
                  chartRef={chartRef}
                />
              )}

              {viewMode === "table" && (
                <ComparisonTable
                  comparison={comparison}
                  isLoading={isLoading && isFetching}
                  error={errorMessage}
                  onRetry={handleRefresh}
                  showRankBadges
                  highlightBest
                  highlightWorst
                  showDeltas
                  enableExpanding
                  defaultExpandedGroups={["performance", "strategy"]}
                  onCampaignClick={handleCampaignClick}
                />
              )}

              {viewMode === "chart" && (
                <NormalizedMetricsChart
                  comparison={comparison}
                  isLoading={isLoading && isFetching}
                  error={errorMessage}
                  onRetry={handleRefresh}
                  title="Normalized Performance Metrics"
                  description="All metrics normalized to 0-100% scale for fair comparison. Higher values indicate better performance."
                  height={isFullscreen ? 600 : 450}
                  showLegend
                  showMetricControls
                  showDownloadButton={!!onExport}
                  onExport={onExport ? (format) => onExport(format) : undefined}
                  chartRef={chartRef}
                />
              )}
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Convenience Variants
// =============================================================================

/**
 * Simple comparison panel without header controls
 */
export function SimpleComparisonPanel({
  initialCampaignIds,
  onCampaignChange,
  className,
}: {
  initialCampaignIds?: string[];
  onCampaignChange?: (ids: string[]) => void;
  className?: string;
}) {
  return (
    <CampaignComparisonPanel
      initialCampaignIds={initialCampaignIds}
      onCampaignChange={onCampaignChange}
      showViewModeToggle={false}
      showExportButton={false}
      allowFullscreen={false}
      compact
      className={className}
    />
  );
}

/**
 * Compact comparison panel for dashboard widgets
 */
export function CompactComparisonPanel({
  initialCampaignIds,
  onCampaignChange,
  className,
}: {
  initialCampaignIds?: string[];
  onCampaignChange?: (ids: string[]) => void;
  className?: string;
}) {
  return (
    <CampaignComparisonPanel
      initialCampaignIds={initialCampaignIds}
      onCampaignChange={onCampaignChange}
      defaultViewMode="chart"
      showViewModeToggle={false}
      allowFullscreen={false}
      compact
      title="Quick Compare"
      className={className}
    />
  );
}

/**
 * Detailed comparison panel with all features
 */
export function DetailedComparisonPanel({
  initialCampaignIds,
  onCampaignChange,
  onExport,
  className,
}: {
  initialCampaignIds?: string[];
  onCampaignChange?: (ids: string[]) => void;
  onExport?: (format: "png" | "svg" | "csv") => void;
  className?: string;
}) {
  return (
    <CampaignComparisonPanel
      initialCampaignIds={initialCampaignIds}
      onCampaignChange={onCampaignChange}
      onExport={onExport}
      title="Detailed Campaign Comparison"
      description="Compare performance metrics, techniques, and cost efficiency across campaigns."
      showViewModeToggle
      showExportButton
      allowFullscreen
      className={className}
    />
  );
}

// =============================================================================
// Exports
// =============================================================================

export type { ComparisonViewMode };
export default CampaignComparisonPanel;
