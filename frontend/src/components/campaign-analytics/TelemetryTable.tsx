"use client";

/**
 * TelemetryTable Component for Campaign Analytics
 *
 * Paginated table of individual telemetry events.
 * Columns: timestamp, technique, provider, success, latency.
 * Row click opens detail modal. Support sorting and export.
 *
 * @module components/campaign-analytics/TelemetryTable
 */

import * as React from "react";
import {
  Clock,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Check,
  X,
  AlertCircle,
  Loader2,
  ExternalLink,
  Timer,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { TelemetryDetailModal } from "./TelemetryDetailModal";
import { ExportButton, type CSVExportOptions } from "./ExportButton";
import { useTelemetryEvents } from "@/lib/api/query/campaign-queries";
import type {
  TelemetryEventSummary,
  TelemetryFilterParams,
  ExecutionStatusEnum,
  SortConfig,
} from "@/types/campaign-analytics";

// =============================================================================
// Types
// =============================================================================

/**
 * Sortable columns in the telemetry table.
 */
export type TelemetrySortField =
  | "timestamp"
  | "technique"
  | "provider"
  | "status"
  | "latency"
  | "tokens"
  | "potency";

/**
 * Sort direction type.
 */
export type TelemetrySortDirection = "asc" | "desc";

/**
 * Combined sort configuration.
 */
export interface TelemetrySortConfig {
  field: TelemetrySortField;
  direction: TelemetrySortDirection;
}

/**
 * Props for the TelemetryTable component.
 */
export interface TelemetryTableProps {
  /** Campaign ID to fetch telemetry events for */
  campaignId: string | null;
  /** Optional filters to apply to the telemetry query */
  filters?: TelemetryFilterParams | null;
  /** Initial page size (default: 25) */
  initialPageSize?: number;
  /** Page size options (default: [10, 25, 50, 100]) */
  pageSizeOptions?: number[];
  /** Initial sort configuration */
  initialSort?: TelemetrySortConfig;
  /** Whether to enable row click to open detail modal (default: true) */
  enableDetailModal?: boolean;
  /** Whether to enable export button (default: true) */
  enableExport?: boolean;
  /** Whether to enable sorting (default: true) */
  enableSorting?: boolean;
  /** Whether to enable pagination (default: true) */
  enablePagination?: boolean;
  /** Callback when row is clicked */
  onRowClick?: (event: TelemetryEventSummary) => void;
  /** Callback when page changes */
  onPageChange?: (page: number) => void;
  /** Callback when sort changes */
  onSortChange?: (sort: TelemetrySortConfig) => void;
  /** Custom CSV export options */
  csvExportOptions?: CSVExportOptions<TelemetryEventSummary>;
  /** Export filename base */
  exportFilename?: string;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show column headers (default: true) */
  showHeaders?: boolean;
  /** Compact mode with reduced padding (default: false) */
  compact?: boolean;
}

// =============================================================================
// Constants
// =============================================================================

/** Default page size options */
const DEFAULT_PAGE_SIZE_OPTIONS = [10, 25, 50, 100];

/** Default page size */
const DEFAULT_PAGE_SIZE = 25;

/** Status configuration for badges */
interface StatusConfig {
  label: string;
  icon: LucideIcon;
  className: string;
  bgClassName: string;
}

const STATUS_CONFIGS: Record<string, StatusConfig> = {
  success: {
    label: "Success",
    icon: Check,
    className: "text-green-600 dark:text-green-400",
    bgClassName: "bg-green-100 dark:bg-green-900/30",
  },
  partial_success: {
    label: "Partial",
    icon: AlertCircle,
    className: "text-amber-600 dark:text-amber-400",
    bgClassName: "bg-amber-100 dark:bg-amber-900/30",
  },
  failure: {
    label: "Failed",
    icon: X,
    className: "text-red-600 dark:text-red-400",
    bgClassName: "bg-red-100 dark:bg-red-900/30",
  },
  timeout: {
    label: "Timeout",
    icon: Timer,
    className: "text-orange-600 dark:text-orange-400",
    bgClassName: "bg-orange-100 dark:bg-orange-900/30",
  },
  pending: {
    label: "Pending",
    icon: Loader2,
    className: "text-blue-600 dark:text-blue-400",
    bgClassName: "bg-blue-100 dark:bg-blue-900/30",
  },
  skipped: {
    label: "Skipped",
    icon: AlertCircle,
    className: "text-slate-600 dark:text-slate-400",
    bgClassName: "bg-slate-100 dark:bg-slate-800",
  },
};

/** Default CSV columns for telemetry export */
const DEFAULT_CSV_COLUMNS = [
  { accessor: "id", header: "Event ID" },
  { accessor: "sequence_number", header: "Sequence" },
  { accessor: "created_at", header: "Timestamp", format: "datetime" as const },
  { accessor: "technique_suite", header: "Technique" },
  { accessor: "provider", header: "Provider" },
  { accessor: "model", header: "Model" },
  { accessor: "status", header: "Status" },
  { accessor: "success_indicator", header: "Success" },
  { accessor: "total_latency_ms", header: "Latency (ms)" },
  { accessor: "total_tokens", header: "Total Tokens" },
  { accessor: "potency_level", header: "Potency" },
];

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format timestamp to readable format.
 */
function formatTimestamp(timestamp: string | null | undefined): string {
  if (!timestamp) return "—";
  try {
    const date = new Date(timestamp);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "—";
  }
}

/**
 * Format duration in milliseconds to human-readable string.
 */
function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${ms.toFixed(0)}ms`;
}

/**
 * Sort events locally based on sort configuration.
 */
function sortEvents(
  events: TelemetryEventSummary[],
  sort: TelemetrySortConfig
): TelemetryEventSummary[] {
  const sorted = [...events].sort((a, b) => {
    let comparison = 0;

    switch (sort.field) {
      case "timestamp":
        comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        break;
      case "technique":
        comparison = a.technique_suite.localeCompare(b.technique_suite);
        break;
      case "provider":
        comparison = a.provider.localeCompare(b.provider);
        break;
      case "status":
        comparison = a.status.localeCompare(b.status);
        break;
      case "latency":
        comparison = (a.total_latency_ms || 0) - (b.total_latency_ms || 0);
        break;
      case "tokens":
        comparison = (a.total_tokens || 0) - (b.total_tokens || 0);
        break;
      case "potency":
        comparison = (a.potency_level || 0) - (b.potency_level || 0);
        break;
    }

    return sort.direction === "desc" ? -comparison : comparison;
  });

  return sorted;
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Status badge for telemetry event.
 */
function StatusBadge({ status }: { status: ExecutionStatusEnum }) {
  const config = STATUS_CONFIGS[status] || STATUS_CONFIGS.pending;
  const Icon = config.icon;

  return (
    <Badge
      variant="secondary"
      className={cn(
        "gap-1 font-medium text-xs",
        config.className,
        config.bgClassName
      )}
    >
      <Icon className={cn("size-3", status === "pending" && "animate-spin")} />
      <span className="hidden sm:inline">{config.label}</span>
    </Badge>
  );
}

/**
 * Sortable column header.
 */
function SortableHeader({
  field,
  label,
  currentSort,
  onSort,
  className,
}: {
  field: TelemetrySortField;
  label: string;
  currentSort: TelemetrySortConfig;
  onSort: (field: TelemetrySortField) => void;
  className?: string;
}) {
  const isActive = currentSort.field === field;

  return (
    <button
      type="button"
      className={cn(
        "flex items-center gap-1 hover:text-foreground transition-colors text-left",
        isActive && "text-foreground font-medium",
        className
      )}
      onClick={() => onSort(field)}
    >
      {label}
      <span className="size-4 flex items-center justify-center">
        {isActive ? (
          currentSort.direction === "asc" ? (
            <ChevronUp className="size-3.5" />
          ) : (
            <ChevronDown className="size-3.5" />
          )
        ) : (
          <ChevronDown className="size-3 opacity-30" />
        )}
      </span>
    </button>
  );
}

/**
 * Pagination controls component.
 */
function PaginationControls({
  currentPage,
  totalPages,
  pageSize,
  totalItems,
  pageSizeOptions,
  onPageChange,
  onPageSizeChange,
  className,
}: {
  currentPage: number;
  totalPages: number;
  pageSize: number;
  totalItems: number;
  pageSizeOptions: number[];
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  className?: string;
}) {
  const startItem = (currentPage - 1) * pageSize + 1;
  const endItem = Math.min(currentPage * pageSize, totalItems);

  return (
    <div className={cn("flex items-center justify-between gap-4", className)}>
      {/* Page size selector */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span className="hidden sm:inline">Rows per page:</span>
        <Select
          value={String(pageSize)}
          onValueChange={(value) => onPageSizeChange(Number(value))}
        >
          <SelectTrigger className="h-8 w-16">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {pageSizeOptions.map((size) => (
              <SelectItem key={size} value={String(size)}>
                {size}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Page info */}
      <div className="text-sm text-muted-foreground">
        {totalItems > 0 ? (
          <>
            {startItem}–{endItem} of {totalItems.toLocaleString()}
          </>
        ) : (
          "No results"
        )}
      </div>

      {/* Navigation buttons */}
      <div className="flex items-center gap-1">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8"
                disabled={currentPage <= 1}
                onClick={() => onPageChange(1)}
              >
                <ChevronsLeft className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>First page</TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8"
                disabled={currentPage <= 1}
                onClick={() => onPageChange(currentPage - 1)}
              >
                <ChevronLeft className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Previous page</TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <span className="px-2 text-sm font-medium">
          {currentPage} / {totalPages || 1}
        </span>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8"
                disabled={currentPage >= totalPages}
                onClick={() => onPageChange(currentPage + 1)}
              >
                <ChevronRight className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Next page</TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8"
                disabled={currentPage >= totalPages}
                onClick={() => onPageChange(totalPages)}
              >
                <ChevronsRight className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Last page</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}

/**
 * Single table row for a telemetry event.
 */
function TelemetryRow({
  event,
  onClick,
  compact,
}: {
  event: TelemetryEventSummary;
  onClick?: (event: TelemetryEventSummary) => void;
  compact?: boolean;
}) {
  const isClickable = !!onClick;

  return (
    <TableRow
      className={cn(
        isClickable && "cursor-pointer hover:bg-muted/50 transition-colors"
      )}
      onClick={() => onClick?.(event)}
    >
      {/* Sequence & Timestamp */}
      <TableCell className={cn("font-mono text-xs", compact && "py-2")}>
        <div className="flex flex-col gap-0.5">
          <span className="text-muted-foreground">#{event.sequence_number}</span>
          <span>{formatTimestamp(event.created_at)}</span>
        </div>
      </TableCell>

      {/* Technique */}
      <TableCell className={cn(compact && "py-2")}>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="font-medium truncate max-w-[150px] block">
                {event.technique_suite}
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p>{event.technique_suite}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </TableCell>

      {/* Provider */}
      <TableCell className={cn(compact && "py-2")}>
        <div className="flex flex-col gap-0.5">
          <span className="font-medium">{event.provider}</span>
          <span className="text-xs text-muted-foreground truncate max-w-[120px]">
            {event.model}
          </span>
        </div>
      </TableCell>

      {/* Status */}
      <TableCell className={cn(compact && "py-2")}>
        <StatusBadge status={event.status} />
      </TableCell>

      {/* Latency */}
      <TableCell className={cn("font-mono", compact && "py-2")}>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                className={cn(
                  event.total_latency_ms > 5000 && "text-amber-600 dark:text-amber-400",
                  event.total_latency_ms > 10000 && "text-red-600 dark:text-red-400"
                )}
              >
                {formatDuration(event.total_latency_ms)}
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p>Total latency: {event.total_latency_ms?.toLocaleString()}ms</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </TableCell>

      {/* Tokens */}
      <TableCell className={cn("font-mono text-right", compact && "py-2")}>
        {event.total_tokens?.toLocaleString() ?? "—"}
      </TableCell>

      {/* Actions */}
      {isClickable && (
        <TableCell className={cn("w-10", compact && "py-2")}>
          <ExternalLink className="size-4 text-muted-foreground opacity-50" />
        </TableCell>
      )}
    </TableRow>
  );
}

// =============================================================================
// Loading Skeleton
// =============================================================================

/**
 * Loading skeleton for the telemetry table.
 */
export function TelemetryTableSkeleton({
  rows = 10,
  className,
}: {
  rows?: number;
  className?: string;
}) {
  return (
    <div className={cn("space-y-4", className)}>
      {/* Header skeleton */}
      <div className="flex items-center justify-between">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-9 w-24" />
      </div>

      {/* Table skeleton */}
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[140px]">
                <Skeleton className="h-4 w-20" />
              </TableHead>
              <TableHead className="w-[160px]">
                <Skeleton className="h-4 w-24" />
              </TableHead>
              <TableHead className="w-[140px]">
                <Skeleton className="h-4 w-20" />
              </TableHead>
              <TableHead className="w-[100px]">
                <Skeleton className="h-4 w-16" />
              </TableHead>
              <TableHead className="w-[100px]">
                <Skeleton className="h-4 w-16" />
              </TableHead>
              <TableHead className="w-[80px]">
                <Skeleton className="h-4 w-12" />
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {Array.from({ length: rows }).map((_, i) => (
              <TableRow key={i}>
                <TableCell>
                  <div className="space-y-1">
                    <Skeleton className="h-3 w-8" />
                    <Skeleton className="h-3 w-24" />
                  </div>
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-32" />
                </TableCell>
                <TableCell>
                  <div className="space-y-1">
                    <Skeleton className="h-4 w-20" />
                    <Skeleton className="h-3 w-28" />
                  </div>
                </TableCell>
                <TableCell>
                  <Skeleton className="h-6 w-16 rounded-full" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-12" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-10" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination skeleton */}
      <div className="flex items-center justify-between">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-4 w-24" />
        <div className="flex gap-1">
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-16" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Error State
// =============================================================================

/**
 * Error state component for the telemetry table.
 */
export function TelemetryTableError({
  error,
  onRetry,
  className,
}: {
  error: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-4 py-12 text-center rounded-lg border border-dashed",
        className
      )}
    >
      <AlertCircle className="size-12 text-red-500" />
      <div className="space-y-1">
        <h4 className="text-lg font-semibold">Failed to load telemetry events</h4>
        <p className="text-sm text-muted-foreground max-w-md">{error}</p>
      </div>
      {onRetry && (
        <Button variant="outline" onClick={onRetry}>
          Try Again
        </Button>
      )}
    </div>
  );
}

// =============================================================================
// Empty State
// =============================================================================

/**
 * Empty state component for when no events match filters.
 */
export function TelemetryTableEmpty({
  hasFilters,
  onClearFilters,
  className,
}: {
  hasFilters?: boolean;
  onClearFilters?: () => void;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-4 py-12 text-center rounded-lg border border-dashed",
        className
      )}
    >
      <Clock className="size-12 text-muted-foreground/50" />
      <div className="space-y-1">
        <h4 className="text-lg font-semibold">No telemetry events found</h4>
        <p className="text-sm text-muted-foreground max-w-md">
          {hasFilters
            ? "No events match your current filters. Try adjusting or clearing the filters."
            : "This campaign doesn't have any telemetry events yet."}
        </p>
      </div>
      {hasFilters && onClearFilters && (
        <Button variant="outline" onClick={onClearFilters}>
          Clear Filters
        </Button>
      )}
    </div>
  );
}

// =============================================================================
// Waiting State
// =============================================================================

/**
 * Waiting state for when no campaign is selected.
 */
export function TelemetryTableWaiting({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-4 py-12 text-center rounded-lg border border-dashed",
        className
      )}
    >
      <Clock className="size-12 text-muted-foreground/50" />
      <div className="space-y-1">
        <h4 className="text-lg font-semibold">Select a Campaign</h4>
        <p className="text-sm text-muted-foreground">
          Choose a campaign to view its telemetry events.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * TelemetryTable - Paginated table of individual telemetry events.
 *
 * Features:
 * - Paginated data with configurable page size
 * - Sortable columns (timestamp, technique, provider, status, latency)
 * - Row click opens TelemetryDetailModal
 * - Export to CSV
 * - Loading, error, and empty states
 *
 * @example
 * ```tsx
 * <TelemetryTable
 *   campaignId={selectedCampaignId}
 *   initialPageSize={25}
 *   enableExport={true}
 *   onRowClick={(event) => console.log('Selected:', event)}
 * />
 * ```
 */
export function TelemetryTable({
  campaignId,
  filters,
  initialPageSize = DEFAULT_PAGE_SIZE,
  pageSizeOptions = DEFAULT_PAGE_SIZE_OPTIONS,
  initialSort = { field: "timestamp", direction: "desc" },
  enableDetailModal = true,
  enableExport = true,
  enableSorting = true,
  enablePagination = true,
  onRowClick,
  onPageChange,
  onSortChange,
  csvExportOptions,
  exportFilename = "telemetry-events",
  className,
  showHeaders = true,
  compact = false,
}: TelemetryTableProps) {
  // Local state
  const [page, setPage] = React.useState(1);
  const [pageSize, setPageSize] = React.useState(initialPageSize);
  const [sort, setSort] = React.useState<TelemetrySortConfig>(initialSort);
  const [selectedEventId, setSelectedEventId] = React.useState<string | null>(null);
  const [isDetailModalOpen, setIsDetailModalOpen] = React.useState(false);

  // Fetch telemetry events
  const {
    data: response,
    isLoading,
    error,
    refetch,
  } = useTelemetryEvents(
    campaignId,
    {
      page,
      pageSize,
      filters: filters ?? undefined,
    },
    !!campaignId
  );

  // Reset page when filters or campaign changes
  React.useEffect(() => {
    setPage(1);
  }, [campaignId, filters]);

  // Sort events locally (API should handle this, but we add client-side fallback)
  const sortedEvents = React.useMemo(() => {
    if (!response?.items) return [];
    return sortEvents(response.items, sort);
  }, [response?.items, sort]);

  // Handle sort column click
  const handleSort = React.useCallback(
    (field: TelemetrySortField) => {
      const newSort: TelemetrySortConfig = {
        field,
        direction:
          sort.field === field && sort.direction === "asc" ? "desc" : "asc",
      };
      setSort(newSort);
      onSortChange?.(newSort);
    },
    [sort, onSortChange]
  );

  // Handle page change
  const handlePageChange = React.useCallback(
    (newPage: number) => {
      setPage(newPage);
      onPageChange?.(newPage);
    },
    [onPageChange]
  );

  // Handle page size change
  const handlePageSizeChange = React.useCallback((newSize: number) => {
    setPageSize(newSize);
    setPage(1); // Reset to first page
  }, []);

  // Handle row click
  const handleRowClick = React.useCallback(
    (event: TelemetryEventSummary) => {
      onRowClick?.(event);

      if (enableDetailModal) {
        setSelectedEventId(event.id);
        setIsDetailModalOpen(true);
      }
    },
    [onRowClick, enableDetailModal]
  );

  // Handle detail modal navigation
  const handleDetailNavigate = React.useCallback(
    (_direction: "prev" | "next", newEventId: string) => {
      setSelectedEventId(newEventId);
    },
    []
  );

  // CSV export columns
  const csvColumns = React.useMemo(
    () => csvExportOptions?.columns ?? DEFAULT_CSV_COLUMNS,
    [csvExportOptions?.columns]
  );

  // Show waiting state if no campaign selected
  if (!campaignId) {
    return <TelemetryTableWaiting className={className} />;
  }

  // Show loading state
  if (isLoading && !response) {
    return <TelemetryTableSkeleton rows={pageSize} className={className} />;
  }

  // Show error state
  if (error) {
    return (
      <TelemetryTableError
        error={error instanceof Error ? error.message : "Unknown error"}
        onRetry={() => refetch()}
        className={className}
      />
    );
  }

  // Show empty state
  if (!response?.items?.length) {
    const hasActiveFilters =
      filters &&
      (filters.status?.length ||
        filters.technique_suite?.length ||
        filters.provider?.length ||
        filters.success_only !== null ||
        filters.start_time ||
        filters.end_time);

    return (
      <TelemetryTableEmpty
        hasFilters={!!hasActiveFilters}
        className={className}
      />
    );
  }

  const totalPages = response.total_pages || Math.ceil(response.total / pageSize);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header with export button */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          {response.total.toLocaleString()} telemetry event{response.total !== 1 ? "s" : ""}
        </div>
        {enableExport && (
          <ExportButton
            data={sortedEvents}
            filename={exportFilename}
            csvOptions={{
              columns: csvColumns,
              ...csvExportOptions,
            }}
            formats={["csv"]}
            variant="outline"
            size="sm"
          />
        )}
      </div>

      {/* Table */}
      <div className="rounded-md border overflow-auto">
        <Table>
          {showHeaders && (
            <TableHeader>
              <TableRow>
                <TableHead className="w-[140px]">
                  {enableSorting ? (
                    <SortableHeader
                      field="timestamp"
                      label="Timestamp"
                      currentSort={sort}
                      onSort={handleSort}
                    />
                  ) : (
                    "Timestamp"
                  )}
                </TableHead>
                <TableHead className="w-[160px]">
                  {enableSorting ? (
                    <SortableHeader
                      field="technique"
                      label="Technique"
                      currentSort={sort}
                      onSort={handleSort}
                    />
                  ) : (
                    "Technique"
                  )}
                </TableHead>
                <TableHead className="w-[140px]">
                  {enableSorting ? (
                    <SortableHeader
                      field="provider"
                      label="Provider"
                      currentSort={sort}
                      onSort={handleSort}
                    />
                  ) : (
                    "Provider"
                  )}
                </TableHead>
                <TableHead className="w-[100px]">
                  {enableSorting ? (
                    <SortableHeader
                      field="status"
                      label="Status"
                      currentSort={sort}
                      onSort={handleSort}
                    />
                  ) : (
                    "Status"
                  )}
                </TableHead>
                <TableHead className="w-[100px]">
                  {enableSorting ? (
                    <SortableHeader
                      field="latency"
                      label="Latency"
                      currentSort={sort}
                      onSort={handleSort}
                    />
                  ) : (
                    "Latency"
                  )}
                </TableHead>
                <TableHead className="w-[80px] text-right">
                  {enableSorting ? (
                    <SortableHeader
                      field="tokens"
                      label="Tokens"
                      currentSort={sort}
                      onSort={handleSort}
                      className="justify-end"
                    />
                  ) : (
                    "Tokens"
                  )}
                </TableHead>
                {enableDetailModal && <TableHead className="w-10" />}
              </TableRow>
            </TableHeader>
          )}
          <TableBody>
            {sortedEvents.map((event) => (
              <TelemetryRow
                key={event.id}
                event={event}
                onClick={enableDetailModal || onRowClick ? handleRowClick : undefined}
                compact={compact}
              />
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {enablePagination && totalPages > 1 && (
        <PaginationControls
          currentPage={page}
          totalPages={totalPages}
          pageSize={pageSize}
          totalItems={response.total}
          pageSizeOptions={pageSizeOptions}
          onPageChange={handlePageChange}
          onPageSizeChange={handlePageSizeChange}
        />
      )}

      {/* Detail Modal */}
      {enableDetailModal && (
        <TelemetryDetailModal
          open={isDetailModalOpen}
          onOpenChange={setIsDetailModalOpen}
          campaignId={campaignId}
          eventId={selectedEventId}
          events={sortedEvents}
          onNavigate={handleDetailNavigate}
        />
      )}
    </div>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * Simple telemetry table without export or modal features.
 */
export function SimpleTelemetryTable({
  campaignId,
  filters,
  pageSize = 25,
  className,
}: {
  campaignId: string | null;
  filters?: TelemetryFilterParams | null;
  pageSize?: number;
  className?: string;
}) {
  return (
    <TelemetryTable
      campaignId={campaignId}
      filters={filters}
      initialPageSize={pageSize}
      enableExport={false}
      enableDetailModal={false}
      className={className}
    />
  );
}

/**
 * Compact telemetry table for dashboard widgets.
 */
export function CompactTelemetryTable({
  campaignId,
  filters,
  pageSize = 10,
  className,
}: {
  campaignId: string | null;
  filters?: TelemetryFilterParams | null;
  pageSize?: number;
  className?: string;
}) {
  return (
    <TelemetryTable
      campaignId={campaignId}
      filters={filters}
      initialPageSize={pageSize}
      pageSizeOptions={[5, 10, 20]}
      enableExport={false}
      compact={true}
      className={className}
    />
  );
}

/**
 * Full-featured telemetry table with all options enabled.
 */
export function DetailedTelemetryTable({
  campaignId,
  filters,
  onRowClick,
  className,
}: {
  campaignId: string | null;
  filters?: TelemetryFilterParams | null;
  onRowClick?: (event: TelemetryEventSummary) => void;
  className?: string;
}) {
  return (
    <TelemetryTable
      campaignId={campaignId}
      filters={filters}
      initialPageSize={50}
      pageSizeOptions={[25, 50, 100, 200]}
      enableExport={true}
      enableDetailModal={true}
      enableSorting={true}
      onRowClick={onRowClick}
      className={className}
    />
  );
}
