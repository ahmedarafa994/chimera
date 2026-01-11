"use client";

/**
 * ExportButton Component for Campaign Analytics
 *
 * Dropdown button with export options: PNG, SVG, CSV.
 * Accepts chartRef and data props. Shows loading state during export.
 *
 * @module components/campaign-analytics/ExportButton
 */

import * as React from "react";
import {
  Download,
  Image,
  FileCode2,
  FileSpreadsheet,
  Loader2,
  Check,
  AlertCircle,
  ChevronDown,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  exportChartAsPNG,
  exportChartAsSVG,
  downloadFile,
  generateChartFilename,
  type ChartRef,
  type PNGExportOptions,
  type SVGExportOptions,
} from "@/lib/utils/chart-export";
import {
  generateCSV,
  downloadCSV,
  generateCSVFilename,
  type CSVColumn,
  type CSVGenerateOptions,
} from "@/lib/utils/csv-export";

// =============================================================================
// Types
// =============================================================================

/**
 * Export format types supported by the ExportButton.
 */
export type ExportFormat = "png" | "svg" | "csv";

/**
 * Result of an export operation.
 */
export interface ExportResult {
  /** Whether the export was successful */
  success: boolean;
  /** Format that was exported */
  format: ExportFormat;
  /** Error message if export failed */
  error?: string;
  /** Filename used for download */
  filename?: string;
}

/**
 * Options for chart export (PNG/SVG).
 */
export interface ChartExportOptions {
  /** Options for PNG export */
  pngOptions?: PNGExportOptions;
  /** Options for SVG export */
  svgOptions?: SVGExportOptions;
}

/**
 * Options for CSV export.
 */
export interface CSVExportOptions<T = unknown> {
  /** Column definitions for CSV export */
  columns?: CSVColumn<T>[];
  /** CSV generation options */
  options?: CSVGenerateOptions;
}

/**
 * Props for the ExportButton component.
 */
export interface ExportButtonProps<T = unknown> {
  /** Reference to the chart container element for PNG/SVG export */
  chartRef?: ChartRef;
  /** Data for CSV export */
  data?: T[];
  /** Base filename for exports (without extension) */
  filename?: string;
  /** Whether to include timestamp in filename */
  includeTimestamp?: boolean;
  /** Options for chart export (PNG/SVG) */
  chartOptions?: ChartExportOptions;
  /** Options for CSV export */
  csvOptions?: CSVExportOptions<T>;
  /** Which export formats to show (default: all available based on props) */
  formats?: ExportFormat[];
  /** Callback when export starts */
  onExportStart?: (format: ExportFormat) => void;
  /** Callback when export completes */
  onExportComplete?: (result: ExportResult) => void;
  /** Callback when export fails */
  onExportError?: (error: Error, format: ExportFormat) => void;
  /** Button variant */
  variant?: "default" | "outline" | "secondary" | "ghost";
  /** Button size */
  size?: "default" | "sm" | "lg" | "icon" | "icon-sm";
  /** Whether the button is disabled */
  disabled?: boolean;
  /** Additional class names */
  className?: string;
  /** Label for the button (default: "Export") */
  label?: string;
  /** Whether to show the label (default: true for non-icon sizes) */
  showLabel?: boolean;
  /** Whether to show the dropdown arrow (default: true) */
  showChevron?: boolean;
  /** Custom trigger element (replaces default button) */
  trigger?: React.ReactNode;
  /** Dropdown menu alignment */
  align?: "start" | "center" | "end";
  /** Dropdown menu side */
  side?: "top" | "right" | "bottom" | "left";
}

/**
 * State for tracking export status.
 */
interface ExportState {
  /** Currently exporting format (null if idle) */
  exporting: ExportFormat | null;
  /** Last export result (for showing success/error feedback) */
  lastResult: ExportResult | null;
}

// =============================================================================
// Constants
// =============================================================================

/** Export format metadata */
const FORMAT_CONFIG: Record<
  ExportFormat,
  {
    label: string;
    description: string;
    icon: React.ComponentType<{ className?: string }>;
  }
> = {
  png: {
    label: "PNG Image",
    description: "High-resolution image for publications",
    icon: Image,
  },
  svg: {
    label: "SVG Vector",
    description: "Scalable vector for editing",
    icon: FileCode2,
  },
  csv: {
    label: "CSV Data",
    description: "Raw data for spreadsheets",
    icon: FileSpreadsheet,
  },
};

/** Feedback display duration in ms */
const FEEDBACK_DURATION = 2000;

// =============================================================================
// Main Component
// =============================================================================

/**
 * ExportButton - Dropdown button with export options.
 *
 * Provides PNG, SVG, and CSV export functionality with loading states
 * and user feedback.
 *
 * @example
 * ```tsx
 * // Basic usage with chart export
 * const chartRef = useRef<HTMLDivElement>(null);
 * <ExportButton
 *   chartRef={chartRef}
 *   filename="success-rate-chart"
 * />
 *
 * // With CSV data export
 * <ExportButton
 *   data={telemetryEvents}
 *   csvOptions={{
 *     columns: CAMPAIGN_CSV_COLUMNS.telemetryEvent,
 *   }}
 *   filename="campaign-telemetry"
 * />
 *
 * // Combined chart and data export
 * <ExportButton
 *   chartRef={chartRef}
 *   data={telemetryEvents}
 *   csvOptions={{
 *     columns: CAMPAIGN_CSV_COLUMNS.telemetryEvent,
 *   }}
 *   filename="campaign-analysis"
 *   onExportComplete={(result) => {
 *     toast.success(`Exported ${result.filename}`);
 *   }}
 * />
 * ```
 */
export function ExportButton<T = unknown>({
  chartRef,
  data,
  filename = "export",
  includeTimestamp = true,
  chartOptions,
  csvOptions,
  formats,
  onExportStart,
  onExportComplete,
  onExportError,
  variant = "outline",
  size = "default",
  disabled = false,
  className,
  label = "Export",
  showLabel = true,
  showChevron = true,
  trigger,
  align = "end",
  side = "bottom",
}: ExportButtonProps<T>) {
  // Track export state
  const [exportState, setExportState] = React.useState<ExportState>({
    exporting: null,
    lastResult: null,
  });

  // Determine available formats based on props
  const availableFormats = React.useMemo(() => {
    if (formats) {
      return formats;
    }

    const available: ExportFormat[] = [];

    // Chart exports require chartRef
    if (chartRef) {
      available.push("png", "svg");
    }

    // CSV export requires data
    if (data && data.length > 0) {
      available.push("csv");
    }

    return available;
  }, [formats, chartRef, data]);

  // Clear feedback after delay
  React.useEffect(() => {
    if (exportState.lastResult) {
      const timer = setTimeout(() => {
        setExportState((prev) => ({
          ...prev,
          lastResult: null,
        }));
      }, FEEDBACK_DURATION);

      return () => clearTimeout(timer);
    }
  }, [exportState.lastResult]);

  /**
   * Handle export for a specific format.
   */
  const handleExport = React.useCallback(
    async (format: ExportFormat) => {
      // Prevent multiple exports
      if (exportState.exporting) {
        return;
      }

      // Set exporting state
      setExportState({
        exporting: format,
        lastResult: null,
      });

      // Notify start
      onExportStart?.(format);

      try {
        let result: ExportResult;

        switch (format) {
          case "png": {
            if (!chartRef?.current) {
              throw new Error("Chart reference is not available");
            }

            const pngResult = await exportChartAsPNG(
              chartRef,
              chartOptions?.pngOptions
            );

            if (!pngResult.success || !pngResult.blob) {
              throw new Error(pngResult.error || "Failed to export PNG");
            }

            const pngFilename = generateChartFilename(
              filename,
              "png",
              includeTimestamp
            );
            await downloadFile(pngResult.blob, pngFilename);

            result = {
              success: true,
              format: "png",
              filename: pngFilename,
            };
            break;
          }

          case "svg": {
            if (!chartRef?.current) {
              throw new Error("Chart reference is not available");
            }

            const svgResult = await exportChartAsSVG(
              chartRef,
              chartOptions?.svgOptions
            );

            if (!svgResult.success || !svgResult.blob) {
              throw new Error(svgResult.error || "Failed to export SVG");
            }

            const svgFilename = generateChartFilename(
              filename,
              "svg",
              includeTimestamp
            );
            await downloadFile(svgResult.blob, svgFilename);

            result = {
              success: true,
              format: "svg",
              filename: svgFilename,
            };
            break;
          }

          case "csv": {
            if (!data || data.length === 0) {
              throw new Error("No data available for CSV export");
            }

            const csvFilename = generateCSVFilename(filename, includeTimestamp);

            if (csvOptions?.columns) {
              const csvResult = generateCSV(
                data,
                csvOptions.columns,
                csvOptions.options
              );

              if (!csvResult.success || !csvResult.blob) {
                throw new Error(csvResult.error || "Failed to generate CSV");
              }

              await downloadCSV(csvResult.blob, csvFilename);
            } else {
              // Auto-detect columns
              await downloadCSV(data, csvFilename, undefined, csvOptions?.options);
            }

            result = {
              success: true,
              format: "csv",
              filename: csvFilename,
            };
            break;
          }

          default:
            throw new Error(`Unsupported export format: ${format}`);
        }

        // Update state and notify
        setExportState({
          exporting: null,
          lastResult: result,
        });
        onExportComplete?.(result);
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Unknown export error";

        const result: ExportResult = {
          success: false,
          format,
          error: errorMessage,
        };

        setExportState({
          exporting: null,
          lastResult: result,
        });

        onExportError?.(
          error instanceof Error ? error : new Error(errorMessage),
          format
        );
        onExportComplete?.(result);
      }
    },
    [
      chartRef,
      data,
      filename,
      includeTimestamp,
      chartOptions,
      csvOptions,
      exportState.exporting,
      onExportStart,
      onExportComplete,
      onExportError,
    ]
  );

  // Determine if we're in a loading state
  const isExporting = exportState.exporting !== null;

  // Determine button icon based on state
  const getButtonIcon = () => {
    if (isExporting) {
      return <Loader2 className="size-4 animate-spin" />;
    }

    if (exportState.lastResult?.success) {
      return <Check className="size-4 text-green-500" />;
    }

    if (exportState.lastResult && !exportState.lastResult.success) {
      return <AlertCircle className="size-4 text-destructive" />;
    }

    return <Download className="size-4" />;
  };

  // If no formats available, disable the button
  const isDisabled = disabled || availableFormats.length === 0;

  // Build the trigger button
  const triggerContent = trigger ?? (
    <Button
      variant={variant}
      size={size}
      disabled={isDisabled}
      className={cn(
        "gap-1.5",
        isExporting && "pointer-events-none",
        className
      )}
      aria-label={
        isExporting
          ? `Exporting as ${exportState.exporting?.toUpperCase()}`
          : "Export options"
      }
    >
      {getButtonIcon()}
      {showLabel && size !== "icon" && size !== "icon-sm" && (
        <span>{isExporting ? "Exporting..." : label}</span>
      )}
      {showChevron && size !== "icon" && size !== "icon-sm" && (
        <ChevronDown className="size-3 opacity-60" />
      )}
    </Button>
  );

  // If only one format is available, simplify to direct export
  if (availableFormats.length === 1) {
    const format = availableFormats[0];
    const FormatIcon = FORMAT_CONFIG[format].icon;

    return (
      <Button
        variant={variant}
        size={size}
        disabled={isDisabled || isExporting}
        className={cn(
          "gap-1.5",
          isExporting && "pointer-events-none",
          className
        )}
        onClick={() => handleExport(format)}
        aria-label={
          isExporting
            ? `Exporting as ${format.toUpperCase()}`
            : `Export as ${format.toUpperCase()}`
        }
      >
        {isExporting ? (
          <Loader2 className="size-4 animate-spin" />
        ) : exportState.lastResult?.success ? (
          <Check className="size-4 text-green-500" />
        ) : exportState.lastResult && !exportState.lastResult.success ? (
          <AlertCircle className="size-4 text-destructive" />
        ) : (
          <FormatIcon className="size-4" />
        )}
        {showLabel && size !== "icon" && size !== "icon-sm" && (
          <span>
            {isExporting
              ? "Exporting..."
              : `${label} ${format.toUpperCase()}`}
          </span>
        )}
      </Button>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild disabled={isDisabled}>
        {triggerContent}
      </DropdownMenuTrigger>
      <DropdownMenuContent align={align} side={side} className="w-56">
        <DropdownMenuLabel className="flex items-center gap-2">
          <Download className="size-4" />
          Export Options
        </DropdownMenuLabel>
        <DropdownMenuSeparator />

        {/* Chart export options */}
        {(availableFormats.includes("png") ||
          availableFormats.includes("svg")) && (
          <>
            {availableFormats.includes("png") && (
              <DropdownMenuItem
                onClick={() => handleExport("png")}
                disabled={exportState.exporting === "png"}
                className="flex items-start gap-3 py-2"
              >
                <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-orange-500/10 text-orange-600 dark:bg-orange-500/20">
                  {exportState.exporting === "png" ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Image className="size-4" />
                  )}
                </div>
                <div className="flex flex-col">
                  <span className="font-medium">
                    {FORMAT_CONFIG.png.label}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {FORMAT_CONFIG.png.description}
                  </span>
                </div>
              </DropdownMenuItem>
            )}

            {availableFormats.includes("svg") && (
              <DropdownMenuItem
                onClick={() => handleExport("svg")}
                disabled={exportState.exporting === "svg"}
                className="flex items-start gap-3 py-2"
              >
                <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-blue-500/10 text-blue-600 dark:bg-blue-500/20">
                  {exportState.exporting === "svg" ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <FileCode2 className="size-4" />
                  )}
                </div>
                <div className="flex flex-col">
                  <span className="font-medium">
                    {FORMAT_CONFIG.svg.label}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {FORMAT_CONFIG.svg.description}
                  </span>
                </div>
              </DropdownMenuItem>
            )}
          </>
        )}

        {/* Separator between chart and data exports */}
        {(availableFormats.includes("png") ||
          availableFormats.includes("svg")) &&
          availableFormats.includes("csv") && <DropdownMenuSeparator />}

        {/* CSV export option */}
        {availableFormats.includes("csv") && (
          <DropdownMenuItem
            onClick={() => handleExport("csv")}
            disabled={exportState.exporting === "csv"}
            className="flex items-start gap-3 py-2"
          >
            <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-green-500/10 text-green-600 dark:bg-green-500/20">
              {exportState.exporting === "csv" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FileSpreadsheet className="size-4" />
              )}
            </div>
            <div className="flex flex-col">
              <span className="font-medium">{FORMAT_CONFIG.csv.label}</span>
              <span className="text-xs text-muted-foreground">
                {FORMAT_CONFIG.csv.description}
              </span>
            </div>
          </DropdownMenuItem>
        )}

        {/* Show feedback for last export */}
        {exportState.lastResult && (
          <>
            <DropdownMenuSeparator />
            <div
              className={cn(
                "flex items-center gap-2 px-2 py-1.5 text-xs",
                exportState.lastResult.success
                  ? "text-green-600 dark:text-green-400"
                  : "text-destructive"
              )}
            >
              {exportState.lastResult.success ? (
                <>
                  <Check className="size-3" />
                  <span>
                    Exported: {exportState.lastResult.filename}
                  </span>
                </>
              ) : (
                <>
                  <AlertCircle className="size-3" />
                  <span>{exportState.lastResult.error}</span>
                </>
              )}
            </div>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * ChartExportButton - Simplified export button for chart-only export.
 *
 * Only shows PNG and SVG options.
 */
export function ChartExportButton({
  chartRef,
  ...props
}: Omit<ExportButtonProps, "data" | "csvOptions" | "formats"> & {
  chartRef: ChartRef;
}) {
  return (
    <ExportButton chartRef={chartRef} formats={["png", "svg"]} {...props} />
  );
}

/**
 * DataExportButton - Simplified export button for CSV-only export.
 *
 * Only shows CSV option with single-click export.
 */
export function DataExportButton<T = unknown>({
  data,
  csvOptions,
  ...props
}: Omit<ExportButtonProps<T>, "chartRef" | "chartOptions" | "formats"> & {
  data: T[];
}) {
  return (
    <ExportButton
      data={data}
      csvOptions={csvOptions}
      formats={["csv"]}
      {...props}
    />
  );
}

/**
 * CompactExportButton - Icon-only export button.
 *
 * Shows just the download icon, good for toolbars.
 */
export function CompactExportButton<T = unknown>(
  props: ExportButtonProps<T>
) {
  return (
    <ExportButton
      size="icon-sm"
      showLabel={false}
      showChevron={false}
      variant="ghost"
      {...props}
    />
  );
}

/**
 * FullExportButton - Full-featured export button with all options.
 *
 * Shows all available export formats based on provided props.
 */
export function FullExportButton<T = unknown>(props: ExportButtonProps<T>) {
  return <ExportButton variant="default" {...props} />;
}

// =============================================================================
// Skeleton and State Components
// =============================================================================

/**
 * ExportButtonSkeleton - Loading skeleton for ExportButton.
 */
export function ExportButtonSkeleton({
  size = "default",
  className,
}: {
  size?: "default" | "sm" | "lg" | "icon" | "icon-sm";
  className?: string;
}) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-md bg-muted",
        size === "icon" && "size-9",
        size === "icon-sm" && "size-8",
        size === "sm" && "h-8 w-20",
        size === "default" && "h-9 w-24",
        size === "lg" && "h-10 w-28",
        className
      )}
    />
  );
}

/**
 * ExportButtonDisabled - Disabled state placeholder.
 */
export function ExportButtonDisabled({
  reason = "No data available for export",
  size = "default",
  className,
}: {
  reason?: string;
  size?: "default" | "sm" | "lg" | "icon" | "icon-sm";
  className?: string;
}) {
  return (
    <Button
      variant="outline"
      size={size}
      disabled
      className={className}
      title={reason}
      aria-label={reason}
    >
      <Download className="size-4 opacity-50" />
      {size !== "icon" && size !== "icon-sm" && (
        <span className="opacity-50">Export</span>
      )}
    </Button>
  );
}
