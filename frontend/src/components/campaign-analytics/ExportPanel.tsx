"use client";

/**
 * ExportPanel Component for Campaign Analytics
 *
 * Panel for bulk export: select charts to include, choose format,
 * configure options (resolution, include metadata). Generate zip for
 * multiple exports.
 *
 * @module components/campaign-analytics/ExportPanel
 */

import * as React from "react";
import {
  Download,
  Image,
  FileCode2,
  FileSpreadsheet,
  Package,
  Loader2,
  Check,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Settings2,
  X,
  RefreshCw,
  FileText,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Progress } from "@/components/ui/progress";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  SheetFooter,
  SheetClose,
} from "@/components/ui/sheet";
import {
  exportChartAsPNG,
  exportChartAsSVG,
  generateChartFilename,
  type ChartRef,
  type PNGExportOptions,
  type SVGExportOptions,
} from "@/lib/utils/chart-export";
import {
  generateCSV,
  generateCSVFilename,
  type CSVColumn,
  type CSVGenerateOptions,
} from "@/lib/utils/csv-export";
import {
  generateZip,
  downloadZip,
  generateZipFilename,
  type ZipFileEntry,
} from "@/lib/utils/zip-export";

// =============================================================================
// Types
// =============================================================================

/**
 * Exportable chart definition.
 */
export interface ExportableChart {
  /** Unique identifier for the chart */
  id: string;
  /** Display name for the chart */
  name: string;
  /** Optional description */
  description?: string;
  /** Reference to the chart container element */
  chartRef: ChartRef;
  /** Category for grouping (e.g., "Performance", "Comparison") */
  category?: string;
  /** Whether this chart is enabled by default */
  defaultSelected?: boolean;
  /** Icon for the chart (component) */
  icon?: React.ReactNode;
}

/**
 * Exportable data definition.
 */
export interface ExportableData<T = unknown> {
  /** Unique identifier for the data export */
  id: string;
  /** Display name for the data export */
  name: string;
  /** Optional description */
  description?: string;
  /** Data to export */
  data: T[];
  /** Column definitions for CSV export */
  columns?: CSVColumn<T>[];
  /** Category for grouping */
  category?: string;
  /** Whether this data is enabled by default */
  defaultSelected?: boolean;
  /** Icon for the data (component) */
  icon?: React.ReactNode;
}

/**
 * Export format type.
 */
export type BulkExportFormat = "png" | "svg" | "csv" | "zip";

/**
 * Chart export options.
 */
export interface ChartExportConfig {
  /** Export format for charts */
  format: "png" | "svg";
  /** PNG scale factor (1-4) */
  scale: number;
  /** Background color for PNG */
  backgroundColor: string;
  /** Padding around charts in pixels */
  padding: number;
  /** Whether to inline styles in SVG */
  inlineStyles: boolean;
}

/**
 * CSV export options.
 */
export interface CSVExportConfig {
  /** Whether to include headers */
  includeHeaders: boolean;
  /** Date format */
  dateFormat: "ISO" | "US" | "EU";
  /** Whether to add BOM for Excel */
  addBOM: boolean;
  /** Decimal precision */
  decimalPrecision: number;
}

/**
 * Export options configuration.
 */
export interface ExportConfig {
  /** Chart export settings */
  chart: ChartExportConfig;
  /** CSV export settings */
  csv: CSVExportConfig;
  /** Whether to include timestamp in filenames */
  includeTimestamp: boolean;
  /** Whether to include metadata file in ZIP */
  includeMetadata: boolean;
  /** Base filename for exports */
  baseFilename: string;
}

/**
 * Export progress state.
 */
export interface ExportProgress {
  /** Current phase of export */
  phase: "idle" | "preparing" | "exporting" | "packaging" | "complete" | "error";
  /** Current item being processed */
  currentItem?: string;
  /** Number of items processed */
  processed: number;
  /** Total number of items */
  total: number;
  /** Error message if any */
  error?: string;
}

/**
 * Export result.
 */
export interface BulkExportResult {
  /** Whether the export was successful */
  success: boolean;
  /** Number of items exported */
  exportedCount: number;
  /** Total size in bytes */
  totalSize?: number;
  /** Filename of the exported file */
  filename?: string;
  /** Error message if any */
  error?: string;
}

/**
 * Props for the ExportPanel component.
 */
export interface ExportPanelProps {
  /** Available charts to export */
  charts?: ExportableChart[];
  /** Available data exports */
  dataExports?: ExportableData[];
  /** Initial export configuration */
  initialConfig?: Partial<ExportConfig>;
  /** Base filename for exports */
  filename?: string;
  /** Callback when export starts */
  onExportStart?: () => void;
  /** Callback when export completes */
  onExportComplete?: (result: BulkExportResult) => void;
  /** Callback when export fails */
  onExportError?: (error: Error) => void;
  /** Additional class names */
  className?: string;
  /** Whether to show as a sheet (side panel) */
  asSheet?: boolean;
  /** Whether the panel is open (controlled mode for sheet) */
  open?: boolean;
  /** Callback when open state changes (controlled mode for sheet) */
  onOpenChange?: (open: boolean) => void;
  /** Trigger element for sheet mode */
  trigger?: React.ReactNode;
  /** Sheet side */
  sheetSide?: "left" | "right" | "top" | "bottom";
  /** Whether the panel is disabled */
  disabled?: boolean;
  /** Compact mode (less padding/margins) */
  compact?: boolean;
}

// =============================================================================
// Constants
// =============================================================================

/** Default export configuration */
const DEFAULT_CONFIG: ExportConfig = {
  chart: {
    format: "png",
    scale: 2,
    backgroundColor: "#ffffff",
    padding: 20,
    inlineStyles: true,
  },
  csv: {
    includeHeaders: true,
    dateFormat: "ISO",
    addBOM: true,
    decimalPrecision: 2,
  },
  includeTimestamp: true,
  includeMetadata: true,
  baseFilename: "campaign-export",
};

/** Resolution presets */
const RESOLUTION_PRESETS = [
  { label: "Standard (1x)", value: 1, description: "72 DPI" },
  { label: "Retina (2x)", value: 2, description: "144 DPI, recommended" },
  { label: "Print (3x)", value: 3, description: "216 DPI, for publications" },
  { label: "Ultra (4x)", value: 4, description: "288 DPI, highest quality" },
];

/** Background color presets */
const BACKGROUND_PRESETS = [
  { label: "White", value: "#ffffff", color: "bg-white border" },
  { label: "Light Gray", value: "#f5f5f5", color: "bg-gray-100" },
  { label: "Dark", value: "#1f2937", color: "bg-gray-800" },
  { label: "Transparent", value: "transparent", color: "bg-transparent border border-dashed" },
];

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generates metadata JSON for the export.
 */
function generateMetadata(
  charts: string[],
  dataExports: string[],
  config: ExportConfig
): string {
  const metadata = {
    exportedAt: new Date().toISOString(),
    version: "1.0.0",
    contents: {
      charts: charts,
      dataExports: dataExports,
    },
    configuration: {
      chartFormat: config.chart.format,
      chartScale: config.chart.scale,
      csvDateFormat: config.csv.dateFormat,
    },
    generator: "Chimera Campaign Analytics",
  };

  return JSON.stringify(metadata, null, 2);
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Selection list item for charts/data.
 */
function SelectionItem({
  id,
  name,
  description,
  category,
  selected,
  onSelectionChange,
  icon,
  disabled,
}: {
  id: string;
  name: string;
  description?: string;
  category?: string;
  selected: boolean;
  onSelectionChange: (id: string, selected: boolean) => void;
  icon?: React.ReactNode;
  disabled?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-lg border p-3 transition-colors",
        selected && !disabled && "border-primary/50 bg-primary/5",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <Checkbox
        id={id}
        checked={selected}
        onCheckedChange={(checked) => onSelectionChange(id, !!checked)}
        disabled={disabled}
        className="mt-0.5"
      />
      <div className="flex-1 min-w-0">
        <Label
          htmlFor={id}
          className={cn(
            "flex items-center gap-2 cursor-pointer",
            disabled && "cursor-not-allowed"
          )}
        >
          {icon && <span className="text-muted-foreground">{icon}</span>}
          <span className="font-medium">{name}</span>
          {category && (
            <Badge variant="secondary" className="text-xs">
              {category}
            </Badge>
          )}
        </Label>
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
      </div>
    </div>
  );
}

/**
 * Progress indicator for export operation.
 */
function ExportProgressIndicator({
  progress,
}: {
  progress: ExportProgress;
}) {
  const percentage =
    progress.total > 0
      ? Math.round((progress.processed / progress.total) * 100)
      : 0;

  if (progress.phase === "idle") return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">
          {progress.phase === "preparing" && "Preparing export..."}
          {progress.phase === "exporting" &&
            `Exporting ${progress.currentItem || "..."}  (${progress.processed}/${progress.total})`}
          {progress.phase === "packaging" && "Creating ZIP archive..."}
          {progress.phase === "complete" && "Export complete!"}
          {progress.phase === "error" && "Export failed"}
        </span>
        <span className="font-medium">{percentage}%</span>
      </div>
      <Progress
        value={percentage}
        className={cn(
          "h-2",
          progress.phase === "error" && "[&>div]:bg-destructive",
          progress.phase === "complete" && "[&>div]:bg-green-500"
        )}
      />
      {progress.error && (
        <p className="text-xs text-destructive flex items-center gap-1">
          <AlertCircle className="size-3" />
          {progress.error}
        </p>
      )}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * ExportPanel - Panel for bulk export with chart selection and configuration.
 *
 * Allows users to:
 * - Select which charts to include in export
 * - Choose export format (PNG/SVG for charts, CSV for data)
 * - Configure resolution, background color, and other options
 * - Generate a ZIP file containing all selected exports
 *
 * @example
 * ```tsx
 * const chartRef1 = useRef<HTMLDivElement>(null);
 * const chartRef2 = useRef<HTMLDivElement>(null);
 *
 * const charts = [
 *   { id: "success-rate", name: "Success Rate", chartRef: chartRef1 },
 *   { id: "latency", name: "Latency Distribution", chartRef: chartRef2 },
 * ];
 *
 * const dataExports = [
 *   { id: "telemetry", name: "Telemetry Events", data: events, columns: telemetryColumns },
 * ];
 *
 * <ExportPanel
 *   charts={charts}
 *   dataExports={dataExports}
 *   filename="campaign-analysis"
 *   onExportComplete={(result) => toast.success(`Exported ${result.exportedCount} items`)}
 * />
 * ```
 */
export function ExportPanel({
  charts = [],
  dataExports = [],
  initialConfig,
  filename = "campaign-export",
  onExportStart,
  onExportComplete,
  onExportError,
  className,
  asSheet = false,
  open: controlledOpen,
  onOpenChange,
  trigger,
  sheetSide = "right",
  disabled = false,
  compact = false,
}: ExportPanelProps) {
  // State
  const [config, setConfig] = React.useState<ExportConfig>(() => ({
    ...DEFAULT_CONFIG,
    ...initialConfig,
    baseFilename: filename,
  }));

  const [selectedCharts, setSelectedCharts] = React.useState<Set<string>>(
    () => new Set(charts.filter((c) => c.defaultSelected !== false).map((c) => c.id))
  );

  const [selectedData, setSelectedData] = React.useState<Set<string>>(
    () => new Set(dataExports.filter((d) => d.defaultSelected !== false).map((d) => d.id))
  );

  const [progress, setProgress] = React.useState<ExportProgress>({
    phase: "idle",
    processed: 0,
    total: 0,
  });

  const [optionsExpanded, setOptionsExpanded] = React.useState(false);
  const [internalOpen, setInternalOpen] = React.useState(false);

  // Controlled vs uncontrolled open state
  const isOpen = controlledOpen ?? internalOpen;
  const setIsOpen = onOpenChange ?? setInternalOpen;

  // Selection helpers
  const handleChartSelection = React.useCallback((id: string, selected: boolean) => {
    setSelectedCharts((prev) => {
      const next = new Set(prev);
      if (selected) {
        next.add(id);
      } else {
        next.delete(id);
      }
      return next;
    });
  }, []);

  const handleDataSelection = React.useCallback((id: string, selected: boolean) => {
    setSelectedData((prev) => {
      const next = new Set(prev);
      if (selected) {
        next.add(id);
      } else {
        next.delete(id);
      }
      return next;
    });
  }, []);

  const selectAllCharts = React.useCallback(() => {
    setSelectedCharts(new Set(charts.map((c) => c.id)));
  }, [charts]);

  const deselectAllCharts = React.useCallback(() => {
    setSelectedCharts(new Set());
  }, []);

  const selectAllData = React.useCallback(() => {
    setSelectedData(new Set(dataExports.map((d) => d.id)));
  }, [dataExports]);

  const deselectAllData = React.useCallback(() => {
    setSelectedData(new Set());
  }, []);

  // Calculate totals
  const totalSelected = selectedCharts.size + selectedData.size;
  const totalAvailable = charts.length + dataExports.length;

  // Export logic
  const handleExport = React.useCallback(async () => {
    if (totalSelected === 0) return;

    const selectedChartItems = charts.filter((c) => selectedCharts.has(c.id));
    const selectedDataItems = dataExports.filter((d) => selectedData.has(d.id));
    const total = selectedChartItems.length + selectedDataItems.length;

    setProgress({ phase: "preparing", processed: 0, total });
    onExportStart?.();

    try {
      const files: ZipFileEntry[] = [];
      let processed = 0;

      // Export charts
      setProgress({ phase: "exporting", processed, total, currentItem: "charts" });

      for (const chart of selectedChartItems) {
        setProgress({
          phase: "exporting",
          processed,
          total,
          currentItem: chart.name,
        });

        if (!chart.chartRef?.current) {
          continue; // Skip if ref is not available
        }

        const chartFilename = generateChartFilename(
          chart.id,
          config.chart.format,
          false
        );

        if (config.chart.format === "png") {
          const pngOptions: PNGExportOptions = {
            scale: config.chart.scale,
            backgroundColor: config.chart.backgroundColor,
            padding: config.chart.padding,
          };

          const result = await exportChartAsPNG(chart.chartRef, pngOptions);
          if (result.success && result.blob) {
            files.push({
              filename: `charts/${chartFilename}`,
              content: result.blob,
              mimeType: "image/png",
            });
          }
        } else {
          const svgOptions: SVGExportOptions = {
            inlineStyles: config.chart.inlineStyles,
            padding: config.chart.padding,
          };

          const result = await exportChartAsSVG(chart.chartRef, svgOptions);
          if (result.success && result.blob) {
            files.push({
              filename: `charts/${chartFilename}`,
              content: result.blob,
              mimeType: "image/svg+xml",
            });
          }
        }

        processed++;
        setProgress({
          phase: "exporting",
          processed,
          total,
          currentItem: chart.name,
        });
      }

      // Export data as CSV
      for (const dataExport of selectedDataItems) {
        setProgress({
          phase: "exporting",
          processed,
          total,
          currentItem: dataExport.name,
        });

        const csvFilename = generateCSVFilename(dataExport.id, false);

        const csvOptions: CSVGenerateOptions = {
          includeHeaders: config.csv.includeHeaders,
          dateFormat: config.csv.dateFormat,
          addBOM: config.csv.addBOM,
          decimalPrecision: config.csv.decimalPrecision,
        };

        const result = dataExport.columns
          ? generateCSV(dataExport.data, dataExport.columns, csvOptions)
          : generateCSV(
              dataExport.data,
              Object.keys(dataExport.data[0] || {}).map((key) => ({
                header: key,
                accessor: key,
              })),
              csvOptions
            );

        if (result.success && result.blob) {
          files.push({
            filename: `data/${csvFilename}`,
            content: result.blob,
            mimeType: "text/csv",
          });
        }

        processed++;
        setProgress({
          phase: "exporting",
          processed,
          total,
          currentItem: dataExport.name,
        });
      }

      // Add metadata if enabled
      if (config.includeMetadata) {
        const metadata = generateMetadata(
          selectedChartItems.map((c) => c.name),
          selectedDataItems.map((d) => d.name),
          config
        );
        files.push({
          filename: "metadata.json",
          content: new TextEncoder().encode(metadata),
          mimeType: "application/json",
        });
      }

      // Create ZIP
      setProgress({ phase: "packaging", processed: total, total });

      const zipFilename = generateZipFilename(
        config.baseFilename,
        config.includeTimestamp
      );

      const zipResult = await generateZip(files, {
        comment: `Chimera Campaign Analytics Export - ${new Date().toISOString()}`,
      });

      if (!zipResult.success || !zipResult.blob) {
        throw new Error(zipResult.error || "Failed to create ZIP archive");
      }

      // Download
      await downloadZip(zipResult.blob, zipFilename);

      // Complete
      setProgress({ phase: "complete", processed: total, total });

      const result: BulkExportResult = {
        success: true,
        exportedCount: files.length,
        totalSize: zipResult.totalSize,
        filename: zipFilename,
      };

      onExportComplete?.(result);

      // Reset after delay
      setTimeout(() => {
        setProgress({ phase: "idle", processed: 0, total: 0 });
      }, 2000);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown export error";

      setProgress({
        phase: "error",
        processed: 0,
        total: 0,
        error: errorMessage,
      });

      onExportError?.(error instanceof Error ? error : new Error(errorMessage));
    }
  }, [
    charts,
    dataExports,
    selectedCharts,
    selectedData,
    config,
    totalSelected,
    onExportStart,
    onExportComplete,
    onExportError,
  ]);

  // Reset progress on open
  React.useEffect(() => {
    if (isOpen) {
      setProgress({ phase: "idle", processed: 0, total: 0 });
    }
  }, [isOpen]);

  // Panel content
  const panelContent = (
    <div className={cn("flex flex-col h-full", compact && "gap-3", !compact && "gap-4")}>
      {/* Chart Selection */}
      {charts.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Image className="size-4 text-muted-foreground" />
              <span className="font-medium text-sm">
                Charts ({selectedCharts.size}/{charts.length})
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={selectAllCharts}
                className="text-xs h-7"
                disabled={disabled || progress.phase !== "idle"}
              >
                Select All
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={deselectAllCharts}
                className="text-xs h-7"
                disabled={disabled || progress.phase !== "idle"}
              >
                Clear
              </Button>
            </div>
          </div>

          <div className={cn("space-y-2", compact ? "max-h-32" : "max-h-48", "overflow-y-auto pr-1")}>
            {charts.map((chart) => (
              <SelectionItem
                key={chart.id}
                id={`chart-${chart.id}`}
                name={chart.name}
                description={chart.description}
                category={chart.category}
                selected={selectedCharts.has(chart.id)}
                onSelectionChange={(_, selected) =>
                  handleChartSelection(chart.id, selected)
                }
                icon={chart.icon || <Image className="size-4" />}
                disabled={disabled || progress.phase !== "idle"}
              />
            ))}
          </div>
        </div>
      )}

      {charts.length > 0 && dataExports.length > 0 && <Separator />}

      {/* Data Selection */}
      {dataExports.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileSpreadsheet className="size-4 text-muted-foreground" />
              <span className="font-medium text-sm">
                Data Exports ({selectedData.size}/{dataExports.length})
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={selectAllData}
                className="text-xs h-7"
                disabled={disabled || progress.phase !== "idle"}
              >
                Select All
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={deselectAllData}
                className="text-xs h-7"
                disabled={disabled || progress.phase !== "idle"}
              >
                Clear
              </Button>
            </div>
          </div>

          <div className={cn("space-y-2", compact ? "max-h-32" : "max-h-48", "overflow-y-auto pr-1")}>
            {dataExports.map((dataExport) => (
              <SelectionItem
                key={dataExport.id}
                id={`data-${dataExport.id}`}
                name={dataExport.name}
                description={dataExport.description}
                category={dataExport.category}
                selected={selectedData.has(dataExport.id)}
                onSelectionChange={(_, selected) =>
                  handleDataSelection(dataExport.id, selected)
                }
                icon={dataExport.icon || <FileSpreadsheet className="size-4" />}
                disabled={disabled || progress.phase !== "idle"}
              />
            ))}
          </div>
        </div>
      )}

      <Separator />

      {/* Export Options */}
      <Collapsible open={optionsExpanded} onOpenChange={setOptionsExpanded}>
        <CollapsibleTrigger asChild>
          <Button
            variant="ghost"
            className="flex w-full items-center justify-between px-0 hover:bg-transparent"
            disabled={disabled || progress.phase !== "idle"}
          >
            <div className="flex items-center gap-2">
              <Settings2 className="size-4 text-muted-foreground" />
              <span className="font-medium text-sm">Export Options</span>
            </div>
            {optionsExpanded ? (
              <ChevronUp className="size-4" />
            ) : (
              <ChevronDown className="size-4" />
            )}
          </Button>
        </CollapsibleTrigger>

        <CollapsibleContent className="space-y-4 pt-3">
          {/* Chart Format */}
          {charts.length > 0 && (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Chart Format</Label>
              <RadioGroup
                value={config.chart.format}
                onValueChange={(value) =>
                  setConfig((prev) => ({
                    ...prev,
                    chart: { ...prev.chart, format: value as "png" | "svg" },
                  }))
                }
                className="flex gap-4"
                disabled={disabled || progress.phase !== "idle"}
              >
                <div className="flex items-center gap-2">
                  <RadioGroupItem value="png" id="format-png" />
                  <Label htmlFor="format-png" className="flex items-center gap-1.5 cursor-pointer">
                    <Image className="size-3.5" />
                    PNG
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <RadioGroupItem value="svg" id="format-svg" />
                  <Label htmlFor="format-svg" className="flex items-center gap-1.5 cursor-pointer">
                    <FileCode2 className="size-3.5" />
                    SVG
                  </Label>
                </div>
              </RadioGroup>
            </div>
          )}

          {/* Resolution (PNG only) */}
          {charts.length > 0 && config.chart.format === "png" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs text-muted-foreground">Resolution</Label>
                <span className="text-xs font-medium">
                  {RESOLUTION_PRESETS.find((p) => p.value === config.chart.scale)?.label}
                </span>
              </div>
              <Slider
                value={[config.chart.scale]}
                onValueChange={([value]) =>
                  setConfig((prev) => ({
                    ...prev,
                    chart: { ...prev.chart, scale: value },
                  }))
                }
                min={1}
                max={4}
                step={1}
                disabled={disabled || progress.phase !== "idle"}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>1x</span>
                <span>2x</span>
                <span>3x</span>
                <span>4x</span>
              </div>
            </div>
          )}

          {/* Background Color (PNG only) */}
          {charts.length > 0 && config.chart.format === "png" && (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Background Color</Label>
              <div className="flex gap-2">
                {BACKGROUND_PRESETS.map((preset) => (
                  <TooltipProvider key={preset.value}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          type="button"
                          className={cn(
                            "size-8 rounded-md transition-all",
                            preset.color,
                            config.chart.backgroundColor === preset.value &&
                              "ring-2 ring-primary ring-offset-2",
                            (disabled || progress.phase !== "idle") &&
                              "opacity-50 cursor-not-allowed"
                          )}
                          onClick={() =>
                            setConfig((prev) => ({
                              ...prev,
                              chart: { ...prev.chart, backgroundColor: preset.value },
                            }))
                          }
                          disabled={disabled || progress.phase !== "idle"}
                          aria-label={preset.label}
                        />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>{preset.label}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                ))}
              </div>
            </div>
          )}

          {/* CSV Options */}
          {dataExports.length > 0 && (
            <div className="space-y-3">
              <Label className="text-xs text-muted-foreground">CSV Options</Label>

              <div className="flex items-center justify-between">
                <Label htmlFor="include-headers" className="text-sm cursor-pointer">
                  Include Headers
                </Label>
                <Switch
                  id="include-headers"
                  checked={config.csv.includeHeaders}
                  onCheckedChange={(checked) =>
                    setConfig((prev) => ({
                      ...prev,
                      csv: { ...prev.csv, includeHeaders: checked },
                    }))
                  }
                  disabled={disabled || progress.phase !== "idle"}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="add-bom" className="text-sm cursor-pointer">
                  Excel Compatible (BOM)
                </Label>
                <Switch
                  id="add-bom"
                  checked={config.csv.addBOM}
                  onCheckedChange={(checked) =>
                    setConfig((prev) => ({
                      ...prev,
                      csv: { ...prev.csv, addBOM: checked },
                    }))
                  }
                  disabled={disabled || progress.phase !== "idle"}
                />
              </div>
            </div>
          )}

          {/* General Options */}
          <div className="space-y-3">
            <Label className="text-xs text-muted-foreground">General</Label>

            <div className="flex items-center justify-between">
              <Label htmlFor="include-timestamp" className="text-sm cursor-pointer">
                Include Timestamp in Filename
              </Label>
              <Switch
                id="include-timestamp"
                checked={config.includeTimestamp}
                onCheckedChange={(checked) =>
                  setConfig((prev) => ({ ...prev, includeTimestamp: checked }))
                }
                disabled={disabled || progress.phase !== "idle"}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="include-metadata" className="text-sm cursor-pointer">
                Include Metadata File
              </Label>
              <Switch
                id="include-metadata"
                checked={config.includeMetadata}
                onCheckedChange={(checked) =>
                  setConfig((prev) => ({ ...prev, includeMetadata: checked }))
                }
                disabled={disabled || progress.phase !== "idle"}
              />
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Progress */}
      {progress.phase !== "idle" && (
        <>
          <Separator />
          <ExportProgressIndicator progress={progress} />
        </>
      )}

      {/* Export Button */}
      <div className="mt-auto pt-4">
        <Button
          className="w-full gap-2"
          onClick={handleExport}
          disabled={
            disabled ||
            totalSelected === 0 ||
            (progress.phase !== "idle" && progress.phase !== "complete" && progress.phase !== "error")
          }
        >
          {progress.phase === "idle" || progress.phase === "error" ? (
            <>
              <Package className="size-4" />
              Export {totalSelected} {totalSelected === 1 ? "Item" : "Items"} as ZIP
            </>
          ) : progress.phase === "complete" ? (
            <>
              <Check className="size-4" />
              Export Complete
            </>
          ) : (
            <>
              <Loader2 className="size-4 animate-spin" />
              Exporting...
            </>
          )}
        </Button>

        {totalSelected === 0 && (
          <p className="text-xs text-muted-foreground text-center mt-2">
            Select at least one item to export
          </p>
        )}
      </div>
    </div>
  );

  // Render as sheet or card
  if (asSheet) {
    return (
      <Sheet open={isOpen} onOpenChange={setIsOpen}>
        {trigger && <SheetTrigger asChild>{trigger}</SheetTrigger>}
        <SheetContent side={sheetSide} className="w-[400px] sm:max-w-[400px]">
          <SheetHeader>
            <SheetTitle className="flex items-center gap-2">
              <Download className="size-5" />
              Bulk Export
            </SheetTitle>
            <SheetDescription>
              Select charts and data to export as a ZIP archive
            </SheetDescription>
          </SheetHeader>
          <div className="flex-1 overflow-auto py-4">{panelContent}</div>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className={cn(compact && "pb-3")}>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Download className="size-5" />
          Bulk Export
        </CardTitle>
        <CardDescription>
          Select charts and data to export as a ZIP archive
        </CardDescription>
      </CardHeader>
      <CardContent>{panelContent}</CardContent>
    </Card>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * ExportPanelSheet - ExportPanel rendered as a slide-out sheet.
 */
export function ExportPanelSheet(props: Omit<ExportPanelProps, "asSheet">) {
  return <ExportPanel asSheet {...props} />;
}

/**
 * ExportPanelCompact - Compact version of ExportPanel.
 */
export function ExportPanelCompact(props: Omit<ExportPanelProps, "compact">) {
  return <ExportPanel compact {...props} />;
}

/**
 * ExportPanelTrigger - Button trigger for ExportPanel sheet.
 */
export function ExportPanelTrigger({
  charts = [],
  dataExports = [],
  ...props
}: ExportPanelProps) {
  const totalItems = charts.length + dataExports.length;

  return (
    <ExportPanel
      charts={charts}
      dataExports={dataExports}
      asSheet
      trigger={
        <Button variant="outline" className="gap-2">
          <Package className="size-4" />
          Bulk Export
          {totalItems > 0 && (
            <Badge variant="secondary" className="ml-1">
              {totalItems}
            </Badge>
          )}
        </Button>
      }
      {...props}
    />
  );
}

// =============================================================================
// Skeleton and State Components
// =============================================================================

/**
 * ExportPanelSkeleton - Loading skeleton for ExportPanel.
 */
export function ExportPanelSkeleton({ className }: { className?: string }) {
  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="size-5 rounded bg-muted animate-pulse" />
          <div className="h-5 w-24 rounded bg-muted animate-pulse" />
        </div>
        <div className="h-4 w-48 rounded bg-muted animate-pulse" />
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Chart selection skeleton */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="h-4 w-20 rounded bg-muted animate-pulse" />
            <div className="h-6 w-24 rounded bg-muted animate-pulse" />
          </div>
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex items-center gap-3 p-3 border rounded-lg">
              <div className="size-4 rounded bg-muted animate-pulse" />
              <div className="flex-1 space-y-1">
                <div className="h-4 w-32 rounded bg-muted animate-pulse" />
                <div className="h-3 w-24 rounded bg-muted animate-pulse" />
              </div>
            </div>
          ))}
        </div>

        {/* Export button skeleton */}
        <div className="h-10 w-full rounded bg-muted animate-pulse" />
      </CardContent>
    </Card>
  );
}

/**
 * ExportPanelEmpty - Empty state when no items available.
 */
export function ExportPanelEmpty({
  className,
  message = "No charts or data available for export",
}: {
  className?: string;
  message?: string;
}) {
  return (
    <Card className={cn("w-full", className)}>
      <CardContent className="flex flex-col items-center justify-center py-8 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-muted">
          <Package className="size-6 text-muted-foreground" />
        </div>
        <h3 className="mt-4 font-medium">Nothing to Export</h3>
        <p className="mt-1 text-sm text-muted-foreground">{message}</p>
      </CardContent>
    </Card>
  );
}

/**
 * ExportPanelDisabled - Disabled state for ExportPanel.
 */
export function ExportPanelDisabled({
  className,
  reason = "Export is currently unavailable",
}: {
  className?: string;
  reason?: string;
}) {
  return (
    <Card className={cn("w-full opacity-60", className)}>
      <CardContent className="flex flex-col items-center justify-center py-8 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-muted">
          <AlertCircle className="size-6 text-muted-foreground" />
        </div>
        <h3 className="mt-4 font-medium">Export Unavailable</h3>
        <p className="mt-1 text-sm text-muted-foreground">{reason}</p>
      </CardContent>
    </Card>
  );
}
