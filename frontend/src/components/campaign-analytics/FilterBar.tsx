"use client";

import * as React from "react";
import {
  Filter,
  X,
  Calendar,
  CheckCircle,
  Circle,
  AlertCircle,
  Clock,
  XCircle,
  Ban,
  ChevronDown,
  Search,
  RotateCcw,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  useTechniqueBreakdown,
  useProviderBreakdown,
} from "@/lib/api/query/campaign-queries";
import type {
  ExecutionStatusEnum,
  TelemetryFilterParams,
} from "@/types/campaign-analytics";

// =============================================================================
// Types
// =============================================================================

/**
 * Date range for filtering.
 */
export interface DateRange {
  start: Date | null;
  end: Date | null;
}

/**
 * Filter state managed by FilterBar.
 */
export interface FilterState {
  /** Selected technique suites */
  techniques: string[];
  /** Selected providers */
  providers: string[];
  /** Date range */
  dateRange: DateRange;
  /** Success status filter */
  successStatus: ("success" | "failure" | "partial" | "all")[];
}

/**
 * Props for the FilterBar component.
 */
export interface FilterBarProps {
  /** Campaign ID to fetch technique/provider options */
  campaignId?: string | null;
  /** Current filter state */
  filters: FilterState;
  /** Callback when filters change */
  onFiltersChange: (filters: FilterState) => void;
  /** Callback when filters are cleared */
  onClearAll?: () => void;
  /** Disabled state */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Show loading skeleton while fetching options */
  isLoading?: boolean;
  /** Compact mode for smaller displays */
  compact?: boolean;
  /** Custom technique options (overrides API fetch) */
  techniqueOptions?: string[];
  /** Custom provider options (overrides API fetch) */
  providerOptions?: string[];
  /** Show active filter count badge */
  showActiveCount?: boolean;
  /** Aria label */
  ariaLabel?: string;
}

/**
 * Props for multi-select popover.
 */
interface MultiSelectPopoverProps {
  label: string;
  icon: LucideIcon;
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
  emptyMessage?: string;
  searchable?: boolean;
  maxHeight?: string;
}

/**
 * Props for status filter popover.
 */
interface StatusFilterPopoverProps {
  selected: FilterState["successStatus"];
  onChange: (selected: FilterState["successStatus"]) => void;
  disabled?: boolean;
}

/**
 * Props for date range picker popover.
 */
interface DateRangePopoverProps {
  dateRange: DateRange;
  onChange: (range: DateRange) => void;
  disabled?: boolean;
}

/**
 * Date range preset options.
 */
type DateRangePreset =
  | "today"
  | "yesterday"
  | "last_7_days"
  | "last_30_days"
  | "last_90_days"
  | "this_month"
  | "last_month"
  | "all_time"
  | "custom";

// =============================================================================
// Status Configuration
// =============================================================================

interface StatusOption {
  value: "success" | "failure" | "partial" | "all";
  label: string;
  icon: LucideIcon;
  className: string;
  description: string;
}

const STATUS_OPTIONS: StatusOption[] = [
  {
    value: "all",
    label: "All Statuses",
    icon: Circle,
    className: "text-muted-foreground",
    description: "Show all execution statuses",
  },
  {
    value: "success",
    label: "Success",
    icon: CheckCircle,
    className: "text-green-600 dark:text-green-400",
    description: "Only successful executions",
  },
  {
    value: "partial",
    label: "Partial Success",
    icon: Clock,
    className: "text-amber-600 dark:text-amber-400",
    description: "Partially successful executions",
  },
  {
    value: "failure",
    label: "Failure",
    icon: XCircle,
    className: "text-red-600 dark:text-red-400",
    description: "Failed executions only",
  },
];

// =============================================================================
// Date Range Presets
// =============================================================================

interface DatePreset {
  value: DateRangePreset;
  label: string;
  getRange: () => DateRange;
}

const DATE_PRESETS: DatePreset[] = [
  {
    value: "today",
    label: "Today",
    getRange: () => {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      return { start: today, end: new Date() };
    },
  },
  {
    value: "yesterday",
    label: "Yesterday",
    getRange: () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      yesterday.setHours(0, 0, 0, 0);
      const end = new Date();
      end.setDate(end.getDate() - 1);
      end.setHours(23, 59, 59, 999);
      return { start: yesterday, end };
    },
  },
  {
    value: "last_7_days",
    label: "Last 7 days",
    getRange: () => {
      const start = new Date();
      start.setDate(start.getDate() - 7);
      start.setHours(0, 0, 0, 0);
      return { start, end: new Date() };
    },
  },
  {
    value: "last_30_days",
    label: "Last 30 days",
    getRange: () => {
      const start = new Date();
      start.setDate(start.getDate() - 30);
      start.setHours(0, 0, 0, 0);
      return { start, end: new Date() };
    },
  },
  {
    value: "last_90_days",
    label: "Last 90 days",
    getRange: () => {
      const start = new Date();
      start.setDate(start.getDate() - 90);
      start.setHours(0, 0, 0, 0);
      return { start, end: new Date() };
    },
  },
  {
    value: "this_month",
    label: "This month",
    getRange: () => {
      const start = new Date();
      start.setDate(1);
      start.setHours(0, 0, 0, 0);
      return { start, end: new Date() };
    },
  },
  {
    value: "last_month",
    label: "Last month",
    getRange: () => {
      const start = new Date();
      start.setMonth(start.getMonth() - 1, 1);
      start.setHours(0, 0, 0, 0);
      const end = new Date();
      end.setDate(0); // Last day of previous month
      end.setHours(23, 59, 59, 999);
      return { start, end };
    },
  },
  {
    value: "all_time",
    label: "All time",
    getRange: () => ({ start: null, end: null }),
  },
];

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format a date for display.
 */
function formatDate(date: Date | null): string {
  if (!date) return "";
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

/**
 * Format a date for input field.
 */
function formatDateForInput(date: Date | null): string {
  if (!date) return "";
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/**
 * Parse date string from input.
 */
function parseDateInput(value: string): Date | null {
  if (!value) return null;
  const date = new Date(value);
  return isNaN(date.getTime()) ? null : date;
}

/**
 * Get the current date range preset.
 */
function getCurrentPreset(dateRange: DateRange): DateRangePreset {
  if (!dateRange.start && !dateRange.end) return "all_time";

  for (const preset of DATE_PRESETS) {
    if (preset.value === "custom" || preset.value === "all_time") continue;

    const presetRange = preset.getRange();
    if (
      presetRange.start &&
      dateRange.start &&
      presetRange.end &&
      dateRange.end
    ) {
      // Check if dates match (within a minute tolerance)
      const startDiff = Math.abs(
        presetRange.start.getTime() - dateRange.start.getTime()
      );
      const endDiff = Math.abs(
        presetRange.end.getTime() - dateRange.end.getTime()
      );
      if (startDiff < 60000 && endDiff < 60000) {
        return preset.value;
      }
    }
  }

  return "custom";
}

/**
 * Count active filters.
 */
function countActiveFilters(filters: FilterState): number {
  let count = 0;
  if (filters.techniques.length > 0) count++;
  if (filters.providers.length > 0) count++;
  if (filters.dateRange.start || filters.dateRange.end) count++;
  if (filters.successStatus.length > 0 && !filters.successStatus.includes("all")) count++;
  return count;
}

/**
 * Create default empty filter state.
 */
export function createDefaultFilterState(): FilterState {
  return {
    techniques: [],
    providers: [],
    dateRange: { start: null, end: null },
    successStatus: [],
  };
}

/**
 * Convert FilterState to TelemetryFilterParams for API.
 */
export function filterStateToParams(filters: FilterState): TelemetryFilterParams {
  const params: TelemetryFilterParams = {};

  if (filters.techniques.length > 0) {
    params.technique_suite = filters.techniques;
  }

  if (filters.providers.length > 0) {
    params.provider = filters.providers;
  }

  if (filters.dateRange.start) {
    params.start_time = filters.dateRange.start.toISOString();
  }

  if (filters.dateRange.end) {
    params.end_time = filters.dateRange.end.toISOString();
  }

  if (filters.successStatus.length > 0 && !filters.successStatus.includes("all")) {
    const statusMap: Record<string, ExecutionStatusEnum> = {
      success: "success" as ExecutionStatusEnum,
      failure: "failure" as ExecutionStatusEnum,
      partial: "partial_success" as ExecutionStatusEnum,
    };
    params.status = filters.successStatus
      .filter((s) => s !== "all")
      .map((s) => statusMap[s]);
  }

  return params;
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Multi-select popover for techniques/providers.
 */
function MultiSelectPopover({
  label,
  icon: Icon,
  options,
  selected,
  onChange,
  disabled = false,
  isLoading = false,
  placeholder = "Select...",
  emptyMessage = "No options available",
  searchable = true,
  maxHeight = "250px",
}: MultiSelectPopoverProps) {
  const [open, setOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const searchInputRef = React.useRef<HTMLInputElement>(null);

  // Filter options based on search
  const filteredOptions = React.useMemo(() => {
    if (!search.trim()) return options;
    const query = search.toLowerCase();
    return options.filter((opt) => opt.toLowerCase().includes(query));
  }, [options, search]);

  // Handle selection toggle
  const handleToggle = React.useCallback(
    (option: string) => {
      if (selected.includes(option)) {
        onChange(selected.filter((s) => s !== option));
      } else {
        onChange([...selected, option]);
      }
    },
    [selected, onChange]
  );

  // Handle select/deselect all
  const handleSelectAll = React.useCallback(() => {
    if (selected.length === filteredOptions.length) {
      onChange([]);
    } else {
      onChange([...filteredOptions]);
    }
  }, [selected.length, filteredOptions, onChange]);

  // Focus search on open
  React.useEffect(() => {
    if (open && searchable && searchInputRef.current) {
      const timer = setTimeout(() => {
        searchInputRef.current?.focus();
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [open, searchable]);

  // Clear search when closed
  React.useEffect(() => {
    if (!open) {
      setSearch("");
    }
  }, [open]);

  const hasSelection = selected.length > 0;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant={hasSelection ? "secondary" : "outline"}
          size="sm"
          disabled={disabled}
          className={cn(
            "h-8 gap-1.5 font-normal",
            hasSelection && "bg-primary/10 border-primary/30"
          )}
          aria-label={`Filter by ${label}`}
        >
          <Icon className="h-3.5 w-3.5 shrink-0" />
          <span className="hidden sm:inline">{label}</span>
          {hasSelection && (
            <Badge
              variant="secondary"
              className="h-4 px-1 text-[10px] font-semibold ml-0.5"
            >
              {selected.length}
            </Badge>
          )}
          <ChevronDown
            className={cn(
              "h-3 w-3 text-muted-foreground transition-transform ml-0.5",
              open && "rotate-180"
            )}
          />
        </Button>
      </PopoverTrigger>

      <PopoverContent
        className="w-64 p-0"
        align="start"
        sideOffset={4}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b">
          <span className="text-sm font-medium">{label}</span>
          {hasSelection && (
            <button
              type="button"
              onClick={() => onChange([])}
              className="text-xs text-primary hover:underline"
            >
              Clear all
            </button>
          )}
        </div>

        {/* Search */}
        {searchable && (
          <div className="flex items-center gap-2 px-3 py-2 border-b">
            <Search className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <input
              ref={searchInputRef}
              type="text"
              placeholder={`Search ${label.toLowerCase()}...`}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="flex-1 text-sm bg-transparent border-none outline-none placeholder:text-muted-foreground"
              aria-label={`Search ${label}`}
            />
            {search && (
              <button
                type="button"
                onClick={() => setSearch("")}
                className="rounded-full p-0.5 hover:bg-muted transition-colors"
                aria-label="Clear search"
              >
                <X className="h-3 w-3 text-muted-foreground" />
              </button>
            )}
          </div>
        )}

        {/* Options list */}
        <ScrollArea style={{ maxHeight }}>
          <div className="p-2">
            {isLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex items-center gap-2 p-2">
                    <Skeleton className="h-4 w-4 rounded-sm" />
                    <Skeleton className="h-4 w-24" />
                  </div>
                ))}
              </div>
            ) : filteredOptions.length === 0 ? (
              <div className="py-4 text-center text-sm text-muted-foreground">
                {search ? `No ${label.toLowerCase()} match "${search}"` : emptyMessage}
              </div>
            ) : (
              <>
                {/* Select all option */}
                <button
                  type="button"
                  onClick={handleSelectAll}
                  className="w-full flex items-center gap-2 p-2 rounded-md text-left text-sm hover:bg-accent transition-colors"
                >
                  <Checkbox
                    checked={
                      filteredOptions.length > 0 &&
                      selected.length === filteredOptions.length
                    }
                    className="shrink-0"
                    aria-hidden="true"
                  />
                  <span className="font-medium">
                    {selected.length === filteredOptions.length
                      ? "Deselect all"
                      : "Select all"}
                  </span>
                </button>
                <Separator className="my-1" />

                {/* Individual options */}
                {filteredOptions.map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => handleToggle(option)}
                    className="w-full flex items-center gap-2 p-2 rounded-md text-left text-sm hover:bg-accent transition-colors"
                  >
                    <Checkbox
                      checked={selected.includes(option)}
                      className="shrink-0"
                      aria-hidden="true"
                    />
                    <span className="truncate">{option}</span>
                  </button>
                ))}
              </>
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        {!isLoading && filteredOptions.length > 0 && (
          <div className="px-3 py-1.5 border-t text-xs text-muted-foreground text-center">
            {selected.length} of {filteredOptions.length} selected
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

/**
 * Status filter popover.
 */
function StatusFilterPopover({
  selected,
  onChange,
  disabled = false,
}: StatusFilterPopoverProps) {
  const [open, setOpen] = React.useState(false);

  const handleToggle = React.useCallback(
    (value: StatusOption["value"]) => {
      if (value === "all") {
        // Toggle "all" clears other selections or selects all
        if (selected.includes("all")) {
          onChange([]);
        } else {
          onChange(["all"]);
        }
      } else {
        // Toggle individual status
        const withoutAll = selected.filter((s) => s !== "all");
        if (withoutAll.includes(value)) {
          onChange(withoutAll.filter((s) => s !== value));
        } else {
          onChange([...withoutAll, value]);
        }
      }
    },
    [selected, onChange]
  );

  const hasSelection = selected.length > 0 && !selected.includes("all");
  const activeCount = selected.filter((s) => s !== "all").length;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant={hasSelection ? "secondary" : "outline"}
          size="sm"
          disabled={disabled}
          className={cn(
            "h-8 gap-1.5 font-normal",
            hasSelection && "bg-primary/10 border-primary/30"
          )}
          aria-label="Filter by status"
        >
          <CheckCircle className="h-3.5 w-3.5 shrink-0" />
          <span className="hidden sm:inline">Status</span>
          {hasSelection && (
            <Badge
              variant="secondary"
              className="h-4 px-1 text-[10px] font-semibold ml-0.5"
            >
              {activeCount}
            </Badge>
          )}
          <ChevronDown
            className={cn(
              "h-3 w-3 text-muted-foreground transition-transform ml-0.5",
              open && "rotate-180"
            )}
          />
        </Button>
      </PopoverTrigger>

      <PopoverContent className="w-56 p-0" align="start" sideOffset={4}>
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b">
          <span className="text-sm font-medium">Status</span>
          {hasSelection && (
            <button
              type="button"
              onClick={() => onChange([])}
              className="text-xs text-primary hover:underline"
            >
              Clear
            </button>
          )}
        </div>

        {/* Options */}
        <div className="p-2">
          {STATUS_OPTIONS.map((option) => {
            const Icon = option.icon;
            const isSelected =
              option.value === "all"
                ? selected.includes("all") || selected.length === 0
                : selected.includes(option.value);

            return (
              <button
                key={option.value}
                type="button"
                onClick={() => handleToggle(option.value)}
                className="w-full flex items-start gap-2 p-2 rounded-md text-left hover:bg-accent transition-colors"
              >
                <Checkbox
                  checked={isSelected}
                  className="shrink-0 mt-0.5"
                  aria-hidden="true"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <Icon className={cn("h-3.5 w-3.5", option.className)} />
                    <span className="text-sm font-medium">{option.label}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {option.description}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </PopoverContent>
    </Popover>
  );
}

/**
 * Date range picker popover.
 */
function DateRangePopover({
  dateRange,
  onChange,
  disabled = false,
}: DateRangePopoverProps) {
  const [open, setOpen] = React.useState(false);
  const [startInput, setStartInput] = React.useState(
    formatDateForInput(dateRange.start)
  );
  const [endInput, setEndInput] = React.useState(
    formatDateForInput(dateRange.end)
  );

  // Sync inputs with props
  React.useEffect(() => {
    setStartInput(formatDateForInput(dateRange.start));
    setEndInput(formatDateForInput(dateRange.end));
  }, [dateRange]);

  const handlePresetClick = React.useCallback(
    (preset: DatePreset) => {
      const range = preset.getRange();
      onChange(range);
      if (preset.value !== "custom") {
        setOpen(false);
      }
    },
    [onChange]
  );

  const handleStartChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setStartInput(value);
      const date = parseDateInput(value);
      if (date || !value) {
        onChange({ ...dateRange, start: date });
      }
    },
    [dateRange, onChange]
  );

  const handleEndChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setEndInput(value);
      const date = parseDateInput(value);
      if (date || !value) {
        onChange({ ...dateRange, end: date });
      }
    },
    [dateRange, onChange]
  );

  const handleClear = React.useCallback(() => {
    onChange({ start: null, end: null });
  }, [onChange]);

  const hasSelection = dateRange.start || dateRange.end;
  const currentPreset = getCurrentPreset(dateRange);

  // Build display label
  const displayLabel = React.useMemo(() => {
    if (!hasSelection) return "Date range";

    const preset = DATE_PRESETS.find((p) => p.value === currentPreset);
    if (preset && currentPreset !== "custom") {
      return preset.label;
    }

    if (dateRange.start && dateRange.end) {
      return `${formatDate(dateRange.start)} - ${formatDate(dateRange.end)}`;
    }
    if (dateRange.start) {
      return `From ${formatDate(dateRange.start)}`;
    }
    if (dateRange.end) {
      return `Until ${formatDate(dateRange.end)}`;
    }
    return "Date range";
  }, [hasSelection, currentPreset, dateRange]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant={hasSelection ? "secondary" : "outline"}
          size="sm"
          disabled={disabled}
          className={cn(
            "h-8 gap-1.5 font-normal",
            hasSelection && "bg-primary/10 border-primary/30"
          )}
          aria-label="Filter by date range"
        >
          <Calendar className="h-3.5 w-3.5 shrink-0" />
          <span className="hidden lg:inline max-w-[120px] truncate">
            {displayLabel}
          </span>
          <span className="lg:hidden">Date</span>
          <ChevronDown
            className={cn(
              "h-3 w-3 text-muted-foreground transition-transform ml-0.5",
              open && "rotate-180"
            )}
          />
        </Button>
      </PopoverTrigger>

      <PopoverContent className="w-72 p-0" align="start" sideOffset={4}>
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b">
          <span className="text-sm font-medium">Date Range</span>
          {hasSelection && (
            <button
              type="button"
              onClick={handleClear}
              className="text-xs text-primary hover:underline"
            >
              Clear
            </button>
          )}
        </div>

        {/* Presets */}
        <div className="p-2 border-b">
          <div className="grid grid-cols-2 gap-1">
            {DATE_PRESETS.map((preset) => (
              <button
                key={preset.value}
                type="button"
                onClick={() => handlePresetClick(preset)}
                className={cn(
                  "px-2 py-1.5 rounded-md text-sm text-left hover:bg-accent transition-colors",
                  currentPreset === preset.value &&
                    "bg-primary/10 text-primary font-medium"
                )}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Custom date inputs */}
        <div className="p-3 space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="filter-start-date" className="text-xs">
              Start Date
            </Label>
            <input
              id="filter-start-date"
              type="date"
              value={startInput}
              onChange={handleStartChange}
              className={cn(
                "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm",
                "shadow-sm transition-colors placeholder:text-muted-foreground",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              )}
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="filter-end-date" className="text-xs">
              End Date
            </Label>
            <input
              id="filter-end-date"
              type="date"
              value={endInput}
              onChange={handleEndChange}
              min={startInput || undefined}
              className={cn(
                "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm",
                "shadow-sm transition-colors placeholder:text-muted-foreground",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              )}
            />
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

/**
 * Active filter chip display.
 */
function ActiveFilterChip({
  label,
  onRemove,
}: {
  label: string;
  onRemove: () => void;
}) {
  return (
    <Badge
      variant="secondary"
      className="gap-1 pr-1 pl-2 py-0.5 max-w-[150px] shrink-0"
    >
      <span className="truncate text-xs">{label}</span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        className="shrink-0 rounded-full p-0.5 hover:bg-muted-foreground/20 transition-colors"
        aria-label={`Remove ${label} filter`}
      >
        <X className="h-3 w-3" />
      </button>
    </Badge>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * FilterBar Component
 *
 * Horizontal bar with filter controls for campaign telemetry analysis:
 * - Technique multi-select
 * - Provider multi-select
 * - Date range picker with presets
 * - Success status filter
 * - Clear all filters button
 *
 * @example Basic usage
 * ```tsx
 * const [filters, setFilters] = useState(createDefaultFilterState());
 *
 * <FilterBar
 *   campaignId={selectedCampaignId}
 *   filters={filters}
 *   onFiltersChange={setFilters}
 *   onClearAll={() => setFilters(createDefaultFilterState())}
 * />
 * ```
 *
 * @example With custom options
 * ```tsx
 * <FilterBar
 *   filters={filters}
 *   onFiltersChange={setFilters}
 *   techniqueOptions={['cognitive_hacking', 'dan_persona', 'payload_splitting']}
 *   providerOptions={['openai', 'anthropic', 'google']}
 * />
 * ```
 */
export function FilterBar({
  campaignId,
  filters,
  onFiltersChange,
  onClearAll,
  disabled = false,
  className,
  isLoading = false,
  compact = false,
  techniqueOptions: customTechniqueOptions,
  providerOptions: customProviderOptions,
  showActiveCount = true,
  ariaLabel = "Filter controls",
}: FilterBarProps) {
  // Fetch technique options from API if campaignId provided and no custom options
  const {
    data: techniqueBreakdown,
    isLoading: techniquesLoading,
  } = useTechniqueBreakdown(campaignId || "", !!campaignId && !customTechniqueOptions);

  // Fetch provider options from API if campaignId provided and no custom options
  const {
    data: providerBreakdown,
    isLoading: providersLoading,
  } = useProviderBreakdown(campaignId || "", !!campaignId && !customProviderOptions);

  // Build technique options
  const techniqueOptions = React.useMemo(() => {
    if (customTechniqueOptions) return customTechniqueOptions;
    return techniqueBreakdown?.items.map((item) => item.name) || [];
  }, [customTechniqueOptions, techniqueBreakdown]);

  // Build provider options
  const providerOptions = React.useMemo(() => {
    if (customProviderOptions) return customProviderOptions;
    return providerBreakdown?.items.map((item) => item.name) || [];
  }, [customProviderOptions, providerBreakdown]);

  // Handle individual filter changes
  const handleTechniquesChange = React.useCallback(
    (techniques: string[]) => {
      onFiltersChange({ ...filters, techniques });
    },
    [filters, onFiltersChange]
  );

  const handleProvidersChange = React.useCallback(
    (providers: string[]) => {
      onFiltersChange({ ...filters, providers });
    },
    [filters, onFiltersChange]
  );

  const handleDateRangeChange = React.useCallback(
    (dateRange: DateRange) => {
      onFiltersChange({ ...filters, dateRange });
    },
    [filters, onFiltersChange]
  );

  const handleStatusChange = React.useCallback(
    (successStatus: FilterState["successStatus"]) => {
      onFiltersChange({ ...filters, successStatus });
    },
    [filters, onFiltersChange]
  );

  const handleClearAll = React.useCallback(() => {
    if (onClearAll) {
      onClearAll();
    } else {
      onFiltersChange(createDefaultFilterState());
    }
  }, [onClearAll, onFiltersChange]);

  // Count active filters
  const activeFilterCount = countActiveFilters(filters);
  const hasActiveFilters = activeFilterCount > 0;

  // Build active filter chips for display
  const activeChips = React.useMemo(() => {
    const chips: Array<{ id: string; label: string; onRemove: () => void }> = [];

    if (filters.techniques.length > 0) {
      if (filters.techniques.length <= 2) {
        filters.techniques.forEach((t) => {
          chips.push({
            id: `technique-${t}`,
            label: t,
            onRemove: () =>
              handleTechniquesChange(
                filters.techniques.filter((x) => x !== t)
              ),
          });
        });
      } else {
        chips.push({
          id: "techniques",
          label: `${filters.techniques.length} techniques`,
          onRemove: () => handleTechniquesChange([]),
        });
      }
    }

    if (filters.providers.length > 0) {
      if (filters.providers.length <= 2) {
        filters.providers.forEach((p) => {
          chips.push({
            id: `provider-${p}`,
            label: p,
            onRemove: () =>
              handleProvidersChange(filters.providers.filter((x) => x !== p)),
          });
        });
      } else {
        chips.push({
          id: "providers",
          label: `${filters.providers.length} providers`,
          onRemove: () => handleProvidersChange([]),
        });
      }
    }

    if (filters.dateRange.start || filters.dateRange.end) {
      const currentPreset = getCurrentPreset(filters.dateRange);
      const preset = DATE_PRESETS.find((p) => p.value === currentPreset);
      const label =
        preset && currentPreset !== "custom" && currentPreset !== "all_time"
          ? preset.label
          : "Custom dates";
      chips.push({
        id: "dateRange",
        label,
        onRemove: () => handleDateRangeChange({ start: null, end: null }),
      });
    }

    if (filters.successStatus.length > 0 && !filters.successStatus.includes("all")) {
      const statusLabels = filters.successStatus
        .map((s) => STATUS_OPTIONS.find((o) => o.value === s)?.label || s)
        .join(", ");
      chips.push({
        id: "status",
        label: statusLabels,
        onRemove: () => handleStatusChange([]),
      });
    }

    return chips;
  }, [
    filters,
    handleTechniquesChange,
    handleProvidersChange,
    handleDateRangeChange,
    handleStatusChange,
  ]);

  const isFilterLoading = isLoading || techniquesLoading || providersLoading;

  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-2",
        compact && "gap-1",
        className
      )}
      role="toolbar"
      aria-label={ariaLabel}
    >
      {/* Filter icon with count */}
      <div className="flex items-center gap-1.5 shrink-0">
        <Filter className="h-4 w-4 text-muted-foreground" />
        {showActiveCount && activeFilterCount > 0 && (
          <Badge
            variant="default"
            className="h-5 px-1.5 text-[10px] font-bold"
          >
            {activeFilterCount}
          </Badge>
        )}
      </div>

      <Separator orientation="vertical" className="h-6" />

      {/* Filter controls */}
      <div className="flex flex-wrap items-center gap-1.5">
        {/* Technique filter */}
        <MultiSelectPopover
          label="Techniques"
          icon={Filter}
          options={techniqueOptions}
          selected={filters.techniques}
          onChange={handleTechniquesChange}
          disabled={disabled}
          isLoading={isFilterLoading && techniqueOptions.length === 0}
          placeholder="Filter techniques"
          emptyMessage="No techniques available"
        />

        {/* Provider filter */}
        <MultiSelectPopover
          label="Providers"
          icon={Filter}
          options={providerOptions}
          selected={filters.providers}
          onChange={handleProvidersChange}
          disabled={disabled}
          isLoading={isFilterLoading && providerOptions.length === 0}
          placeholder="Filter providers"
          emptyMessage="No providers available"
        />

        {/* Date range filter */}
        <DateRangePopover
          dateRange={filters.dateRange}
          onChange={handleDateRangeChange}
          disabled={disabled}
        />

        {/* Status filter */}
        <StatusFilterPopover
          selected={filters.successStatus}
          onChange={handleStatusChange}
          disabled={disabled}
        />
      </div>

      {/* Active filter chips (on larger screens) */}
      {!compact && activeChips.length > 0 && (
        <>
          <Separator orientation="vertical" className="h-6 hidden lg:block" />
          <div className="hidden lg:flex flex-wrap items-center gap-1">
            {activeChips.slice(0, 4).map((chip) => (
              <ActiveFilterChip
                key={chip.id}
                label={chip.label}
                onRemove={chip.onRemove}
              />
            ))}
            {activeChips.length > 4 && (
              <Badge variant="outline" className="text-xs">
                +{activeChips.length - 4} more
              </Badge>
            )}
          </div>
        </>
      )}

      {/* Clear all button */}
      {hasActiveFilters && (
        <>
          <div className="flex-1 min-w-0" />
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClearAll}
            className="h-8 gap-1.5 text-muted-foreground hover:text-foreground shrink-0"
            aria-label="Clear all filters"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Clear all</span>
          </Button>
        </>
      )}
    </div>
  );
}

// =============================================================================
// Loading & State Components
// =============================================================================

/**
 * Loading skeleton for FilterBar.
 */
export function FilterBarSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <Skeleton className="h-4 w-4" />
      <Separator orientation="vertical" className="h-6" />
      <div className="flex items-center gap-1.5">
        <Skeleton className="h-8 w-24 rounded-md" />
        <Skeleton className="h-8 w-24 rounded-md" />
        <Skeleton className="h-8 w-28 rounded-md" />
        <Skeleton className="h-8 w-20 rounded-md" />
      </div>
    </div>
  );
}

/**
 * Empty/disabled state for FilterBar.
 */
export function FilterBarEmpty({
  message = "Select a campaign to enable filters",
  className,
}: {
  message?: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 text-muted-foreground",
        className
      )}
    >
      <Filter className="h-4 w-4" />
      <Separator orientation="vertical" className="h-6" />
      <span className="text-sm">{message}</span>
    </div>
  );
}

// =============================================================================
// Convenience Variants
// =============================================================================

/**
 * CompactFilterBar - Pre-configured for smaller displays.
 */
export function CompactFilterBar(
  props: Omit<FilterBarProps, "compact" | "showActiveCount">
) {
  return <FilterBar {...props} compact showActiveCount={false} />;
}

/**
 * InlineFilterBar - Minimal inline variant.
 */
export function InlineFilterBar({
  filters,
  onFiltersChange,
  onClearAll,
  disabled,
  className,
}: Pick<
  FilterBarProps,
  "filters" | "onFiltersChange" | "onClearAll" | "disabled" | "className"
>) {
  return (
    <FilterBar
      filters={filters}
      onFiltersChange={onFiltersChange}
      onClearAll={onClearAll}
      disabled={disabled}
      className={className}
      compact
      showActiveCount={false}
    />
  );
}

export default FilterBar;
