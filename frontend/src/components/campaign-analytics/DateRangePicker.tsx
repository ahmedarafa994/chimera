"use client";

import * as React from "react";
import {
  Calendar,
  ChevronDown,
  X,
  CalendarClock,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import {
  format,
  subDays,
  subMonths,
  startOfMonth,
  endOfMonth,
  startOfWeek,
  endOfWeek,
  startOfYear,
  endOfYear,
  isSameDay,
  isWithinInterval,
  addMonths,
  getDaysInMonth,
  startOfDay,
  endOfDay,
  isAfter,
  isBefore,
  isValid,
  parseISO,
} from "date-fns";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

// =============================================================================
// Types
// =============================================================================

/**
 * Date range value with start and end dates.
 */
export interface DateRangeValue {
  /** Start date of the range (null if not set) */
  start: Date | null;
  /** End date of the range (null if not set) */
  end: Date | null;
}

/**
 * Preset identifier for quick date range selection.
 */
export type DateRangePreset =
  | "today"
  | "yesterday"
  | "last_7_days"
  | "last_30_days"
  | "last_90_days"
  | "this_week"
  | "last_week"
  | "this_month"
  | "last_month"
  | "this_year"
  | "last_year"
  | "all_time"
  | "custom";

/**
 * Size variant for the date range picker.
 */
export type DateRangePickerSize = "sm" | "md" | "lg";

/**
 * Alignment for the popover.
 */
export type DateRangePickerAlign = "start" | "center" | "end";

/**
 * Configuration for a preset option.
 */
export interface PresetConfig {
  /** Unique identifier */
  value: DateRangePreset;
  /** Display label */
  label: string;
  /** Short label for compact display */
  shortLabel?: string;
  /** Description for tooltip */
  description?: string;
  /** Function to calculate the date range */
  getRange: () => DateRangeValue;
}

/**
 * Props for the DateRangePicker component.
 */
export interface DateRangePickerProps {
  /** Current date range value */
  value: DateRangeValue;
  /** Callback when the date range changes */
  onChange: (value: DateRangeValue) => void;
  /** Placeholder text when no range is selected */
  placeholder?: string;
  /** Disabled state */
  disabled?: boolean;
  /** Additional CSS classes for the trigger button */
  className?: string;
  /** Size variant */
  size?: DateRangePickerSize;
  /** Popover alignment */
  align?: DateRangePickerAlign;
  /** Minimum selectable date */
  minDate?: Date;
  /** Maximum selectable date */
  maxDate?: Date;
  /** Whether to show preset options */
  showPresets?: boolean;
  /** Custom preset configurations (replaces defaults if provided) */
  presets?: PresetConfig[];
  /** Which presets to show (subset of defaults) */
  enabledPresets?: DateRangePreset[];
  /** Whether to show the calendar for custom selection */
  showCalendar?: boolean;
  /** Whether to show clear button */
  showClear?: boolean;
  /** Whether to close popover on selection */
  closeOnSelect?: boolean;
  /** Date format for display (date-fns format string) */
  dateFormat?: string;
  /** Aria label for the trigger button */
  ariaLabel?: string;
  /** Callback when popover opens/closes */
  onOpenChange?: (open: boolean) => void;
  /** Controlled open state */
  open?: boolean;
}

/**
 * Props for the mini calendar component.
 */
interface MiniCalendarProps {
  /** Currently displayed month */
  month: Date;
  /** Selected date range */
  selectedRange: DateRangeValue;
  /** Callback when a date is clicked */
  onDateClick: (date: Date) => void;
  /** Minimum selectable date */
  minDate?: Date;
  /** Maximum selectable date */
  maxDate?: Date;
  /** Currently hovered date (for range preview) */
  hoveredDate: Date | null;
  /** Callback when date is hovered */
  onDateHover: (date: Date | null) => void;
  /** Selection mode: start or end */
  selectionMode: "start" | "end";
}

// =============================================================================
// Default Presets Configuration
// =============================================================================

const DEFAULT_PRESETS: PresetConfig[] = [
  {
    value: "today",
    label: "Today",
    shortLabel: "Today",
    description: "Only today's data",
    getRange: () => ({
      start: startOfDay(new Date()),
      end: endOfDay(new Date()),
    }),
  },
  {
    value: "yesterday",
    label: "Yesterday",
    shortLabel: "Yesterday",
    description: "Only yesterday's data",
    getRange: () => {
      const yesterday = subDays(new Date(), 1);
      return {
        start: startOfDay(yesterday),
        end: endOfDay(yesterday),
      };
    },
  },
  {
    value: "last_7_days",
    label: "Last 7 days",
    shortLabel: "7 days",
    description: "Past 7 days including today",
    getRange: () => ({
      start: startOfDay(subDays(new Date(), 6)),
      end: endOfDay(new Date()),
    }),
  },
  {
    value: "last_30_days",
    label: "Last 30 days",
    shortLabel: "30 days",
    description: "Past 30 days including today",
    getRange: () => ({
      start: startOfDay(subDays(new Date(), 29)),
      end: endOfDay(new Date()),
    }),
  },
  {
    value: "last_90_days",
    label: "Last 90 days",
    shortLabel: "90 days",
    description: "Past 90 days including today",
    getRange: () => ({
      start: startOfDay(subDays(new Date(), 89)),
      end: endOfDay(new Date()),
    }),
  },
  {
    value: "this_week",
    label: "This week",
    shortLabel: "This week",
    description: "Current calendar week",
    getRange: () => ({
      start: startOfWeek(new Date(), { weekStartsOn: 0 }),
      end: endOfDay(new Date()),
    }),
  },
  {
    value: "last_week",
    label: "Last week",
    shortLabel: "Last week",
    description: "Previous calendar week",
    getRange: () => {
      const lastWeekStart = startOfWeek(subDays(new Date(), 7), { weekStartsOn: 0 });
      const lastWeekEnd = endOfWeek(subDays(new Date(), 7), { weekStartsOn: 0 });
      return {
        start: lastWeekStart,
        end: endOfDay(lastWeekEnd),
      };
    },
  },
  {
    value: "this_month",
    label: "This month",
    shortLabel: "This month",
    description: "Current calendar month",
    getRange: () => ({
      start: startOfMonth(new Date()),
      end: endOfDay(new Date()),
    }),
  },
  {
    value: "last_month",
    label: "Last month",
    shortLabel: "Last month",
    description: "Previous calendar month",
    getRange: () => {
      const lastMonth = subMonths(new Date(), 1);
      return {
        start: startOfMonth(lastMonth),
        end: endOfMonth(lastMonth),
      };
    },
  },
  {
    value: "this_year",
    label: "This year",
    shortLabel: "This year",
    description: "Current calendar year",
    getRange: () => ({
      start: startOfYear(new Date()),
      end: endOfDay(new Date()),
    }),
  },
  {
    value: "last_year",
    label: "Last year",
    shortLabel: "Last year",
    description: "Previous calendar year",
    getRange: () => {
      const lastYear = subMonths(new Date(), 12);
      return {
        start: startOfYear(lastYear),
        end: endOfYear(lastYear),
      };
    },
  },
  {
    value: "all_time",
    label: "All time",
    shortLabel: "All",
    description: "No date restrictions",
    getRange: () => ({ start: null, end: null }),
  },
];

// Commonly used presets for quick access
const QUICK_PRESETS: DateRangePreset[] = [
  "last_7_days",
  "last_30_days",
  "this_month",
  "last_month",
  "all_time",
];

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Parse a date string to Date object safely.
 */
function parseDate(dateStr: string | Date | null | undefined): Date | null {
  if (!dateStr) return null;
  if (dateStr instanceof Date) {
    return isValid(dateStr) ? dateStr : null;
  }
  try {
    const parsed = parseISO(dateStr);
    return isValid(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * Format date for input field (yyyy-MM-dd).
 */
function formatDateForInput(date: Date | null): string {
  if (!date || !isValid(date)) return "";
  return format(date, "yyyy-MM-dd");
}

/**
 * Check if a date is within the allowed range.
 */
function isDateInRange(
  date: Date,
  minDate?: Date,
  maxDate?: Date
): boolean {
  if (minDate && isBefore(date, startOfDay(minDate))) return false;
  if (maxDate && isAfter(date, endOfDay(maxDate))) return false;
  return true;
}

/**
 * Detect which preset matches the current range.
 */
function detectPreset(
  range: DateRangeValue,
  presets: PresetConfig[]
): DateRangePreset {
  if (!range.start && !range.end) return "all_time";

  for (const preset of presets) {
    if (preset.value === "custom" || preset.value === "all_time") continue;

    const presetRange = preset.getRange();
    if (presetRange.start && range.start && presetRange.end && range.end) {
      // Check if dates match (within a second tolerance for time differences)
      const startMatch = Math.abs(
        presetRange.start.getTime() - range.start.getTime()
      ) < 1000;
      const endMatch = Math.abs(
        presetRange.end.getTime() - range.end.getTime()
      ) < 1000;
      if (startMatch && endMatch) {
        return preset.value;
      }
    }
  }

  return "custom";
}

/**
 * Get display label for a date range.
 */
function getDisplayLabel(
  range: DateRangeValue,
  presets: PresetConfig[],
  dateFormat: string,
  placeholder: string
): string {
  if (!range.start && !range.end) return placeholder;

  const preset = detectPreset(range, presets);
  if (preset !== "custom") {
    const presetConfig = presets.find((p) => p.value === preset);
    if (presetConfig) return presetConfig.shortLabel || presetConfig.label;
  }

  if (range.start && range.end) {
    if (isSameDay(range.start, range.end)) {
      return format(range.start, dateFormat);
    }
    return `${format(range.start, dateFormat)} – ${format(range.end, dateFormat)}`;
  }
  if (range.start) {
    return `From ${format(range.start, dateFormat)}`;
  }
  if (range.end) {
    return `Until ${format(range.end, dateFormat)}`;
  }
  return placeholder;
}

// =============================================================================
// Mini Calendar Component
// =============================================================================

const WEEKDAYS = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];

/**
 * Mini calendar for date selection.
 */
function MiniCalendar({
  month,
  selectedRange,
  onDateClick,
  minDate,
  maxDate,
  hoveredDate,
  onDateHover,
  selectionMode,
}: MiniCalendarProps) {
  const year = month.getFullYear();
  const monthIndex = month.getMonth();
  const firstDayOfMonth = new Date(year, monthIndex, 1);
  const startDayOfWeek = firstDayOfMonth.getDay();
  const daysInMonth = getDaysInMonth(month);

  // Generate calendar days
  const calendarDays: (Date | null)[] = [];

  // Add empty cells for days before the first of the month
  for (let i = 0; i < startDayOfWeek; i++) {
    calendarDays.push(null);
  }

  // Add days of the month
  for (let day = 1; day <= daysInMonth; day++) {
    calendarDays.push(new Date(year, monthIndex, day));
  }

  // Check if a date is in the selected range
  const isInRange = (date: Date): boolean => {
    if (!selectedRange.start || !selectedRange.end) return false;
    return isWithinInterval(date, {
      start: startOfDay(selectedRange.start),
      end: endOfDay(selectedRange.end),
    });
  };

  // Check if a date is in the preview range (when hovering)
  const isInPreviewRange = (date: Date): boolean => {
    if (!hoveredDate) return false;
    if (selectionMode === "end" && selectedRange.start && !selectedRange.end) {
      const rangeStart = selectedRange.start;
      const rangeEnd = hoveredDate;
      if (isAfter(rangeEnd, rangeStart)) {
        return isWithinInterval(date, {
          start: startOfDay(rangeStart),
          end: endOfDay(rangeEnd),
        });
      }
    }
    return false;
  };

  return (
    <div className="w-full">
      {/* Weekday headers */}
      <div className="grid grid-cols-7 gap-0.5 mb-1">
        {WEEKDAYS.map((day) => (
          <div
            key={day}
            className="h-6 flex items-center justify-center text-xs font-medium text-muted-foreground"
          >
            {day}
          </div>
        ))}
      </div>

      {/* Calendar grid */}
      <div className="grid grid-cols-7 gap-0.5">
        {calendarDays.map((date, index) => {
          if (!date) {
            return <div key={`empty-${index}`} className="h-7" />;
          }

          const isDisabled = !isDateInRange(date, minDate, maxDate);
          const isStart = selectedRange.start && isSameDay(date, selectedRange.start);
          const isEnd = selectedRange.end && isSameDay(date, selectedRange.end);
          const isSelected = isStart || isEnd;
          const inRange = isInRange(date);
          const inPreview = isInPreviewRange(date);
          const isToday = isSameDay(date, new Date());

          return (
            <button
              key={date.getTime()}
              type="button"
              disabled={isDisabled}
              onClick={() => onDateClick(date)}
              onMouseEnter={() => onDateHover(date)}
              onMouseLeave={() => onDateHover(null)}
              className={cn(
                "h-7 w-full flex items-center justify-center text-xs rounded-sm transition-colors",
                "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-1",
                // Disabled state
                isDisabled && "text-muted-foreground/40 cursor-not-allowed",
                // Default state
                !isDisabled && !isSelected && !inRange && !inPreview &&
                  "hover:bg-accent hover:text-accent-foreground",
                // Today indicator
                isToday && !isSelected && "ring-1 ring-primary/30",
                // In range (between start and end)
                (inRange || inPreview) && !isSelected && "bg-primary/10",
                // Selected (start or end)
                isSelected && "bg-primary text-primary-foreground font-medium",
                // Start of range (rounded left)
                isStart && selectedRange.end && "rounded-l-md rounded-r-none",
                // End of range (rounded right)
                isEnd && selectedRange.start && "rounded-l-none rounded-r-md",
                // Both (single day range)
                isStart && isEnd && "rounded-md"
              )}
              aria-label={format(date, "MMMM d, yyyy")}
              aria-selected={isSelected || undefined}
              aria-disabled={isDisabled}
            >
              {date.getDate()}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * DateRangePicker Component
 *
 * A comprehensive date range selector with preset options and custom calendar.
 * Uses date-fns for formatting and follows shadcn/ui patterns.
 *
 * @example Basic usage
 * ```tsx
 * const [dateRange, setDateRange] = useState<DateRangeValue>({
 *   start: null,
 *   end: null,
 * });
 *
 * <DateRangePicker
 *   value={dateRange}
 *   onChange={setDateRange}
 *   placeholder="Select date range"
 * />
 * ```
 *
 * @example With presets and calendar
 * ```tsx
 * <DateRangePicker
 *   value={dateRange}
 *   onChange={setDateRange}
 *   showPresets
 *   showCalendar
 *   enabledPresets={["last_7_days", "last_30_days", "this_month"]}
 * />
 * ```
 *
 * @example With min/max constraints
 * ```tsx
 * <DateRangePicker
 *   value={dateRange}
 *   onChange={setDateRange}
 *   minDate={new Date(2024, 0, 1)}
 *   maxDate={new Date()}
 * />
 * ```
 */
export function DateRangePicker({
  value,
  onChange,
  placeholder = "Select date range",
  disabled = false,
  className,
  size = "md",
  align = "start",
  minDate,
  maxDate,
  showPresets = true,
  presets: customPresets,
  enabledPresets,
  showCalendar = true,
  showClear = true,
  closeOnSelect = true,
  dateFormat = "MMM d, yyyy",
  ariaLabel = "Select date range",
  onOpenChange,
  open: controlledOpen,
}: DateRangePickerProps) {
  const [internalOpen, setInternalOpen] = React.useState(false);
  const [calendarMonth, setCalendarMonth] = React.useState(() =>
    value.start || new Date()
  );
  const [hoveredDate, setHoveredDate] = React.useState<Date | null>(null);
  const [selectionMode, setSelectionMode] = React.useState<"start" | "end">("start");
  const [startInput, setStartInput] = React.useState(formatDateForInput(value.start));
  const [endInput, setEndInput] = React.useState(formatDateForInput(value.end));

  // Use controlled or internal open state
  const isOpen = controlledOpen !== undefined ? controlledOpen : internalOpen;
  const setIsOpen = React.useCallback(
    (open: boolean) => {
      if (controlledOpen === undefined) {
        setInternalOpen(open);
      }
      onOpenChange?.(open);
    },
    [controlledOpen, onOpenChange]
  );

  // Build presets list
  const presets = React.useMemo(() => {
    if (customPresets) return customPresets;
    if (enabledPresets) {
      return DEFAULT_PRESETS.filter((p) => enabledPresets.includes(p.value));
    }
    return DEFAULT_PRESETS;
  }, [customPresets, enabledPresets]);

  // Sync inputs with value prop
  React.useEffect(() => {
    setStartInput(formatDateForInput(value.start));
    setEndInput(formatDateForInput(value.end));
  }, [value]);

  // Detect current preset
  const currentPreset = React.useMemo(
    () => detectPreset(value, presets),
    [value, presets]
  );

  // Display label
  const displayLabel = React.useMemo(
    () => getDisplayLabel(value, presets, dateFormat, placeholder),
    [value, presets, dateFormat, placeholder]
  );

  // Has selection
  const hasSelection = value.start || value.end;

  // Handle preset click
  const handlePresetClick = React.useCallback(
    (preset: PresetConfig) => {
      const range = preset.getRange();
      onChange(range);
      if (closeOnSelect && preset.value !== "custom") {
        setIsOpen(false);
      }
      if (range.start) {
        setCalendarMonth(range.start);
      }
    },
    [onChange, closeOnSelect, setIsOpen]
  );

  // Handle date click from calendar
  const handleDateClick = React.useCallback(
    (date: Date) => {
      if (selectionMode === "start" || !value.start) {
        // Starting a new selection
        onChange({ start: startOfDay(date), end: null });
        setSelectionMode("end");
      } else {
        // Completing the selection
        const newStart = value.start;
        const newEnd = date;

        // Ensure start is before end
        if (isAfter(newStart, newEnd)) {
          onChange({
            start: startOfDay(newEnd),
            end: endOfDay(newStart),
          });
        } else {
          onChange({
            start: startOfDay(newStart),
            end: endOfDay(newEnd),
          });
        }
        setSelectionMode("start");
        if (closeOnSelect) {
          setIsOpen(false);
        }
      }
    },
    [value.start, selectionMode, onChange, closeOnSelect, setIsOpen]
  );

  // Handle input change
  const handleStartInputChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const inputValue = e.target.value;
      setStartInput(inputValue);
      const parsed = parseDate(inputValue);
      if (parsed && isDateInRange(parsed, minDate, maxDate)) {
        onChange({ ...value, start: startOfDay(parsed) });
        setCalendarMonth(parsed);
      } else if (!inputValue) {
        onChange({ ...value, start: null });
      }
    },
    [value, onChange, minDate, maxDate]
  );

  const handleEndInputChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const inputValue = e.target.value;
      setEndInput(inputValue);
      const parsed = parseDate(inputValue);
      if (parsed && isDateInRange(parsed, minDate, maxDate)) {
        onChange({ ...value, end: endOfDay(parsed) });
        setCalendarMonth(parsed);
      } else if (!inputValue) {
        onChange({ ...value, end: null });
      }
    },
    [value, onChange, minDate, maxDate]
  );

  // Handle clear
  const handleClear = React.useCallback(() => {
    onChange({ start: null, end: null });
    setSelectionMode("start");
  }, [onChange]);

  // Navigate calendar month
  const goToPreviousMonth = React.useCallback(() => {
    setCalendarMonth((prev) => addMonths(prev, -1));
  }, []);

  const goToNextMonth = React.useCallback(() => {
    setCalendarMonth((prev) => addMonths(prev, 1));
  }, []);

  // Size classes
  const sizeClasses = {
    sm: "h-7 text-xs px-2 gap-1",
    md: "h-8 text-sm px-3 gap-1.5",
    lg: "h-10 text-base px-4 gap-2",
  };

  const iconSize = {
    sm: "h-3 w-3",
    md: "h-3.5 w-3.5",
    lg: "h-4 w-4",
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant={hasSelection ? "secondary" : "outline"}
          disabled={disabled}
          className={cn(
            "font-normal justify-start",
            sizeClasses[size],
            hasSelection && "bg-primary/10 border-primary/30",
            className
          )}
          aria-label={ariaLabel}
        >
          <Calendar className={cn("shrink-0", iconSize[size])} />
          <span className="truncate max-w-[200px]">{displayLabel}</span>
          <ChevronDown
            className={cn(
              "shrink-0 text-muted-foreground transition-transform ml-auto",
              iconSize[size],
              isOpen && "rotate-180"
            )}
          />
        </Button>
      </PopoverTrigger>

      <PopoverContent
        className="w-auto p-0"
        align={align}
        sideOffset={4}
      >
        <div className="flex">
          {/* Presets sidebar */}
          {showPresets && (
            <div className="border-r p-2 w-36">
              <div className="text-xs font-medium text-muted-foreground px-2 py-1 mb-1">
                Quick Select
              </div>
              <div className="space-y-0.5">
                {presets.map((preset) => (
                  <button
                    key={preset.value}
                    type="button"
                    onClick={() => handlePresetClick(preset)}
                    className={cn(
                      "w-full text-left px-2 py-1.5 text-sm rounded-md transition-colors",
                      "hover:bg-accent hover:text-accent-foreground",
                      currentPreset === preset.value &&
                        "bg-primary/10 text-primary font-medium"
                    )}
                    title={preset.description}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Calendar section */}
          {showCalendar && (
            <div className="p-3 min-w-[240px]">
              {/* Month navigation */}
              <div className="flex items-center justify-between mb-2">
                <button
                  type="button"
                  onClick={goToPreviousMonth}
                  className="p-1 rounded-md hover:bg-accent transition-colors"
                  aria-label="Previous month"
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <span className="text-sm font-medium">
                  {format(calendarMonth, "MMMM yyyy")}
                </span>
                <button
                  type="button"
                  onClick={goToNextMonth}
                  className="p-1 rounded-md hover:bg-accent transition-colors"
                  aria-label="Next month"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>

              {/* Calendar */}
              <MiniCalendar
                month={calendarMonth}
                selectedRange={value}
                onDateClick={handleDateClick}
                minDate={minDate}
                maxDate={maxDate}
                hoveredDate={hoveredDate}
                onDateHover={setHoveredDate}
                selectionMode={selectionMode}
              />

              {/* Date inputs */}
              <div className="mt-3 pt-3 border-t space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label htmlFor="drp-start" className="text-xs">
                      Start
                    </Label>
                    <input
                      id="drp-start"
                      type="date"
                      value={startInput}
                      onChange={handleStartInputChange}
                      min={minDate ? formatDateForInput(minDate) : undefined}
                      max={maxDate ? formatDateForInput(maxDate) : undefined}
                      className={cn(
                        "flex h-8 w-full rounded-md border border-input bg-background px-2 py-1 text-xs",
                        "shadow-sm transition-colors placeholder:text-muted-foreground",
                        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      )}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="drp-end" className="text-xs">
                      End
                    </Label>
                    <input
                      id="drp-end"
                      type="date"
                      value={endInput}
                      onChange={handleEndInputChange}
                      min={startInput || (minDate ? formatDateForInput(minDate) : undefined)}
                      max={maxDate ? formatDateForInput(maxDate) : undefined}
                      className={cn(
                        "flex h-8 w-full rounded-md border border-input bg-background px-2 py-1 text-xs",
                        "shadow-sm transition-colors placeholder:text-muted-foreground",
                        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      )}
                    />
                  </div>
                </div>

                {/* Selection hint */}
                {selectionMode === "end" && value.start && !value.end && (
                  <p className="text-xs text-muted-foreground text-center">
                    Click another date to complete the range
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer with clear button */}
        {showClear && hasSelection && (
          <div className="border-t px-3 py-2 flex justify-end">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              className="h-7 gap-1.5 text-xs"
            >
              <X className="h-3 w-3" />
              Clear
            </Button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

// =============================================================================
// Convenience Variants
// =============================================================================

/**
 * SimpleDateRangePicker - Presets only, no calendar.
 */
export function SimpleDateRangePicker(
  props: Omit<DateRangePickerProps, "showCalendar" | "showPresets">
) {
  return (
    <DateRangePicker
      {...props}
      showPresets
      showCalendar={false}
      enabledPresets={QUICK_PRESETS}
    />
  );
}

/**
 * CompactDateRangePicker - Smaller size with essential presets.
 */
export function CompactDateRangePicker(
  props: Omit<DateRangePickerProps, "size" | "enabledPresets">
) {
  return (
    <DateRangePicker
      {...props}
      size="sm"
      enabledPresets={QUICK_PRESETS}
    />
  );
}

/**
 * CalendarOnlyDateRangePicker - Calendar without presets.
 */
export function CalendarOnlyDateRangePicker(
  props: Omit<DateRangePickerProps, "showPresets" | "showCalendar">
) {
  return (
    <DateRangePicker
      {...props}
      showPresets={false}
      showCalendar
    />
  );
}

/**
 * AnalyticsDateRangePicker - Optimized for analytics dashboards.
 * Shows commonly used presets for data analysis.
 */
export function AnalyticsDateRangePicker(
  props: Omit<DateRangePickerProps, "enabledPresets" | "showPresets">
) {
  return (
    <DateRangePicker
      {...props}
      showPresets
      showCalendar
      enabledPresets={[
        "today",
        "yesterday",
        "last_7_days",
        "last_30_days",
        "last_90_days",
        "this_month",
        "last_month",
        "all_time",
      ]}
    />
  );
}

// =============================================================================
// State Components
// =============================================================================

/**
 * Loading skeleton for DateRangePicker.
 */
export function DateRangePickerSkeleton({
  size = "md",
  className,
}: {
  size?: DateRangePickerSize;
  className?: string;
}) {
  const sizeClasses = {
    sm: "h-7 w-28",
    md: "h-8 w-36",
    lg: "h-10 w-44",
  };

  return (
    <Skeleton className={cn("rounded-md", sizeClasses[size], className)} />
  );
}

/**
 * Empty/disabled state for DateRangePicker.
 */
export function DateRangePickerEmpty({
  message = "No date range available",
  className,
}: {
  message?: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 text-muted-foreground text-sm",
        className
      )}
    >
      <CalendarClock className="h-4 w-4" />
      <span>{message}</span>
    </div>
  );
}

// =============================================================================
// Export Type-Only Exports
// =============================================================================

// Note: PresetConfig is already exported where it's defined above
export default DateRangePicker;
