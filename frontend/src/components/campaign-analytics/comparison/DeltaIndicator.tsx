/**
 * DeltaIndicator Component
 *
 * A small, reusable component showing the delta (difference) between values,
 * typically used for comparing campaigns. Displays color-coded positive/negative
 * changes with trend icons and supports both percentage and absolute modes.
 *
 * @example
 * ```tsx
 * // Percentage mode (default)
 * <DeltaIndicator value={15.5} />           // Shows: +15.5%
 * <DeltaIndicator value={-8.2} />           // Shows: -8.2%
 *
 * // Absolute mode
 * <DeltaIndicator value={150} mode="absolute" />        // Shows: +150
 * <DeltaIndicator value={-50} mode="absolute" unit="ms" /> // Shows: -50ms
 *
 * // With direction context
 * <DeltaIndicator value={-10} direction="lower_is_better" /> // Shows green (good)
 *
 * // With baseline comparison
 * <DeltaIndicator current={85} baseline={70} />  // Shows: +21.4%
 * ```
 */

"use client";

import * as React from "react";
import { useMemo } from "react";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowUp,
  ArrowDown,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// =============================================================================
// Types
// =============================================================================

/**
 * Display mode for delta values.
 */
export type DeltaMode = "percentage" | "absolute";

/**
 * Metric direction - determines color coding for positive/negative values.
 */
export type DeltaDirection = "higher_is_better" | "lower_is_better" | "neutral";

/**
 * Size variants for the delta indicator.
 */
export type DeltaSize = "xs" | "sm" | "md" | "lg";

/**
 * Icon style for trend display.
 */
export type DeltaIconStyle = "trending" | "arrow" | "none";

/**
 * Color preset for the indicator.
 */
export type DeltaColorPreset = "default" | "success" | "warning" | "error" | "info" | "muted";

/**
 * Props for the DeltaIndicator component.
 */
export interface DeltaIndicatorProps {
  /**
   * The delta value to display.
   * If using percentage mode with current/baseline, this is ignored.
   */
  value?: number | null;

  /**
   * Current value for automatic delta calculation.
   * Used with `baseline` to calculate percentage or absolute difference.
   */
  current?: number | null;

  /**
   * Baseline value for comparison.
   * Used with `current` to calculate the delta.
   */
  baseline?: number | null;

  /**
   * Display mode - percentage (default) or absolute.
   */
  mode?: DeltaMode;

  /**
   * Direction preference for color coding.
   * - "higher_is_better": Positive values are good (green)
   * - "lower_is_better": Negative values are good (green)
   * - "neutral": No color preference (muted colors)
   */
  direction?: DeltaDirection;

  /**
   * Unit suffix for absolute mode (e.g., "ms", "KB", "¢").
   */
  unit?: string;

  /**
   * Number of decimal places for the value.
   */
  precision?: number;

  /**
   * Size variant for the indicator.
   */
  size?: DeltaSize;

  /**
   * Icon style for trend display.
   */
  iconStyle?: DeltaIconStyle;

  /**
   * Show the icon alongside the value.
   */
  showIcon?: boolean;

  /**
   * Show positive sign (+) for positive values.
   */
  showSign?: boolean;

  /**
   * Show tooltip with additional context.
   */
  showTooltip?: boolean;

  /**
   * Custom tooltip content.
   */
  tooltipContent?: React.ReactNode;

  /**
   * Label for the metric (shown in tooltip).
   */
  label?: string;

  /**
   * Force a specific color preset (overrides direction-based coloring).
   */
  colorPreset?: DeltaColorPreset;

  /**
   * Render as inline element (span) instead of div.
   */
  inline?: boolean;

  /**
   * Additional CSS classes.
   */
  className?: string;

  /**
   * ARIA label for accessibility.
   */
  ariaLabel?: string;
}

// =============================================================================
// Constants
// =============================================================================

/**
 * Threshold for considering a delta as "no change" (within this range shows neutral icon).
 */
const NEUTRAL_THRESHOLD = 0.5;

/**
 * Size configuration mapping.
 */
const SIZE_CONFIG: Record<DeltaSize, { text: string; icon: string; gap: string }> = {
  xs: { text: "text-[10px]", icon: "h-2.5 w-2.5", gap: "gap-0.5" },
  sm: { text: "text-xs", icon: "h-3 w-3", gap: "gap-0.5" },
  md: { text: "text-sm", icon: "h-4 w-4", gap: "gap-1" },
  lg: { text: "text-base", icon: "h-5 w-5", gap: "gap-1.5" },
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Calculate delta value from current and baseline.
 */
function calculateDelta(
  current: number | null | undefined,
  baseline: number | null | undefined,
  mode: DeltaMode
): number | null {
  if (
    current === null ||
    current === undefined ||
    baseline === null ||
    baseline === undefined
  ) {
    return null;
  }

  if (mode === "percentage") {
    // Avoid division by zero
    if (baseline === 0) {
      // If baseline is 0 and current is not, return 100% increase or null
      return current === 0 ? 0 : current > 0 ? 100 : -100;
    }
    return ((current - baseline) / Math.abs(baseline)) * 100;
  }

  // Absolute mode
  return current - baseline;
}

/**
 * Get trend icon based on delta value and icon style.
 */
function getTrendIcon(
  delta: number,
  iconStyle: DeltaIconStyle,
  neutralThreshold: number = NEUTRAL_THRESHOLD
): LucideIcon | null {
  if (iconStyle === "none") return null;

  if (Math.abs(delta) < neutralThreshold) {
    return Minus;
  }

  if (iconStyle === "arrow") {
    return delta > 0 ? ArrowUp : ArrowDown;
  }

  // Default: trending icons
  return delta > 0 ? TrendingUp : TrendingDown;
}

/**
 * Get color classes based on delta value and direction preference.
 */
function getDeltaColorClasses(
  delta: number,
  direction: DeltaDirection,
  colorPreset?: DeltaColorPreset
): string {
  // Use preset if specified
  if (colorPreset) {
    switch (colorPreset) {
      case "success":
        return "text-green-600 dark:text-green-400";
      case "warning":
        return "text-amber-600 dark:text-amber-400";
      case "error":
        return "text-red-600 dark:text-red-400";
      case "info":
        return "text-blue-600 dark:text-blue-400";
      case "muted":
        return "text-muted-foreground";
      default:
        // Fall through to direction-based logic
        break;
    }
  }

  // Neutral direction = muted colors
  if (direction === "neutral") {
    return "text-muted-foreground";
  }

  // No change = muted
  if (Math.abs(delta) < NEUTRAL_THRESHOLD) {
    return "text-muted-foreground";
  }

  const isPositive = delta > 0;
  const isGood =
    (direction === "higher_is_better" && isPositive) ||
    (direction === "lower_is_better" && !isPositive);

  return isGood
    ? "text-green-600 dark:text-green-400"
    : "text-red-600 dark:text-red-400";
}

/**
 * Format delta value for display.
 */
function formatDeltaValue(
  delta: number,
  mode: DeltaMode,
  precision: number,
  unit?: string,
  showSign: boolean = true
): string {
  const sign = showSign && delta > 0 ? "+" : "";
  const value = delta.toFixed(precision);

  if (mode === "percentage") {
    return `${sign}${value}%`;
  }

  // Absolute mode
  const suffix = unit ? ` ${unit}` : "";
  return `${sign}${value}${suffix}`;
}

/**
 * Generate accessibility label for the delta.
 */
function generateAriaLabel(
  delta: number,
  mode: DeltaMode,
  direction: DeltaDirection,
  label?: string
): string {
  const prefix = label ? `${label}: ` : "";
  const sign = delta > 0 ? "increased" : delta < 0 ? "decreased" : "unchanged";
  const value = Math.abs(delta).toFixed(1);
  const unit = mode === "percentage" ? "percent" : "";

  let quality = "";
  if (direction !== "neutral") {
    const isGood =
      (direction === "higher_is_better" && delta > 0) ||
      (direction === "lower_is_better" && delta < 0);
    quality = isGood ? " (improvement)" : " (regression)";
  }

  return `${prefix}${sign} by ${value} ${unit}${quality}`.trim();
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * DeltaIndicator Component
 *
 * A small, reusable component for displaying the difference between two values
 * with color-coded positive/negative indicators and trend icons.
 *
 * @accessibility
 * - Includes ARIA labels describing the change
 * - Color is not the only indicator (uses icons and text)
 * - Screen reader friendly with meaningful descriptions
 */
export function DeltaIndicator({
  value,
  current,
  baseline,
  mode = "percentage",
  direction = "higher_is_better",
  unit,
  precision = 1,
  size = "sm",
  iconStyle = "trending",
  showIcon = true,
  showSign = true,
  showTooltip = false,
  tooltipContent,
  label,
  colorPreset,
  inline = false,
  className,
  ariaLabel,
}: DeltaIndicatorProps) {
  // Calculate delta value
  const delta = useMemo(() => {
    // If value is directly provided, use it
    if (value !== undefined && value !== null) {
      return value;
    }
    // Otherwise calculate from current and baseline
    return calculateDelta(current, baseline, mode);
  }, [value, current, baseline, mode]);

  // If no valid delta, render nothing
  if (delta === null) {
    return null;
  }

  // Get configuration
  const sizeConfig = SIZE_CONFIG[size];
  const colorClasses = getDeltaColorClasses(delta, direction, colorPreset);
  const TrendIcon = getTrendIcon(delta, iconStyle);
  const formattedValue = formatDeltaValue(delta, mode, precision, unit, showSign);
  const accessibilityLabel = ariaLabel || generateAriaLabel(delta, mode, direction, label);

  // Build the indicator element
  const Wrapper = inline ? "span" : "div";

  const indicatorContent = (
    <Wrapper
      className={cn(
        "flex items-center",
        sizeConfig.text,
        sizeConfig.gap,
        colorClasses,
        "font-medium tabular-nums",
        inline && "inline-flex",
        className
      )}
      role="status"
      aria-label={accessibilityLabel}
    >
      {showIcon && TrendIcon && (
        <TrendIcon className={cn(sizeConfig.icon, "flex-shrink-0")} aria-hidden="true" />
      )}
      <span>{formattedValue}</span>
    </Wrapper>
  );

  // Wrap in tooltip if enabled
  if (showTooltip) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            {indicatorContent}
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-xs">
            {tooltipContent || (
              <div className="text-sm">
                {label && <p className="font-medium">{label}</p>}
                <p className="text-muted-foreground">
                  {delta > 0 ? "Increased" : delta < 0 ? "Decreased" : "No change"} by{" "}
                  <span className="font-mono">{formatDeltaValue(Math.abs(delta), mode, precision, unit, false)}</span>
                  {direction !== "neutral" && (
                    <span className={cn(
                      "ml-1",
                      getDeltaColorClasses(delta, direction, colorPreset)
                    )}>
                      {(direction === "higher_is_better" && delta > 0) ||
                       (direction === "lower_is_better" && delta < 0)
                        ? "(better)"
                        : "(worse)"}
                    </span>
                  )}
                </p>
              </div>
            )}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return indicatorContent;
}

// =============================================================================
// Convenience Variants
// =============================================================================

/**
 * Percentage delta indicator (default mode).
 */
export function PercentageDeltaIndicator(
  props: Omit<DeltaIndicatorProps, "mode">
) {
  return <DeltaIndicator {...props} mode="percentage" />;
}

/**
 * Absolute delta indicator.
 */
export function AbsoluteDeltaIndicator(
  props: Omit<DeltaIndicatorProps, "mode">
) {
  return <DeltaIndicator {...props} mode="absolute" />;
}

/**
 * Success rate delta indicator (higher is better, percentage mode).
 */
export function SuccessRateDelta(
  props: Omit<DeltaIndicatorProps, "mode" | "direction" | "label">
) {
  return (
    <DeltaIndicator
      {...props}
      mode="percentage"
      direction="higher_is_better"
      label="Success Rate"
    />
  );
}

/**
 * Latency delta indicator (lower is better, absolute mode).
 */
export function LatencyDelta({
  unit = "ms",
  ...props
}: Omit<DeltaIndicatorProps, "mode" | "direction" | "label">) {
  return (
    <DeltaIndicator
      {...props}
      mode="absolute"
      direction="lower_is_better"
      unit={unit}
      label="Latency"
    />
  );
}

/**
 * Cost delta indicator (lower is better, currency-aware).
 */
export function CostDelta({
  unit = "¢",
  precision = 2,
  ...props
}: Omit<DeltaIndicatorProps, "mode" | "direction" | "label">) {
  return (
    <DeltaIndicator
      {...props}
      mode="absolute"
      direction="lower_is_better"
      unit={unit}
      precision={precision}
      label="Cost"
    />
  );
}

/**
 * Token usage delta indicator (neutral direction, absolute mode).
 */
export function TokenDelta({
  unit = "tokens",
  precision = 0,
  ...props
}: Omit<DeltaIndicatorProps, "mode" | "label">) {
  return (
    <DeltaIndicator
      {...props}
      mode="absolute"
      unit={unit}
      precision={precision}
      label="Token Usage"
    />
  );
}

/**
 * Compact inline delta indicator (smaller size, inline display).
 */
export function InlineDelta(
  props: Omit<DeltaIndicatorProps, "inline" | "size">
) {
  return <DeltaIndicator {...props} inline size="xs" />;
}

/**
 * Large delta indicator for prominent display.
 */
export function LargeDelta(
  props: Omit<DeltaIndicatorProps, "size" | "showTooltip">
) {
  return <DeltaIndicator {...props} size="lg" showTooltip />;
}

// =============================================================================
// Composite Components
// =============================================================================

/**
 * Props for DeltaComparison component.
 */
export interface DeltaComparisonProps {
  /** Current campaign value */
  currentValue: number | null | undefined;
  /** Current campaign label */
  currentLabel?: string;
  /** Baseline campaign value */
  baselineValue: number | null | undefined;
  /** Baseline campaign label */
  baselineLabel?: string;
  /** Mode for delta calculation */
  mode?: DeltaMode;
  /** Direction preference */
  direction?: DeltaDirection;
  /** Unit for absolute mode */
  unit?: string;
  /** Number of decimal places */
  precision?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * DeltaComparison Component
 *
 * Shows a comparison between two values with the delta displayed prominently.
 * Useful for campaign comparison panels.
 *
 * @example
 * ```tsx
 * <DeltaComparison
 *   currentValue={85}
 *   currentLabel="Campaign A"
 *   baselineValue={70}
 *   baselineLabel="Campaign B"
 *   direction="higher_is_better"
 * />
 * ```
 */
export function DeltaComparison({
  currentValue,
  currentLabel = "Current",
  baselineValue,
  baselineLabel = "Baseline",
  mode = "percentage",
  direction = "higher_is_better",
  unit,
  precision = 1,
  className,
}: DeltaComparisonProps) {
  const delta = calculateDelta(currentValue, baselineValue, mode);

  if (delta === null) {
    return (
      <div className={cn("text-sm text-muted-foreground", className)}>
        Cannot compare values
      </div>
    );
  }

  const colorClasses = getDeltaColorClasses(delta, direction);
  const formattedCurrent = currentValue?.toFixed(precision) ?? "—";
  const formattedBaseline = baselineValue?.toFixed(precision) ?? "—";

  return (
    <div className={cn("flex flex-col gap-1", className)}>
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{currentLabel}</span>
        <span className="font-mono font-medium">{formattedCurrent}</span>
      </div>
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{baselineLabel}</span>
        <span className="font-mono">{formattedBaseline}</span>
      </div>
      <div className="border-t pt-1 mt-1">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Difference</span>
          <DeltaIndicator
            value={delta}
            mode={mode}
            direction={direction}
            unit={unit}
            precision={precision}
            size="sm"
            showTooltip
          />
        </div>
      </div>
    </div>
  );
}

/**
 * Props for DeltaBadge component.
 */
export interface DeltaBadgeProps extends Omit<DeltaIndicatorProps, "showIcon" | "iconStyle"> {
  /** Badge variant style */
  variant?: "subtle" | "outline" | "solid";
}

/**
 * DeltaBadge Component
 *
 * A badge-style delta indicator with background color.
 * Good for use in tables or lists.
 *
 * @example
 * ```tsx
 * <DeltaBadge value={15.5} variant="subtle" />
 * ```
 */
export function DeltaBadge({
  value,
  current,
  baseline,
  mode = "percentage",
  direction = "higher_is_better",
  unit,
  precision = 1,
  size = "sm",
  showSign = true,
  label,
  variant = "subtle",
  className,
  ...props
}: DeltaBadgeProps) {
  // Calculate delta value
  const delta = useMemo(() => {
    if (value !== undefined && value !== null) {
      return value;
    }
    return calculateDelta(current, baseline, mode);
  }, [value, current, baseline, mode]);

  if (delta === null) {
    return null;
  }

  const sizeConfig = SIZE_CONFIG[size];
  const isPositive = delta > 0;
  const isNeutral = direction === "neutral" || Math.abs(delta) < NEUTRAL_THRESHOLD;
  const isGood = !isNeutral && (
    (direction === "higher_is_better" && isPositive) ||
    (direction === "lower_is_better" && !isPositive)
  );

  const formattedValue = formatDeltaValue(delta, mode, precision, unit, showSign);

  // Get variant-specific classes
  const getVariantClasses = () => {
    if (isNeutral) {
      switch (variant) {
        case "solid":
          return "bg-muted text-muted-foreground";
        case "outline":
          return "border border-border text-muted-foreground";
        default:
          return "bg-muted/50 text-muted-foreground";
      }
    }

    if (isGood) {
      switch (variant) {
        case "solid":
          return "bg-green-600 text-white dark:bg-green-500";
        case "outline":
          return "border border-green-500/50 text-green-600 dark:text-green-400";
        default:
          return "bg-green-500/10 text-green-600 dark:text-green-400";
      }
    }

    // Not good (regression)
    switch (variant) {
      case "solid":
        return "bg-red-600 text-white dark:bg-red-500";
      case "outline":
        return "border border-red-500/50 text-red-600 dark:text-red-400";
      default:
        return "bg-red-500/10 text-red-600 dark:text-red-400";
    }
  };

  const TrendIcon = getTrendIcon(delta, "arrow");

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-medium tabular-nums",
        sizeConfig.text,
        getVariantClasses(),
        className
      )}
      role="status"
      aria-label={generateAriaLabel(delta, mode, direction, label)}
    >
      {TrendIcon && <TrendIcon className={cn(sizeConfig.icon, "flex-shrink-0")} aria-hidden="true" />}
      {formattedValue}
    </span>
  );
}

// =============================================================================
// Export Default
// =============================================================================

export default DeltaIndicator;
