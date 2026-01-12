"use client";

import * as React from "react";
import { useMemo } from "react";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";

// =============================================================================
// Types
// =============================================================================

export interface UptimeIndicatorProps {
  uptimePercent: number;
  size?: "xs" | "sm" | "md" | "lg";
  showLabel?: boolean;
  showPercentage?: boolean;
  showIcon?: boolean;
  label?: string;
  className?: string;
}

export interface UptimeBadgesProps {
  lastHour?: number;
  last24Hours?: number;
  last7Days?: number;
  last30Days?: number;
  className?: string;
}

// =============================================================================
// Configuration
// =============================================================================

const sizeConfig = {
  xs: {
    height: "h-1",
    iconSize: "h-3 w-3",
    textSize: "text-xs",
    percentageSize: "text-xs",
  },
  sm: {
    height: "h-1.5",
    iconSize: "h-4 w-4",
    textSize: "text-sm",
    percentageSize: "text-sm",
  },
  md: {
    height: "h-2",
    iconSize: "h-5 w-5",
    textSize: "text-base",
    percentageSize: "text-lg",
  },
  lg: {
    height: "h-3",
    iconSize: "h-6 w-6",
    textSize: "text-lg",
    percentageSize: "text-2xl",
  },
};

// =============================================================================
// Helper Functions
// =============================================================================

function getUptimeConfig(percent: number) {
  if (percent >= 99.5) {
    return {
      icon: CheckCircle2,
      color: "text-emerald-500",
      bgColor: "bg-emerald-500/10",
      progressColor: "bg-emerald-500",
      label: "Excellent",
      description: "Service is performing optimally",
    };
  }
  if (percent >= 99) {
    return {
      icon: CheckCircle2,
      color: "text-green-500",
      bgColor: "bg-green-500/10",
      progressColor: "bg-green-500",
      label: "Good",
      description: "Service is performing well",
    };
  }
  if (percent >= 95) {
    return {
      icon: AlertTriangle,
      color: "text-yellow-500",
      bgColor: "bg-yellow-500/10",
      progressColor: "bg-yellow-500",
      label: "Degraded",
      description: "Service is experiencing minor issues",
    };
  }
  if (percent >= 90) {
    return {
      icon: AlertTriangle,
      color: "text-orange-500",
      bgColor: "bg-orange-500/10",
      progressColor: "bg-orange-500",
      label: "Poor",
      description: "Service is experiencing significant issues",
    };
  }
  return {
    icon: XCircle,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    progressColor: "bg-red-500",
    label: "Critical",
    description: "Service is experiencing critical issues",
  };
}

function formatUptime(percent: number): string {
  if (percent === 100) return "100%";
  if (percent >= 99.99) return "99.99%";
  if (percent >= 99.9) return percent.toFixed(2) + "%";
  if (percent >= 99) return percent.toFixed(2) + "%";
  return percent.toFixed(1) + "%";
}

// =============================================================================
// Main Component
// =============================================================================

export function UptimeIndicator({
  uptimePercent,
  size = "md",
  showLabel = true,
  showPercentage = true,
  showIcon = true,
  label = "Uptime",
  className,
}: UptimeIndicatorProps) {
  const config = useMemo(() => getUptimeConfig(uptimePercent), [uptimePercent]);
  const sizeClasses = sizeConfig[size];
  const StatusIcon = config.icon;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={cn("space-y-1.5", className)}>
            {/* Header with label and percentage */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                {showIcon && (
                  <StatusIcon className={cn(sizeClasses.iconSize, config.color)} />
                )}
                {showLabel && (
                  <span className={cn(sizeClasses.textSize, "text-muted-foreground")}>
                    {label}
                  </span>
                )}
              </div>
              {showPercentage && (
                <span
                  className={cn(
                    sizeClasses.percentageSize,
                    "font-semibold",
                    config.color
                  )}
                >
                  {formatUptime(uptimePercent)}
                </span>
              )}
            </div>

            {/* Progress bar */}
            <div className={cn("relative w-full rounded-full bg-muted", sizeClasses.height)}>
              <div
                className={cn(
                  "absolute inset-y-0 left-0 rounded-full transition-all duration-500",
                  config.progressColor
                )}
                style={{ width: `${Math.min(uptimePercent, 100)}%` }}
              />
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent side="top">
          <div className="text-center">
            <p className="font-medium">{config.label}</p>
            <p className="text-xs text-muted-foreground">{config.description}</p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// Uptime Badges Component
// =============================================================================

interface UptimeBadgeProps {
  label: string;
  value?: number;
}

function UptimeBadge({ label, value }: UptimeBadgeProps) {
  if (value === undefined) return null;

  const config = getUptimeConfig(value);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn(
              "flex flex-col items-center p-2 rounded-lg border",
              config.bgColor,
              "border-transparent"
            )}
          >
            <span className="text-xs text-muted-foreground">{label}</span>
            <span className={cn("text-sm font-semibold", config.color)}>
              {formatUptime(value)}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {label}: {config.label}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function UptimeBadges({
  lastHour,
  last24Hours,
  last7Days,
  last30Days,
  className,
}: UptimeBadgesProps) {
  const hasData = lastHour !== undefined || last24Hours !== undefined ||
                  last7Days !== undefined || last30Days !== undefined;

  if (!hasData) {
    return (
      <div className={cn("text-center text-sm text-muted-foreground", className)}>
        No uptime data available
      </div>
    );
  }

  return (
    <div className={cn("grid grid-cols-2 sm:grid-cols-4 gap-2", className)}>
      <UptimeBadge label="Last Hour" value={lastHour} />
      <UptimeBadge label="24 Hours" value={last24Hours} />
      <UptimeBadge label="7 Days" value={last7Days} />
      <UptimeBadge label="30 Days" value={last30Days} />
    </div>
  );
}

// =============================================================================
// Circular Uptime Indicator
// =============================================================================

export interface CircularUptimeProps {
  uptimePercent: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
}

export function CircularUptime({
  uptimePercent,
  size = 80,
  strokeWidth = 6,
  className,
}: CircularUptimeProps) {
  const config = useMemo(() => getUptimeConfig(uptimePercent), [uptimePercent]);

  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (uptimePercent / 100) * circumference;

  const StatusIcon = config.icon;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn("relative inline-flex items-center justify-center", className)}
            style={{ width: size, height: size }}
          >
            {/* Background circle */}
            <svg
              className="absolute inset-0 transform -rotate-90"
              width={size}
              height={size}
            >
              <circle
                className="text-muted"
                strokeWidth={strokeWidth}
                stroke="currentColor"
                fill="transparent"
                r={radius}
                cx={size / 2}
                cy={size / 2}
              />
              <circle
                className={config.color}
                strokeWidth={strokeWidth}
                strokeDasharray={circumference}
                strokeDashoffset={offset}
                strokeLinecap="round"
                stroke="currentColor"
                fill="transparent"
                r={radius}
                cx={size / 2}
                cy={size / 2}
                style={{
                  transition: "stroke-dashoffset 0.5s ease-in-out",
                }}
              />
            </svg>

            {/* Center content */}
            <div className="flex flex-col items-center">
              <span className={cn("text-lg font-bold", config.color)}>
                {formatUptime(uptimePercent)}
              </span>
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <div className="text-center">
            <p className="font-medium">{config.label}</p>
            <p className="text-xs text-muted-foreground">{config.description}</p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// Uptime Status Dots (Mini visualization)
// =============================================================================

export interface UptimeDotsProps {
  /** Array of uptime values for each period (e.g., last 30 days) */
  history: Array<{ date: string; uptime: number }>;
  /** Number of dots to display */
  maxDots?: number;
  /** Size of each dot */
  dotSize?: "xs" | "sm" | "md";
  className?: string;
}

export function UptimeDots({
  history,
  maxDots = 30,
  dotSize = "sm",
  className,
}: UptimeDotsProps) {
  const dotSizeClass = {
    xs: "w-1 h-4",
    sm: "w-1.5 h-5",
    md: "w-2 h-6",
  }[dotSize];

  const displayHistory = history.slice(-maxDots);

  return (
    <div className={cn("flex items-end gap-0.5", className)}>
      {displayHistory.map((entry, index) => {
        const config = getUptimeConfig(entry.uptime);
        const height = Math.max(20, Math.min(100, entry.uptime));

        return (
          <TooltipProvider key={entry.date || index}>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "rounded-sm transition-all hover:opacity-80",
                    dotSizeClass,
                    config.progressColor
                  )}
                  style={{
                    height: `${height}%`,
                    minHeight: "4px",
                  }}
                />
              </TooltipTrigger>
              <TooltipContent>
                <p className="font-medium">{entry.date}</p>
                <p className="text-sm">{formatUptime(entry.uptime)}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      })}
    </div>
  );
}

export default UptimeIndicator;
