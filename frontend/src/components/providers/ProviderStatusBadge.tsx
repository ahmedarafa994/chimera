/**
 * ProviderStatusBadge Component
 *
 * Visual indicator for provider health status with tooltip showing details
 * like last checked time, error rate, and latency.
 *
 * @module components/providers/ProviderStatusBadge
 */

"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { ProviderStatus, ProviderStatusBadgeProps } from "@/types/unified-providers";
import { STATUS_CONFIG } from "@/types/unified-providers";

// =============================================================================
// Status Icons
// =============================================================================

interface StatusIconProps {
  status: ProviderStatus;
  className?: string;
}

function StatusIcon({ status, className }: StatusIconProps) {
  const baseClasses = "h-3 w-3 rounded-full";

  switch (status) {
    case "healthy":
      return (
        <span
          className={cn(baseClasses, "bg-green-500 animate-pulse", className)}
          aria-hidden="true"
        />
      );
    case "degraded":
      return (
        <span
          className={cn(baseClasses, "bg-yellow-500", className)}
          aria-hidden="true"
        />
      );
    case "unhealthy":
      return (
        <span
          className={cn(baseClasses, "bg-red-500", className)}
          aria-hidden="true"
        />
      );
    case "unknown":
    default:
      return (
        <span
          className={cn(baseClasses, "bg-gray-400", className)}
          aria-hidden="true"
        />
      );
  }
}

// =============================================================================
// Helper Functions
// =============================================================================

function formatLastChecked(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSeconds = Math.floor(diffMs / 1000);
    const diffMinutes = Math.floor(diffSeconds / 60);
    const diffHours = Math.floor(diffMinutes / 60);

    if (diffSeconds < 60) {
      return "Just now";
    } else if (diffMinutes < 60) {
      return `${diffMinutes}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  } catch {
    return "Unknown";
  }
}

function formatLatency(latencyMs: number): string {
  if (latencyMs < 1000) {
    return `${Math.round(latencyMs)}ms`;
  }
  return `${(latencyMs / 1000).toFixed(1)}s`;
}

function formatErrorRate(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

function getHealthScoreColor(score: number): string {
  if (score >= 0.9) return "text-green-500";
  if (score >= 0.7) return "text-yellow-500";
  if (score >= 0.5) return "text-orange-500";
  return "text-red-500";
}

// =============================================================================
// Component
// =============================================================================

export function ProviderStatusBadge({
  status,
  healthScore,
  compact = false,
  lastChecked,
  errorRate,
  latency,
  className,
}: ProviderStatusBadgeProps) {
  const config = STATUS_CONFIG[status];

  // Compact mode: just show the status dot
  if (compact) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className={cn("inline-flex items-center", className)}>
              <StatusIcon status={status} />
              <span className="sr-only">{config.label}</span>
            </span>
          </TooltipTrigger>
          <TooltipContent side="top" className="text-xs">
            <p className="font-medium">{config.label}</p>
            <p className="text-muted-foreground">{config.description}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  // Full mode: show badge with details
  const hasDetails = lastChecked || errorRate !== undefined || latency !== undefined || healthScore !== undefined;

  const badgeContent = (
    <Badge
      variant={config.color}
      className={cn(
        "inline-flex items-center gap-1.5 font-medium",
        className
      )}
    >
      <StatusIcon status={status} />
      <span>{config.label}</span>
    </Badge>
  );

  if (!hasDetails) {
    return badgeContent;
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          {badgeContent}
        </TooltipTrigger>
        <TooltipContent side="top" className="w-48 p-3">
          <div className="space-y-2">
            <div>
              <p className="font-semibold text-sm">{config.label}</p>
              <p className="text-xs text-muted-foreground">{config.description}</p>
            </div>

            <div className="border-t pt-2 space-y-1.5">
              {healthScore !== undefined && (
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Health Score</span>
                  <span className={cn("font-medium", getHealthScoreColor(healthScore))}>
                    {Math.round(healthScore * 100)}%
                  </span>
                </div>
              )}

              {lastChecked && (
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Last Checked</span>
                  <span className="font-medium">{formatLastChecked(lastChecked)}</span>
                </div>
              )}

              {latency !== undefined && (
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Latency</span>
                  <span className="font-medium">{formatLatency(latency)}</span>
                </div>
              )}

              {errorRate !== undefined && (
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Error Rate</span>
                  <span className={cn(
                    "font-medium",
                    errorRate > 0.1 ? "text-red-500" : errorRate > 0.05 ? "text-yellow-500" : "text-green-500"
                  )}>
                    {formatErrorRate(errorRate)}
                  </span>
                </div>
              )}
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// Additional Status Utilities
// =============================================================================

/**
 * Get the appropriate status for a given health score
 */
export function getStatusFromHealthScore(score: number): ProviderStatus {
  if (score >= 0.9) return "healthy";
  if (score >= 0.5) return "degraded";
  if (score > 0) return "unhealthy";
  return "unknown";
}

/**
 * Inline status indicator (just the dot, no badge styling)
 */
export function ProviderStatusIndicator({
  status,
  className,
}: {
  status: ProviderStatus;
  className?: string;
}) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={cn("inline-flex items-center", className)}>
            <StatusIcon status={status} />
            <span className="sr-only">{STATUS_CONFIG[status].label}</span>
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" className="text-xs">
          {STATUS_CONFIG[status].label}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export default ProviderStatusBadge;
