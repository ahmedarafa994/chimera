"use client";

/**
 * Provider Status Badge Component
 *
 * Visual indicator for provider health status with tooltip details.
 * Color coding: green (healthy), yellow (degraded), red (unavailable)
 */

import React from "react";
import { cn } from "@/lib/utils";

export interface ProviderStatusBadgeProps {
  status: "healthy" | "degraded" | "unavailable" | "unknown";
  latencyMs?: number | null;
  errorMessage?: string | null;
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const statusConfig = {
  healthy: {
    color: "bg-green-500",
    pulseColor: "bg-green-400",
    label: "Healthy",
    textColor: "text-green-600 dark:text-green-400",
  },
  degraded: {
    color: "bg-yellow-500",
    pulseColor: "bg-yellow-400",
    label: "Degraded",
    textColor: "text-yellow-600 dark:text-yellow-400",
  },
  unavailable: {
    color: "bg-red-500",
    pulseColor: "bg-red-400",
    label: "Unavailable",
    textColor: "text-red-600 dark:text-red-400",
  },
  unknown: {
    color: "bg-gray-400",
    pulseColor: "bg-gray-300",
    label: "Unknown",
    textColor: "text-gray-600 dark:text-gray-400",
  },
};

const sizeConfig = {
  sm: {
    dot: "h-2 w-2",
    text: "text-xs",
    gap: "gap-1",
  },
  md: {
    dot: "h-2.5 w-2.5",
    text: "text-sm",
    gap: "gap-1.5",
  },
  lg: {
    dot: "h-3 w-3",
    text: "text-base",
    gap: "gap-2",
  },
};

export function ProviderStatusBadge({
  status,
  latencyMs,
  errorMessage,
  showLabel = false,
  size = "md",
  className,
}: ProviderStatusBadgeProps) {
  const config = statusConfig[status];
  const sizes = sizeConfig[size];

  const tooltipContent = React.useMemo(() => {
    const parts = [config.label];
    if (latencyMs !== null && latencyMs !== undefined) {
      parts.push(`Latency: ${Math.round(latencyMs)}ms`);
    }
    if (errorMessage) {
      parts.push(`Error: ${errorMessage}`);
    }
    return parts.join(" | ");
  }, [config.label, latencyMs, errorMessage]);

  return (
    <div
      className={cn("inline-flex items-center", sizes.gap, className)}
      title={tooltipContent}
    >
      <span className="relative flex">
        <span
          className={cn(
            "absolute inline-flex h-full w-full rounded-full opacity-75",
            status === "healthy" && "animate-ping",
            config.pulseColor
          )}
          style={{ animationDuration: "2s" }}
        />
        <span
          className={cn("relative inline-flex rounded-full", sizes.dot, config.color)}
        />
      </span>
      {showLabel && (
        <span className={cn(sizes.text, config.textColor, "font-medium")}>
          {config.label}
        </span>
      )}
    </div>
  );
}

/**
 * Provider Card with Status
 */
export interface ProviderCardProps {
  provider: string;
  displayName: string;
  status: "healthy" | "degraded" | "unavailable" | "unknown";
  isSelected?: boolean;
  latencyMs?: number | null;
  modelCount?: number;
  onClick?: () => void;
  disabled?: boolean;
}

export function ProviderCard({
  provider,
  displayName,
  status,
  isSelected = false,
  latencyMs,
  modelCount,
  onClick,
  disabled = false,
}: ProviderCardProps) {
  const providerIcons: Record<string, string> = {
    // Gemini AI
    gemini: "✨",
    google: "✨",
    "gemini-cli": "✨",
    "gemini-openai": "✨",
    // Hybrid
    antigravity: "⚡",
    // Other providers
    deepseek: "🔍",
    openai: "🤖",
    anthropic: "🧠",
    kiro: "🧠",
    qwen: "💻",
    cursor: "🖱️",
  };

  const icon = providerIcons[provider.toLowerCase()] || "🔮";

  return (
    <button
      onClick={onClick}
      disabled={disabled || status === "unavailable"}
      className={cn(
        "relative flex flex-col items-start p-4 rounded-lg border-2 transition-all duration-200",
        "hover:shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2",
        isSelected
          ? "border-primary bg-primary/5 ring-2 ring-primary/20"
          : "border-border hover:border-primary/50",
        disabled && "opacity-50 cursor-not-allowed",
        status === "unavailable" && "opacity-60"
      )}
    >
      {/* Status indicator */}
      <div className="absolute top-2 right-2">
        <ProviderStatusBadge status={status} latencyMs={latencyMs} />
      </div>

      {/* Provider icon and name */}
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">{icon}</span>
        <span className="font-semibold text-foreground">{displayName}</span>
      </div>

      {/* Model count */}
      {modelCount !== undefined && (
        <span className="text-xs text-muted-foreground">
          {modelCount} model{modelCount !== 1 ? "s" : ""} available
        </span>
      )}

      {/* Latency info */}
      {latencyMs !== null && latencyMs !== undefined && (
        <span className="text-xs text-muted-foreground mt-1">
          ~{Math.round(latencyMs)}ms latency
        </span>
      )}

      {/* Selected indicator */}
      {isSelected && (
        <div className="absolute bottom-2 right-2">
          <svg
            className="w-5 h-5 text-primary"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      )}
    </button>
  );
}

export default ProviderStatusBadge;
