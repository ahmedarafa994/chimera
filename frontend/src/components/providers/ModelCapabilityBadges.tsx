/**
 * ModelCapabilityBadges Component
 *
 * Displays model capabilities as color-coded badges with icons.
 * Supports collapsible display when there are many capabilities.
 *
 * @module components/providers/ModelCapabilityBadges
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
import type { ModelCapability, ModelCapabilityBadgesProps } from "@/types/unified-providers";
import { CAPABILITY_CONFIG } from "@/types/unified-providers";

// =============================================================================
// Capability Icons (using simple SVG icons)
// =============================================================================

interface IconProps {
  className?: string;
}

const Icons: Record<string, React.FC<IconProps>> = {
  zap: ({ className }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  ),
  eye: ({ className }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ),
  code: ({ className }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  ),
  braces: ({ className }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1" />
      <path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1" />
    </svg>
  ),
  terminal: ({ className }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  ),
  brain: ({ className }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-1.54Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-1.54Z" />
    </svg>
  ),
  layers: ({ className }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  ),
  wrench: ({ className }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
  ),
};

// =============================================================================
// Badge Color Mapping
// =============================================================================

type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

function getCapabilityVariant(color: string): BadgeVariant {
  switch (color) {
    case "primary":
    case "success":
      return "default";
    case "secondary":
    case "warning":
      return "secondary";
    case "destructive":
      return "destructive";
    default:
      return "outline";
  }
}

function getCapabilityColorClasses(color: string): string {
  switch (color) {
    case "primary":
      return "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20 hover:bg-blue-500/20";
    case "secondary":
      return "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20 hover:bg-purple-500/20";
    case "success":
      return "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20 hover:bg-green-500/20";
    case "warning":
      return "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20 hover:bg-amber-500/20";
    case "destructive":
      return "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20 hover:bg-red-500/20";
    default:
      return "bg-gray-500/10 text-gray-600 dark:text-gray-400 border-gray-500/20 hover:bg-gray-500/20";
  }
}

// =============================================================================
// Single Capability Badge
// =============================================================================

interface CapabilityBadgeProps {
  capability: ModelCapability;
  size?: "sm" | "md";
  showIcon?: boolean;
  className?: string;
}

export function CapabilityBadge({
  capability,
  size = "md",
  showIcon = true,
  className,
}: CapabilityBadgeProps) {
  const config = CAPABILITY_CONFIG[capability];
  if (!config) {
    return null;
  }

  const IconComponent = Icons[config.icon];
  const iconSize = size === "sm" ? "h-2.5 w-2.5" : "h-3 w-3";
  const badgeSize = size === "sm" ? "text-[10px] px-1.5 py-0" : "text-xs px-2 py-0.5";

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant={getCapabilityVariant(config.color)}
            className={cn(
              "inline-flex items-center gap-1 font-medium border",
              getCapabilityColorClasses(config.color),
              badgeSize,
              className
            )}
          >
            {showIcon && IconComponent && (
              <IconComponent className={iconSize} />
            )}
            <span>{config.label}</span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="text-xs max-w-[200px]">
          {config.description}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// Multiple Capability Badges with Collapse
// =============================================================================

export function ModelCapabilityBadges({
  capabilities,
  maxVisible = 3,
  size = "md",
  className,
}: ModelCapabilityBadgesProps) {
  const [expanded, setExpanded] = React.useState(false);

  if (!capabilities || capabilities.length === 0) {
    return null;
  }

  // Filter to valid capabilities
  const validCapabilities = capabilities.filter(
    (cap) => cap in CAPABILITY_CONFIG
  );

  if (validCapabilities.length === 0) {
    return null;
  }

  const visibleCapabilities = expanded
    ? validCapabilities
    : validCapabilities.slice(0, maxVisible);

  const hiddenCount = validCapabilities.length - maxVisible;
  const hasHidden = hiddenCount > 0 && !expanded;

  return (
    <div className={cn("flex flex-wrap items-center gap-1", className)}>
      {visibleCapabilities.map((capability) => (
        <CapabilityBadge
          key={capability}
          capability={capability}
          size={size}
        />
      ))}

      {hasHidden && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="outline"
                className={cn(
                  "cursor-pointer hover:bg-accent",
                  size === "sm" ? "text-[10px] px-1.5 py-0" : "text-xs px-2 py-0.5"
                )}
                onClick={() => setExpanded(true)}
              >
                +{hiddenCount} more
              </Badge>
            </TooltipTrigger>
            <TooltipContent side="top" className="text-xs">
              <div className="space-y-1">
                {validCapabilities.slice(maxVisible).map((cap) => (
                  <div key={cap} className="flex items-center gap-1">
                    <span>{CAPABILITY_CONFIG[cap].label}</span>
                  </div>
                ))}
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}

      {expanded && hiddenCount > 0 && (
        <Badge
          variant="outline"
          className={cn(
            "cursor-pointer hover:bg-accent",
            size === "sm" ? "text-[10px] px-1.5 py-0" : "text-xs px-2 py-0.5"
          )}
          onClick={() => setExpanded(false)}
        >
          Show less
        </Badge>
      )}
    </div>
  );
}

// =============================================================================
// Compact Capability Icons (for tight spaces)
// =============================================================================

interface CompactCapabilityIconsProps {
  capabilities: ModelCapability[];
  maxVisible?: number;
  className?: string;
}

export function CompactCapabilityIcons({
  capabilities,
  maxVisible = 4,
  className,
}: CompactCapabilityIconsProps) {
  if (!capabilities || capabilities.length === 0) {
    return null;
  }

  const validCapabilities = capabilities.filter(
    (cap) => cap in CAPABILITY_CONFIG
  );

  const visibleCapabilities = validCapabilities.slice(0, maxVisible);
  const hiddenCount = validCapabilities.length - maxVisible;

  return (
    <TooltipProvider>
      <div className={cn("flex items-center gap-0.5", className)}>
        {visibleCapabilities.map((capability) => {
          const config = CAPABILITY_CONFIG[capability];
          const IconComponent = Icons[config.icon];

          return (
            <Tooltip key={capability}>
              <TooltipTrigger asChild>
                <span
                  className={cn(
                    "inline-flex items-center justify-center h-5 w-5 rounded",
                    getCapabilityColorClasses(config.color)
                  )}
                >
                  {IconComponent && <IconComponent className="h-3 w-3" />}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" className="text-xs">
                <p className="font-medium">{config.label}</p>
                <p className="text-muted-foreground">{config.description}</p>
              </TooltipContent>
            </Tooltip>
          );
        })}

        {hiddenCount > 0 && (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="inline-flex items-center justify-center h-5 w-5 rounded bg-muted text-muted-foreground text-[10px] font-medium">
                +{hiddenCount}
              </span>
            </TooltipTrigger>
            <TooltipContent side="top" className="text-xs">
              <div className="space-y-0.5">
                {validCapabilities.slice(maxVisible).map((cap) => (
                  <div key={cap}>{CAPABILITY_CONFIG[cap].label}</div>
                ))}
              </div>
            </TooltipContent>
          </Tooltip>
        )}
      </div>
    </TooltipProvider>
  );
}

// =============================================================================
// Capability Check (utility for checking if model has capability)
// =============================================================================

export function hasCapability(
  capabilities: ModelCapability[] | undefined,
  capability: ModelCapability
): boolean {
  return capabilities?.includes(capability) ?? false;
}

export function hasAnyCapability(
  capabilities: ModelCapability[] | undefined,
  check: ModelCapability[]
): boolean {
  return check.some((cap) => capabilities?.includes(cap));
}

export function hasAllCapabilities(
  capabilities: ModelCapability[] | undefined,
  check: ModelCapability[]
): boolean {
  return check.every((cap) => capabilities?.includes(cap));
}

export default ModelCapabilityBadges;
