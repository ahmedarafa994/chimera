/**
 * ConnectionStatus Component
 *
 * Displays WebSocket connection status for Aegis Campaign Dashboard with:
 * - Visual status indicators for connected/disconnected/connecting/reconnecting states
 * - Animated pulse indicator for connected state
 * - Reconnect button when disconnected
 * - Last connected timestamp display
 * - Reconnection attempt counter
 *
 * Follows glass morphism styling pattern from existing components.
 */

"use client";

import { memo, useMemo } from "react";
import {
  Wifi,
  WifiOff,
  RefreshCw,
  AlertCircle,
  Clock,
  Activity,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { GlassCard } from "@/components/ui/glass-card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  WebSocketConnectionStatus,
  MAX_RECONNECT_ATTEMPTS,
} from "@/types/aegis-telemetry";

// ============================================================================
// Types
// ============================================================================

export interface ConnectionStatusProps {
  /** Current WebSocket connection status */
  status: WebSocketConnectionStatus;
  /** Callback to trigger manual reconnection */
  onReconnect: () => void;
  /** Timestamp of last successful connection (ISO string) */
  lastConnected?: string | null;
  /** Number of reconnection attempts made */
  reconnectAttempts?: number;
  /** Campaign ID being monitored */
  campaignId?: string;
  /** Whether to show compact version */
  compact?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Status Configuration
// ============================================================================

interface StatusConfig {
  label: string;
  icon: typeof Wifi;
  colorClass: string;
  bgClass: string;
  pulseClass?: string;
  description: string;
}

const STATUS_CONFIG: Record<WebSocketConnectionStatus, StatusConfig> = {
  connected: {
    label: "Connected",
    icon: Wifi,
    colorClass: "text-emerald-400",
    bgClass: "bg-emerald-500/20 border-emerald-500/30",
    pulseClass: "animate-pulse",
    description: "Real-time telemetry stream active",
  },
  connecting: {
    label: "Connecting",
    icon: RefreshCw,
    colorClass: "text-blue-400",
    bgClass: "bg-blue-500/20 border-blue-500/30",
    pulseClass: "animate-spin",
    description: "Establishing WebSocket connection...",
  },
  reconnecting: {
    label: "Reconnecting",
    icon: RefreshCw,
    colorClass: "text-amber-400",
    bgClass: "bg-amber-500/20 border-amber-500/30",
    pulseClass: "animate-spin",
    description: "Attempting to restore connection...",
  },
  disconnected: {
    label: "Disconnected",
    icon: WifiOff,
    colorClass: "text-gray-400",
    bgClass: "bg-gray-500/20 border-gray-500/30",
    description: "No active connection",
  },
  error: {
    label: "Error",
    icon: AlertCircle,
    colorClass: "text-red-400",
    bgClass: "bg-red-500/20 border-red-500/30",
    description: "Connection error occurred",
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format a timestamp as relative time (e.g., "2 minutes ago")
 */
function formatRelativeTime(timestamp: string | null | undefined): string {
  if (!timestamp) return "Never";

  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);

  if (diffSec < 10) return "Just now";
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHour < 24) return `${diffHour}h ago`;

  return date.toLocaleDateString();
}

/**
 * Format timestamp for tooltip
 */
function formatTimestamp(timestamp: string | null | undefined): string {
  if (!timestamp) return "Never connected";
  return new Date(timestamp).toLocaleString();
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Animated pulse indicator for connection status
 */
const PulseIndicator = memo(function PulseIndicator({
  status,
}: {
  status: WebSocketConnectionStatus;
}) {
  return (
    <div className="relative flex items-center justify-center">
      {/* Outer pulse ring for connected state */}
      {status === "connected" && (
        <span
          className={cn(
            "absolute inline-flex h-4 w-4 rounded-full opacity-75",
            "bg-emerald-500 animate-ping"
          )}
        />
      )}
      {/* Inner solid indicator */}
      <span
        className={cn(
          "relative inline-flex h-3 w-3 rounded-full",
          status === "connected" && "bg-emerald-500",
          status === "connecting" && "bg-blue-500",
          status === "reconnecting" && "bg-amber-500",
          status === "disconnected" && "bg-gray-500",
          status === "error" && "bg-red-500"
        )}
      />
    </div>
  );
});

/**
 * Connection status badge
 */
const StatusBadge = memo(function StatusBadge({
  status,
}: {
  status: WebSocketConnectionStatus;
}) {
  const config = STATUS_CONFIG[status];
  const Icon = config.icon;

  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1.5 px-2.5 py-1 font-medium transition-all duration-300",
        config.bgClass,
        config.colorClass
      )}
    >
      <Icon
        className={cn("h-3.5 w-3.5", config.pulseClass)}
        aria-hidden="true"
      />
      <span>{config.label}</span>
    </Badge>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * ConnectionStatus displays WebSocket connection status for the Aegis Dashboard.
 *
 * Features:
 * - Visual status indicators for all connection states
 * - Animated pulse indicator for active connections
 * - Reconnect button with attempt counter
 * - Last connected timestamp with relative time
 * - Responsive compact/full layouts
 */
export const ConnectionStatus = memo(function ConnectionStatus({
  status,
  onReconnect,
  lastConnected,
  reconnectAttempts = 0,
  campaignId,
  compact = false,
  className,
}: ConnectionStatusProps) {
  const config = STATUS_CONFIG[status];

  // Determine if reconnect should be shown/enabled
  const showReconnect = status === "disconnected" || status === "error";
  const isReconnecting = status === "reconnecting";
  const isConnecting = status === "connecting";

  // Memoized relative time
  const relativeTime = useMemo(
    () => formatRelativeTime(lastConnected),
    [lastConnected]
  );

  const formattedTimestamp = useMemo(
    () => formatTimestamp(lastConnected),
    [lastConnected]
  );

  // Compact version - just badge and reconnect button
  if (compact) {
    return (
      <div className={cn("flex items-center gap-2", className)}>
        <PulseIndicator status={status} />
        <StatusBadge status={status} />
        {showReconnect && (
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onReconnect}
            disabled={isReconnecting || isConnecting}
            className="h-7 w-7"
            aria-label="Reconnect to telemetry stream"
          >
            <RefreshCw
              className={cn(
                "h-3.5 w-3.5",
                (isReconnecting || isConnecting) && "animate-spin"
              )}
            />
          </Button>
        )}
      </div>
    );
  }

  // Full version with glass card
  return (
    <GlassCard
      variant="default"
      intensity="medium"
      className={cn("p-4", className)}
    >
      <div className="flex items-center justify-between gap-4">
        {/* Left side: Status info */}
        <div className="flex items-center gap-3">
          {/* Pulse indicator */}
          <div
            className={cn(
              "flex items-center justify-center w-10 h-10 rounded-full",
              config.bgClass
            )}
          >
            <PulseIndicator status={status} />
          </div>

          {/* Status text */}
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <StatusBadge status={status} />

              {/* Reconnect attempts counter */}
              {isReconnecting && reconnectAttempts > 0 && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge
                        variant="outline"
                        className="bg-amber-500/10 text-amber-400 border-amber-500/30 text-xs"
                      >
                        Attempt {reconnectAttempts}/{MAX_RECONNECT_ATTEMPTS}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Automatic reconnection in progress</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>

            {/* Description and last connected */}
            <p className="text-xs text-muted-foreground mt-1">
              {config.description}
            </p>
          </div>
        </div>

        {/* Right side: Last connected & reconnect button */}
        <div className="flex items-center gap-3">
          {/* Last connected timestamp */}
          {lastConnected && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" aria-hidden="true" />
                    <span>{relativeTime}</span>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="left">
                  <p>Last connected: {formattedTimestamp}</p>
                  {campaignId && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Campaign: {campaignId}
                    </p>
                  )}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}

          {/* Reconnect button */}
          {showReconnect && (
            <Button
              variant="outline"
              size="sm"
              onClick={onReconnect}
              disabled={isReconnecting || isConnecting}
              className={cn(
                "gap-1.5 transition-all duration-300",
                "bg-white/5 hover:bg-white/10 border-white/10 hover:border-white/20"
              )}
            >
              <RefreshCw
                className={cn(
                  "h-3.5 w-3.5",
                  (isReconnecting || isConnecting) && "animate-spin"
                )}
              />
              Reconnect
            </Button>
          )}

          {/* Activity indicator when connected */}
          {status === "connected" && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/20">
                    <Activity className="h-3.5 w-3.5 text-emerald-400" />
                    <span className="text-xs text-emerald-400 font-medium">
                      Live
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Receiving real-time telemetry updates</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      </div>
    </GlassCard>
  );
});

// Named export for index
export default ConnectionStatus;
