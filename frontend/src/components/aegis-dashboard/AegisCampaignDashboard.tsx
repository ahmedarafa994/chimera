/**
 * AegisCampaignDashboard Component
 *
 * Main container component for the Real-Time Aegis Campaign Dashboard that orchestrates
 * all sub-components including:
 * - ConnectionStatus: WebSocket connection status indicator
 * - SuccessRateCard: Attack success rate with trend
 * - SuccessRateTrendChart: Success rate over time chart
 * - TechniqueBreakdown: Performance breakdown by technique
 * - TokenUsageCard: Token consumption and cost tracking
 * - LatencyCard: API and processing latency metrics
 * - LiveEventFeed: Real-time event stream
 * - PromptEvolutionTimeline: Prompt transformation visualization
 *
 * Features:
 * - Responsive grid layout for metrics cards
 * - Campaign selector/ID input
 * - Start/stop campaign controls
 * - Loading and error states
 * - WebSocket connection management via useAegisTelemetry hook
 */

"use client";

import { useState, useCallback, memo, useMemo, useEffect, useRef } from "react";
import {
  Play,
  Pause,
  Square,
  RefreshCw,
  AlertCircle,
  Activity,
  Target,
  FileSearch,
  Settings2,
  Loader2,
  WifiOff,
  AlertTriangle,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

// Import sub-components
import { ConnectionStatus } from "./ConnectionStatus";
import { SuccessRateCard } from "./SuccessRateCard";
import { SuccessRateTrendChart } from "./SuccessRateTrendChart";
import { TechniqueBreakdown } from "./TechniqueBreakdown";
import { TokenUsageCard } from "./TokenUsageCard";
import { LatencyCard } from "./LatencyCard";
import { LiveEventFeed } from "./LiveEventFeed";
import { PromptEvolutionTimeline } from "./PromptEvolutionTimeline";

// Import hook and types
import { useAegisTelemetry } from "@/lib/hooks/useAegisTelemetry";
import {
  CampaignStatus,
  CampaignSummary,
  WebSocketConnectionStatus,
  MAX_RECONNECT_ATTEMPTS,
} from "@/types/aegis-telemetry";

// ============================================================================
// Types
// ============================================================================

export interface AegisCampaignDashboardProps {
  /** Initial campaign ID to connect to */
  initialCampaignId?: string;
  /** API key for WebSocket authentication */
  apiKey?: string;
  /** Whether to auto-connect on mount */
  autoConnect?: boolean;
  /** Enable debug logging */
  debug?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Constants
// ============================================================================

const CAMPAIGN_STATUS_CONFIG: Record<
  CampaignStatus,
  {
    label: string;
    icon: typeof Activity;
    colorClass: string;
    bgClass: string;
    borderClass: string;
  }
> = {
  [CampaignStatus.PENDING]: {
    label: "Pending",
    icon: Loader2,
    colorClass: "text-gray-400",
    bgClass: "bg-gray-500/10",
    borderClass: "border-gray-500/20",
  },
  [CampaignStatus.RUNNING]: {
    label: "Running",
    icon: Activity,
    colorClass: "text-emerald-400",
    bgClass: "bg-emerald-500/10",
    borderClass: "border-emerald-500/20",
  },
  [CampaignStatus.PAUSED]: {
    label: "Paused",
    icon: Pause,
    colorClass: "text-amber-400",
    bgClass: "bg-amber-500/10",
    borderClass: "border-amber-500/20",
  },
  [CampaignStatus.COMPLETED]: {
    label: "Completed",
    icon: Target,
    colorClass: "text-blue-400",
    bgClass: "bg-blue-500/10",
    borderClass: "border-blue-500/20",
  },
  [CampaignStatus.FAILED]: {
    label: "Failed",
    icon: AlertCircle,
    colorClass: "text-red-400",
    bgClass: "bg-red-500/10",
    borderClass: "border-red-500/20",
  },
  [CampaignStatus.CANCELLED]: {
    label: "Cancelled",
    icon: Square,
    colorClass: "text-gray-400",
    bgClass: "bg-gray-500/10",
    borderClass: "border-gray-500/20",
  },
  [CampaignStatus.IDLE]: {
    label: "Idle",
    icon: Pause,
    colorClass: "text-gray-400",
    bgClass: "bg-gray-500/10",
    borderClass: "border-gray-500/20",
  },
};

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Campaign header with title, status, and controls
 */
const CampaignHeader = memo(function CampaignHeader({
  campaignSummary,
  connectionStatus,
  onReconnect,
  reconnectAttempts,
}: {
  campaignSummary: CampaignSummary | null;
  connectionStatus: WebSocketConnectionStatus;
  onReconnect: () => void;
  reconnectAttempts: number;
}) {
  const statusConfig = campaignSummary?.status
    ? CAMPAIGN_STATUS_CONFIG[campaignSummary.status]
    : null;
  const StatusIcon = statusConfig?.icon || Activity;

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight gradient-text">
            Aegis Campaign Dashboard
          </h1>
          {campaignSummary && statusConfig && (
            <Badge
              variant="outline"
              className={cn(
                "gap-1.5 text-xs font-medium",
                statusConfig.bgClass,
                statusConfig.borderClass,
                statusConfig.colorClass
              )}
            >
              <StatusIcon className="h-3 w-3" aria-hidden="true" />
              {statusConfig.label}
            </Badge>
          )}
        </div>
        {campaignSummary?.objective && (
          <p className="text-sm text-muted-foreground line-clamp-1">
            Objective: {campaignSummary.objective}
          </p>
        )}
        {campaignSummary?.target_model && (
          <p className="text-xs text-muted-foreground/70">
            Target: {campaignSummary.target_model}
          </p>
        )}
      </div>

      <ConnectionStatus
        status={connectionStatus}
        onReconnect={onReconnect}
        reconnectAttempts={reconnectAttempts}
        campaignId={campaignSummary?.campaign_id}
        compact
      />
    </div>
  );
});

/**
 * Campaign selector input with connect button
 */
const CampaignSelector = memo(function CampaignSelector({
  campaignId,
  onCampaignIdChange,
  onConnect,
  isConnecting,
  isConnected,
}: {
  campaignId: string;
  onCampaignIdChange: (id: string) => void;
  onConnect: () => void;
  isConnecting: boolean;
  isConnected: boolean;
}) {
  return (
    <GlassCard variant="default" intensity="medium" className="p-4">
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="flex-1">
          <label
            htmlFor="campaign-id-input"
            className="block text-xs font-medium text-muted-foreground mb-1.5"
          >
            Campaign ID
          </label>
          <div className="relative">
            <FileSearch
              className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              id="campaign-id-input"
              type="text"
              placeholder="Enter campaign ID (e.g., aegis-2024-01-01-abc123)"
              value={campaignId}
              onChange={(e) => onCampaignIdChange(e.target.value)}
              className="pl-10 bg-white/5 border-white/10 focus:border-white/20"
              disabled={isConnecting}
            />
          </div>
        </div>

        <div className="flex items-end gap-2">
          <Button
            variant={isConnected ? "outline" : "default"}
            onClick={onConnect}
            disabled={!campaignId.trim() || isConnecting}
            className={cn(
              "gap-2 min-w-[120px]",
              isConnected && "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
            )}
          >
            {isConnecting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Connecting...
              </>
            ) : isConnected ? (
              <>
                <Activity className="h-4 w-4" />
                Connected
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Connect
              </>
            )}
          </Button>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9"
                  aria-label="Dashboard settings"
                >
                  <Settings2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Dashboard settings</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </GlassCard>
  );
});

/**
 * Progress indicator showing iteration progress
 */
const IterationProgress = memo(function IterationProgress({
  current,
  max,
}: {
  current: number;
  max: number;
}) {
  const percentage = max > 0 ? (current / max) * 100 : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">Iteration Progress</span>
        <span className="font-medium tabular-nums">
          {current} / {max}
        </span>
      </div>
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
});

/**
 * Loading skeleton for the dashboard
 */
const DashboardSkeleton = memo(function DashboardSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Header skeleton */}
      <div className="flex items-center justify-between">
        <Skeleton className="h-9 w-64" />
        <Skeleton className="h-8 w-32" />
      </div>

      {/* Selector skeleton */}
      <GlassCard variant="default" intensity="medium" className="p-4">
        <div className="flex gap-3">
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-9 w-full" />
          </div>
          <Skeleton className="h-9 w-32 self-end" />
        </div>
      </GlassCard>

      {/* Metrics grid skeleton */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <GlassCard key={i} variant="default" intensity="medium" className="p-4">
            <Skeleton className="h-4 w-24 mb-3" />
            <Skeleton className="h-12 w-20 mb-4" />
            <Skeleton className="h-8 w-full" />
          </GlassCard>
        ))}
      </div>

      {/* Charts skeleton */}
      <div className="grid gap-4 lg:grid-cols-2">
        <GlassCard variant="default" intensity="medium" className="p-4">
          <Skeleton className="h-4 w-32 mb-4" />
          <Skeleton className="h-48 w-full" />
        </GlassCard>
        <GlassCard variant="default" intensity="medium" className="p-4">
          <Skeleton className="h-4 w-32 mb-4" />
          <Skeleton className="h-48 w-full" />
        </GlassCard>
      </div>

      {/* Event feed skeleton */}
      <GlassCard variant="default" intensity="medium" className="p-4">
        <Skeleton className="h-4 w-28 mb-4" />
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </div>
      </GlassCard>
    </div>
  );
});

/**
 * Empty state when no campaign is connected
 */
const EmptyState = memo(function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="rounded-full bg-white/5 p-6 mb-6">
        <Target className="h-12 w-12 text-muted-foreground/50" />
      </div>
      <h3 className="text-lg font-semibold text-foreground mb-2">
        No Campaign Connected
      </h3>
      <p className="text-sm text-muted-foreground max-w-md mb-6">
        Enter a campaign ID above to connect to a live Aegis campaign and monitor
        real-time telemetry including attack success rates, technique performance,
        token usage, and prompt evolution.
      </p>
      <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline" className="bg-white/5">
          <Activity className="h-3 w-3 mr-1" />
          Real-time Updates
        </Badge>
        <Badge variant="outline" className="bg-white/5">
          <Target className="h-3 w-3 mr-1" />
          Attack Metrics
        </Badge>
        <Badge variant="outline" className="bg-white/5">
          <Settings2 className="h-3 w-3 mr-1" />
          Technique Analysis
        </Badge>
      </div>
    </div>
  );
});

/**
 * Error state display
 */
const ErrorState = memo(function ErrorState({
  error,
  onRetry,
}: {
  error: {
    error_code?: string;
    error_message: string;
    recoverable?: boolean;
  };
  onRetry?: () => void;
}) {
  return (
    <Alert variant="destructive" className="bg-red-500/10 border-red-500/20">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle className="text-red-400">
        {error.error_code || "Connection Error"}
      </AlertTitle>
      <AlertDescription className="text-red-400/80">
        {error.error_message}
        {error.recoverable && onRetry && (
          <Button
            variant="outline"
            size="sm"
            onClick={onRetry}
            className="mt-2 gap-1.5 bg-red-500/10 border-red-500/20 text-red-400 hover:bg-red-500/20"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Retry Connection
          </Button>
        )}
      </AlertDescription>
    </Alert>
  );
});

/**
 * Disconnection warning banner component
 * Shows when WebSocket connection is lost but dashboard still has data
 */
const DisconnectionBanner = memo(function DisconnectionBanner({
  status,
  reconnectAttempts,
  lastConnected,
  onReconnect,
}: {
  status: WebSocketConnectionStatus;
  reconnectAttempts: number;
  lastConnected: string | null;
  onReconnect: () => void;
}) {
  // Only show when disconnected or in error/reconnecting state (not connecting/connected)
  if (status === "connected" || status === "connecting") {
    return null;
  }

  const isReconnecting = status === "reconnecting";
  const isError = status === "error";

  // Calculate time since last connection
  const getTimeSinceDisconnect = () => {
    if (!lastConnected) return "Unknown";
    const diff = Date.now() - new Date(lastConnected).getTime();
    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border p-3 transition-all duration-300",
        isError
          ? "bg-red-500/10 border-red-500/20"
          : isReconnecting
          ? "bg-amber-500/10 border-amber-500/20"
          : "bg-gray-500/10 border-gray-500/20"
      )}
      role="alert"
      aria-live="polite"
    >
      {/* Animated background for reconnecting state */}
      {isReconnecting && (
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-amber-500/5 to-transparent animate-pulse" />
      )}

      <div className="relative flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {/* Icon */}
          <div
            className={cn(
              "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
              isError
                ? "bg-red-500/20"
                : isReconnecting
                ? "bg-amber-500/20"
                : "bg-gray-500/20"
            )}
          >
            {isReconnecting ? (
              <RefreshCw
                className="h-4 w-4 text-amber-400 animate-spin"
                aria-hidden="true"
              />
            ) : isError ? (
              <AlertTriangle className="h-4 w-4 text-red-400" aria-hidden="true" />
            ) : (
              <WifiOff className="h-4 w-4 text-gray-400" aria-hidden="true" />
            )}
          </div>

          {/* Text content */}
          <div className="min-w-0">
            <p
              className={cn(
                "text-sm font-medium",
                isError
                  ? "text-red-400"
                  : isReconnecting
                  ? "text-amber-400"
                  : "text-gray-300"
              )}
            >
              {isReconnecting
                ? `Reconnecting... (Attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`
                : isError
                ? "Connection Error"
                : "Connection Lost"}
            </p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {isReconnecting
                ? "Attempting to restore real-time updates..."
                : `Data may be stale. Last update: ${getTimeSinceDisconnect()}`}
            </p>
          </div>
        </div>

        {/* Reconnect button */}
        {!isReconnecting && (
          <Button
            variant="outline"
            size="sm"
            onClick={onReconnect}
            className={cn(
              "shrink-0 gap-1.5",
              isError
                ? "bg-red-500/10 border-red-500/20 text-red-400 hover:bg-red-500/20"
                : "bg-white/5 border-white/10 text-foreground hover:bg-white/10"
            )}
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Reconnect
          </Button>
        )}
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * AegisCampaignDashboard is the main container component for the Real-Time
 * Aegis Campaign Dashboard. It orchestrates all sub-components and manages
 * the WebSocket connection to stream live telemetry data.
 *
 * @example
 * ```tsx
 * <AegisCampaignDashboard
 *   initialCampaignId="aegis-2024-01-01-abc123"
 *   autoConnect={true}
 * />
 * ```
 */
export const AegisCampaignDashboard = memo(function AegisCampaignDashboard({
  initialCampaignId = "",
  apiKey,
  autoConnect = false,
  debug = false,
  className,
}: AegisCampaignDashboardProps) {
  // ============================================================================
  // State
  // ============================================================================

  const [campaignIdInput, setCampaignIdInput] = useState(initialCampaignId);
  const [activeCampaignId, setActiveCampaignId] = useState(
    autoConnect && initialCampaignId ? initialCampaignId : ""
  );
  const [lastConnectedTime, setLastConnectedTime] = useState<string | null>(null);
  const wasConnectedRef = useRef(false);

  // ============================================================================
  // WebSocket Hook
  // ============================================================================

  const {
    connectionStatus,
    metrics,
    campaignSummary,
    techniqueBreakdown,
    tokenUsage,
    latencyMetrics,
    recentEvents,
    successRateHistory,
    tokenUsageHistory,
    latencyHistory,
    promptEvolutions,
    error,
    reconnectAttempts,
    isConnected,
    reconnect,
    disconnect,
  } = useAegisTelemetry(activeCampaignId, {
    autoConnect: Boolean(activeCampaignId),
    apiKey,
    debug,
  });

  // ============================================================================
  // Track Last Connected Time
  // ============================================================================

  useEffect(() => {
    if (isConnected && !wasConnectedRef.current) {
      // Just connected, update last connected time
      setLastConnectedTime(new Date().toISOString());
      wasConnectedRef.current = true;
    } else if (!isConnected && wasConnectedRef.current) {
      // Just disconnected, keep last connected time but update ref
      wasConnectedRef.current = false;
    } else if (isConnected) {
      // Still connected, continuously update last connected time
      setLastConnectedTime(new Date().toISOString());
    }
  }, [isConnected, connectionStatus, recentEvents.length]);

  // ============================================================================
  // Handlers
  // ============================================================================

  const handleConnect = useCallback(() => {
    if (campaignIdInput.trim()) {
      setActiveCampaignId(campaignIdInput.trim());
    }
  }, [campaignIdInput]);

  const handleReconnect = useCallback(() => {
    if (activeCampaignId) {
      reconnect();
    }
  }, [activeCampaignId, reconnect]);

  const handleCampaignIdChange = useCallback((id: string) => {
    setCampaignIdInput(id);
  }, []);

  // ============================================================================
  // Computed Values
  // ============================================================================

  const isLoading = connectionStatus === "connecting";
  const hasError = error !== null;
  const showDashboard = isConnected || connectionStatus === "reconnecting";

  // Calculate trend from history
  const trend = useMemo(() => {
    if (successRateHistory.length < 2) {
      return { trend: "stable" as const, change: 0 };
    }
    const recentHistory = successRateHistory.slice(-10);
    const firstRate = recentHistory[0].success_rate;
    const change = metrics.success_rate - firstRate;

    let direction: "up" | "down" | "stable";
    if (change > 5) {
      direction = "up";
    } else if (change < -5) {
      direction = "down";
    } else {
      direction = "stable";
    }

    return { trend: direction, change };
  }, [successRateHistory, metrics.success_rate]);

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div className={cn("space-y-6", className)}>
      {/* Campaign Header */}
      <CampaignHeader
        campaignSummary={campaignSummary}
        connectionStatus={connectionStatus}
        onReconnect={handleReconnect}
        reconnectAttempts={reconnectAttempts}
      />

      {/* Campaign Selector */}
      <CampaignSelector
        campaignId={campaignIdInput}
        onCampaignIdChange={handleCampaignIdChange}
        onConnect={handleConnect}
        isConnecting={isLoading}
        isConnected={isConnected}
      />

      {/* Disconnection Warning Banner - shows when we have data but connection is lost */}
      {showDashboard && (connectionStatus === "disconnected" || connectionStatus === "reconnecting" || connectionStatus === "error") && (
        <DisconnectionBanner
          status={connectionStatus}
          reconnectAttempts={reconnectAttempts}
          lastConnected={lastConnectedTime}
          onReconnect={handleReconnect}
        />
      )}

      {/* Error State */}
      {hasError && (
        <ErrorState
          error={{
            error_code: error?.error_code,
            error_message: error?.error_message || "An error occurred",
            recoverable: error?.recoverable,
          }}
          onRetry={handleReconnect}
        />
      )}

      {/* Loading State */}
      {isLoading && !showDashboard && <DashboardSkeleton />}

      {/* Empty State */}
      {!activeCampaignId && !isLoading && <EmptyState />}

      {/* Dashboard Content */}
      {showDashboard && (
        <div className="space-y-6">
          {/* Iteration Progress */}
          {campaignSummary && (
            <GlassCard variant="default" intensity="medium" className="p-4">
              <IterationProgress
                current={campaignSummary.current_iteration}
                max={campaignSummary.max_iterations}
              />
            </GlassCard>
          )}

          {/* Metrics Cards Grid */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <SuccessRateCard
              metrics={metrics}
              successRateHistory={successRateHistory}
              trend={trend.trend}
              trendChange={trend.change}
              compact
            />
            <TokenUsageCard
              tokenUsage={tokenUsage}
              tokenUsageHistory={tokenUsageHistory}
              successfulAttacks={metrics.successful_attacks}
              totalAttacks={metrics.total_attempts}
              compact
            />
            <LatencyCard
              latencyMetrics={latencyMetrics}
              latencyHistory={latencyHistory}
              compact
            />
            <GlassCard variant="default" intensity="medium" className="p-4">
              <div className="flex flex-col h-full">
                <h3 className="text-sm font-medium text-muted-foreground mb-3">
                  Techniques Applied
                </h3>
                <div className="flex-1 flex items-center">
                  <span className="text-3xl font-bold tabular-nums gradient-text">
                    {techniqueBreakdown.length}
                  </span>
                  <span className="ml-2 text-sm text-muted-foreground">
                    unique techniques
                  </span>
                </div>
                <div className="mt-2 text-xs text-muted-foreground">
                  Avg success rate:{" "}
                  {techniqueBreakdown.length > 0
                    ? (
                        techniqueBreakdown.reduce(
                          (sum, t) => sum + t.success_rate,
                          0
                        ) / techniqueBreakdown.length
                      ).toFixed(1)
                    : 0}
                  %
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Charts Row */}
          <div className="grid gap-4 lg:grid-cols-2">
            <SuccessRateTrendChart
              data={successRateHistory}
              timeWindowMinutes={30}
            />
            <TechniqueBreakdown
              techniques={techniqueBreakdown}
              compact
            />
          </div>

          {/* Bottom Section: Event Feed and Prompt Evolution */}
          <div className="grid gap-4 lg:grid-cols-2">
            <LiveEventFeed
              events={recentEvents}
              maxHeight={400}
            />
            <PromptEvolutionTimeline
              evolutions={promptEvolutions}
              compact
            />
          </div>
        </div>
      )}
    </div>
  );
});

// Named export for index
export default AegisCampaignDashboard;
