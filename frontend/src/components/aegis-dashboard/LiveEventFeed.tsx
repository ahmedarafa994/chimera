/**
 * LiveEventFeed Component
 *
 * Displays a scrolling feed showing real-time campaign events for Aegis Campaign Dashboard:
 * - Virtualized scrolling list for performance (max 100 events)
 * - Event icons and color coding by type
 * - Timestamp and relative time display
 * - Auto-scroll to latest with manual scroll lock
 * - Event detail expansion on click
 *
 * Follows glass morphism styling pattern from existing components.
 */

"use client";

import { memo, useMemo, useRef, useEffect, useState, useCallback } from "react";
import {
  Activity,
  Play,
  Pause,
  Square,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Zap,
  Target,
  RefreshCw,
  Clock,
  DollarSign,
  Timer,
  Sparkles,
  AlertCircle,
  Heart,
  Link,
  ChevronDown,
  ChevronUp,
  ArrowDown,
  Filter,
  Copy,
  Check,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import {
  AegisTelemetryEventBase,
  AegisTelemetryEventType,
  EVENT_TYPE_LABELS,
  MAX_EVENT_HISTORY,
} from "@/types/aegis-telemetry";

// ============================================================================
// Types
// ============================================================================

export interface LiveEventFeedProps {
  /** Array of telemetry events to display */
  events: AegisTelemetryEventBase[];
  /** Whether the component is in loading state */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show in compact mode */
  compact?: boolean;
  /** Maximum height for the feed (default: 400px) */
  maxHeight?: number;
  /** Whether auto-scroll is initially enabled (default: true) */
  autoScrollEnabled?: boolean;
  /** Callback when an event is clicked */
  onEventClick?: (event: AegisTelemetryEventBase) => void;
}

// ============================================================================
// Configuration
// ============================================================================

/**
 * Event type configuration with icons and colors
 */
interface EventTypeConfig {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  colorClass: string;
  bgClass: string;
  borderClass: string;
  description: string;
}

const EVENT_TYPE_CONFIG: Record<AegisTelemetryEventType, EventTypeConfig> = {
  [AegisTelemetryEventType.CAMPAIGN_STARTED]: {
    icon: Play,
    label: "Campaign Started",
    colorClass: "text-emerald-400",
    bgClass: "bg-emerald-500/10",
    borderClass: "border-emerald-500/20",
    description: "Campaign has been initialized and started",
  },
  [AegisTelemetryEventType.CAMPAIGN_PAUSED]: {
    icon: Pause,
    label: "Campaign Paused",
    colorClass: "text-amber-400",
    bgClass: "bg-amber-500/10",
    borderClass: "border-amber-500/20",
    description: "Campaign execution has been paused",
  },
  [AegisTelemetryEventType.CAMPAIGN_RESUMED]: {
    icon: Play,
    label: "Campaign Resumed",
    colorClass: "text-emerald-400",
    bgClass: "bg-emerald-500/10",
    borderClass: "border-emerald-500/20",
    description: "Campaign execution has resumed",
  },
  [AegisTelemetryEventType.CAMPAIGN_COMPLETED]: {
    icon: CheckCircle,
    label: "Campaign Completed",
    colorClass: "text-cyan-400",
    bgClass: "bg-cyan-500/10",
    borderClass: "border-cyan-500/20",
    description: "Campaign has completed successfully",
  },
  [AegisTelemetryEventType.CAMPAIGN_FAILED]: {
    icon: XCircle,
    label: "Campaign Failed",
    colorClass: "text-red-400",
    bgClass: "bg-red-500/10",
    borderClass: "border-red-500/20",
    description: "Campaign has failed with an error",
  },
  [AegisTelemetryEventType.ITERATION_STARTED]: {
    icon: RefreshCw,
    label: "Iteration Started",
    colorClass: "text-blue-400",
    bgClass: "bg-blue-500/10",
    borderClass: "border-blue-500/20",
    description: "New iteration has started",
  },
  [AegisTelemetryEventType.ITERATION_COMPLETED]: {
    icon: CheckCircle,
    label: "Iteration Completed",
    colorClass: "text-blue-400",
    bgClass: "bg-blue-500/10",
    borderClass: "border-blue-500/20",
    description: "Iteration has completed",
  },
  [AegisTelemetryEventType.ATTACK_STARTED]: {
    icon: Target,
    label: "Attack Started",
    colorClass: "text-purple-400",
    bgClass: "bg-purple-500/10",
    borderClass: "border-purple-500/20",
    description: "Attack attempt has been initiated",
  },
  [AegisTelemetryEventType.ATTACK_COMPLETED]: {
    icon: Target,
    label: "Attack Completed",
    colorClass: "text-purple-400",
    bgClass: "bg-purple-500/10",
    borderClass: "border-purple-500/20",
    description: "Attack attempt has completed",
  },
  [AegisTelemetryEventType.ATTACK_RESULT]: {
    icon: Zap,
    label: "Attack Result",
    colorClass: "text-amber-400",
    bgClass: "bg-amber-500/10",
    borderClass: "border-amber-500/20",
    description: "Result from an attack attempt",
  },
  [AegisTelemetryEventType.TECHNIQUE_APPLIED]: {
    icon: Sparkles,
    label: "Technique Applied",
    colorClass: "text-violet-400",
    bgClass: "bg-violet-500/10",
    borderClass: "border-violet-500/20",
    description: "Transformation technique has been applied",
  },
  [AegisTelemetryEventType.TECHNIQUE_RESULT]: {
    icon: Sparkles,
    label: "Technique Result",
    colorClass: "text-violet-400",
    bgClass: "bg-violet-500/10",
    borderClass: "border-violet-500/20",
    description: "Result from technique application",
  },
  [AegisTelemetryEventType.COST_UPDATE]: {
    icon: DollarSign,
    label: "Cost Update",
    colorClass: "text-green-400",
    bgClass: "bg-green-500/10",
    borderClass: "border-green-500/20",
    description: "Token usage and cost updated",
  },
  [AegisTelemetryEventType.TOKEN_USAGE]: {
    icon: DollarSign,
    label: "Token Usage",
    colorClass: "text-green-400",
    bgClass: "bg-green-500/10",
    borderClass: "border-green-500/20",
    description: "Token consumption recorded",
  },
  [AegisTelemetryEventType.LATENCY_UPDATE]: {
    icon: Timer,
    label: "Latency Update",
    colorClass: "text-cyan-400",
    bgClass: "bg-cyan-500/10",
    borderClass: "border-cyan-500/20",
    description: "Latency metrics updated",
  },
  [AegisTelemetryEventType.PROMPT_EVOLVED]: {
    icon: Sparkles,
    label: "Prompt Evolved",
    colorClass: "text-pink-400",
    bgClass: "bg-pink-500/10",
    borderClass: "border-pink-500/20",
    description: "Prompt has been transformed",
  },
  [AegisTelemetryEventType.SCORE_UPDATE]: {
    icon: Activity,
    label: "Score Update",
    colorClass: "text-indigo-400",
    bgClass: "bg-indigo-500/10",
    borderClass: "border-indigo-500/20",
    description: "Attack score updated",
  },
  [AegisTelemetryEventType.HEARTBEAT]: {
    icon: Heart,
    label: "Heartbeat",
    colorClass: "text-gray-400",
    bgClass: "bg-gray-500/10",
    borderClass: "border-gray-500/20",
    description: "WebSocket heartbeat received",
  },
  [AegisTelemetryEventType.CONNECTION_ACK]: {
    icon: Link,
    label: "Connected",
    colorClass: "text-emerald-400",
    bgClass: "bg-emerald-500/10",
    borderClass: "border-emerald-500/20",
    description: "WebSocket connection established",
  },
  [AegisTelemetryEventType.ERROR]: {
    icon: AlertCircle,
    label: "Error",
    colorClass: "text-red-400",
    bgClass: "bg-red-500/10",
    borderClass: "border-red-500/20",
    description: "An error has occurred",
  },
};

/**
 * Event categories for filtering
 */
const EVENT_CATEGORIES = {
  campaign: [
    AegisTelemetryEventType.CAMPAIGN_STARTED,
    AegisTelemetryEventType.CAMPAIGN_PAUSED,
    AegisTelemetryEventType.CAMPAIGN_RESUMED,
    AegisTelemetryEventType.CAMPAIGN_COMPLETED,
    AegisTelemetryEventType.CAMPAIGN_FAILED,
  ],
  iteration: [
    AegisTelemetryEventType.ITERATION_STARTED,
    AegisTelemetryEventType.ITERATION_COMPLETED,
  ],
  attack: [
    AegisTelemetryEventType.ATTACK_STARTED,
    AegisTelemetryEventType.ATTACK_COMPLETED,
    AegisTelemetryEventType.ATTACK_RESULT,
  ],
  technique: [
    AegisTelemetryEventType.TECHNIQUE_APPLIED,
    AegisTelemetryEventType.TECHNIQUE_RESULT,
    AegisTelemetryEventType.PROMPT_EVOLVED,
  ],
  resource: [
    AegisTelemetryEventType.COST_UPDATE,
    AegisTelemetryEventType.TOKEN_USAGE,
    AegisTelemetryEventType.LATENCY_UPDATE,
  ],
  connection: [
    AegisTelemetryEventType.HEARTBEAT,
    AegisTelemetryEventType.CONNECTION_ACK,
    AegisTelemetryEventType.ERROR,
  ],
} as const;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format timestamp as relative time (e.g., "2s ago", "5m ago")
 */
function formatRelativeTime(timestamp: string): string {
  const now = new Date();
  const eventTime = new Date(timestamp);
  const diffMs = now.getTime() - eventTime.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);

  if (diffSeconds < 5) return "just now";
  if (diffSeconds < 60) return `${diffSeconds}s ago`;

  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m ago`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/**
 * Get event summary for display
 */
function getEventSummary(event: AegisTelemetryEventBase): string {
  const data = event.data;

  switch (event.event_type) {
    case AegisTelemetryEventType.CAMPAIGN_STARTED:
      return `Target: ${data?.target_model || "Unknown"} | ${data?.max_iterations || "?"} iterations`;
    case AegisTelemetryEventType.CAMPAIGN_COMPLETED:
      return `${data?.successful_attacks || 0}/${data?.total_attacks || 0} successful | ${(data?.final_success_rate || 0).toFixed(1)}%`;
    case AegisTelemetryEventType.CAMPAIGN_FAILED:
      return data?.error_message || "Unknown error";
    case AegisTelemetryEventType.ITERATION_STARTED:
      return `Iteration #${data?.iteration || "?"}`;
    case AegisTelemetryEventType.ITERATION_COMPLETED:
      return `Iteration #${data?.iteration || "?"} | Score: ${(data?.score || 0).toFixed(2)}`;
    case AegisTelemetryEventType.ATTACK_STARTED:
      return `Technique: ${data?.technique || "Mixed"} | ${data?.attack_id?.substring(0, 8) || ""}...`;
    case AegisTelemetryEventType.ATTACK_COMPLETED:
      return `${data?.success ? "Success" : "Failed"} | Score: ${(data?.score || 0).toFixed(2)} | ${(data?.duration_ms || 0)}ms`;
    case AegisTelemetryEventType.TECHNIQUE_APPLIED:
      return `${data?.technique_name || "Unknown"} | ${data?.success ? "Success" : "Failed"}`;
    case AegisTelemetryEventType.COST_UPDATE:
      return `Total: $${(data?.total_cost_usd || 0).toFixed(4)} | Session: $${(data?.session_cost_usd || 0).toFixed(4)}`;
    case AegisTelemetryEventType.PROMPT_EVOLVED:
      return `Iteration #${data?.iteration || "?"} | +${(data?.improvement || 0).toFixed(2)} improvement`;
    case AegisTelemetryEventType.LATENCY_UPDATE:
      return `API: ${data?.api_latency_ms || 0}ms | Processing: ${data?.processing_latency_ms || 0}ms`;
    case AegisTelemetryEventType.HEARTBEAT:
      return `Server time: ${data?.server_time ? formatTimestamp(data.server_time) : "Unknown"}`;
    case AegisTelemetryEventType.CONNECTION_ACK:
      return `Client: ${data?.client_id?.substring(0, 8) || ""}...`;
    case AegisTelemetryEventType.ERROR:
      return `[${data?.severity || "error"}] ${data?.error_message || "Unknown error"}`;
    default:
      return `Sequence: ${event.sequence}`;
  }
}

/**
 * Get event detail data for expansion
 */
function getEventDetails(event: AegisTelemetryEventBase): Record<string, unknown> | null {
  if (!event.data || typeof event.data !== "object") {
    return null;
  }
  return event.data as Record<string, unknown>;
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Event icon with animation based on event type
 */
const EventIcon = memo(function EventIcon({
  eventType,
  className,
}: {
  eventType: AegisTelemetryEventType;
  className?: string;
}) {
  const config = EVENT_TYPE_CONFIG[eventType];
  const Icon = config?.icon || Activity;

  return (
    <div
      className={cn(
        "flex items-center justify-center rounded-lg p-1.5",
        config?.bgClass,
        config?.borderClass,
        "border transition-all duration-300",
        className
      )}
    >
      <Icon
        className={cn("h-3.5 w-3.5", config?.colorClass)}
        aria-hidden="true"
      />
    </div>
  );
});

/**
 * Event timestamp display with tooltip
 */
const EventTimestamp = memo(function EventTimestamp({
  timestamp,
  compact,
}: {
  timestamp: string;
  compact?: boolean;
}) {
  const relativeTime = useMemo(() => formatRelativeTime(timestamp), [timestamp]);
  const fullTime = useMemo(() => formatTimestamp(timestamp), [timestamp]);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <Clock className="h-2.5 w-2.5" aria-hidden="true" />
            <span className="tabular-nums">
              {compact ? relativeTime : fullTime}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>{new Date(timestamp).toLocaleString()}</p>
          <p className="text-xs text-muted-foreground">{relativeTime}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Copy button for event details
 */
const CopyButton = memo(function CopyButton({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      // Clipboard API not available
    }
  }, [text]);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              handleCopy();
            }}
            className={cn(
              "h-6 w-6 p-0",
              copied && "text-emerald-400",
              className
            )}
          >
            {copied ? (
              <Check className="h-3 w-3" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{copied ? "Copied!" : "Copy event data"}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Event details expansion panel
 */
const EventDetails = memo(function EventDetails({
  event,
}: {
  event: AegisTelemetryEventBase;
}) {
  const details = getEventDetails(event);

  if (!details) {
    return (
      <div className="text-xs text-muted-foreground italic">
        No additional data available
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">
          Event Data
        </span>
        <CopyButton text={JSON.stringify(details, null, 2)} />
      </div>
      <pre className="overflow-x-auto rounded-lg bg-black/30 p-2 text-[10px] text-gray-300 font-mono">
        {JSON.stringify(details, null, 2)}
      </pre>
    </div>
  );
});

/**
 * Single event row component
 */
const EventRow = memo(function EventRow({
  event,
  compact,
  isExpanded,
  onToggleExpand,
}: {
  event: AegisTelemetryEventBase;
  compact?: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
}) {
  const config = EVENT_TYPE_CONFIG[event.event_type];
  const summary = useMemo(() => getEventSummary(event), [event]);

  return (
    <div
      className={cn(
        "group relative border-b border-white/5 transition-all duration-200",
        "hover:bg-white/[0.02]",
        isExpanded && "bg-white/[0.03]"
      )}
    >
      {/* Main row */}
      <button
        onClick={onToggleExpand}
        className="w-full text-left p-2 flex items-start gap-2"
        aria-expanded={isExpanded}
        aria-label={`${config?.label || event.event_type} event - click to expand`}
      >
        {/* Event icon */}
        <EventIcon eventType={event.event_type} />

        {/* Event content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "text-xs font-medium truncate",
                config?.colorClass
              )}
            >
              {config?.label || EVENT_TYPE_LABELS[event.event_type]}
            </span>
            {!compact && (
              <Badge
                variant="outline"
                className="text-[10px] px-1 py-0 h-4 text-muted-foreground border-white/10"
              >
                #{event.sequence}
              </Badge>
            )}
          </div>
          <p className="text-[11px] text-muted-foreground truncate mt-0.5">
            {summary}
          </p>
        </div>

        {/* Timestamp and expand indicator */}
        <div className="flex items-center gap-1.5 shrink-0">
          <EventTimestamp timestamp={event.timestamp} compact={compact} />
          {isExpanded ? (
            <ChevronUp className="h-3 w-3 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
          )}
        </div>
      </button>

      {/* Expanded details */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-1 border-t border-white/5">
          <EventDetails event={event} />
        </div>
      )}
    </div>
  );
});

/**
 * Filter dropdown for event types
 */
const EventFilter = memo(function EventFilter({
  enabledTypes,
  onToggleType,
  onSelectAll,
  onDeselectAll,
}: {
  enabledTypes: Set<AegisTelemetryEventType>;
  onToggleType: (type: AegisTelemetryEventType) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}) {
  const allTypes = Object.values(AegisTelemetryEventType);
  const allEnabled = enabledTypes.size === allTypes.length;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs text-muted-foreground"
        >
          <Filter className="h-3 w-3 mr-1" />
          Filter
          {enabledTypes.size < allTypes.length && (
            <Badge
              variant="secondary"
              className="ml-1 h-4 w-4 p-0 text-[10px] flex items-center justify-center"
            >
              {enabledTypes.size}
            </Badge>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-56 max-h-80 overflow-y-auto bg-black/90 border-white/10"
      >
        <div className="flex items-center justify-between px-2 py-1.5 border-b border-white/10">
          <Button
            variant="ghost"
            size="sm"
            onClick={allEnabled ? onDeselectAll : onSelectAll}
            className="h-6 px-2 text-xs"
          >
            {allEnabled ? "Deselect All" : "Select All"}
          </Button>
        </div>
        {Object.entries(EVENT_CATEGORIES).map(([category, types]) => (
          <div key={category} className="py-1">
            <div className="px-2 py-1 text-[10px] uppercase text-muted-foreground font-medium">
              {category}
            </div>
            {types.map((type) => {
              const config = EVENT_TYPE_CONFIG[type];
              return (
                <DropdownMenuCheckboxItem
                  key={type}
                  checked={enabledTypes.has(type)}
                  onCheckedChange={() => onToggleType(type)}
                  className="text-xs"
                >
                  <span className={config?.colorClass}>{config?.label}</span>
                </DropdownMenuCheckboxItem>
              );
            })}
          </div>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
});

/**
 * Auto-scroll toggle button
 */
const AutoScrollButton = memo(function AutoScrollButton({
  enabled,
  onToggle,
  hasNewEvents,
}: {
  enabled: boolean;
  onToggle: () => void;
  hasNewEvents?: boolean;
}) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className={cn(
              "h-6 px-2 text-xs",
              enabled
                ? "text-emerald-400 hover:text-emerald-300"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <ArrowDown
              className={cn(
                "h-3 w-3 mr-1",
                enabled && "animate-bounce",
                hasNewEvents && !enabled && "text-amber-400"
              )}
            />
            Auto-scroll
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{enabled ? "Auto-scroll enabled" : "Click to enable auto-scroll"}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Empty state component
 */
const EmptyState = memo(function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <Activity className="h-8 w-8 text-muted-foreground/50 mb-3" />
      <p className="text-sm text-muted-foreground">No events yet</p>
      <p className="text-xs text-muted-foreground/70 mt-1">
        Events will appear here in real-time
      </p>
    </div>
  );
});

/**
 * Loading skeleton component
 */
const LoadingSkeleton = memo(function LoadingSkeleton() {
  return (
    <div className="animate-pulse space-y-2 p-3">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="flex items-start gap-2">
          <div className="h-7 w-7 bg-white/10 rounded-lg shrink-0" />
          <div className="flex-1 space-y-1.5">
            <div className="h-3 w-24 bg-white/10 rounded" />
            <div className="h-2.5 w-48 bg-white/10 rounded" />
          </div>
          <div className="h-2.5 w-12 bg-white/10 rounded" />
        </div>
      ))}
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * LiveEventFeed displays a scrolling feed of real-time campaign events.
 *
 * Features:
 * - Virtualized-like scrolling with max 100 events for performance
 * - Event icons and color coding by type
 * - Timestamp with relative time display
 * - Auto-scroll to latest with manual scroll lock
 * - Event detail expansion on click
 * - Event type filtering
 */
export const LiveEventFeed = memo(function LiveEventFeed({
  events,
  isLoading = false,
  className,
  compact = false,
  maxHeight = 400,
  autoScrollEnabled: initialAutoScroll = true,
  onEventClick,
}: LiveEventFeedProps) {
  // State
  const [expandedEventId, setExpandedEventId] = useState<string | null>(null);
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(initialAutoScroll);
  const [enabledEventTypes, setEnabledEventTypes] = useState<Set<AegisTelemetryEventType>>(
    new Set(Object.values(AegisTelemetryEventType))
  );
  const [hasNewEvents, setHasNewEvents] = useState(false);
  const lastEventCountRef = useRef(0);

  // Refs
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isUserScrollingRef = useRef(false);

  // Filter events by enabled types and limit to MAX_EVENT_HISTORY
  const filteredEvents = useMemo(() => {
    const filtered = events.filter((e) => enabledEventTypes.has(e.event_type));
    // Return most recent events (events should be sorted by sequence/timestamp)
    return filtered.slice(-MAX_EVENT_HISTORY);
  }, [events, enabledEventTypes]);

  // Generate unique key for each event
  const getEventKey = useCallback(
    (event: AegisTelemetryEventBase) => `${event.campaign_id}-${event.sequence}`,
    []
  );

  // Handle auto-scroll
  useEffect(() => {
    if (
      autoScrollEnabled &&
      scrollContainerRef.current &&
      !isUserScrollingRef.current
    ) {
      scrollContainerRef.current.scrollTop =
        scrollContainerRef.current.scrollHeight;
    }
  }, [filteredEvents, autoScrollEnabled]);

  // Detect new events when auto-scroll is disabled
  useEffect(() => {
    if (!autoScrollEnabled && events.length > lastEventCountRef.current) {
      setHasNewEvents(true);
    }
    lastEventCountRef.current = events.length;
  }, [events.length, autoScrollEnabled]);

  // Handle scroll events to detect user scrolling
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;

    if (isAtBottom) {
      setHasNewEvents(false);
      if (!autoScrollEnabled) {
        // User scrolled to bottom manually, re-enable auto-scroll
      }
    } else if (autoScrollEnabled) {
      // User scrolled up, disable auto-scroll
      isUserScrollingRef.current = true;
      setAutoScrollEnabled(false);
      setTimeout(() => {
        isUserScrollingRef.current = false;
      }, 100);
    }
  }, [autoScrollEnabled]);

  // Toggle event expansion
  const toggleEventExpand = useCallback(
    (event: AegisTelemetryEventBase) => {
      const key = getEventKey(event);
      setExpandedEventId((prev) => (prev === key ? null : key));
      onEventClick?.(event);
    },
    [getEventKey, onEventClick]
  );

  // Event filter handlers
  const toggleEventType = useCallback((type: AegisTelemetryEventType) => {
    setEnabledEventTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  }, []);

  const selectAllEventTypes = useCallback(() => {
    setEnabledEventTypes(new Set(Object.values(AegisTelemetryEventType)));
  }, []);

  const deselectAllEventTypes = useCallback(() => {
    setEnabledEventTypes(new Set());
  }, []);

  // Toggle auto-scroll
  const toggleAutoScroll = useCallback(() => {
    setAutoScrollEnabled((prev) => !prev);
    if (!autoScrollEnabled) {
      setHasNewEvents(false);
      // Scroll to bottom when enabling
      if (scrollContainerRef.current) {
        scrollContainerRef.current.scrollTop =
          scrollContainerRef.current.scrollHeight;
      }
    }
  }, [autoScrollEnabled]);

  // Loading state
  if (isLoading) {
    return (
      <GlassCard variant="default" intensity="medium" className={cn("p-0", className)}>
        <div className="flex items-center justify-between p-3 border-b border-white/5">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-cyan-400" aria-hidden="true" />
            <h3 className="text-sm font-medium text-muted-foreground">
              Live Events
            </h3>
          </div>
        </div>
        <LoadingSkeleton />
      </GlassCard>
    );
  }

  return (
    <GlassCard
      variant="default"
      intensity="medium"
      className={cn("p-0 overflow-hidden", className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-white/5">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-cyan-400" aria-hidden="true" />
          <h3 className="text-sm font-medium text-muted-foreground">
            Live Events
          </h3>
          <Badge
            variant="outline"
            className="text-[10px] px-1.5 py-0 h-4 text-muted-foreground border-white/10"
          >
            {filteredEvents.length}
            {events.length !== filteredEvents.length && (
              <span className="text-muted-foreground/50">/{events.length}</span>
            )}
          </Badge>
        </div>
        <div className="flex items-center gap-1">
          <EventFilter
            enabledTypes={enabledEventTypes}
            onToggleType={toggleEventType}
            onSelectAll={selectAllEventTypes}
            onDeselectAll={deselectAllEventTypes}
          />
          <AutoScrollButton
            enabled={autoScrollEnabled}
            onToggle={toggleAutoScroll}
            hasNewEvents={hasNewEvents}
          />
        </div>
      </div>

      {/* Event list */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="overflow-y-auto overflow-x-hidden"
        style={{ maxHeight }}
      >
        {filteredEvents.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="divide-y divide-white/5">
            {filteredEvents.map((event) => {
              const key = getEventKey(event);
              return (
                <EventRow
                  key={key}
                  event={event}
                  compact={compact}
                  isExpanded={expandedEventId === key}
                  onToggleExpand={() => toggleEventExpand(event)}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* New events indicator */}
      {hasNewEvents && !autoScrollEnabled && (
        <button
          onClick={toggleAutoScroll}
          className="absolute bottom-2 left-1/2 transform -translate-x-1/2 px-3 py-1.5 rounded-full bg-cyan-500/20 border border-cyan-500/30 text-cyan-400 text-xs font-medium hover:bg-cyan-500/30 transition-all animate-bounce"
        >
          <ArrowDown className="h-3 w-3 inline mr-1" />
          New events
        </button>
      )}
    </GlassCard>
  );
});

// Named export for index
export default LiveEventFeed;
