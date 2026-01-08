"use client";

/**
 * Activity Feed Component for Project Chimera
 * 
 * Displays real-time activity events with filtering, search,
 * and auto-scroll capabilities.
 * 
 * WCAG 2.1 AA Compliant:
 * - Semantic HTML structure with proper landmarks
 * - Keyboard navigation support
 * - Screen reader announcements for live updates
 * - Sufficient color contrast (4.5:1 minimum)
 * - Focus management and visible focus indicators
 */

import React, { useRef, useEffect, useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowDown,
  Brain,
  Bug,
  CheckCircle2,
  Clock,
  Cpu,
  Filter,
  Info,
  Loader2,
  Pause,
  Search,
  Server,
  ServerOff,
  Skull,
  Sparkles,
  Target,
  Trash2,
  TrendingUp,
  UserMinus,
  UserPlus,
  Wand2,
  Wifi,
  WifiOff,
  XCircle,
} from "lucide-react";
import type {
  ActivityEvent,
  ActivityEventType,
  ActivityStatus,
} from "@/types/activity-types";
import {
  formatActivityTime,
  formatDuration,
  getActivityStatusColor,
  getActivityStatusBgColor,
} from "@/types/activity-types";
import { useActivityFeed } from "@/hooks/use-activity-feed";

// =============================================================================
// Error Boundary Component
// =============================================================================

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ActivityFeedErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ActivityFeed Error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <Card className="p-6">
            <div 
              className="flex flex-col items-center justify-center text-center"
              role="alert"
              aria-live="assertive"
            >
              <AlertCircle className="h-12 w-12 text-destructive mb-4" aria-hidden="true" />
              <h3 className="text-lg font-semibold mb-2">Something went wrong</h3>
              <p className="text-sm text-muted-foreground mb-4">
                The activity feed encountered an error. Please try refreshing the page.
              </p>
              <Button
                onClick={() => this.setState({ hasError: false })}
                variant="outline"
              >
                Try Again
              </Button>
            </div>
          </Card>
        )
      );
    }

    return this.props.children;
  }
}

// =============================================================================
// Screen Reader Live Region Component
// =============================================================================

interface LiveRegionProps {
  message: string;
  politeness?: "polite" | "assertive";
}

function LiveRegion({ message, politeness = "polite" }: LiveRegionProps) {
  return (
    <div
      role="status"
      aria-live={politeness}
      aria-atomic="true"
      className="sr-only"
    >
      {message}
    </div>
  );
}

// =============================================================================
// Loading Skeleton Component
// =============================================================================

function ActivityFeedSkeleton() {
  return (
    <Card className="flex flex-col" aria-busy="true" aria-label="Loading activity feed">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-5 w-5 rounded" />
            <Skeleton className="h-6 w-32" />
          </div>
          <Skeleton className="h-6 w-16 rounded-full" />
        </div>
        <Skeleton className="h-4 w-64 mt-2" />
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-3">
        {/* Stats skeleton */}
        <div className="grid grid-cols-4 gap-2 p-3 bg-muted/30 rounded-lg">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="text-center">
              <Skeleton className="h-7 w-8 mx-auto mb-1" />
              <Skeleton className="h-3 w-12 mx-auto" />
            </div>
          ))}
        </div>
        
        {/* Filter skeleton */}
        <div className="flex items-center gap-2">
          <Skeleton className="h-8 flex-1" />
          <Skeleton className="h-8 w-20" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
        </div>
        
        {/* Events skeleton */}
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex gap-3 p-3 rounded-lg border border-border/50">
              <Skeleton className="h-10 w-10 rounded-lg flex-shrink-0" />
              <div className="flex-1 space-y-2">
                <div className="flex items-center justify-between">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-3 w-16" />
                </div>
                <Skeleton className="h-3 w-full" />
                <div className="flex gap-1.5">
                  <Skeleton className="h-5 w-16 rounded-full" />
                  <Skeleton className="h-5 w-20 rounded-full" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Icon Mapping
// =============================================================================

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  Skull,
  Brain,
  Bug,
  Target,
  TrendingUp,
  Wand2,
  Sparkles,
  Server,
  ServerOff,
  Cpu,
  UserPlus,
  UserMinus,
  AlertCircle,
  AlertTriangle,
  Info,
  Activity,
};

function renderActivityIcon(type: ActivityEventType, className?: string): React.ReactNode {
  const iconName: Record<ActivityEventType, string> = {
    jailbreak_started: "Skull",
    jailbreak_completed: "Skull",
    jailbreak_failed: "Skull",
    autodan_started: "Brain",
    autodan_completed: "Brain",
    autodan_failed: "Brain",
    gptfuzz_started: "Bug",
    gptfuzz_progress: "Bug",
    gptfuzz_completed: "Bug",
    gptfuzz_failed: "Bug",
    houyi_started: "Target",
    houyi_completed: "Target",
    houyi_failed: "Target",
    gradient_started: "TrendingUp",
    gradient_completed: "TrendingUp",
    gradient_failed: "TrendingUp",
    transform_completed: "Wand2",
    generation_completed: "Sparkles",
    provider_connected: "Server",
    provider_disconnected: "ServerOff",
    model_changed: "Cpu",
    session_created: "UserPlus",
    session_expired: "UserMinus",
    error: "AlertCircle",
    warning: "AlertTriangle",
    info: "Info",
  };
  const IconComponent = iconMap[iconName[type]] || Activity;
  return <IconComponent className={className} />;
}

// =============================================================================
// Status Icon Component
// =============================================================================

interface StatusIconProps {
  status: ActivityStatus;
  label?: string;
}

function StatusIcon({ status, label }: StatusIconProps) {
  const statusLabels: Record<ActivityStatus, string> = {
    success: "Completed successfully",
    failed: "Failed",
    running: "In progress",
    pending: "Pending",
    warning: "Warning",
    info: "Information",
  };

  const ariaLabel = label || statusLabels[status];

  switch (status) {
    case "success":
      return <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" aria-label={ariaLabel} role="img" />;
    case "failed":
      return <XCircle className="h-4 w-4 text-red-600 dark:text-red-400" aria-label={ariaLabel} role="img" />;
    case "running":
      return <Loader2 className="h-4 w-4 text-blue-600 dark:text-blue-400 animate-spin" aria-label={ariaLabel} role="img" />;
    case "pending":
      return <Clock className="h-4 w-4 text-yellow-600 dark:text-yellow-400" aria-label={ariaLabel} role="img" />;
    case "warning":
      return <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400" aria-label={ariaLabel} role="img" />;
    case "info":
      return <Info className="h-4 w-4 text-sky-600 dark:text-sky-400" aria-label={ariaLabel} role="img" />;
    default:
      return <Activity className="h-4 w-4 text-gray-600 dark:text-gray-400" aria-label="Activity" role="img" />;
  }
}

// =============================================================================
// Activity Event Item Component
// =============================================================================

interface ActivityEventItemProps {
  event: ActivityEvent;
  isNew?: boolean;
  isFocused?: boolean;
  onKeyDown?: (e: React.KeyboardEvent) => void;
}

const ActivityEventItem = React.forwardRef<HTMLElement, ActivityEventItemProps>(
  ({ event, isNew, isFocused, onKeyDown }, ref) => {
    const statusColor = getActivityStatusColor(event.status);
    const statusBgColor = getActivityStatusBgColor(event.status);

    const accessibleDescription = useMemo(() => {
      const parts = [event.title];
      if (event.description) parts.push(event.description);
      if (event.duration_ms) parts.push(`Duration: ${formatDuration(event.duration_ms)}`);
      if (event.metadata?.provider) parts.push(`Provider: ${event.metadata.provider}`);
      if (event.metadata?.model) parts.push(`Model: ${event.metadata.model}`);
      return parts.join(". ");
    }, [event]);

    return (
      <article
        ref={ref as React.Ref<HTMLElement>}
        className={cn(
          "group relative flex gap-3 p-3 rounded-lg transition-all duration-300",
          "hover:bg-muted/50 border border-transparent hover:border-border",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          isNew && "animate-in slide-in-from-top-2 fade-in-0 duration-300",
          isFocused && "ring-2 ring-ring ring-offset-2",
          statusBgColor
        )}
        tabIndex={0}
        role="article"
        aria-label={accessibleDescription}
        aria-describedby={`event-${event.id}-details`}
        onKeyDown={onKeyDown}
      >
        {/* Icon */}
        <div
          className={cn(
            "flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center",
            "bg-background/80 border border-border/50"
          )}
          aria-hidden="true"
        >
          {renderActivityIcon(event.type, cn("h-5 w-5", statusColor))}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0" id={`event-${event.id}-details`}>
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <StatusIcon status={event.status} />
              <h3 className="font-medium text-sm truncate">{event.title}</h3>
            </div>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <time 
                    dateTime={event.timestamp}
                    className="text-xs text-muted-foreground whitespace-nowrap"
                  >
                    {formatActivityTime(event.timestamp)}
                  </time>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{new Date(event.timestamp).toLocaleString()}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          {event.description && (
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
              {event.description}
            </p>
          )}

          {/* Progress bar for running events */}
          {event.status === "running" && event.progress !== undefined && (
            <div className="mt-2">
              <Progress 
                value={event.progress} 
                className="h-1.5" 
                aria-label={`Progress: ${event.progress}%`}
              />
              <span className="text-xs text-muted-foreground mt-0.5">
                {event.progress}% complete
              </span>
            </div>
          )}

          {/* Metadata badges */}
          <div className="flex flex-wrap gap-1.5 mt-2" role="list" aria-label="Event metadata">
            {event.duration_ms && (
              <Badge variant="outline" className="text-xs px-1.5 py-0" role="listitem">
                <Clock className="h-3 w-3 mr-1" aria-hidden="true" />
                <span className="sr-only">Duration: </span>
                {formatDuration(event.duration_ms)}
              </Badge>
            )}
            {event.metadata?.provider && (
              <Badge variant="outline" className="text-xs px-1.5 py-0" role="listitem">
                <Server className="h-3 w-3 mr-1" aria-hidden="true" />
                <span className="sr-only">Provider: </span>
                {event.metadata.provider}
              </Badge>
            )}
            {event.metadata?.model && (
              <Badge variant="outline" className="text-xs px-1.5 py-0" role="listitem">
                <Cpu className="h-3 w-3 mr-1" aria-hidden="true" />
                <span className="sr-only">Model: </span>
                {event.metadata.model}
              </Badge>
            )}
            {event.metadata?.technique_suite && (
              <Badge variant="outline" className="text-xs px-1.5 py-0" role="listitem">
                <Wand2 className="h-3 w-3 mr-1" aria-hidden="true" />
                <span className="sr-only">Technique: </span>
                {event.metadata.technique_suite}
              </Badge>
            )}
            {event.metadata?.success_rate !== undefined && (
              <Badge
                variant="outline"
                className={cn(
                  "text-xs px-1.5 py-0",
                  event.metadata.success_rate >= 50 ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"
                )}
                role="listitem"
              >
                <span className="sr-only">Success rate: </span>
                {event.metadata.success_rate}% success
              </Badge>
            )}
          </div>
        </div>
      </article>
    );
  }
);

ActivityEventItem.displayName = "ActivityEventItem";

// =============================================================================
// Activity Feed Stats Component
// =============================================================================

interface ActivityFeedStatsProps {
  stats: {
    total: number;
    byStatus: Record<ActivityStatus, number>;
    successRate: number;
    avgDuration: number;
  };
}

function ActivityFeedStats({ stats }: ActivityFeedStatsProps) {
  return (
    <section 
      className="grid grid-cols-4 gap-2 p-3 bg-muted/30 rounded-lg"
      aria-label="Activity statistics"
      role="region"
    >
      <div className="text-center">
        <div className="text-lg font-bold" aria-label={`${stats.total} total events`}>{stats.total}</div>
        <div className="text-xs text-muted-foreground">Total</div>
      </div>
      <div className="text-center">
        <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400" aria-label={`${stats.byStatus.success} successful events`}>
          {stats.byStatus.success}
        </div>
        <div className="text-xs text-muted-foreground">Success</div>
      </div>
      <div className="text-center">
        <div className="text-lg font-bold text-red-600 dark:text-red-400" aria-label={`${stats.byStatus.failed} failed events`}>
          {stats.byStatus.failed}
        </div>
        <div className="text-xs text-muted-foreground">Failed</div>
      </div>
      <div className="text-center">
        <div className="text-lg font-bold text-blue-600 dark:text-blue-400" aria-label={`${stats.byStatus.running} running events`}>
          {stats.byStatus.running}
        </div>
        <div className="text-xs text-muted-foreground">Running</div>
      </div>
    </section>
  );
}

// =============================================================================
// Empty State Component
// =============================================================================

function EmptyState() {
  return (
    <div 
      className="flex flex-col items-center justify-center py-12 text-muted-foreground"
      role="status"
      aria-label="No activity events"
    >
      <Activity className="h-12 w-12 mb-4 opacity-20" aria-hidden="true" />
      <p className="text-sm font-medium">No activity events yet</p>
      <p className="text-xs mt-1">Events will appear here in real-time</p>
    </div>
  );
}

// =============================================================================
// Main Activity Feed Component
// =============================================================================

export interface ActivityFeedProps {
  /** Enable demo mode with simulated events */
  demoMode?: boolean;
  /** Maximum events to display (for performance) */
  maxEvents?: number;
  /** Maximum height of the feed */
  maxHeight?: string;
  /** Show statistics panel */
  showStats?: boolean;
  /** Show filter controls */
  showFilters?: boolean;
  /** Enable auto-scroll by default */
  autoScroll?: boolean;
  /** Compact mode for sidebar */
  compact?: boolean;
  /** Show loading skeleton initially */
  isLoading?: boolean;
  /** Custom class name */
  className?: string;
}

function ActivityFeedInner({
  demoMode = true,
  maxEvents = 100,
  maxHeight = "500px",
  showStats = true,
  showFilters = true,
  autoScroll: initialAutoScroll = true,
  compact = false,
  isLoading = false,
  className,
}: ActivityFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const eventRefs = useRef<Map<string, HTMLElement>>(new Map());
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStatuses, setSelectedStatuses] = useState<ActivityStatus[]>([]);
  const [newEventIds, setNewEventIds] = useState<Set<string>>(new Set());
  const [focusedEventIndex, setFocusedEventIndex] = useState<number>(-1);
  const [liveAnnouncement, setLiveAnnouncement] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);

  const {
    events,
    stats,
    isConnected,
    autoScroll,
    setAutoScroll,
    clearEvents,
    updateFilter,
    isDemoMode,
  } = useActivityFeed({
    demoMode,
    autoConnect: true,
    maxEvents,
    onEvent: (event) => {
      setNewEventIds((prev) => new Set(prev).add(event.id));
      setLiveAnnouncement(`New activity: ${event.title}. Status: ${event.status}`);
      setTimeout(() => {
        setNewEventIds((prev) => {
          const next = new Set(prev);
          next.delete(event.id);
          return next;
        });
      }, 1000);
    },
  });

  useEffect(() => {
    setAutoScroll(initialAutoScroll);
  }, [initialAutoScroll, setAutoScroll]);

  useEffect(() => {
    updateFilter({
      search: searchQuery || undefined,
      statuses: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    });
  }, [searchQuery, selectedStatuses, updateFilter]);

  useEffect(() => {
    if (autoScroll && scrollRef.current && events.length > 0) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events.length, autoScroll]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    const eventCount = events.length;
    if (eventCount === 0) return;

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setFocusedEventIndex((prev) => Math.min(prev + 1, eventCount - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setFocusedEventIndex((prev) => Math.max(prev - 1, 0));
        break;
      case "Home":
        e.preventDefault();
        setFocusedEventIndex(0);
        break;
      case "End":
        e.preventDefault();
        setFocusedEventIndex(eventCount - 1);
        break;
      case "Escape":
        e.preventDefault();
        setFocusedEventIndex(-1);
        searchInputRef.current?.focus();
        break;
    }
  }, [events.length]);

  useEffect(() => {
    if (focusedEventIndex >= 0 && focusedEventIndex < events.length) {
      const eventId = events[focusedEventIndex].id;
      const element = eventRefs.current.get(eventId);
      element?.focus();
    }
  }, [focusedEventIndex, events]);

  const toggleStatus = useCallback((status: ActivityStatus) => {
    setSelectedStatuses((prev) =>
      prev.includes(status) ? prev.filter((s) => s !== status) : [...prev, status]
    );
  }, []);

  const handleClearEvents = useCallback(() => {
    clearEvents();
    setLiveAnnouncement("All events cleared");
  }, [clearEvents]);

  const statusOptions: ActivityStatus[] = ["success", "failed", "running", "pending", "warning", "info"];

  if (isLoading) {
    return <ActivityFeedSkeleton />;
  }

  return (
    <Card className={cn("flex flex-col", className)} role="region" aria-labelledby="activity-feed-title">
      <LiveRegion message={liveAnnouncement} />
      
      <CardHeader className={cn("pb-3", compact && "p-3")}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" aria-hidden="true" />
            <CardTitle className={cn(compact && "text-base")} id="activity-feed-title">
              Activity Feed
            </CardTitle>
            {isDemoMode && (
              <Badge variant="secondary" className="text-xs">
                Demo
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className={cn(
                      "flex items-center gap-1 px-2 py-1 rounded-full text-xs",
                      isConnected
                        ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                        : "bg-red-500/10 text-red-600 dark:text-red-400"
                    )}
                    role="status"
                    aria-label={isConnected ? "Connected to activity stream" : "Disconnected from activity stream"}
                  >
                    {isConnected ? (
                      <Wifi className="h-3 w-3" aria-hidden="true" />
                    ) : (
                      <WifiOff className="h-3 w-3" aria-hidden="true" />
                    )}
                    <span className="hidden sm:inline">
                      {isConnected ? "Live" : "Offline"}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{isConnected ? "Connected to activity stream" : "Disconnected"}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
        {!compact && (
          <CardDescription>
            Real-time events from jailbreak generation, fuzzing, and optimization operations
          </CardDescription>
        )}
      </CardHeader>

      <CardContent className={cn("flex-1 flex flex-col gap-3", compact && "p-3 pt-0")}>
        {showStats && !compact && <ActivityFeedStats stats={stats} />}

        {showFilters && (
          <div className="flex items-center gap-2" role="search" aria-label="Filter events">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" aria-hidden="true" />
              <Input
                ref={searchInputRef}
                type="search"
                placeholder="Search events..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 h-8 text-sm"
                aria-label="Search events"
              />
            </div>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="h-8 gap-1"
                  aria-label={`Filter by status${selectedStatuses.length > 0 ? `, ${selectedStatuses.length} selected` : ""}`}
                >
                  <Filter className="h-3.5 w-3.5" aria-hidden="true" />
                  <span className="hidden sm:inline">Filter</span>
                  {selectedStatuses.length > 0 && (
                    <Badge variant="secondary" className="ml-1 px-1 text-xs">
                      {selectedStatuses.length}
                    </Badge>
                  )}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-40">
                <DropdownMenuLabel>Filter by Status</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {statusOptions.map((status) => (
                  <DropdownMenuCheckboxItem
                    key={status}
                    checked={selectedStatuses.includes(status)}
                    onCheckedChange={() => toggleStatus(status)}
                    className="capitalize"
                  >
                    <StatusIcon status={status} />
                    <span className="ml-2">{status}</span>
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={autoScroll ? "default" : "outline"}
                    size="sm"
                    className="h-8 w-8 p-0"
                    onClick={() => setAutoScroll(!autoScroll)}
                    aria-label={autoScroll ? "Disable auto-scroll" : "Enable auto-scroll"}
                    aria-pressed={autoScroll}
                  >
                    {autoScroll ? (
                      <ArrowDown className="h-3.5 w-3.5" aria-hidden="true" />
                    ) : (
                      <Pause className="h-3.5 w-3.5" aria-hidden="true" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{autoScroll ? "Auto-scroll enabled" : "Auto-scroll paused"}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 w-8 p-0"
                    onClick={handleClearEvents}
                    aria-label="Clear all events"
                  >
                    <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Clear all events</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
{/* Events list */}
<ScrollArea
  ref={scrollRef}
  className="flex-1"
  style={{ maxHeight }}
>
  <div
    className="space-y-2 pr-4"
    role="feed"
    aria-label="Activity events"
    aria-busy={false}
    onKeyDown={handleKeyDown}
  >
    {events.length === 0 ? (
      <EmptyState />
    ) : (
      events.map((event, index) => (
        <ActivityEventItem
          key={event.id}
          ref={(el) => {
            if (el) {
              eventRefs.current.set(event.id, el);
            } else {
              eventRefs.current.delete(event.id);
            }
          }}
          event={event}
          isNew={newEventIds.has(event.id)}
          isFocused={focusedEventIndex === index}
          onKeyDown={handleKeyDown}
        />
      ))
    )}
  </div>
</ScrollArea>

{/* Keyboard navigation hint */}
<div className="sr-only" aria-live="polite">
  Use arrow keys to navigate between events. Press Escape to return to search.
</div>
</CardContent>
</Card>
);
}

// =============================================================================
// Exported Component with Error Boundary
// =============================================================================

export function ActivityFeed(props: ActivityFeedProps) {
return (
<ActivityFeedErrorBoundary>
<ActivityFeedInner {...props} />
</ActivityFeedErrorBoundary>
);
}

export default ActivityFeed;
        