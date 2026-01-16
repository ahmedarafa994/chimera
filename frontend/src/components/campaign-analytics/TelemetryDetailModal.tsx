"use client";

import * as React from "react";
import {
  ChevronLeft,
  ChevronRight,
  Clock,
  Zap,
  Check,
  X,
  AlertCircle,
  Timer,
  Sparkles,
  Shield,
  Target,
  Loader2,
  Copy,
  ArrowRight,
  Tag,
  MessageSquare,
  Code2,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useTelemetryEventDetail } from "@/lib/api/query/campaign-queries";
import type {
  TelemetryEventDetail,
  TelemetryEventSummary,
  ExecutionStatusEnum,
} from "@/types/campaign-analytics";

// =============================================================================
// Types
// =============================================================================

/**
 * Navigation direction for event switching.
 */
export type NavigationDirection = "prev" | "next";

/**
 * Tab identifiers for the detail modal.
 */
export type DetailTab = "prompt" | "response" | "timing" | "quality";

/**
 * Props for the TelemetryDetailModal component.
 */
export interface TelemetryDetailModalProps {
  /** Whether the modal is open */
  open: boolean;
  /** Callback when the modal is closed */
  onOpenChange: (open: boolean) => void;
  /** Campaign ID for fetching event details */
  campaignId: string | null;
  /** Event ID to display details for */
  eventId: string | null;
  /** All event summaries for navigation (optional) */
  events?: TelemetryEventSummary[];
  /** Callback when navigating to a different event */
  onNavigate?: (direction: NavigationDirection, newEventId: string) => void;
  /** Default tab to show */
  defaultTab?: DetailTab;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Props for the TimingBreakdown component.
 */
interface TimingBreakdownProps {
  event: TelemetryEventDetail;
  className?: string;
}

/**
 * Props for the QualityScores component.
 */
interface QualityScoresProps {
  event: TelemetryEventDetail;
  className?: string;
}

/**
 * Props for the PromptDisplay component.
 */
interface PromptDisplayProps {
  event: TelemetryEventDetail;
  className?: string;
}

/**
 * Props for the ResponseDisplay component.
 */
interface ResponseDisplayProps {
  event: TelemetryEventDetail;
  className?: string;
}

// =============================================================================
// Status Configuration
// =============================================================================

interface StatusConfig {
  label: string;
  icon: LucideIcon;
  className: string;
  bgClassName: string;
}

const STATUS_CONFIGS: Record<string, StatusConfig> = {
  success: {
    label: "Success",
    icon: Check,
    className: "text-green-600 dark:text-green-400",
    bgClassName: "bg-green-100 dark:bg-green-900/30",
  },
  partial_success: {
    label: "Partial Success",
    icon: AlertCircle,
    className: "text-amber-600 dark:text-amber-400",
    bgClassName: "bg-amber-100 dark:bg-amber-900/30",
  },
  failure: {
    label: "Failed",
    icon: X,
    className: "text-red-600 dark:text-red-400",
    bgClassName: "bg-red-100 dark:bg-red-900/30",
  },
  timeout: {
    label: "Timeout",
    icon: Timer,
    className: "text-orange-600 dark:text-orange-400",
    bgClassName: "bg-orange-100 dark:bg-orange-900/30",
  },
  pending: {
    label: "Pending",
    icon: Loader2,
    className: "text-blue-600 dark:text-blue-400",
    bgClassName: "bg-blue-100 dark:bg-blue-900/30",
  },
  skipped: {
    label: "Skipped",
    icon: AlertCircle,
    className: "text-slate-600 dark:text-slate-400",
    bgClassName: "bg-slate-100 dark:bg-slate-800",
  },
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format duration in milliseconds to human-readable string.
 */
function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${ms.toFixed(0)}ms`;
}

/**
 * Format a timestamp to a readable date/time string.
 */
function formatTimestamp(timestamp: string | null | undefined): string {
  if (!timestamp) return "—";
  try {
    const date = new Date(timestamp);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "—";
  }
}

/**
 * Format a score as a percentage.
 */
function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined) return "—";
  return `${(score * 100).toFixed(1)}%`;
}

/**
 * Get score color class based on value.
 */
function getScoreColorClass(score: number | null | undefined): string {
  if (score === null || score === undefined) return "text-muted-foreground";
  if (score >= 0.8) return "text-green-600 dark:text-green-400";
  if (score >= 0.6) return "text-blue-600 dark:text-blue-400";
  if (score >= 0.4) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

/**
 * Truncate text to a maximum length.
 */
function truncateText(text: string | null | undefined, maxLength: number): string {
  if (!text) return "";
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength)}...`;
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Status badge for the event.
 */
export function StatusBadge({ status }: { status: ExecutionStatusEnum }) {
  const config = STATUS_CONFIGS[status] || STATUS_CONFIGS.pending;
  const Icon = config.icon;

  return (
    <Badge
      variant="secondary"
      className={cn(
        "gap-1 font-medium",
        config.className,
        config.bgClassName
      )}
    >
      <Icon className="size-3" />
      {config.label}
    </Badge>
  );
}

/**
 * Copy button for text content.
 */
export function CopyButton({
  text,
  label = "Copy",
  className,
}: {
  text: string;
  label?: string;
  className?: string;
}) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Handle copy failure silently
    }
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className={cn("h-7 gap-1.5 px-2 text-xs", className)}
            onClick={handleCopy}
          >
            {copied ? (
              <>
                <Check className="size-3" />
                Copied
              </>
            ) : (
              <>
                <Copy className="size-3" />
                {label}
              </>
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{copied ? "Copied to clipboard!" : "Copy to clipboard"}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Metadata row component for consistent display.
 */
export function MetadataRow({
  icon: Icon,
  label,
  value,
  valueClassName,
}: {
  icon?: LucideIcon;
  label: string;
  value: React.ReactNode;
  valueClassName?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        {Icon && <Icon className="size-4" />}
        <span>{label}</span>
      </div>
      <span className={cn("text-sm font-medium", valueClassName)}>{value}</span>
    </div>
  );
}

/**
 * Score card for quality metrics.
 */
export function ScoreCard({
  label,
  score,
  description,
  icon: Icon,
}: {
  label: string;
  score: number | null | undefined;
  description?: string;
  icon?: LucideIcon;
}) {
  const colorClass = getScoreColorClass(score);

  return (
    <div className="flex flex-col gap-1 rounded-lg border bg-card p-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {Icon && <Icon className="size-3" />}
        <span>{label}</span>
      </div>
      <div className={cn("text-2xl font-bold", colorClass)}>
        {formatScore(score)}
      </div>
      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}
    </div>
  );
}

/**
 * Prompt display component with original and transformed prompts.
 */
export function PromptDisplay({ event, className }: PromptDisplayProps) {
  const [showTransformed, setShowTransformed] = React.useState(true);

  const hasTransformedPrompt = !!event.transformed_prompt;

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      {/* Toggle for original vs transformed */}
      {hasTransformedPrompt && (
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">View:</span>
          <div className="flex rounded-lg border p-1">
            <Button
              variant={!showTransformed ? "secondary" : "ghost"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => setShowTransformed(false)}
            >
              Original
            </Button>
            <Button
              variant={showTransformed ? "secondary" : "ghost"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => setShowTransformed(true)}
            >
              Transformed
            </Button>
          </div>
        </div>
      )}

      {/* Prompt content */}
      <div className="relative">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-muted-foreground">
            {showTransformed && hasTransformedPrompt ? "Transformed Prompt" : "Original Prompt"}
          </span>
          <CopyButton
            text={
              showTransformed && hasTransformedPrompt
                ? event.transformed_prompt!
                : event.original_prompt
            }
          />
        </div>
        <ScrollArea className="h-[300px] rounded-lg border bg-muted/30 p-4">
          <pre className="whitespace-pre-wrap text-sm font-mono">
            {showTransformed && hasTransformedPrompt
              ? event.transformed_prompt
              : event.original_prompt}
          </pre>
        </ScrollArea>
      </div>

      {/* Applied techniques */}
      {event.applied_techniques && event.applied_techniques.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium text-muted-foreground">
            Applied Techniques
          </span>
          <div className="flex flex-wrap gap-1.5">
            {event.applied_techniques.map((technique, index) => (
              <Badge key={index} variant="outline" className="text-xs">
                <Tag className="mr-1 size-3" />
                {technique}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Transformation arrow visualization */}
      {hasTransformedPrompt && (
        <div className="flex items-center justify-center gap-4 py-2">
          <div className="flex flex-col items-center gap-1">
            <MessageSquare className="size-5 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Original</span>
          </div>
          <ArrowRight className="size-5 text-blue-500" />
          <div className="flex flex-col items-center gap-1">
            <Code2 className="size-5 text-blue-500" />
            <span className="text-xs text-blue-500">Transformed</span>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Response display component.
 */
export function ResponseDisplay({ event, className }: ResponseDisplayProps) {
  return (
    <div className={cn("flex flex-col gap-4", className)}>
      {/* Response content */}
      <div className="relative">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-muted-foreground">
            Model Response
          </span>
          {event.response_text && <CopyButton text={event.response_text} />}
        </div>
        <ScrollArea className="h-[300px] rounded-lg border bg-muted/30 p-4">
          {event.response_text ? (
            <pre className="whitespace-pre-wrap text-sm font-mono">
              {event.response_text}
            </pre>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              No response text available
            </div>
          )}
        </ScrollArea>
      </div>

      {/* Bypass indicators */}
      {event.bypass_indicators && event.bypass_indicators.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium text-muted-foreground">
            Bypass Indicators
          </span>
          <div className="flex flex-wrap gap-1.5">
            {event.bypass_indicators.map((indicator, index) => (
              <Badge
                key={index}
                variant="secondary"
                className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300 text-xs"
              >
                <Check className="mr-1 size-3" />
                {indicator}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Safety trigger warning */}
      {event.safety_trigger_detected && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-900/20 p-3">
          <Shield className="size-5 text-amber-600 dark:text-amber-400" />
          <div className="flex flex-col">
            <span className="text-sm font-medium text-amber-700 dark:text-amber-300">
              Safety Trigger Detected
            </span>
            <span className="text-xs text-amber-600 dark:text-amber-400">
              The model's safety mechanisms were triggered during this execution
            </span>
          </div>
        </div>
      )}

      {/* Error information */}
      {(event.error_message || event.error_code) && (
        <div className="flex flex-col gap-2 rounded-lg border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20 p-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="size-5 text-red-600 dark:text-red-400" />
            <span className="text-sm font-medium text-red-700 dark:text-red-300">
              Error {event.error_code && `(${event.error_code})`}
            </span>
          </div>
          {event.error_message && (
            <p className="text-sm text-red-600 dark:text-red-400">
              {event.error_message}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Timing breakdown component.
 */
export function TimingBreakdown({ event, className }: TimingBreakdownProps) {
  const totalLatency = event.total_latency_ms || 0;
  const transformationTime = event.transformation_time_ms || 0;
  const executionTime = event.execution_time_ms || 0;
  const overheadTime = Math.max(0, totalLatency - transformationTime - executionTime);

  // Calculate percentages for the visual bar
  const transformPct = totalLatency > 0 ? (transformationTime / totalLatency) * 100 : 0;
  const executionPct = totalLatency > 0 ? (executionTime / totalLatency) * 100 : 0;
  const overheadPct = totalLatency > 0 ? (overheadTime / totalLatency) * 100 : 0;

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Total latency header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className="size-5 text-muted-foreground" />
          <span className="text-lg font-semibold">Total Latency</span>
        </div>
        <span className="text-2xl font-bold">
          {formatDuration(totalLatency)}
        </span>
      </div>

      {/* Visual timing bar */}
      <div className="flex flex-col gap-2">
        <div className="flex h-6 overflow-hidden rounded-full border">
          {transformPct > 0 && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className="bg-blue-500 transition-all hover:brightness-110"
                    style={{ width: `${transformPct}%` }}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Transformation: {formatDuration(transformationTime)}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          {executionPct > 0 && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className="bg-green-500 transition-all hover:brightness-110"
                    style={{ width: `${executionPct}%` }}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Execution: {formatDuration(executionTime)}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          {overheadPct > 0 && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className="bg-slate-300 dark:bg-slate-600 transition-all hover:brightness-110"
                    style={{ width: `${overheadPct}%` }}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Overhead: {formatDuration(overheadTime)}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="size-3 rounded-sm bg-blue-500" />
            <span>Transformation</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="size-3 rounded-sm bg-green-500" />
            <span>Execution</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="size-3 rounded-sm bg-slate-300 dark:bg-slate-600" />
            <span>Overhead</span>
          </div>
        </div>
      </div>

      <Separator />

      {/* Detailed timing metrics */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
            <Code2 className="size-4" />
            <span>Transformation Time</span>
          </div>
          <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {formatDuration(transformationTime)}
          </span>
          {totalLatency > 0 && (
            <span className="ml-2 text-sm text-muted-foreground">
              ({transformPct.toFixed(1)}%)
            </span>
          )}
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
            <Zap className="size-4" />
            <span>Execution Time</span>
          </div>
          <span className="text-2xl font-bold text-green-600 dark:text-green-400">
            {formatDuration(executionTime)}
          </span>
          {totalLatency > 0 && (
            <span className="ml-2 text-sm text-muted-foreground">
              ({executionPct.toFixed(1)}%)
            </span>
          )}
        </div>
      </div>

      <Separator />

      {/* Additional timing metadata */}
      <div className="flex flex-col gap-1">
        <MetadataRow
          icon={Clock}
          label="Created At"
          value={formatTimestamp(event.created_at)}
        />
        <Separator />
        <MetadataRow
          icon={Target}
          label="Sequence Number"
          value={`#${event.sequence_number}`}
        />
      </div>
    </div>
  );
}

/**
 * Quality scores component.
 */
export function QualityScores({ event, className }: QualityScoresProps) {
  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Main quality scores grid */}
      <div className="grid gap-4 sm:grid-cols-2">
        <ScoreCard
          label="Semantic Success"
          score={event.semantic_success_score}
          description="How well the response meets the semantic goal"
          icon={Target}
        />
        <ScoreCard
          label="Effectiveness"
          score={event.effectiveness_score}
          description="Overall effectiveness of the transformation"
          icon={Sparkles}
        />
        <ScoreCard
          label="Naturalness"
          score={event.naturalness_score}
          description="How natural the transformed prompt appears"
          icon={MessageSquare}
        />
        <ScoreCard
          label="Detectability"
          score={event.detectability_score}
          description="Likelihood of detection (lower is better)"
          icon={Shield}
        />
      </div>

      <Separator />

      {/* Token usage */}
      <div className="flex flex-col gap-4">
        <h4 className="text-sm font-medium text-muted-foreground">Token Usage</h4>
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border bg-card p-3">
            <span className="text-xs text-muted-foreground">Prompt Tokens</span>
            <p className="text-xl font-bold">{event.prompt_tokens?.toLocaleString() ?? "—"}</p>
          </div>
          <div className="rounded-lg border bg-card p-3">
            <span className="text-xs text-muted-foreground">Completion Tokens</span>
            <p className="text-xl font-bold">{event.completion_tokens?.toLocaleString() ?? "—"}</p>
          </div>
          <div className="rounded-lg border bg-card p-3">
            <span className="text-xs text-muted-foreground">Total Tokens</span>
            <p className="text-xl font-bold">{event.total_tokens?.toLocaleString() ?? "—"}</p>
          </div>
        </div>
      </div>

      <Separator />

      {/* Additional metadata */}
      <div className="flex flex-col gap-1">
        <MetadataRow
          icon={Zap}
          label="Potency Level"
          value={event.potency_level}
        />
        <Separator />
        <MetadataRow
          label="Provider"
          value={event.provider}
        />
        <Separator />
        <MetadataRow
          label="Model"
          value={event.model}
        />
        <Separator />
        <MetadataRow
          label="Technique Suite"
          value={event.technique_suite}
        />
      </div>
    </div>
  );
}

// =============================================================================
// Loading Skeleton
// =============================================================================

/**
 * Loading skeleton for the modal content.
 */
export function TelemetryDetailModalSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      {/* Header skeleton */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Skeleton className="h-6 w-24" />
          <Skeleton className="h-6 w-32" />
        </div>
        <Skeleton className="h-8 w-20" />
      </div>

      {/* Tabs skeleton */}
      <Skeleton className="h-10 w-full" />

      {/* Content skeleton */}
      <div className="space-y-4">
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-[200px] w-full" />
      </div>

      {/* Footer skeleton */}
      <div className="flex items-center justify-between pt-4">
        <Skeleton className="h-9 w-24" />
        <Skeleton className="h-9 w-24" />
      </div>
    </div>
  );
}

// =============================================================================
// Error State
// =============================================================================

/**
 * Error state for the modal.
 */
export function TelemetryDetailModalError({
  error,
  onRetry,
}: {
  error: string;
  onRetry?: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-12">
      <AlertCircle className="size-12 text-red-500" />
      <div className="text-center">
        <h4 className="text-lg font-semibold">Failed to load event details</h4>
        <p className="text-sm text-muted-foreground">{error}</p>
      </div>
      {onRetry && (
        <Button variant="outline" onClick={onRetry}>
          Try Again
        </Button>
      )}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * Modal component for displaying detailed telemetry for a single execution.
 *
 * Features:
 * - Full prompt display (original and transformed)
 * - Complete response text with syntax highlighting
 * - Visual timing breakdown with percentage bars
 * - Quality scores dashboard
 * - Navigation between executions
 * - Copy to clipboard functionality
 *
 * @example
 * ```tsx
 * <TelemetryDetailModal
 *   open={isOpen}
 *   onOpenChange={setIsOpen}
 *   campaignId={campaignId}
 *   eventId={selectedEventId}
 *   events={allEvents}
 *   onNavigate={(direction, newId) => setSelectedEventId(newId)}
 * />
 * ```
 */
export function TelemetryDetailModal({
  open,
  onOpenChange,
  campaignId,
  eventId,
  events,
  onNavigate,
  defaultTab = "prompt",
  className,
}: TelemetryDetailModalProps) {
  const [activeTab, setActiveTab] = React.useState<DetailTab>(defaultTab);

  // Fetch event details
  const {
    data: event,
    isLoading,
    error,
    refetch,
  } = useTelemetryEventDetail(campaignId, eventId, open && !!eventId);

  // Find current position in events list for navigation
  const currentIndex = React.useMemo(() => {
    if (!events || !eventId) return -1;
    return events.findIndex((e) => e.id === eventId);
  }, [events, eventId]);

  const canNavigatePrev = currentIndex > 0;
  const canNavigateNext = events ? currentIndex < events.length - 1 : false;

  // Handle navigation
  const handleNavigate = React.useCallback(
    (direction: NavigationDirection) => {
      if (!events || !onNavigate) return;

      const newIndex = direction === "prev" ? currentIndex - 1 : currentIndex + 1;
      if (newIndex >= 0 && newIndex < events.length) {
        onNavigate(direction, events[newIndex].id);
      }
    },
    [events, currentIndex, onNavigate]
  );

  // Keyboard navigation
  React.useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft" && canNavigatePrev) {
        e.preventDefault();
        handleNavigate("prev");
      } else if (e.key === "ArrowRight" && canNavigateNext) {
        e.preventDefault();
        handleNavigate("next");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, canNavigatePrev, canNavigateNext, handleNavigate]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          "max-w-4xl max-h-[90vh] overflow-hidden flex flex-col",
          className
        )}
      >
        {isLoading ? (
          <TelemetryDetailModalSkeleton />
        ) : error ? (
          <TelemetryDetailModalError
            error={error instanceof Error ? error.message : "Unknown error"}
            onRetry={() => refetch()}
          />
        ) : event ? (
          <>
            {/* Header */}
            <DialogHeader className="flex-shrink-0">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <DialogTitle className="text-lg font-semibold">
                    Execution #{event.sequence_number}
                  </DialogTitle>
                  <StatusBadge status={event.status} />
                </div>
                {events && events.length > 1 && (
                  <span className="text-sm text-muted-foreground">
                    {currentIndex + 1} of {events.length}
                  </span>
                )}
              </div>
              <DialogDescription>
                {event.technique_suite} • {event.provider}/{event.model} •{" "}
                {formatTimestamp(event.created_at)}
              </DialogDescription>
            </DialogHeader>

            {/* Tabs */}
            <Tabs
              value={activeTab}
              onValueChange={(v) => setActiveTab(v as DetailTab)}
              className="flex-1 flex flex-col min-h-0"
            >
              <TabsList className="grid w-full grid-cols-4 flex-shrink-0">
                <TabsTrigger value="prompt" className="gap-1.5">
                  <MessageSquare className="size-4" />
                  <span className="hidden sm:inline">Prompt</span>
                </TabsTrigger>
                <TabsTrigger value="response" className="gap-1.5">
                  <Sparkles className="size-4" />
                  <span className="hidden sm:inline">Response</span>
                </TabsTrigger>
                <TabsTrigger value="timing" className="gap-1.5">
                  <Clock className="size-4" />
                  <span className="hidden sm:inline">Timing</span>
                </TabsTrigger>
                <TabsTrigger value="quality" className="gap-1.5">
                  <Target className="size-4" />
                  <span className="hidden sm:inline">Quality</span>
                </TabsTrigger>
              </TabsList>

              <div className="flex-1 min-h-0 mt-4">
                <ScrollArea className="h-full max-h-[50vh]">
                  <TabsContent value="prompt" className="mt-0 px-1">
                    <PromptDisplay event={event} />
                  </TabsContent>

                  <TabsContent value="response" className="mt-0 px-1">
                    <ResponseDisplay event={event} />
                  </TabsContent>

                  <TabsContent value="timing" className="mt-0 px-1">
                    <TimingBreakdown event={event} />
                  </TabsContent>

                  <TabsContent value="quality" className="mt-0 px-1">
                    <QualityScores event={event} />
                  </TabsContent>
                </ScrollArea>
              </div>
            </Tabs>

            {/* Navigation footer */}
            {events && events.length > 1 && onNavigate && (
              <>
                <Separator className="flex-shrink-0" />
                <div className="flex items-center justify-between pt-2 flex-shrink-0">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!canNavigatePrev}
                    onClick={() => handleNavigate("prev")}
                    className="gap-1.5"
                  >
                    <ChevronLeft className="size-4" />
                    Previous
                  </Button>

                  <span className="text-xs text-muted-foreground">
                    Use ← → arrow keys to navigate
                  </span>

                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!canNavigateNext}
                    onClick={() => handleNavigate("next")}
                    className="gap-1.5"
                  >
                    Next
                    <ChevronRight className="size-4" />
                  </Button>
                </div>
              </>
            )}
          </>
        ) : (
          <div className="flex items-center justify-center py-12 text-muted-foreground">
            No event selected
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * Simple modal without navigation, just shows event details.
 */
export function SimpleTelemetryDetailModal({
  open,
  onOpenChange,
  campaignId,
  eventId,
  className,
}: Omit<TelemetryDetailModalProps, "events" | "onNavigate">) {
  return (
    <TelemetryDetailModal
      open={open}
      onOpenChange={onOpenChange}
      campaignId={campaignId}
      eventId={eventId}
      className={className}
    />
  );
}

/**
 * Compact modal variant with smaller sizing.
 */
export function CompactTelemetryDetailModal({
  open,
  onOpenChange,
  campaignId,
  eventId,
  events,
  onNavigate,
  className,
}: TelemetryDetailModalProps) {
  return (
    <TelemetryDetailModal
      open={open}
      onOpenChange={onOpenChange}
      campaignId={campaignId}
      eventId={eventId}
      events={events}
      onNavigate={onNavigate}
      className={cn("max-w-2xl", className)}
    />
  );
}

// =============================================================================
// Exports
// =============================================================================

export {
  StatusBadge,
  CopyButton,
  MetadataRow,
  ScoreCard,
  PromptDisplay,
  ResponseDisplay,
  TimingBreakdown,
  QualityScores,
  TelemetryDetailModalSkeleton,
  TelemetryDetailModalError,
};
