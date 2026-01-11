"use client";

import * as React from "react";
import {
  Target,
  Calendar,
  Clock,
  Play,
  Pause,
  StopCircle,
  RotateCcw,
  Download,
  Share2,
  BarChart3,
  ExternalLink,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  MoreHorizontal,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
  CardAction,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type {
  CampaignSummary,
  CampaignDetail,
  CampaignStatusEnum,
} from "@/types/campaign-analytics";

// =============================================================================
// Types
// =============================================================================

/**
 * Quick actions available for the campaign.
 */
export type CampaignQuickAction =
  | "view_details"
  | "view_analytics"
  | "export_csv"
  | "export_chart"
  | "share"
  | "copy_id"
  | "resume"
  | "pause"
  | "stop"
  | "retry";

/**
 * Props for the CampaignOverviewCard component.
 */
export interface CampaignOverviewCardProps {
  /** Campaign data to display */
  campaign: CampaignSummary | CampaignDetail | null | undefined;
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: string | null;
  /** Compact mode for smaller displays */
  compact?: boolean;
  /** Show techniques as list instead of badges */
  showTechniqueList?: boolean;
  /** Maximum number of techniques to show before truncating */
  maxTechniques?: number;
  /** Maximum number of tags to show before truncating */
  maxTags?: number;
  /** Callback when view details is clicked */
  onViewDetails?: (campaignId: string) => void;
  /** Callback when view analytics is clicked */
  onViewAnalytics?: (campaignId: string) => void;
  /** Callback when export CSV is clicked */
  onExportCSV?: (campaignId: string) => void;
  /** Callback when export chart is clicked */
  onExportChart?: (campaignId: string) => void;
  /** Callback when share is clicked */
  onShare?: (campaignId: string) => void;
  /** Callback when copy ID is clicked */
  onCopyId?: (campaignId: string) => void;
  /** Callback when resume is clicked (for paused campaigns) */
  onResume?: (campaignId: string) => void;
  /** Callback when pause is clicked (for running campaigns) */
  onPause?: (campaignId: string) => void;
  /** Callback when stop is clicked (for running campaigns) */
  onStop?: (campaignId: string) => void;
  /** Callback when retry is clicked (for failed campaigns) */
  onRetry?: (campaignId: string) => void;
  /** Additional CSS classes */
  className?: string;
  /** Show quick actions */
  showActions?: boolean;
  /** Show footer with metadata */
  showFooter?: boolean;
}

// =============================================================================
// Status Configuration
// =============================================================================

interface StatusConfig {
  label: string;
  variant: "default" | "secondary" | "destructive" | "outline";
  className: string;
  icon: LucideIcon;
  iconClassName?: string;
}

const STATUS_CONFIGS: Record<string, StatusConfig> = {
  draft: {
    label: "Draft",
    variant: "secondary",
    className: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    icon: Clock,
  },
  running: {
    label: "Running",
    variant: "default",
    className: "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300",
    icon: Loader2,
    iconClassName: "animate-spin",
  },
  paused: {
    label: "Paused",
    variant: "outline",
    className: "bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-300",
    icon: Pause,
  },
  completed: {
    label: "Completed",
    variant: "default",
    className: "bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300",
    icon: Check,
  },
  failed: {
    label: "Failed",
    variant: "destructive",
    className: "bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300",
    icon: AlertCircle,
  },
  cancelled: {
    label: "Cancelled",
    variant: "secondary",
    className: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
    icon: StopCircle,
  },
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format a date string for display.
 */
function formatDate(dateString?: string | null): string {
  if (!dateString) return "—";
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return "—";
  }
}

/**
 * Format date range for display.
 */
function formatDateRange(
  startDate?: string | null,
  endDate?: string | null
): string {
  const start = formatDate(startDate);
  const end = formatDate(endDate);

  if (start === "—" && end === "—") return "Not started";
  if (end === "—") return `Started ${start}`;
  return `${start} — ${end}`;
}

/**
 * Format duration in seconds to human-readable format.
 */
function formatDuration(seconds?: number | null): string {
  if (seconds === null || seconds === undefined) return "—";

  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
  }

  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
}

/**
 * Format success rate as percentage.
 */
function formatSuccessRate(rate?: number | null): string {
  if (rate === null || rate === undefined) return "—";
  return `${(rate * 100).toFixed(1)}%`;
}

/**
 * Truncate text with ellipsis.
 */
function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + "...";
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Status badge component.
 */
function StatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIGS[status] || STATUS_CONFIGS.draft;
  const Icon = config.icon;

  return (
    <Badge variant={config.variant} className={cn("gap-1", config.className)}>
      <Icon className={cn("h-3 w-3", config.iconClassName)} aria-hidden="true" />
      {config.label}
    </Badge>
  );
}

/**
 * Metadata row component.
 */
function MetadataRow({
  icon: Icon,
  label,
  value,
  className,
}: {
  icon: LucideIcon;
  label: string;
  value: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-2 text-sm", className)}>
      <Icon className="h-4 w-4 text-muted-foreground shrink-0" aria-hidden="true" />
      <span className="text-muted-foreground shrink-0">{label}:</span>
      <span className="font-medium truncate">{value}</span>
    </div>
  );
}

/**
 * Technique/tag badge list with truncation.
 */
function BadgeList({
  items,
  maxItems = 3,
  variant = "secondary",
  emptyText = "None",
}: {
  items: string[];
  maxItems?: number;
  variant?: "default" | "secondary" | "destructive" | "outline";
  emptyText?: string;
}) {
  if (!items || items.length === 0) {
    return <span className="text-sm text-muted-foreground">{emptyText}</span>;
  }

  const visibleItems = items.slice(0, maxItems);
  const remainingCount = items.length - maxItems;

  return (
    <div className="flex flex-wrap gap-1">
      {visibleItems.map((item) => (
        <Badge key={item} variant={variant} className="text-xs">
          {truncateText(item, 20)}
        </Badge>
      ))}
      {remainingCount > 0 && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="outline" className="text-xs cursor-help">
                +{remainingCount} more
              </Badge>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-xs">
              <div className="flex flex-wrap gap-1">
                {items.slice(maxItems).map((item) => (
                  <Badge key={item} variant={variant} className="text-xs">
                    {item}
                  </Badge>
                ))}
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}

/**
 * Quick action button component.
 */
function ActionButton({
  icon: Icon,
  label,
  onClick,
  disabled,
  variant = "outline",
  size = "sm",
  className,
}: {
  icon: LucideIcon;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link";
  size?: "default" | "sm" | "lg" | "icon";
  className?: string;
}) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant={variant}
            size={size}
            onClick={onClick}
            disabled={disabled}
            className={cn("gap-1.5", className)}
          >
            <Icon className="h-4 w-4" aria-hidden="true" />
            <span className="sr-only">{label}</span>
          </Button>
        </TooltipTrigger>
        <TooltipContent>{label}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// Loading Skeleton
// =============================================================================

/**
 * Loading skeleton for CampaignOverviewCard.
 */
export function CampaignOverviewCardSkeleton({
  className,
  compact,
}: {
  className?: string;
  compact?: boolean;
}) {
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-2 flex-1">
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-full max-w-md" />
          </div>
          <Skeleton className="h-6 w-20" />
        </div>
      </CardHeader>
      <CardContent className={cn("space-y-3", compact && "py-2")}>
        <div className="grid gap-3 sm:grid-cols-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-4 w-36" />
          <Skeleton className="h-4 w-28" />
        </div>
        <Separator className="my-3" />
        <div className="space-y-2">
          <Skeleton className="h-4 w-24" />
          <div className="flex gap-1">
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-5 w-20" />
            <Skeleton className="h-5 w-14" />
          </div>
        </div>
      </CardContent>
      <CardFooter className="pt-3 border-t">
        <div className="flex gap-2">
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
        </div>
      </CardFooter>
    </Card>
  );
}

// =============================================================================
// Error State
// =============================================================================

/**
 * Error state for CampaignOverviewCard.
 */
export function CampaignOverviewCardError({
  error,
  onRetry,
  className,
}: {
  error: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <Card className={cn("overflow-hidden border-red-500/30", className)}>
      <CardContent className="py-8">
        <div className="flex flex-col items-center justify-center text-center gap-3">
          <AlertCircle className="h-10 w-10 text-red-500" />
          <div className="space-y-1">
            <p className="font-medium text-foreground">Failed to load campaign</p>
            <p className="text-sm text-muted-foreground">{error}</p>
          </div>
          {onRetry && (
            <Button variant="outline" size="sm" onClick={onRetry} className="mt-2">
              <RotateCcw className="h-4 w-4 mr-1.5" />
              Retry
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Empty State
// =============================================================================

/**
 * Empty state when no campaign is selected.
 */
export function CampaignOverviewCardEmpty({
  message = "No campaign selected",
  description = "Select a campaign to view its details",
  className,
}: {
  message?: string;
  description?: string;
  className?: string;
}) {
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardContent className="py-8">
        <div className="flex flex-col items-center justify-center text-center gap-2">
          <Target className="h-10 w-10 text-muted-foreground/50" />
          <p className="font-medium text-muted-foreground">{message}</p>
          <p className="text-sm text-muted-foreground/70">{description}</p>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * CampaignOverviewCard Component
 *
 * Displays campaign metadata including name, objective, status, date range,
 * provider, and techniques applied. Includes quick action buttons for common
 * operations.
 *
 * @example
 * ```tsx
 * <CampaignOverviewCard
 *   campaign={campaign}
 *   onViewDetails={(id) => router.push(`/campaigns/${id}`)}
 *   onViewAnalytics={(id) => router.push(`/campaigns/${id}/analytics`)}
 *   onExportCSV={(id) => exportService.exportCSV(id)}
 * />
 * ```
 *
 * @accessibility
 * - Semantic structure with proper heading hierarchy
 * - ARIA labels for interactive elements
 * - Keyboard navigable buttons
 * - Screen reader friendly status badges
 */
export function CampaignOverviewCard({
  campaign,
  isLoading = false,
  error,
  compact = false,
  showTechniqueList = false,
  maxTechniques = 4,
  maxTags = 3,
  onViewDetails,
  onViewAnalytics,
  onExportCSV,
  onExportChart,
  onShare,
  onCopyId,
  onResume,
  onPause,
  onStop,
  onRetry,
  className,
  showActions = true,
  showFooter = true,
}: CampaignOverviewCardProps) {
  const [copiedId, setCopiedId] = React.useState(false);

  // Handle copy ID action
  const handleCopyId = React.useCallback(() => {
    if (!campaign?.id) return;

    navigator.clipboard.writeText(campaign.id).then(() => {
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
      onCopyId?.(campaign.id);
    });
  }, [campaign?.id, onCopyId]);

  // Loading state
  if (isLoading) {
    return <CampaignOverviewCardSkeleton className={className} compact={compact} />;
  }

  // Error state
  if (error) {
    return (
      <CampaignOverviewCardError
        error={error}
        onRetry={onRetry ? () => onRetry("") : undefined}
        className={className}
      />
    );
  }

  // Empty state
  if (!campaign) {
    return <CampaignOverviewCardEmpty className={className} />;
  }

  // Determine available actions based on status
  const canResume = campaign.status === "paused" && onResume;
  const canPause = campaign.status === "running" && onPause;
  const canStop = campaign.status === "running" && onStop;
  const canRetryFailed = campaign.status === "failed" && onRetry;

  return (
    <Card
      className={cn(
        "overflow-hidden transition-all duration-200 hover:shadow-md",
        className
      )}
    >
      {/* Header with Name and Status */}
      <CardHeader className={cn("pb-3", compact && "py-3")}>
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1 flex-1 min-w-0">
            <CardTitle className="text-lg font-semibold truncate">
              {campaign.name}
            </CardTitle>
            {campaign.description && !compact && (
              <CardDescription className="line-clamp-2">
                {campaign.description}
              </CardDescription>
            )}
          </div>
          <CardAction>
            <StatusBadge status={campaign.status} />
          </CardAction>
        </div>
      </CardHeader>

      {/* Content with Metadata */}
      <CardContent className={cn("space-y-4", compact && "py-2 space-y-3")}>
        {/* Objective */}
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Target className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span className="font-medium">Objective</span>
          </div>
          <p className="text-sm pl-6 line-clamp-2">{campaign.objective}</p>
        </div>

        {/* Metadata Grid */}
        <div className={cn("grid gap-2", compact ? "grid-cols-1" : "sm:grid-cols-2")}>
          {/* Date Range */}
          <MetadataRow
            icon={Calendar}
            label="Period"
            value={formatDateRange(campaign.started_at, campaign.completed_at)}
          />

          {/* Duration */}
          <MetadataRow
            icon={Clock}
            label="Duration"
            value={formatDuration(campaign.duration_seconds)}
          />

          {/* Provider */}
          {campaign.target_provider && (
            <MetadataRow
              icon={BarChart3}
              label="Provider"
              value={
                <span className="capitalize">
                  {campaign.target_provider}
                  {campaign.target_model && (
                    <span className="text-muted-foreground ml-1">
                      ({campaign.target_model})
                    </span>
                  )}
                </span>
              }
            />
          )}

          {/* Quick Stats */}
          {(campaign.total_attempts > 0 || campaign.success_rate !== null) && (
            <MetadataRow
              icon={Target}
              label="Results"
              value={
                <span>
                  {campaign.total_attempts} attempts
                  {campaign.success_rate !== null && (
                    <span className="text-muted-foreground ml-1">
                      ({formatSuccessRate(campaign.success_rate)} success)
                    </span>
                  )}
                </span>
              }
            />
          )}
        </div>

        {/* Separator */}
        <Separator />

        {/* Techniques */}
        {campaign.technique_suites && campaign.technique_suites.length > 0 && (
          <div className="space-y-2">
            <span className="text-sm font-medium text-muted-foreground">
              Techniques Applied
            </span>
            {showTechniqueList ? (
              <ul className="text-sm space-y-1 pl-4 list-disc text-foreground">
                {campaign.technique_suites.slice(0, maxTechniques).map((tech) => (
                  <li key={tech}>{tech}</li>
                ))}
                {campaign.technique_suites.length > maxTechniques && (
                  <li className="text-muted-foreground">
                    +{campaign.technique_suites.length - maxTechniques} more
                  </li>
                )}
              </ul>
            ) : (
              <BadgeList
                items={campaign.technique_suites}
                maxItems={maxTechniques}
                variant="secondary"
                emptyText="No techniques"
              />
            )}
          </div>
        )}

        {/* Tags */}
        {campaign.tags && campaign.tags.length > 0 && (
          <div className="space-y-2">
            <span className="text-sm font-medium text-muted-foreground">Tags</span>
            <BadgeList
              items={campaign.tags}
              maxItems={maxTags}
              variant="outline"
              emptyText="No tags"
            />
          </div>
        )}
      </CardContent>

      {/* Footer with Actions */}
      {showFooter && showActions && (
        <CardFooter className="pt-3 border-t flex items-center justify-between gap-2">
          {/* Primary Actions */}
          <div className="flex items-center gap-1">
            {onViewAnalytics && (
              <Button
                variant="default"
                size="sm"
                onClick={() => onViewAnalytics(campaign.id)}
                className="gap-1.5"
              >
                <BarChart3 className="h-4 w-4" aria-hidden="true" />
                Analytics
              </Button>
            )}
            {onViewDetails && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onViewDetails(campaign.id)}
                className="gap-1.5"
              >
                <ExternalLink className="h-4 w-4" aria-hidden="true" />
                Details
              </Button>
            )}
          </div>

          {/* Secondary Actions */}
          <div className="flex items-center gap-1">
            {/* Status-specific actions */}
            {canResume && (
              <ActionButton
                icon={Play}
                label="Resume Campaign"
                onClick={() => onResume(campaign.id)}
              />
            )}
            {canPause && (
              <ActionButton
                icon={Pause}
                label="Pause Campaign"
                onClick={() => onPause(campaign.id)}
              />
            )}
            {canStop && (
              <ActionButton
                icon={StopCircle}
                label="Stop Campaign"
                onClick={() => onStop(campaign.id)}
                variant="destructive"
              />
            )}
            {canRetryFailed && (
              <ActionButton
                icon={RotateCcw}
                label="Retry Campaign"
                onClick={() => onRetry(campaign.id)}
              />
            )}

            {/* Export Actions */}
            {onExportCSV && (
              <ActionButton
                icon={Download}
                label="Export CSV"
                onClick={() => onExportCSV(campaign.id)}
              />
            )}

            {/* More Actions Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <MoreHorizontal className="h-4 w-4" />
                  <span className="sr-only">More actions</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={handleCopyId}>
                  {copiedId ? (
                    <Check className="mr-2 h-4 w-4" />
                  ) : (
                    <Copy className="mr-2 h-4 w-4" />
                  )}
                  {copiedId ? "Copied!" : "Copy Campaign ID"}
                </DropdownMenuItem>
                {onExportChart && (
                  <DropdownMenuItem onClick={() => onExportChart(campaign.id)}>
                    <BarChart3 className="mr-2 h-4 w-4" />
                    Export Chart
                  </DropdownMenuItem>
                )}
                {onShare && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => onShare(campaign.id)}>
                      <Share2 className="mr-2 h-4 w-4" />
                      Share
                    </DropdownMenuItem>
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </CardFooter>
      )}
    </Card>
  );
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * Compact variant of CampaignOverviewCard.
 */
export function CompactCampaignOverviewCard(
  props: Omit<CampaignOverviewCardProps, "compact">
) {
  return <CampaignOverviewCard {...props} compact showFooter={false} />;
}

/**
 * Campaign card for list displays (minimal actions).
 */
export function CampaignListCard({
  campaign,
  onSelect,
  isSelected,
  className,
  ...props
}: Omit<CampaignOverviewCardProps, "showActions" | "showFooter"> & {
  onSelect?: (campaignId: string) => void;
  isSelected?: boolean;
}) {
  return (
    <Card
      className={cn(
        "overflow-hidden transition-all duration-200 cursor-pointer",
        "hover:shadow-md hover:border-primary/50",
        isSelected && "ring-2 ring-primary border-primary",
        className
      )}
      onClick={() => campaign && onSelect?.(campaign.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && campaign) {
          e.preventDefault();
          onSelect?.(campaign.id);
        }
      }}
    >
      <CampaignOverviewCard
        {...props}
        campaign={campaign}
        compact
        showActions={false}
        showFooter={false}
        className="border-0 shadow-none"
      />
    </Card>
  );
}

export default CampaignOverviewCard;
