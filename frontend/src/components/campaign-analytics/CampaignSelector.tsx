"use client";

import * as React from "react";
import {
  Search,
  ChevronDown,
  Check,
  X,
  Calendar,
  Target,
  AlertCircle,
  Loader2,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useCampaigns } from "@/lib/api/query/campaign-queries";
import type {
  CampaignSummary,
  CampaignStatusEnum,
} from "@/types/campaign-analytics";

// =============================================================================
// Types
// =============================================================================

/**
 * Selection mode for the campaign selector.
 */
export type SelectionMode = "single" | "multi";

/**
 * Props for the CampaignSelector component.
 */
export interface CampaignSelectorProps {
  /** Selected campaign ID (single mode) */
  value?: string | null;
  /** Selected campaign IDs (multi mode) */
  values?: string[];
  /** Callback when selection changes (single mode) */
  onChange?: (campaignId: string | null) => void;
  /** Callback when selection changes (multi mode) */
  onMultiChange?: (campaignIds: string[]) => void;
  /** Selection mode */
  mode?: SelectionMode;
  /** Maximum selections in multi mode (default: 4 for comparison) */
  maxSelections?: number;
  /** Placeholder text when nothing selected */
  placeholder?: string;
  /** Disabled state */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Filter campaigns by status */
  statusFilter?: CampaignStatusEnum[];
  /** Exclude specific campaign IDs */
  excludeIds?: string[];
  /** Show only completed campaigns */
  completedOnly?: boolean;
  /** Custom empty state message */
  emptyMessage?: string;
  /** Aria label */
  ariaLabel?: string;
  /** Width of the popover */
  popoverWidth?: string;
  /** Allow clearing selection */
  allowClear?: boolean;
}

/**
 * Props for individual campaign option items.
 */
interface CampaignOptionProps {
  campaign: CampaignSummary;
  isSelected: boolean;
  isMultiMode: boolean;
  onSelect: () => void;
  disabled?: boolean;
}

// =============================================================================
// Status Badge Configuration
// =============================================================================

interface StatusConfig {
  label: string;
  variant: "default" | "secondary" | "destructive" | "outline";
  className: string;
  icon?: LucideIcon;
}

const STATUS_CONFIGS: Record<string, StatusConfig> = {
  draft: {
    label: "Draft",
    variant: "secondary",
    className: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  },
  running: {
    label: "Running",
    variant: "default",
    className: "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300",
    icon: Loader2,
  },
  paused: {
    label: "Paused",
    variant: "outline",
    className: "bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-300",
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
    icon: X,
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
 * Format relative time (e.g., "2 days ago").
 */
function formatRelativeTime(dateString?: string | null): string {
  if (!dateString) return "";
  try {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffMinutes < 1) return "just now";
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays === 1) return "yesterday";
    if (diffDays < 7) return `${diffDays}d ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
    return formatDate(dateString);
  } catch {
    return "";
  }
}

/**
 * Format success rate as percentage.
 */
function formatSuccessRate(rate?: number | null): string {
  if (rate === null || rate === undefined) return "—";
  return `${(rate * 100).toFixed(1)}%`;
}

/**
 * Filter campaigns based on search query.
 */
function filterCampaigns(
  campaigns: CampaignSummary[],
  query: string,
  excludeIds?: string[]
): CampaignSummary[] {
  const normalizedQuery = query.toLowerCase().trim();

  let filtered = campaigns;

  // Exclude specific IDs
  if (excludeIds && excludeIds.length > 0) {
    filtered = filtered.filter((c) => !excludeIds.includes(c.id));
  }

  // Apply search filter
  if (normalizedQuery) {
    filtered = filtered.filter((campaign) => {
      const nameMatch = campaign.name.toLowerCase().includes(normalizedQuery);
      const objectiveMatch = campaign.objective.toLowerCase().includes(normalizedQuery);
      const providerMatch = campaign.target_provider?.toLowerCase().includes(normalizedQuery);
      const tagMatch = campaign.tags?.some((tag) =>
        tag.toLowerCase().includes(normalizedQuery)
      );
      return nameMatch || objectiveMatch || providerMatch || tagMatch;
    });
  }

  return filtered;
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Status badge for campaign status.
 */
function CampaignStatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIGS[status] || STATUS_CONFIGS.draft;
  const Icon = config.icon;

  return (
    <Badge
      variant={config.variant}
      className={cn(
        "text-[10px] px-1.5 py-0 font-medium shrink-0",
        config.className
      )}
    >
      {Icon && (
        <Icon
          className={cn(
            "h-2.5 w-2.5 mr-0.5",
            status === "running" && "animate-spin"
          )}
        />
      )}
      {config.label}
    </Badge>
  );
}

/**
 * Loading skeleton for campaign list.
 */
function CampaignSelectorSkeleton() {
  return (
    <div className="space-y-2 p-1">
      {[1, 2, 3].map((i) => (
        <div key={i} className="flex items-center gap-2 p-2 rounded-md">
          <Skeleton className="h-4 w-4 rounded-sm" />
          <div className="flex-1 space-y-1.5">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-48" />
          </div>
          <Skeleton className="h-5 w-16 rounded-full" />
        </div>
      ))}
    </div>
  );
}

/**
 * Empty state when no campaigns found.
 */
function CampaignEmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
      <Target className="h-8 w-8 text-muted-foreground/50 mb-2" />
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}

/**
 * Individual campaign option in the list.
 */
function CampaignOption({
  campaign,
  isSelected,
  isMultiMode,
  onSelect,
  disabled,
}: CampaignOptionProps) {
  return (
    <button
      type="button"
      role="option"
      aria-selected={isSelected}
      disabled={disabled}
      onClick={onSelect}
      className={cn(
        "w-full flex items-start gap-3 p-2.5 rounded-md text-left transition-colors",
        "hover:bg-accent focus:bg-accent focus:outline-none",
        isSelected && "bg-accent",
        disabled && "opacity-50 cursor-not-allowed hover:bg-transparent"
      )}
    >
      {/* Selection indicator */}
      {isMultiMode ? (
        <Checkbox
          checked={isSelected}
          disabled={disabled}
          className="mt-0.5 shrink-0"
          aria-hidden="true"
        />
      ) : (
        <div
          className={cn(
            "mt-0.5 h-4 w-4 shrink-0 flex items-center justify-center",
            isSelected ? "text-primary" : "text-transparent"
          )}
        >
          <Check className="h-4 w-4" />
        </div>
      )}

      {/* Campaign info */}
      <div className="flex-1 min-w-0 space-y-1">
        {/* Name and status */}
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm truncate">{campaign.name}</span>
          <CampaignStatusBadge status={campaign.status} />
        </div>

        {/* Objective (truncated) */}
        <p className="text-xs text-muted-foreground line-clamp-1">
          {campaign.objective}
        </p>

        {/* Metadata row */}
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          {/* Date */}
          <span className="flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            {formatRelativeTime(campaign.created_at)}
          </span>

          {/* Provider */}
          {campaign.target_provider && (
            <span className="flex items-center gap-1">
              <Target className="h-3 w-3" />
              {campaign.target_provider}
            </span>
          )}

          {/* Success rate */}
          {campaign.success_rate !== null && campaign.success_rate !== undefined && (
            <span
              className={cn(
                "font-medium",
                campaign.success_rate >= 0.7
                  ? "text-green-600 dark:text-green-400"
                  : campaign.success_rate >= 0.4
                    ? "text-yellow-600 dark:text-yellow-400"
                    : "text-red-600 dark:text-red-400"
              )}
            >
              {formatSuccessRate(campaign.success_rate)}
            </span>
          )}

          {/* Attempts count */}
          {campaign.total_attempts > 0 && (
            <span>
              {campaign.total_attempts.toLocaleString()} attempts
            </span>
          )}
        </div>
      </div>
    </button>
  );
}

/**
 * Selected campaign chip for multi-select mode.
 */
function SelectedCampaignChip({
  campaign,
  onRemove,
}: {
  campaign: CampaignSummary;
  onRemove: () => void;
}) {
  return (
    <Badge
      variant="secondary"
      className="gap-1 pr-1 pl-2 py-0.5 max-w-[150px]"
    >
      <span className="truncate text-xs">{campaign.name}</span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        className="shrink-0 rounded-full p-0.5 hover:bg-muted-foreground/20 transition-colors"
        aria-label={`Remove ${campaign.name}`}
      >
        <X className="h-3 w-3" />
      </button>
    </Badge>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * CampaignSelector Component
 *
 * A searchable dropdown/combobox for selecting campaigns with:
 * - Real-time search filtering
 * - Status badges with color coding
 * - Date and metadata display
 * - Single and multi-select modes
 * - Support for comparison feature (max 4 selections)
 *
 * @example Single select mode
 * ```tsx
 * <CampaignSelector
 *   value={selectedCampaignId}
 *   onChange={(id) => setSelectedCampaignId(id)}
 *   placeholder="Select a campaign"
 * />
 * ```
 *
 * @example Multi-select mode for comparison
 * ```tsx
 * <CampaignSelector
 *   mode="multi"
 *   values={comparisonIds}
 *   onMultiChange={(ids) => setComparisonIds(ids)}
 *   maxSelections={4}
 *   placeholder="Select campaigns to compare"
 * />
 * ```
 *
 * @accessibility
 * - Full keyboard navigation support
 * - ARIA combobox pattern implementation
 * - Screen reader friendly with proper labels
 * - Focus management for dropdown
 */
export function CampaignSelector({
  value,
  values = [],
  onChange,
  onMultiChange,
  mode = "single",
  maxSelections = 4,
  placeholder = "Select campaign...",
  disabled = false,
  className,
  statusFilter,
  excludeIds,
  completedOnly = false,
  emptyMessage = "No campaigns found",
  ariaLabel = "Campaign selector",
  popoverWidth = "400px",
  allowClear = true,
}: CampaignSelectorProps) {
  const [open, setOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const searchInputRef = React.useRef<HTMLInputElement>(null);

  // Fetch campaigns
  const {
    data: campaignsResponse,
    isLoading,
    isError,
    error,
  } = useCampaigns({
    pageSize: 100, // Get more campaigns for selector
    filters: statusFilter
      ? { status: statusFilter }
      : completedOnly
        ? { status: ["completed" as CampaignStatusEnum] }
        : undefined,
  });

  const campaigns = React.useMemo(() => {
    return campaignsResponse?.items || [];
  }, [campaignsResponse]);

  // Filter campaigns based on search and exclusions
  const filteredCampaigns = React.useMemo(() => {
    return filterCampaigns(campaigns, search, excludeIds);
  }, [campaigns, search, excludeIds]);

  // Get selected campaigns for display
  const selectedCampaigns = React.useMemo(() => {
    if (mode === "single") {
      return campaigns.filter((c) => c.id === value);
    }
    return campaigns.filter((c) => values.includes(c.id));
  }, [campaigns, mode, value, values]);

  // Handle selection in single mode
  const handleSingleSelect = React.useCallback(
    (campaignId: string) => {
      if (onChange) {
        onChange(campaignId === value ? null : campaignId);
      }
      setOpen(false);
      setSearch("");
    },
    [onChange, value]
  );

  // Handle selection in multi mode
  const handleMultiSelect = React.useCallback(
    (campaignId: string) => {
      if (!onMultiChange) return;

      const isCurrentlySelected = values.includes(campaignId);

      if (isCurrentlySelected) {
        // Remove from selection
        onMultiChange(values.filter((id) => id !== campaignId));
      } else {
        // Add to selection (respecting max limit)
        if (values.length < maxSelections) {
          onMultiChange([...values, campaignId]);
        }
      }
    },
    [onMultiChange, values, maxSelections]
  );

  // Handle removing a selection
  const handleRemoveSelection = React.useCallback(
    (campaignId: string) => {
      if (mode === "single" && onChange) {
        onChange(null);
      } else if (mode === "multi" && onMultiChange) {
        onMultiChange(values.filter((id) => id !== campaignId));
      }
    },
    [mode, onChange, onMultiChange, values]
  );

  // Handle clearing all selections
  const handleClearAll = React.useCallback(() => {
    if (mode === "single" && onChange) {
      onChange(null);
    } else if (mode === "multi" && onMultiChange) {
      onMultiChange([]);
    }
  }, [mode, onChange, onMultiChange]);

  // Focus search input when popover opens
  React.useEffect(() => {
    if (open && searchInputRef.current) {
      // Small delay to ensure popover is rendered
      const timer = setTimeout(() => {
        searchInputRef.current?.focus();
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [open]);

  // Check if a campaign can be selected (for multi-mode max limit)
  const canSelectMore = mode === "multi" ? values.length < maxSelections : true;

  // Build trigger label
  const triggerLabel = React.useMemo(() => {
    if (mode === "single") {
      return selectedCampaigns[0]?.name || placeholder;
    }
    if (values.length === 0) {
      return placeholder;
    }
    return `${values.length} campaign${values.length > 1 ? "s" : ""} selected`;
  }, [mode, selectedCampaigns, values.length, placeholder]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          aria-label={ariaLabel}
          aria-haspopup="listbox"
          disabled={disabled}
          className={cn(
            "justify-between font-normal",
            mode === "multi" && values.length > 0 && "h-auto min-h-9 py-1.5",
            !value && mode === "single" && "text-muted-foreground",
            className
          )}
        >
          <div className="flex-1 flex flex-wrap gap-1 items-center text-left">
            {/* Multi-select: Show chips */}
            {mode === "multi" && selectedCampaigns.length > 0 ? (
              selectedCampaigns.map((campaign) => (
                <SelectedCampaignChip
                  key={campaign.id}
                  campaign={campaign}
                  onRemove={() => handleRemoveSelection(campaign.id)}
                />
              ))
            ) : (
              <span className={cn(!value && mode === "single" && "text-muted-foreground")}>
                {triggerLabel}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1 shrink-0 ml-2">
            {/* Clear button */}
            {allowClear && selectedCampaigns.length > 0 && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleClearAll();
                }}
                className="rounded-full p-0.5 hover:bg-muted transition-colors"
                aria-label="Clear selection"
              >
                <X className="h-3.5 w-3.5 text-muted-foreground" />
              </button>
            )}
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform duration-200",
                open && "rotate-180"
              )}
            />
          </div>
        </Button>
      </PopoverTrigger>

      <PopoverContent
        className="p-0"
        style={{ width: popoverWidth }}
        align="start"
        sideOffset={4}
      >
        {/* Search input */}
        <div className="flex items-center gap-2 px-3 py-2 border-b">
          <Search className="h-4 w-4 text-muted-foreground shrink-0" />
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search campaigns..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 text-sm bg-transparent border-none outline-none placeholder:text-muted-foreground"
            aria-label="Search campaigns"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              className="rounded-full p-0.5 hover:bg-muted transition-colors"
              aria-label="Clear search"
            >
              <X className="h-3.5 w-3.5 text-muted-foreground" />
            </button>
          )}
        </div>

        {/* Multi-select info bar */}
        {mode === "multi" && (
          <div className="flex items-center justify-between px-3 py-1.5 bg-muted/50 text-xs text-muted-foreground border-b">
            <span>
              {values.length} of {maxSelections} selected
            </span>
            {values.length > 0 && (
              <button
                type="button"
                onClick={handleClearAll}
                className="text-primary hover:underline"
              >
                Clear all
              </button>
            )}
          </div>
        )}

        {/* Campaign list */}
        <ScrollArea className="max-h-[300px]">
          <div role="listbox" aria-label="Campaigns" className="p-1">
            {isLoading ? (
              <CampaignSelectorSkeleton />
            ) : isError ? (
              <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
                <AlertCircle className="h-8 w-8 text-destructive/50 mb-2" />
                <p className="text-sm text-destructive">
                  Failed to load campaigns
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {error instanceof Error ? error.message : "Unknown error"}
                </p>
              </div>
            ) : filteredCampaigns.length === 0 ? (
              <CampaignEmptyState
                message={search ? `No campaigns match "${search}"` : emptyMessage}
              />
            ) : (
              filteredCampaigns.map((campaign) => {
                const isSelected =
                  mode === "single"
                    ? campaign.id === value
                    : values.includes(campaign.id);
                const isDisabled =
                  mode === "multi" && !isSelected && !canSelectMore;

                return (
                  <CampaignOption
                    key={campaign.id}
                    campaign={campaign}
                    isSelected={isSelected}
                    isMultiMode={mode === "multi"}
                    disabled={isDisabled}
                    onSelect={() =>
                      mode === "single"
                        ? handleSingleSelect(campaign.id)
                        : handleMultiSelect(campaign.id)
                    }
                  />
                );
              })
            )}
          </div>
        </ScrollArea>

        {/* Footer with count */}
        {!isLoading && !isError && filteredCampaigns.length > 0 && (
          <div className="px-3 py-1.5 border-t text-xs text-muted-foreground text-center">
            {filteredCampaigns.length} campaign
            {filteredCampaigns.length !== 1 ? "s" : ""} available
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

// =============================================================================
// Convenience Variants
// =============================================================================

/**
 * CampaignSelectorSingle - Pre-configured for single selection.
 */
export function CampaignSelectorSingle({
  value,
  onChange,
  ...props
}: Omit<CampaignSelectorProps, "mode" | "values" | "onMultiChange">) {
  return (
    <CampaignSelector
      mode="single"
      value={value}
      onChange={onChange}
      {...props}
    />
  );
}

/**
 * CampaignSelectorMulti - Pre-configured for multi-selection (comparison).
 */
export function CampaignSelectorMulti({
  values,
  onMultiChange,
  maxSelections = 4,
  ...props
}: Omit<CampaignSelectorProps, "mode" | "value" | "onChange">) {
  return (
    <CampaignSelector
      mode="multi"
      values={values}
      onMultiChange={onMultiChange}
      maxSelections={maxSelections}
      placeholder={props.placeholder || "Select campaigns to compare..."}
      {...props}
    />
  );
}

/**
 * CampaignComparisonSelector - Specialized for comparison feature.
 * Only shows completed campaigns and limits to 4 selections.
 */
export function CampaignComparisonSelector({
  values,
  onMultiChange,
  ...props
}: Omit<
  CampaignSelectorProps,
  "mode" | "value" | "onChange" | "completedOnly" | "maxSelections"
>) {
  return (
    <CampaignSelector
      mode="multi"
      values={values}
      onMultiChange={onMultiChange}
      completedOnly
      maxSelections={4}
      placeholder="Select 2-4 campaigns to compare..."
      emptyMessage="No completed campaigns available for comparison"
      {...props}
    />
  );
}

export default CampaignSelector;
