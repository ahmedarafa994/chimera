"use client";

import * as React from "react";
import {
  Star,
  Copy,
  MoreVertical,
  Clock,
  CheckCircle2,
  Eye,
  Trash2,
  Edit3,
  ExternalLink,
  Shield,
  Target,
  Lock,
  Users,
  Globe,
  Tag,
  TrendingUp,
  Beaker,
  Archive,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
  CardDescription,
  CardAction,
} from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type {
  PromptTemplate,
  TechniqueType,
  VulnerabilityType,
  SharingLevel,
  TemplateStatus,
} from "@/types/prompt-library-types";
import {
  formatTechniqueType,
  formatVulnerabilityType,
  formatSharingLevel,
} from "@/types/prompt-library-types";

// =============================================================================
// Types & Interfaces
// =============================================================================

interface PromptTemplateCardProps {
  /** The prompt template data */
  template: PromptTemplate;
  /** Card display variant */
  variant?: "default" | "compact" | "featured";
  /** Whether the card is currently selected */
  isSelected?: boolean;
  /** Current user ID for permission checks */
  currentUserId?: string | null;
  /** Callback when clicking the card */
  onClick?: (template: PromptTemplate) => void;
  /** Callback when clicking the "Use Template" action */
  onUseTemplate?: (template: PromptTemplate) => void;
  /** Callback when clicking the "Copy" action */
  onCopy?: (template: PromptTemplate) => void;
  /** Callback when clicking the "Edit" action */
  onEdit?: (template: PromptTemplate) => void;
  /** Callback when clicking the "Delete" action */
  onDelete?: (template: PromptTemplate) => void;
  /** Callback when clicking the "View Details" action */
  onViewDetails?: (template: PromptTemplate) => void;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// Configuration
// =============================================================================

/** Configuration for sharing level badges */
const SHARING_LEVEL_CONFIG: Record<
  SharingLevel,
  { icon: React.ElementType; color: string; bgColor: string }
> = {
  private: {
    icon: Lock,
    color: "text-orange-600",
    bgColor: "bg-orange-50 dark:bg-orange-950/30",
  },
  team: {
    icon: Users,
    color: "text-blue-600",
    bgColor: "bg-blue-50 dark:bg-blue-950/30",
  },
  public: {
    icon: Globe,
    color: "text-green-600",
    bgColor: "bg-green-50 dark:bg-green-950/30",
  },
};

/** Configuration for template status badges */
const STATUS_CONFIG: Record<
  TemplateStatus,
  { icon: React.ElementType; color: string; bgColor: string; label: string }
> = {
  draft: {
    icon: Edit3,
    color: "text-yellow-600",
    bgColor: "bg-yellow-50 dark:bg-yellow-950/30",
    label: "Draft",
  },
  active: {
    icon: CheckCircle2,
    color: "text-green-600",
    bgColor: "bg-green-50 dark:bg-green-950/30",
    label: "Active",
  },
  archived: {
    icon: Archive,
    color: "text-gray-500",
    bgColor: "bg-gray-50 dark:bg-gray-800/50",
    label: "Archived",
  },
  deprecated: {
    icon: AlertTriangle,
    color: "text-red-600",
    bgColor: "bg-red-50 dark:bg-red-950/30",
    label: "Deprecated",
  },
};

/** Color palette for technique type badges */
const TECHNIQUE_COLORS: string[] = [
  "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
  "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300",
  "bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300",
  "bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300",
];

/** Color palette for vulnerability type badges */
const VULNERABILITY_COLORS: string[] = [
  "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
  "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
];

// =============================================================================
// Helper Components
// =============================================================================

/**
 * Star Rating Display Component
 */
function StarRating({
  rating,
  totalRatings,
  size = "default",
}: {
  rating: number;
  totalRatings: number;
  size?: "sm" | "default";
}) {
  const starSize = size === "sm" ? "h-3 w-3" : "h-4 w-4";
  const textSize = size === "sm" ? "text-xs" : "text-sm";

  // Render 5 stars with proper fill based on rating
  const renderStars = () => {
    const stars = [];
    for (let i = 1; i <= 5; i++) {
      const isFilled = i <= Math.floor(rating);
      const isHalfFilled = i === Math.ceil(rating) && rating % 1 !== 0;

      stars.push(
        <Star
          key={i}
          className={cn(
            starSize,
            isFilled
              ? "fill-yellow-400 text-yellow-400"
              : isHalfFilled
                ? "fill-yellow-400/50 text-yellow-400"
                : "text-muted-foreground/30"
          )}
          aria-hidden="true"
        />
      );
    }
    return stars;
  };

  return (
    <div
      className="flex items-center gap-1.5"
      aria-label={`Rating: ${rating.toFixed(1)} out of 5 stars based on ${totalRatings} ratings`}
    >
      <div className="flex items-center gap-0.5">{renderStars()}</div>
      <span className={cn("font-semibold text-foreground", textSize)}>
        {rating.toFixed(1)}
      </span>
      <span className={cn("text-muted-foreground", textSize)}>
        ({totalRatings.toLocaleString()})
      </span>
    </div>
  );
}

/**
 * Success Rate Badge Component
 */
function SuccessRateBadge({
  successRate,
  testCount,
}: {
  successRate: number | null;
  testCount: number;
}) {
  if (successRate === null || testCount === 0) {
    return (
      <Badge variant="outline" className="text-muted-foreground">
        <Beaker className="h-3 w-3 mr-1" aria-hidden="true" />
        Not tested
      </Badge>
    );
  }

  const percentage = (successRate * 100).toFixed(0);
  const colorClass =
    successRate >= 0.7
      ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
      : successRate >= 0.4
        ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300"
        : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge className={cn("border-0", colorClass)}>
            <TrendingUp className="h-3 w-3 mr-1" aria-hidden="true" />
            {percentage}%
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {percentage}% success rate ({testCount} tests)
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Metadata Badge Group Component
 */
function MetadataBadges({
  techniques,
  vulnerabilities,
  maxDisplay = 2,
}: {
  techniques: TechniqueType[];
  vulnerabilities: VulnerabilityType[];
  maxDisplay?: number;
}) {
  const displayTechniques = techniques.slice(0, maxDisplay);
  const remainingTechniques = techniques.length - maxDisplay;
  const displayVulnerabilities = vulnerabilities.slice(0, maxDisplay);
  const remainingVulnerabilities = vulnerabilities.length - maxDisplay;

  return (
    <div className="flex flex-wrap gap-1.5">
      {/* Technique badges */}
      {displayTechniques.map((technique, idx) => (
        <Badge
          key={`technique-${technique}`}
          className={cn(
            "border-0 text-xs font-medium",
            TECHNIQUE_COLORS[idx % TECHNIQUE_COLORS.length]
          )}
        >
          <Shield className="h-3 w-3 mr-1" aria-hidden="true" />
          {formatTechniqueType(technique)}
        </Badge>
      ))}
      {remainingTechniques > 0 && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="secondary" className="text-xs">
                +{remainingTechniques}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <p>
                {techniques
                  .slice(maxDisplay)
                  .map((t) => formatTechniqueType(t))
                  .join(", ")}
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}

      {/* Vulnerability badges */}
      {displayVulnerabilities.map((vulnerability, idx) => (
        <Badge
          key={`vulnerability-${vulnerability}`}
          className={cn(
            "border-0 text-xs font-medium",
            VULNERABILITY_COLORS[idx % VULNERABILITY_COLORS.length]
          )}
        >
          <Target className="h-3 w-3 mr-1" aria-hidden="true" />
          {formatVulnerabilityType(vulnerability)}
        </Badge>
      ))}
      {remainingVulnerabilities > 0 && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="secondary" className="text-xs">
                +{remainingVulnerabilities}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <p>
                {vulnerabilities
                  .slice(maxDisplay)
                  .map((v) => formatVulnerabilityType(v))
                  .join(", ")}
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}

/**
 * Tags Display Component
 */
function TagsDisplay({ tags, maxDisplay = 3 }: { tags: string[]; maxDisplay?: number }) {
  if (tags.length === 0) return null;

  const displayTags = tags.slice(0, maxDisplay);
  const remainingTags = tags.length - maxDisplay;

  return (
    <div className="flex flex-wrap gap-1">
      {displayTags.map((tag) => (
        <Badge
          key={tag}
          variant="outline"
          className="text-xs font-normal px-1.5 py-0"
        >
          <Tag className="h-2.5 w-2.5 mr-0.5" aria-hidden="true" />
          {tag}
        </Badge>
      ))}
      {remainingTags > 0 && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="outline" className="text-xs font-normal px-1.5 py-0">
                +{remainingTags}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <p>{tags.slice(maxDisplay).join(", ")}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * PromptTemplateCard Component
 *
 * Displays a prompt template in a card format with:
 * - Template name and description
 * - Technique and vulnerability type badges
 * - Star rating display
 * - Success rate indicator
 * - Sharing level and status badges
 * - Action buttons (Use, Copy, Edit, Delete)
 *
 * @accessibility
 * - Semantic HTML structure with proper heading hierarchy
 * - ARIA labels for interactive elements
 * - Keyboard navigable
 * - Focus indicators for all interactive elements
 * - Screen reader friendly stats and badges
 */
export function PromptTemplateCard({
  template,
  variant = "default",
  isSelected = false,
  currentUserId,
  onClick,
  onUseTemplate,
  onCopy,
  onEdit,
  onDelete,
  onViewDetails,
  className,
}: PromptTemplateCardProps) {
  const isCompact = variant === "compact";
  const isFeatured = variant === "featured";

  // Check if current user can edit this template
  const canEdit = currentUserId && template.created_by === currentUserId;
  const canDelete = canEdit;

  // Get sharing level config
  const sharingConfig = SHARING_LEVEL_CONFIG[template.sharing_level];
  const SharingIcon = sharingConfig.icon;

  // Get status config
  const statusConfig = STATUS_CONFIG[template.status];
  const StatusIcon = statusConfig.icon;

  // Handle card click
  const handleCardClick = () => {
    onClick?.(template);
  };

  // Handle copy action
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(template.prompt_content);
      onCopy?.(template);
    } catch (err) {
      // Clipboard API failed, try fallback
      onCopy?.(template);
    }
  };

  // Handle use template action
  const handleUseTemplate = (e: React.MouseEvent) => {
    e.stopPropagation();
    onUseTemplate?.(template);
  };

  // Format relative time
  const formatRelativeTime = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
  };

  return (
    <Card
      className={cn(
        "group relative overflow-hidden transition-all duration-300",
        "hover:shadow-lg hover:border-primary/20",
        "focus-within:ring-2 focus-within:ring-primary/50",
        isSelected && "ring-2 ring-primary border-primary",
        isFeatured &&
          "border-primary/30 bg-gradient-to-br from-primary/5 to-transparent",
        template.status === "archived" && "opacity-75",
        template.status === "deprecated" && "opacity-60",
        onClick && "cursor-pointer",
        className
      )}
      onClick={onClick ? handleCardClick : undefined}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleCardClick();
              }
            }
          : undefined
      }
      aria-label={`Prompt template: ${template.name}`}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            {/* Status and Sharing Level Badges */}
            <div className="flex items-center gap-2 mb-2">
              {/* Sharing Level Badge */}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge
                      className={cn(
                        "border-0 text-xs",
                        sharingConfig.bgColor,
                        sharingConfig.color
                      )}
                    >
                      <SharingIcon className="h-3 w-3 mr-1" aria-hidden="true" />
                      {formatSharingLevel(template.sharing_level)}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>
                      {template.sharing_level === "private"
                        ? "Only you can see this template"
                        : template.sharing_level === "team"
                          ? "Visible to your team"
                          : "Visible to everyone"}
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              {/* Status Badge (if not active) */}
              {template.status !== "active" && (
                <Badge
                  className={cn(
                    "border-0 text-xs",
                    statusConfig.bgColor,
                    statusConfig.color
                  )}
                >
                  <StatusIcon className="h-3 w-3 mr-1" aria-hidden="true" />
                  {statusConfig.label}
                </Badge>
              )}

              {/* Version Badge */}
              {template.current_version > 1 && (
                <Badge variant="outline" className="text-xs">
                  v{template.current_version}
                </Badge>
              )}
            </div>

            {/* Title */}
            <CardTitle
              className={cn(
                "line-clamp-2 transition-colors",
                isFeatured ? "text-lg" : "text-base",
                onClick && "group-hover:text-primary"
              )}
            >
              {template.name}
            </CardTitle>

            {/* Description */}
            {!isCompact && (
              <CardDescription className="line-clamp-2 mt-1">
                {template.description || "No description provided."}
              </CardDescription>
            )}
          </div>

          {/* Actions Dropdown */}
          <CardAction>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={(e) => e.stopPropagation()}
                  aria-label="Template actions"
                >
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem
                  onClick={(e) => {
                    e.stopPropagation();
                    onViewDetails?.(template);
                  }}
                >
                  <Eye className="h-4 w-4 mr-2" />
                  View Details
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleCopy}>
                  <Copy className="h-4 w-4 mr-2" />
                  Copy Prompt
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleUseTemplate}>
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Use Template
                </DropdownMenuItem>

                {canEdit && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      onClick={(e) => {
                        e.stopPropagation();
                        onEdit?.(template);
                      }}
                    >
                      <Edit3 className="h-4 w-4 mr-2" />
                      Edit
                    </DropdownMenuItem>
                  </>
                )}

                {canDelete && (
                  <DropdownMenuItem
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete?.(template);
                    }}
                    variant="destructive"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </CardAction>
        </div>
      </CardHeader>

      <CardContent className="pb-3">
        {/* Metadata Badges */}
        {!isCompact && (
          <div className="mb-3">
            <MetadataBadges
              techniques={template.metadata.technique_types}
              vulnerabilities={template.metadata.vulnerability_types}
              maxDisplay={isFeatured ? 3 : 2}
            />
          </div>
        )}

        {/* Tags */}
        {!isCompact && template.metadata.tags.length > 0 && (
          <div className="mb-3">
            <TagsDisplay tags={template.metadata.tags} maxDisplay={isFeatured ? 4 : 3} />
          </div>
        )}

        {/* Stats Row */}
        <div
          className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground"
          role="list"
          aria-label="Template statistics"
        >
          {/* Rating */}
          <div role="listitem">
            <StarRating
              rating={template.rating_stats.average_rating}
              totalRatings={template.rating_stats.total_ratings}
              size={isCompact ? "sm" : "default"}
            />
          </div>

          {/* Success Rate */}
          <div role="listitem">
            <SuccessRateBadge
              successRate={template.metadata.success_rate}
              testCount={template.metadata.test_count}
            />
          </div>

          {/* Target Models (if any) */}
          {template.metadata.target_models.length > 0 && !isCompact && (
            <div
              className="flex items-center gap-1"
              role="listitem"
              aria-label={`Tested on: ${template.metadata.target_models.join(", ")}`}
            >
              <span className="text-xs bg-muted px-1.5 py-0.5 rounded">
                {template.metadata.target_models[0]}
                {template.metadata.target_models.length > 1 &&
                  ` +${template.metadata.target_models.length - 1}`}
              </span>
            </div>
          )}
        </div>

        {/* Last Updated */}
        <div className="flex items-center gap-1 mt-3 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" aria-hidden="true" />
          <span>Updated {formatRelativeTime(template.updated_at)}</span>
        </div>
      </CardContent>

      <CardFooter className="pt-3 border-t justify-between gap-2">
        {/* Quick Actions */}
        <div className="flex items-center gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={handleCopy}
                  aria-label="Copy prompt to clipboard"
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Copy prompt</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        {/* Primary Action */}
        <Button size="sm" onClick={handleUseTemplate} aria-label={`Use ${template.name}`}>
          <ExternalLink className="h-4 w-4 mr-1" />
          Use Template
        </Button>
      </CardFooter>
    </Card>
  );
}

// =============================================================================
// Skeleton Component
// =============================================================================

/**
 * PromptTemplateCardSkeleton
 *
 * Loading skeleton for the PromptTemplateCard component.
 */
export function PromptTemplateCardSkeleton({
  variant = "default",
}: {
  variant?: "default" | "compact" | "featured";
}) {
  const isCompact = variant === "compact";

  return (
    <Card className="overflow-hidden animate-pulse">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            {/* Status badges skeleton */}
            <div className="flex gap-2 mb-2">
              <div className="h-5 w-16 bg-muted rounded-full" />
              <div className="h-5 w-12 bg-muted rounded-full" />
            </div>
            {/* Title skeleton */}
            <div className="h-5 bg-muted rounded w-3/4" />
            {/* Description skeleton */}
            {!isCompact && (
              <>
                <div className="h-4 bg-muted rounded w-full mt-2" />
                <div className="h-4 bg-muted rounded w-2/3 mt-1" />
              </>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pb-3">
        {/* Badges skeleton */}
        {!isCompact && (
          <div className="flex gap-1.5 mb-3">
            <div className="h-5 w-24 bg-muted rounded-full" />
            <div className="h-5 w-20 bg-muted rounded-full" />
            <div className="h-5 w-28 bg-muted rounded-full" />
          </div>
        )}

        {/* Stats skeleton */}
        <div className="flex gap-4">
          <div className="h-4 w-24 bg-muted rounded" />
          <div className="h-4 w-16 bg-muted rounded" />
        </div>

        {/* Updated time skeleton */}
        <div className="flex gap-1 mt-3">
          <div className="h-3 w-3 bg-muted rounded" />
          <div className="h-3 w-24 bg-muted rounded" />
        </div>
      </CardContent>

      <CardFooter className="pt-3 border-t justify-between">
        <div className="h-8 w-8 bg-muted rounded" />
        <div className="h-8 w-28 bg-muted rounded" />
      </CardFooter>
    </Card>
  );
}

export default PromptTemplateCard;
