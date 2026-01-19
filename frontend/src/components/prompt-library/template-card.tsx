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
import {
  PromptTemplate,
  TemplateListItem,
  TechniqueType,
  VulnerabilityType,
  SharingLevel,
  TemplateStatus,
  formatTechniqueType,
  formatVulnerabilityType,
  formatSharingLevel,
} from "@/types/prompt-library-types";

// =============================================================================
// Types & Interfaces
// =============================================================================

interface PromptTemplateCardProps {
  /** The prompt template data */
  template: TemplateListItem;
  /** Card display variant */
  variant?: "default" | "compact" | "featured";
  /** Whether the card is currently selected */
  isSelected?: boolean;
  /** Current user ID for permission checks */
  currentUserId?: string | null;
  /** Callback when clicking the card */
  onClick?: (template: TemplateListItem) => void;
  /** Callback when clicking the "Use Template" action */
  onUseTemplate?: (template: TemplateListItem) => void;
  /** Callback when clicking the "Copy" action */
  onCopy?: (template: TemplateListItem) => void;
  /** Callback when clicking the "Edit" action */
  onEdit?: (template: TemplateListItem) => void;
  /** Callback when clicking the "Delete" action */
  onDelete?: (template: TemplateListItem) => void;
  /** Callback when clicking the "View Details" action */
  onViewDetails?: (template: TemplateListItem) => void;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// Configuration
// =============================================================================

const SHARING_LEVEL_CONFIG: Record<
  SharingLevel,
  { icon: React.ElementType; color: string; bgColor: string }
> = {
  [SharingLevel.PRIVATE]: {
    icon: Lock,
    color: "text-orange-600",
    bgColor: "bg-orange-50 dark:bg-orange-950/30",
  },
  [SharingLevel.TEAM]: {
    icon: Users,
    color: "text-blue-600",
    bgColor: "bg-blue-50 dark:bg-blue-950/30",
  },
  [SharingLevel.PUBLIC]: {
    icon: Globe,
    color: "text-green-600",
    bgColor: "bg-green-50 dark:bg-green-950/30",
  },
};

const STATUS_CONFIG: Record<
  TemplateStatus,
  { icon: React.ElementType; color: string; bgColor: string; label: string }
> = {
  [TemplateStatus.DRAFT]: {
    icon: Edit3,
    color: "text-yellow-600",
    bgColor: "bg-yellow-50 dark:bg-yellow-950/30",
    label: "Draft",
  },
  [TemplateStatus.ACTIVE]: {
    icon: CheckCircle2,
    color: "text-green-600",
    bgColor: "bg-green-50 dark:bg-green-950/30",
    label: "Active",
  },
  [TemplateStatus.ARCHIVED]: {
    icon: Archive,
    color: "text-gray-500",
    bgColor: "bg-gray-50 dark:bg-gray-800/50",
    label: "Archived",
  },
  [TemplateStatus.DEPRECATED]: {
    icon: AlertTriangle,
    color: "text-red-600",
    bgColor: "bg-red-50 dark:bg-red-950/30",
    label: "Deprecated",
  },
};

const TECHNIQUE_COLORS: string[] = [
  "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
  "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300",
  "bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300",
  "bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300",
];

const VULNERABILITY_COLORS: string[] = [
  "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
  "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
];

// =============================================================================
// Helper Components
// =============================================================================

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
        ({totalRatings})
      </span>
    </div>
  );
}

function SuccessRateIndicator({
  effectivenessScore,
}: {
  effectivenessScore: number;
}) {
  const percentage = (effectivenessScore * 100).toFixed(0);
  const colorClass =
    effectivenessScore >= 0.7
      ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
      : effectivenessScore >= 0.4
        ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300"
        : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge className={cn("border-0", colorClass)}>
            <TrendingUp className="h-3 w-3 mr-1" aria-hidden="true" />
            {percentage}% Effective
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {percentage}% effectiveness based on user votes
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

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
        <Badge variant="secondary" className="text-xs">
          +{remainingTechniques}
        </Badge>
      )}

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
        <Badge variant="secondary" className="text-xs">
          +{remainingVulnerabilities}
        </Badge>
      )}
    </div>
  );
}

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
        <Badge variant="outline" className="text-xs font-normal px-1.5 py-0">
          +{remainingTags}
        </Badge>
      )}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

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

  const canEdit = currentUserId && template.owner_id === currentUserId;
  const canDelete = canEdit;

  const sharingConfig = SHARING_LEVEL_CONFIG[template.sharing_level];
  const SharingIcon = sharingConfig.icon;

  const statusConfig = STATUS_CONFIG[template.status];
  const StatusIcon = statusConfig.icon;

  const handleCardClick = () => {
    onClick?.(template);
  };

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    onCopy?.(template);
  };

  const handleUseTemplate = (e: React.MouseEvent) => {
    e.stopPropagation();
    onUseTemplate?.(template);
  };

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
        template.status === TemplateStatus.ARCHIVED && "opacity-75",
        template.status === TemplateStatus.DEPRECATED && "opacity-60",
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
      aria-label={`Prompt template: ${template.title}`}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
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
                      {template.sharing_level === SharingLevel.PRIVATE
                        ? "Only you can see this template"
                        : template.sharing_level === SharingLevel.TEAM
                          ? "Visible to your team"
                          : "Visible to everyone"}
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              {template.status !== TemplateStatus.ACTIVE && (
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
            </div>

            <CardTitle
              className={cn(
                "line-clamp-2 transition-colors",
                isFeatured ? "text-lg" : "text-base",
                onClick && "group-hover:text-primary"
              )}
            >
              {template.title}
            </CardTitle>

            {!isCompact && (
              <CardDescription className="line-clamp-2 mt-1">
                {template.description || "No description provided."}
              </CardDescription>
            )}
          </div>

          <div className="flex items-center gap-1">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
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
                    className="text-destructive focus:text-destructive"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pb-3">
        {!isCompact && (
          <div className="mb-3">
            <MetadataBadges
              techniques={template.technique_types}
              vulnerabilities={template.vulnerability_types}
              maxDisplay={isFeatured ? 3 : 2}
            />
          </div>
        )}

        {!isCompact && template.tags && template.tags.length > 0 && (
          <div className="mb-3">
            <TagsDisplay tags={template.tags} maxDisplay={isFeatured ? 4 : 3} />
          </div>
        )}

        <div
          className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground"
          role="list"
          aria-label="Template statistics"
        >
          <div role="listitem">
            <StarRating
              rating={template.avg_rating}
              totalRatings={template.total_ratings}
              size={isCompact ? "sm" : "default"}
            />
          </div>

          <div role="listitem">
            <SuccessRateIndicator
              effectivenessScore={template.effectiveness_score}
            />
          </div>
        </div>

        <div className="flex items-center gap-1 mt-3 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" aria-hidden="true" />
          <span>Updated {formatRelativeTime(template.created_at)}</span>
        </div>
      </CardContent>

      <CardFooter className="pt-3 border-t justify-between gap-2">
        <div className="flex items-center gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
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

        <Button size="sm" onClick={handleUseTemplate} aria-label={`Use ${template.title}`}>
          <ExternalLink className="h-4 w-4 mr-1" />
          Use Template
        </Button>
      </CardFooter>
    </Card>
  );
}

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
            <div className="flex gap-2 mb-2">
              <div className="h-5 w-16 bg-muted rounded-full" />
              <div className="h-5 w-12 bg-muted rounded-full" />
            </div>
            <div className="h-5 bg-muted rounded w-3/4" />
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
        {!isCompact && (
          <div className="flex gap-1.5 mb-3">
            <div className="h-5 w-24 bg-muted rounded-full" />
            <div className="h-5 w-20 bg-muted rounded-full" />
            <div className="h-5 w-28 bg-muted rounded-full" />
          </div>
        )}

        <div className="flex gap-4">
          <div className="h-4 w-24 bg-muted rounded" />
          <div className="h-4 w-16 bg-muted rounded" />
        </div>

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