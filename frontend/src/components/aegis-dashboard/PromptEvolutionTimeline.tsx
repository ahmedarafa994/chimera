/**
 * PromptEvolutionTimeline Component
 *
 * Displays how prompts evolve through iterations for Aegis Campaign Dashboard:
 * - Timeline visualization of prompt transformations
 * - Score indicator per iteration
 * - Diff view between iterations
 * - Expandable prompt text with copy button
 *
 * Follows glass morphism styling pattern from existing components.
 */

"use client";

import { memo, useMemo, useState, useCallback } from "react";
import {
  Sparkles,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Copy,
  Check,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Minus,
  Clock,
  Zap,
  FileText,
  Diff,
  LayoutList,
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
import { cn } from "@/lib/utils";
import { PromptEvolution } from "@/types/aegis-telemetry";

// ============================================================================
// Types
// ============================================================================

export interface PromptEvolutionTimelineProps {
  /** Array of prompt evolutions to display */
  evolutions: PromptEvolution[];
  /** Whether the component is in loading state */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show in compact mode */
  compact?: boolean;
  /** Maximum height for the timeline (default: 500px) */
  maxHeight?: number;
  /** Callback when an evolution is selected */
  onEvolutionSelect?: (evolution: PromptEvolution) => void;
}

type ViewMode = "timeline" | "diff";

// ============================================================================
// Configuration
// ============================================================================

/**
 * Score color configuration based on thresholds
 */
interface ScoreColorConfig {
  label: string;
  textClass: string;
  bgClass: string;
  borderClass: string;
}

const getScoreColorConfig = (score: number): ScoreColorConfig => {
  if (score >= 0.7) {
    return {
      label: "High",
      textClass: "text-emerald-400",
      bgClass: "bg-emerald-500/10",
      borderClass: "border-emerald-500/20",
    };
  } else if (score >= 0.4) {
    return {
      label: "Medium",
      textClass: "text-amber-400",
      bgClass: "bg-amber-500/10",
      borderClass: "border-amber-500/20",
    };
  } else {
    return {
      label: "Low",
      textClass: "text-red-400",
      bgClass: "bg-red-500/10",
      borderClass: "border-red-500/20",
    };
  }
};

const getImprovementConfig = (improvement: number) => {
  if (improvement > 0.05) {
    return {
      icon: TrendingUp,
      textClass: "text-emerald-400",
      bgClass: "bg-emerald-500/10",
      borderClass: "border-emerald-500/20",
      label: "Improved",
    };
  } else if (improvement < -0.05) {
    return {
      icon: TrendingDown,
      textClass: "text-red-400",
      bgClass: "bg-red-500/10",
      borderClass: "border-red-500/20",
      label: "Declined",
    };
  } else {
    return {
      icon: Minus,
      textClass: "text-gray-400",
      bgClass: "bg-gray-500/10",
      borderClass: "border-gray-500/20",
      label: "Stable",
    };
  }
};

// ============================================================================
// Helper Functions
// ============================================================================

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
 * Format relative time
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
 * Format score as percentage
 */
function formatScore(score: number): string {
  return `${(score * 100).toFixed(1)}%`;
}

/**
 * Truncate text with ellipsis
 */
function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + "...";
}

/**
 * Calculate simple diff between two strings
 * Returns an array of diff parts with type: 'same' | 'added' | 'removed'
 */
interface DiffPart {
  type: "same" | "added" | "removed";
  text: string;
}

function calculateDiff(original: string, evolved: string): DiffPart[] {
  const parts: DiffPart[] = [];

  // Simple word-based diff for visualization
  const originalWords = original.split(/(\s+)/);
  const evolvedWords = evolved.split(/(\s+)/);

  let i = 0;
  let j = 0;

  while (i < originalWords.length || j < evolvedWords.length) {
    if (i >= originalWords.length) {
      // Remaining words in evolved are additions
      parts.push({ type: "added", text: evolvedWords.slice(j).join("") });
      break;
    }
    if (j >= evolvedWords.length) {
      // Remaining words in original are removals
      parts.push({ type: "removed", text: originalWords.slice(i).join("") });
      break;
    }

    if (originalWords[i] === evolvedWords[j]) {
      // Words match
      if (parts.length > 0 && parts[parts.length - 1].type === "same") {
        parts[parts.length - 1].text += originalWords[i];
      } else {
        parts.push({ type: "same", text: originalWords[i] });
      }
      i++;
      j++;
    } else {
      // Words differ - look ahead for match
      let foundInOriginal = -1;
      let foundInEvolved = -1;

      // Look for evolvedWords[j] in remaining originalWords
      for (let k = i + 1; k < Math.min(i + 5, originalWords.length); k++) {
        if (originalWords[k] === evolvedWords[j]) {
          foundInOriginal = k;
          break;
        }
      }

      // Look for originalWords[i] in remaining evolvedWords
      for (let k = j + 1; k < Math.min(j + 5, evolvedWords.length); k++) {
        if (evolvedWords[k] === originalWords[i]) {
          foundInEvolved = k;
          break;
        }
      }

      if (foundInOriginal !== -1 && (foundInEvolved === -1 || foundInOriginal - i <= foundInEvolved - j)) {
        // Remove words from original until match
        parts.push({ type: "removed", text: originalWords.slice(i, foundInOriginal).join("") });
        i = foundInOriginal;
      } else if (foundInEvolved !== -1) {
        // Add words from evolved until match
        parts.push({ type: "added", text: evolvedWords.slice(j, foundInEvolved).join("") });
        j = foundInEvolved;
      } else {
        // No match found - mark as removed and added
        parts.push({ type: "removed", text: originalWords[i] });
        parts.push({ type: "added", text: evolvedWords[j] });
        i++;
        j++;
      }
    }
  }

  return parts;
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Copy button component
 */
const CopyButton = memo(function CopyButton({
  text,
  className,
  size = "sm",
}: {
  text: string;
  className?: string;
  size?: "sm" | "xs";
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
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
              size === "xs" ? "h-5 w-5 p-0" : "h-6 w-6 p-0",
              copied && "text-emerald-400",
              className
            )}
          >
            {copied ? (
              <Check className={cn(size === "xs" ? "h-2.5 w-2.5" : "h-3 w-3")} />
            ) : (
              <Copy className={cn(size === "xs" ? "h-2.5 w-2.5" : "h-3 w-3")} />
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{copied ? "Copied!" : "Copy prompt"}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Score badge component
 */
const ScoreBadge = memo(function ScoreBadge({
  score,
  compact,
}: {
  score: number;
  compact?: boolean;
}) {
  const config = getScoreColorConfig(score);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className={cn(
              "font-medium tabular-nums",
              config.bgClass,
              config.borderClass,
              config.textClass,
              compact ? "text-[10px] px-1 py-0" : "text-xs px-1.5 py-0.5"
            )}
          >
            {formatScore(score)}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p>{config.label} score: {formatScore(score)}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Improvement indicator component
 */
const ImprovementIndicator = memo(function ImprovementIndicator({
  improvement,
  compact,
}: {
  improvement: number;
  compact?: boolean;
}) {
  const config = getImprovementConfig(improvement);
  const Icon = config.icon;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn(
              "flex items-center gap-1 rounded-full px-1.5 py-0.5",
              config.bgClass,
              config.borderClass,
              "border"
            )}
          >
            <Icon
              className={cn(
                config.textClass,
                compact ? "h-2.5 w-2.5" : "h-3 w-3"
              )}
            />
            <span
              className={cn(
                "font-medium tabular-nums",
                config.textClass,
                compact ? "text-[10px]" : "text-xs"
              )}
            >
              {improvement > 0 ? "+" : ""}
              {formatScore(improvement)}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>{config.label}: {improvement > 0 ? "+" : ""}{formatScore(improvement)}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
});

/**
 * Techniques applied badges
 */
const TechniquesBadges = memo(function TechniquesBadges({
  techniques,
  maxVisible = 3,
}: {
  techniques: string[];
  maxVisible?: number;
}) {
  const visible = techniques.slice(0, maxVisible);
  const hidden = techniques.length - maxVisible;

  return (
    <div className="flex flex-wrap gap-1">
      {visible.map((technique, index) => (
        <Badge
          key={index}
          variant="outline"
          className="text-[10px] px-1 py-0 h-4 text-violet-400 bg-violet-500/10 border-violet-500/20"
        >
          {technique}
        </Badge>
      ))}
      {hidden > 0 && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="outline"
                className="text-[10px] px-1 py-0 h-4 text-muted-foreground"
              >
                +{hidden}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <p>{techniques.slice(maxVisible).join(", ")}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
});

/**
 * Diff view component showing changes between prompts
 */
const DiffView = memo(function DiffView({
  original,
  evolved,
}: {
  original: string;
  evolved: string;
}) {
  const diffParts = useMemo(() => calculateDiff(original, evolved), [original, evolved]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Diff className="h-3 w-3" />
        <span>Changes</span>
      </div>
      <div className="rounded-lg bg-black/30 p-3 text-sm font-mono overflow-x-auto">
        {diffParts.map((part, index) => (
          <span
            key={index}
            className={cn(
              part.type === "added" && "bg-emerald-500/20 text-emerald-300",
              part.type === "removed" && "bg-red-500/20 text-red-300 line-through",
              part.type === "same" && "text-gray-300"
            )}
          >
            {part.text}
          </span>
        ))}
      </div>
    </div>
  );
});

/**
 * Prompt text display with expand/collapse
 */
const PromptText = memo(function PromptText({
  prompt,
  label,
  isExpanded,
  colorClass,
}: {
  prompt: string;
  label: string;
  isExpanded: boolean;
  colorClass: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className={cn("text-xs font-medium", colorClass)}>{label}</span>
        <CopyButton text={prompt} size="xs" />
      </div>
      <div
        className={cn(
          "rounded-lg bg-black/30 p-2 text-xs font-mono overflow-x-auto",
          !isExpanded && "max-h-16 overflow-hidden relative"
        )}
      >
        <p className="whitespace-pre-wrap text-gray-300">
          {isExpanded ? prompt : truncateText(prompt, 200)}
        </p>
        {!isExpanded && prompt.length > 200 && (
          <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-black/60 to-transparent" />
        )}
      </div>
    </div>
  );
});

/**
 * Timeline node representing a single evolution
 */
const TimelineNode = memo(function TimelineNode({
  evolution,
  index,
  isFirst,
  isLast,
  isExpanded,
  onToggleExpand,
  compact,
}: {
  evolution: PromptEvolution;
  index: number;
  isFirst: boolean;
  isLast: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
  compact?: boolean;
}) {
  const scoreConfig = getScoreColorConfig(evolution.score);
  const [showDiff, setShowDiff] = useState(false);

  return (
    <div className="relative flex gap-3">
      {/* Timeline line */}
      <div className="flex flex-col items-center">
        {/* Node indicator */}
        <div
          className={cn(
            "relative z-10 flex items-center justify-center rounded-full border-2 transition-all duration-300",
            scoreConfig.bgClass,
            scoreConfig.borderClass,
            evolution.is_successful
              ? "ring-2 ring-emerald-500/30"
              : "",
            compact ? "h-6 w-6" : "h-8 w-8"
          )}
        >
          {evolution.is_successful ? (
            <Zap
              className={cn("text-emerald-400", compact ? "h-3 w-3" : "h-4 w-4")}
            />
          ) : (
            <span
              className={cn(
                "font-semibold tabular-nums",
                scoreConfig.textClass,
                compact ? "text-[10px]" : "text-xs"
              )}
            >
              {evolution.iteration}
            </span>
          )}
        </div>
        {/* Vertical line */}
        {!isLast && (
          <div
            className={cn(
              "flex-1 w-0.5 bg-gradient-to-b",
              scoreConfig.bgClass.replace("bg-", "from-"),
              "to-white/5",
              compact ? "min-h-[40px]" : "min-h-[60px]"
            )}
          />
        )}
      </div>

      {/* Content */}
      <div className={cn("flex-1 pb-4", compact && "pb-2")}>
        {/* Header row */}
        <button
          onClick={onToggleExpand}
          className="w-full text-left group"
          aria-expanded={isExpanded}
          aria-label={`Iteration ${evolution.iteration} - click to ${isExpanded ? "collapse" : "expand"}`}
        >
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-sm font-medium text-foreground truncate">
                Iteration {evolution.iteration}
              </span>
              <ScoreBadge score={evolution.score} compact={compact} />
              <ImprovementIndicator improvement={evolution.improvement} compact={compact} />
              {evolution.is_successful && (
                <Badge
                  variant="outline"
                  className="text-[10px] px-1 py-0 h-4 text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
                >
                  Success
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                      <Clock className="h-2.5 w-2.5" />
                      {formatRelativeTime(evolution.timestamp)}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{formatTimestamp(evolution.timestamp)}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              {isExpanded ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
              )}
            </div>
          </div>

          {/* Techniques row (always visible) */}
          {evolution.techniques_applied.length > 0 && (
            <div className="mt-1.5">
              <TechniquesBadges techniques={evolution.techniques_applied} />
            </div>
          )}

          {/* Preview when collapsed */}
          {!isExpanded && (
            <p className="mt-2 text-xs text-muted-foreground truncate">
              {truncateText(evolution.evolved_prompt, 100)}
            </p>
          )}
        </button>

        {/* Expanded content */}
        {isExpanded && (
          <div className="mt-3 space-y-3">
            {/* View mode toggle */}
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDiff(false)}
                className={cn(
                  "h-6 px-2 text-xs",
                  !showDiff
                    ? "text-cyan-400 bg-cyan-500/10"
                    : "text-muted-foreground"
                )}
              >
                <LayoutList className="h-3 w-3 mr-1" />
                Side by Side
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDiff(true)}
                className={cn(
                  "h-6 px-2 text-xs",
                  showDiff
                    ? "text-cyan-400 bg-cyan-500/10"
                    : "text-muted-foreground"
                )}
              >
                <Diff className="h-3 w-3 mr-1" />
                Diff View
              </Button>
            </div>

            {showDiff ? (
              <DiffView
                original={evolution.original_prompt}
                evolved={evolution.evolved_prompt}
              />
            ) : (
              <div className="grid gap-3 md:grid-cols-2">
                <PromptText
                  prompt={evolution.original_prompt}
                  label="Original"
                  isExpanded={true}
                  colorClass="text-gray-400"
                />
                <div className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-cyan-400 mt-6 shrink-0 hidden md:block" />
                  <div className="flex-1">
                    <PromptText
                      prompt={evolution.evolved_prompt}
                      label="Evolved"
                      isExpanded={true}
                      colorClass={scoreConfig.textClass}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Stats footer */}
            <div className="flex items-center gap-4 pt-2 border-t border-white/5 text-xs text-muted-foreground">
              <span>
                Score: <span className={cn("font-medium", scoreConfig.textClass)}>{formatScore(evolution.score)}</span>
              </span>
              <span>
                Change: <span className={cn("font-medium", getImprovementConfig(evolution.improvement).textClass)}>
                  {evolution.improvement > 0 ? "+" : ""}{formatScore(evolution.improvement)}
                </span>
              </span>
              <span>
                Techniques: <span className="font-medium text-violet-400">{evolution.techniques_applied.length}</span>
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

/**
 * Empty state component
 */
const EmptyState = memo(function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <Sparkles className="h-8 w-8 text-muted-foreground/50 mb-3" />
      <p className="text-sm text-muted-foreground">No prompt evolutions yet</p>
      <p className="text-xs text-muted-foreground/70 mt-1">
        Evolutions will appear here as the campaign progresses
      </p>
    </div>
  );
});

/**
 * Loading skeleton component
 */
const LoadingSkeleton = memo(function LoadingSkeleton() {
  return (
    <div className="animate-pulse space-y-4 p-4">
      {[...Array(3)].map((_, i) => (
        <div key={i} className="flex gap-3">
          <div className="h-8 w-8 bg-white/10 rounded-full shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="flex items-center gap-2">
              <div className="h-4 w-20 bg-white/10 rounded" />
              <div className="h-4 w-12 bg-white/10 rounded" />
              <div className="h-4 w-16 bg-white/10 rounded" />
            </div>
            <div className="h-3 w-full bg-white/10 rounded" />
            <div className="h-3 w-3/4 bg-white/10 rounded" />
          </div>
        </div>
      ))}
    </div>
  );
});

/**
 * Summary stats header
 */
const SummaryStats = memo(function SummaryStats({
  evolutions,
  compact,
}: {
  evolutions: PromptEvolution[];
  compact?: boolean;
}) {
  const stats = useMemo(() => {
    if (evolutions.length === 0) {
      return {
        totalIterations: 0,
        successfulIterations: 0,
        bestScore: 0,
        totalImprovement: 0,
        avgImprovement: 0,
      };
    }

    const successful = evolutions.filter((e) => e.is_successful).length;
    const bestScore = Math.max(...evolutions.map((e) => e.score));
    const totalImprovement = evolutions.reduce((sum, e) => sum + e.improvement, 0);

    return {
      totalIterations: evolutions.length,
      successfulIterations: successful,
      bestScore,
      totalImprovement,
      avgImprovement: evolutions.length > 0 ? totalImprovement / evolutions.length : 0,
    };
  }, [evolutions]);

  if (compact) return null;

  return (
    <div className="grid grid-cols-4 gap-2 p-3 border-b border-white/5">
      <div className="flex flex-col items-center justify-center rounded-lg bg-white/5 p-2">
        <span className="text-xs text-muted-foreground">Iterations</span>
        <span className="text-sm font-semibold text-foreground tabular-nums">
          {stats.totalIterations}
        </span>
      </div>
      <div className="flex flex-col items-center justify-center rounded-lg bg-emerald-500/10 p-2">
        <span className="text-xs text-emerald-400/80">Successful</span>
        <span className="text-sm font-semibold text-emerald-400 tabular-nums">
          {stats.successfulIterations}
        </span>
      </div>
      <div className="flex flex-col items-center justify-center rounded-lg bg-violet-500/10 p-2">
        <span className="text-xs text-violet-400/80">Best Score</span>
        <span className="text-sm font-semibold text-violet-400 tabular-nums">
          {formatScore(stats.bestScore)}
        </span>
      </div>
      <div className="flex flex-col items-center justify-center rounded-lg bg-cyan-500/10 p-2">
        <span className="text-xs text-cyan-400/80">Avg Improve</span>
        <span className={cn(
          "text-sm font-semibold tabular-nums",
          stats.avgImprovement >= 0 ? "text-cyan-400" : "text-red-400"
        )}>
          {stats.avgImprovement >= 0 ? "+" : ""}{formatScore(stats.avgImprovement)}
        </span>
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * PromptEvolutionTimeline displays how prompts evolve through iterations.
 *
 * Features:
 * - Timeline visualization of prompt transformations
 * - Score indicator per iteration with color coding
 * - Diff view between original and evolved prompts
 * - Expandable prompt text with copy button
 * - Summary statistics header
 */
export const PromptEvolutionTimeline = memo(function PromptEvolutionTimeline({
  evolutions,
  isLoading = false,
  className,
  compact = false,
  maxHeight = 500,
  onEvolutionSelect,
}: PromptEvolutionTimelineProps) {
  // State
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  // Sort evolutions by iteration (most recent first for timeline)
  const sortedEvolutions = useMemo(
    () => [...evolutions].sort((a, b) => b.iteration - a.iteration),
    [evolutions]
  );

  // Toggle expansion
  const toggleExpand = useCallback(
    (index: number, evolution: PromptEvolution) => {
      setExpandedIndex((prev) => (prev === index ? null : index));
      onEvolutionSelect?.(evolution);
    },
    [onEvolutionSelect]
  );

  // Loading state
  if (isLoading) {
    return (
      <GlassCard variant="default" intensity="medium" className={cn("p-0", className)}>
        <div className="flex items-center gap-2 p-3 border-b border-white/5">
          <Sparkles className="h-4 w-4 text-pink-400" aria-hidden="true" />
          <h3 className="text-sm font-medium text-muted-foreground">
            Prompt Evolution
          </h3>
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
          <Sparkles className="h-4 w-4 text-pink-400" aria-hidden="true" />
          <h3 className="text-sm font-medium text-muted-foreground">
            Prompt Evolution
          </h3>
          <Badge
            variant="outline"
            className="text-[10px] px-1.5 py-0 h-4 text-muted-foreground border-white/10"
          >
            {evolutions.length} iterations
          </Badge>
        </div>
        {evolutions.some((e) => e.is_successful) && (
          <Badge
            variant="outline"
            className="text-[10px] px-1.5 py-0 h-4 text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
          >
            <Zap className="h-2.5 w-2.5 mr-0.5" />
            Has Success
          </Badge>
        )}
      </div>

      {/* Summary stats */}
      <SummaryStats evolutions={evolutions} compact={compact} />

      {/* Timeline content */}
      <div
        className="overflow-y-auto p-4"
        style={{ maxHeight }}
      >
        {sortedEvolutions.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="space-y-0">
            {sortedEvolutions.map((evolution, index) => (
              <TimelineNode
                key={`${evolution.iteration}-${evolution.timestamp}`}
                evolution={evolution}
                index={index}
                isFirst={index === 0}
                isLast={index === sortedEvolutions.length - 1}
                isExpanded={expandedIndex === index}
                onToggleExpand={() => toggleExpand(index, evolution)}
                compact={compact}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer with hint */}
      {sortedEvolutions.length > 0 && !compact && (
        <div className="flex items-center justify-center gap-1 p-2 border-t border-white/5 text-[10px] text-muted-foreground">
          <FileText className="h-2.5 w-2.5" />
          <span>Click on an iteration to view details and diff</span>
        </div>
      )}
    </GlassCard>
  );
});

// Named export for index
export default PromptEvolutionTimeline;
