"use client";

/**
 * Password Strength Meter Component
 *
 * A visual indicator for password strength with:
 * - Color-coded strength bar (red -> yellow -> green)
 * - Score display (0-100)
 * - Errors, warnings, and suggestions display
 * - Real-time feedback
 *
 * @module components/auth/PasswordStrengthMeter
 */

import React, { useMemo } from "react";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Lightbulb,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Shield,
} from "lucide-react";

import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { PasswordStrengthResult } from "@/contexts/AuthContext";

// =============================================================================
// Types
// =============================================================================

interface PasswordStrengthMeterProps {
  /** Password strength result from API */
  result: PasswordStrengthResult | null;
  /** Loading state */
  isLoading?: boolean;
  /** Additional class names */
  className?: string;
  /** Show detailed feedback (errors, warnings, suggestions) */
  showDetails?: boolean;
  /** Compact mode (smaller text, tighter spacing) */
  compact?: boolean;
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Get strength label and color based on score
 */
function getStrengthInfo(score: number): {
  label: string;
  color: string;
  bgColor: string;
  textColor: string;
  Icon: typeof Shield;
} {
  if (score >= 80) {
    return {
      label: "Strong",
      color: "bg-green-500",
      bgColor: "bg-green-500/20",
      textColor: "text-green-400",
      Icon: ShieldCheck,
    };
  }
  if (score >= 60) {
    return {
      label: "Good",
      color: "bg-emerald-500",
      bgColor: "bg-emerald-500/20",
      textColor: "text-emerald-400",
      Icon: ShieldCheck,
    };
  }
  if (score >= 40) {
    return {
      label: "Fair",
      color: "bg-yellow-500",
      bgColor: "bg-yellow-500/20",
      textColor: "text-yellow-400",
      Icon: ShieldAlert,
    };
  }
  if (score >= 20) {
    return {
      label: "Weak",
      color: "bg-orange-500",
      bgColor: "bg-orange-500/20",
      textColor: "text-orange-400",
      Icon: ShieldAlert,
    };
  }
  return {
    label: "Very Weak",
    color: "bg-red-500",
    bgColor: "bg-red-500/20",
    textColor: "text-red-400",
    Icon: ShieldX,
  };
}

// =============================================================================
// Component
// =============================================================================

export function PasswordStrengthMeter({
  result,
  isLoading,
  className,
  showDetails = true,
  compact = false,
}: PasswordStrengthMeterProps) {
  // Get strength info based on score
  const strengthInfo = useMemo(() => {
    if (!result) return null;
    return getStrengthInfo(result.score);
  }, [result]);

  // Don't render anything if no result and not loading
  if (!result && !isLoading) {
    return null;
  }

  return (
    <div className={cn("space-y-2", className)}>
      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <div className="w-3 h-3 border border-primary/30 border-t-primary rounded-full animate-spin" />
          <span className={cn("text-sm", compact && "text-xs")}>
            Checking password strength...
          </span>
        </div>
      )}

      {/* Strength Result */}
      {result && strengthInfo && !isLoading && (
        <>
          {/* Strength Bar */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <strengthInfo.Icon
                  className={cn(
                    "w-4 h-4",
                    strengthInfo.textColor,
                    compact && "w-3.5 h-3.5"
                  )}
                />
                <span
                  className={cn(
                    "text-sm font-medium",
                    strengthInfo.textColor,
                    compact && "text-xs"
                  )}
                >
                  {strengthInfo.label}
                </span>
              </div>
              <span
                className={cn(
                  "text-xs text-muted-foreground",
                  compact && "text-[10px]"
                )}
              >
                {result.score}/100
              </span>
            </div>

            {/* Custom Progress Bar with Color */}
            <div
              className={cn(
                "relative h-2 w-full overflow-hidden rounded-full",
                strengthInfo.bgColor,
                compact && "h-1.5"
              )}
            >
              <div
                className={cn(
                  "h-full transition-all duration-500 ease-out rounded-full",
                  strengthInfo.color
                )}
                style={{ width: `${result.score}%` }}
              />
            </div>
          </div>

          {/* Detailed Feedback */}
          {showDetails && (
            <div className="space-y-2 pt-1">
              {/* Errors */}
              {result.errors.length > 0 && (
                <FeedbackList
                  items={result.errors}
                  type="error"
                  compact={compact}
                />
              )}

              {/* Warnings */}
              {result.warnings.length > 0 && (
                <FeedbackList
                  items={result.warnings}
                  type="warning"
                  compact={compact}
                />
              )}

              {/* Suggestions (only show if password is not strong enough) */}
              {result.suggestions.length > 0 && result.score < 80 && (
                <FeedbackList
                  items={result.suggestions}
                  type="suggestion"
                  compact={compact}
                />
              )}

              {/* Success message for strong passwords */}
              {result.is_valid && result.score >= 80 && (
                <div className="flex items-center gap-2 text-green-400 animate-fade-in">
                  <CheckCircle2
                    className={cn("w-4 h-4", compact && "w-3.5 h-3.5")}
                  />
                  <span className={cn("text-sm", compact && "text-xs")}>
                    Your password meets all requirements
                  </span>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// =============================================================================
// Feedback List Sub-component
// =============================================================================

interface FeedbackListProps {
  items: string[];
  type: "error" | "warning" | "suggestion";
  compact?: boolean;
}

function FeedbackList({ items, type, compact = false }: FeedbackListProps) {
  // Get icon and colors based on type
  const config = useMemo(() => {
    switch (type) {
      case "error":
        return {
          Icon: XCircle,
          iconColor: "text-red-400",
          bgColor: "bg-red-500/5",
          borderColor: "border-red-500/20",
        };
      case "warning":
        return {
          Icon: AlertTriangle,
          iconColor: "text-yellow-400",
          bgColor: "bg-yellow-500/5",
          borderColor: "border-yellow-500/20",
        };
      case "suggestion":
        return {
          Icon: Lightbulb,
          iconColor: "text-blue-400",
          bgColor: "bg-blue-500/5",
          borderColor: "border-blue-500/20",
        };
    }
  }, [type]);

  if (items.length === 0) return null;

  return (
    <ul className="space-y-1">
      {items.map((item, index) => (
        <li
          key={index}
          className={cn(
            "flex items-start gap-2 text-muted-foreground animate-fade-in",
            compact && "gap-1.5"
          )}
          style={{ animationDelay: `${index * 50}ms` }}
        >
          <config.Icon
            className={cn(
              "w-4 h-4 mt-0.5 flex-shrink-0",
              config.iconColor,
              compact && "w-3.5 h-3.5"
            )}
          />
          <span className={cn("text-sm", compact && "text-xs")}>{item}</span>
        </li>
      ))}
    </ul>
  );
}

// =============================================================================
// Password Requirements Display
// =============================================================================

interface PasswordRequirement {
  label: string;
  met: boolean;
}

interface PasswordRequirementsProps {
  password: string;
  className?: string;
  compact?: boolean;
}

/**
 * Display password requirements with visual check/uncheck indicators
 */
export function PasswordRequirements({
  password,
  className,
  compact = false,
}: PasswordRequirementsProps) {
  // Check which requirements are met
  const requirements: PasswordRequirement[] = useMemo(() => {
    return [
      { label: "At least 12 characters", met: password.length >= 12 },
      { label: "At least one uppercase letter", met: /[A-Z]/.test(password) },
      { label: "At least one lowercase letter", met: /[a-z]/.test(password) },
      { label: "At least one number", met: /[0-9]/.test(password) },
      {
        label: "At least one special character (!@#$%^&*)",
        met: /[!@#$%^&*(),.?":{}|<>]/.test(password),
      },
    ];
  }, [password]);

  // Count how many requirements are met
  const metCount = requirements.filter((r) => r.met).length;
  const allMet = metCount === requirements.length;

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <span
          className={cn(
            "text-xs text-muted-foreground font-medium",
            compact && "text-[10px]"
          )}
        >
          Password Requirements
        </span>
        <span
          className={cn(
            "text-xs",
            allMet ? "text-green-400" : "text-muted-foreground",
            compact && "text-[10px]"
          )}
        >
          {metCount}/{requirements.length}
        </span>
      </div>

      <ul className="space-y-1">
        {requirements.map((req, index) => (
          <li
            key={index}
            className={cn(
              "flex items-center gap-2 transition-colors duration-200",
              req.met ? "text-green-400" : "text-muted-foreground",
              compact && "gap-1.5"
            )}
          >
            {req.met ? (
              <CheckCircle2
                className={cn("w-3.5 h-3.5", compact && "w-3 h-3")}
              />
            ) : (
              <div
                className={cn(
                  "w-3.5 h-3.5 rounded-full border border-muted-foreground/50",
                  compact && "w-3 h-3"
                )}
              />
            )}
            <span className={cn("text-xs", compact && "text-[10px]")}>
              {req.label}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default PasswordStrengthMeter;
