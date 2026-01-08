"use client";

import * as React from "react";
import { Star, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Instructor } from "./types";

interface InstructorBadgeProps {
  instructor: Instructor;
  variant?: "compact" | "detailed";
  showStats?: boolean;
  className?: string;
}

/**
 * InstructorBadge Component
 * 
 * Displays instructor information with avatar, name, title, and optional stats.
 * Supports compact (for cards) and detailed (for course pages) variants.
 * 
 * @accessibility
 * - Uses semantic HTML with proper heading hierarchy
 * - Avatar has alt text for screen readers
 * - Stats are labeled for assistive technologies
 */
export function InstructorBadge({
  instructor,
  variant = "compact",
  showStats = false,
  className,
}: InstructorBadgeProps) {
  const isCompact = variant === "compact";

  return (
    <div
      className={cn(
        "flex items-center gap-3",
        isCompact ? "py-2" : "py-4",
        className
      )}
      role="group"
      aria-label={`Instructor: ${instructor.name}`}
    >
      {/* Avatar */}
      <div
        className={cn(
          "relative flex-shrink-0 rounded-full bg-gradient-to-br from-primary/20 to-primary/40 flex items-center justify-center overflow-hidden",
          isCompact ? "h-10 w-10" : "h-14 w-14"
        )}
      >
        {instructor.avatar ? (
          <img
            src={instructor.avatar}
            alt={`${instructor.name}'s profile picture`}
            className="h-full w-full object-cover"
            loading="lazy"
          />
        ) : (
          <span
            className={cn(
              "font-semibold text-primary",
              isCompact ? "text-sm" : "text-lg"
            )}
            aria-hidden="true"
          >
            {instructor.name
              .split(" ")
              .map((n) => n[0])
              .join("")
              .toUpperCase()
              .slice(0, 2)}
          </span>
        )}
      </div>

      {/* Info */}
      <div className="flex flex-col min-w-0">
        <span
          className={cn(
            "font-medium text-foreground truncate",
            isCompact ? "text-sm" : "text-base"
          )}
        >
          {instructor.name}
        </span>
        <span
          className={cn(
            "text-muted-foreground truncate",
            isCompact ? "text-xs" : "text-sm"
          )}
        >
          {instructor.title}
        </span>

        {/* Stats (detailed variant only) */}
        {showStats && !isCompact && (
          <div className="flex items-center gap-4 mt-2">
            {instructor.rating && (
              <div
                className="flex items-center gap-1 text-sm"
                aria-label={`Rating: ${instructor.rating} out of 5 stars`}
              >
                <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" aria-hidden="true" />
                <span className="font-medium">{instructor.rating.toFixed(1)}</span>
              </div>
            )}
            {instructor.totalStudents && (
              <div
                className="flex items-center gap-1 text-sm text-muted-foreground"
                aria-label={`${instructor.totalStudents.toLocaleString()} students`}
              >
                <Users className="h-4 w-4" aria-hidden="true" />
                <span>{instructor.totalStudents.toLocaleString()} students</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * InstructorBadgeSkeleton
 * 
 * Loading skeleton for the InstructorBadge component.
 */
export function InstructorBadgeSkeleton({
  variant = "compact",
}: {
  variant?: "compact" | "detailed";
}) {
  const isCompact = variant === "compact";

  return (
    <div className={cn("flex items-center gap-3 animate-pulse", isCompact ? "py-2" : "py-4")}>
      <div
        className={cn(
          "rounded-full bg-muted",
          isCompact ? "h-10 w-10" : "h-14 w-14"
        )}
      />
      <div className="flex flex-col gap-2">
        <div className={cn("h-4 bg-muted rounded", isCompact ? "w-24" : "w-32")} />
        <div className={cn("h-3 bg-muted rounded", isCompact ? "w-16" : "w-24")} />
      </div>
    </div>
  );
}

export default InstructorBadge;