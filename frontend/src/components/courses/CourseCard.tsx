"use client";

import * as React from "react";
import Link from "next/link";
import {
  Clock,
  BookOpen,
  Users,
  Star,
  TrendingUp,
  Sparkles,
  Award,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";
import { InstructorBadge } from "./InstructorBadge";
import type { Course } from "./types";
import { LEVEL_CONFIG, CATEGORY_CONFIG } from "./types";

interface CourseCardProps {
  course: Course;
  variant?: "default" | "featured" | "compact";
  onEnroll?: (courseId: string) => void;
  className?: string;
}

/**
 * CourseCard Component
 *
 * Displays course information in a card format with:
 * - Course thumbnail with badges (New, Bestseller, Featured)
 * - Title and short description
 * - Instructor information
 * - Course stats (duration, lessons, students, rating)
 * - Price with optional discount
 * - Clear CTAs (Enroll Now, Learn More)
 *
 * @accessibility
 * - Semantic HTML structure with proper heading hierarchy
 * - ARIA labels for interactive elements
 * - Keyboard navigable
 * - Focus indicators for all interactive elements
 * - Screen reader friendly stats and badges
 */
export function CourseCard({
  course,
  variant = "default",
  onEnroll,
  className,
}: CourseCardProps) {
  const isFeatured = variant === "featured";
  const isCompact = variant === "compact";
  const levelConfig = LEVEL_CONFIG[course.level];
  const categoryConfig = CATEGORY_CONFIG[course.category];

  const handleEnroll = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onEnroll?.(course.id);
  };

  return (
    <Card
      className={cn(
        "group relative overflow-hidden transition-all duration-300",
        "hover:shadow-lg hover:border-primary/20",
        "focus-within:ring-2 focus-within:ring-primary/50",
        isFeatured && "border-primary/30 bg-gradient-to-br from-primary/5 to-transparent",
        className
      )}
    >
      {/* Thumbnail Section */}
      <div className="relative aspect-video overflow-hidden bg-muted">
        {course.thumbnail ? (
          <img
            src={course.thumbnail}
            alt={`${course.title} course thumbnail`}
            className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
            loading="lazy"
          />
        ) : (
          <div
            className={cn(
              "h-full w-full flex items-center justify-center",
              categoryConfig.color,
              "bg-opacity-20"
            )}
            aria-hidden="true"
          >
            <BookOpen className="h-12 w-12 text-muted-foreground/50" />
          </div>
        )}

        {/* Badges Overlay */}
        <div className="absolute top-3 left-3 flex flex-wrap gap-2">
          {course.isNew && (
            <span
              className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full bg-green-500 text-white"
              aria-label="New course"
            >
              <Sparkles className="h-3 w-3" aria-hidden="true" />
              New
            </span>
          )}
          {course.isBestseller && (
            <span
              className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full bg-amber-500 text-white"
              aria-label="Bestseller course"
            >
              <TrendingUp className="h-3 w-3" aria-hidden="true" />
              Bestseller
            </span>
          )}
          {course.isFeatured && (
            <span
              className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full bg-primary text-primary-foreground"
              aria-label="Featured course"
            >
              <Award className="h-3 w-3" aria-hidden="true" />
              Featured
            </span>
          )}
        </div>

        {/* Level Badge */}
        <div className="absolute top-3 right-3">
          <span
            className={cn(
              "px-2 py-1 text-xs font-medium rounded-full",
              levelConfig.color
            )}
          >
            {levelConfig.label}
          </span>
        </div>

        {/* Category Tag */}
        <div className="absolute bottom-3 left-3">
          <span className="px-2 py-1 text-xs font-medium rounded-full bg-background/80 backdrop-blur-sm text-foreground">
            {categoryConfig.label}
          </span>
        </div>
      </div>

      <CardHeader className="pb-2">
        {/* Title */}
        <Link
          href={`/courses/${course.slug}` as unknown as "/courses"}
          className="group/link focus:outline-none"
        >
          <h3
            className={cn(
              "font-semibold text-foreground line-clamp-2 transition-colors",
              "group-hover/link:text-primary",
              isFeatured ? "text-lg" : "text-base"
            )}
          >
            {course.title}
          </h3>
        </Link>

        {/* Short Description */}
        {!isCompact && (
          <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
            {course.shortDescription}
          </p>
        )}
      </CardHeader>

      <CardContent className="pb-3">
        {/* Instructor */}
        <InstructorBadge instructor={course.instructor} variant="compact" />

        {/* Stats Row */}
        <div
          className="flex flex-wrap items-center gap-x-4 gap-y-2 mt-3 text-sm text-muted-foreground"
          role="list"
          aria-label="Course statistics"
        >
          <div
            className="flex items-center gap-1"
            role="listitem"
            aria-label={`Duration: ${course.duration}`}
          >
            <Clock className="h-4 w-4" aria-hidden="true" />
            <span>{course.duration}</span>
          </div>
          <div
            className="flex items-center gap-1"
            role="listitem"
            aria-label={`${course.lessonsCount} lessons`}
          >
            <BookOpen className="h-4 w-4" aria-hidden="true" />
            <span>{course.lessonsCount} lessons</span>
          </div>
          <div
            className="flex items-center gap-1"
            role="listitem"
            aria-label={`${course.enrolledCount.toLocaleString()} students enrolled`}
          >
            <Users className="h-4 w-4" aria-hidden="true" />
            <span>{course.enrolledCount.toLocaleString()}</span>
          </div>
        </div>

        {/* Rating */}
        <div
          className="flex items-center gap-2 mt-3"
          aria-label={`Rating: ${course.rating} out of 5 stars based on ${course.reviewsCount} reviews`}
        >
          <div className="flex items-center gap-1">
            <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" aria-hidden="true" />
            <span className="font-semibold text-foreground">{course.rating.toFixed(1)}</span>
          </div>
          <span className="text-sm text-muted-foreground">
            ({course.reviewsCount.toLocaleString()} reviews)
          </span>
        </div>
      </CardContent>

      <CardFooter className="flex items-center justify-between pt-3 border-t">
        {/* Price */}
        <div className="flex items-baseline gap-2">
          <span className="text-xl font-bold text-foreground">
            ${course.price.toFixed(2)}
          </span>
          {course.originalPrice && course.originalPrice > course.price && (
            <span className="text-sm text-muted-foreground line-through">
              ${course.originalPrice.toFixed(2)}
            </span>
          )}
        </div>

        {/* CTAs */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            asChild
            className="hidden sm:inline-flex"
          >
            <Link href={`/courses/${course.slug}` as unknown as "/courses"}>
              Learn More
            </Link>
          </Button>
          <Button
            size="sm"
            onClick={handleEnroll}
            aria-label={`Enroll in ${course.title}`}
          >
            Enroll Now
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
}

/**
 * CourseCardSkeleton
 *
 * Loading skeleton for the CourseCard component.
 */
export function CourseCardSkeleton({ variant = "default" }: { variant?: "default" | "featured" | "compact" }) {
  return (
    <Card className="overflow-hidden animate-pulse">
      {/* Thumbnail Skeleton */}
      <div className="aspect-video bg-muted" />

      <CardHeader className="pb-2">
        <div className="h-5 bg-muted rounded w-3/4" />
        {variant !== "compact" && (
          <div className="h-4 bg-muted rounded w-full mt-2" />
        )}
      </CardHeader>

      <CardContent className="pb-3">
        {/* Instructor Skeleton */}
        <div className="flex items-center gap-3 py-2">
          <div className="h-10 w-10 rounded-full bg-muted" />
          <div className="flex flex-col gap-2">
            <div className="h-4 bg-muted rounded w-24" />
            <div className="h-3 bg-muted rounded w-16" />
          </div>
        </div>

        {/* Stats Skeleton */}
        <div className="flex gap-4 mt-3">
          <div className="h-4 bg-muted rounded w-16" />
          <div className="h-4 bg-muted rounded w-20" />
          <div className="h-4 bg-muted rounded w-12" />
        </div>

        {/* Rating Skeleton */}
        <div className="flex items-center gap-2 mt-3">
          <div className="h-4 bg-muted rounded w-12" />
          <div className="h-4 bg-muted rounded w-20" />
        </div>
      </CardContent>

      <CardFooter className="flex items-center justify-between pt-3 border-t">
        <div className="h-6 bg-muted rounded w-16" />
        <div className="flex gap-2">
          <div className="h-8 bg-muted rounded w-20" />
          <div className="h-8 bg-muted rounded w-24" />
        </div>
      </CardFooter>
    </Card>
  );
}

export default CourseCard;
