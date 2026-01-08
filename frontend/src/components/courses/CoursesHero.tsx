"use client";

import * as React from "react";
import { Search, GraduationCap, Users, Award, Play } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface CoursesHeroProps {
  title?: string;
  subtitle?: string;
  searchQuery?: string;
  onSearchChange?: (query: string) => void;
  onSearchSubmit?: (query: string) => void;
  stats?: {
    totalCourses: number;
    totalStudents: number;
    totalInstructors: number;
  };
  className?: string;
}

/**
 * CoursesHero Component
 * 
 * Hero section for the courses landing page featuring:
 * - Compelling headline and value proposition
 * - Search functionality
 * - Key statistics
 * - Visual appeal with gradient background
 * 
 * @accessibility
 * - Semantic HTML with proper heading hierarchy
 * - Search input has associated label
 * - Stats are announced to screen readers
 * - Keyboard accessible search
 */
export function CoursesHero({
  title = "Unlock Your Potential",
  subtitle = "Discover world-class courses taught by industry experts. Start learning today and transform your career.",
  searchQuery = "",
  onSearchChange,
  onSearchSubmit,
  stats,
  className,
}: CoursesHeroProps) {
  const [localQuery, setLocalQuery] = React.useState(searchQuery);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearchSubmit?.(localQuery);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setLocalQuery(value);
    onSearchChange?.(value);
  };

  return (
    <section
      className={cn(
        "relative overflow-hidden bg-gradient-to-br from-background via-background to-primary/5",
        "py-16 md:py-24 lg:py-32",
        className
      )}
      aria-labelledby="courses-hero-title"
    >
      {/* Background Decoration */}
      <div
        className="absolute inset-0 bg-grid-pattern opacity-5 pointer-events-none"
        aria-hidden="true"
      />
      <div
        className="absolute top-0 right-0 w-1/2 h-1/2 bg-gradient-to-bl from-primary/10 to-transparent rounded-full blur-3xl"
        aria-hidden="true"
      />
      <div
        className="absolute bottom-0 left-0 w-1/3 h-1/3 bg-gradient-to-tr from-secondary/10 to-transparent rounded-full blur-3xl"
        aria-hidden="true"
      />

      <div className="container relative mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
            <GraduationCap className="h-4 w-4" aria-hidden="true" />
            <span>Learn from the best</span>
          </div>

          {/* Title */}
          <h1
            id="courses-hero-title"
            className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-foreground mb-6"
          >
            {title}
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
            {subtitle}
          </p>

          {/* Search Form */}
          <form
            onSubmit={handleSubmit}
            className="max-w-2xl mx-auto mb-12"
            role="search"
            aria-label="Search courses"
          >
            <div className="relative flex items-center">
              <label htmlFor="course-search" className="sr-only">
                Search for courses
              </label>
              <Search
                className="absolute left-4 h-5 w-5 text-muted-foreground pointer-events-none"
                aria-hidden="true"
              />
              <input
                id="course-search"
                type="search"
                placeholder="Search for courses, topics, or instructors..."
                value={localQuery}
                onChange={handleChange}
                className={cn(
                  "w-full h-14 pl-12 pr-32 rounded-full",
                  "bg-background border border-input",
                  "text-foreground placeholder:text-muted-foreground",
                  "focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent",
                  "transition-all duration-200"
                )}
              />
              <Button
                type="submit"
                size="lg"
                className="absolute right-2 rounded-full"
              >
                Search
              </Button>
            </div>
          </form>

          {/* Quick Actions */}
          <div className="flex flex-wrap items-center justify-center gap-4 mb-12">
            <Button variant="outline" size="lg" className="gap-2">
              <Play className="h-4 w-4" aria-hidden="true" />
              Watch Demo
            </Button>
            <Button variant="ghost" size="lg">
              Browse All Courses
            </Button>
          </div>

          {/* Stats */}
          {stats && (
            <div
              className="grid grid-cols-1 sm:grid-cols-3 gap-8 max-w-3xl mx-auto"
              role="list"
              aria-label="Platform statistics"
            >
              <div
                className="flex flex-col items-center p-6 rounded-2xl bg-card/50 backdrop-blur-sm border"
                role="listitem"
              >
                <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary/10 text-primary mb-3">
                  <GraduationCap className="h-6 w-6" aria-hidden="true" />
                </div>
                <span className="text-3xl font-bold text-foreground">
                  {stats.totalCourses.toLocaleString()}+
                </span>
                <span className="text-sm text-muted-foreground">Courses</span>
              </div>

              <div
                className="flex flex-col items-center p-6 rounded-2xl bg-card/50 backdrop-blur-sm border"
                role="listitem"
              >
                <div className="flex items-center justify-center h-12 w-12 rounded-full bg-green-500/10 text-green-500 mb-3">
                  <Users className="h-6 w-6" aria-hidden="true" />
                </div>
                <span className="text-3xl font-bold text-foreground">
                  {stats.totalStudents.toLocaleString()}+
                </span>
                <span className="text-sm text-muted-foreground">Students</span>
              </div>

              <div
                className="flex flex-col items-center p-6 rounded-2xl bg-card/50 backdrop-blur-sm border"
                role="listitem"
              >
                <div className="flex items-center justify-center h-12 w-12 rounded-full bg-amber-500/10 text-amber-500 mb-3">
                  <Award className="h-6 w-6" aria-hidden="true" />
                </div>
                <span className="text-3xl font-bold text-foreground">
                  {stats.totalInstructors.toLocaleString()}+
                </span>
                <span className="text-sm text-muted-foreground">Expert Instructors</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export default CoursesHero;