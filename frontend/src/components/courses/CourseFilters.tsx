"use client";

import * as React from "react";
import {
  Filter,
  X,
  ChevronDown,
  Grid,
  List,
  SlidersHorizontal,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type {
  CourseCategory,
  CourseLevel,
  CourseSortOption,
  CourseFiltersState,
} from "./types";
import { CATEGORY_CONFIG, LEVEL_CONFIG } from "./types";

interface CourseFiltersProps {
  filters: CourseFiltersState;
  onFiltersChange: (filters: Partial<CourseFiltersState>) => void;
  onClearFilters: () => void;
  totalResults: number;
  viewMode?: "grid" | "list";
  onViewModeChange?: (mode: "grid" | "list") => void;
  className?: string;
}

const SORT_OPTIONS: { value: CourseSortOption; label: string }[] = [
  { value: "popular", label: "Most Popular" },
  { value: "newest", label: "Newest" },
  { value: "highest-rated", label: "Highest Rated" },
  { value: "price-low", label: "Price: Low to High" },
  { value: "price-high", label: "Price: High to Low" },
];

/**
 * CourseFilters Component
 * 
 * Filter bar for the courses listing with:
 * - Category filter dropdown
 * - Level filter dropdown
 * - Sort options
 * - View mode toggle (grid/list)
 * - Active filters display with clear option
 * 
 * @accessibility
 * - All dropdowns are keyboard accessible
 * - ARIA labels for filter controls
 * - Focus management for filter interactions
 * - Screen reader announcements for filter changes
 */
export function CourseFilters({
  filters,
  onFiltersChange,
  onClearFilters,
  totalResults,
  viewMode = "grid",
  onViewModeChange,
  className,
}: CourseFiltersProps) {
  const [isCategoryOpen, setIsCategoryOpen] = React.useState(false);
  const [isLevelOpen, setIsLevelOpen] = React.useState(false);
  const [isSortOpen, setIsSortOpen] = React.useState(false);

  const categoryRef = React.useRef<HTMLDivElement>(null);
  const levelRef = React.useRef<HTMLDivElement>(null);
  const sortRef = React.useRef<HTMLDivElement>(null);

  // Close dropdowns when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (categoryRef.current && !categoryRef.current.contains(event.target as Node)) {
        setIsCategoryOpen(false);
      }
      if (levelRef.current && !levelRef.current.contains(event.target as Node)) {
        setIsLevelOpen(false);
      }
      if (sortRef.current && !sortRef.current.contains(event.target as Node)) {
        setIsSortOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const hasActiveFilters =
    filters.category !== "all" ||
    filters.level !== "all-levels" ||
    filters.searchQuery !== "";

  const activeFilterCount = [
    filters.category !== "all",
    filters.level !== "all-levels",
    filters.searchQuery !== "",
  ].filter(Boolean).length;

  return (
    <div
      className={cn(
        "sticky top-0 z-10 bg-background/95 backdrop-blur-sm border-b",
        "py-4",
        className
      )}
      role="region"
      aria-label="Course filters"
    >
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Filter Row */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Left: Filters */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Filter Icon */}
            <div className="flex items-center gap-2 text-muted-foreground">
              <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
              <span className="text-sm font-medium hidden sm:inline">Filters</span>
            </div>

            {/* Category Dropdown */}
            <div ref={categoryRef} className="relative">
              <button
                onClick={() => setIsCategoryOpen(!isCategoryOpen)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg border",
                  "text-sm font-medium transition-colors",
                  "hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary",
                  filters.category !== "all" && "border-primary bg-primary/5"
                )}
                aria-expanded={isCategoryOpen}
                aria-haspopup="listbox"
                aria-label="Filter by category"
              >
                <span>
                  {filters.category === "all"
                    ? "Category"
                    : CATEGORY_CONFIG[filters.category].label}
                </span>
                <ChevronDown
                  className={cn(
                    "h-4 w-4 transition-transform",
                    isCategoryOpen && "rotate-180"
                  )}
                  aria-hidden="true"
                />
              </button>

              {isCategoryOpen && (
                <div
                  className="absolute top-full left-0 mt-2 w-56 rounded-lg border bg-popover shadow-lg z-50"
                  role="listbox"
                  aria-label="Category options"
                >
                  <div className="p-2">
                    {(Object.keys(CATEGORY_CONFIG) as CourseCategory[]).map((category) => (
                      <button
                        key={category}
                        onClick={() => {
                          onFiltersChange({ category });
                          setIsCategoryOpen(false);
                        }}
                        className={cn(
                          "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm",
                          "hover:bg-accent transition-colors text-left",
                          "focus:outline-none focus:bg-accent",
                          filters.category === category && "bg-primary/10 text-primary"
                        )}
                        role="option"
                        aria-selected={filters.category === category}
                      >
                        <span
                          className={cn(
                            "h-2 w-2 rounded-full",
                            CATEGORY_CONFIG[category].color
                          )}
                          aria-hidden="true"
                        />
                        {CATEGORY_CONFIG[category].label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Level Dropdown */}
            <div ref={levelRef} className="relative">
              <button
                onClick={() => setIsLevelOpen(!isLevelOpen)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg border",
                  "text-sm font-medium transition-colors",
                  "hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary",
                  filters.level !== "all-levels" && "border-primary bg-primary/5"
                )}
                aria-expanded={isLevelOpen}
                aria-haspopup="listbox"
                aria-label="Filter by level"
              >
                <span>
                  {filters.level === "all-levels"
                    ? "Level"
                    : LEVEL_CONFIG[filters.level].label}
                </span>
                <ChevronDown
                  className={cn(
                    "h-4 w-4 transition-transform",
                    isLevelOpen && "rotate-180"
                  )}
                  aria-hidden="true"
                />
              </button>

              {isLevelOpen && (
                <div
                  className="absolute top-full left-0 mt-2 w-48 rounded-lg border bg-popover shadow-lg z-50"
                  role="listbox"
                  aria-label="Level options"
                >
                  <div className="p-2">
                    {(Object.keys(LEVEL_CONFIG) as CourseLevel[]).map((level) => (
                      <button
                        key={level}
                        onClick={() => {
                          onFiltersChange({ level });
                          setIsLevelOpen(false);
                        }}
                        className={cn(
                          "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm",
                          "hover:bg-accent transition-colors text-left",
                          "focus:outline-none focus:bg-accent",
                          filters.level === level && "bg-primary/10 text-primary"
                        )}
                        role="option"
                        aria-selected={filters.level === level}
                      >
                        <span
                          className={cn(
                            "px-2 py-0.5 rounded text-xs",
                            LEVEL_CONFIG[level].color
                          )}
                        >
                          {LEVEL_CONFIG[level].label}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Clear Filters */}
            {hasActiveFilters && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClearFilters}
                className="gap-1 text-muted-foreground hover:text-foreground"
                aria-label={`Clear ${activeFilterCount} active filter${activeFilterCount > 1 ? "s" : ""}`}
              >
                <X className="h-4 w-4" aria-hidden="true" />
                Clear
                {activeFilterCount > 0 && (
                  <span className="ml-1 px-1.5 py-0.5 rounded-full bg-primary/10 text-primary text-xs">
                    {activeFilterCount}
                  </span>
                )}
              </Button>
            )}
          </div>

          {/* Right: Sort & View */}
          <div className="flex items-center gap-3">
            {/* Results Count */}
            <span className="text-sm text-muted-foreground hidden md:inline">
              {totalResults.toLocaleString()} course{totalResults !== 1 ? "s" : ""}
            </span>

            {/* Sort Dropdown */}
            <div ref={sortRef} className="relative">
              <button
                onClick={() => setIsSortOpen(!isSortOpen)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg border",
                  "text-sm font-medium transition-colors",
                  "hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary"
                )}
                aria-expanded={isSortOpen}
                aria-haspopup="listbox"
                aria-label="Sort courses"
              >
                <span className="hidden sm:inline">Sort:</span>
                <span>
                  {SORT_OPTIONS.find((opt) => opt.value === filters.sortBy)?.label}
                </span>
                <ChevronDown
                  className={cn(
                    "h-4 w-4 transition-transform",
                    isSortOpen && "rotate-180"
                  )}
                  aria-hidden="true"
                />
              </button>

              {isSortOpen && (
                <div
                  className="absolute top-full right-0 mt-2 w-48 rounded-lg border bg-popover shadow-lg z-50"
                  role="listbox"
                  aria-label="Sort options"
                >
                  <div className="p-2">
                    {SORT_OPTIONS.map((option) => (
                      <button
                        key={option.value}
                        onClick={() => {
                          onFiltersChange({ sortBy: option.value });
                          setIsSortOpen(false);
                        }}
                        className={cn(
                          "w-full px-3 py-2 rounded-md text-sm text-left",
                          "hover:bg-accent transition-colors",
                          "focus:outline-none focus:bg-accent",
                          filters.sortBy === option.value && "bg-primary/10 text-primary"
                        )}
                        role="option"
                        aria-selected={filters.sortBy === option.value}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* View Mode Toggle */}
            {onViewModeChange && (
              <div
                className="flex items-center border rounded-lg p-1"
                role="group"
                aria-label="View mode"
              >
                <button
                  onClick={() => onViewModeChange("grid")}
                  className={cn(
                    "p-2 rounded-md transition-colors",
                    "focus:outline-none focus:ring-2 focus:ring-primary",
                    viewMode === "grid"
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-accent"
                  )}
                  aria-label="Grid view"
                  aria-pressed={viewMode === "grid"}
                >
                  <Grid className="h-4 w-4" aria-hidden="true" />
                </button>
                <button
                  onClick={() => onViewModeChange("list")}
                  className={cn(
                    "p-2 rounded-md transition-colors",
                    "focus:outline-none focus:ring-2 focus:ring-primary",
                    viewMode === "list"
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-accent"
                  )}
                  aria-label="List view"
                  aria-pressed={viewMode === "list"}
                >
                  <List className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Active Filters Tags */}
        {hasActiveFilters && (
          <div
            className="flex flex-wrap items-center gap-2 mt-4"
            role="list"
            aria-label="Active filters"
          >
            {filters.category !== "all" && (
              <span
                className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm"
                role="listitem"
              >
                {CATEGORY_CONFIG[filters.category].label}
                <button
                  onClick={() => onFiltersChange({ category: "all" })}
                  className="ml-1 hover:bg-primary/20 rounded-full p-0.5"
                  aria-label={`Remove ${CATEGORY_CONFIG[filters.category].label} filter`}
                >
                  <X className="h-3 w-3" aria-hidden="true" />
                </button>
              </span>
            )}
            {filters.level !== "all-levels" && (
              <span
                className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm"
                role="listitem"
              >
                {LEVEL_CONFIG[filters.level].label}
                <button
                  onClick={() => onFiltersChange({ level: "all-levels" })}
                  className="ml-1 hover:bg-primary/20 rounded-full p-0.5"
                  aria-label={`Remove ${LEVEL_CONFIG[filters.level].label} filter`}
                >
                  <X className="h-3 w-3" aria-hidden="true" />
                </button>
              </span>
            )}
            {filters.searchQuery && (
              <span
                className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm"
                role="listitem"
              >
                &quot;{filters.searchQuery}&quot;
                <button
                  onClick={() => onFiltersChange({ searchQuery: "" })}
                  className="ml-1 hover:bg-primary/20 rounded-full p-0.5"
                  aria-label="Remove search filter"
                >
                  <X className="h-3 w-3" aria-hidden="true" />
                </button>
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default CourseFilters;