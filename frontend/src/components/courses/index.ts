/**
 * Courses Components
 * 
 * A comprehensive set of components for building a courses landing page.
 * 
 * @example
 * ```tsx
 * import { CoursesPage, CourseCard, CoursesHero } from "@/components/courses";
 * 
 * // Use the full page component
 * <CoursesPage />
 * 
 * // Or compose your own layout
 * <CoursesHero />
 * <CourseFilters />
 * <CourseCard course={course} />
 * ```
 */

// Main page component
export { CoursesPage, default as CoursesPageDefault } from "./CoursesPage";

// Individual components
export { CourseCard, CourseCardSkeleton } from "./CourseCard";
export { CoursesHero } from "./CoursesHero";
export { CourseFilters } from "./CourseFilters";
export { InstructorBadge, InstructorBadgeSkeleton } from "./InstructorBadge";

// Types
export type {
  Course,
  CourseCategory,
  CourseLevel,
  CourseModule,
  CourseFiltersState,
  CourseSortOption,
  CoursesPageProps,
  Instructor,
} from "./types";

// Configuration
export { CATEGORY_CONFIG, LEVEL_CONFIG } from "./types";