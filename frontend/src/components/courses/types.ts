/**
 * Course Landing Page Types
 * 
 * Type definitions for the courses showcase page components.
 */

export interface Instructor {
  id: string;
  name: string;
  title: string;
  avatar?: string;
  bio?: string;
  expertise: string[];
  rating?: number;
  totalStudents?: number;
}

export interface CourseModule {
  id: string;
  title: string;
  duration: string;
  lessons: number;
}

export interface Course {
  id: string;
  title: string;
  slug: string;
  description: string;
  shortDescription: string;
  thumbnail?: string;
  instructor: Instructor;
  category: CourseCategory;
  level: CourseLevel;
  duration: string;
  lessonsCount: number;
  enrolledCount: number;
  rating: number;
  reviewsCount: number;
  price: number;
  originalPrice?: number;
  isFeatured?: boolean;
  isNew?: boolean;
  isBestseller?: boolean;
  tags: string[];
  modules?: CourseModule[];
  learningOutcomes?: string[];
  prerequisites?: string[];
  lastUpdated: string;
}

export type CourseCategory = 
  | "web-development"
  | "mobile-development"
  | "data-science"
  | "machine-learning"
  | "cloud-computing"
  | "cybersecurity"
  | "devops"
  | "design"
  | "business"
  | "all";

export type CourseLevel = "beginner" | "intermediate" | "advanced" | "all-levels";

export interface CourseFiltersState {
  category: CourseCategory;
  level: CourseLevel;
  priceRange: [number, number];
  rating: number;
  searchQuery: string;
  sortBy: CourseSortOption;
}

export type CourseSortOption = 
  | "popular"
  | "newest"
  | "highest-rated"
  | "price-low"
  | "price-high";

export interface CoursesPageProps {
  initialCourses?: Course[];
  featuredCourses?: Course[];
}

// Category display configuration
export const CATEGORY_CONFIG: Record<CourseCategory, { label: string; icon: string; color: string }> = {
  "web-development": { label: "Web Development", icon: "Globe", color: "bg-blue-500" },
  "mobile-development": { label: "Mobile Development", icon: "Smartphone", color: "bg-green-500" },
  "data-science": { label: "Data Science", icon: "BarChart", color: "bg-purple-500" },
  "machine-learning": { label: "Machine Learning", icon: "Brain", color: "bg-pink-500" },
  "cloud-computing": { label: "Cloud Computing", icon: "Cloud", color: "bg-cyan-500" },
  "cybersecurity": { label: "Cybersecurity", icon: "Shield", color: "bg-red-500" },
  "devops": { label: "DevOps", icon: "GitBranch", color: "bg-orange-500" },
  "design": { label: "Design", icon: "Palette", color: "bg-indigo-500" },
  "business": { label: "Business", icon: "Briefcase", color: "bg-amber-500" },
  "all": { label: "All Categories", icon: "Grid", color: "bg-gray-500" },
};

// Level display configuration
export const LEVEL_CONFIG: Record<CourseLevel, { label: string; color: string }> = {
  "beginner": { label: "Beginner", color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" },
  "intermediate": { label: "Intermediate", color: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200" },
  "advanced": { label: "Advanced", color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200" },
  "all-levels": { label: "All Levels", color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200" },
};