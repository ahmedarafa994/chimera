"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { CoursesHero } from "./CoursesHero";
import { CourseFilters } from "./CourseFilters";
import { CourseCard, CourseCardSkeleton } from "./CourseCard";
import type { Course, CourseFiltersState, CoursesPageProps } from "./types";

// Sample course data for demonstration
const SAMPLE_COURSES: Course[] = [
  {
    id: "1",
    title: "Complete Web Development Bootcamp 2024",
    slug: "complete-web-development-bootcamp-2024",
    description: "Learn web development from scratch with HTML, CSS, JavaScript, React, Node.js, and more.",
    shortDescription: "Master full-stack web development with modern technologies and best practices.",
    thumbnail: "/images/courses/web-dev.jpg",
    instructor: {
      id: "inst-1",
      name: "Sarah Johnson",
      title: "Senior Software Engineer at Google",
      expertise: ["React", "Node.js", "TypeScript"],
      rating: 4.9,
      totalStudents: 125000,
    },
    category: "web-development",
    level: "beginner",
    duration: "52 hours",
    lessonsCount: 420,
    enrolledCount: 45230,
    rating: 4.8,
    reviewsCount: 12450,
    price: 89.99,
    originalPrice: 199.99,
    isBestseller: true,
    tags: ["HTML", "CSS", "JavaScript", "React", "Node.js"],
    lastUpdated: "2024-01-15",
  },
  {
    id: "2",
    title: "Machine Learning A-Z: AI, Python & R",
    slug: "machine-learning-az-ai-python-r",
    description: "Learn to create Machine Learning Algorithms in Python and R from two Data Science experts.",
    shortDescription: "Master machine learning with hands-on projects using Python and R.",
    thumbnail: "/images/courses/ml.jpg",
    instructor: {
      id: "inst-2",
      name: "Dr. Michael Chen",
      title: "AI Research Lead at OpenAI",
      expertise: ["Machine Learning", "Deep Learning", "Python"],
      rating: 4.9,
      totalStudents: 89000,
    },
    category: "machine-learning",
    level: "intermediate",
    duration: "44 hours",
    lessonsCount: 320,
    enrolledCount: 32100,
    rating: 4.7,
    reviewsCount: 8920,
    price: 94.99,
    originalPrice: 189.99,
    isFeatured: true,
    tags: ["Python", "R", "TensorFlow", "Scikit-learn"],
    lastUpdated: "2024-02-01",
  },
  {
    id: "3",
    title: "iOS & Swift - The Complete iOS App Development",
    slug: "ios-swift-complete-app-development",
    description: "From beginner to iOS App Developer with just one course. Includes SwiftUI and UIKit.",
    shortDescription: "Build iOS apps from scratch using Swift, SwiftUI, and UIKit.",
    thumbnail: "/images/courses/ios.jpg",
    instructor: {
      id: "inst-3",
      name: "Angela Yu",
      title: "Lead iOS Developer & Instructor",
      expertise: ["Swift", "SwiftUI", "iOS Development"],
      rating: 4.8,
      totalStudents: 156000,
    },
    category: "mobile-development",
    level: "beginner",
    duration: "60 hours",
    lessonsCount: 540,
    enrolledCount: 67800,
    rating: 4.9,
    reviewsCount: 18200,
    price: 84.99,
    originalPrice: 179.99,
    isNew: true,
    isBestseller: true,
    tags: ["Swift", "SwiftUI", "UIKit", "Xcode"],
    lastUpdated: "2024-01-20",
  },
  {
    id: "4",
    title: "AWS Certified Solutions Architect 2024",
    slug: "aws-certified-solutions-architect-2024",
    description: "Pass the AWS Certified Solutions Architect Associate exam with confidence.",
    shortDescription: "Comprehensive AWS certification prep with hands-on labs and practice exams.",
    thumbnail: "/images/courses/aws.jpg",
    instructor: {
      id: "inst-4",
      name: "Ryan Kroonenburg",
      title: "AWS Solutions Architect & Cloud Expert",
      expertise: ["AWS", "Cloud Architecture", "DevOps"],
      rating: 4.7,
      totalStudents: 210000,
    },
    category: "cloud-computing",
    level: "intermediate",
    duration: "28 hours",
    lessonsCount: 180,
    enrolledCount: 89500,
    rating: 4.6,
    reviewsCount: 24100,
    price: 79.99,
    originalPrice: 149.99,
    tags: ["AWS", "Cloud", "EC2", "S3", "Lambda"],
    lastUpdated: "2024-01-10",
  },
  {
    id: "5",
    title: "Ethical Hacking & Penetration Testing",
    slug: "ethical-hacking-penetration-testing",
    description: "Learn ethical hacking, penetration testing, web testing, and wifi hacking using Kali Linux.",
    shortDescription: "Master cybersecurity skills with real-world hacking techniques and tools.",
    thumbnail: "/images/courses/security.jpg",
    instructor: {
      id: "inst-5",
      name: "Nathan House",
      title: "CEO of StationX & Security Expert",
      expertise: ["Penetration Testing", "Network Security", "Kali Linux"],
      rating: 4.8,
      totalStudents: 78000,
    },
    category: "cybersecurity",
    level: "advanced",
    duration: "36 hours",
    lessonsCount: 280,
    enrolledCount: 34200,
    rating: 4.7,
    reviewsCount: 9800,
    price: 99.99,
    originalPrice: 199.99,
    isFeatured: true,
    tags: ["Kali Linux", "Penetration Testing", "Network Security"],
    lastUpdated: "2024-02-05",
  },
  {
    id: "6",
    title: "UI/UX Design Masterclass",
    slug: "ui-ux-design-masterclass",
    description: "Learn UI/UX design from scratch. Create beautiful designs using Figma.",
    shortDescription: "Design stunning user interfaces and experiences with industry-standard tools.",
    thumbnail: "/images/courses/design.jpg",
    instructor: {
      id: "inst-6",
      name: "Daniel Scott",
      title: "Adobe Certified Instructor",
      expertise: ["Figma", "UI Design", "UX Research"],
      rating: 4.9,
      totalStudents: 92000,
    },
    category: "design",
    level: "beginner",
    duration: "24 hours",
    lessonsCount: 160,
    enrolledCount: 28900,
    rating: 4.8,
    reviewsCount: 7600,
    price: 74.99,
    originalPrice: 159.99,
    isNew: true,
    tags: ["Figma", "UI Design", "UX Design", "Prototyping"],
    lastUpdated: "2024-01-25",
  },
];

const DEFAULT_FILTERS: CourseFiltersState = {
  category: "all",
  level: "all-levels",
  priceRange: [0, 200],
  rating: 0,
  searchQuery: "",
  sortBy: "popular",
};

/**
 * CoursesPage Component
 * 
 * Main courses landing page that combines:
 * - Hero section with search and stats
 * - Filter bar with category, level, and sort options
 * - Responsive course grid
 * - Loading states and empty states
 * 
 * @accessibility
 * - Semantic page structure with landmarks
 * - Skip links for keyboard navigation
 * - Live region for filter result announcements
 * - Responsive design for all screen sizes
 */
export function CoursesPage({
  initialCourses = SAMPLE_COURSES,
}: CoursesPageProps) {
  const [courses] = React.useState<Course[]>(initialCourses);
  const [filters, setFilters] = React.useState<CourseFiltersState>(DEFAULT_FILTERS);
  const [viewMode, setViewMode] = React.useState<"grid" | "list">("grid");
  const [isLoading, setIsLoading] = React.useState(false);

  // Filter courses based on current filters
  const filteredCourses = React.useMemo(() => {
    let result = [...courses];

    // Category filter
    if (filters.category !== "all") {
      result = result.filter((course) => course.category === filters.category);
    }

    // Level filter
    if (filters.level !== "all-levels") {
      result = result.filter((course) => course.level === filters.level);
    }

    // Search filter
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      result = result.filter(
        (course) =>
          course.title.toLowerCase().includes(query) ||
          course.shortDescription.toLowerCase().includes(query) ||
          course.instructor.name.toLowerCase().includes(query) ||
          course.tags.some((tag) => tag.toLowerCase().includes(query))
      );
    }

    // Sort
    switch (filters.sortBy) {
      case "newest":
        result.sort((a, b) => new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime());
        break;
      case "highest-rated":
        result.sort((a, b) => b.rating - a.rating);
        break;
      case "price-low":
        result.sort((a, b) => a.price - b.price);
        break;
      case "price-high":
        result.sort((a, b) => b.price - a.price);
        break;
      case "popular":
      default:
        result.sort((a, b) => b.enrolledCount - a.enrolledCount);
        break;
    }

    return result;
  }, [courses, filters]);

  const handleFiltersChange = (newFilters: Partial<CourseFiltersState>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
  };

  const handleClearFilters = () => {
    setFilters(DEFAULT_FILTERS);
  };

  const handleSearch = (query: string) => {
    setFilters((prev) => ({ ...prev, searchQuery: query }));
  };

  const handleEnroll = (courseId: string) => {
    // Handle enrollment - could open modal, redirect to checkout, etc.
    console.log("Enrolling in course:", courseId);
    alert(`Enrolling in course: ${courseId}`);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Skip Link for Accessibility */}
      <a
        href="#courses-grid"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-md"
      >
        Skip to courses
      </a>

      {/* Hero Section */}
      <CoursesHero
        title="Unlock Your Potential"
        subtitle="Discover world-class courses taught by industry experts. Start learning today and transform your career."
        onSearchChange={handleSearch}
        onSearchSubmit={handleSearch}
        stats={{
          totalCourses: 500,
          totalStudents: 250000,
          totalInstructors: 150,
        }}
      />

      {/* Filters */}
      <CourseFilters
        filters={filters}
        onFiltersChange={handleFiltersChange}
        onClearFilters={handleClearFilters}
        totalResults={filteredCourses.length}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
      />

      {/* Main Content */}
      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Results Announcement for Screen Readers */}
        <div
          role="status"
          aria-live="polite"
          aria-atomic="true"
          className="sr-only"
        >
          {filteredCourses.length} courses found
        </div>

        {/* Course Grid */}
        <section
          id="courses-grid"
          aria-label="Course listings"
          className={cn(
            "grid gap-6",
            viewMode === "grid"
              ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
              : "grid-cols-1 max-w-4xl mx-auto"
          )}
        >
          {isLoading ? (
            // Loading Skeletons
            Array.from({ length: 8 }).map((_, index) => (
              <CourseCardSkeleton key={index} />
            ))
          ) : filteredCourses.length > 0 ? (
            // Course Cards
            filteredCourses.map((course) => (
              <CourseCard
                key={course.id}
                course={course}
                variant={course.isFeatured ? "featured" : "default"}
                onEnroll={handleEnroll}
              />
            ))
          ) : (
            // Empty State
            <div className="col-span-full flex flex-col items-center justify-center py-16 text-center">
              <div className="h-24 w-24 rounded-full bg-muted flex items-center justify-center mb-6">
                <svg
                  className="h-12 w-12 text-muted-foreground"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">
                No courses found
              </h3>
              <p className="text-muted-foreground max-w-md mb-6">
                We couldn&apos;t find any courses matching your criteria. Try adjusting
                your filters or search terms.
              </p>
              <button
                onClick={handleClearFilters}
                className="text-primary hover:underline font-medium"
              >
                Clear all filters
              </button>
            </div>
          )}
        </section>

        {/* Load More (for pagination) */}
        {filteredCourses.length > 0 && filteredCourses.length >= 8 && (
          <div className="flex justify-center mt-12">
            <button
              className={cn(
                "px-8 py-3 rounded-full border-2 border-primary",
                "text-primary font-medium",
                "hover:bg-primary hover:text-primary-foreground",
                "transition-colors duration-200",
                "focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
              )}
            >
              Load More Courses
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default CoursesPage;