# Courses Landing Page Design Documentation

**Version:** 1.0.0  
**Last Updated:** 2025-12-19  
**Author:** Web Design Consultant

---

## Executive Summary

This document outlines the design specifications for a courses landing page (white page) that showcases courses with course listings, brief descriptions, instructor highlights, and clear calls to action.

---

## 1. Information Architecture

### 1.1 Page Structure

```
┌─────────────────────────────────────────────────────────────┐
│                        HERO SECTION                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Badge: "Learn from the best"                           ││
│  │  Headline: "Unlock Your Potential"                      ││
│  │  Subheadline: Value proposition                         ││
│  │  [Search Bar with CTA]                                  ││
│  │  Quick Actions: [Watch Demo] [Browse All]               ││
│  │  Stats: Courses | Students | Instructors                ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                      FILTER BAR (Sticky)                     │
│  [Category ▼] [Level ▼] [Clear] | Results | [Sort ▼] [Grid]│
│  Active Filters: [Tag] [Tag] [×]                            │
├─────────────────────────────────────────────────────────────┤
│                      COURSE GRID                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  Course  │ │  Course  │ │  Course  │ │  Course  │       │
│  │   Card   │ │   Card   │ │   Card   │ │   Card   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  Course  │ │  Course  │ │  Course  │ │  Course  │       │
│  │   Card   │ │   Card   │ │   Card   │ │   Card   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                    [Load More Courses]                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Hierarchy

```
CoursesPage
├── CoursesHero
│   ├── Badge
│   ├── Title
│   ├── Subtitle
│   ├── SearchForm
│   ├── QuickActions
│   └── StatsGrid
├── CourseFilters
│   ├── CategoryDropdown
│   ├── LevelDropdown
│   ├── ClearButton
│   ├── ResultsCount
│   ├── SortDropdown
│   ├── ViewModeToggle
│   └── ActiveFilterTags
└── CourseGrid
    └── CourseCard (×n)
        ├── Thumbnail
        │   ├── Image
        │   ├── Badges (New, Bestseller, Featured)
        │   ├── LevelBadge
        │   └── CategoryTag
        ├── CardHeader
        │   ├── Title (Link)
        │   └── ShortDescription
        ├── CardContent
        │   ├── InstructorBadge
        │   │   ├── Avatar
        │   │   ├── Name
        │   │   └── Title
        │   ├── StatsRow
        │   │   ├── Duration
        │   │   ├── Lessons
        │   │   └── Students
        │   └── Rating
        └── CardFooter
            ├── Price
            └── CTAs
                ├── LearnMore
                └── EnrollNow
```

---

## 2. Visual Design Specifications

### 2.1 Color Palette

The design uses CSS custom properties for theming support:

| Token | Light Mode | Dark Mode | Usage |
|-------|------------|-----------|-------|
| `--background` | `#ffffff` | `#0a0a0a` | Page background |
| `--foreground` | `#0a0a0a` | `#fafafa` | Primary text |
| `--primary` | `#2563eb` | `#3b82f6` | CTAs, links, accents |
| `--muted` | `#f4f4f5` | `#27272a` | Secondary backgrounds |
| `--muted-foreground` | `#71717a` | `#a1a1aa` | Secondary text |

### 2.2 Typography

| Element | Font | Size | Weight | Line Height |
|---------|------|------|--------|-------------|
| Hero Title | Geist Sans | 48-60px | 700 | 1.1 |
| Hero Subtitle | Geist Sans | 18-20px | 400 | 1.6 |
| Card Title | Geist Sans | 16-18px | 600 | 1.3 |
| Card Description | Geist Sans | 14px | 400 | 1.5 |
| Stats | Geist Sans | 30px | 700 | 1.2 |
| Labels | Geist Sans | 12-14px | 500 | 1.4 |

### 2.3 Spacing System

Based on 4px grid:

| Token | Value | Usage |
|-------|-------|-------|
| `space-1` | 4px | Tight spacing |
| `space-2` | 8px | Component internal |
| `space-3` | 12px | Small gaps |
| `space-4` | 16px | Standard gaps |
| `space-6` | 24px | Section padding |
| `space-8` | 32px | Large gaps |
| `space-12` | 48px | Section margins |

### 2.4 Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `radius-sm` | 4px | Small elements |
| `radius-md` | 8px | Cards, inputs |
| `radius-lg` | 12px | Large cards |
| `radius-xl` | 16px | Hero elements |
| `radius-full` | 9999px | Pills, avatars |

---

## 3. Component Specifications

### 3.1 CourseCard

**Purpose:** Display course information in a scannable, visually appealing format.

**Variants:**
- `default` - Standard card layout
- `featured` - Highlighted with border accent
- `compact` - Reduced information for list view

**Props:**
```typescript
interface CourseCardProps {
  course: Course;
  variant?: "default" | "featured" | "compact";
  onEnroll?: (courseId: string) => void;
  className?: string;
}
```

**Visual States:**
- Default: Standard appearance
- Hover: Elevated shadow, slight scale
- Focus: Ring outline for accessibility
- Featured: Primary border accent, gradient background

### 3.2 InstructorBadge

**Purpose:** Display instructor information with credibility indicators.

**Variants:**
- `compact` - Avatar + name + title (for cards)
- `detailed` - Includes stats (for course pages)

**Props:**
```typescript
interface InstructorBadgeProps {
  instructor: Instructor;
  variant?: "compact" | "detailed";
  showStats?: boolean;
  className?: string;
}
```

### 3.3 CoursesHero

**Purpose:** Capture attention and communicate value proposition.

**Features:**
- Gradient background with decorative elements
- Prominent search functionality
- Platform statistics for social proof
- Quick action buttons

### 3.4 CourseFilters

**Purpose:** Enable users to find relevant courses quickly.

**Features:**
- Category dropdown with color indicators
- Level dropdown with styled badges
- Sort options (Popular, Newest, Rating, Price)
- View mode toggle (Grid/List)
- Active filter tags with remove buttons
- Sticky positioning for easy access

---

## 4. Responsive Design

### 4.1 Breakpoints

| Breakpoint | Width | Grid Columns |
|------------|-------|--------------|
| Mobile | < 640px | 1 |
| Tablet | 640-1024px | 2 |
| Desktop | 1024-1280px | 3 |
| Large | > 1280px | 4 |

### 4.2 Mobile Adaptations

- Hero: Stacked layout, smaller text
- Filters: Horizontal scroll, collapsed dropdowns
- Cards: Full width, stacked CTAs
- Stats: Vertical stack

---

## 5. Accessibility (WCAG 2.1 AA)

### 5.1 Implemented Features

| Feature | Implementation |
|---------|----------------|
| Semantic HTML | Proper heading hierarchy, landmarks |
| ARIA Labels | All interactive elements labeled |
| Keyboard Navigation | Full tab support, focus indicators |
| Screen Reader | Live regions for filter changes |
| Skip Links | Jump to main content |
| Color Contrast | 4.5:1 minimum ratio |
| Focus Indicators | Visible ring on all focusable elements |

### 5.2 Testing Checklist

- [ ] Navigate entire page with keyboard only
- [ ] Test with screen reader (NVDA/VoiceOver)
- [ ] Verify color contrast ratios
- [ ] Test at 200% zoom
- [ ] Verify focus order is logical

---

## 6. Interactive Features

### 6.1 Search

- Real-time filtering as user types
- Searches: title, description, instructor, tags
- Debounced input (300ms)

### 6.2 Filtering

- Multi-select categories
- Single-select level
- Persistent filter state in URL
- Clear all filters button

### 6.3 Sorting

- Popular (default) - by enrollment count
- Newest - by last updated date
- Highest Rated - by rating
- Price Low/High - by price

### 6.4 Pagination

- Load more button (not infinite scroll)
- 8 courses per page
- Smooth scroll to new content

---

## 7. Performance Considerations

### 7.1 Image Optimization

- Lazy loading for course thumbnails
- WebP format with fallbacks
- Responsive images with srcset
- Placeholder skeletons during load

### 7.2 Code Splitting

- Dynamic imports for filter dropdowns
- Skeleton components for loading states
- Memoized filter calculations

### 7.3 Caching

- React Query for API responses
- 60-second stale time
- Background refetch on focus

---

## 8. File Structure

```
frontend/src/components/courses/
├── index.ts              # Barrel exports
├── types.ts              # TypeScript interfaces
├── CourseCard.tsx        # Course card component
├── CoursesHero.tsx       # Hero section
├── CourseFilters.tsx     # Filter bar
├── InstructorBadge.tsx   # Instructor display
└── CoursesPage.tsx       # Main page component

frontend/src/app/courses/
└── page.tsx              # Next.js route
```

---

## 9. Usage Examples

### 9.1 Full Page

```tsx
import { CoursesPage } from "@/components/courses";

export default function Page() {
  return <CoursesPage />;
}
```

### 9.2 Custom Layout

```tsx
import {
  CoursesHero,
  CourseFilters,
  CourseCard,
  type Course,
} from "@/components/courses";

export default function CustomCoursesPage({ courses }: { courses: Course[] }) {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);

  return (
    <div>
      <CoursesHero onSearchSubmit={(q) => setFilters({ ...filters, searchQuery: q })} />
      <CourseFilters filters={filters} onFiltersChange={setFilters} />
      <div className="grid grid-cols-3 gap-6">
        {courses.map((course) => (
          <CourseCard key={course.id} course={course} />
        ))}
      </div>
    </div>
  );
}
```

### 9.3 Individual Card

```tsx
import { CourseCard } from "@/components/courses";

<CourseCard
  course={course}
  variant="featured"
  onEnroll={(id) => router.push(`/checkout/${id}`)}
/>
```

---

## 10. Future Enhancements

### Phase 2
- [ ] Course detail page
- [ ] Instructor profile pages
- [ ] Course preview modal
- [ ] Wishlist functionality

### Phase 3
- [ ] Course comparison
- [ ] Learning paths
- [ ] Personalized recommendations
- [ ] Progress tracking

---

## Appendix A: Sample Data Schema

```typescript
interface Course {
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
  lastUpdated: string;
}

interface Instructor {
  id: string;
  name: string;
  title: string;
  avatar?: string;
  expertise: string[];
  rating?: number;
  totalStudents?: number;
}
```

---

**Document Status:** Complete  
**Review Date:** 2025-01-19