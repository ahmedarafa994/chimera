import { Metadata } from "next";
import { CoursesPage } from "@/components/courses";

export const metadata: Metadata = {
  title: "Courses | Learn from Industry Experts",
  description:
    "Discover world-class courses taught by industry experts. Master web development, machine learning, cloud computing, and more. Start learning today.",
  keywords: [
    "online courses",
    "web development",
    "machine learning",
    "programming",
    "cloud computing",
    "cybersecurity",
    "design",
  ],
  openGraph: {
    title: "Courses | Learn from Industry Experts",
    description:
      "Discover world-class courses taught by industry experts. Start learning today.",
    type: "website",
    images: [
      {
        url: "/images/og/courses.jpg",
        width: 1200,
        height: 630,
        alt: "Online Courses Platform",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Courses | Learn from Industry Experts",
    description:
      "Discover world-class courses taught by industry experts. Start learning today.",
  },
};

/**
 * Courses Landing Page
 * 
 * A white page showcasing courses with:
 * - Hero section with search and platform stats
 * - Filter bar for category, level, and sorting
 * - Responsive course grid with instructor highlights
 * - Clear CTAs for enrollment
 * 
 * @see /docs/COURSES_DESIGN.md for design documentation
 */
export default function CoursesLandingPage() {
  return <CoursesPage />;
}