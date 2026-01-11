/**
 * Campaign Analytics Dashboard Page
 *
 * Next.js page at /dashboard/analytics that wraps the CampaignAnalyticsDashboard
 * component with proper metadata for SEO and Suspense for loading states.
 *
 * Features:
 * - Post-campaign telemetry analysis
 * - Statistical summaries (mean, median, p95 success rates)
 * - Side-by-side campaign comparison (up to 4 campaigns)
 * - Exportable charts (PNG/SVG) for research publications
 * - Time-series visualization of prompt evolution
 * - Filter and drill-down by technique, provider, or time range
 * - CSV export for external analysis
 *
 * @module app/dashboard/analytics/page
 */

import { Suspense } from "react";
import { Metadata } from "next";
import { Loader2, BarChart3 } from "lucide-react";
import { CampaignAnalyticsDashboard } from "@/components/campaign-analytics";

/**
 * Page metadata for SEO and social sharing
 */
export const metadata: Metadata = {
  title: "Campaign Analytics | Chimera",
  description:
    "Post-campaign analytics view with detailed telemetry breakdowns, statistical analysis, comparative metrics, and exportable charts for research papers and security reports.",
  keywords: [
    "campaign analytics",
    "telemetry",
    "jailbreak analysis",
    "prompt research",
    "AI security",
    "attack metrics",
    "success rates",
    "campaign comparison",
  ],
  openGraph: {
    title: "Campaign Analytics | Chimera",
    description:
      "Analyze campaign telemetry with statistical summaries, side-by-side comparisons, and exportable visualizations for research.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Campaign Analytics | Chimera",
    description:
      "Analyze campaign telemetry with statistical summaries, side-by-side comparisons, and exportable visualizations.",
  },
};

/**
 * Dynamic rendering configuration
 * Force dynamic rendering to ensure fresh data on each request
 */
export const dynamic = "force-dynamic";
export const fetchCache = "force-no-store";

/**
 * Loading component displayed while the dashboard is loading
 * Provides visual feedback during Suspense fallback
 */
function AnalyticsLoading() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="relative">
          <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />
          <div className="relative flex items-center justify-center rounded-full bg-primary/10 p-4">
            <BarChart3
              className="h-8 w-8 text-primary animate-pulse"
              aria-hidden="true"
            />
          </div>
        </div>
        <div className="flex flex-col items-center gap-1">
          <div className="flex items-center gap-2">
            <Loader2
              className="h-4 w-4 animate-spin text-muted-foreground"
              aria-hidden="true"
            />
            <p className="text-sm font-medium text-foreground">
              Loading Campaign Analytics
            </p>
          </div>
          <p className="text-xs text-muted-foreground">
            Preparing telemetry data and visualizations...
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * Campaign Analytics Dashboard Page
 *
 * Main page component that renders the CampaignAnalyticsDashboard
 * wrapped in Suspense for loading states. Error handling is provided
 * by the parent dashboard error boundary at /dashboard/error.tsx.
 *
 * @returns The campaign analytics dashboard page
 */
export default function CampaignAnalyticsPage() {
  return (
    <div className="flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-primary" aria-hidden="true" />
          <h1 className="text-2xl font-bold tracking-tight">
            Campaign Analytics
          </h1>
        </div>
        <p className="text-muted-foreground">
          Analyze campaign telemetry with statistical summaries, comparative
          metrics, and exportable visualizations for research.
        </p>
      </div>

      {/* Main Dashboard with Suspense Boundary */}
      <Suspense fallback={<AnalyticsLoading />}>
        <CampaignAnalyticsDashboard className="min-h-[calc(100vh-200px)]" />
      </Suspense>
    </div>
  );
}
