/**
 * Aegis Campaign Dashboard Page
 *
 * Real-time dashboard for monitoring Aegis campaign telemetry including:
 * - Attack success rates and trends
 * - Technique performance breakdown
 * - Token usage and cost tracking
 * - API latency metrics
 * - Live event feed
 * - Prompt evolution timeline
 *
 * @module app/dashboard/aegis-campaign
 */

import type { Metadata } from "next";
import dynamicImport from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";
import { GlassCard } from "@/components/ui/glass-card";

// ============================================================================
// Metadata for SEO
// ============================================================================

export const metadata: Metadata = {
  title: "Aegis Campaign Dashboard | Chimera",
  description:
    "Real-time monitoring dashboard for Aegis security research campaigns. Track attack success rates, transformation techniques, token costs, and prompt evolution in real-time via WebSocket streaming.",
  keywords: [
    "aegis",
    "campaign",
    "dashboard",
    "real-time",
    "telemetry",
    "jailbreak",
    "attack",
    "metrics",
    "security research",
    "prompt optimization",
  ],
  robots: "noindex, nofollow", // Security research tool - don't index
};

// Force dynamic rendering for WebSocket support
export const dynamic = "force-dynamic";

// ============================================================================
// Dynamic Import with Loading State
// ============================================================================

/**
 * Loading skeleton matching the dashboard layout
 */
function DashboardLoadingSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Header skeleton */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-9 w-64" />
          <Skeleton className="h-4 w-48" />
        </div>
        <Skeleton className="h-8 w-32" />
      </div>

      {/* Campaign selector skeleton */}
      <GlassCard variant="default" intensity="medium" className="p-4">
        <div className="flex gap-3">
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-9 w-full" />
          </div>
          <Skeleton className="h-9 w-32 self-end" />
        </div>
      </GlassCard>

      {/* Metrics grid skeleton */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <GlassCard key={i} variant="default" intensity="medium" className="p-4">
            <Skeleton className="h-4 w-24 mb-3" />
            <Skeleton className="h-12 w-20 mb-4" />
            <Skeleton className="h-8 w-full" />
          </GlassCard>
        ))}
      </div>

      {/* Charts skeleton */}
      <div className="grid gap-4 lg:grid-cols-2">
        <GlassCard variant="default" intensity="medium" className="p-4">
          <Skeleton className="h-4 w-32 mb-4" />
          <Skeleton className="h-48 w-full" />
        </GlassCard>
        <GlassCard variant="default" intensity="medium" className="p-4">
          <Skeleton className="h-4 w-32 mb-4" />
          <Skeleton className="h-48 w-full" />
        </GlassCard>
      </div>

      {/* Event feed and timeline skeleton */}
      <div className="grid gap-4 lg:grid-cols-2">
        <GlassCard variant="default" intensity="medium" className="p-4">
          <Skeleton className="h-4 w-28 mb-4" />
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        </GlassCard>
        <GlassCard variant="default" intensity="medium" className="p-4">
          <Skeleton className="h-4 w-36 mb-4" />
          <div className="space-y-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="flex items-start gap-3">
                <Skeleton className="h-8 w-8 rounded-full flex-shrink-0" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}

/**
 * Dynamically imported dashboard component with loading state
 * This allows for code splitting and reduces initial bundle size
 */
const AegisCampaignDashboard = dynamicImport(
  () =>
    import("@/components/aegis-dashboard/AegisCampaignDashboard").then(
      (mod) => mod.AegisCampaignDashboard
    ),
  {
    loading: () => <DashboardLoadingSkeleton />,
    ssr: false, // Disable SSR for WebSocket-based component
  }
);

// ============================================================================
// Page Component
// ============================================================================

/**
 * Aegis Campaign Dashboard Page
 *
 * Renders the real-time Aegis campaign monitoring dashboard.
 * Uses dynamic import for code splitting and SSR disabled for WebSocket support.
 *
 * @example
 * // Navigate to /dashboard/aegis-campaign to access this page
 */
export default function AegisCampaignPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex-1 space-y-4 p-8 pt-6">
        <AegisCampaignDashboard />
      </div>
    </div>
  );
}
