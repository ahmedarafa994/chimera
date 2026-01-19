/**
 * Aegis Campaign Dashboard Page
 *
 * Client Component for real-time WebSocket-based campaign monitoring
 */

"use client";

import { AegisCampaignDashboard } from "@/components/aegis-dashboard/AegisCampaignDashboard";

export default function AegisCampaignPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex-1 space-y-4 p-8 pt-6">
        <AegisCampaignDashboard />
      </div>
    </div>
  );
}