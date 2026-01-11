"use client";

import SystemArchitectureDashboard from "@/components/SystemArchitectureDashboard";
import { AdminOnly } from "@/components/auth/RoleGuard";

export default function SystemPage() {
  return (
    <AdminOnly>
      <SystemArchitectureDashboard />
    </AdminOnly>
  );
}
