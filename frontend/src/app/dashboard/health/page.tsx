import { HealthDashboard } from "@/components/health/HealthDashboard";

export default function HealthPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">System Health</h1>
        <p className="text-muted-foreground">
          Monitor the health status of all Chimera platform services and dependencies.
        </p>
      </div>
      <HealthDashboard autoRefresh={true} refreshInterval={30000} />
    </div>
  );
}