import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

const AutoDANReasoningDashboard = dynamic(
  () => import("@/components/autodan/AutoDANReasoningDashboard").then((mod) => mod.AutoDANReasoningDashboard),
  {
    loading: () => <DashboardSkeleton />,
  }
);

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Skeleton className="h-24 rounded-xl" />
        <Skeleton className="h-24 rounded-xl" />
        <Skeleton className="h-24 rounded-xl" />
        <Skeleton className="h-24 rounded-xl" />
      </div>
      <Skeleton className="h-[400px] rounded-xl" />
    </div>
  )
}

export default function AutoDANPage() {
  return (
    <div className="space-y-6">
      <AutoDANReasoningDashboard />
    </div>
  );
}
