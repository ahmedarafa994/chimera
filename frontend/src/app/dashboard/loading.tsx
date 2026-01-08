import { Loader2 } from 'lucide-react';

export default function DashboardLoading() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <Loader2 
          className="h-8 w-8 animate-spin text-primary" 
          aria-hidden="true"
        />
        <p className="text-sm text-muted-foreground">
          Loading dashboard...
        </p>
      </div>
    </div>
  );
}