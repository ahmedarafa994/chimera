/**
 * Unified Attack Page
 * 
 * Next.js page for multi-vector attack dashboard.
 */

import { Suspense } from 'react';
import { Loader2 } from 'lucide-react';
import { AttackDashboard } from '@/components/unified-attack';

export const metadata = {
  title: 'Multi-Vector Attack Dashboard | Chimera',
  description: 'Unified obfuscation and mutation attack framework',
};

export const dynamic = 'force-dynamic';
export const fetchCache = 'force-no-store';

function DashboardLoading() {
  return (
    <div className="flex items-center justify-center h-screen bg-slate-950">
      <div className="flex flex-col items-center gap-4">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <p className="text-sm text-slate-400">Loading Attack Dashboard...</p>
      </div>
    </div>
  );
}

export default function UnifiedAttackPage() {
  return (
    <Suspense fallback={<DashboardLoading />}>
      <AttackDashboard className="h-screen" />
    </Suspense>
  );
}