'use client';

import { Suspense, useMemo } from 'react';
import { PerformantSuspense, ConditionalFeature } from '@/lib/components/performance-suspense';
import { RechartsComponents, FeatureComponents } from '@/lib/components/lazy-components';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

// Enhanced loading skeletons for different component types
const DashboardSkeleton = () => (
  <div className="space-y-6">
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {[...Array(4)].map((_, i) => (
        <Card key={i}>
          <CardContent className="pt-6">
            <Skeleton className="h-8 w-16 mb-2" />
            <Skeleton className="h-4 w-24" />
          </CardContent>
        </Card>
      ))}
    </div>

    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-4 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-4 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    </div>
  </div>
);

const MetricsSkeleton = () => (
  <div className="space-y-6">
    <div className="flex items-center justify-between">
      <Skeleton className="h-8 w-48" />
      <div className="flex items-center space-x-2">
        <Skeleton className="h-6 w-16" />
        <Skeleton className="h-6 w-20" />
      </div>
    </div>

    <div className="grid gap-4 md:grid-cols-3">
      {[...Array(3)].map((_, i) => (
        <Card key={i}>
          <CardContent className="pt-6">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Skeleton className="h-5 w-8" />
                <Skeleton className="h-5 w-12" />
              </div>
              <Skeleton className="h-8 w-16" />
              <Skeleton className="h-2 w-full" />
              <div className="flex justify-between">
                <Skeleton className="h-3 w-12" />
                <Skeleton className="h-3 w-12" />
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  </div>
);

const ChartSkeleton = () => {
  const heights = useMemo(() => Array.from({ length: 12 }, () => Math.random() * 200 + 20), []);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <Skeleton className="h-6 w-32" />
            <Skeleton className="h-4 w-48" />
          </div>
          <div className="flex items-center space-x-2">
            <Skeleton className="h-6 w-16" />
            <Skeleton className="h-6 w-20" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-end space-x-2 h-64">
            {heights.map((height: number, i: number) => (
              <Skeleton
                key={i}
                className="flex-1"
                style={{ height: `${height}px` }}
              />
            ))}
          </div>
          <div className="flex items-center justify-center space-x-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="flex items-center space-x-2">
                <Skeleton className="h-3 w-3 rounded-full" />
                <Skeleton className="h-3 w-16" />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const FormSkeleton = () => (
  <Card>
    <CardHeader>
      <Skeleton className="h-6 w-32" />
      <Skeleton className="h-4 w-48" />
    </CardHeader>
    <CardContent className="space-y-4">
      {[...Array(4)].map((_, i) => (
        <div key={i} className="space-y-2">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-10 w-full" />
        </div>
      ))}
      <div className="flex space-x-2">
        <Skeleton className="h-10 w-20" />
        <Skeleton className="h-10 w-16" />
      </div>
    </CardContent>
  </Card>
);

// Progressive enhancement wrapper for charts
interface ProgressiveChartProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  enableHighPerformanceMode?: boolean;
}

export function ProgressiveChart({
  children,
  fallback = <ChartSkeleton />,
  enableHighPerformanceMode = true
}: ProgressiveChartProps) {
  return (
    <ConditionalFeature
      fallback={fallback}
      condition={enableHighPerformanceMode}
    >
      <PerformantSuspense
        fallback={fallback}
        timeout={8000} // Longer timeout for charts
      >
        {children}
      </PerformantSuspense>
    </ConditionalFeature>
  );
}

// Progressive enhancement wrapper for metrics
interface ProgressiveMetricsProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export function ProgressiveMetrics({ children, fallback = <MetricsSkeleton /> }: ProgressiveMetricsProps) {
  return (
    <PerformantSuspense fallback={fallback} timeout={5000}>
      {children}
    </PerformantSuspense>
  );
}

// Progressive enhancement wrapper for forms
interface ProgressiveFormProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export function ProgressiveForm({ children, fallback = <FormSkeleton /> }: ProgressiveFormProps) {
  return (
    <PerformantSuspense fallback={fallback} timeout={3000}>
      {children}
    </PerformantSuspense>
  );
}

// Performance-aware dashboard wrapper
interface ProgressiveDashboardProps {
  children: React.ReactNode;
  showPerformanceMetrics?: boolean;
}

export function ProgressiveDashboard({ children, showPerformanceMetrics = false }: ProgressiveDashboardProps) {
  return (
    <div className="space-y-6">
      {/* Core Web Vitals Monitor - Load first for performance insights */}
      {showPerformanceMetrics && (
        <PerformantSuspense
          fallback={<MetricsSkeleton />}
          timeout={3000}
        >
          <FeatureComponents.CoreWebVitalsMonitor />
        </PerformantSuspense>
      )}

      {/* Main dashboard content */}
      <PerformantSuspense
        fallback={<DashboardSkeleton />}
        timeout={5000}
      >
        {children}
      </PerformantSuspense>
    </div>
  );
}

// Smart loading indicator that adapts to connection speed
export function SmartLoadingIndicator() {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center space-x-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
          <div className="space-y-2">
            <Skeleton className="h-4 w-32" />
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="animate-pulse">
                Loading...
              </Badge>
              <Skeleton className="h-3 w-20" />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Adaptive loading based on device capabilities
export function AdaptiveLoader({ children, heavyComponent }: { children: React.ReactNode; heavyComponent: React.ReactNode }) {
  return (
    <ConditionalFeature
      fallback={children}
      condition={true} // Will check hardware capabilities internally
    >
      <PerformantSuspense fallback={<SmartLoadingIndicator />}>
        {heavyComponent}
      </PerformantSuspense>
    </ConditionalFeature>
  );
}

export {
  DashboardSkeleton,
  MetricsSkeleton,
  ChartSkeleton,
  FormSkeleton,
};