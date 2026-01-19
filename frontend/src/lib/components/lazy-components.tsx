import dynamic from 'next/dynamic';
import { Suspense, ComponentType } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type {
  XAxisProps,
  YAxisProps,
  CartesianGridProps,
  TooltipProps,
  LegendProps,
  LineProps,
  AreaProps,
  BarProps,
  CellProps,
  PieProps,
  ResponsiveContainerProps,
  RadarProps as RadarChartProps,
  RadarProps,
  PolarGridProps,
  PolarAngleAxisProps,
  PolarRadiusAxisProps,
  ScatterProps,
  ReferenceLineProps,
  ReferenceAreaProps,
  ZAxisProps,
} from 'recharts';

// Create loading components for better UX
const ChartLoading = () => (
  <Card>
    <CardContent className="pt-6">
      <div className="space-y-4">
        <Skeleton className="h-4 w-1/3" />
        <Skeleton className="h-40 w-full" />
        <div className="space-y-2">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-2/3" />
        </div>
      </div>
    </CardContent>
  </Card>
);

const PerformanceMonitorLoading = () => (
  <div className="space-y-6">
    <Card>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <Skeleton className="h-6 w-1/4" />
          <div className="grid gap-4 md:grid-cols-3">
            {[...Array(3)].map((_, i) => (
              <Card key={i}>
                <CardContent className="pt-4">
                  <Skeleton className="h-4 w-12 mb-2" />
                  <Skeleton className="h-8 w-20 mb-3" />
                  <Skeleton className="h-2 w-full mb-2" />
                  <div className="flex justify-between">
                    <Skeleton className="h-3 w-16" />
                    <Skeleton className="h-3 w-16" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  </div>
);

// Dynamic imports with loading states
export const RechartsComponents = {
  // Lazy load recharts components
  LineChart: dynamic(() => import('recharts').then(mod => ({ default: mod.LineChart })), {
    loading: () => <ChartLoading />,
    ssr: false, // Disable SSR for better performance
  }),

  AreaChart: dynamic(() => import('recharts').then(mod => ({ default: mod.AreaChart })), {
    loading: () => <ChartLoading />,
    ssr: false,
  }),

  BarChart: dynamic(() => import('recharts').then(mod => ({ default: mod.BarChart })), {
    loading: () => <ChartLoading />,
    ssr: false,
  }),

  PieChart: dynamic(() => import('recharts').then(mod => ({ default: mod.PieChart })), {
    loading: () => <ChartLoading />,
    ssr: false,
  }),

  // Common recharts exports with proper types
  XAxis: dynamic(() => import('recharts').then(mod => ({ default: mod.XAxis as ComponentType<XAxisProps> })), { ssr: false }),
  YAxis: dynamic(() => import('recharts').then(mod => ({ default: mod.YAxis as ComponentType<YAxisProps> })), { ssr: false }),
  CartesianGrid: dynamic(() => import('recharts').then(mod => ({ default: mod.CartesianGrid as ComponentType<CartesianGridProps> })), { ssr: false }),
  Tooltip: dynamic(() => import('recharts').then(mod => ({ default: mod.Tooltip as ComponentType<TooltipProps<number, string>> })), { ssr: false }),
  Legend: dynamic(() => import('recharts').then(mod => ({ default: mod.Legend as ComponentType<LegendProps> })), { ssr: false }),
  Line: dynamic(() => import('recharts').then(mod => ({ default: mod.Line as ComponentType<LineProps> })), { ssr: false }),
  Area: dynamic(() => import('recharts').then(mod => ({ default: mod.Area as ComponentType<AreaProps> })), { ssr: false }),
  Bar: dynamic(() => import('recharts').then(mod => ({ default: mod.Bar as ComponentType<BarProps> })), { ssr: false }),
  Cell: dynamic(() => import('recharts').then(mod => ({ default: mod.Cell as ComponentType<CellProps> })), { ssr: false }),
  Pie: dynamic(() => import('recharts').then(mod => ({ default: mod.Pie as ComponentType<PieProps> })), { ssr: false }),
  ResponsiveContainer: dynamic(() => import('recharts').then(mod => ({ default: mod.ResponsiveContainer as ComponentType<ResponsiveContainerProps> })), { ssr: false }),

  // Radar chart components
  RadarChart: dynamic(() => import('recharts').then(mod => ({ default: mod.RadarChart })), {
    loading: () => <ChartLoading />,
    ssr: false,
  }),
  Radar: dynamic(() => import('recharts').then(mod => ({ default: mod.Radar as ComponentType<RadarProps> })), { ssr: false }),
  PolarGrid: dynamic(() => import('recharts').then(mod => ({ default: mod.PolarGrid as ComponentType<PolarGridProps> })), { ssr: false }),
  PolarAngleAxis: dynamic(() => import('recharts').then(mod => ({ default: mod.PolarAngleAxis as ComponentType<PolarAngleAxisProps> })), { ssr: false }),
  PolarRadiusAxis: dynamic(() => import('recharts').then(mod => ({ default: mod.PolarRadiusAxis as ComponentType<PolarRadiusAxisProps> })), { ssr: false }),

  // Scatter chart components
  ScatterChart: dynamic(() => import('recharts').then(mod => ({ default: mod.ScatterChart })), {
    loading: () => <ChartLoading />,
    ssr: false,
  }),
  Scatter: dynamic(() => import('recharts').then(mod => ({ default: mod.Scatter as ComponentType<ScatterProps> })), { ssr: false }),
  ZAxis: dynamic(() => import('recharts').then(mod => ({ default: mod.ZAxis as ComponentType<ZAxisProps> })), { ssr: false }),

  // Reference components (for trend lines and annotations)
  ReferenceLine: dynamic(() => import('recharts').then(mod => ({ default: mod.ReferenceLine as ComponentType<ReferenceLineProps> })), { ssr: false }),
  ReferenceArea: dynamic(() => import('recharts').then(mod => ({ default: mod.ReferenceArea as ComponentType<ReferenceAreaProps> })), { ssr: false }),

  // ComposedChart for combining scatter + line (trend)
  ComposedChart: dynamic(() => import('recharts').then(mod => ({ default: mod.ComposedChart })), {
    loading: () => <ChartLoading />,
    ssr: false,
  }),
};

// Performance monitoring components
export const PerformanceComponents = {
  CoreWebVitalsMonitor: dynamic(() => import('@/components/performance/CoreWebVitalsMonitor'), {
    loading: () => <PerformanceMonitorLoading />,
    ssr: false,
  }),

  // Commented out until component is created
  // PerformanceMetricsChart: dynamic(() => import('@/components/performance/PerformanceMetricsChart'), {
  //   loading: () => <ChartLoading />,
  //   ssr: false,
  // }),
};

// Heavy UI components that can be lazy loaded
export const HeavyUIComponents = {
  // Radix UI Dialog (often heavy due to portal)
  Dialog: dynamic(() => import('@radix-ui/react-dialog').then(mod => ({
    default: mod.Root,
  })), {
    ssr: true, // Keep SSR for accessibility
  }),

  // Date picker (if we add one)
  // DatePicker: dynamic(() => import('@/components/ui/date-picker'), {
  //   loading: () => <Skeleton className="h-10 w-full" />,
  //   ssr: false,
  // }),

  // Code editor (if we add one)
  // CodeEditor: dynamic(() => import('@/components/ui/code-editor'), {
  //   loading: () => (
  //     <Card>
  //       <CardContent className="pt-6">
  //         <Skeleton className="h-64 w-full" />
  //       </CardContent>
  //     </Card>
  //   ),
  //   ssr: false,
  // }),

  // Advanced form components
  // AdvancedFormBuilder: dynamic(() => import('@/components/forms/AdvancedFormBuilder'), {
  //   loading: () => <Skeleton className="h-96 w-full" />,
  //   ssr: false,
  // }),
};

// Feature-specific dynamic imports
export const FeatureComponents = {
  // AutoDAN components (commented out - component doesn't exist)
  // AutoDanInterface: dynamic(() => import('@/components/autoadv/AutoDanInterface'), {
  //   loading: () => <Skeleton className="h-96 w-full" />,
  //   ssr: false,
  // }),

  // GPTFuzz components
  GPTFuzzInterface: dynamic(() => import('@/components/gptfuzz/GPTFuzzInterface').then(mod => ({ default: mod.GPTFuzzInterface })), {
    loading: () => <Skeleton className="h-96 w-full" />,
    ssr: false,
  }),

  // Performance monitoring
  CoreWebVitalsMonitor: dynamic(() => import('@/components/performance/CoreWebVitalsMonitor'), {
    loading: () => <Skeleton className="h-32 w-full" />,
    ssr: false,
  }),
};

// Utility function to create suspense wrapper
export function createSuspenseWrapper<T extends Record<string, any>>(
  Component: React.ComponentType<T>,
  fallback: React.ReactNode = <Skeleton className="h-32 w-full" />
) {
  return function SuspenseWrapper(props: T) {
    return (
      <Suspense fallback={fallback}>
        <Component {...(props as T)} />
      </Suspense>
    );
  };
}

// Preload functions for critical routes
export const preloadComponents = {
  dashboard: () => {
    // Preload dashboard components
    (PerformanceComponents.CoreWebVitalsMonitor as any).preload?.();
    (RechartsComponents.LineChart as any).preload?.();
  },

  generation: () => {
    // Preload generation components
    // HeavyUIComponents.AdvancedFormBuilder.preload?.();
  },

  metrics: () => {
    // Preload metrics components
    (RechartsComponents.AreaChart as any).preload?.();
    (RechartsComponents.BarChart as any).preload?.();
  },

  performance: () => {
    // Preload performance monitoring
    (PerformanceComponents.CoreWebVitalsMonitor as any).preload?.();
    // PerformanceComponents.PerformanceMetricsChart.preload?.();
  },
};

// Hook to preload components based on route
export function useComponentPreloading() {
  const preload = (route: keyof typeof preloadComponents) => {
    // Use requestIdleCallback if available, otherwise setTimeout
    if ('requestIdleCallback' in window) {
      window.requestIdleCallback(() => {
        preloadComponents[route]();
      });
    } else {
      setTimeout(() => {
        preloadComponents[route]();
      }, 100);
    }
  };

  return { preload };
}
