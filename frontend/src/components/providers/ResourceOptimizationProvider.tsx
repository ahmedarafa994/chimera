'use client';

import { useEffect, useCallback } from 'react';

interface ResourceOptimizationProviderProps {
  children: React.ReactNode;
}

export default function ResourceOptimizationProvider({ children }: ResourceOptimizationProviderProps) {
  // Performance monitoring - disabled in development to reduce console noise
  useEffect(() => {
    if (typeof window === 'undefined' || !('PerformanceObserver' in window)) {
      return;
    }

    // Skip monitoring in development unless explicitly enabled
    const enableMonitoring = process.env.NODE_ENV === 'production' ||
                            process.env.NEXT_PUBLIC_ENABLE_PERF_MONITORING === 'true';
    
    if (!enableMonitoring) {
      return;
    }

    let longTaskObserver: PerformanceObserver | null = null;
    let layoutShiftObserver: PerformanceObserver | null = null;

    try {
      // Register performance observer for long tasks
      longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          // Log significant long tasks (> 100ms)
          if (entry.duration > 100) {
            console.warn(`[Perf] Long task: ${Math.round(entry.duration)}ms`);
          }
        }
      });

      longTaskObserver.observe({ entryTypes: ['longtask'] });

      // Monitor layout shifts
      layoutShiftObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if ((entry as PerformanceEntry & { hadRecentInput?: boolean }).hadRecentInput) continue;

          const cls = (entry as PerformanceEntry & { value?: number }).value || 0;
          // Log significant CLS (> 0.1)
          if (cls > 0.1) {
            console.warn(`[Perf] Layout shift: ${cls.toFixed(3)}`);
          }
        }
      });

      layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
    } catch {
      // PerformanceObserver may not be fully supported
    }

    return () => {
      longTaskObserver?.disconnect();
      layoutShiftObserver?.disconnect();
    };
  }, []);

  // Preload critical routes on idle - debounced
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const preloadRoutes = () => {
      if ('requestIdleCallback' in window) {
        window.requestIdleCallback(
          () => {
            // Preload critical route chunks silently
            import('@/app/dashboard/page').catch(() => {});
          },
          { timeout: 2000 }
        );
      }
    };

    // Delay preloading to not interfere with initial render
    const timer = setTimeout(preloadRoutes, 1000);
    return () => clearTimeout(timer);
  }, []);

  return <>{children}</>;
}

// Utility hook for route-specific optimizations
export function useRouteOptimization(route: string) {
  const preloadForRoute = useCallback((routePath: string) => {
    if (typeof window === 'undefined' || !('requestIdleCallback' in window)) return;

    const optimizations: Record<string, () => void> = {
      '/dashboard': () => {
        window.requestIdleCallback(
          () => {
            import('recharts').catch(() => {});
          },
          { timeout: 3000 }
        );
      },
      '/dashboard/generation': () => {
        window.requestIdleCallback(
          () => {
            import('@/components/generation-panel').catch(() => {});
          },
          { timeout: 3000 }
        );
      },
      '/dashboard/metrics': () => {
        window.requestIdleCallback(
          () => {
            import('recharts').catch(() => {});
          },
          { timeout: 3000 }
        );
      },
    };

    optimizations[routePath]?.();
  }, []);

  useEffect(() => {
    preloadForRoute(route);
  }, [route, preloadForRoute]);
}