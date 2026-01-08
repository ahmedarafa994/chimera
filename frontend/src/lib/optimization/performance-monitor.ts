/**
 * Performance Monitoring System for Project Chimera
 * 
 * Provides comprehensive performance monitoring including:
 * - Core Web Vitals tracking (LCP, FID, CLS, FCP, TTFB)
 * - Custom performance metrics
 * - Resource timing analysis
 * - Long task detection
 * - Memory monitoring
 * - Network performance tracking
 * 
 * @module lib/optimization/performance-monitor
 */

// ============================================================================
// Types
// ============================================================================

export interface WebVitalsMetric {
  name: 'LCP' | 'FID' | 'CLS' | 'FCP' | 'TTFB' | 'INP';
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  delta: number;
  id: string;
  navigationType: string;
  entries: PerformanceEntry[];
}

export interface PerformanceMetrics {
  // Core Web Vitals
  lcp?: number;
  fid?: number;
  cls?: number;
  fcp?: number;
  ttfb?: number;
  inp?: number;

  // Custom metrics
  timeToInteractive?: number;
  domContentLoaded?: number;
  windowLoad?: number;
  firstPaint?: number;

  // Resource metrics
  totalResources?: number;
  totalResourceSize?: number;
  totalResourceTime?: number;

  // JavaScript metrics
  jsHeapSize?: number;
  jsHeapSizeLimit?: number;
  jsHeapUsedSize?: number;

  // Network metrics
  connectionType?: string;
  effectiveType?: string;
  downlink?: number;
  rtt?: number;
}

export interface ResourceMetrics {
  name: string;
  type: string;
  duration: number;
  transferSize: number;
  decodedBodySize: number;
  startTime: number;
  responseEnd: number;
}

export interface LongTaskMetrics {
  duration: number;
  startTime: number;
  attribution: string[];
}

export interface PerformanceReport {
  timestamp: number;
  url: string;
  metrics: PerformanceMetrics;
  resources: ResourceMetrics[];
  longTasks: LongTaskMetrics[];
  marks: Record<string, number>;
  measures: Record<string, number>;
}

export type MetricCallback = (metric: WebVitalsMetric) => void;
export type ReportCallback = (report: PerformanceReport) => void;

// ============================================================================
// Web Vitals Thresholds
// ============================================================================

const WEB_VITALS_THRESHOLDS = {
  LCP: { good: 2500, poor: 4000 },
  FID: { good: 100, poor: 300 },
  CLS: { good: 0.1, poor: 0.25 },
  FCP: { good: 1800, poor: 3000 },
  TTFB: { good: 800, poor: 1800 },
  INP: { good: 200, poor: 500 },
} as const;

/**
 * Get rating for a metric value
 */
function getRating(
  name: keyof typeof WEB_VITALS_THRESHOLDS,
  value: number
): 'good' | 'needs-improvement' | 'poor' {
  const thresholds = WEB_VITALS_THRESHOLDS[name];
  if (value <= thresholds.good) return 'good';
  if (value <= thresholds.poor) return 'needs-improvement';
  return 'poor';
}

// ============================================================================
// Performance Monitor Class
// ============================================================================

export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: PerformanceMetrics = {};
  private resources: ResourceMetrics[] = [];
  private longTasks: LongTaskMetrics[] = [];
  private marks: Map<string, number> = new Map();
  private measures: Map<string, number> = new Map();
  private callbacks: Set<MetricCallback> = new Set();
  private reportCallbacks: Set<ReportCallback> = new Set();
  private observers: PerformanceObserver[] = [];
  private isInitialized = false;

  private constructor() { }

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  /**
   * Initialize performance monitoring
   */
  initialize(): void {
    if (this.isInitialized || typeof window === 'undefined') return;

    this.isInitialized = true;
    this.observeWebVitals();
    this.observeResources();
    this.observeLongTasks();
    this.collectNavigationMetrics();
    this.observeMemory();
    this.observeNetwork();
  }

  /**
   * Subscribe to metric updates
   */
  subscribe(callback: MetricCallback): () => void {
    this.callbacks.add(callback);
    return () => this.callbacks.delete(callback);
  }

  /**
   * Subscribe to performance reports
   */
  subscribeToReports(callback: ReportCallback): () => void {
    this.reportCallbacks.add(callback);
    return () => this.reportCallbacks.delete(callback);
  }

  /**
   * Get current metrics
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  /**
   * Get performance report
   */
  getReport(): PerformanceReport {
    return {
      timestamp: Date.now(),
      url: typeof window !== 'undefined' ? window.location.href : '',
      metrics: this.getMetrics(),
      resources: [...this.resources],
      longTasks: [...this.longTasks],
      marks: Object.fromEntries(this.marks),
      measures: Object.fromEntries(this.measures),
    };
  }

  /**
   * Create a performance mark
   */
  mark(name: string): void {
    if (typeof performance === 'undefined') return;

    performance.mark(name);
    this.marks.set(name, performance.now());
  }

  /**
   * Create a performance measure
   */
  measure(name: string, startMark: string, endMark?: string): number | undefined {
    if (typeof performance === 'undefined') return;

    try {
      const measure = performance.measure(name, startMark, endMark);
      this.measures.set(name, measure.duration);
      return measure.duration;
    } catch {
      return undefined;
    }
  }

  /**
   * Clear all marks and measures
   */
  clearMarks(): void {
    if (typeof performance === 'undefined') return;

    performance.clearMarks();
    performance.clearMeasures();
    this.marks.clear();
    this.measures.clear();
  }

  /**
   * Cleanup observers
   */
  destroy(): void {
    this.observers.forEach((observer) => observer.disconnect());
    this.observers = [];
    this.callbacks.clear();
    this.reportCallbacks.clear();
    this.isInitialized = false;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private notifyCallbacks(metric: WebVitalsMetric): void {
    this.callbacks.forEach((callback) => callback(metric));
  }

  private notifyReportCallbacks(): void {
    const report = this.getReport();
    this.reportCallbacks.forEach((callback) => callback(report));
  }

  private observeWebVitals(): void {
    // Largest Contentful Paint
    this.observeLCP();

    // First Input Delay
    this.observeFID();

    // Cumulative Layout Shift
    this.observeCLS();

    // First Contentful Paint
    this.observeFCP();

    // Time to First Byte
    this.observeTTFB();

    // Interaction to Next Paint
    this.observeINP();
  }

  private observeLCP(): void {
    try {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1] as PerformanceEntry & { startTime: number };

        if (lastEntry) {
          const value = lastEntry.startTime;
          this.metrics.lcp = value;

          this.notifyCallbacks({
            name: 'LCP',
            value,
            rating: getRating('LCP', value),
            delta: value,
            id: `lcp-${Date.now()}`,
            navigationType: this.getNavigationType(),
            entries: [lastEntry],
          });
        }
      });

      observer.observe({ type: 'largest-contentful-paint', buffered: true });
      this.observers.push(observer);
    } catch {
      // LCP not supported
    }
  }

  private observeFID(): void {
    try {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const firstEntry = entries[0] as PerformanceEntry & { processingStart: number; startTime: number };

        if (firstEntry) {
          const value = firstEntry.processingStart - firstEntry.startTime;
          this.metrics.fid = value;

          this.notifyCallbacks({
            name: 'FID',
            value,
            rating: getRating('FID', value),
            delta: value,
            id: `fid-${Date.now()}`,
            navigationType: this.getNavigationType(),
            entries: [firstEntry],
          });
        }
      });

      observer.observe({ type: 'first-input', buffered: true });
      this.observers.push(observer);
    } catch {
      // FID not supported
    }
  }

  private observeCLS(): void {
    let clsValue = 0;
    let clsEntries: PerformanceEntry[] = [];
    let sessionValue = 0;
    let sessionEntries: PerformanceEntry[] = [];

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as (PerformanceEntry & { hadRecentInput: boolean; value: number })[]) {
          if (!entry.hadRecentInput) {
            const firstSessionEntry = sessionEntries[0] as PerformanceEntry | undefined;
            const lastSessionEntry = sessionEntries[sessionEntries.length - 1] as PerformanceEntry | undefined;

            if (
              sessionValue &&
              firstSessionEntry &&
              lastSessionEntry &&
              entry.startTime - lastSessionEntry.startTime < 1000 &&
              entry.startTime - firstSessionEntry.startTime < 5000
            ) {
              sessionValue += entry.value;
              sessionEntries.push(entry);
            } else {
              sessionValue = entry.value;
              sessionEntries = [entry];
            }

            if (sessionValue > clsValue) {
              clsValue = sessionValue;
              clsEntries = sessionEntries;

              this.metrics.cls = clsValue;

              this.notifyCallbacks({
                name: 'CLS',
                value: clsValue,
                rating: getRating('CLS', clsValue),
                delta: entry.value,
                id: `cls-${Date.now()}`,
                navigationType: this.getNavigationType(),
                entries: clsEntries,
              });
            }
          }
        }
      });

      observer.observe({ type: 'layout-shift', buffered: true });
      this.observers.push(observer);
    } catch {
      // CLS not supported
    }
  }

  private observeFCP(): void {
    try {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const fcpEntry = entries.find((e) => e.name === 'first-contentful-paint');

        if (fcpEntry) {
          const value = fcpEntry.startTime;
          this.metrics.fcp = value;

          this.notifyCallbacks({
            name: 'FCP',
            value,
            rating: getRating('FCP', value),
            delta: value,
            id: `fcp-${Date.now()}`,
            navigationType: this.getNavigationType(),
            entries: [fcpEntry],
          });
        }
      });

      observer.observe({ type: 'paint', buffered: true });
      this.observers.push(observer);
    } catch {
      // FCP not supported
    }
  }

  private observeTTFB(): void {
    try {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries() as PerformanceNavigationTiming[];
        const navEntry = entries[0];

        if (navEntry) {
          const value = navEntry.responseStart - navEntry.requestStart;
          this.metrics.ttfb = value;

          this.notifyCallbacks({
            name: 'TTFB',
            value,
            rating: getRating('TTFB', value),
            delta: value,
            id: `ttfb-${Date.now()}`,
            navigationType: this.getNavigationType(),
            entries: [navEntry],
          });
        }
      });

      observer.observe({ type: 'navigation', buffered: true });
      this.observers.push(observer);
    } catch {
      // TTFB not supported
    }
  }

  private observeINP(): void {
    let maxINP = 0;

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as (PerformanceEntry & { duration: number })[]) {
          if (entry.duration > maxINP) {
            maxINP = entry.duration;
            this.metrics.inp = maxINP;

            this.notifyCallbacks({
              name: 'INP',
              value: maxINP,
              rating: getRating('INP', maxINP),
              delta: entry.duration,
              id: `inp-${Date.now()}`,
              navigationType: this.getNavigationType(),
              entries: [entry],
            });
          }
        }
      });

      observer.observe({ type: 'event', buffered: true } as any);
      this.observers.push(observer);
    } catch {
      // INP not supported
    }
  }

  private observeResources(): void {
    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as PerformanceResourceTiming[]) {
          this.resources.push({
            name: entry.name,
            type: entry.initiatorType,
            duration: entry.duration,
            transferSize: entry.transferSize,
            decodedBodySize: entry.decodedBodySize,
            startTime: entry.startTime,
            responseEnd: entry.responseEnd,
          });
        }

        // Update aggregate metrics
        this.metrics.totalResources = this.resources.length;
        this.metrics.totalResourceSize = this.resources.reduce(
          (sum, r) => sum + r.transferSize,
          0
        );
        this.metrics.totalResourceTime = this.resources.reduce(
          (sum, r) => sum + r.duration,
          0
        );
      });

      observer.observe({ type: 'resource', buffered: true });
      this.observers.push(observer);
    } catch {
      // Resource timing not supported
    }
  }

  private observeLongTasks(): void {
    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as (PerformanceEntry & { attribution?: { name: string }[] })[]) {
          this.longTasks.push({
            duration: entry.duration,
            startTime: entry.startTime,
            attribution: entry.attribution?.map((a) => a.name) || [],
          });
        }
      });

      observer.observe({ type: 'longtask', buffered: true });
      this.observers.push(observer);
    } catch {
      // Long task not supported
    }
  }

  private collectNavigationMetrics(): void {
    if (typeof window === 'undefined') return;

    window.addEventListener('load', () => {
      setTimeout(() => {
        const timing = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;

        if (timing) {
          this.metrics.domContentLoaded = timing.domContentLoadedEventEnd - timing.startTime;
          this.metrics.windowLoad = timing.loadEventEnd - timing.startTime;

          // First paint
          const paintEntries = performance.getEntriesByType('paint');
          const fpEntry = paintEntries.find((e) => e.name === 'first-paint');
          if (fpEntry) {
            this.metrics.firstPaint = fpEntry.startTime;
          }
        }

        this.notifyReportCallbacks();
      }, 0);
    });
  }

  private observeMemory(): void {
    if (typeof window === 'undefined') return;

    const updateMemory = () => {
      const memory = (performance as Performance & { memory?: { jsHeapSizeLimit: number; totalJSHeapSize: number; usedJSHeapSize: number } }).memory;
      if (memory) {
        this.metrics.jsHeapSizeLimit = memory.jsHeapSizeLimit;
        this.metrics.jsHeapSize = memory.totalJSHeapSize;
        this.metrics.jsHeapUsedSize = memory.usedJSHeapSize;
      }
    };

    updateMemory();
    setInterval(updateMemory, 10000); // Update every 10 seconds
  }

  private observeNetwork(): void {
    if (typeof navigator === 'undefined') return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const connection = (navigator as any).connection;
    if (connection) {
      const updateNetwork = () => {
        this.metrics.connectionType = connection.type;
        this.metrics.effectiveType = connection.effectiveType;
        this.metrics.downlink = connection.downlink;
        this.metrics.rtt = connection.rtt;
      };

      updateNetwork();
      connection.addEventListener?.('change', updateNetwork);
    }
  }

  private getNavigationType(): string {
    if (typeof window === 'undefined') return 'unknown';

    const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    return navEntry?.type || 'unknown';
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const performanceMonitor = PerformanceMonitor.getInstance();

// ============================================================================
// React Hook
// ============================================================================

import { useEffect, useState, useCallback } from 'react';

/**
 * React hook for performance monitoring
 */
export function usePerformanceMonitor(): {
  metrics: PerformanceMetrics;
  report: PerformanceReport | null;
  mark: (name: string) => void;
  measure: (name: string, startMark: string, endMark?: string) => number | undefined;
} {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({});
  const [report, setReport] = useState<PerformanceReport | null>(null);

  useEffect(() => {
    performanceMonitor.initialize();

    const unsubscribeMetrics = performanceMonitor.subscribe((metric) => {
      setMetrics((prev) => ({
        ...prev,
        [metric.name.toLowerCase()]: metric.value,
      }));
    });

    const unsubscribeReports = performanceMonitor.subscribeToReports((newReport) => {
      setReport(newReport);
    });

    // Get initial metrics
    setMetrics(performanceMonitor.getMetrics());

    return () => {
      unsubscribeMetrics();
      unsubscribeReports();
    };
  }, []);

  const mark = useCallback((name: string) => {
    performanceMonitor.mark(name);
  }, []);

  const measure = useCallback(
    (name: string, startMark: string, endMark?: string) => {
      return performanceMonitor.measure(name, startMark, endMark);
    },
    []
  );

  return { metrics, report, mark, measure };
}

// ============================================================================
// Performance Utilities
// ============================================================================

/**
 * Measure function execution time
 */
export function measureExecutionTime<T>(
  fn: () => T,
  label: string
): { result: T; duration: number } {
  const start = performance.now();
  const result = fn();
  const duration = performance.now() - start;

  if (process.env.NODE_ENV === 'development') {
    console.log(`[Performance] ${label}: ${duration.toFixed(2)}ms`);
  }

  return { result, duration };
}

/**
 * Measure async function execution time
 */
export async function measureAsyncExecutionTime<T>(
  fn: () => Promise<T>,
  label: string
): Promise<{ result: T; duration: number }> {
  const start = performance.now();
  const result = await fn();
  const duration = performance.now() - start;

  if (process.env.NODE_ENV === 'development') {
    console.log(`[Performance] ${label}: ${duration.toFixed(2)}ms`);
  }

  return { result, duration };
}

/**
 * Create a performance-tracked function
 */
export function withPerformanceTracking<T extends (...args: unknown[]) => unknown>(
  fn: T,
  label: string
): T {
  return ((...args: Parameters<T>) => {
    const start = performance.now();
    const result = fn(...args);

    if (result instanceof Promise) {
      return result.finally(() => {
        const duration = performance.now() - start;
        if (process.env.NODE_ENV === 'development') {
          console.log(`[Performance] ${label}: ${duration.toFixed(2)}ms`);
        }
      });
    }

    const duration = performance.now() - start;
    if (process.env.NODE_ENV === 'development') {
      console.log(`[Performance] ${label}: ${duration.toFixed(2)}ms`);
    }

    return result;
  }) as T;
}

/**
 * Debounce with performance tracking
 */
export function debounceWithTracking<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number,
  label: string
): T & { cancel: () => void } {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let callCount = 0;

  const debounced = ((...args: Parameters<T>) => {
    callCount++;

    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    timeoutId = setTimeout(() => {
      if (process.env.NODE_ENV === 'development') {
        console.log(`[Performance] ${label}: Debounced ${callCount} calls`);
      }
      callCount = 0;
      fn(...args);
    }, delay);
  }) as T & { cancel: () => void };

  debounced.cancel = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
  };

  return debounced;
}

/**
 * Throttle with performance tracking
 */
export function throttleWithTracking<T extends (...args: unknown[]) => unknown>(
  fn: T,
  limit: number,
  label: string
): T {
  let lastCall = 0;
  let skippedCalls = 0;

  return ((...args: Parameters<T>) => {
    const now = Date.now();

    if (now - lastCall >= limit) {
      if (skippedCalls > 0 && process.env.NODE_ENV === 'development') {
        console.log(`[Performance] ${label}: Throttled ${skippedCalls} calls`);
      }
      skippedCalls = 0;
      lastCall = now;
      return fn(...args);
    }

    skippedCalls++;
  }) as T;
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

export interface BenchmarkResult {
  name: string;
  iterations: number;
  totalTime: number;
  averageTime: number;
  minTime: number;
  maxTime: number;
  opsPerSecond: number;
}

/**
 * Run a performance benchmark
 */
export async function benchmark(
  name: string,
  fn: () => void | Promise<void>,
  iterations: number = 1000
): Promise<BenchmarkResult> {
  const times: number[] = [];

  // Warmup
  for (let i = 0; i < Math.min(10, iterations / 10); i++) {
    await fn();
  }

  // Actual benchmark
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }

  const totalTime = times.reduce((a, b) => a + b, 0);
  const averageTime = totalTime / iterations;
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const opsPerSecond = 1000 / averageTime;

  return {
    name,
    iterations,
    totalTime,
    averageTime,
    minTime,
    maxTime,
    opsPerSecond,
  };
}

/**
 * Compare multiple implementations
 */
export async function compareBenchmarks(
  benchmarks: { name: string; fn: () => void | Promise<void> }[],
  iterations: number = 1000
): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  for (const { name, fn } of benchmarks) {
    results.push(await benchmark(name, fn, iterations));
  }

  // Sort by average time
  results.sort((a, b) => a.averageTime - b.averageTime);

  return results;
}

// ============================================================================
// Performance Budget
// ============================================================================

export interface PerformanceBudget {
  lcp?: number;
  fid?: number;
  cls?: number;
  fcp?: number;
  ttfb?: number;
  totalResourceSize?: number;
  totalResources?: number;
  jsHeapSize?: number;
}

export const DEFAULT_PERFORMANCE_BUDGET: PerformanceBudget = {
  lcp: 2500,
  fid: 100,
  cls: 0.1,
  fcp: 1800,
  ttfb: 800,
  totalResourceSize: 1024 * 1024 * 2, // 2MB
  totalResources: 50,
  jsHeapSize: 1024 * 1024 * 50, // 50MB
};

/**
 * Check if metrics are within budget
 */
export function checkPerformanceBudget(
  metrics: PerformanceMetrics,
  budget: PerformanceBudget = DEFAULT_PERFORMANCE_BUDGET
): { passed: boolean; violations: string[] } {
  const violations: string[] = [];

  if (budget.lcp && metrics.lcp && metrics.lcp > budget.lcp) {
    violations.push(`LCP: ${metrics.lcp}ms exceeds budget of ${budget.lcp}ms`);
  }

  if (budget.fid && metrics.fid && metrics.fid > budget.fid) {
    violations.push(`FID: ${metrics.fid}ms exceeds budget of ${budget.fid}ms`);
  }

  if (budget.cls && metrics.cls && metrics.cls > budget.cls) {
    violations.push(`CLS: ${metrics.cls} exceeds budget of ${budget.cls}`);
  }

  if (budget.fcp && metrics.fcp && metrics.fcp > budget.fcp) {
    violations.push(`FCP: ${metrics.fcp}ms exceeds budget of ${budget.fcp}ms`);
  }

  if (budget.ttfb && metrics.ttfb && metrics.ttfb > budget.ttfb) {
    violations.push(`TTFB: ${metrics.ttfb}ms exceeds budget of ${budget.ttfb}ms`);
  }

  if (
    budget.totalResourceSize &&
    metrics.totalResourceSize &&
    metrics.totalResourceSize > budget.totalResourceSize
  ) {
    violations.push(
      `Total resource size: ${(metrics.totalResourceSize / 1024 / 1024).toFixed(2)}MB exceeds budget of ${(budget.totalResourceSize / 1024 / 1024).toFixed(2)}MB`
    );
  }

  if (
    budget.totalResources &&
    (metrics.totalResources ?? 0) > budget.totalResources
  ) {
    violations.push(
      `Total resources: ${metrics.totalResources} exceeds budget of ${budget.totalResources}`
    );
  }

  if (budget.jsHeapSize && metrics.jsHeapSize && metrics.jsHeapSize > budget.jsHeapSize) {
    violations.push(
      `JS heap size: ${(metrics.jsHeapSize / 1024 / 1024).toFixed(2)}MB exceeds budget of ${(budget.jsHeapSize / 1024 / 1024).toFixed(2)}MB`
    );
  }

  return {
    passed: violations.length === 0,
    violations,
  };
}