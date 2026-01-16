/**
 * Aegis Telemetry Performance Monitor
 *
 * Performance tracking utilities for the Real-Time Aegis Campaign Dashboard.
 * Ensures event processing meets the <500ms requirement from the spec.
 *
 * Features:
 * - Event processing time tracking
 * - Performance metrics aggregation
 * - Threshold violation detection
 * - Performance reporting
 */

"use client";

// ============================================================================
// Constants
// ============================================================================

/**
 * Maximum acceptable event processing time in milliseconds
 * Per spec requirement: "Updates within 500ms of backend telemetry event emission"
 */
export const MAX_EVENT_PROCESSING_MS = 500;

/**
 * Warning threshold for event processing time
 */
export const WARNING_THRESHOLD_MS = 300;

/**
 * Maximum number of samples to track for rolling averages
 */
export const MAX_SAMPLES = 100;

// ============================================================================
// Types
// ============================================================================

export interface PerformanceSample {
  /** Timestamp when processing started */
  startTime: number;
  /** Timestamp when processing completed */
  endTime: number;
  /** Duration in milliseconds */
  duration: number;
  /** Event type that was processed */
  eventType: string;
  /** Whether the sample exceeded the threshold */
  exceedsThreshold: boolean;
}

export interface PerformanceMetrics {
  /** Total number of events processed */
  totalEvents: number;
  /** Average processing time in milliseconds */
  averageProcessingTime: number;
  /** Minimum processing time in milliseconds */
  minProcessingTime: number;
  /** Maximum processing time in milliseconds */
  maxProcessingTime: number;
  /** P50 (median) processing time */
  p50ProcessingTime: number;
  /** P95 processing time */
  p95ProcessingTime: number;
  /** P99 processing time */
  p99ProcessingTime: number;
  /** Number of events that exceeded the threshold */
  thresholdViolations: number;
  /** Threshold violation rate as percentage */
  violationRate: number;
  /** Events processed per second */
  eventsPerSecond: number;
  /** Last update timestamp */
  lastUpdated: number;
}

export interface PerformanceMonitorConfig {
  /** Enable performance monitoring */
  enabled: boolean;
  /** Log warnings when thresholds are exceeded */
  logWarnings: boolean;
  /** Custom threshold in milliseconds */
  thresholdMs?: number;
}

// ============================================================================
// Performance Monitor Class
// ============================================================================

/**
 * Performance monitor for tracking Aegis telemetry event processing times.
 *
 * @example
 * ```tsx
 * const monitor = new AegisPerformanceMonitor({ enabled: true, logWarnings: true });
 *
 * // Track event processing
 * const endMeasure = monitor.startMeasure("attack_completed");
 * await processEvent(event);
 * endMeasure();
 *
 * // Get metrics
 * const metrics = monitor.getMetrics();
 * console.log(`Average processing time: ${metrics.averageProcessingTime}ms`);
 * ```
 */
export class AegisPerformanceMonitor {
  private samples: PerformanceSample[] = [];
  private config: PerformanceMonitorConfig;
  private startTimeRef: number | null = null;
  private eventCountWindow: { timestamp: number; count: number }[] = [];

  constructor(config: Partial<PerformanceMonitorConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      logWarnings: config.logWarnings ?? false,
      thresholdMs: config.thresholdMs ?? MAX_EVENT_PROCESSING_MS,
    };
  }

  /**
   * Start measuring event processing time
   * @returns Function to call when processing is complete
   */
  startMeasure(eventType: string): () => void {
    if (!this.config.enabled) {
      return () => {};
    }

    const startTime = performance.now();

    return () => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      const exceedsThreshold = duration > (this.config.thresholdMs ?? MAX_EVENT_PROCESSING_MS);

      const sample: PerformanceSample = {
        startTime,
        endTime,
        duration,
        eventType,
        exceedsThreshold,
      };

      this.addSample(sample);

      if (exceedsThreshold && this.config.logWarnings) {
        // Using performance warning format
        if (typeof window !== "undefined" && "console" in window) {
          // eslint-disable-next-line no-console
          console.warn(
            `[AegisPerformance] Event processing exceeded threshold: ${eventType} took ${duration.toFixed(2)}ms (threshold: ${this.config.thresholdMs}ms)`
          );
        }
      }
    };
  }

  /**
   * Record a pre-calculated processing time
   */
  recordProcessingTime(eventType: string, durationMs: number): void {
    if (!this.config.enabled) return;

    const now = performance.now();
    const sample: PerformanceSample = {
      startTime: now - durationMs,
      endTime: now,
      duration: durationMs,
      eventType,
      exceedsThreshold: durationMs > (this.config.thresholdMs ?? MAX_EVENT_PROCESSING_MS),
    };

    this.addSample(sample);
  }

  /**
   * Add a sample to the rolling window
   */
  private addSample(sample: PerformanceSample): void {
    this.samples.push(sample);

    // Maintain rolling window
    if (this.samples.length > MAX_SAMPLES) {
      this.samples.shift();
    }

    // Track event count for events/second calculation
    const now = Date.now();
    this.eventCountWindow.push({ timestamp: now, count: 1 });

    // Remove events older than 1 second
    const oneSecondAgo = now - 1000;
    this.eventCountWindow = this.eventCountWindow.filter(
      (e) => e.timestamp > oneSecondAgo
    );
  }

  /**
   * Calculate percentile value from sorted array
   */
  private calculatePercentile(sortedValues: number[], percentile: number): number {
    if (sortedValues.length === 0) return 0;

    const index = Math.ceil((percentile / 100) * sortedValues.length) - 1;
    return sortedValues[Math.max(0, Math.min(index, sortedValues.length - 1))];
  }

  /**
   * Get aggregated performance metrics
   */
  getMetrics(): PerformanceMetrics {
    if (this.samples.length === 0) {
      return {
        totalEvents: 0,
        averageProcessingTime: 0,
        minProcessingTime: 0,
        maxProcessingTime: 0,
        p50ProcessingTime: 0,
        p95ProcessingTime: 0,
        p99ProcessingTime: 0,
        thresholdViolations: 0,
        violationRate: 0,
        eventsPerSecond: 0,
        lastUpdated: Date.now(),
      };
    }

    const durations = this.samples.map((s) => s.duration);
    const sortedDurations = [...durations].sort((a, b) => a - b);
    const sum = durations.reduce((acc, d) => acc + d, 0);
    const violations = this.samples.filter((s) => s.exceedsThreshold).length;

    return {
      totalEvents: this.samples.length,
      averageProcessingTime: sum / this.samples.length,
      minProcessingTime: sortedDurations[0],
      maxProcessingTime: sortedDurations[sortedDurations.length - 1],
      p50ProcessingTime: this.calculatePercentile(sortedDurations, 50),
      p95ProcessingTime: this.calculatePercentile(sortedDurations, 95),
      p99ProcessingTime: this.calculatePercentile(sortedDurations, 99),
      thresholdViolations: violations,
      violationRate: (violations / this.samples.length) * 100,
      eventsPerSecond: this.eventCountWindow.length,
      lastUpdated: Date.now(),
    };
  }

  /**
   * Check if performance meets requirements
   */
  meetsRequirements(): boolean {
    const metrics = this.getMetrics();
    return metrics.p95ProcessingTime < (this.config.thresholdMs ?? MAX_EVENT_PROCESSING_MS);
  }

  /**
   * Reset all samples
   */
  reset(): void {
    this.samples = [];
    this.eventCountWindow = [];
  }

  /**
   * Get recent samples for debugging
   */
  getRecentSamples(count: number = 10): PerformanceSample[] {
    return this.samples.slice(-count);
  }

  /**
   * Enable or disable monitoring
   */
  setEnabled(enabled: boolean): void {
    this.config.enabled = enabled;
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

/**
 * Global performance monitor instance for Aegis telemetry
 */
export const aegisPerformanceMonitor = new AegisPerformanceMonitor({
  enabled: typeof window !== "undefined",
  logWarnings: process.env.NODE_ENV === "development",
});

// ============================================================================
// React Hook
// ============================================================================

import { useRef, useCallback, useMemo } from "react";

/**
 * React hook for performance monitoring in components
 */
export function usePerformanceMonitor(componentName: string) {
  const monitorRef = useRef(aegisPerformanceMonitor);

  const startMeasure = useCallback(
    (eventType: string) => {
      return monitorRef.current.startMeasure(`${componentName}:${eventType}`);
    },
    [componentName]
  );

  const recordProcessingTime = useCallback(
    (eventType: string, durationMs: number) => {
      monitorRef.current.recordProcessingTime(
        `${componentName}:${eventType}`,
        durationMs
      );
    },
    [componentName]
  );

  const getMetrics = useCallback(() => {
    return monitorRef.current.getMetrics();
  }, []);

  return useMemo(
    () => ({
      startMeasure,
      recordProcessingTime,
      getMetrics,
    }),
    [startMeasure, recordProcessingTime, getMetrics]
  );
}

// ============================================================================
// Exports
// ============================================================================

export default aegisPerformanceMonitor;
