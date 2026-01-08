/**
 * Core Web Vitals Monitoring Implementation
 *
 * This module provides comprehensive monitoring for Core Web Vitals:
 * - Largest Contentful Paint (LCP) - Target: < 2.5s
 * - First Input Delay (FID) - Target: < 100ms
 * - Cumulative Layout Shift (CLS) - Target: < 0.1
 * - First Contentful Paint (FCP) - Target: < 1.8s
 * - Time to First Byte (TTFB) - Target: < 800ms
 * - Interaction to Next Paint (INP) - Target: < 200ms
 */

import { onCLS, onFCP, onFID, onLCP, onTTFB, onINP } from 'web-vitals';

// Performance thresholds aligned with Google recommendations
export const PERFORMANCE_THRESHOLDS = {
  LCP: { good: 2500, poor: 4000 },
  FID: { good: 100, poor: 300 },
  CLS: { good: 0.1, poor: 0.25 },
  FCP: { good: 1800, poor: 3000 },
  TTFB: { good: 800, poor: 1800 },
  INP: { good: 200, poor: 500 },
} as const;

export type MetricName = keyof typeof PERFORMANCE_THRESHOLDS;
export type MetricRating = 'good' | 'needs-improvement' | 'poor';

interface VitalMetric {
  name: MetricName;
  value: number;
  rating: MetricRating;
  delta: number;
  id: string;
  navigationType: string;
  timestamp: number;
}

interface PerformanceReport {
  url: string;
  timestamp: number;
  metrics: VitalMetric[];
  deviceInfo: {
    userAgent: string;
    connection?: {
      effectiveType: string;
      downlink: number;
    };
    memory?: number;
    cores?: number;
  };
}

// Performance analytics endpoint
const ANALYTICS_ENDPOINT = '/api/analytics/performance';

class CoreWebVitalsMonitor {
  private metrics: Map<MetricName, VitalMetric> = new Map();
  private reportQueue: PerformanceReport[] = [];
  private isReporting = false;

  constructor() {
    this.initializeMonitoring();
    this.setupReporting();
  }

  private initializeMonitoring(): void {
    // Only run in browser environment
    if (typeof window === 'undefined') return;

    // Monitor Core Web Vitals
    onCLS(this.handleMetric.bind(this));
    onFCP(this.handleMetric.bind(this));
    onFID(this.handleMetric.bind(this));
    onLCP(this.handleMetric.bind(this));
    onTTFB(this.handleMetric.bind(this));
    onINP(this.handleMetric.bind(this));

    // Monitor custom performance metrics
    this.monitorCustomMetrics();

    // Setup page visibility change handler
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.sendReport();
      }
    });

    // Setup beforeunload handler
    window.addEventListener('beforeunload', () => {
      this.sendReport();
    });
  }

  private handleMetric(metric: any): void {
    const rating = this.calculateRating(metric.name as MetricName, metric.value);

    const vitalMetric: VitalMetric = {
      name: metric.name as MetricName,
      value: metric.value,
      rating,
      delta: metric.delta,
      id: metric.id,
      navigationType: metric.navigationType || 'unknown',
      timestamp: Date.now(),
    };

    this.metrics.set(vitalMetric.name, vitalMetric);

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`[Core Web Vitals] ${metric.name}:`, {
        value: metric.value,
        rating,
        threshold: PERFORMANCE_THRESHOLDS[metric.name as MetricName],
      });
    }

    // Send immediate alert for poor metrics
    if (rating === 'poor') {
      this.sendAlert(vitalMetric);
    }
  }

  private calculateRating(metricName: MetricName, value: number): MetricRating {
    const thresholds = PERFORMANCE_THRESHOLDS[metricName];

    if (value <= thresholds.good) return 'good';
    if (value <= thresholds.poor) return 'needs-improvement';
    return 'poor';
  }

  private monitorCustomMetrics(): void {
    // Monitor JavaScript bundle loading time
    if ('performance' in window && performance.getEntriesByType) {
      const navigationEntries = performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];
      if (navigationEntries.length > 0) {
        const navigation = navigationEntries[0];

        // DNS resolution time
        const dnsTime = navigation.domainLookupEnd - navigation.domainLookupStart;
        this.trackCustomMetric('dns-resolution', dnsTime);

        // TCP connection time
        const tcpTime = navigation.connectEnd - navigation.connectStart;
        this.trackCustomMetric('tcp-connection', tcpTime);

        // SSL negotiation time
        if (navigation.secureConnectionStart > 0) {
          const sslTime = navigation.connectEnd - navigation.secureConnectionStart;
          this.trackCustomMetric('ssl-negotiation', sslTime);
        }

        // DOM processing time
        const domTime = navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart;
        this.trackCustomMetric('dom-processing', domTime);

        // Resource loading time
        const resourceTime = navigation.loadEventEnd - navigation.domContentLoadedEventEnd;
        this.trackCustomMetric('resource-loading', resourceTime);
      }
    }

    // Monitor resource loading
    setTimeout(() => {
      const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
      const jsResources = resources.filter(r => r.name.includes('.js'));
      const cssResources = resources.filter(r => r.name.includes('.css'));

      if (jsResources.length > 0) {
        const totalJsSize = jsResources.reduce((acc, r) => acc + (r.transferSize || 0), 0);
        const avgJsLoadTime = jsResources.reduce((acc, r) => acc + r.duration, 0) / jsResources.length;

        this.trackCustomMetric('js-bundle-size', totalJsSize);
        this.trackCustomMetric('js-load-time', avgJsLoadTime);
      }

      if (cssResources.length > 0) {
        const totalCssSize = cssResources.reduce((acc, r) => acc + (r.transferSize || 0), 0);
        const avgCssLoadTime = cssResources.reduce((acc, r) => acc + r.duration, 0) / cssResources.length;

        this.trackCustomMetric('css-bundle-size', totalCssSize);
        this.trackCustomMetric('css-load-time', avgCssLoadTime);
      }
    }, 5000); // Wait for resources to load
  }

  private trackCustomMetric(name: string, value: number): void {
    if (process.env.NODE_ENV === 'development') {
      console.log(`[Custom Metric] ${name}:`, value);
    }
  }

  private setupReporting(): void {
    // Send report every 30 seconds in development, 5 minutes in production
    const interval = process.env.NODE_ENV === 'development' ? 30000 : 300000;

    setInterval(() => {
      this.sendReport();
    }, interval);
  }

  private sendAlert(metric: VitalMetric): void {
    // Send immediate alert for critical performance issues
    const alertData = {
      type: 'performance-alert',
      metric,
      url: window.location.href,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
    };

    // In development, just log to console
    if (process.env.NODE_ENV === 'development') {
      console.warn('[Performance Alert]', alertData);
      return;
    }

    // Send to analytics endpoint
    fetch('/api/analytics/alerts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alertData),
    }).catch(console.error);
  }

  private sendReport(): void {
    if (this.isReporting || this.metrics.size === 0) return;

    this.isReporting = true;

    const report: PerformanceReport = {
      url: window.location.href,
      timestamp: Date.now(),
      metrics: Array.from(this.metrics.values()),
      deviceInfo: {
        userAgent: navigator.userAgent,
        connection: (navigator as any).connection ? {
          effectiveType: (navigator as any).connection.effectiveType,
          downlink: (navigator as any).connection.downlink,
        } : undefined,
        memory: (performance as any).memory?.usedJSHeapSize,
        cores: navigator.hardwareConcurrency,
      },
    };

    // In development, just log the report
    if (process.env.NODE_ENV === 'development') {
      console.log('[Performance Report]', report);
      this.isReporting = false;
      return;
    }

    // Queue report for sending
    this.reportQueue.push(report);
    this.flushReports();
  }

  private async flushReports(): Promise<void> {
    if (this.reportQueue.length === 0) {
      this.isReporting = false;
      return;
    }

    const reports = [...this.reportQueue];
    this.reportQueue = [];

    try {
      await fetch(ANALYTICS_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reports }),
      });
    } catch (error) {
      console.error('Failed to send performance reports:', error);
      // Re-queue failed reports
      this.reportQueue.unshift(...reports);
    } finally {
      this.isReporting = false;
    }
  }

  // Public API
  public getCurrentMetrics(): VitalMetric[] {
    return Array.from(this.metrics.values());
  }

  public getMetricScore(): number {
    const metrics = this.getCurrentMetrics();
    if (metrics.length === 0) return 0;

    const scores = metrics.map(metric => {
      const rating = metric.rating;
      switch (rating) {
        case 'good': return 100;
        case 'needs-improvement': return 50;
        case 'poor': return 0;
        default: return 0;
      }
    });

    return Math.round(scores.reduce((a: number, b: number) => a + b, 0) / scores.length);
  }

  public generateReport(): string {
    const metrics = this.getCurrentMetrics();
    const score = this.getMetricScore();

    let report = `🚀 Core Web Vitals Report (Score: ${score}/100)\n\n`;

    metrics.forEach(metric => {
      const emoji = metric.rating === 'good' ? '✅' : metric.rating === 'needs-improvement' ? '⚠️' : '❌';
      const threshold = PERFORMANCE_THRESHOLDS[metric.name];

      report += `${emoji} ${metric.name}: ${metric.value.toFixed(1)}${metric.name === 'CLS' ? '' : 'ms'} (${metric.rating})\n`;
      report += `   Target: ≤ ${threshold.good}${metric.name === 'CLS' ? '' : 'ms'} (good), ≤ ${threshold.poor}${metric.name === 'CLS' ? '' : 'ms'} (poor)\n\n`;
    });

    return report;
  }
}

// Global instance
export const coreWebVitalsMonitor = new CoreWebVitalsMonitor();

// React hook for using Core Web Vitals in components
export function useCoreWebVitals() {
  const getCurrentMetrics = () => coreWebVitalsMonitor.getCurrentMetrics();
  const getMetricScore = () => coreWebVitalsMonitor.getMetricScore();
  const generateReport = () => coreWebVitalsMonitor.generateReport();

  return {
    getCurrentMetrics,
    getMetricScore,
    generateReport,
  };
}

// Initialize monitoring when module loads
if (typeof window !== 'undefined') {
  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      // Monitor is already initialized in constructor
    });
  }
}