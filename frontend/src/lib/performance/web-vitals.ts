'use client';

import { onCLS, onFID, onFCP, onLCP, onTTFB } from 'web-vitals';

export interface VitalsMetric {
  name: 'CLS' | 'FID' | 'FCP' | 'LCP' | 'TTFB';
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  delta: number;
  navigationType: 'navigate' | 'reload' | 'back-forward' | 'prerender';
}

interface VitalsReport {
  id: string;
  timestamp: number;
  url: string;
  userAgent: string;
  connectionType?: string;
  metrics: VitalsMetric[];
}

// Thresholds based on Google's Core Web Vitals recommendations
const VITALS_THRESHOLDS = {
  LCP: { good: 2500, poor: 4000 },
  FID: { good: 100, poor: 300 },
  CLS: { good: 0.1, poor: 0.25 },
  FCP: { good: 1800, poor: 3000 },
  TTFB: { good: 800, poor: 1800 },
} as const;

function getVitalRating(name: VitalsMetric['name'], value: number): VitalsMetric['rating'] {
  const thresholds = VITALS_THRESHOLDS[name];
  if (value <= thresholds.good) return 'good';
  if (value <= thresholds.poor) return 'needs-improvement';
  return 'poor';
}

class WebVitalsManager {
  private metrics: Map<string, VitalsMetric> = new Map();
  private callbacks: Array<(metric: VitalsMetric) => void> = [];
  private reportCallbacks: Array<(report: VitalsReport) => void> = [];
  private reportTimer: NodeJS.Timeout | null = null;

  constructor() {
    this.initVitalsCollection();
  }

  private initVitalsCollection() {
    // Core Web Vitals
    onLCP((metric) => this.handleMetric('LCP', metric));
    onFID((metric) => this.handleMetric('FID', metric));
    onCLS((metric) => this.handleMetric('CLS', metric));

    // Additional performance metrics
    onFCP((metric) => this.handleMetric('FCP', metric));
    onTTFB((metric) => this.handleMetric('TTFB', metric));

    // Schedule report generation
    this.scheduleReport();
  }

  private handleMetric(name: VitalsMetric['name'], metric: any) {
    const vitalMetric: VitalsMetric = {
      name,
      value: metric.value,
      rating: getVitalRating(name, metric.value),
      delta: metric.delta,
      navigationType: metric.navigationType || 'navigate',
    };

    this.metrics.set(name, vitalMetric);

    // Call individual metric callbacks
    this.callbacks.forEach(callback => callback(vitalMetric));

    console.log(`[Core Web Vitals] ${name}:`, {
      value: vitalMetric.value,
      rating: vitalMetric.rating,
      threshold: VITALS_THRESHOLDS[name],
    });
  }

  private scheduleReport() {
    // Clear existing timer
    if (this.reportTimer) {
      clearTimeout(this.reportTimer);
    }

    // Schedule report generation after 5 seconds of inactivity
    this.reportTimer = setTimeout(() => {
      this.generateReport();
    }, 5000);
  }

  private generateReport() {
    if (this.metrics.size === 0) return;

    const report: VitalsReport = {
      id: `vitals-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      connectionType: this.getConnectionType(),
      metrics: Array.from(this.metrics.values()),
    };

    // Call report callbacks
    this.reportCallbacks.forEach(callback => callback(report));

    // Send to analytics
    this.sendToAnalytics(report);
  }

  private getConnectionType(): string | undefined {
    // connection API is experimental and not in standard Navigator type
    const connection = (navigator as Navigator & { connection?: { effectiveType?: string }; mozConnection?: { effectiveType?: string }; webkitConnection?: { effectiveType?: string } }).connection || (navigator as Navigator & { mozConnection?: { effectiveType?: string } }).mozConnection || (navigator as Navigator & { webkitConnection?: { effectiveType?: string } }).webkitConnection;
    return connection?.effectiveType;
  }

  private async sendToAnalytics(report: VitalsReport) {
    try {
      // Send to your analytics endpoint
      await fetch('/api/analytics/web-vitals', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(report),
      });
    } catch (error) {
      console.warn('Failed to send Web Vitals report:', error);
    }
  }

  // Public API
  onMetric(callback: (metric: VitalsMetric) => void) {
    this.callbacks.push(callback);
  }

  onReport(callback: (report: VitalsReport) => void) {
    this.reportCallbacks.push(callback);
  }

  getCurrentMetrics(): VitalsMetric[] {
    return Array.from(this.metrics.values());
  }

  getMetric(name: VitalsMetric['name']): VitalsMetric | undefined {
    return this.metrics.get(name);
  }

  // Force immediate report generation
  generateImmediateReport() {
    this.generateReport();
  }
}

// Singleton instance
let webVitalsManager: WebVitalsManager | null = null;

export function getWebVitalsManager(): WebVitalsManager {
  if (typeof window === 'undefined') {
    // SSR safe - return a mock object
    return {
      onMetric: () => { },
      onReport: () => { },
      getCurrentMetrics: () => [],
      getMetric: () => undefined,
      generateImmediateReport: () => { },
    } as unknown as WebVitalsManager;
  }

  if (!webVitalsManager) {
    webVitalsManager = new WebVitalsManager();
  }

  return webVitalsManager;
}

// Helper function for React components
export function useWebVitals() {
  const manager = getWebVitalsManager();

  return {
    manager,
    getCurrentMetrics: () => manager.getCurrentMetrics(),
    getMetric: (name: VitalsMetric['name']) => manager.getMetric(name),
    onMetric: (callback: (metric: VitalsMetric) => void) => manager.onMetric(callback),
    onReport: (callback: (report: VitalsReport) => void) => manager.onReport(callback),
  };
}

export { VITALS_THRESHOLDS };
