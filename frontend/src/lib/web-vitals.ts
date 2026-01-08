// Client-side Web Vitals Implementation for Chimera Frontend
// This script should be imported in the root layout or _app file

'use client';

import { onCLS, onFCP, onFID, onLCP, onTTFB, onINP } from 'web-vitals';

// Performance monitoring configuration
const PERFORMANCE_CONFIG = {
  apiEndpoint: '/api/performance-fixed',
  sessionId: typeof window !== 'undefined' ?
    (sessionStorage.getItem('chimera-session-id') ||
     (() => {
       const id = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
       sessionStorage.setItem('chimera-session-id', id);
       return id;
     })()) : null,
  enableDebugLogging: process.env.NODE_ENV === 'development',
  batchMetrics: true,
  batchInterval: 5000, // Send metrics every 5 seconds
  maxBatchSize: 10
};

// Metrics queue for batching
const metricsQueue: any[] = [];
let batchTimeout: NodeJS.Timeout | null = null;

// Enhanced metric data collection
function getEnhancedMetricData() {
  return {
    url: window.location.pathname,
    timestamp: Date.now(),
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight
    },
    connectionType: (navigator as any)?.connection?.effectiveType || 'unknown',
    deviceMemory: (navigator as any)?.deviceMemory || 'unknown',
    hardwareConcurrency: navigator.hardwareConcurrency || 'unknown',
    referrer: document.referrer,
    userAgent: navigator.userAgent
  };
}

// Send metric to API endpoint
async function sendMetric(metricType: string, metricData: any) {
  if (!PERFORMANCE_CONFIG.sessionId) {
    console.warn('No session ID available for metric tracking');
    return;
  }

  const payload = {
    sessionId: PERFORMANCE_CONFIG.sessionId,
    metricType,
    metric: {
      ...metricData,
      ...getEnhancedMetricData()
    }
  };

  if (PERFORMANCE_CONFIG.batchMetrics) {
    metricsQueue.push(payload);

    // Set up batch sending
    if (!batchTimeout) {
      batchTimeout = setTimeout(flushMetricsBatch, PERFORMANCE_CONFIG.batchInterval);
    }

    // Send immediately if batch is full
    if (metricsQueue.length >= PERFORMANCE_CONFIG.maxBatchSize) {
      flushMetricsBatch();
    }
  } else {
    await sendSingleMetric(payload);
  }
}

// Send batched metrics
async function flushMetricsBatch() {
  if (metricsQueue.length === 0) return;

  const batch = metricsQueue.splice(0, PERFORMANCE_CONFIG.maxBatchSize);
  batchTimeout = null;

  try {
    await fetch(PERFORMANCE_CONFIG.apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: PERFORMANCE_CONFIG.sessionId,
        metricType: 'batch',
        metrics: batch
      }),
      keepalive: true
    });

    if (PERFORMANCE_CONFIG.enableDebugLogging) {
      console.log(`📊 Sent batch of ${batch.length} performance metrics`);
    }
  } catch (error) {
    console.warn('Failed to send metrics batch:', error);
    // Return metrics to queue for retry
    metricsQueue.unshift(...batch);
  }

  // Continue batching if more metrics are queued
  if (metricsQueue.length > 0) {
    batchTimeout = setTimeout(flushMetricsBatch, PERFORMANCE_CONFIG.batchInterval);
  }
}

// Send individual metric
async function sendSingleMetric(payload: any) {
  try {
    await fetch(PERFORMANCE_CONFIG.apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      keepalive: true
    });

    if (PERFORMANCE_CONFIG.enableDebugLogging) {
      console.log(`📊 Sent ${payload.metricType} metric:`, payload.metric);
    }
  } catch (error) {
    console.warn('Failed to send performance metric:', error);
  }
}

// Core Web Vitals implementation
export function initializeWebVitals() {
  // Skip if running on server
  if (typeof window === 'undefined') {
    return;
  }

  // Largest Contentful Paint (LCP)
  onLCP((metric) => {
    sendMetric('core_web_vitals', {
      name: 'LCP',
      value: metric.value,
      rating: metric.rating,
      unit: 'ms',
      delta: metric.delta,
      entries: metric.entries.map(entry => ({
        element: entry.element?.tagName,
        url: entry.url,
        size: entry.size,
        loadTime: entry.loadTime,
        renderTime: entry.renderTime
      }))
    });
  });

  // First Input Delay (FID)
  onFID((metric) => {
    sendMetric('core_web_vitals', {
      name: 'FID',
      value: metric.value,
      rating: metric.rating,
      unit: 'ms',
      delta: metric.delta,
      entries: metric.entries.map(entry => ({
        eventType: entry.name,
        startTime: entry.startTime,
        processingStart: entry.processingStart,
        processingEnd: entry.processingEnd,
        cancelable: entry.cancelable
      }))
    });
  });

  // Interaction to Next Paint (INP) - Modern alternative to FID
  onINP((metric) => {
    sendMetric('core_web_vitals', {
      name: 'INP',
      value: metric.value,
      rating: metric.rating,
      unit: 'ms',
      delta: metric.delta,
      entries: metric.entries.map(entry => ({
        eventType: entry.name,
        startTime: entry.startTime,
        processingStart: entry.processingStart,
        processingEnd: entry.processingEnd,
        duration: entry.duration,
        interactionId: (entry as any).interactionId
      }))
    });
  });

  // Cumulative Layout Shift (CLS)
  onCLS((metric) => {
    sendMetric('core_web_vitals', {
      name: 'CLS',
      value: metric.value,
      rating: metric.rating,
      unit: 'score',
      delta: metric.delta,
      entries: metric.entries.map(entry => ({
        value: entry.value,
        hadRecentInput: entry.hadRecentInput,
        sources: entry.sources?.map(source => ({
          element: (source.node as Element)?.tagName,
          currentRect: source.currentRect,
          previousRect: source.previousRect
        }))
      }))
    });
  });

  // First Contentful Paint (FCP)
  onFCP((metric) => {
    sendMetric('core_web_vitals', {
      name: 'FCP',
      value: metric.value,
      rating: metric.rating,
      unit: 'ms',
      delta: metric.delta
    });
  });

  // Time to First Byte (TTFB)
  onTTFB((metric) => {
    sendMetric('core_web_vitals', {
      name: 'TTFB',
      value: metric.value,
      rating: metric.rating,
      unit: 'ms',
      delta: metric.delta
    });
  });

  // Additional custom metrics for Chimera-specific tracking
  initializeCustomMetrics();

  // Page visibility change tracking
  document.addEventListener('visibilitychange', () => {
    sendMetric('page_lifecycle', {
      name: 'visibility_change',
      value: document.hidden ? 1 : 0,
      rating: 'info',
      visibilityState: document.visibilityState,
      hidden: document.hidden
    });
  });

  // Unload handler for final metrics
  window.addEventListener('beforeunload', () => {
    flushMetricsBatch();

    // Send session duration
    const sessionStart = parseInt(sessionStorage.getItem('chimera-session-start') || '0');
    if (sessionStart) {
      const sessionDuration = Date.now() - sessionStart;
      sendMetric('session', {
        name: 'session_duration',
        value: sessionDuration,
        rating: sessionDuration < 300000 ? 'good' : 'poor', // 5 minutes threshold
        unit: 'ms'
      });
    }
  });

  // Track session start
  if (!sessionStorage.getItem('chimera-session-start')) {
    sessionStorage.setItem('chimera-session-start', Date.now().toString());
  }

  console.log('✅ Chimera Web Vitals monitoring initialized');
}

// Custom metrics for Chimera-specific user journeys
function initializeCustomMetrics() {
  // Track route changes in Next.js
  let routeChangeStart: number;

  const trackRouteChange = () => {
    if (routeChangeStart) {
      const routeChangeDuration = performance.now() - routeChangeStart;
      sendMetric('navigation', {
        name: 'route_change',
        value: routeChangeDuration,
        rating: routeChangeDuration < 1000 ? 'good' : routeChangeDuration < 3000 ? 'needs-improvement' : 'poor',
        unit: 'ms',
        fromUrl: document.referrer,
        toUrl: window.location.pathname
      });
    }
    routeChangeStart = performance.now();
  };

  // Listen for navigation events
  window.addEventListener('popstate', trackRouteChange);

  // Track initial page load
  trackRouteChange();

  // Track API call performance
  const originalFetch = window.fetch;
  window.fetch = async function(...args) {
    const startTime = performance.now();
    const url = typeof args[0] === 'string'
      ? args[0]
      : args[0] instanceof Request
        ? args[0].url
        : args[0].toString();

    try {
      const response = await originalFetch.apply(this, args);
      const duration = performance.now() - startTime;

      // Only track API calls to our backend
      if (url.includes('/api/')) {
        sendMetric('api_timing', {
          name: 'api_call',
          value: duration,
          rating: duration < 500 ? 'good' : duration < 1000 ? 'needs-improvement' : 'poor',
          unit: 'ms',
          endpoint: url,
          method: args[1]?.method || 'GET',
          status: response.status,
          success: response.ok
        });
      }

      return response;
    } catch (error) {
      const duration = performance.now() - startTime;

      if (url.includes('/api/')) {
        sendMetric('api_timing', {
          name: 'api_call_error',
          value: duration,
          rating: 'poor',
          unit: 'ms',
          endpoint: url,
          error: (error as Error).message,
          method: args[1]?.method || 'GET'
        });
      }

      throw error;
    }
  };

  // Track form submission performance
  document.addEventListener('submit', (event) => {
    const form = event.target as HTMLFormElement;
    const formId = form.id || form.className || 'unknown-form';

    sendMetric('user_interaction', {
      name: 'form_submission',
      value: performance.now(),
      rating: 'info',
      formId,
      action: form.action,
      method: form.method
    });
  });

  // Track critical user interactions
  ['click', 'keydown', 'touchstart'].forEach(eventType => {
    document.addEventListener(eventType, (event) => {
      const target = event.target as HTMLElement;
      const isImportant = target.matches('button, a, input, [role="button"]');

      if (isImportant) {
        sendMetric('user_interaction', {
          name: eventType,
          value: performance.now(),
          rating: 'info',
          elementType: target.tagName,
          elementId: target.id,
          elementClass: target.className
        });
      }
    }, { passive: true });
  });
}

// User journey tracking utility
export class UserJourneyTracker {
  private startTime: number;
  private journeyName: string;
  private interactions: any[] = [];
  private apiCalls: any[] = [];
  private errors: any[] = [];

  constructor(journeyName: string) {
    this.journeyName = journeyName;
    this.startTime = performance.now();
  }

  addInteraction(type: string, details: any = {}) {
    this.interactions.push({
      type,
      timestamp: performance.now() - this.startTime,
      ...details
    });
  }

  addApiCall(endpoint: string, duration: number, success: boolean = true) {
    this.apiCalls.push({
      endpoint,
      duration,
      success,
      timestamp: performance.now() - this.startTime
    });
  }

  addError(error: string | Error, context: any = {}) {
    this.errors.push({
      error: error instanceof Error ? error.message : error,
      context,
      timestamp: performance.now() - this.startTime
    });
  }

  complete() {
    const totalDuration = performance.now() - this.startTime;

    sendMetric('user_journey', {
      name: this.journeyName,
      value: totalDuration,
      rating: totalDuration < 5000 ? 'good' : totalDuration < 10000 ? 'needs-improvement' : 'poor',
      unit: 'ms',
      interactionCount: this.interactions.length,
      apiCallCount: this.apiCalls.length,
      errorCount: this.errors.length,
      avgApiCallDuration: this.apiCalls.length > 0
        ? this.apiCalls.reduce((sum, call) => sum + call.duration, 0) / this.apiCalls.length
        : 0,
      interactions: this.interactions,
      apiCalls: this.apiCalls,
      errors: this.errors
    });
  }
}

// Export for external usage
export { sendMetric };