// Frontend Performance Monitoring for Chimera Next.js Application
// Provides Core Web Vitals tracking, React performance monitoring, and user experience metrics

export interface PerformanceMetric {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  timestamp: number;
  url: string;
  connection?: string;
  navigationType?: string;
}

export interface UserSession {
  sessionId: string;
  userId?: string;
  startTime: number;
  userAgent: string;
  viewport: {
    width: number;
    height: number;
  };
  connection: {
    effectiveType: string;
    downlink: number;
    rtt: number;
  } | null;
}

export interface ReactComponentMetric {
  componentName: string;
  renderTime: number;
  updateTime: number;
  mountTime: number;
  propsSize: number;
  renderCount: number;
  timestamp: number;
}

export interface APICallMetric {
  endpoint: string;
  method: string;
  duration: number;
  status: number;
  size: number;
  fromCache: boolean;
  timestamp: number;
}

class PerformanceMonitor {
  private sessionId: string;
  private metrics: PerformanceMetric[] = [];
  private componentMetrics: ReactComponentMetric[] = [];
  private apiMetrics: APICallMetric[] = [];
  private observer: PerformanceObserver | null = null;
  private isInitialized = false;

  constructor() {
    this.sessionId = this.generateSessionId();
    this.initialize();
  }

  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  public initialize(): void {
    if (this.isInitialized || typeof window === 'undefined') return;

    this.setupPerformanceObserver();
    this.trackCoreWebVitals();
    this.setupResourceObserver();
    this.setupNavigationObserver();
    this.setupUserSession();
    this.setupBeforeUnloadTracking();

    this.isInitialized = true;
  }

  private setupPerformanceObserver(): void {
    try {
      this.observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();

        for (const entry of entries) {
          this.processPerformanceEntry(entry);
        }
      });

      // Observe various performance entry types
      this.observer.observe({
        entryTypes: ['measure', 'navigation', 'resource', 'paint', 'largest-contentful-paint']
      });
    } catch (error) {
      console.warn('Performance Observer not supported:', error);
    }
  }

  private processPerformanceEntry(entry: PerformanceEntry): void {
    const metric: PerformanceMetric = {
      name: entry.name,
      value: entry.duration || entry.startTime,
      rating: this.getRating(entry.name, entry.duration || entry.startTime),
      timestamp: Date.now(),
      url: window.location.href
    };

    if (entry.entryType === 'navigation') {
      const navEntry = entry as PerformanceNavigationTiming;
      metric.navigationType = navEntry.type;
    }

    this.metrics.push(metric);
    this.sendMetricToBackend(metric);
  }

  private trackCoreWebVitals(): void {
    // First Contentful Paint (FCP)
    this.observePerformanceMetric('first-contentful-paint', (entry) => {
      this.trackMetric('FCP', entry.startTime, this.getFCPRating(entry.startTime));
    });

    // Largest Contentful Paint (LCP)
    this.observePerformanceMetric('largest-contentful-paint', (entry) => {
      this.trackMetric('LCP', entry.startTime, this.getLCPRating(entry.startTime));
    });

    // First Input Delay (FID) - using polyfill or estimate
    if ('PerformanceEventTiming' in window) {
      this.observePerformanceMetric('first-input', (entry: PerformanceEntry) => {
        const eventEntry = entry as PerformanceEntry & { processingStart?: number };
        const fid = (eventEntry.processingStart ?? entry.startTime) - entry.startTime;
        this.trackMetric('FID', fid, this.getFIDRating(fid));
      });
    }

    // Cumulative Layout Shift (CLS)
    this.trackCLS();

    // Custom metrics
    this.trackTimeToInteractive();
    this.trackFirstInputDelay();
  }

  private observePerformanceMetric(
    type: string,
    callback: (entry: PerformanceEntry) => void
  ): void {
    try {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(callback);
      });

      observer.observe({ entryTypes: [type] });
    } catch (error) {
      console.warn(`Cannot observe ${type}:`, error);
    }
  }

  private trackCLS(): void {
    let clsValue = 0;
    const sessionEntries: PerformanceEntry[] = [];
    let sessionValue = 0;

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
            // Only count unexpected layout shifts
            const layoutShiftEntry = entry as PerformanceEntry & { hadRecentInput?: boolean; value?: number };
            if (!layoutShiftEntry.hadRecentInput) {
              sessionEntries.push(entry);
              sessionValue += layoutShiftEntry.value ?? 0;
            }
          }

        clsValue = Math.max(clsValue, sessionValue);
        this.trackMetric('CLS', clsValue, this.getCLSRating(clsValue));
      });

      observer.observe({ entryTypes: ['layout-shift'] });
    } catch (error) {
      console.warn('Layout shift tracking not supported:', error);
    }
  }

  private trackTimeToInteractive(): void {
    // Estimate TTI based on main thread idle time
    if (document.readyState === 'complete') {
      this.measureTTI();
    } else {
      window.addEventListener('load', () => this.measureTTI());
    }
  }

  private measureTTI(): void {
    setTimeout(() => {
      const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navEntry) {
        // Simplified TTI calculation
        const tti = navEntry.domContentLoadedEventEnd || navEntry.loadEventEnd;
        this.trackMetric('TTI', tti, this.getTTIRating(tti));
      }
    }, 100);
  }

  private trackFirstInputDelay(): void {
    let firstInput = true;

    const measureFID = () => {
      if (!firstInput) return;
      firstInput = false;

      const startTime = performance.now();
      setTimeout(() => {
        const fid = performance.now() - startTime;
        this.trackMetric('FID', fid, this.getFIDRating(fid));
      }, 0);
    };

    ['click', 'keydown', 'pointerdown', 'touchstart'].forEach(type => {
      document.addEventListener(type, measureFID, { once: true, passive: true });
    });
  }

  private setupResourceObserver(): void {
    try {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();

        for (const entry of entries) {
          const resourceEntry = entry as PerformanceResourceTiming;

          this.trackResourceMetric({
            name: entry.name,
            duration: entry.duration,
            size: resourceEntry.transferSize || 0,
            type: this.getResourceType(entry.name),
            fromCache: resourceEntry.transferSize === 0 && resourceEntry.decodedBodySize > 0,
            timestamp: Date.now()
          });
        }
      });

      observer.observe({ entryTypes: ['resource'] });
    } catch (error) {
      console.warn('Resource timing not supported:', error);
    }
  }

  private setupNavigationObserver(): void {
    try {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();

        for (const entry of entries) {
          const navEntry = entry as PerformanceNavigationTiming;

          this.trackNavigationMetrics(navEntry);
        }
      });

      observer.observe({ entryTypes: ['navigation'] });
    } catch (error) {
      console.warn('Navigation timing not supported:', error);
    }
  }

  private trackNavigationMetrics(entry: PerformanceNavigationTiming): void {
    const metrics = {
      'DNS Lookup': entry.domainLookupEnd - entry.domainLookupStart,
      'TCP Connect': entry.connectEnd - entry.connectStart,
      'SSL Handshake': entry.secureConnectionStart ? entry.connectEnd - entry.secureConnectionStart : 0,
      'Request': entry.responseStart - entry.requestStart,
      'Response': entry.responseEnd - entry.responseStart,
      'DOM Processing': entry.domContentLoadedEventStart - entry.responseEnd,
      'Load Complete': entry.loadEventEnd - entry.loadEventStart
    };

    Object.entries(metrics).forEach(([name, value]) => {
      if (value > 0) {
        this.trackMetric(name, value, this.getTimingRating(name, value));
      }
    });
  }

  private setupUserSession(): void {
    const session: UserSession = {
      sessionId: this.sessionId,
      startTime: Date.now(),
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      connection: this.getConnectionInfo()
    };

    this.sendSessionInfo(session);

    // Track viewport changes
    window.addEventListener('resize', () => {
      this.trackMetric('Viewport Change', Date.now(), 'good');
    });
  }

  private getConnectionInfo() {
    const nav = navigator as Navigator & { connection?: { effectiveType?: string; downlink?: number; rtt?: number }; mozConnection?: { effectiveType?: string; downlink?: number; rtt?: number }; webkitConnection?: { effectiveType?: string; downlink?: number; rtt?: number } };
    const conn = nav.connection || nav.mozConnection || nav.webkitConnection;

    if (!conn) return null;

    return {
      effectiveType: conn.effectiveType || 'unknown',
      downlink: conn.downlink || 0,
      rtt: conn.rtt || 0
    };
  }

  private setupBeforeUnloadTracking(): void {
    window.addEventListener('beforeunload', () => {
      // Send remaining metrics before page unload
      this.flushMetrics();
    });
  }

  public trackReactComponent(componentName: string, renderTime: number): void {
    const metric: ReactComponentMetric = {
      componentName,
      renderTime,
      updateTime: 0,
      mountTime: 0,
      propsSize: 0,
      renderCount: 1,
      timestamp: Date.now()
    };

    this.componentMetrics.push(metric);
    this.sendComponentMetric(metric);
  }

  public trackAPICall(
    endpoint: string,
    method: string,
    duration: number,
    status: number,
    size: number = 0,
    fromCache: boolean = false
  ): void {
    const metric: APICallMetric = {
      endpoint,
      method,
      duration,
      status,
      size,
      fromCache,
      timestamp: Date.now()
    };

    this.apiMetrics.push(metric);
    this.sendAPIMetric(metric);

    // Track slow API calls
    if (duration > 2000) {
      this.trackMetric('Slow API Call', duration, 'poor');
    }
  }

  public trackUserInteraction(action: string, target: string, duration: number = 0): void {
    this.trackMetric(`User Interaction: ${action}`, duration, 'good');

    // Send interaction data
    this.sendInteractionMetric({
      action,
      target,
      duration,
      timestamp: Date.now(),
      url: window.location.href
    });
  }

  private trackMetric(name: string, value: number, rating: 'good' | 'needs-improvement' | 'poor'): void {
    const metric: PerformanceMetric = {
      name,
      value,
      rating,
      timestamp: Date.now(),
      url: window.location.href
    };

    this.metrics.push(metric);
    this.sendMetricToBackend(metric);
  }

  private trackResourceMetric(resource: { name?: string; duration: number; type?: string; size?: number; fromCache?: boolean; timestamp?: number }): void {
    // Track resource loading performance
    if (resource.duration > 1000) {
      this.trackMetric(`Slow Resource: ${resource.type}`, resource.duration, 'poor');
    }
  }

  // Rating functions based on Core Web Vitals thresholds
  private getFCPRating(value: number): 'good' | 'needs-improvement' | 'poor' {
    return value <= 1800 ? 'good' : value <= 3000 ? 'needs-improvement' : 'poor';
  }

  private getLCPRating(value: number): 'good' | 'needs-improvement' | 'poor' {
    return value <= 2500 ? 'good' : value <= 4000 ? 'needs-improvement' : 'poor';
  }

  private getFIDRating(value: number): 'good' | 'needs-improvement' | 'poor' {
    return value <= 100 ? 'good' : value <= 300 ? 'needs-improvement' : 'poor';
  }

  private getCLSRating(value: number): 'good' | 'needs-improvement' | 'poor' {
    return value <= 0.1 ? 'good' : value <= 0.25 ? 'needs-improvement' : 'poor';
  }

  private getTTIRating(value: number): 'good' | 'needs-improvement' | 'poor' {
    return value <= 3800 ? 'good' : value <= 7300 ? 'needs-improvement' : 'poor';
  }

  private getTimingRating(name: string, value: number): 'good' | 'needs-improvement' | 'poor' {
    // General timing thresholds
    switch (name) {
      case 'DNS Lookup':
        return value <= 50 ? 'good' : value <= 200 ? 'needs-improvement' : 'poor';
      case 'TCP Connect':
        return value <= 100 ? 'good' : value <= 300 ? 'needs-improvement' : 'poor';
      case 'Request':
        return value <= 200 ? 'good' : value <= 500 ? 'needs-improvement' : 'poor';
      default:
        return value <= 1000 ? 'good' : value <= 3000 ? 'needs-improvement' : 'poor';
    }
  }

  private getRating(name: string, value: number): 'good' | 'needs-improvement' | 'poor' {
    // Default rating based on common performance thresholds
    if (name.includes('paint') || name.includes('load')) {
      return this.getLCPRating(value);
    }
    return value <= 100 ? 'good' : value <= 300 ? 'needs-improvement' : 'poor';
  }

  private getNavigationType(type: number): string {
    switch (type) {
      case 0: return 'navigate';
      case 1: return 'reload';
      case 2: return 'back_forward';
      case 255: return 'prerender';
      default: return 'unknown';
    }
  }

  private getResourceType(url: string): string {
    const ext = url.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'js': return 'script';
      case 'css': return 'stylesheet';
      case 'png':
      case 'jpg':
      case 'jpeg':
      case 'gif':
      case 'svg':
      case 'webp': return 'image';
      case 'woff':
      case 'woff2':
      case 'ttf':
      case 'otf': return 'font';
      default: return 'other';
    }
  }

  // Data transmission methods
  private async sendMetricToBackend(metric: PerformanceMetric): Promise<void> {
    try {
      await fetch('/api/performance/metrics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          metric
        })
      });
    } catch (error) {
      console.warn('Failed to send metric:', error);
      // Fallback to localStorage for offline storage
      this.storeMetricLocally(metric);
    }
  }

  private async sendComponentMetric(metric: ReactComponentMetric): Promise<void> {
    try {
      await fetch('/api/performance/components', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          metric
        })
      });
    } catch (error) {
      console.warn('Failed to send component metric:', error);
    }
  }

  private async sendAPIMetric(metric: APICallMetric): Promise<void> {
    try {
      await fetch('/api/performance/api-calls', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          metric
        })
      });
    } catch (error) {
      console.warn('Failed to send API metric:', error);
    }
  }

  private async sendSessionInfo(session: UserSession): Promise<void> {
    try {
      await fetch('/api/performance/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(session)
      });
    } catch (error) {
      console.warn('Failed to send session info:', error);
    }
  }

  private async sendInteractionMetric(interaction: { action: string; target: string; duration: number; timestamp: number; url: string }): Promise<void> {
    try {
      await fetch('/api/performance/interactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          interaction
        })
      });
    } catch (error) {
      console.warn('Failed to send interaction metric:', error);
    }
  }

  private storeMetricLocally(metric: PerformanceMetric): void {
    try {
      const stored = localStorage.getItem('chimera-performance-metrics') || '[]';
      const metrics = JSON.parse(stored);
      metrics.push(metric);

      // Keep only last 100 metrics to prevent storage overflow
      if (metrics.length > 100) {
        metrics.splice(0, metrics.length - 100);
      }

      localStorage.setItem('chimera-performance-metrics', JSON.stringify(metrics));
    } catch (error) {
      console.warn('Failed to store metric locally:', error);
    }
  }

  public flushMetrics(): void {
    // Send any remaining metrics using sendBeacon if available
    if (navigator.sendBeacon && this.metrics.length > 0) {
      const data = JSON.stringify({
        sessionId: this.sessionId,
        metrics: this.metrics,
        componentMetrics: this.componentMetrics,
        apiMetrics: this.apiMetrics
      });

      navigator.sendBeacon('/api/performance/flush', data);
    }
  }

  public getSessionMetrics() {
    return {
      sessionId: this.sessionId,
      metrics: this.metrics,
      componentMetrics: this.componentMetrics,
      apiMetrics: this.apiMetrics
    };
  }

  public getPerformanceSummary() {
    const coreVitals = this.metrics.filter(m =>
      ['FCP', 'LCP', 'FID', 'CLS', 'TTI'].includes(m.name)
    );

    const summary = {
      sessionId: this.sessionId,
      url: window.location.href,
      timestamp: Date.now(),
      coreVitals: coreVitals.reduce((acc, metric) => {
        acc[metric.name] = {
          value: metric.value,
          rating: metric.rating
        };
        return acc;
      }, {} as Record<string, { value: number; rating: string }>),
      apiCalls: this.apiMetrics.length,
      slowApiCalls: this.apiMetrics.filter(m => m.duration > 2000).length,
      componentRenders: this.componentMetrics.length,
      totalMetrics: this.metrics.length
    };

    return summary;
  }
}

// Global performance monitor instance
export const performanceMonitor = new PerformanceMonitor();

// React Hook for component performance tracking
export const usePerformanceTracking = (componentName: string) => {
  const trackRender = (renderTime: number) => {
    performanceMonitor.trackReactComponent(componentName, renderTime);
  };

  const trackInteraction = (action: string, target: string = componentName) => {
    performanceMonitor.trackUserInteraction(action, target);
  };

  return { trackRender, trackInteraction };
};

export default performanceMonitor;