// Performance Monitoring Implementation for Chimera AI Frontend
// This script implements comprehensive Web Vitals and RUM monitoring

(function() {
  'use strict';

  // Performance metrics collection
  const PerformanceMonitor = {
    sessionId: Date.now().toString(36) + Math.random().toString(36).substr(2),
    metrics: {},
    navigationStart: performance.timing?.navigationStart || performance.timeOrigin,

    // Core Web Vitals implementation
    initCoreWebVitals() {
      // Largest Contentful Paint (LCP)
      this.measureLCP();

      // First Input Delay (FID) / Interaction to Next Paint (INP)
      this.measureFID();

      // Cumulative Layout Shift (CLS)
      this.measureCLS();

      // First Contentful Paint (FCP)
      this.measureFCP();

      // Time to Interactive (TTI)
      this.measureTTI();

      // Total Blocking Time (TBT)
      this.measureTBT();
    },

    // LCP measurement
    measureLCP() {
      if (typeof PerformanceObserver !== 'undefined') {
        try {
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];

            const lcp = lastEntry.startTime;
            const rating = lcp <= 2500 ? 'good' : lcp <= 4000 ? 'needs-improvement' : 'poor';

            this.recordMetric('LCP', lcp, rating);
            this.sendToAnalytics('lcp', lcp, rating, {
              element: lastEntry.element?.tagName,
              url: lastEntry.url,
              size: lastEntry.size
            });
          });

          observer.observe({ entryTypes: ['largest-contentful-paint'] });
        } catch (e) {
          console.warn('LCP measurement not supported:', e);
        }
      }
    },

    // FID measurement (for legacy browsers) and INP for modern browsers
    measureFID() {
      if (typeof PerformanceObserver !== 'undefined') {
        try {
          // First Input Delay
          const fidObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach((entry) => {
              const fid = entry.processingStart - entry.startTime;
              const rating = fid <= 100 ? 'good' : fid <= 300 ? 'needs-improvement' : 'poor';

              this.recordMetric('FID', fid, rating);
              this.sendToAnalytics('fid', fid, rating, {
                eventType: entry.name,
                target: entry.target?.tagName
              });
            });
          });

          fidObserver.observe({ entryTypes: ['first-input'] });

          // Interaction to Next Paint (INP) for modern browsers
          const inpObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach((entry) => {
              if (entry.interactionId) {
                const inp = entry.duration;
                const rating = inp <= 200 ? 'good' : inp <= 500 ? 'needs-improvement' : 'poor';

                this.recordMetric('INP', inp, rating);
                this.sendToAnalytics('inp', inp, rating, {
                  interactionType: entry.name,
                  target: entry.target?.tagName
                });
              }
            });
          });

          inpObserver.observe({ entryTypes: ['event'] });
        } catch (e) {
          console.warn('FID/INP measurement not supported:', e);
        }
      }
    },

    // CLS measurement
    measureCLS() {
      if (typeof PerformanceObserver !== 'undefined') {
        try {
          let clsValue = 0;
          let clsEntries = [];

          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach((entry) => {
              if (!entry.hadRecentInput) {
                clsValue += entry.value;
                clsEntries.push(entry);
              }
            });

            const rating = clsValue <= 0.1 ? 'good' : clsValue <= 0.25 ? 'needs-improvement' : 'poor';

            this.recordMetric('CLS', clsValue, rating);
            this.sendToAnalytics('cls', clsValue, rating, {
              entryCount: clsEntries.length,
              sources: clsEntries.map(e => e.sources?.[0]?.node?.tagName).filter(Boolean)
            });
          });

          observer.observe({ entryTypes: ['layout-shift'] });
        } catch (e) {
          console.warn('CLS measurement not supported:', e);
        }
      }
    },

    // FCP measurement
    measureFCP() {
      if (typeof PerformanceObserver !== 'undefined') {
        try {
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach((entry) => {
              if (entry.name === 'first-contentful-paint') {
                const fcp = entry.startTime;
                const rating = fcp <= 1800 ? 'good' : fcp <= 3000 ? 'needs-improvement' : 'poor';

                this.recordMetric('FCP', fcp, rating);
                this.sendToAnalytics('fcp', fcp, rating);
              }
            });
          });

          observer.observe({ entryTypes: ['paint'] });
        } catch (e) {
          console.warn('FCP measurement not supported:', e);
        }
      }
    },

    // Time to Interactive (TTI) estimation
    measureTTI() {
      if (document.readyState === 'complete') {
        this.calculateTTI();
      } else {
        window.addEventListener('load', () => {
          setTimeout(() => this.calculateTTI(), 0);
        });
      }
    },

    calculateTTI() {
      const navigation = performance.getEntriesByType('navigation')[0];
      if (!navigation) return;

      // Simple TTI estimation based on load events and long tasks
      const domContentLoaded = navigation.domContentLoadedEventEnd;
      const loadComplete = navigation.loadEventEnd;

      // Check for long tasks after DCL
      if (typeof PerformanceObserver !== 'undefined') {
        try {
          const longTasks = performance.getEntriesByType('longtask')
            .filter(task => task.startTime > domContentLoaded);

          const tti = longTasks.length > 0 ?
            Math.max(...longTasks.map(t => t.startTime + t.duration)) :
            loadComplete;

          const rating = tti <= 3800 ? 'good' : tti <= 7300 ? 'needs-improvement' : 'poor';

          this.recordMetric('TTI', tti, rating);
          this.sendToAnalytics('tti', tti, rating, {
            longTaskCount: longTasks.length,
            dclTime: domContentLoaded
          });
        } catch (e) {
          console.warn('TTI estimation failed:', e);
        }
      }
    },

    // Total Blocking Time (TBT) measurement
    measureTBT() {
      if (typeof PerformanceObserver !== 'undefined') {
        try {
          let tbtValue = 0;
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach((entry) => {
              // Tasks longer than 50ms contribute to TBT
              if (entry.duration > 50) {
                tbtValue += entry.duration - 50;
              }
            });

            const rating = tbtValue <= 200 ? 'good' : tbtValue <= 600 ? 'needs-improvement' : 'poor';

            this.recordMetric('TBT', tbtValue, rating);
            this.sendToAnalytics('tbt', tbtValue, rating, {
              longTaskCount: entries.length
            });
          });

          observer.observe({ entryTypes: ['longtask'] });
        } catch (e) {
          console.warn('TBT measurement not supported:', e);
        }
      }
    },

    // Resource timing analysis
    analyzeResourceTiming() {
      const resources = performance.getEntriesByType('resource');

      const analysis = {
        totalResources: resources.length,
        slowResources: [],
        largeResources: [],
        cacheHitRatio: 0,
        downloadTime: 0,
        dnsTime: 0,
        connectionTime: 0
      };

      let cacheHits = 0;
      let totalDownloadTime = 0;
      let totalDnsTime = 0;
      let totalConnectionTime = 0;

      resources.forEach(resource => {
        const downloadTime = resource.responseEnd - resource.responseStart;
        const dnsTime = resource.domainLookupEnd - resource.domainLookupStart;
        const connectionTime = resource.connectEnd - resource.connectStart;

        // Identify slow resources (>1s download)
        if (downloadTime > 1000) {
          analysis.slowResources.push({
            name: resource.name,
            duration: downloadTime,
            size: resource.transferSize
          });
        }

        // Identify large resources (>1MB)
        if (resource.transferSize > 1024 * 1024) {
          analysis.largeResources.push({
            name: resource.name,
            size: resource.transferSize,
            duration: downloadTime
          });
        }

        // Cache analysis
        if (resource.transferSize === 0 && resource.decodedBodySize > 0) {
          cacheHits++;
        }

        totalDownloadTime += downloadTime;
        totalDnsTime += dnsTime || 0;
        totalConnectionTime += connectionTime || 0;
      });

      analysis.cacheHitRatio = resources.length > 0 ? (cacheHits / resources.length) * 100 : 0;
      analysis.avgDownloadTime = totalDownloadTime / resources.length;
      analysis.avgDnsTime = totalDnsTime / resources.length;
      analysis.avgConnectionTime = totalConnectionTime / resources.length;

      this.sendToAnalytics('resource_analysis', analysis.avgDownloadTime,
        analysis.avgDownloadTime <= 500 ? 'good' : 'poor', analysis);

      return analysis;
    },

    // User journey tracking
    trackUserJourney(routeName, startTime = performance.now()) {
      const journeyKey = `journey_${routeName}`;

      if (!this.metrics[journeyKey]) {
        this.metrics[journeyKey] = {
          startTime,
          interactions: [],
          apiCalls: [],
          errors: []
        };
      }

      return {
        addInteraction: (type, target, duration) => {
          this.metrics[journeyKey].interactions.push({
            type, target, duration, timestamp: performance.now()
          });
        },
        addApiCall: (endpoint, duration, status) => {
          this.metrics[journeyKey].apiCalls.push({
            endpoint, duration, status, timestamp: performance.now()
          });
        },
        addError: (error, context) => {
          this.metrics[journeyKey].errors.push({
            error: error.message, context, timestamp: performance.now()
          });
        },
        complete: () => {
          const journey = this.metrics[journeyKey];
          journey.totalDuration = performance.now() - journey.startTime;

          this.sendToAnalytics('user_journey', journey.totalDuration,
            journey.totalDuration <= 5000 ? 'good' : 'poor', {
              routeName,
              interactionCount: journey.interactions.length,
              apiCallCount: journey.apiCalls.length,
              errorCount: journey.errors.length,
              avgApiCallTime: journey.apiCalls.reduce((sum, call) => sum + call.duration, 0) / journey.apiCalls.length || 0
            });
        }
      };
    },

    // Bundle size analysis
    analyzeBundleSize() {
      const scripts = Array.from(document.querySelectorAll('script[src]'));
      const styles = Array.from(document.querySelectorAll('link[rel="stylesheet"]'));

      const bundleAnalysis = {
        scriptCount: scripts.length,
        styleCount: styles.length,
        totalSize: 0,
        largeAssets: []
      };

      // Get resource sizes from performance API
      const resources = performance.getEntriesByType('resource');

      scripts.forEach(script => {
        const resource = resources.find(r => r.name.includes(script.src));
        if (resource && resource.transferSize > 100 * 1024) { // >100KB
          bundleAnalysis.largeAssets.push({
            type: 'script',
            name: script.src,
            size: resource.transferSize
          });
        }
        bundleAnalysis.totalSize += resource?.transferSize || 0;
      });

      styles.forEach(style => {
        const resource = resources.find(r => r.name.includes(style.href));
        if (resource && resource.transferSize > 50 * 1024) { // >50KB
          bundleAnalysis.largeAssets.push({
            type: 'stylesheet',
            name: style.href,
            size: resource.transferSize
          });
        }
        bundleAnalysis.totalSize += resource?.transferSize || 0;
      });

      this.sendToAnalytics('bundle_analysis', bundleAnalysis.totalSize,
        bundleAnalysis.totalSize <= 500 * 1024 ? 'good' : 'poor', bundleAnalysis);

      return bundleAnalysis;
    },

    // Record metric locally
    recordMetric(name, value, rating) {
      this.metrics[name] = { value, rating, timestamp: Date.now() };
    },

    // Send data to analytics endpoint
    sendToAnalytics(metricName, value, rating, additionalData = {}) {
      const data = {
        sessionId: this.sessionId,
        metric: {
          name: metricName,
          value: Math.round(value * 100) / 100, // Round to 2 decimals
          rating,
          timestamp: Date.now(),
          url: window.location.pathname,
          userAgent: navigator.userAgent,
          connectionType: navigator.connection?.effectiveType,
          ...additionalData
        }
      };

      // Send to Next.js API endpoint
      fetch('/api/performance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
        keepalive: true
      }).catch(error => {
        console.warn('Failed to send performance data:', error);
        // Fallback to localStorage for offline analysis
        const stored = localStorage.getItem('chimera_performance') || '[]';
        const metrics = JSON.parse(stored);
        metrics.push(data);
        localStorage.setItem('chimera_performance', JSON.stringify(metrics.slice(-100))); // Keep last 100
      });
    },

    // Initialize all monitoring
    init() {
      // Wait for page to stabilize
      if (document.readyState === 'complete') {
        this.initCoreWebVitals();
        setTimeout(() => {
          this.analyzeResourceTiming();
          this.analyzeBundleSize();
        }, 1000);
      } else {
        window.addEventListener('load', () => {
          setTimeout(() => {
            this.initCoreWebVitals();
            this.analyzeResourceTiming();
            this.analyzeBundleSize();
          }, 1000);
        });
      }

      // Track page visibility changes
      document.addEventListener('visibilitychange', () => {
        this.sendToAnalytics('visibility_change',
          document.hidden ? 1 : 0, 'info', {
            hidden: document.hidden,
            visibilityState: document.visibilityState
          });
      });

      // Track unload to catch incomplete sessions
      window.addEventListener('beforeunload', () => {
        // Send any pending metrics
        const pendingMetrics = Object.keys(this.metrics).length;
        if (pendingMetrics > 0) {
          navigator.sendBeacon('/api/performance', JSON.stringify({
            sessionId: this.sessionId,
            metric: {
              name: 'session_end',
              value: performance.now(),
              rating: 'info',
              pendingMetrics
            }
          }));
        }
      });
    },

    // Get current metrics snapshot
    getMetrics() {
      return { ...this.metrics };
    }
  };

  // Auto-initialize when script loads
  PerformanceMonitor.init();

  // Expose to global scope for manual usage
  window.ChimeraPerformanceMonitor = PerformanceMonitor;

  console.log('Chimera Performance Monitoring initialized');
})();