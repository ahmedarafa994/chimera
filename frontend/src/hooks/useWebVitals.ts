'use client';

import { useEffect } from 'react';

export function useWebVitals() {
  useEffect(() => {
    // Only run in production or when explicitly enabled
    if (process.env.NODE_ENV !== 'production' && !process.env.NEXT_PUBLIC_ENABLE_VITALS) {
      return;
    }

    const loadWebVitals = async () => {
      try {
        const { onCLS, onINP, onFCP, onLCP, onTTFB } = await import('web-vitals');

        function sendToAnalytics(metric: { name: string; value: number; id: string }) {
          // Log to console in development
          if (process.env.NODE_ENV === 'development') {
            console.debug(`[Web Vitals] ${metric.name}:`, metric.value);
            return;
          }

          // Send to analytics endpoint in production
          const endpoint = process.env.NEXT_PUBLIC_ANALYTICS_ENDPOINT || '/api/analytics/web-vitals';

          fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(metric),
          }).catch(() => {
            // Silently fail - analytics should not break the app
          });
        }

        onCLS(sendToAnalytics);
        onINP(sendToAnalytics); // INP replaced FID in web-vitals v4
        onFCP(sendToAnalytics);
        onLCP(sendToAnalytics);
        onTTFB(sendToAnalytics);
      } catch {
        // Silently fail - web vitals are optional
      }
    };

    loadWebVitals();
  }, []);
}