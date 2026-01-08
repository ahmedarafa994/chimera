'use client';

import { useEffect, useCallback } from 'react';

// Resource preloading utilities
export const preloadResource = (href: string, as: string, crossorigin?: string) => {
  const link = document.createElement('link');
  link.rel = 'preload';
  link.href = href;
  link.as = as;
  if (crossorigin) link.crossOrigin = crossorigin;
  document.head.appendChild(link);
};

export const prefetchResource = (href: string) => {
  const link = document.createElement('link');
  link.rel = 'prefetch';
  link.href = href;
  document.head.appendChild(link);
};

export const preconnectOrigin = (origin: string, crossorigin?: boolean) => {
  const link = document.createElement('link');
  link.rel = 'preconnect';
  link.href = origin;
  if (crossorigin) link.crossOrigin = 'anonymous';
  document.head.appendChild(link);
};

// Critical CSS inlining (for server components)
export const inlineCriticalCSS = (css: string) => {
  const style = document.createElement('style');
  style.innerHTML = css;
  document.head.appendChild(style);
};

// Font optimization
export const optimizeFont = (fontFamily: string, weights: number[] = [400, 700]) => {
  weights.forEach(weight => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = `https://fonts.googleapis.com/css2?family=${fontFamily}:wght@${weight}&display=swap`;
    link.as = 'style';
    link.crossOrigin = 'anonymous';
    document.head.appendChild(link);
  });
};

// Resource hint management hook
interface ResourceHints {
  preload: Array<{ href: string; as: string; crossorigin?: string }>;
  prefetch: Array<string>;
  preconnect: Array<{ origin: string; crossorigin?: boolean }>;
  fonts: Array<{ family: string; weights: number[] }>;
}

export function useResourceHints(hints: ResourceHints) {
  useEffect(() => {
    // Apply preload hints
    hints.preload.forEach(({ href, as, crossorigin }) => {
      preloadResource(href, as, crossorigin);
    });

    // Apply prefetch hints
    hints.prefetch.forEach(href => {
      prefetchResource(href);
    });

    // Apply preconnect hints
    hints.preconnect.forEach(({ origin, crossorigin }) => {
      preconnectOrigin(origin, crossorigin);
    });

    // Optimize fonts
    hints.fonts.forEach(({ family, weights }) => {
      optimizeFont(family, weights);
    });
  }, [hints]);
}

// Critical path optimization hook
interface CriticalPathOptions {
  deferNonCriticalCSS?: boolean;
  optimizeImages?: boolean;
  preloadCriticalAssets?: boolean;
}

export function useCriticalPath(options: CriticalPathOptions = {}) {
  const {
    deferNonCriticalCSS = true,
    optimizeImages = true,
    // Disabled by default - only enable if assets exist
    preloadCriticalAssets = false
  } = options;

  const deferCSS = useCallback((href: string) => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'style';
    link.href = href;
    link.onload = () => {
      // Change from preload to stylesheet after load
      link.onload = null;
      link.rel = 'stylesheet';
    };
    document.head.appendChild(link);

    // Fallback for browsers that don't support preload
    const noscript = document.createElement('noscript');
    const fallbackLink = document.createElement('link');
    fallbackLink.rel = 'stylesheet';
    fallbackLink.href = href;
    noscript.appendChild(fallbackLink);
    document.head.appendChild(noscript);
  }, []);

  const optimizeImageLoading = useCallback(() => {
    // Add loading="lazy" to images below the fold
    const images = document.querySelectorAll('img');
    images.forEach((img, index) => {
      if (index > 2) { // Assume first 3 images are above the fold
        img.setAttribute('loading', 'lazy');
      }
    });
  }, []);

  const preloadCritical = useCallback(() => {
    // Preconnect to external font domains (always safe)
    preconnectOrigin('https://fonts.googleapis.com');
    preconnectOrigin('https://fonts.gstatic.com', true);

    // NOTE: Only preload assets that actually exist in /public
    // Available assets: favicon.ico, next.svg, vercel.svg, globe.svg, file.svg, window.svg
    // Do NOT preload: logo.svg, logo.png, hero-bg.webp (these don't exist)
  }, []);

  useEffect(() => {
    if (preloadCriticalAssets) {
      preloadCritical();
    }

    if (optimizeImages) {
      optimizeImageLoading();
    }

    // Defer non-critical CSS loading
    if (deferNonCriticalCSS) {
      // This would typically be handled at build time
      // but can be done dynamically for certain scenarios
    }
  }, [deferNonCriticalCSS, optimizeImages, preloadCriticalAssets, deferCSS, optimizeImageLoading, preloadCritical]);

  return {
    deferCSS,
    optimizeImageLoading,
    preloadCritical,
  };
}

// Performance budget monitoring
interface PerformanceBudget {
  totalSize: number; // KB
  jsSize: number; // KB
  cssSize: number; // KB
  imageSize: number; // KB
  firstContentfulPaint: number; // ms
  largestContentfulPaint: number; // ms
}

export function usePerformanceBudget(budget: PerformanceBudget) {
  useEffect(() => {
    // Skip performance monitoring in development to reduce console noise
    if (process.env.NODE_ENV !== 'production') {
      return;
    }

    // Monitor performance budget
    if ('performance' in window) {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'navigation') {
            const navEntry = entry as PerformanceNavigationTiming;
            const loadTime = navEntry.loadEventEnd - navEntry.fetchStart;

            // Only log if budget exceeded
            if (loadTime > budget.largestContentfulPaint) {
              console.warn(`LCP budget exceeded: ${loadTime}ms > ${budget.largestContentfulPaint}ms`);
            }
          }

          if (entry.entryType === 'paint') {
            if (entry.name === 'first-contentful-paint') {
              // Only log if budget exceeded
              if (entry.startTime > budget.firstContentfulPaint) {
                console.warn(`FCP budget exceeded: ${entry.startTime}ms > ${budget.firstContentfulPaint}ms`);
              }
            }
          }
        }
      });

      observer.observe({ entryTypes: ['navigation', 'paint'] });

      return () => observer.disconnect();
    }
  }, [budget]);
}

// Intersection Observer for lazy loading
export function useIntersectionObserver(
  callback: IntersectionObserverCallback,
  options?: IntersectionObserverInit
) {
  useEffect(() => {
    const observer = new IntersectionObserver(callback, {
      rootMargin: '50px',
      threshold: 0.1,
      ...options,
    });

    return () => observer.disconnect();
  }, [callback, options]);
}

// Service Worker registration for caching
export function useServiceWorker(swPath = '/sw.js') {
  useEffect(() => {
    if ('serviceWorker' in navigator && process.env.NODE_ENV === 'production') {
      navigator.serviceWorker.register(swPath)
        .then((registration) => {
          console.log('SW registered: ', registration);
        })
        .catch((registrationError) => {
          console.log('SW registration failed: ', registrationError);
        });
    }
  }, [swPath]);
}
