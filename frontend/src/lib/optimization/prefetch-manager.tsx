/**
 * Prefetch Manager for Project Chimera
 *
 * Intelligent prefetching system that:
 * - Predicts user navigation patterns
 * - Prefetches routes and data based on viewport visibility
 * - Manages prefetch queue with priority
 * - Respects network conditions and data saver mode
 * - Integrates with React Query for data prefetching
 *
 * @module lib/optimization/prefetch-manager
 */

import { useEffect, useCallback, useRef } from 'react';

// ============================================================================
// Types
// ============================================================================

export type PrefetchPriority = 'critical' | 'high' | 'medium' | 'low';

export interface PrefetchItem {
  /** Unique identifier */
  id: string;
  /** Type of prefetch */
  type: 'route' | 'data' | 'asset' | 'component';
  /** URL or key to prefetch */
  target: string;
  /** Priority level */
  priority: PrefetchPriority;
  /** Prefetch function */
  prefetch: () => Promise<void>;
  /** Whether this item has been prefetched */
  prefetched: boolean;
  /** Timestamp when added */
  addedAt: number;
  /** Timestamp when prefetched */
  prefetchedAt?: number;
}

export interface PrefetchConfig {
  /** Maximum concurrent prefetches */
  maxConcurrent?: number;
  /** Delay before starting prefetch (ms) */
  delay?: number;
  /** Respect data saver mode */
  respectDataSaver?: boolean;
  /** Minimum effective connection type */
  minConnectionType?: '4g' | '3g' | '2g' | 'slow-2g';
  /** Enable viewport-based prefetching */
  viewportPrefetch?: boolean;
  /** Viewport intersection threshold */
  intersectionThreshold?: number;
  /** Enable hover-based prefetching */
  hoverPrefetch?: boolean;
  /** Hover delay before prefetch (ms) */
  hoverDelay?: number;
}

export interface NetworkInfo {
  effectiveType: '4g' | '3g' | '2g' | 'slow-2g';
  saveData: boolean;
  downlink: number;
  rtt: number;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_CONFIG: Required<PrefetchConfig> = {
  maxConcurrent: 2,
  delay: 100,
  respectDataSaver: true,
  minConnectionType: '3g',
  viewportPrefetch: true,
  intersectionThreshold: 0.1,
  hoverPrefetch: true,
  hoverDelay: 100,
};

const CONNECTION_PRIORITY: Record<string, number> = {
  '4g': 4,
  '3g': 3,
  '2g': 2,
  'slow-2g': 1,
};

const PRIORITY_WEIGHTS: Record<PrefetchPriority, number> = {
  critical: 100,
  high: 75,
  medium: 50,
  low: 25,
};

// ============================================================================
// Prefetch Manager Class
// ============================================================================

export class PrefetchManager {
  private static instance: PrefetchManager;
  private queue: Map<string, PrefetchItem> = new Map();
  private activeCount = 0;
  private config: Required<PrefetchConfig>;
  private observers: Map<string, IntersectionObserver> = new Map();
  private hoverTimeouts: Map<string, ReturnType<typeof setTimeout>> = new Map();
  private isEnabled = true;

  private constructor(config: PrefetchConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.checkNetworkConditions();
  }

  static getInstance(config?: PrefetchConfig): PrefetchManager {
    if (!PrefetchManager.instance) {
      PrefetchManager.instance = new PrefetchManager(config);
    }
    return PrefetchManager.instance;
  }

  /**
   * Update configuration
   */
  configure(config: Partial<PrefetchConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Add item to prefetch queue
   */
  add(item: Omit<PrefetchItem, 'prefetched' | 'addedAt'>): void {
    if (!this.isEnabled) return;
    if (this.queue.has(item.id)) return;

    this.queue.set(item.id, {
      ...item,
      prefetched: false,
      addedAt: Date.now(),
    });

    this.processQueue();
  }

  /**
   * Remove item from queue
   */
  remove(id: string): void {
    this.queue.delete(id);
  }

  /**
   * Clear all pending prefetches
   */
  clear(): void {
    this.queue.clear();
    this.observers.forEach((observer) => observer.disconnect());
    this.observers.clear();
    this.hoverTimeouts.forEach((timeout) => clearTimeout(timeout));
    this.hoverTimeouts.clear();
  }

  /**
   * Enable/disable prefetching
   */
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
    if (enabled) {
      this.processQueue();
    }
  }

  /**
   * Get queue status
   */
  getStatus(): {
    queueSize: number;
    activeCount: number;
    prefetchedCount: number;
    isEnabled: boolean;
  } {
    const prefetchedCount = Array.from(this.queue.values()).filter(
      (item) => item.prefetched
    ).length;

    return {
      queueSize: this.queue.size,
      activeCount: this.activeCount,
      prefetchedCount,
      isEnabled: this.isEnabled,
    };
  }

  /**
   * Prefetch a route
   */
  prefetchRoute(path: string, priority: PrefetchPriority = 'medium'): void {
    this.add({
      id: `route:${path}`,
      type: 'route',
      target: path,
      priority,
      prefetch: async () => {
        // Next.js router prefetch
        if (typeof window !== 'undefined') {
          // Use manual link injection as fallback for App Router
          await this.prefetchNextRoute(path);
        }
      },
    });
  }

  /**
   * Prefetch data using a query function
   */
  prefetchData(
    key: string,
    queryFn: () => Promise<unknown>,
    priority: PrefetchPriority = 'medium'
  ): void {
    this.add({
      id: `data:${key}`,
      type: 'data',
      target: key,
      priority,
      prefetch: async () => {
        await queryFn();
      },
    });
  }

  /**
   * Prefetch an asset (image, script, etc.)
   */
  prefetchAsset(
    url: string,
    assetType: 'image' | 'script' | 'style' | 'font',
    priority: PrefetchPriority = 'low'
  ): void {
    this.add({
      id: `asset:${url}`,
      type: 'asset',
      target: url,
      priority,
      prefetch: async () => {
        await this.loadAsset(url, assetType);
      },
    });
  }

  /**
   * Prefetch a component
   */
  prefetchComponent(
    name: string,
    importFn: () => Promise<unknown>,
    priority: PrefetchPriority = 'medium'
  ): void {
    this.add({
      id: `component:${name}`,
      type: 'component',
      target: name,
      priority,
      prefetch: async () => {
        await importFn();
      },
    });
  }

  /**
   * Setup viewport-based prefetching for an element
   */
  observeElement(
    element: HTMLElement,
    onVisible: () => void,
    options: { threshold?: number; rootMargin?: string } = {}
  ): () => void {
    if (!this.config.viewportPrefetch) return () => { };

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            onVisible();
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: options.threshold ?? this.config.intersectionThreshold,
        rootMargin: options.rootMargin ?? '100px',
      }
    );

    observer.observe(element);
    const id = `observer:${Date.now()}`;
    this.observers.set(id, observer);

    return () => {
      observer.disconnect();
      this.observers.delete(id);
    };
  }

  /**
   * Setup hover-based prefetching for an element
   */
  observeHover(element: HTMLElement, onHover: () => void): () => void {
    if (!this.config.hoverPrefetch) return () => { };

    const handleMouseEnter = () => {
      const timeout = setTimeout(() => {
        onHover();
      }, this.config.hoverDelay);

      this.hoverTimeouts.set(element.id || 'hover', timeout);
    };

    const handleMouseLeave = () => {
      const timeout = this.hoverTimeouts.get(element.id || 'hover');
      if (timeout) {
        clearTimeout(timeout);
        this.hoverTimeouts.delete(element.id || 'hover');
      }
    };

    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mouseleave', handleMouseLeave);
      handleMouseLeave();
    };
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private async processQueue(): Promise<void> {
    if (!this.isEnabled) return;
    if (this.activeCount >= this.config.maxConcurrent) return;

    // Get pending items sorted by priority
    const pendingItems = Array.from(this.queue.values())
      .filter((item) => !item.prefetched)
      .sort((a, b) => {
        const priorityDiff = PRIORITY_WEIGHTS[b.priority] - PRIORITY_WEIGHTS[a.priority];
        if (priorityDiff !== 0) return priorityDiff;
        return a.addedAt - b.addedAt; // FIFO for same priority
      });

    if (pendingItems.length === 0) return;

    // Process items up to max concurrent
    const itemsToProcess = pendingItems.slice(
      0,
      this.config.maxConcurrent - this.activeCount
    );

    for (const item of itemsToProcess) {
      this.activeCount++;

      // Add delay for non-critical items
      if (item.priority !== 'critical') {
        await new Promise((resolve) => setTimeout(resolve, this.config.delay));
      }

      try {
        await item.prefetch();
        item.prefetched = true;
        item.prefetchedAt = Date.now();
      } catch (error) {
        console.warn(`[Prefetch] Failed to prefetch ${item.id}:`, error);
      } finally {
        this.activeCount--;
        this.processQueue(); // Process next items
      }
    }
  }

  private checkNetworkConditions(): void {
    if (typeof navigator === 'undefined') return;

    const connection = (navigator as Navigator & { connection?: NetworkInfo }).connection;
    if (!connection) return;

    // Check data saver mode
    if (this.config.respectDataSaver && connection.saveData) {
      this.isEnabled = false;
      return;
    }

    // Check connection type
    const currentPriority = CONNECTION_PRIORITY[connection.effectiveType] || 0;
    const minPriority = CONNECTION_PRIORITY[this.config.minConnectionType] || 0;

    if (currentPriority < minPriority) {
      this.isEnabled = false;
    }

    // Listen for connection changes
    (connection as any).addEventListener?.('change', () => {
      this.checkNetworkConditions();
    });
  }

  private async prefetchNextRoute(path: string): Promise<void> {
    // Create a prefetch link
    if (typeof document === 'undefined') return;

    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = path;
    link.as = 'document';
    document.head.appendChild(link);

    // Also prefetch the JS chunks
    const jsLink = document.createElement('link');
    jsLink.rel = 'prefetch';
    jsLink.href = `/_next/static/chunks/pages${path === '/' ? '/index' : path}.js`;
    jsLink.as = 'script';
    document.head.appendChild(jsLink);
  }

  private async loadAsset(
    url: string,
    type: 'image' | 'script' | 'style' | 'font'
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      if (typeof document === 'undefined') {
        resolve();
        return;
      }

      switch (type) {
        case 'image': {
          const img = new Image();
          img.onload = () => resolve();
          img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
          img.src = url;
          break;
        }

        case 'script': {
          const link = document.createElement('link');
          link.rel = 'prefetch';
          link.as = 'script';
          link.href = url;
          link.onload = () => resolve();
          link.onerror = () => reject(new Error(`Failed to prefetch script: ${url}`));
          document.head.appendChild(link);
          break;
        }

        case 'style': {
          const link = document.createElement('link');
          link.rel = 'prefetch';
          link.as = 'style';
          link.href = url;
          link.onload = () => resolve();
          link.onerror = () => reject(new Error(`Failed to prefetch style: ${url}`));
          document.head.appendChild(link);
          break;
        }

        case 'font': {
          const link = document.createElement('link');
          link.rel = 'prefetch';
          link.as = 'font';
          link.href = url;
          link.crossOrigin = 'anonymous';
          link.onload = () => resolve();
          link.onerror = () => reject(new Error(`Failed to prefetch font: ${url}`));
          document.head.appendChild(link);
          break;
        }

        default:
          resolve();
      }
    });
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const prefetchManager = PrefetchManager.getInstance();

// ============================================================================
// React Hooks
// ============================================================================

/**
 * Hook for prefetching routes on viewport visibility
 */
export function usePrefetchOnVisible(
  path: string,
  options: {
    priority?: PrefetchPriority;
    threshold?: number;
    rootMargin?: string;
  } = {}
): React.RefObject<HTMLElement | null> {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    return prefetchManager.observeElement(
      element,
      () => prefetchManager.prefetchRoute(path, options.priority),
      { threshold: options.threshold, rootMargin: options.rootMargin }
    );
  }, [path, options.priority, options.threshold, options.rootMargin]);

  return ref;
}

/**
 * Hook for prefetching routes on hover
 */
export function usePrefetchOnHover(
  path: string,
  priority: PrefetchPriority = 'high'
): {
  onMouseEnter: () => void;
  onMouseLeave: () => void;
} {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const onMouseEnter = useCallback(() => {
    timeoutRef.current = setTimeout(() => {
      prefetchManager.prefetchRoute(path, priority);
    }, 100);
  }, [path, priority]);

  const onMouseLeave = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  }, []);

  return { onMouseEnter, onMouseLeave };
}

/**
 * Hook for prefetching data
 */
export function usePrefetchData(
  key: string,
  queryFn: () => Promise<unknown>,
  options: {
    priority?: PrefetchPriority;
    enabled?: boolean;
  } = {}
): void {
  const { priority = 'medium', enabled = true } = options;

  useEffect(() => {
    if (!enabled) return;
    prefetchManager.prefetchData(key, queryFn, priority);
  }, [key, queryFn, priority, enabled]);
}

/**
 * Hook for prefetching components
 */
export function usePrefetchComponent(
  name: string,
  importFn: () => Promise<unknown>,
  options: {
    priority?: PrefetchPriority;
    enabled?: boolean;
  } = {}
): void {
  const { priority = 'medium', enabled = true } = options;

  useEffect(() => {
    if (!enabled) return;
    prefetchManager.prefetchComponent(name, importFn, priority);
  }, [name, importFn, priority, enabled]);
}

/**
 * Hook for managing prefetch queue
 */
export function usePrefetchManager(): {
  status: ReturnType<PrefetchManager['getStatus']>;
  prefetchRoute: (path: string, priority?: PrefetchPriority) => void;
  prefetchData: (key: string, queryFn: () => Promise<unknown>, priority?: PrefetchPriority) => void;
  prefetchAsset: (url: string, type: 'image' | 'script' | 'style' | 'font', priority?: PrefetchPriority) => void;
  clear: () => void;
  setEnabled: (enabled: boolean) => void;
} {
  const getStatus = useCallback(() => prefetchManager.getStatus(), []);

  return {
    status: getStatus(),
    prefetchRoute: useCallback(
      (path: string, priority?: PrefetchPriority) =>
        prefetchManager.prefetchRoute(path, priority),
      []
    ),
    prefetchData: useCallback(
      (key: string, queryFn: () => Promise<unknown>, priority?: PrefetchPriority) =>
        prefetchManager.prefetchData(key, queryFn, priority),
      []
    ),
    prefetchAsset: useCallback(
      (url: string, type: 'image' | 'script' | 'style' | 'font', priority?: PrefetchPriority) =>
        prefetchManager.prefetchAsset(url, type, priority),
      []
    ),
    clear: useCallback(() => prefetchManager.clear(), []),
    setEnabled: useCallback((enabled: boolean) => prefetchManager.setEnabled(enabled), []),
  };
}

// ============================================================================
// Link Component with Prefetch
// ============================================================================

import React, { forwardRef, AnchorHTMLAttributes } from 'react';
import Link from 'next/link';

// LinkProps not imported explicitly or used in interface def?
export interface PrefetchLinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> {
  children: React.ReactNode;
  href: string;
  prefetchPriority?: PrefetchPriority;
  prefetchOnHover?: boolean;
  prefetchOnVisible?: boolean;
}

/**
 * Enhanced Link component with intelligent prefetching
 */
export const PrefetchLink = forwardRef<HTMLAnchorElement, PrefetchLinkProps>(
  function PrefetchLink(
    {
      children,
      className,
      prefetchPriority = 'medium',
      prefetchOnHover = true,
      prefetchOnVisible = true,
      href,
      ...props
    },
    ref
  ) {
    const internalRef = useRef<HTMLAnchorElement>(null);

    // Sync external ref with internal ref
    useEffect(() => {
      if (!ref) return;
      if (typeof ref === 'function') {
        ref(internalRef.current);
      } else {
        (ref as React.MutableRefObject<HTMLAnchorElement | null>).current = internalRef.current;
      }
    }, [ref]);

    const path = href;

    // Viewport-based prefetching
    useEffect(() => {
      if (!prefetchOnVisible) return;
      const element = internalRef.current;
      if (!element) return;

      return prefetchManager.observeElement(element, () =>
        prefetchManager.prefetchRoute(path, prefetchPriority)
      );
    }, [path, prefetchPriority, prefetchOnVisible]);

    // Hover-based prefetching
    const hoverHandlers = usePrefetchOnHover(path, 'high');

    return (
      <Link
        ref={internalRef}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        href={href as any}
        className={className}
        prefetch={false} // Disable Next.js default prefetch, use our own
        {...(prefetchOnHover ? hoverHandlers : {})}
        {...props}
      >
        {children}
      </Link>
    );
  }
);

// ============================================================================
// Predictive Prefetching
// ============================================================================

export interface NavigationPattern {
  from: string;
  to: string;
  count: number;
  lastVisited: number;
}

/**
 * Predictive prefetch manager that learns navigation patterns
 */
export class PredictivePrefetchManager {
  private patterns: Map<string, NavigationPattern[]> = new Map();
  private currentPath: string = '';
  private storageKey = 'chimera:navigation-patterns';

  constructor() {
    this.loadPatterns();
  }

  /**
   * Record a navigation event
   */
  recordNavigation(from: string, to: string): void {
    const patterns = this.patterns.get(from) || [];
    const existingPattern = patterns.find((p) => p.to === to);

    if (existingPattern) {
      existingPattern.count++;
      existingPattern.lastVisited = Date.now();
    } else {
      patterns.push({
        from,
        to,
        count: 1,
        lastVisited: Date.now(),
      });
    }

    this.patterns.set(from, patterns);
    this.savePatterns();
  }

  /**
   * Get predicted next routes based on current path
   */
  getPredictedRoutes(currentPath: string, limit: number = 3): string[] {
    const patterns = this.patterns.get(currentPath) || [];

    return patterns
      .sort((a, b) => {
        // Score based on frequency and recency
        const scoreA = a.count * (1 / (Date.now() - a.lastVisited + 1));
        const scoreB = b.count * (1 / (Date.now() - b.lastVisited + 1));
        return scoreB - scoreA;
      })
      .slice(0, limit)
      .map((p) => p.to);
  }

  /**
   * Prefetch predicted routes
   */
  prefetchPredicted(currentPath: string): void {
    const predicted = this.getPredictedRoutes(currentPath);

    predicted.forEach((route, index) => {
      const priority: PrefetchPriority = index === 0 ? 'high' : index === 1 ? 'medium' : 'low';
      prefetchManager.prefetchRoute(route, priority);
    });
  }

  /**
   * Update current path and trigger predictive prefetch
   */
  setCurrentPath(path: string): void {
    if (this.currentPath && this.currentPath !== path) {
      this.recordNavigation(this.currentPath, path);
    }
    this.currentPath = path;
    this.prefetchPredicted(path);
  }

  private loadPatterns(): void {
    if (typeof localStorage === 'undefined') return;

    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const data = JSON.parse(stored) as [string, NavigationPattern[]][];
        this.patterns = new Map(data);
      }
    } catch {
      // Ignore errors
    }
  }

  private savePatterns(): void {
    if (typeof localStorage === 'undefined') return;

    try {
      const data = Array.from(this.patterns.entries());
      localStorage.setItem(this.storageKey, JSON.stringify(data));
    } catch {
      // Ignore errors
    }
  }
}

export const predictivePrefetch = new PredictivePrefetchManager();

/**
 * Hook for predictive prefetching
 */
export function usePredictivePrefetch(): void {
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleRouteChange = () => {
      predictivePrefetch.setCurrentPath(window.location.pathname);
    };

    // Initial path
    handleRouteChange();

    // Listen for route changes
    window.addEventListener('popstate', handleRouteChange);

    return () => {
      window.removeEventListener('popstate', handleRouteChange);
    };
  }, []);
}
