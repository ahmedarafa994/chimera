/**
 * Lazy Loading Utilities for Project Chimera
 * 
 * Provides intelligent lazy loading with:
 * - Component-level code splitting
 * - Intersection Observer-based loading
 * - Preloading strategies
 * - Loading state management
 * 
 * @module lib/optimization/lazy-loader
 */

import React, {
  ComponentType,
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
  ReactNode,
  createElement
} from 'react';

// ============================================================================
// Types
// ============================================================================

export interface LazyLoadOptions {
  /** Root margin for intersection observer */
  rootMargin?: string;
  /** Threshold for intersection observer */
  threshold?: number | number[];
  /** Delay before loading (ms) */
  delay?: number;
  /** Preload on hover */
  preloadOnHover?: boolean;
  /** Preload on focus */
  preloadOnFocus?: boolean;
  /** Retry attempts on failure */
  retryAttempts?: number;
  /** Retry delay (ms) */
  retryDelay?: number;
}

export interface LazyComponentOptions<P = Record<string, unknown>> {
  /** Loading fallback component */
  fallback?: ReactNode;
  /** Error fallback component */
  errorFallback?: ComponentType<{ error: Error; retry: () => void }>;
  /** Preload function */
  preload?: () => Promise<void>;
  /** Component display name */
  displayName?: string;
  /** SSR support */
  ssr?: boolean;
}

export interface LazyLoadState {
  isLoading: boolean;
  isLoaded: boolean;
  isError: boolean;
  error: Error | null;
}

export interface LazyLoadConfig {
  /** Root margin for intersection observer */
  rootMargin?: string;
  /** Threshold for intersection observer */
  threshold?: number | number[];
  /** Delay before loading (ms) */
  delay?: number;
  /** Preload on hover */
  preloadOnHover?: boolean;
  /** Preload on focus */
  preloadOnFocus?: boolean;
  /** Retry attempts on failure */
  retryAttempts?: number;
  /** Retry delay (ms) */
  retryDelay?: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: Required<LazyLoadConfig> = {
  rootMargin: '100px',
  threshold: 0.1,
  delay: 0,
  preloadOnHover: true,
  preloadOnFocus: true,
  retryAttempts: 3,
  retryDelay: 1000,
};

// ============================================================================
// Lazy Component Factory
// ============================================================================

/**
 * Create a lazy-loaded component with enhanced features
 */
export function createLazyComponent<P extends Record<string, unknown>>(
  importFn: () => Promise<{ default: ComponentType<P> }>,
  options: LazyComponentOptions<P> = {}
): ComponentType<P> {
  const {
    fallback = null,
    errorFallback: ErrorFallback,
    displayName,
    ssr = false,
  } = options;

  // Create the lazy component
  const LazyComponent = lazy(importFn);

  // Wrapper component with error boundary and suspense
  const WrappedComponent: React.FC<P> = (props) => {
    const [error, setError] = useState<Error | null>(null);
    const [retryCount, setRetryCount] = useState(0);

    const retry = () => {
      setError(null);
      setRetryCount((c) => c + 1);
    };

    if (error && ErrorFallback) {
      return createElement(ErrorFallback, { error, retry });
    }

    return createElement(
      Suspense,
      { fallback },
      createElement(LazyComponent, { ...props, key: retryCount } as P & { key: number })
    );
  };

  WrappedComponent.displayName = displayName || 'LazyComponent';

  return WrappedComponent as ComponentType<P>;
}

/**
 * Create a lazy component with preloading capability
 */
export function createPreloadableLazyComponent<P extends Record<string, unknown>>(
  importFn: () => Promise<{ default: ComponentType<P> }>,
  options: LazyComponentOptions<P> = {}
): ComponentType<P> & { preload: () => Promise<void> } {
  let preloadPromise: Promise<{ default: ComponentType<P> }> | null = null;

  const preload = async (): Promise<void> => {
    if (!preloadPromise) {
      preloadPromise = importFn();
    }
    await preloadPromise;
  };

  const LazyComponent = createLazyComponent(
    () => preloadPromise || importFn(),
    options
  );

  return Object.assign(LazyComponent, { preload });
}

// ============================================================================
// Intersection Observer Hook
// ============================================================================

/**
 * Hook for lazy loading based on viewport intersection
 */
export function useLazyLoad(
  config: LazyLoadConfig = {}
): {
  ref: React.RefObject<HTMLDivElement | null>;
  isInView: boolean;
  isLoaded: boolean;
} {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config };
  const ref = useRef<HTMLDivElement | null>(null);
  const [isInView, setIsInView] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element || isLoaded) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            if (mergedConfig.delay > 0) {
              setTimeout(() => {
                setIsInView(true);
                setIsLoaded(true);
              }, mergedConfig.delay);
            } else {
              setIsInView(true);
              setIsLoaded(true);
            }
            observer.unobserve(element);
          }
        });
      },
      {
        rootMargin: mergedConfig.rootMargin,
        threshold: mergedConfig.threshold,
      }
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [isLoaded, mergedConfig.delay, mergedConfig.rootMargin, mergedConfig.threshold]);

  return { ref, isInView, isLoaded };
}

// ============================================================================
// Lazy Image Component
// ============================================================================

export interface LazyImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  /** Placeholder image or color */
  placeholder?: string;
  /** Blur placeholder data URL */
  blurDataURL?: string;
  /** Loading strategy */
  loadingStrategy?: 'lazy' | 'eager' | 'intersection';
  /** Intersection observer config */
  intersectionConfig?: LazyLoadConfig;
  /** On load callback */
  onImageLoad?: () => void;
  /** On error callback */
  onImageError?: (error: Error) => void;
}

/**
 * Lazy loading image component with blur-up effect
 */
export function LazyImage({
  src,
  alt,
  placeholder,
  blurDataURL,
  loadingStrategy = 'intersection',
  intersectionConfig,
  onImageLoad,
  onImageError,
  className,
  style,
  ...props
}: LazyImageProps): React.ReactElement {
  const { ref, isLoaded: isInView } = useLazyLoad(intersectionConfig);
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  const shouldLoad = loadingStrategy === 'eager' ||
    (loadingStrategy === 'lazy') ||
    (loadingStrategy === 'intersection' && isInView);

  const handleLoad = () => {
    setIsLoaded(true);
    onImageLoad?.();
  };

  const handleError = () => {
    setHasError(true);
    onImageError?.(new Error(`Failed to load image: ${src}`));
  };

  return createElement(
    'div',
    {
      ref,
      className: `relative overflow-hidden ${className || ''}`,
      style: { ...style },
    },
    [
      // Placeholder/blur background
      (blurDataURL || placeholder) && !isLoaded && createElement('div', {
        key: 'placeholder',
        className: 'absolute inset-0 transition-opacity duration-300',
        style: {
          backgroundImage: blurDataURL ? `url(${blurDataURL})` : undefined,
          backgroundColor: placeholder && !blurDataURL ? placeholder : undefined,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          filter: 'blur(20px)',
          transform: 'scale(1.1)',
          opacity: isLoaded ? 0 : 1,
        },
      }),
      // Actual image
      shouldLoad && createElement('img', {
        key: 'image',
        src,
        alt,
        loading: loadingStrategy === 'lazy' ? 'lazy' : undefined,
        onLoad: handleLoad,
        onError: handleError,
        className: `transition-opacity duration-500 ${isLoaded ? 'opacity-100' : 'opacity-0'}`,
        style: { width: '100%', height: '100%', objectFit: 'cover' },
        ...props,
      }),
      // Error state
      hasError && createElement('div', {
        key: 'error',
        className: 'absolute inset-0 flex items-center justify-center bg-muted',
      }, createElement('span', { className: 'text-muted-foreground text-sm' }, 'Failed to load image')),
    ]
  );
}

// ============================================================================
// Lazy Section Component
// ============================================================================

export interface LazySectionProps {
  children: ReactNode;
  /** Fallback while loading */
  fallback?: ReactNode;
  /** Intersection observer config */
  config?: LazyLoadConfig;
  /** Minimum height to prevent layout shift */
  minHeight?: string | number;
  /** CSS class name */
  className?: string;
  /** On visible callback */
  onVisible?: () => void;
}

/**
 * Lazy loading section that renders children when in viewport
 */
export function LazySection({
  children,
  fallback,
  config,
  minHeight = 200,
  className,
  onVisible,
}: LazySectionProps): React.ReactElement {
  const { ref, isLoaded } = useLazyLoad(config);

  useEffect(() => {
    if (isLoaded) {
      onVisible?.();
    }
  }, [isLoaded, onVisible]);

  return createElement(
    'div',
    {
      ref,
      className,
      style: { minHeight: isLoaded ? undefined : minHeight },
    },
    isLoaded ? children : fallback
  );
}

// ============================================================================
// Preload Manager
// ============================================================================

type PreloadFn = () => Promise<unknown>;

export class PreloadManager {
  private preloadQueue: Map<string, PreloadFn> = new Map();
  private preloaded: Set<string> = new Set();
  private preloading: Set<string> = new Set();

  /**
   * Register a component for preloading
   */
  register(key: string, preloadFn: PreloadFn): void {
    if (!this.preloaded.has(key) && !this.preloadQueue.has(key)) {
      this.preloadQueue.set(key, preloadFn);
    }
  }

  /**
   * Preload a specific component
   */
  async preload(key: string): Promise<void> {
    if (this.preloaded.has(key) || this.preloading.has(key)) {
      return;
    }

    const preloadFn = this.preloadQueue.get(key);
    if (!preloadFn) return;

    this.preloading.add(key);

    try {
      await preloadFn();
      this.preloaded.add(key);
      this.preloadQueue.delete(key);
    } finally {
      this.preloading.delete(key);
    }
  }

  /**
   * Preload all registered components
   */
  async preloadAll(): Promise<void> {
    const keys = Array.from(this.preloadQueue.keys());
    await Promise.all(keys.map((key) => this.preload(key)));
  }

  /**
   * Preload components during idle time
   */
  preloadOnIdle(): void {
    if (typeof requestIdleCallback !== 'undefined') {
      requestIdleCallback(() => {
        this.preloadAll();
      });
    } else {
      setTimeout(() => {
        this.preloadAll();
      }, 1);
    }
  }

  /**
   * Check if a component is preloaded
   */
  isPreloaded(key: string): boolean {
    return this.preloaded.has(key);
  }

  /**
   * Get preload status
   */
  getStatus(): {
    queued: number;
    preloaded: number;
    preloading: number;
  } {
    return {
      queued: this.preloadQueue.size,
      preloaded: this.preloaded.size,
      preloading: this.preloading.size,
    };
  }
}

export const preloadManager = new PreloadManager();

// ============================================================================
// Route Preloading Hook
// ============================================================================

/**
 * Hook for preloading routes on hover/focus
 */
export function useRoutePreload(
  routes: Record<string, () => Promise<unknown>>
): {
  preloadRoute: (route: string) => void;
  getPreloadHandlers: (route: string) => {
    onMouseEnter: () => void;
    onFocus: () => void;
  };
} {
  const preloadedRoutes = useRef<Set<string>>(new Set());

  const preloadRoute = (route: string) => {
    if (preloadedRoutes.current.has(route)) return;

    const preloadFn = routes[route];
    if (preloadFn) {
      preloadedRoutes.current.add(route);
      preloadFn().catch(console.error);
    }
  };

  const getPreloadHandlers = (route: string) => ({
    onMouseEnter: () => preloadRoute(route),
    onFocus: () => preloadRoute(route),
  });

  return { preloadRoute, getPreloadHandlers };
}

// ============================================================================
// Dynamic Import with Retry
// ============================================================================

/**
 * Dynamic import with automatic retry on failure
 */
export async function dynamicImportWithRetry<T>(
  importFn: () => Promise<T>,
  options: {
    retries?: number;
    delay?: number;
    onRetry?: (attempt: number, error: Error) => void;
  } = {}
): Promise<T> {
  const { retries = 3, delay = 1000, onRetry } = options;

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await importFn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < retries) {
        onRetry?.(attempt + 1, lastError);
        await new Promise((resolve) => setTimeout(resolve, delay * (attempt + 1)));
      }
    }
  }

  throw lastError;
}

// ============================================================================
// Chunk Loading Utilities
// ============================================================================

/**
 * Preload a webpack chunk by name
 */
export function preloadChunk(chunkName: string): void {
  if (typeof window !== 'undefined') {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'script';
    link.href = `/_next/static/chunks/${chunkName}.js`;
    document.head.appendChild(link);
  }
}

/**
 * Prefetch a webpack chunk by name
 */
export function prefetchChunk(chunkName: string): void {
  if (typeof window !== 'undefined') {
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = `/_next/static/chunks/${chunkName}.js`;
    document.head.appendChild(link);
  }
}

// ============================================================================
// Loading State Component
// ============================================================================

export interface LoadingSkeletonProps {
  /** Width of skeleton */
  width?: string | number;
  /** Height of skeleton */
  height?: string | number;
  /** Border radius */
  borderRadius?: string | number;
  /** Animation type */
  animation?: 'pulse' | 'shimmer' | 'none';
  /** CSS class name */
  className?: string;
}

/**
 * Loading skeleton component for lazy-loaded content
 */
export function LoadingSkeleton({
  width = '100%',
  height = 20,
  borderRadius = 4,
  animation = 'shimmer',
  className,
}: LoadingSkeletonProps): React.ReactElement {
  const animationClass = animation === 'shimmer'
    ? 'shimmer'
    : animation === 'pulse'
      ? 'animate-pulse'
      : '';

  return createElement('div', {
    className: `bg-muted ${animationClass} ${className || ''}`,
    style: {
      width,
      height,
      borderRadius,
    },
  });
}