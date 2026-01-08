/**
 * Code Splitting Strategies for Project Chimera
 * 
 * Provides intelligent code splitting with:
 * - Route-based splitting
 * - Component-based splitting
 * - Vendor chunk optimization
 * - Dynamic import utilities
 * 
 * @module lib/optimization/code-splitting
 */

import { ComponentType, lazy } from 'react';

// ============================================================================
// Types
// ============================================================================

export interface ChunkConfig {
  /** Chunk name for webpack magic comments */
  chunkName: string;
  /** Preload priority */
  priority?: 'high' | 'low' | 'auto';
  /** Prefetch hint */
  prefetch?: boolean;
  /** Preload hint */
  preload?: boolean;
}

export interface SplitPointConfig {
  /** Minimum size for splitting (bytes) */
  minSize?: number;
  /** Maximum size for splitting (bytes) */
  maxSize?: number;
  /** Maximum async requests */
  maxAsyncRequests?: number;
  /** Maximum initial requests */
  maxInitialRequests?: number;
}

export interface RouteChunk {
  path: string;
  component: () => Promise<{ default: ComponentType<unknown> }>;
  chunkName: string;
  preload?: boolean;
}

// ============================================================================
// Route-Based Code Splitting
// ============================================================================

/**
 * Create a route-based lazy component with chunk naming
 */
export function createRouteComponent<P extends Record<string, unknown>>(
  importFn: () => Promise<{ default: ComponentType<P> }>,
  config: ChunkConfig
): ComponentType<P> {
  // Add webpack magic comments for chunk naming
  const enhancedImport = () => {
    // The actual magic comment would be in the import statement
    // This is a runtime wrapper
    return importFn();
  };

  const LazyComponent = lazy(enhancedImport);

  // Add preload/prefetch hints
  if (typeof window !== 'undefined') {
    if (config.preload) {
      addResourceHint(config.chunkName, 'preload');
    } else if (config.prefetch) {
      addResourceHint(config.chunkName, 'prefetch');
    }
  }

  return LazyComponent;
}

/**
 * Create multiple route components with automatic chunk naming
 */
export function createRouteComponents<T extends Record<string, () => Promise<{ default: ComponentType<Record<string, unknown>> }>>>(
  routes: T,
  options: { prefetchAll?: boolean; preloadCritical?: string[] } = {}
): { [K in keyof T]: ComponentType<Record<string, unknown>> } {
  const result = {} as { [K in keyof T]: ComponentType<Record<string, unknown>> };

  for (const [key, importFn] of Object.entries(routes)) {
    const isCritical = options.preloadCritical?.includes(key);

    result[key as keyof T] = createRouteComponent<Record<string, unknown>>(importFn, {
      chunkName: `route-${key}`,
      preload: isCritical,
      prefetch: options.prefetchAll && !isCritical,
    });
  }

  return result;
}

// ============================================================================
// Component-Based Code Splitting
// ============================================================================

/**
 * Split a heavy component into its own chunk
 */
export function splitComponent<P extends Record<string, unknown>>(
  importFn: () => Promise<{ default: ComponentType<P> }>,
  chunkName: string
): ComponentType<P> {
  return lazy(() => importFn());
}

/**
 * Create a component that loads different implementations based on conditions
 */
export function createConditionalComponent<P extends Record<string, unknown>>(
  condition: () => boolean,
  lightImport: () => Promise<{ default: ComponentType<P> }>,
  heavyImport: () => Promise<{ default: ComponentType<P> }>
): ComponentType<P> {
  return lazy(() => {
    if (condition()) {
      return heavyImport();
    }
    return lightImport();
  });
}

/**
 * Create a component that loads based on feature flags
 */
export function createFeatureFlaggedComponent<P extends Record<string, unknown>>(
  featureFlag: string,
  enabledImport: () => Promise<{ default: ComponentType<P> }>,
  disabledImport: () => Promise<{ default: ComponentType<P> }>,
  getFeatureFlag: (flag: string) => boolean
): ComponentType<P> {
  return lazy(() => {
    if (getFeatureFlag(featureFlag)) {
      return enabledImport();
    }
    return disabledImport();
  });
}

// ============================================================================
// Vendor Chunk Optimization
// ============================================================================

export interface VendorChunkConfig {
  /** Vendor packages to group together */
  packages: string[];
  /** Chunk name */
  name: string;
  /** Priority (higher = more likely to be in this chunk) */
  priority?: number;
}

/**
 * Generate webpack splitChunks configuration for vendors
 */
export function generateVendorChunksConfig(
  vendors: VendorChunkConfig[]
): Record<string, unknown> {
  const cacheGroups: Record<string, unknown> = {};

  vendors.forEach((vendor, index) => {
    const regex = new RegExp(`[\\\\/]node_modules[\\\\/](${vendor.packages.join('|')})[\\\\/]`);

    cacheGroups[vendor.name] = {
      test: regex,
      name: vendor.name,
      chunks: 'all',
      priority: vendor.priority ?? (vendors.length - index),
      reuseExistingChunk: true,
    };
  });

  return {
    splitChunks: {
      chunks: 'all',
      cacheGroups,
    },
  };
}

/**
 * Recommended vendor chunk configuration for Chimera
 */
export const CHIMERA_VENDOR_CHUNKS: VendorChunkConfig[] = [
  {
    name: 'vendor-react',
    packages: ['react', 'react-dom', 'scheduler'],
    priority: 40,
  },
  {
    name: 'vendor-radix',
    packages: ['@radix-ui'],
    priority: 30,
  },
  {
    name: 'vendor-tanstack',
    packages: ['@tanstack'],
    priority: 25,
  },
  {
    name: 'vendor-charts',
    packages: ['recharts', 'd3'],
    priority: 20,
  },
  {
    name: 'vendor-forms',
    packages: ['react-hook-form', '@hookform', 'zod'],
    priority: 15,
  },
  {
    name: 'vendor-utils',
    packages: ['lodash', 'date-fns', 'clsx', 'tailwind-merge'],
    priority: 10,
  },
];

// ============================================================================
// Dynamic Import Utilities
// ============================================================================

/**
 * Dynamic import with named export support
 */
export function importNamed<T>(
  importFn: () => Promise<Record<string, unknown>>,
  exportName: string
): () => Promise<{ default: T }> {
  return async () => {
    const module = await importFn();
    return { default: module[exportName] as T };
  };
}

/**
 * Dynamic import with multiple named exports
 */
export async function importMultiple<T extends Record<string, unknown>>(
  importFn: () => Promise<Record<string, unknown>>,
  exportNames: (keyof T)[]
): Promise<T> {
  const module = await importFn();
  const result = {} as T;

  for (const name of exportNames) {
    result[name] = module[name as string] as T[keyof T];
  }

  return result;
}

/**
 * Batch import multiple modules
 */
export async function batchImport<T extends Record<string, () => Promise<unknown>>>(
  imports: T
): Promise<{ [K in keyof T]: Awaited<ReturnType<T[K]>> }> {
  const entries = Object.entries(imports);
  const results = await Promise.all(
    entries.map(async ([key, importFn]) => [key, await importFn()])
  );

  return Object.fromEntries(results) as { [K in keyof T]: Awaited<ReturnType<T[K]>> };
}

// ============================================================================
// Resource Hints
// ============================================================================

/**
 * Add a resource hint link element
 */
export function addResourceHint(
  chunkName: string,
  type: 'preload' | 'prefetch' | 'preconnect' | 'dns-prefetch'
): void {
  if (typeof window === 'undefined') return;

  const existingLink = document.querySelector(`link[data-chunk="${chunkName}"]`);
  if (existingLink) return;

  const link = document.createElement('link');
  link.rel = type;
  link.setAttribute('data-chunk', chunkName);

  if (type === 'preload' || type === 'prefetch') {
    link.as = 'script';
    link.href = `/_next/static/chunks/${chunkName}.js`;
  }

  document.head.appendChild(link);
}

/**
 * Remove a resource hint
 */
export function removeResourceHint(chunkName: string): void {
  if (typeof window === 'undefined') return;

  const link = document.querySelector(`link[data-chunk="${chunkName}"]`);
  if (link) {
    link.remove();
  }
}

/**
 * Preconnect to an origin
 */
export function preconnect(origin: string): void {
  if (typeof window === 'undefined') return;

  const existingLink = document.querySelector(`link[rel="preconnect"][href="${origin}"]`);
  if (existingLink) return;

  const link = document.createElement('link');
  link.rel = 'preconnect';
  link.href = origin;
  link.crossOrigin = 'anonymous';
  document.head.appendChild(link);
}

/**
 * DNS prefetch for an origin
 */
export function dnsPrefetch(origin: string): void {
  if (typeof window === 'undefined') return;

  const existingLink = document.querySelector(`link[rel="dns-prefetch"][href="${origin}"]`);
  if (existingLink) return;

  const link = document.createElement('link');
  link.rel = 'dns-prefetch';
  link.href = origin;
  document.head.appendChild(link);
}

// ============================================================================
// Chunk Loading Strategies
// ============================================================================

export type ChunkLoadingStrategy = 'eager' | 'lazy' | 'idle' | 'interaction';

/**
 * Load a chunk based on strategy
 */
export async function loadChunkWithStrategy<T>(
  importFn: () => Promise<T>,
  strategy: ChunkLoadingStrategy
): Promise<T> {
  switch (strategy) {
    case 'eager':
      return importFn();

    case 'lazy':
      // Load after a short delay
      await new Promise((resolve) => setTimeout(resolve, 100));
      return importFn();

    case 'idle':
      // Load during idle time
      return new Promise((resolve, reject) => {
        if (typeof requestIdleCallback !== 'undefined') {
          requestIdleCallback(async () => {
            try {
              resolve(await importFn());
            } catch (error) {
              reject(error);
            }
          });
        } else {
          setTimeout(async () => {
            try {
              resolve(await importFn());
            } catch (error) {
              reject(error);
            }
          }, 1);
        }
      });

    case 'interaction':
      // This should be triggered by user interaction
      // Return a promise that resolves when called
      return importFn();

    default:
      return importFn();
  }
}

// ============================================================================
// Module Federation Support (for micro-frontends)
// ============================================================================

export interface RemoteModuleConfig {
  /** Remote name */
  name: string;
  /** Remote URL */
  url: string;
  /** Exposed module path */
  module: string;
  /** Fallback component */
  fallback?: ComponentType<unknown>;
}

/**
 * Load a remote module (for module federation)
 */
export async function loadRemoteModule<T>(
  config: RemoteModuleConfig
): Promise<T> {
  // This is a placeholder for module federation support
  // Actual implementation would depend on webpack module federation setup

  const { url, module } = config;

  // Dynamic script loading for remote entry
  await loadRemoteEntry(url);

  // Get the remote container using type assertion for dynamic property access
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const windowAny = window as any;
  
  const container = windowAny[config.name] as {
    get: (module: string) => Promise<() => T>;
    init: (shareScope: unknown) => Promise<void>;
  } | undefined;

  if (!container) {
    throw new Error(`Remote container ${config.name} not found`);
  }

  // Initialize sharing
  const shareScopes = windowAny.__webpack_share_scopes__ as { default?: unknown } | undefined;
  await container.init(shareScopes?.default || {});

  // Get the module
  const factory = await container.get(module);
  return factory();
}

/**
 * Load remote entry script
 */
async function loadRemoteEntry(url: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existingScript = document.querySelector(`script[src="${url}"]`);
    if (existingScript) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = url;
    script.type = 'text/javascript';
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load remote entry: ${url}`));
    document.head.appendChild(script);
  });
}

// ============================================================================
// Chunk Analysis Utilities
// ============================================================================

export interface ChunkInfo {
  name: string;
  size: number;
  modules: string[];
  isAsync: boolean;
  isInitial: boolean;
}

/**
 * Get information about loaded chunks (development only)
 */
export function getLoadedChunks(): ChunkInfo[] {
  if (typeof window === 'undefined' || process.env.NODE_ENV !== 'development') {
    return [];
  }

  // This would need webpack runtime access
  // Placeholder implementation
  return [];
}

/**
 * Log chunk loading performance
 */
export function logChunkPerformance(chunkName: string, loadTime: number): void {
  if (process.env.NODE_ENV === 'development') {
    console.log(`[Chunk] ${chunkName} loaded in ${loadTime.toFixed(2)}ms`);
  }
}

// ============================================================================
// Critical CSS Extraction
// ============================================================================

/**
 * Extract critical CSS for above-the-fold content
 * This is typically done at build time, but this provides runtime utilities
 */
export function injectCriticalCSS(css: string): void {
  if (typeof window === 'undefined') return;

  const existingStyle = document.querySelector('style[data-critical]');
  if (existingStyle) return;

  const style = document.createElement('style');
  style.setAttribute('data-critical', 'true');
  style.textContent = css;
  document.head.insertBefore(style, document.head.firstChild);
}

/**
 * Load non-critical CSS asynchronously
 */
export function loadAsyncCSS(href: string): void {
  if (typeof window === 'undefined') return;

  const existingLink = document.querySelector(`link[href="${href}"]`);
  if (existingLink) return;

  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = href;
  link.media = 'print';
  link.onload = () => {
    link.media = 'all';
  };
  document.head.appendChild(link);
}